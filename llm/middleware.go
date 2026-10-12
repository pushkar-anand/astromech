package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strconv"
	"strings"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

// scrub removes the SDK's X-Stainless-* headers, which report the host's
// operating system, architecture and Go version to whichever server the
// client talks to.
func scrub(req *http.Request, next option.MiddlewareNext) (*http.Response, error) {
	for name := range req.Header {
		if strings.HasPrefix(name, "X-Stainless-") {
			req.Header.Del(name)
		}
	}

	return next(req)
}

// attempts follows one call through the SDK's retries. It sees every
// attempt's response before the SDK does, which is how a failure keeps the
// server's own explanation: the SDK reads only OpenAI's error object and
// turns any other shape into a decoding error.
type attempts struct {
	// onRetry, when not nil, hears of each retry as it starts.
	onRetry func(Retrying)

	count  int
	last   *APIError
	lastTx error
	status int
}

func (a *attempts) track(req *http.Request, next option.MiddlewareNext) (*http.Response, error) {
	a.count++

	if a.count > 1 && a.onRetry != nil {
		prev := error(a.last)
		if a.last == nil {
			prev = a.lastTx
		}

		a.onRetry(Retrying{Attempt: a.count - 1, Err: prev})
	}

	res, err := next(req)
	if err != nil {
		if req.Context().Err() == nil {
			a.last, a.lastTx = nil, &TransportError{Err: err}
		}

		return res, err
	}

	a.status = res.StatusCode

	if res.StatusCode < http.StatusBadRequest {
		a.last, a.lastTx = nil, nil

		return res, nil
	}

	raw, _ := io.ReadAll(io.LimitReader(res.Body, maxErrorBody))
	_ = res.Body.Close()

	a.last, a.lastTx = newAPIError(res.StatusCode, res.Header, raw), nil

	// Hand the SDK an error object it can read, so it reports the status
	// in place of failing to decode the body.
	body, _ := json.Marshal(map[string]any{"error": map[string]string{
		"message": a.last.Message,
		"type":    a.last.Type,
		"code":    a.last.Code,
		"param":   a.last.Param,
	}})

	res.Body = io.NopCloser(bytes.NewReader(body))
	res.ContentLength = int64(len(body))
	res.Header.Set("Content-Length", strconv.Itoa(len(body)))

	return res, nil
}

// explain turns an error the SDK returned into this package's: the parsed
// [*APIError] for a refusal, a [*TransportError] for a connection failure,
// and the context's error once it has ended.
func (a *attempts) explain(ctx context.Context, err error) error {
	if ctx.Err() != nil {
		return context.Cause(ctx)
	}

	if _, ok := errors.AsType[*openai.Error](err); ok && a.last != nil {
		return a.last
	}

	if a.lastTx != nil {
		return a.lastTx
	}

	return err
}
