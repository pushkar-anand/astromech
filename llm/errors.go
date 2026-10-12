package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"
)

var (
	// ErrEmptyResponse is returned when a server answers with no choices
	// and no error to say why, which some gateways do when the upstream
	// model fails.
	ErrEmptyResponse = errors.New("llm: the server answered with no reply")

	// ErrIncompleteStream is returned when a stream ends before the server
	// said the reply was finished, such as when a proxy drops the connection.
	ErrIncompleteStream = errors.New("llm: the stream ended before the reply finished")
)

// APIError is a request the server refused or failed.
type APIError struct {
	// StatusCode is the HTTP status. An error a server sends inside a 200
	// response, in a stream or in place of a reply, carries the status its
	// code names, as OpenRouter's do, or 0 when it names none.
	StatusCode int

	// Type, Code and Param are the fields of an OpenAI error object, empty
	// where the server sent something else.
	Type  string
	Code  string
	Param string

	// Message is the server's explanation, or the start of the body when the
	// server sent no structured error.
	Message string

	// RetryAfter is how long the server asked the client to wait, from the
	// Retry-After or retry-after-ms header. Zero when it did not say.
	RetryAfter time.Duration

	// shouldRetry is the server's own verdict from x-should-retry, nil when
	// it gave none.
	shouldRetry *bool
}

func (e *APIError) Error() string {
	var b strings.Builder

	b.WriteString("llm: ")

	if e.StatusCode != 0 {
		fmt.Fprintf(&b, "%d %s", e.StatusCode, http.StatusText(e.StatusCode))
	} else {
		b.WriteString("server error")
	}

	if e.Code != "" {
		fmt.Fprintf(&b, " (%s)", e.Code)
	}

	if e.Message != "" {
		b.WriteString(": ")
		b.WriteString(e.Message)
	}

	return b.String()
}

// retryable reports whether sending the same request again can succeed.
// The statuses are the ones OpenAI's own SDKs retry.
func (e *APIError) retryable() bool {
	if e.shouldRetry != nil {
		return *e.shouldRetry
	}

	switch e.StatusCode {
	case http.StatusRequestTimeout, http.StatusConflict, http.StatusTooManyRequests:
		return true
	}

	return e.StatusCode >= http.StatusInternalServerError
}

// TransportError is a request that failed before a response arrived: the
// connection was refused or dropped, or a TLS handshake failed.
type TransportError struct {
	Err error
}

func (e *TransportError) Error() string {
	return "llm: the server could not be reached: " + e.Err.Error()
}

func (e *TransportError) Unwrap() error { return e.Err }

// IsRetryable reports whether a request that failed with err can succeed if
// sent again: a rate limit, a server-side failure, or a dropped connection.
// It reports false once the request's context has ended.
//
// A [Client] has already retried such a failure as often as
// [WithMaxRetries] allows by the time it returns one, so a true result
// means trying again later, such as on the user's next message.
func IsRetryable(err error) bool {
	if err == nil || errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}

	if e, ok := errors.AsType[*APIError](err); ok {
		return e.retryable()
	}

	_, ok := errors.AsType[*TransportError](err)

	return ok
}

// overflowPatterns match the ways servers say a request is longer than the
// model's context window. No status or code is shared between them, so the
// message is all there is to go on.
var overflowPatterns = regexp.MustCompile(`(?i)` + strings.Join([]string{
	`maximum context length`,
	`context[_ ]length[_ ]exceeded`,
	`context window`,
	`exceeds the available context size`,
	`greater than the context length`,
	`longer than the model'?s context length`,
	`input length.*exceeds.*context length`,
	`exceeds (?:the )?maximum allowed input length`,
	`prompt is too long`,
	`prompt too long`,
	`too many tokens`,
	`token limit exceeded`,
	`request_too_large`,
	`model_context_window_exceeded`,
}, "|"))

// overflowExclusions match messages that look like an overflow and are
// throttling instead.
var overflowExclusions = regexp.MustCompile(`(?i)rate limit|too many requests|max[_ ]tokens must be`)

// IsContextOverflow reports whether err says the request did not fit the
// model's context window, so trimming the conversation and sending it again
// can succeed.
func IsContextOverflow(err error) bool {
	e, ok := errors.AsType[*APIError](err)
	if !ok {
		return false
	}

	if e.StatusCode != http.StatusBadRequest && e.StatusCode != http.StatusRequestEntityTooLarge && e.StatusCode != 0 {
		return false
	}

	text := e.Code + " " + e.Message

	return overflowPatterns.MatchString(text) && !overflowExclusions.MatchString(text)
}

// maxErrorBody caps how much of an error response is read. An error object
// is a few hundred bytes; anything longer is an HTML page from a proxy.
const maxErrorBody = 64 << 10

// maxErrorMessage caps the message kept from an unstructured error body.
const maxErrorMessage = 512

// errorInBody parses an error a server sent inside a 200 response.
func errorInBody(body []byte) *APIError {
	e := parseErrorBody(body)

	if n, err := strconv.Atoi(e.Code); err == nil && n >= http.StatusBadRequest && n < 600 {
		e.StatusCode = n
	}

	return e
}

// newAPIError builds the error for a response with status code and body.
func newAPIError(code int, h http.Header, body []byte) *APIError {
	e := parseErrorBody(body)
	e.StatusCode = code
	e.RetryAfter = retryAfter(h)

	switch h.Get("x-should-retry") {
	case "true":
		e.shouldRetry = new(true)
	case "false":
		e.shouldRetry = new(false)
	}

	return e
}

// parseErrorBody reads the error shapes servers send: OpenAI's error object,
// a bare error string (Ollama), FastAPI's detail (vLLM), or a top-level
// message. Anything else is kept as text.
func parseErrorBody(body []byte) *APIError {
	var shape struct {
		Error   json.RawMessage `json:"error"`
		Detail  json.RawMessage `json:"detail"`
		Message string          `json:"message"`
	}

	if json.Unmarshal(body, &shape) != nil {
		return &APIError{Message: truncate(strings.TrimSpace(string(body)), maxErrorMessage)}
	}

	if len(shape.Error) > 0 {
		var obj struct {
			Message string          `json:"message"`
			Type    string          `json:"type"`
			Param   string          `json:"param"`
			Code    json.RawMessage `json:"code"`
		}

		if json.Unmarshal(shape.Error, &obj) == nil {
			return &APIError{Message: obj.Message, Type: obj.Type, Param: obj.Param, Code: rawString(obj.Code)}
		}

		var text string
		if json.Unmarshal(shape.Error, &text) == nil {
			return &APIError{Message: text}
		}
	}

	if len(shape.Detail) > 0 {
		var text string
		if json.Unmarshal(shape.Detail, &text) == nil {
			return &APIError{Message: text}
		}

		// FastAPI's validation errors are a list of objects; the raw JSON
		// still names the field at fault.
		return &APIError{Message: truncate(string(shape.Detail), maxErrorMessage)}
	}

	return &APIError{Message: shape.Message}
}

// rawString renders a JSON value that servers send as either a string or a
// number, such as an error code.
func rawString(raw json.RawMessage) string {
	if len(raw) == 0 || string(raw) == "null" {
		return ""
	}

	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}

	return string(raw)
}

// retryAfter reads how long the server asked to wait. retry-after-ms is
// OpenAI's finer-grained header and wins when both are present.
func retryAfter(h http.Header) time.Duration {
	if ms, err := strconv.ParseFloat(h.Get("retry-after-ms"), 64); err == nil && ms > 0 {
		return time.Duration(ms * float64(time.Millisecond))
	}

	v := h.Get("Retry-After")
	if v == "" {
		return 0
	}

	if s, err := strconv.ParseFloat(v, 64); err == nil && s > 0 {
		return time.Duration(s * float64(time.Second))
	}

	if t, err := http.ParseTime(v); err == nil {
		return max(time.Until(t), 0)
	}

	return 0
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}

	// Cut on a rune boundary so the message stays valid UTF-8.
	for n > 0 && !utf8.RuneStart(s[n]) {
		n--
	}

	return s[:n] + "…"
}
