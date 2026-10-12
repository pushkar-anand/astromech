package llm

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestIsRetryable(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want bool
	}{
		{"rate limit", &APIError{StatusCode: http.StatusTooManyRequests}, true},
		{"server error", &APIError{StatusCode: http.StatusBadGateway}, true},
		{"refusal", &APIError{StatusCode: http.StatusBadRequest}, false},
		{"server error with no status", &APIError{Type: "server_error"}, true},
		{"refusal with no status", &APIError{Type: "invalid_request_error"}, false},
		{"server says retry", &APIError{StatusCode: http.StatusBadRequest, shouldRetry: new(true)}, true},
		{"server says do not retry", &APIError{StatusCode: http.StatusServiceUnavailable, shouldRetry: new(false)}, false},
		{"dropped connection", &TransportError{Err: errors.New("connection reset by peer")}, true},
		{"wrapped", fmt.Errorf("ask: %w", &APIError{StatusCode: http.StatusInternalServerError}), true},
		{"canceled", context.Canceled, false},
		{"other", errors.New("something else"), false},
		{"nil", nil, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, IsRetryable(tt.err))
		})
	}
}

func TestIsContextOverflow(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want bool
	}{
		{"OpenAI", &APIError{StatusCode: http.StatusBadRequest, Code: "context_length_exceeded"}, true},
		{"vLLM", &APIError{StatusCode: http.StatusBadRequest, Message: "This model's maximum context length is 4096 tokens. However, you requested 5000 tokens."}, true},
		{"llama.cpp", &APIError{StatusCode: http.StatusBadRequest, Message: "the request exceeds the available context size, try increasing it"}, true},
		{"payload", &APIError{StatusCode: http.StatusRequestEntityTooLarge, Message: "request_too_large"}, true},
		{"mid-stream", &APIError{Message: "prompt is too long: 210000 tokens > 200000 maximum"}, true},
		{"rate limit", &APIError{StatusCode: http.StatusBadRequest, Message: "rate limit: too many tokens per minute"}, false},
		{"server error", &APIError{StatusCode: http.StatusInternalServerError, Message: "maximum context length"}, false},
		{"other refusal", &APIError{StatusCode: http.StatusBadRequest, Message: "temperature must be at most 2"}, false},
		{"not an API error", errors.New("maximum context length"), false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, IsContextOverflow(tt.err))
		})
	}
}

func TestParseErrorBody(t *testing.T) {
	tests := []struct {
		name string
		body string
		want APIError
	}{
		{"OpenAI", `{"error":{"message":"m","type":"t","param":"p","code":"c"}}`, APIError{Message: "m", Type: "t", Param: "p", Code: "c"}},
		{"numeric code", `{"error":{"message":"m","code":429}}`, APIError{Message: "m", Code: "429"}},
		{"bare error string", `{"error":"model not found"}`, APIError{Message: "model not found"}},
		{"FastAPI string", `{"detail":"bad request"}`, APIError{Message: "bad request"}},
		{"FastAPI list", `{"detail":[{"loc":["body","model"],"msg":"field required"}]}`, APIError{Message: `[{"loc":["body","model"],"msg":"field required"}]`}},
		{"top-level message", `{"message":"no such model"}`, APIError{Message: "no such model"}},
		{"plain text", "  Bad Gateway\n", APIError{Message: "Bad Gateway"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, &tt.want, parseErrorBody([]byte(tt.body)))
		})
	}
}

func TestRetryAfter(t *testing.T) {
	assert.Equal(t, 1500*time.Millisecond, retryAfter(http.Header{"Retry-After-Ms": {"1500"}, "Retry-After": {"9"}}))
	assert.Equal(t, 2*time.Second, retryAfter(http.Header{"Retry-After": {"2"}}))
	assert.Zero(t, retryAfter(http.Header{"Retry-After": {"soon"}}))
	assert.Zero(t, retryAfter(http.Header{}))

	at := time.Now().Add(time.Hour).UTC().Format(http.TimeFormat)
	assert.InDelta(t, time.Hour, retryAfter(http.Header{"Retry-After": {at}}), float64(2*time.Second))
}

func TestAPIErrorText(t *testing.T) {
	assert.Equal(t, "llm: 429 Too Many Requests (rate_limit_exceeded): slow down",
		(&APIError{StatusCode: 429, Code: "rate_limit_exceeded", Message: "slow down"}).Error())
	assert.Equal(t, "llm: server error: upstream failed", (&APIError{Message: "upstream failed"}).Error())
}

func TestTruncateKeepsUTF8(t *testing.T) {
	assert.Equal(t, "ab…", truncate("abc", 2))
	assert.Equal(t, "a…", truncate("aé", 2), "the cut falls inside é, so it moves before it")
	assert.Equal(t, "abc", truncate("abc", 3))
}
