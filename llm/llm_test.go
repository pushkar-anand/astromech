package llm_test

import (
	"encoding/json/v2"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/pushkar-anand/build-with-go/http/client"
	"github.com/pushkar-anand/build-with-go/secret"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/pushkar-anand/astromech/llm"
)

// backend is a fake LLM backend, on httptest's in-memory network, that
// answers each route with a canned reply and records what it was sent.
type backend struct {
	t       *testing.T
	replies map[string]reply

	mu      sync.Mutex
	headers []http.Header
	bodies  []map[string]any

	// canceled is closed when a hanging reply sees the client go away.
	canceled chan struct{}
}

type reply struct {
	status int
	body   string

	// delay holds back the headers, as a backend loading a model does.
	delay time.Duration

	// hang keeps the connection open after the body until the client goes
	// away.
	hang bool
}

func newBackend(t *testing.T, replies map[string]reply) (*backend, *llm.Client) {
	t.Helper()

	return newBackendWithKey(t, replies, "")
}

func newBackendWithKey(t *testing.T, replies map[string]reply, apiKey string) (*backend, *llm.Client) {
	t.Helper()

	b := &backend{t: t, replies: replies, canceled: make(chan struct{})}
	srv := httptest.NewTestServer(t, b)

	c, err := llm.NewWithTransport(srv.URL+"/v1", secret.New(apiKey), srv.Client().Transport)
	require.NoError(t, err)

	return b, c
}

func (b *backend) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	raw, _ := io.ReadAll(r.Body)

	var body map[string]any
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &body); err != nil {
			b.t.Errorf("request body is not JSON: %v", err)
		}
	}

	b.mu.Lock()
	b.headers = append(b.headers, r.Header.Clone())
	b.bodies = append(b.bodies, body)
	b.mu.Unlock()

	rep, ok := b.replies[r.Method+" "+r.URL.Path]
	if !ok {
		http.NotFound(w, r)
		return
	}

	time.Sleep(rep.delay)

	if strings.HasPrefix(rep.body, "data:") {
		w.Header().Set("Content-Type", "text/event-stream")
	} else {
		w.Header().Set("Content-Type", "application/json")
	}

	if rep.status != 0 {
		w.WriteHeader(rep.status)
	}

	_, _ = io.WriteString(w, rep.body)

	if rep.hang {
		http.NewResponseController(w).Flush()
		<-r.Context().Done()
		close(b.canceled)
	}
}

func (b *backend) lastBody() map[string]any {
	b.mu.Lock()
	defer b.mu.Unlock()

	require.NotEmpty(b.t, b.bodies, "the backend got no request")

	return b.bodies[len(b.bodies)-1]
}

func (b *backend) lastHeader() http.Header {
	b.mu.Lock()
	defer b.mu.Unlock()

	require.NotEmpty(b.t, b.headers, "the backend got no request")

	return b.headers[len(b.headers)-1]
}

const errorBody = `{"error":{"message":"model not found","type":"invalid_request_error","code":"model_not_found"}}`

// requireBackendError checks that err explains errorBody, sent with the
// given status, and still leads to the error response.
func requireBackendError(t *testing.T, err error, statusCode int) {
	t.Helper()

	var backendErr *llm.BackendError
	require.ErrorAs(t, err, &backendErr)
	assert.Equal(t, statusCode, backendErr.StatusCode)
	assert.Equal(t, "model not found", backendErr.Message)
	assert.Equal(t, "invalid_request_error", backendErr.Type)

	var status *client.StatusError
	require.ErrorAs(t, err, &status)
	assert.Equal(t, statusCode, status.StatusCode)
}

var hello = llm.Request{
	Model:    "model-a",
	Messages: []llm.Message{{Role: llm.RoleUser, Content: "Hello"}},
}

func TestNewSendsAPIKey(t *testing.T) {
	b, c := newBackendWithKey(t, map[string]reply{
		"GET /v1/models": {body: `{"data":[]}`},
	}, "placeholder-key")

	_, err := c.Models(t.Context())
	require.NoError(t, err)

	assert.Equal(t, "Bearer placeholder-key", b.lastHeader().Get("Authorization"))
}

func TestNewWithoutAPIKey(t *testing.T) {
	b, c := newBackend(t, map[string]reply{
		"GET /v1/models": {body: `{"data":[]}`},
	})

	_, err := c.Models(t.Context())
	require.NoError(t, err)

	assert.Empty(t, b.lastHeader().Get("Authorization"))
}
