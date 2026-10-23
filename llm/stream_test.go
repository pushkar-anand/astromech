package llm_test

import (
	"net/http"
	"slices"
	"strings"
	"testing"
	"testing/synctest"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/pushkar-anand/astromech/llm"
)

// sse joins chunks into a stream body, ending it the way the LLM backend
// does.
func sse(chunks ...string) string {
	var b strings.Builder
	for _, c := range chunks {
		b.WriteString("data: " + c + "\n\n")
	}

	b.WriteString("data: [DONE]\n\n")

	return b.String()
}

var (
	chunkStart     = `{"id":"r1","model":"model-a-q4","choices":[{"index":0,"delta":{"role":"assistant","content":null}}]}`
	chunkThink     = `{"id":"r1","model":"model-a-q4","choices":[{"index":0,"delta":{"reasoning_content":"They said "}}]}`
	chunkThink2    = `{"id":"r1","model":"model-a-q4","choices":[{"index":0,"delta":{"reasoning_content":"hello."}}]}`
	chunkText      = `{"id":"r1","model":"model-a-q4","choices":[{"index":0,"delta":{"content":"Hi "}}]}`
	chunkText2     = `{"id":"r1","model":"model-a-q4","choices":[{"index":0,"delta":{"content":"there."}}]}`
	chunkStop      = `{"id":"r1","model":"model-a-q4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`
	chunkUsage     = `{"id":"r1","model":"model-a-q4","choices":[],"usage":{"prompt_tokens":9,"completion_tokens":4,"total_tokens":13}}`
	chunkUsageTime = `{"id":"r1","model":"model-a-q4","choices":[],"usage":{"prompt_tokens":9,"completion_tokens":4,"total_tokens":13},` +
		`"timings":{"prompt_ms":100,"predicted_ms":150,"predicted_per_second":26.7}}`
)

func collect(t *testing.T, c *llm.Client) []llm.Event {
	t.Helper()

	return slices.Collect(c.Stream(t.Context(), hello))
}

func TestStream(t *testing.T) {
	b, c := newBackend(t, map[string]reply{
		"POST /v1/chat/completions": {body: sse(chunkStart, chunkThink, chunkThink2, chunkText, chunkText2, chunkStop, chunkUsageTime)},
	})

	events := collect(t, c)

	want := &llm.Response{
		ID:           "r1",
		Model:        "model-a-q4",
		Message:      llm.Message{Role: llm.RoleAssistant, Content: "Hi there."},
		Reasoning:    "They said hello.",
		FinishReason: "stop",
		Usage:        llm.Usage{PromptTokens: 9, CompletionTokens: 4, TotalTokens: 13},
		Stats:        &llm.Stats{PromptMs: 100, PredictedMs: 150, PredictedPerSecond: 26.7},
	}

	assert.Equal(t, []llm.Event{
		llm.ReasoningDelta{Text: "They said "},
		llm.ReasoningDelta{Text: "hello."},
		llm.TextDelta{Text: "Hi "},
		llm.TextDelta{Text: "there."},
		llm.Done{Response: want},
	}, events)

	assert.Equal(t, map[string]any{
		"model":          "model-a",
		"messages":       []any{map[string]any{"role": "user", "content": "Hello"}},
		"stream":         true,
		"stream_options": map[string]any{"include_usage": true},
	}, b.lastBody())
}

func TestStreamWithoutStats(t *testing.T) {
	_, c := newBackend(t, map[string]reply{
		"POST /v1/chat/completions": {body: sse(chunkText, chunkStop, chunkUsage)},
	})

	events := collect(t, c)
	require.Len(t, events, 2)

	done, ok := events[1].(llm.Done)
	require.True(t, ok, "last event is %T, want llm.Done", events[1])

	assert.Equal(t, "Hi ", done.Response.Message.Content)
	assert.Empty(t, done.Response.Reasoning)
	assert.Equal(t, 13, done.Response.Usage.TotalTokens)
	assert.Nil(t, done.Response.Stats)
}

func TestStreamWithoutUsage(t *testing.T) {
	_, c := newBackend(t, map[string]reply{
		"POST /v1/chat/completions": {body: sse(chunkText, chunkStop)},
	})

	events := collect(t, c)
	require.Len(t, events, 2)

	done, ok := events[1].(llm.Done)
	require.True(t, ok, "last event is %T, want llm.Done", events[1])

	assert.Equal(t, "stop", done.Response.FinishReason)
	assert.Zero(t, done.Response.Usage)
}

func TestStreamReasoningField(t *testing.T) {
	_, c := newBackend(t, map[string]reply{
		"POST /v1/chat/completions": {body: sse(
			`{"id":"r1","model":"model-a","choices":[{"index":0,"delta":{"reasoning":"They said "}}]}`,
			`{"id":"r1","model":"model-a","choices":[{"index":0,"delta":{"reasoning":"hello."}}]}`,
			chunkText, chunkStop,
		)},
	})

	events := collect(t, c)
	require.Len(t, events, 4)

	assert.Equal(t, []llm.Event{
		llm.ReasoningDelta{Text: "They said "},
		llm.ReasoningDelta{Text: "hello."},
		llm.TextDelta{Text: "Hi "},
	}, events[:3])

	done, ok := events[3].(llm.Done)
	require.True(t, ok, "last event is %T, want llm.Done", events[3])
	assert.Equal(t, "They said hello.", done.Response.Reasoning)
}

func TestStreamIncomplete(t *testing.T) {
	_, c := newBackend(t, map[string]reply{
		// No finish_reason and no [DONE]: the connection dropped mid-reply.
		"POST /v1/chat/completions": {body: "data: " + chunkText + "\n\n"},
	})

	events := collect(t, c)
	require.Len(t, events, 2)

	assert.Equal(t, llm.TextDelta{Text: "Hi "}, events[0])

	failed, ok := events[1].(llm.Failed)
	require.True(t, ok, "last event is %T, want llm.Failed", events[1])
	assert.ErrorIs(t, failed.Err, llm.ErrIncompleteStream)
}

func TestStreamError(t *testing.T) {
	_, c := newBackend(t, map[string]reply{
		"POST /v1/chat/completions": {status: http.StatusNotFound, body: errorBody},
	})

	events := collect(t, c)
	require.Len(t, events, 1)

	failed, ok := events[0].(llm.Failed)
	require.True(t, ok, "event is %T, want llm.Failed", events[0])
	assert.Contains(t, failed.Err.Error(), "llm: stream")
}

func TestStreamWaitsForSlowModel(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		_, c := newBackend(t, map[string]reply{
			"POST /v1/chat/completions": {delay: 10 * time.Minute, body: sse(chunkText, chunkStop)},
		})

		start := time.Now()

		events := collect(t, c)
		require.Len(t, events, 2)

		assert.IsType(t, llm.Done{}, events[1])
		assert.Equal(t, 10*time.Minute, time.Since(start))
	})
}

func TestStreamStopsEarly(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		b, c := newBackend(t, map[string]reply{
			// The reply never finishes, so only stopping can end the request.
			"POST /v1/chat/completions": {hang: true, body: "data: " + chunkThink + "\n\ndata: " + chunkText + "\n\n"},
		})

		var events []llm.Event
		for ev := range c.Stream(t.Context(), hello) {
			events = append(events, ev)
			if _, ok := ev.(llm.TextDelta); ok {
				break
			}
		}

		assert.Equal(t, []llm.Event{
			llm.ReasoningDelta{Text: "They said "},
			llm.TextDelta{Text: "Hi "},
		}, events)

		select {
		case <-b.canceled:
		case <-time.After(time.Minute):
			t.Fatal("the request to the backend is still open")
		}
	})
}
