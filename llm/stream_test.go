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

// frames joins chunks into a stream body that just stops, as a dropped
// connection does.
func frames(chunks ...string) string {
	var b strings.Builder
	for _, c := range chunks {
		b.WriteString("data: " + c + "\n\n")
	}

	return b.String()
}

// sse joins chunks into a stream body, ending it the way the LLM backend
// does.
func sse(chunks ...string) string {
	return frames(append(chunks, "[DONE]")...)
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

// streamDone collects a stream and returns its Done response, failing the
// test if it ended any other way.
func streamDone(t *testing.T, c *llm.Client) ([]llm.Event, *llm.Response) {
	t.Helper()

	events := collect(t, c)
	require.NotEmpty(t, events)

	done, ok := events[len(events)-1].(llm.Done)
	require.True(t, ok, "last event is %#v, want llm.Done", events[len(events)-1])

	return events[:len(events)-1], done.Response
}

// streamFailed collects a stream and returns its Failed error, failing the
// test if it ended any other way.
func streamFailed(t *testing.T, c *llm.Client) ([]llm.Event, error) {
	t.Helper()

	events := collect(t, c)
	require.NotEmpty(t, events)

	failed, ok := events[len(events)-1].(llm.Failed)
	require.True(t, ok, "last event is %#v, want llm.Failed", events[len(events)-1])

	return events[:len(events)-1], failed.Err
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
		Usage:        llm.Usage{InputTokens: 9, OutputTokens: 4, TotalTokens: 13},
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

	deltas, resp := streamDone(t, c)

	assert.Len(t, deltas, 1)
	assert.Equal(t, "Hi ", resp.Message.Content)
	assert.Empty(t, resp.Reasoning)
	assert.Equal(t, 13, resp.Usage.TotalTokens)
	assert.Nil(t, resp.Stats)
}

func TestStreamWithoutUsage(t *testing.T) {
	_, c := newBackend(t, map[string]reply{
		"POST /v1/chat/completions": {body: sse(chunkText, chunkStop)},
	})

	deltas, resp := streamDone(t, c)

	assert.Len(t, deltas, 1)
	assert.Equal(t, "stop", resp.FinishReason)
	assert.Zero(t, resp.Usage)
}

func TestStreamReasoningField(t *testing.T) {
	_, c := newBackend(t, map[string]reply{
		"POST /v1/chat/completions": {body: sse(
			`{"id":"r1","model":"model-a","choices":[{"index":0,"delta":{"reasoning":"They said "}}]}`,
			`{"id":"r1","model":"model-a","choices":[{"index":0,"delta":{"reasoning":"hello."}}]}`,
			chunkText, chunkStop,
		)},
	})

	deltas, resp := streamDone(t, c)

	assert.Equal(t, []llm.Event{
		llm.ReasoningDelta{Text: "They said "},
		llm.ReasoningDelta{Text: "hello."},
		llm.TextDelta{Text: "Hi "},
	}, deltas)
	assert.Equal(t, "They said hello.", resp.Reasoning)
}

func TestStreamIncomplete(t *testing.T) {
	_, c := newBackend(t, map[string]reply{
		// No finish_reason and no [DONE]: the connection dropped mid-reply.
		"POST /v1/chat/completions": {body: frames(chunkText)},
	})

	deltas, err := streamFailed(t, c)

	assert.Equal(t, []llm.Event{llm.TextDelta{Text: "Hi "}}, deltas)
	assert.ErrorIs(t, err, llm.ErrIncompleteStream)
}

func TestStreamErrorMidway(t *testing.T) {
	_, c := newBackend(t, map[string]reply{
		"POST /v1/chat/completions": {body: sse(
			chunkText,
			`{"error":{"code":500,"message":"out of memory","type":"server_error"}}`,
			chunkText2, chunkStop,
		)},
	})

	deltas, err := streamFailed(t, c)

	assert.Equal(t, []llm.Event{llm.TextDelta{Text: "Hi "}}, deltas)

	var backendErr *llm.BackendError
	require.ErrorAs(t, err, &backendErr)
	assert.Equal(t, &llm.BackendError{Message: "out of memory", Type: "server_error"}, backendErr)
	assert.EqualError(t, err, "llm: stream: the LLM backend failed: out of memory")
}

func TestStreamError(t *testing.T) {
	_, c := newBackend(t, map[string]reply{
		"POST /v1/chat/completions": {status: http.StatusNotFound, body: errorBody},
	})

	deltas, err := streamFailed(t, c)

	assert.Empty(t, deltas)
	assert.EqualError(t, err, "llm: stream: the LLM backend answered 404 Not Found: model not found")
	requireBackendError(t, err, http.StatusNotFound)
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
			"POST /v1/chat/completions": {hang: true, body: frames(chunkThink, chunkText)},
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

// toolChunk is a stream chunk carrying tool-call pieces.
func toolChunk(pieces ...string) string {
	return `{"choices":[{"index":0,"delta":{"tool_calls":[` + strings.Join(pieces, ",") + `]}}]}`
}

const chunkToolStop = `{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`

func TestStreamToolCallPieces(t *testing.T) {
	_, c := newBackend(t, map[string]reply{
		"POST /v1/chat/completions": {body: sse(
			toolChunk(`{"index":0,"id":"call_1","type":"function","function":{"name":"get_time","arguments":""}}`),
			toolChunk(`{"index":0,"function":{"arguments":"{\"zone\""}}`),
			toolChunk(`{"index":0,"function":{"arguments":":\"UTC\"}"}}`),
			chunkToolStop,
		)},
	})

	deltas, resp := streamDone(t, c)

	assert.Empty(t, deltas, "a reply with only a tool call has no text")
	assert.Equal(t, "tool_calls", resp.FinishReason)
	assert.Equal(t, llm.Message{
		Role:      llm.RoleAssistant,
		ToolCalls: []llm.ToolCall{{ID: "call_1", Name: "get_time", Arguments: `{"zone":"UTC"}`}},
	}, resp.Message)
}

func TestStreamToolCallsInterleaved(t *testing.T) {
	_, c := newBackend(t, map[string]reply{
		"POST /v1/chat/completions": {body: sse(
			toolChunk(`{"index":0,"id":"call_1","type":"function","function":{"name":"get_time","arguments":""}}`),
			toolChunk(`{"index":0,"function":{"arguments":"{\"zone\""}}`),
			toolChunk(`{"index":1,"id":"call_2","type":"function","function":{"name":"now","arguments":"{}"}}`),
			toolChunk(`{"index":0,"function":{"arguments":":\"UTC\"}"}}`),
			chunkToolStop,
		)},
	})

	_, resp := streamDone(t, c)

	assert.Equal(t, []llm.ToolCall{
		{ID: "call_1", Name: "get_time", Arguments: `{"zone":"UTC"}`},
		{ID: "call_2", Name: "now", Arguments: "{}"},
	}, resp.Message.ToolCalls)
}

func TestStreamWholeToolCalls(t *testing.T) {
	_, c := newBackend(t, map[string]reply{
		"POST /v1/chat/completions": {body: sse(
			toolChunk(
				`{"index":0,"id":"call_1","type":"function","function":{"name":"get_time","arguments":"{\"zone\":\"UTC\"}"}}`,
				`{"index":1,"id":"call_2","type":"function","function":{"name":"now","arguments":"{}"}}`,
			),
			chunkToolStop,
		)},
	})

	_, resp := streamDone(t, c)

	assert.Equal(t, []llm.ToolCall{
		{ID: "call_1", Name: "get_time", Arguments: `{"zone":"UTC"}`},
		{ID: "call_2", Name: "now", Arguments: "{}"},
	}, resp.Message.ToolCalls)
}

func TestStreamToolCallsReuseIndex(t *testing.T) {
	_, c := newBackend(t, map[string]reply{
		// Both calls at index 0, told apart only by id. The name is sent
		// with every piece, and must not be repeated.
		"POST /v1/chat/completions": {body: sse(
			toolChunk(`{"index":0,"id":"call_1","type":"function","function":{"name":"get_time","arguments":"{\"zone\":"}}`),
			toolChunk(`{"index":0,"id":"call_1","type":"function","function":{"name":"get_time","arguments":"\"UTC\"}"}}`),
			toolChunk(`{"index":0,"id":"call_2","type":"function","function":{"name":"get_time","arguments":"{\"zone\":"}}`),
			toolChunk(`{"index":0,"id":"call_2","type":"function","function":{"name":"get_time","arguments":"\"Asia/Tokyo\"}"}}`),
			chunkToolStop,
		)},
	})

	_, resp := streamDone(t, c)

	assert.Equal(t, []llm.ToolCall{
		{ID: "call_1", Name: "get_time", Arguments: `{"zone":"UTC"}`},
		{ID: "call_2", Name: "get_time", Arguments: `{"zone":"Asia/Tokyo"}`},
	}, resp.Message.ToolCalls)
}

func TestStreamToolCallQuirks(t *testing.T) {
	_, c := newBackend(t, map[string]reply{
		"POST /v1/chat/completions": {body: sse(
			toolChunk(`{"index":0,"type":"function","function":{"name":"now","arguments":""}}`),
			chunkToolStop,
		)},
	})

	_, resp := streamDone(t, c)

	assert.Equal(t, []llm.ToolCall{{ID: "call_0", Name: "now", Arguments: "{}"}}, resp.Message.ToolCalls)
}

func TestStreamTextThenToolCall(t *testing.T) {
	_, c := newBackend(t, map[string]reply{
		"POST /v1/chat/completions": {body: sse(
			chunkText,
			toolChunk(`{"index":0,"id":"call_1","type":"function","function":{"name":"now","arguments":"{}"}}`),
			chunkToolStop,
		)},
	})

	deltas, resp := streamDone(t, c)

	assert.Equal(t, []llm.Event{llm.TextDelta{Text: "Hi "}}, deltas)
	assert.Equal(t, "Hi ", resp.Message.Content)
	assert.Equal(t, []llm.ToolCall{{ID: "call_1", Name: "now", Arguments: "{}"}}, resp.Message.ToolCalls)
}

func TestStreamToolCallCutOff(t *testing.T) {
	_, c := newBackend(t, map[string]reply{
		// The connection drops partway through the arguments.
		"POST /v1/chat/completions": {body: frames(
			toolChunk(`{"index":0,"id":"call_1","type":"function","function":{"name":"get_time","arguments":"{\"zone\""}}`),
		)},
	})

	deltas, err := streamFailed(t, c)

	assert.Empty(t, deltas)
	assert.ErrorIs(t, err, llm.ErrIncompleteStream)
}
