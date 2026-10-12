package llm_test

import (
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/pushkar-anand/astromech/llm"
	"github.com/pushkar-anand/astromech/llm/llmtest"
)

func newClient(t *testing.T, srv *llmtest.Server, opts ...llm.Option) *llm.Client {
	t.Helper()

	c, err := llm.New(srv.URL(), opts...)
	require.NoError(t, err)

	return c
}

func ask(text string) llm.Request {
	return llm.Request{Model: "test-model", Messages: []llm.Message{llm.User(text)}}
}

// collect reads a whole stream, keeping every event.
func collect(t *testing.T, c *llm.Client, req llm.Request) []llm.Event {
	t.Helper()

	var events []llm.Event
	for e := range c.Stream(t.Context(), req) {
		events = append(events, e)
	}

	return events
}

func last(t *testing.T, events []llm.Event) llm.Event {
	t.Helper()
	require.NotEmpty(t, events)

	return events[len(events)-1]
}

func TestNewRefusesAnAddressThatIsNotHTTP(t *testing.T) {
	for _, base := range []string{"", "llm.example.com/v1", "ftp://llm.example.com", "https://"} {
		_, err := llm.New(base)
		assert.Error(t, err, base)
	}
}

func TestCompleteReturnsText(t *testing.T) {
	srv := llmtest.NewServer(t)
	srv.Enqueue(llmtest.Reply{Text: "hello", Usage: llm.Usage{InputTokens: 7, OutputTokens: 2}})

	resp, err := newClient(t, srv).Complete(t.Context(), ask("hi"))
	require.NoError(t, err)

	assert.Equal(t, "hello", resp.Message.Content)
	assert.Equal(t, llm.RoleAssistant, resp.Message.Role)
	assert.Equal(t, llm.StopEnd, resp.Stop)
	assert.Equal(t, llm.Usage{InputTokens: 7, OutputTokens: 2}, resp.Usage)
	assert.Equal(t, "test-model", resp.Model)

	req := srv.Requests()[0]
	assert.Equal(t, "test-model", req.Field("model"))
	assert.Nil(t, req.Field("stream"))
	assert.Empty(t, req.Header.Get("Authorization"), "no key was given, so none is sent")
}

func TestCompleteReturnsToolCalls(t *testing.T) {
	srv := llmtest.NewServer(t)
	srv.Enqueue(llmtest.ToolCalls(llm.ToolCall{ID: "call_1", Name: "list_devices", Arguments: `{"group":"lab"}`}))

	resp, err := newClient(t, srv).Complete(t.Context(), ask("what is in the lab?"))
	require.NoError(t, err)

	assert.Equal(t, llm.StopToolCalls, resp.Stop)
	assert.Equal(t, []llm.ToolCall{{ID: "call_1", Name: "list_devices", Arguments: `{"group":"lab"}`}}, resp.Message.ToolCalls)
}

func TestAPIKeyIsSentAsBearerToken(t *testing.T) {
	srv := llmtest.NewServer(t)
	srv.Enqueue(llmtest.Text("ok"))

	_, err := newClient(t, srv, llm.WithAPIKey("test-key")).Complete(t.Context(), ask("hi"))
	require.NoError(t, err)

	assert.Equal(t, "Bearer test-key", srv.Requests()[0].Header.Get("Authorization"))
}

func TestEnvironmentDoesNotReachTheServer(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "env-key")
	t.Setenv("OPENAI_BASE_URL", "http://203.0.113.1/v1")
	t.Setenv("OPENAI_ORG_ID", "env-org")
	t.Setenv("OPENAI_PROJECT_ID", "env-project")
	t.Setenv("OPENAI_CUSTOM_HEADERS", "X-From-Env: 1\nX-App: from-env")

	srv := llmtest.NewServer(t)
	srv.Enqueue(llmtest.Text("ok"))

	c := newClient(t, srv, llm.WithHeader("X-App", "from-option"))
	_, err := c.Complete(t.Context(), ask("hi"))
	require.NoError(t, err, "the request reached the configured server, not OPENAI_BASE_URL")

	h := srv.Requests()[0].Header
	assert.Empty(t, h.Get("Authorization"))
	assert.Empty(t, h.Get("OpenAI-Organization"))
	assert.Empty(t, h.Get("OpenAI-Project"))
	assert.Empty(t, h.Get("X-From-Env"))
	assert.Equal(t, "from-option", h.Get("X-App"))

	for name := range h {
		assert.False(t, strings.HasPrefix(name, "X-Stainless-"), "%s reports the host to the server", name)
	}
}

func TestStreamYieldsTextThenDone(t *testing.T) {
	srv := llmtest.NewServer(t)
	srv.Enqueue(llmtest.Reply{Text: "Bedroom lights at 40%.", Usage: llm.Usage{InputTokens: 12, OutputTokens: 6}})

	events := collect(t, newClient(t, srv), ask("dim the bedroom"))

	var text strings.Builder
	for _, e := range events {
		if d, ok := e.(llm.TextDelta); ok {
			text.WriteString(d.Text)
		}
	}

	assert.Equal(t, "Bedroom lights at 40%.", text.String())

	done, ok := last(t, events).(llm.Done)
	require.True(t, ok, "the stream ends with Done")
	assert.Equal(t, "Bedroom lights at 40%.", done.Response.Message.Content)
	assert.Equal(t, llm.StopEnd, done.Response.Stop)
	assert.Equal(t, llm.Usage{InputTokens: 12, OutputTokens: 6}, done.Response.Usage)

	opts, _ := srv.Requests()[0].Field("stream_options").(map[string]any)
	assert.Equal(t, true, opts["include_usage"])
}

func TestStreamAssemblesToolCallsFromFragments(t *testing.T) {
	srv := llmtest.NewServer(t)
	srv.Enqueue(llmtest.Reply{
		ToolCalls: []llm.ToolCall{
			{ID: "call_a", Name: "get_device", Arguments: `{"id":42}`},
			{ID: "call_b", Name: "list_events", Arguments: `{"since":"24h","kind":"DEVICE_NEW"}`},
		},
		ChunkSize: 2,
	})

	events := collect(t, newClient(t, srv), ask("what changed?"))

	var started []llm.ToolCallStarted
	var doneCalls []llm.ToolCall
	args := map[int]string{}

	for _, e := range events {
		switch e := e.(type) {
		case llm.ToolCallStarted:
			started = append(started, e)
		case llm.ToolCallDelta:
			args[e.Index] += e.Arguments
		case llm.ToolCallDone:
			doneCalls = append(doneCalls, e.Call)
		}
	}

	assert.Equal(t, []llm.ToolCallStarted{{Index: 0, ID: "call_a", Name: "get_device"}, {Index: 1, ID: "call_b", Name: "list_events"}}, started)
	assert.Equal(t, `{"id":42}`, args[0])
	assert.Equal(t, `{"since":"24h","kind":"DEVICE_NEW"}`, args[1])

	done := last(t, events).(llm.Done)
	assert.Equal(t, llm.StopToolCalls, done.Response.Stop)
	assert.Equal(t, doneCalls, done.Response.Message.ToolCalls)
	assert.Len(t, doneCalls, 2)
}

func TestStreamKeepsReasoningUnderEitherName(t *testing.T) {
	srv := llmtest.NewServer(t)
	srv.Enqueue(
		llmtest.Reply{Reasoning: "The user wants a count.", Text: "3"},
		llmtest.Reply{Raw: llmtest.SSE(
			chunk(map[string]any{"reasoning": "Counting."}, nil),
			chunk(map[string]any{"content": "3"}, "stop"),
			"[DONE]",
		)},
	)

	c := newClient(t, srv)

	for range 2 {
		resp, err := llm.Collect(c.Stream(t.Context(), ask("how many?")))
		require.NoError(t, err)
		assert.NotEmpty(t, resp.Message.Reasoning)
		assert.Equal(t, "3", resp.Message.Content)
	}
}

// chunk is one streamed chunk with delta and finish as its only choice.
func chunk(delta map[string]any, finish any) map[string]any {
	return map[string]any{
		"id":      "chatcmpl-test",
		"object":  "chat.completion.chunk",
		"created": 0,
		"model":   "test-model",
		"choices": []map[string]any{{"index": 0, "delta": delta, "finish_reason": finish}},
	}
}

func TestStreamToleratesLocalServerToolCalls(t *testing.T) {
	tests := []struct {
		name  string
		calls []map[string]any
	}{
		{
			name: "no index, no ID, arguments as an object",
			calls: []map[string]any{
				{"function": map[string]any{"name": "lights_on", "arguments": map[string]any{"room": "bedroom"}}},
				{"function": map[string]any{"name": "lights_on", "arguments": map[string]any{"room": "hall"}}},
			},
		},
		{
			name: "every call at index 0",
			calls: []map[string]any{
				{"index": 0, "id": "call_x", "function": map[string]any{"name": "lights_on", "arguments": `{"room":"bedroom"}`}},
				{"index": 0, "id": "call_y", "function": map[string]any{"name": "lights_on", "arguments": `{"room":"hall"}`}},
			},
		},
		{
			name: "every call at index 0, arguments in pieces",
			calls: []map[string]any{
				{"index": 0, "id": "call_x", "function": map[string]any{"name": "lights_on", "arguments": `{"room":`}},
				{"index": 0, "function": map[string]any{"arguments": `"bedroom"}`}},
				{"index": 0, "id": "call_y", "function": map[string]any{"name": "lights_on", "arguments": `{"room":`}},
				{"index": 0, "function": map[string]any{"arguments": `"hall"}`}},
			},
		},
		{
			name: "no index, arguments in pieces",
			calls: []map[string]any{
				{"function": map[string]any{"name": "lights_on", "arguments": `{"room":`}},
				{"function": map[string]any{"arguments": `"bedroom"}`}},
				{"function": map[string]any{"name": "lights_on", "arguments": `{"room":`}},
				{"function": map[string]any{"arguments": `"hall"}`}},
			},
		},
		{
			name: "indexes out of range",
			calls: []map[string]any{
				{"index": -1, "function": map[string]any{"name": "lights_on", "arguments": `{"room":"bedroom"}`}},
				{"index": 2000000000, "function": map[string]any{"name": "lights_on", "arguments": `{"room":"hall"}`}},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := llmtest.NewServer(t)

			events := []any{}
			for _, call := range tt.calls {
				events = append(events, chunk(map[string]any{"tool_calls": []map[string]any{call}}, nil))
			}

			// Some servers finish a tool-calling reply with "stop".
			events = append(events, chunk(map[string]any{}, "stop"), "[DONE]")
			srv.Enqueue(llmtest.Reply{Raw: llmtest.SSE(events...)})

			resp, err := llm.Collect(newClient(t, srv).Stream(t.Context(), ask("lights on")))
			require.NoError(t, err)

			require.Len(t, resp.Message.ToolCalls, 2)
			assert.Equal(t, llm.StopToolCalls, resp.Stop)

			rooms := []string{}
			for _, call := range resp.Message.ToolCalls {
				assert.NotEmpty(t, call.ID, "every call gets an ID its result can name")
				assert.NotEqual(t, resp.Message.ToolCalls[0].ID, resp.Message.ToolCalls[1].ID)

				var args struct{ Room string }
				require.NoError(t, call.DecodeArguments(&args))
				rooms = append(rooms, args.Room)
			}

			assert.Equal(t, []string{"bedroom", "hall"}, rooms)
		})
	}
}

func TestStreamReportsAnErrorSentMidStream(t *testing.T) {
	srv := llmtest.NewServer(t)
	srv.Enqueue(llmtest.Reply{Raw: llmtest.SSE(
		chunk(map[string]any{"content": "The bedroom"}, nil),
		map[string]any{"error": map[string]any{"message": "upstream model failed", "code": "server_error"}},
	)})

	failed, ok := last(t, collect(t, newClient(t, srv), ask("status?"))).(llm.Failed)
	require.True(t, ok)

	apiErr, ok := errors.AsType[*llm.APIError](failed.Err)
	require.True(t, ok, "got %v", failed.Err)
	assert.Equal(t, "upstream model failed", apiErr.Message)
	assert.Equal(t, "server_error", apiErr.Code)
	assert.True(t, llm.IsRetryable(failed.Err), "a server error with no status can be retried")

	require.NotNil(t, failed.Partial)
	assert.Equal(t, "The bedroom", failed.Partial.Message.Content)
	assert.Equal(t, llm.StopFailed, failed.Partial.Stop)
}

func TestStreamThatEndsEarlyIsIncomplete(t *testing.T) {
	srv := llmtest.NewServer(t)
	srv.Enqueue(llmtest.Reply{Raw: llmtest.SSE(chunk(map[string]any{"content": "Half an answer"}, nil))})

	resp, err := llm.Collect(newClient(t, srv).Stream(t.Context(), ask("status?")))
	require.ErrorIs(t, err, llm.ErrIncompleteStream)
	assert.Equal(t, "Half an answer", resp.Message.Content)
}

func TestStoppingAStreamEndsIt(t *testing.T) {
	srv := llmtest.NewServer(t)
	srv.Enqueue(llmtest.Reply{Text: "a long answer the caller does not want", ChunkSize: 1})

	n := 0
	for e := range newClient(t, srv).Stream(t.Context(), ask("hi")) {
		if _, ok := e.(llm.TextDelta); ok {
			n++
			break
		}
	}

	assert.Equal(t, 1, n)
}

func TestStreamRetriesARateLimitBeforeTheReply(t *testing.T) {
	srv := llmtest.NewServer(t)

	limited := llmtest.Error(http.StatusTooManyRequests, "slow down")
	limited.Header = http.Header{"Retry-After-Ms": {"1"}}
	srv.Enqueue(limited, llmtest.Text("ok"))

	events := collect(t, newClient(t, srv), ask("hi"))

	retry, ok := events[0].(llm.Retrying)
	require.True(t, ok, "the first event is the retry, got %T", events[0])
	assert.Equal(t, 1, retry.Attempt)

	apiErr, ok := errors.AsType[*llm.APIError](retry.Err)
	require.True(t, ok)
	assert.Equal(t, http.StatusTooManyRequests, apiErr.StatusCode)

	_, ok = last(t, events).(llm.Done)
	assert.True(t, ok)
	assert.Len(t, srv.Requests(), 2)
}

func TestARefusalIsNotRetried(t *testing.T) {
	srv := llmtest.NewServer(t)
	srv.Enqueue(llmtest.Error(http.StatusBadRequest, "This model's maximum context length is 8192 tokens."))

	_, err := newClient(t, srv).Complete(t.Context(), ask("hi"))

	apiErr, ok := errors.AsType[*llm.APIError](err)
	require.True(t, ok, "got %v", err)
	assert.Equal(t, http.StatusBadRequest, apiErr.StatusCode)
	assert.True(t, llm.IsContextOverflow(err))
	assert.False(t, llm.IsRetryable(err))
	assert.Len(t, srv.Requests(), 1)
}

func TestErrorsFromServersOtherThanOpenAIKeepTheirMessage(t *testing.T) {
	tests := []struct {
		name   string
		status int
		body   string
		want   string
	}{
		{"Ollama", http.StatusNotFound, `{"error":"model \"tiny\" not found, try pulling it first"}`, `model "tiny" not found, try pulling it first`},
		{"FastAPI", http.StatusUnprocessableEntity, `{"detail":"temperature must be at most 2"}`, "temperature must be at most 2"},
		{"plain text", http.StatusBadRequest, "upstream refused the request", "upstream refused the request"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := llmtest.NewServer(t)
			srv.Enqueue(llmtest.Reply{Status: tt.status, Body: tt.body})

			_, err := newClient(t, srv).Complete(t.Context(), ask("hi"))

			apiErr, ok := errors.AsType[*llm.APIError](err)
			require.True(t, ok, "got %v", err)
			assert.Equal(t, tt.status, apiErr.StatusCode)
			assert.Equal(t, tt.want, apiErr.Message)
		})
	}
}

func TestRequestCarriesEveryField(t *testing.T) {
	srv := llmtest.NewServer(t)
	srv.Enqueue(llmtest.Text("ok"))

	c := newClient(t, srv, llm.WithCompat(llm.Compat{MaxCompletionTokens: true, ToolResultNames: true}))

	call := llm.ToolCall{ID: "call_1", Name: "get_stats"}
	_, err := c.Complete(t.Context(), llm.Request{
		Model: "test-model",
		Messages: []llm.Message{
			llm.System("Be brief."),
			llm.UserWithImages("What is this?", llm.Image{Data: []byte("png"), MediaType: "image/png"}),
			{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{call}},
			llm.ToolResult(call, `{"total":3}`),
		},
		Tools:          []llm.ToolDef{{Name: "get_stats", Description: "Count the devices."}},
		ToolChoice:     llm.ToolChoiceAuto,
		Temperature:    new(0.0),
		MaxTokens:      256,
		Reasoning:      llm.EffortHigh,
		ResponseFormat: &llm.ResponseFormat{Name: "answer", Schema: json.RawMessage(`{"type":"object"}`)},
		Extra:          map[string]any{"keep_alive": "5m"},
	})
	require.NoError(t, err)

	var body struct {
		Messages []struct {
			Role       string          `json:"role"`
			Content    json.RawMessage `json:"content"`
			ToolCalls  []any           `json:"tool_calls"`
			ToolCallID string          `json:"tool_call_id"`
			Name       string          `json:"name"`
		} `json:"messages"`
		Tools []struct {
			Function struct {
				Name       string         `json:"name"`
				Parameters map[string]any `json:"parameters"`
			} `json:"function"`
		} `json:"tools"`
		ToolChoice          string   `json:"tool_choice"`
		Temperature         *float64 `json:"temperature"`
		MaxTokens           *int     `json:"max_tokens"`
		MaxCompletionTokens int      `json:"max_completion_tokens"`
		ReasoningEffort     string   `json:"reasoning_effort"`
		ResponseFormat      struct {
			Type string `json:"type"`
		} `json:"response_format"`
		KeepAlive string `json:"keep_alive"`
	}

	require.NoError(t, srv.Requests()[0].Decode(&body))

	require.Len(t, body.Messages, 4)
	assert.JSONEq(t, `"Be brief."`, string(body.Messages[0].Content))
	assert.JSONEq(t, `[{"type":"text","text":"What is this?"},{"type":"image_url","image_url":{"url":"data:image/png;base64,cG5n"}}]`, string(body.Messages[1].Content))
	assert.Len(t, body.Messages[2].ToolCalls, 1)
	assert.Equal(t, "call_1", body.Messages[3].ToolCallID)
	assert.Equal(t, "get_stats", body.Messages[3].Name)

	require.Len(t, body.Tools, 1)
	assert.Equal(t, map[string]any{"type": "object", "properties": map[string]any{}}, body.Tools[0].Function.Parameters)

	assert.Equal(t, "auto", body.ToolChoice)
	require.NotNil(t, body.Temperature, "a zero temperature is still sent")
	assert.Zero(t, *body.Temperature)
	assert.Nil(t, body.MaxTokens)
	assert.Equal(t, 256, body.MaxCompletionTokens)
	assert.Equal(t, "high", body.ReasoningEffort)
	assert.Equal(t, "json_schema", body.ResponseFormat.Type)
	assert.Equal(t, "5m", body.KeepAlive)
}

func TestReasoningIsSentInTheServersFormat(t *testing.T) {
	tests := []struct {
		format llm.ReasoningFormat
		effort llm.Effort
		field  string
		want   any
	}{
		{llm.ReasoningEffortField, llm.EffortLow, "reasoning_effort", "low"},
		{llm.ReasoningEffortField, llm.EffortOff, "reasoning_effort", "none"},
		{llm.ReasoningObject, llm.EffortMedium, "reasoning", map[string]any{"effort": "medium"}},
		{llm.ReasoningObject, llm.EffortOff, "reasoning", map[string]any{"enabled": false}},
		{llm.ReasoningThink, llm.EffortHigh, "think", true},
		{llm.ReasoningThink, llm.EffortOff, "think", false},
		{llm.ReasoningTemplate, llm.EffortMinimal, "chat_template_kwargs", map[string]any{"enable_thinking": true}},
		{llm.ReasoningTemplate, llm.EffortOff, "chat_template_kwargs", map[string]any{"enable_thinking": false}},
	}

	for _, tt := range tests {
		t.Run(string(tt.format)+"/"+string(tt.effort), func(t *testing.T) {
			srv := llmtest.NewServer(t)
			srv.Enqueue(llmtest.Text("ok"))

			req := ask("hi")
			req.Reasoning = tt.effort

			_, err := newClient(t, srv, llm.WithCompat(llm.Compat{Reasoning: tt.format})).Complete(t.Context(), req)
			require.NoError(t, err)

			assert.Equal(t, tt.want, srv.Requests()[0].Field(tt.field))
		})
	}
}

func TestARequestWithNoModelIsRefusedBeforeSending(t *testing.T) {
	srv := llmtest.NewServer(t)

	_, err := newClient(t, srv).Complete(t.Context(), llm.Request{Messages: []llm.Message{llm.User("hi")}})
	require.Error(t, err)

	failed, ok := last(t, collect(t, newClient(t, srv), llm.Request{})).(llm.Failed)
	require.True(t, ok)
	assert.Error(t, failed.Err)
	assert.Nil(t, failed.Partial)
	assert.Empty(t, srv.Requests())
}

func TestDecodeArgumentsNamesTheTool(t *testing.T) {
	var v struct{ ID int }

	err := llm.ToolCall{Name: "get_device", Arguments: `{"id":`}.DecodeArguments(&v)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "get_device")
}

func TestToolCallEventsNumberCallsInReplyOrder(t *testing.T) {
	srv := llmtest.NewServer(t)
	srv.Enqueue(llmtest.Reply{Raw: llmtest.SSE(
		chunk(map[string]any{"tool_calls": []map[string]any{{"index": 1, "id": "call_a", "function": map[string]any{"name": "get_device", "arguments": `{"id":42}`}}}}, nil),
		chunk(map[string]any{"tool_calls": []map[string]any{{"index": 3, "id": "call_b", "function": map[string]any{"name": "get_device", "arguments": `{"id":7}`}}}}, nil),
		chunk(map[string]any{}, "tool_calls"),
		"[DONE]",
	)})

	var started, deltas, done []int

	for _, e := range collect(t, newClient(t, srv), ask("which devices?")) {
		switch e := e.(type) {
		case llm.ToolCallStarted:
			started = append(started, e.Index)
		case llm.ToolCallDelta:
			deltas = append(deltas, e.Index)
		case llm.ToolCallDone:
			done = append(done, e.Index)
		}
	}

	assert.Equal(t, []int{0, 1}, started)
	assert.Equal(t, []int{0, 1}, deltas)
	assert.Equal(t, []int{0, 1}, done)
}

func TestStreamTakesAWholeReply(t *testing.T) {
	srv := llmtest.NewServer(t)
	srv.Enqueue(llmtest.Reply{
		Whole:     true,
		Text:      "Turning the lights on.",
		ToolCalls: []llm.ToolCall{{ID: "call_a", Name: "lights_on", Arguments: `{"room":"hall"}`}},
		Usage:     llm.Usage{InputTokens: 12, OutputTokens: 6},
	})

	events := collect(t, newClient(t, srv), ask("lights on"))

	assert.Contains(t, events, llm.TextDelta{Text: "Turning the lights on."})
	assert.Contains(t, events, llm.ToolCallStarted{Index: 0, ID: "call_a", Name: "lights_on"})

	done, ok := last(t, events).(llm.Done)
	require.True(t, ok, "got %#v", last(t, events))
	assert.Equal(t, "Turning the lights on.", done.Response.Message.Content)
	assert.Equal(t, []llm.ToolCall{{ID: "call_a", Name: "lights_on", Arguments: `{"room":"hall"}`}}, done.Response.Message.ToolCalls)
	assert.Equal(t, llm.StopToolCalls, done.Response.Stop)
	assert.Equal(t, llm.Usage{InputTokens: 12, OutputTokens: 6}, done.Response.Usage)
}

func TestAnErrorInsideA200TakesTheStatusItsCodeNames(t *testing.T) {
	t.Run("in place of a reply", func(t *testing.T) {
		srv := llmtest.NewServer(t)
		srv.Enqueue(llmtest.Reply{Raw: `{"id":"gen-1","error":{"message":"upstream provider failed","code":502}}`})

		_, err := newClient(t, srv).Complete(t.Context(), ask("status?"))

		apiErr, ok := errors.AsType[*llm.APIError](err)
		require.True(t, ok, "got %v", err)
		assert.Equal(t, http.StatusBadGateway, apiErr.StatusCode)
		assert.Equal(t, "upstream provider failed", apiErr.Message)
		assert.True(t, llm.IsRetryable(err))
	})

	t.Run("in a stream", func(t *testing.T) {
		srv := llmtest.NewServer(t)
		srv.Enqueue(llmtest.Reply{Raw: llmtest.SSE(
			map[string]any{"error": map[string]any{"message": "This model's maximum context length is 8192 tokens", "code": "context_length_exceeded"}},
		)})

		_, err := llm.Collect(newClient(t, srv).Stream(t.Context(), ask("status?")))

		apiErr, ok := errors.AsType[*llm.APIError](err)
		require.True(t, ok, "got %v", err)
		assert.Zero(t, apiErr.StatusCode, "a code that is not a status names none")
		assert.True(t, llm.IsContextOverflow(err))
	})

	t.Run("no error and no reply", func(t *testing.T) {
		srv := llmtest.NewServer(t)
		srv.Enqueue(llmtest.Reply{Raw: `{"id":"gen-1","choices":[]}`})

		_, err := newClient(t, srv).Complete(t.Context(), ask("status?"))
		require.ErrorIs(t, err, llm.ErrEmptyResponse)
	})
}

func TestPartialReplyLeavesOutUnfinishedToolCalls(t *testing.T) {
	call := map[string]any{"tool_calls": []map[string]any{{"index": 0, "function": map[string]any{"name": "lights_on", "arguments": `{"room":`}}}}

	t.Run("with text", func(t *testing.T) {
		srv := llmtest.NewServer(t)
		srv.Enqueue(llmtest.Reply{Raw: llmtest.SSE(chunk(map[string]any{"content": "Checking the hall."}, nil), chunk(call, nil))})

		resp, err := llm.Collect(newClient(t, srv).Stream(t.Context(), ask("lights on")))
		require.ErrorIs(t, err, llm.ErrIncompleteStream)
		require.NotNil(t, resp)
		assert.Equal(t, "Checking the hall.", resp.Message.Content)
		assert.Empty(t, resp.Message.ToolCalls)
	})

	t.Run("with only a tool call", func(t *testing.T) {
		srv := llmtest.NewServer(t)
		srv.Enqueue(llmtest.Reply{Raw: llmtest.SSE(chunk(call, nil))})

		failed, ok := last(t, collect(t, newClient(t, srv), ask("lights on"))).(llm.Failed)
		require.True(t, ok)
		assert.Nil(t, failed.Partial)
	})
}

func TestAssistantMessagesAreSentTheWayServersAccept(t *testing.T) {
	srv := llmtest.NewServer(t)
	srv.Enqueue(llmtest.Text("Done."))

	req := ask("lights on")
	req.Messages = append(req.Messages,
		llm.Message{Role: llm.RoleAssistant, Reasoning: "The user wants light."},
		llm.Message{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "call_a", Name: "lights_on", Arguments: "  "}}},
		llm.ToolResult(llm.ToolCall{ID: "call_a", Name: "lights_on"}, "on"),
	)

	_, err := newClient(t, srv).Complete(t.Context(), req)
	require.NoError(t, err)

	var body struct {
		Messages []map[string]any `json:"messages"`
	}
	require.NoError(t, srv.Requests()[0].Decode(&body))
	require.Len(t, body.Messages, 4)

	content, ok := body.Messages[1]["content"]
	assert.True(t, ok, "a message with neither text nor calls still sends content")
	assert.Equal(t, "", content)

	assert.NotContains(t, body.Messages[2], "content", "a message that only calls tools leaves content out")

	calls, _ := body.Messages[2]["tool_calls"].([]any)
	require.Len(t, calls, 1)
	fn, _ := calls[0].(map[string]any)["function"].(map[string]any)
	assert.Equal(t, "{}", fn["arguments"])
}
