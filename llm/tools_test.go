package llm_test

import (
	"slices"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/pushkar-anand/astromech/llm"
)

const okReply = `{"id":"r1","model":"model-a","choices":[
	{"finish_reason":"stop","message":{"role":"assistant","content":"ok"}}
]}`

func TestRequestSendsTools(t *testing.T) {
	b, c := newBackend(t, map[string]reply{
		"POST /v1/chat/completions": {body: okReply},
	})

	_, err := c.Complete(t.Context(), llm.Request{
		Model: "model-a",
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: "What time is it in UTC?"},
			// The model's earlier turn, sent back: a call and no text.
			{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{
				{ID: "call_1", Name: "get_time", Arguments: `{"tz":"UTC"}`},
			}},
			{Role: llm.RoleTool, ToolCallID: "call_1", Content: "12:00"},
		},
		Tools: []llm.Tool{{
			Name:        "get_time",
			Description: "The current time",
			Parameters: map[string]any{
				"type":       "object",
				"properties": map[string]any{"tz": map[string]any{"type": "string"}},
			},
		}},
	})
	require.NoError(t, err)

	assert.Equal(t, map[string]any{
		"model": "model-a",
		"messages": []any{
			map[string]any{"role": "user", "content": "What time is it in UTC?"},
			map[string]any{"role": "assistant", "tool_calls": []any{
				map[string]any{"id": "call_1", "type": "function", "function": map[string]any{
					"name": "get_time", "arguments": `{"tz":"UTC"}`,
				}},
			}},
			map[string]any{"role": "tool", "tool_call_id": "call_1", "content": "12:00"},
		},
		"tools": []any{
			map[string]any{"type": "function", "function": map[string]any{
				"name":        "get_time",
				"description": "The current time",
				"parameters": map[string]any{
					"type":       "object",
					"properties": map[string]any{"tz": map[string]any{"type": "string"}},
				},
			}},
		},
	}, b.lastBody())
}

func TestRequestToolDefaults(t *testing.T) {
	b, c := newBackend(t, map[string]reply{
		"POST /v1/chat/completions": {body: okReply},
	})

	_, err := c.Complete(t.Context(), llm.Request{
		Model: "model-a",
		Messages: []llm.Message{
			{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "call_1", Name: "now"}}},
			{Role: llm.RoleTool, ToolCallID: "call_1", Content: "12:00"},
		},
		Tools: []llm.Tool{{Name: "now"}},
	})
	require.NoError(t, err)

	body := b.lastBody()

	call := body["messages"].([]any)[0].(map[string]any)["tool_calls"].([]any)[0].(map[string]any)
	assert.Equal(t, "{}", call["function"].(map[string]any)["arguments"], "empty arguments")

	tool := body["tools"].([]any)[0].(map[string]any)["function"].(map[string]any)
	assert.Equal(t, map[string]any{"type": "object", "properties": map[string]any{}}, tool["parameters"], "nil parameters")
	assert.NotContains(t, tool, "description")
}

func TestStreamSendsTools(t *testing.T) {
	b, c := newBackend(t, map[string]reply{
		"POST /v1/chat/completions": {body: sse(chunkText, chunkStop)},
	})

	for range c.Stream(t.Context(), llm.Request{
		Model:    "model-a",
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "Hello"}},
		Tools:    []llm.Tool{{Name: "now"}},
	}) {
	}

	body := b.lastBody()
	assert.Equal(t, true, body["stream"])
	require.Len(t, body["tools"], 1)
	assert.Equal(t, "now", body["tools"].([]any)[0].(map[string]any)["function"].(map[string]any)["name"])
}

func TestCompleteReadsToolCalls(t *testing.T) {
	_, c := newBackend(t, map[string]reply{
		"POST /v1/chat/completions": {body: `{"id":"r1","model":"model-a","choices":[{"finish_reason":"tool_calls","message":{
			"role": "assistant",
			"content": null,
			"tool_calls": [
				{"id":"call_1","type":"function","function":{"name":"get_time","arguments":"{\"zone\":\"Asia/Tokyo\"}"}},
				{"id":"call_2","type":"function","function":{"name":"get_time","arguments":"{\"zone\":\"Europe/London\"}"}}
			]
		}}]}`},
	})

	resp, err := c.Complete(t.Context(), hello)
	require.NoError(t, err)

	assert.Equal(t, "tool_calls", resp.FinishReason)
	assert.Equal(t, llm.Message{
		Role: llm.RoleAssistant,
		ToolCalls: []llm.ToolCall{
			{ID: "call_1", Name: "get_time", Arguments: `{"zone":"Asia/Tokyo"}`},
			{ID: "call_2", Name: "get_time", Arguments: `{"zone":"Europe/London"}`},
		},
	}, resp.Message)
}

func TestCompleteToolCallQuirks(t *testing.T) {
	_, c := newBackend(t, map[string]reply{
		// No ids, and empty arguments for a tool that takes none.
		"POST /v1/chat/completions": {body: `{"choices":[{"finish_reason":"tool_calls","message":{
			"role": "assistant",
			"tool_calls": [
				{"type":"function","function":{"name":"now","arguments":""}},
				{"type":"function","function":{"name":"get_time","arguments":"{\"zone\":\"UTC\"}"}}
			]
		}}]}`},
	})

	resp, err := c.Complete(t.Context(), hello)
	require.NoError(t, err)

	assert.Equal(t, []llm.ToolCall{
		{ID: "call_0", Name: "now", Arguments: "{}"},
		{ID: "call_1", Name: "get_time", Arguments: `{"zone":"UTC"}`},
	}, resp.Message.ToolCalls)
}

func TestCompleteTextAndToolCall(t *testing.T) {
	_, c := newBackend(t, map[string]reply{
		"POST /v1/chat/completions": {body: `{"choices":[{"finish_reason":"tool_calls","message":{
			"role": "assistant",
			"content": "Let me check.",
			"tool_calls": [{"id":"call_1","type":"function","function":{"name":"now","arguments":"{}"}}]
		}}]}`},
	})

	resp, err := c.Complete(t.Context(), hello)
	require.NoError(t, err)

	assert.Equal(t, "Let me check.", resp.Message.Content)
	assert.Equal(t, []llm.ToolCall{{ID: "call_1", Name: "now", Arguments: "{}"}}, resp.Message.ToolCalls)
}

// The agent loop sends the model's reply back as it came: it must turn into
// the same tool calls on the wire.
func TestCompleteToolCallRoundTrip(t *testing.T) {
	b, c := newBackend(t, map[string]reply{
		"POST /v1/chat/completions": {body: `{"choices":[{"finish_reason":"tool_calls","message":{
			"role": "assistant",
			"content": null,
			"tool_calls": [{"id":"call_1","type":"function","function":{"name":"get_time","arguments":"{\"zone\":\"UTC\"}"}}]
		}}]}`},
	})

	first, err := c.Complete(t.Context(), hello)
	require.NoError(t, err)
	require.Len(t, first.Message.ToolCalls, 1)

	history := append(slices.Clone(hello.Messages),
		first.Message,
		llm.Message{Role: llm.RoleTool, ToolCallID: first.Message.ToolCalls[0].ID, Content: "12:00"},
	)

	_, err = c.Complete(t.Context(), llm.Request{Model: "model-a", Messages: history})
	require.NoError(t, err)

	assert.Equal(t, []any{
		map[string]any{"role": "user", "content": "Hello"},
		map[string]any{"role": "assistant", "tool_calls": []any{
			map[string]any{"id": "call_1", "type": "function", "function": map[string]any{
				"name": "get_time", "arguments": `{"zone":"UTC"}`,
			}},
		}},
		map[string]any{"role": "tool", "tool_call_id": "call_1", "content": "12:00"},
	}, b.lastBody()["messages"])
}
