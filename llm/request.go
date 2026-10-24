package llm

import (
	"cmp"
	"fmt"
)

// Request is one call to a model.
type Request struct {
	Model    string
	Messages []Message

	// Tools are the tools the model may ask to call.
	Tools []Tool
}

// Wire format of a request.
//
// Tools and ToolCalls take omitempty, not omitzero: toWire makes them empty
// but not nil, and some LLM backends reject "tools": [] or "tool_calls": [].
type (
	chatRequest struct {
		Model    string        `json:"model"`
		Messages []wireMessage `json:"messages"`
		Tools    []wireTool    `json:"tools,omitempty"`
	}

	wireMessage struct {
		Role       Role           `json:"role"`
		Content    wireContent    `json:"content,omitzero"`
		ToolCalls  []wireToolCall `json:"tool_calls,omitempty"`
		ToolCallID string         `json:"tool_call_id,omitzero"`
	}

	wireToolCall struct {
		ID       string       `json:"id"`
		Type     string       `json:"type"`
		Function wireFunction `json:"function"`
	}

	wireFunction struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	}

	wireTool struct {
		Type     string      `json:"type"`
		Function wireToolDef `json:"function"`
	}

	wireToolDef struct {
		Name        string `json:"name"`
		Description string `json:"description,omitzero"`
		Parameters  any    `json:"parameters"`
	}
)

// toWire is r as the LLM backend expects it.
func (r Request) toWire() (chatRequest, error) {
	out := chatRequest{
		Model:    r.Model,
		Messages: make([]wireMessage, len(r.Messages)),
		Tools:    make([]wireTool, len(r.Tools)),
	}

	for i, message := range r.Messages {
		content, err := message.content()
		if err != nil {
			return chatRequest{}, fmt.Errorf("message %d: %w", i, err)
		}

		wm := wireMessage{
			Role:       message.Role,
			Content:    content,
			ToolCallID: message.ToolCallID,
			ToolCalls:  make([]wireToolCall, len(message.ToolCalls)),
		}

		for j, call := range message.ToolCalls {
			wm.ToolCalls[j] = call.toWire()
		}

		out.Messages[i] = wm
	}

	for i, tool := range r.Tools {
		out.Tools[i] = tool.toWire()
	}

	return out, nil
}

func (tc ToolCall) toWire() wireToolCall {
	return wireToolCall{
		ID:   tc.ID,
		Type: "function",
		Function: wireFunction{
			Name: tc.Name,
			// Some LLM backends send "" for a call with no arguments, and
			// reject it when it comes back.
			Arguments: cmp.Or(tc.Arguments, "{}"),
		},
	}
}

// noParameters is the schema sent for a tool without one, since some LLM
// backends reject a tool with none.
var noParameters any = map[string]any{"type": "object", "properties": map[string]any{}}

func (t Tool) toWire() wireTool {
	return wireTool{
		Type: "function",
		Function: wireToolDef{
			Name:        t.Name,
			Description: t.Description,
			Parameters:  cmp.Or(t.Parameters, noParameters),
		},
	}
}
