package llm

import (
	"cmp"
	"encoding/base64"
	"strings"
)

// Request is one call to a model.
type Request struct {
	Model    string
	Messages []Message

	// Tools are the tools the model may ask to call.
	Tools []Tool
}

// Wire format of a request.
type (
	imageURL struct {
		URL string `json:"url"`
	}

	inputAudio struct {
		Data   string `json:"data"`   // base64, no "data:" prefix
		Format string `json:"format"` // "wav", "mp3"
	}

	contentPart struct {
		Type       string      `json:"type"`
		Text       string      `json:"text,omitzero"`
		ImageURL   *imageURL   `json:"image_url,omitzero"`
		InputAudio *inputAudio `json:"input_audio,omitzero"`
	}

	chatRequest struct {
		Model    string        `json:"model"`
		Messages []wireMessage `json:"messages"`
		Tools    []wireTool    `json:"tools,omitempty"`
	}

	wireMessage struct {
		Role       Role           `json:"role"`
		Content    any            `json:"content,omitzero"`
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
func (r Request) toWire() chatRequest {
	out := chatRequest{
		Model:    r.Model,
		Messages: make([]wireMessage, len(r.Messages)),
		Tools:    make([]wireTool, len(r.Tools)),
	}

	for i, message := range r.Messages {
		txt := message.Content

		parts := make([]contentPart, 0)

		if len(message.Media) > 0 {
			parts = append(parts, contentPart{
				Type: "text",
				Text: txt,
			})
		}

		for _, media := range message.Media {
			var (
				data string
				cp   contentPart
			)

			if media.Data != nil {
				data = base64.StdEncoding.EncodeToString(media.Data)
			} else {
				data = media.URL
			}

			if strings.HasPrefix(media.MIMEType, "image/") {
				cp.Type = "image_url"
				cp.ImageURL = &imageURL{
					URL: data,
				}
			} else if strings.HasPrefix(media.MIMEType, "audio/") {
				cp.Type = "input_audio"
				cp.InputAudio = &inputAudio{
					Data:   data,
					Format: strings.TrimPrefix(media.MIMEType, "audio/"),
				}
			}

			parts = append(parts, cp)
		}

		wm := wireMessage{
			Role:       message.Role,
			ToolCallID: message.ToolCallID,
			ToolCalls:  make([]wireToolCall, len(message.ToolCalls)),
		}

		if len(parts) == 0 {
			wm.Content = txt
		} else {
			wm.Content = parts
		}

		for j, call := range message.ToolCalls {
			wm.ToolCalls[j] = call.toWire()
		}

		out.Messages[i] = wm
	}

	for i, tool := range r.Tools {
		out.Tools[i] = tool.toWire()
	}

	return out
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
