package llm

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"maps"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/shared"
)

// params maps req onto the SDK's request type.
func (c *Client) params(req Request) (openai.ChatCompletionNewParams, error) {
	if req.Model == "" {
		return openai.ChatCompletionNewParams{}, errors.New("llm: the request names no model; set Request.Model to one the server offers")
	}

	p := openai.ChatCompletionNewParams{
		Model:    shared.ChatModel(req.Model),
		Messages: make([]openai.ChatCompletionMessageParamUnion, 0, len(req.Messages)),
	}

	for i, m := range req.Messages {
		mp, err := c.message(m)
		if err != nil {
			return p, fmt.Errorf("llm: message %d: %w", i, err)
		}

		p.Messages = append(p.Messages, mp)
	}

	for _, t := range req.Tools {
		tp, err := tool(t)
		if err != nil {
			return p, err
		}

		p.Tools = append(p.Tools, tp)
	}

	if req.ToolChoice != "" {
		p.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: openai.String(string(req.ToolChoice))}
	}

	if req.Temperature != nil {
		p.Temperature = openai.Float(*req.Temperature)
	}

	if req.TopP != nil {
		p.TopP = openai.Float(*req.TopP)
	}

	if req.MaxTokens > 0 {
		if c.compat.MaxCompletionTokens {
			p.MaxCompletionTokens = openai.Int(int64(req.MaxTokens))
		} else {
			p.MaxTokens = openai.Int(int64(req.MaxTokens))
		}
	}

	if len(req.Stop) > 0 {
		p.Stop = openai.ChatCompletionNewParamsStopUnion{OfStringArray: req.Stop}
	}

	if req.Seed != nil {
		p.Seed = openai.Int(int64(*req.Seed))
	}

	if f := req.ResponseFormat; f != nil {
		if len(f.Schema) == 0 {
			p.ResponseFormat.OfJSONObject = &shared.ResponseFormatJSONObjectParam{}
		} else {
			p.ResponseFormat.OfJSONSchema = &shared.ResponseFormatJSONSchemaParam{
				JSONSchema: shared.ResponseFormatJSONSchemaJSONSchemaParam{
					Name:   f.Name,
					Schema: f.Schema,
					Strict: openai.Bool(true),
				},
			}
		}
	}

	extra := c.reasoning(&p, req.Reasoning)
	maps.Copy(extra, req.Extra)

	if len(extra) > 0 {
		p.SetExtraFields(extra)
	}

	return p, nil
}

func (c *Client) message(m Message) (openai.ChatCompletionMessageParamUnion, error) {
	switch m.Role {
	case RoleSystem:
		return openai.SystemMessage(m.Content), nil

	case RoleUser:
		if len(m.Images) == 0 {
			return openai.UserMessage(m.Content), nil
		}

		parts := make([]openai.ChatCompletionContentPartUnionParam, 0, len(m.Images)+1)
		if m.Content != "" {
			parts = append(parts, openai.TextContentPart(m.Content))
		}

		for _, img := range m.Images {
			parts = append(parts, openai.ImageContentPart(openai.ChatCompletionContentPartImageImageURLParam{
				URL:    img.url(),
				Detail: img.Detail,
			}))
		}

		return openai.UserMessage(parts), nil

	case RoleAssistant:
		var a openai.ChatCompletionAssistantMessageParam

		// An assistant message that only calls tools leaves content out;
		// some servers refuse an empty string there.
		if m.Content != "" {
			a.Content.OfString = openai.String(m.Content)
		}

		for _, tc := range m.ToolCalls {
			args := tc.Arguments
			if args == "" {
				args = "{}"
			}

			a.ToolCalls = append(a.ToolCalls, openai.ChatCompletionMessageToolCallUnionParam{
				OfFunction: &openai.ChatCompletionMessageFunctionToolCallParam{
					ID:       tc.ID,
					Function: openai.ChatCompletionMessageFunctionToolCallFunctionParam{Name: tc.Name, Arguments: args},
				},
			})
		}

		return openai.ChatCompletionMessageParamUnion{OfAssistant: &a}, nil

	case RoleTool:
		mp := openai.ToolMessage(m.Content, m.ToolCallID)
		if c.compat.ToolResultNames {
			mp.OfTool.SetExtraFields(map[string]any{"name": m.ToolName})
		}

		return mp, nil
	}

	return openai.ChatCompletionMessageParamUnion{}, fmt.Errorf("role %q is not one of system, user, assistant or tool", m.Role)
}

func tool(t ToolDef) (openai.ChatCompletionToolUnionParam, error) {
	fn := shared.FunctionDefinitionParam{Name: t.Name}

	if t.Description != "" {
		fn.Description = openai.String(t.Description)
	}

	if t.Strict {
		fn.Strict = openai.Bool(true)
	}

	// A tool with no parameters still sends an empty object schema, which
	// some servers require.
	fn.Parameters = shared.FunctionParameters{"type": "object", "properties": map[string]any{}}

	if len(t.Parameters) > 0 {
		if err := json.Unmarshal(t.Parameters, &fn.Parameters); err != nil {
			return openai.ChatCompletionToolUnionParam{}, fmt.Errorf("llm: parameters of tool %s are not a JSON object: %w", t.Name, err)
		}
	}

	return openai.ChatCompletionFunctionTool(fn), nil
}

// url returns where the server reads the image from: its URL, or its bytes
// as a data URL.
func (img Image) url() string {
	if img.URL != "" {
		return img.URL
	}

	return "data:" + img.MediaType + ";base64," + base64.StdEncoding.EncodeToString(img.Data)
}

// reasoning sets effort on p in the client's reasoning format. Formats the
// SDK has no field for come back as extra body fields.
func (c *Client) reasoning(p *openai.ChatCompletionNewParams, effort Effort) map[string]any {
	extra := make(map[string]any)
	if effort == "" {
		return extra
	}

	on := effort != EffortOff

	switch c.compat.Reasoning {
	case ReasoningObject:
		if on {
			extra["reasoning"] = map[string]any{"effort": string(effort)}
		} else {
			extra["reasoning"] = map[string]any{"enabled": false}
		}
	case ReasoningThink:
		extra["think"] = on
	case ReasoningTemplate:
		extra["chat_template_kwargs"] = map[string]any{"enable_thinking": on}
	default:
		if on {
			p.ReasoningEffort = shared.ReasoningEffort(effort)
		} else {
			p.ReasoningEffort = shared.ReasoningEffortNone
		}
	}

	return extra
}
