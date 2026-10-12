package llm

import (
	"context"
	"encoding/json"
	"slices"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

// Model describes a model a server offers, or one a program offers its
// users, for choosing between them and fitting a conversation to the one
// chosen.
type Model struct {
	ID string `json:"id"`

	// Name is what a user is shown. Empty shows the ID.
	Name string `json:"name,omitzero"`

	// ContextWindow is how many tokens of conversation and reply the model
	// holds at once. Zero when it is not known.
	ContextWindow int `json:"context_window,omitzero"`

	// Capabilities says what the model accepts. Nil when it is not known,
	// which is the case for most servers' model lists; a program then takes
	// it from its own configuration.
	Capabilities *Capabilities `json:"capabilities,omitzero"`
}

// Capabilities are what a model accepts beyond plain text.
type Capabilities struct {
	// Images means the model reads [Image] input.
	Images bool `json:"images,omitzero"`

	// Tools means the model can call tools.
	Tools bool `json:"tools,omitzero"`

	// Reasoning means the model thinks before it answers and takes an
	// [Effort].
	Reasoning bool `json:"reasoning,omitzero"`
}

// Models lists the models the server offers.
//
// Every server reports IDs. Some also report more, and Models reads what
// they do: the context window from OpenRouter's context_length or vLLM's
// max_model_len, and capabilities from OpenRouter's input modalities and
// supported parameters.
func (c *Client) Models(ctx context.Context) ([]Model, error) {
	var att attempts

	pager := c.models.ListAutoPaging(ctx, option.WithMiddleware(att.track))

	var models []Model
	for pager.Next() {
		models = append(models, modelOf(pager.Current()))
	}

	if err := pager.Err(); err != nil {
		return nil, att.explain(ctx, err)
	}

	return models, nil
}

func modelOf(m openai.Model) Model {
	out := Model{ID: m.ID}
	extra := m.JSON.ExtraFields

	var name string
	if json.Unmarshal([]byte(extra["name"].Raw()), &name) == nil {
		out.Name = name
	}

	for _, field := range []string{"context_length", "max_model_len"} {
		var n int
		if json.Unmarshal([]byte(extra[field].Raw()), &n) == nil && n > 0 {
			out.ContextWindow = n

			break
		}
	}

	var arch struct {
		InputModalities []string `json:"input_modalities"`
	}

	var params []string

	hasArch := json.Unmarshal([]byte(extra["architecture"].Raw()), &arch) == nil && arch.InputModalities != nil
	hasParams := json.Unmarshal([]byte(extra["supported_parameters"].Raw()), &params) == nil && params != nil

	if hasArch || hasParams {
		out.Capabilities = &Capabilities{
			Images:    slices.Contains(arch.InputModalities, "image"),
			Tools:     slices.Contains(params, "tools"),
			Reasoning: slices.Contains(params, "reasoning") || slices.Contains(params, "reasoning_effort"),
		}
	}

	return out
}
