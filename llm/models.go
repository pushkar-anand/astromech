package llm

import (
	"context"
	"fmt"
)

type (
	// Model is a model the LLM backend offers.
	Model struct {
		// ID is the model's ID, this is the name the other APIs accept.
		ID          string `json:"id"`
		Name        string `json:"name,omitzero"`
		Description string `json:"description,omitzero"`

		// Created is when the model was added, in Unix seconds.
		Created int64 `json:"created,omitzero"`

		// ContextLength is how many tokens the model holds at once.
		ContextLength int `json:"context_length,omitzero"`
	}

	modelList struct {
		Data []Model `json:"data"`
	}
)

// Models lists the models the LLM backend offers.
func (c *Client) Models(ctx context.Context) ([]Model, error) {
	list, err := c.httpClient.Get[modelList](ctx, modelsPath)
	if err != nil {
		return nil, fmt.Errorf("llm: list models: %w", err)
	}

	return list.Data, nil
}
