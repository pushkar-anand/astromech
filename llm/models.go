package llm

import (
	"context"
	"fmt"
)

type (
	Model struct {
		ID            string `json:"id"`
		Name          string `json:"name"`
		Created       int    `json:"created"`
		Description   string `json:"description"`
		ContextLength int    `json:"context_length,omitempty"`
	}

	modelList struct {
		Data []*Model `json:"data"`
	}
)

func (c *Client) Models(ctx context.Context) ([]*Model, error) {
	list, err := c.httpClient.Get[modelList](ctx, modelsPath)
	if err != nil {
		return nil, fmt.Errorf("llm: failed to get models: %w", err)
	}

	return list.Data, nil
}
