package llm

import (
	"context"
	"fmt"
)

// Complete sends req and waits for the whole reply.
func (c *Client) Complete(ctx context.Context, req Request) (*Response, error) {
	body, err := req.toWire()
	if err != nil {
		return nil, fmt.Errorf("llm: chat completion: %w", err)
	}

	reply, err := c.httpClient.Post[completion](ctx, chatCompletionPath, body)
	if err != nil {
		return nil, fmt.Errorf("llm: chat completion: %w", explain(err))
	}

	if len(reply.Choices) == 0 {
		return nil, ErrNoReply
	}

	return reply.toResponse(), nil
}
