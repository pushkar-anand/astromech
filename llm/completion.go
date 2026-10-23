package llm

import (
	"context"
	"fmt"
)

// Complete sends req and waits for the whole reply.
func (c *Client) Complete(ctx context.Context, req Request) (*Response, error) {
	reply, err := c.httpClient.Post[completion](ctx, chatCompletionPath, req.toWire())
	if err != nil {
		return nil, fmt.Errorf("llm: chat completion: %w", explain(err))
	}

	if len(reply.Choices) == 0 {
		return nil, ErrNoReply
	}

	first := reply.Choices[0]

	return &Response{
		ID:    reply.ID,
		Model: reply.Model,
		Message: Message{
			Role:      RoleAssistant,
			Content:   first.Message.Content,
			ToolCalls: toolCalls(first.Message.ToolCalls),
		},
		Reasoning:    first.Message.thinking(),
		FinishReason: first.FinishReason,
		Usage:        reply.Usage.toUsage(),
		Stats:        reply.Stats,
	}, nil
}
