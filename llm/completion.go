package llm

import (
	"context"
	"fmt"
)

type (
	// Message is one turn of a conversation.
	Message struct {
		Role    Role   `json:"role"`
		Content string `json:"content"`
	}

	// Request is one call to a model.
	Request struct {
		Model    string    `json:"model"`
		Messages []Message `json:"messages"`
	}

	// Response is a model's reply.
	Response struct {
		// ID and Model are as the LLM backend reported them. Model can
		// differ from the requested one.
		ID    string
		Model string

		Message Message

		// Reasoning is the model's thinking before it answered, empty for
		// models that don't reason.
		Reasoning string

		// FinishReason is why the reply ended, such as "stop" or "length".
		FinishReason string

		// Usage is zero when the LLM backend does not report it.
		Usage Usage

		// Stats is nil when the LLM backend sends no timings.
		Stats *Stats
	}

	// Usage counts the tokens a request used.
	Usage struct {
		CompletionTokens    int `json:"completion_tokens"`
		PromptTokens        int `json:"prompt_tokens"`
		TotalTokens         int `json:"total_tokens"`
		PromptTokensDetails struct {
			CachedTokens int `json:"cached_tokens"`
		} `json:"prompt_tokens_details"`
	}

	// Stats is the timings llama.cpp reports for a request.
	Stats struct {
		CacheN              int     `json:"cache_n"`
		PromptN             int     `json:"prompt_n"`
		PromptMs            float64 `json:"prompt_ms"`
		PromptPerTokenMs    float64 `json:"prompt_per_token_ms"`
		PromptPerSecond     float64 `json:"prompt_per_second"`
		PredictedN          int     `json:"predicted_n"`
		PredictedMs         float64 `json:"predicted_ms"`
		PredictedPerTokenMs float64 `json:"predicted_per_token_ms"`
		PredictedPerSecond  float64 `json:"predicted_per_second"`
	}
)

// Wire format of a reply.
type (
	replyMessage struct {
		Role             Role   `json:"role"`
		Content          string `json:"content"`
		ReasoningContent string `json:"reasoning_content"`
	}

	choice struct {
		FinishReason string       `json:"finish_reason"`
		Message      replyMessage `json:"message"`
	}

	completion struct {
		ID      string   `json:"id"`
		Model   string   `json:"model"`
		Choices []choice `json:"choices"`
		Usage   Usage    `json:"usage"`
		Stats   *Stats   `json:"timings,omitzero"`
	}
)

// Complete sends req and waits for the whole reply.
func (c *Client) Complete(ctx context.Context, req Request) (*Response, error) {
	reply, err := c.httpClient.Post[completion](ctx, chatCompletionPath, req)
	if err != nil {
		return nil, fmt.Errorf("llm: chat completion: %w", err)
	}

	if len(reply.Choices) == 0 {
		return nil, ErrNoReply
	}

	first := reply.Choices[0]

	return &Response{
		ID:           reply.ID,
		Model:        reply.Model,
		Message:      Message{Role: RoleAssistant, Content: first.Message.Content},
		Reasoning:    first.Message.ReasoningContent,
		FinishReason: first.FinishReason,
		Usage:        reply.Usage,
		Stats:        reply.Stats,
	}, nil
}
