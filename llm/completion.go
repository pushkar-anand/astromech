package llm

import (
	"context"
	"fmt"
)

type (
	Message struct {
		Role    Role   `json:"role"`
		Content string `json:"content"`
	}

	CompletionRequest struct {
		Model    string    `json:"model"`
		Messages []Message `json:"messages"`
	}
)

type (
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

	Usage struct {
		CompletionTokens    int `json:"completion_tokens"`
		PromptTokens        int `json:"prompt_tokens"`
		TotalTokens         int `json:"total_tokens"`
		PromptTokensDetails struct {
			CachedTokens int `json:"cached_tokens"`
		} `json:"prompt_tokens_details"`
	}

	ResponseMessage struct {
		Message
		ReasoningContent string `json:"reasoning_content"`
	}

	Choice struct {
		FinishReason string          `json:"finish_reason"`
		Index        int             `json:"index"`
		Message      ResponseMessage `json:"message"`
	}

	CompletionResponse struct {
		ID      string   `json:"id"`
		Model   string   `json:"model"`
		Choices []Choice `json:"choices"`
		Stats   *Stats   `json:"timings,omitzero"`
		Usage   Usage    `json:"usage"`
	}
)

func (c *Client) Complete(ctx context.Context, msg CompletionRequest) (*CompletionResponse, error) {
	result, err := c.httpClient.Post[CompletionResponse, CompletionRequest](ctx, chatCompletionPath, msg)
	if err != nil {
		return nil, fmt.Errorf("llm: failed to complete: %w", err)
	}

	if len(result.Choices) == 0 {
		return nil, fmt.Errorf("llm: no choices returned")
	}

	return &result, nil
}
