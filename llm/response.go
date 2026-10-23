package llm

import (
	"cmp"
	"fmt"
)

type (
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
		InputTokens  int
		OutputTokens int
		TotalTokens  int

		// CachedTokens is how many of the input tokens the LLM backend had
		// cached from an earlier request.
		CachedTokens int
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
	replyToolCall struct {
		// Index orders the calls in a stream, where each arrives in pieces.
		Index    int          `json:"index"`
		ID       string       `json:"id"`
		Function wireFunction `json:"function"`
	}

	replyMessage struct {
		Role             Role            `json:"role"`
		Content          string          `json:"content"`
		ReasoningContent string          `json:"reasoning_content"`
		Reasoning        string          `json:"reasoning"`
		ToolCalls        []replyToolCall `json:"tool_calls"`
	}

	choice struct {
		FinishReason string       `json:"finish_reason"`
		Message      replyMessage `json:"message"`
	}

	completion struct {
		ID      string   `json:"id"`
		Model   string   `json:"model"`
		Choices []choice `json:"choices"`
		Usage   usage    `json:"usage"`
		Stats   *Stats   `json:"timings,omitzero"`
	}

	usage struct {
		PromptTokens        int `json:"prompt_tokens"`
		CompletionTokens    int `json:"completion_tokens"`
		TotalTokens         int `json:"total_tokens"`
		PromptTokensDetails struct {
			CachedTokens int `json:"cached_tokens"`
		} `json:"prompt_tokens_details"`
	}
)

func (u usage) toUsage() Usage {
	return Usage{
		InputTokens:  u.PromptTokens,
		OutputTokens: u.CompletionTokens,
		TotalTokens:  u.TotalTokens,
		CachedTokens: u.PromptTokensDetails.CachedTokens,
	}
}

// toToolCall is the call as the harness sees it. i is its place in the
// reply, used for an ID when the LLM backend sends none.
func (tc replyToolCall) toToolCall(i int) ToolCall {
	return ToolCall{
		ID:        cmp.Or(tc.ID, fmt.Sprintf("call_%d", i)),
		Name:      tc.Function.Name,
		Arguments: cmp.Or(tc.Function.Arguments, "{}"),
	}
}

// toolCalls converts the calls in a reply, and is nil when there are none.
func toolCalls(rc []replyToolCall) []ToolCall {
	if len(rc) == 0 {
		return nil
	}

	out := make([]ToolCall, len(rc))
	for i, c := range rc {
		out[i] = c.toToolCall(i)
	}

	return out
}

// thinking returns the model's reasoning, which LLM backends send under
// either name.
func (m replyMessage) thinking() string {
	return cmp.Or(m.ReasoningContent, m.Reasoning)
}
