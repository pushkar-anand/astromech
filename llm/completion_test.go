package llm_test

import (
	"net/http"
	"testing"
	"testing/synctest"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/pushkar-anand/astromech/llm"
)

func TestComplete(t *testing.T) {
	b, c := newBackend(t, map[string]reply{
		"POST /v1/chat/completions": {body: `{
			"id": "r1",
			"model": "model-a-q4",
			"choices": [{"index": 0, "finish_reason": "stop", "message": {
				"role": "assistant",
				"content": "Hi there.",
				"reasoning_content": "They said hello."
			}}],
			"usage": {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13, "prompt_tokens_details": {"cached_tokens": 2}},
			"timings": {"prompt_n": 9, "prompt_ms": 100, "predicted_n": 4, "predicted_ms": 150, "predicted_per_second": 26.7}
		}`},
	})

	resp, err := c.Complete(t.Context(), hello)
	require.NoError(t, err)

	want := &llm.Response{
		ID:           "r1",
		Model:        "model-a-q4",
		Message:      llm.Message{Role: llm.RoleAssistant, Content: "Hi there."},
		Reasoning:    "They said hello.",
		FinishReason: "stop",
		Usage:        llm.Usage{PromptTokens: 9, CompletionTokens: 4, TotalTokens: 13},
		Stats:        &llm.Stats{PromptN: 9, PromptMs: 100, PredictedN: 4, PredictedMs: 150, PredictedPerSecond: 26.7},
	}
	want.Usage.PromptTokensDetails.CachedTokens = 2

	assert.Equal(t, want, resp)

	assert.Equal(t, map[string]any{
		"model":    "model-a",
		"messages": []any{map[string]any{"role": "user", "content": "Hello"}},
	}, b.lastBody())
}

func TestCompleteWithoutStats(t *testing.T) {
	_, c := newBackend(t, map[string]reply{
		"POST /v1/chat/completions": {body: `{"id":"r1","model":"model-a","choices":[
			{"finish_reason":"length","message":{"role":"assistant","content":"Hi"}}
		]}`},
	})

	resp, err := c.Complete(t.Context(), hello)
	require.NoError(t, err)

	assert.Equal(t, "Hi", resp.Message.Content)
	assert.Equal(t, "length", resp.FinishReason)
	assert.Empty(t, resp.Reasoning)
	assert.Zero(t, resp.Usage)
	assert.Nil(t, resp.Stats)
}

func TestCompleteReasoningField(t *testing.T) {
	_, c := newBackend(t, map[string]reply{
		"POST /v1/chat/completions": {body: `{"id":"r1","model":"model-a","choices":[
			{"finish_reason":"stop","message":{"role":"assistant","content":"Hi","reasoning":"They said hello."}}
		]}`},
	})

	resp, err := c.Complete(t.Context(), hello)
	require.NoError(t, err)

	assert.Equal(t, "Hi", resp.Message.Content)
	assert.Equal(t, "They said hello.", resp.Reasoning)
}

func TestCompleteWaitsForSlowModel(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		_, c := newBackend(t, map[string]reply{
			"POST /v1/chat/completions": {delay: 10 * time.Minute, body: `{"id":"r1","model":"model-a","choices":[
				{"finish_reason":"stop","message":{"role":"assistant","content":"Hi"}}
			]}`},
		})

		start := time.Now()

		resp, err := c.Complete(t.Context(), hello)
		require.NoError(t, err)

		assert.Equal(t, "Hi", resp.Message.Content)
		assert.Equal(t, 10*time.Minute, time.Since(start))
	})
}

func TestCompleteNoChoices(t *testing.T) {
	_, c := newBackend(t, map[string]reply{
		"POST /v1/chat/completions": {body: `{"id":"r1","model":"model-a","choices":[]}`},
	})

	resp, err := c.Complete(t.Context(), hello)
	require.ErrorIs(t, err, llm.ErrNoReply)
	assert.Nil(t, resp)
}

func TestCompleteError(t *testing.T) {
	_, c := newBackend(t, map[string]reply{
		"POST /v1/chat/completions": {status: http.StatusNotFound, body: errorBody},
	})

	resp, err := c.Complete(t.Context(), hello)
	require.Error(t, err)
	assert.Nil(t, resp)
	assert.Contains(t, err.Error(), "llm: chat completion")
}
