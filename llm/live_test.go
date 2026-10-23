package llm

import (
	"fmt"
	"os"
	"testing"

	"github.com/pushkar-anand/build-with-go/secret"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLLMLive(t *testing.T) {
	baseURL := os.Getenv("LLM_BASE_URL")
	apiKey := os.Getenv("LLM_API_KEY")

	if baseURL == "" {
		t.Skip("set LLM_BASE_URL, and LLM_API_KEY if the server needs one, to run live tests")
	}

	t.Setenv("DEBUG_ASTROMECH", "true")

	client, err := New(baseURL, secret.New(apiKey))
	require.NoError(t, err)

	t.Run("test models endpoint", func(t *testing.T) {
		models, err := client.Models(t.Context())
		assert.NoError(t, err)
		assert.NotEmpty(t, models)
	})

	t.Run("test completions endpoint", func(t *testing.T) {
		result, err := client.Complete(t.Context(), Request{
			Model: "google/gemma-4-E4B-it-qat-q4_0-gguf",
			Messages: []Message{
				{
					Role:    RoleSystem,
					Content: "You are a helpful assistant.",
				},
				{
					Role:    RoleUser,
					Content: "What is the capital of France?",
				},
			},
		})
		require.NoError(t, err)
		assert.NotEmpty(t, result.Message.Content)

		t.Log(result.Message.Content)
	})

	t.Run("test stream response", func(t *testing.T) {
		// The debug dump would interleave with the streamed text, so this
		// subtest uses a client without it.
		t.Setenv("DEBUG_ASTROMECH", "false")

		client, err := New(baseURL, secret.New(apiKey))
		require.NoError(t, err)

		stream := client.Stream(t.Context(), Request{
			Model: "google/gemma-4-E4B-it-qat-q4_0-gguf",
			Messages: []Message{
				{
					Role:    RoleSystem,
					Content: "You are a helpful assistant.",
				},
				{
					Role:    RoleUser,
					Content: "Give me 10 lines about you",
				},
			},
		})

		var (
			done                *Response
			section             string
			reasoningN, answerN int
		)

		// Print each delta as it arrives, under a heading for each part.
		for ev := range stream {
			switch e := ev.(type) {
			case ReasoningDelta:
				if section != "reasoning" {
					section = "reasoning"
					fmt.Println("\n--- reasoning ---")
				}

				fmt.Print(e.Text)
				reasoningN++
			case TextDelta:
				if section != "answer" {
					section = "answer"
					fmt.Println("\n--- answer ---")
				}

				fmt.Print(e.Text)
				answerN++
			case Done:
				done = e.Response
			case Failed:
				fmt.Println()
				require.NoError(t, e.Err)
			}
		}

		fmt.Println()

		require.NotNil(t, done, "the stream ended without a Done event")
		assert.NotEmpty(t, done.Message.Content)

		t.Logf("%d reasoning deltas, %d answer deltas, finish %q", reasoningN, answerN, done.FinishReason)
		t.Logf("assembled answer is %d bytes, reasoning %d bytes", len(done.Message.Content), len(done.Reasoning))
		t.Logf("usage %+v, stats %+v", done.Usage, done.Stats)
	})

	t.Run("test completions with tool calling", func(t *testing.T) {
		// The tool is offered in Tools; the model decides whether to call it.
		// Complete does not read the call back yet, so this only shows how
		// the backend took the request: "tool_calls" means the model asked.
		add := Tool{
			Name:        "add",
			Description: "Adds two integers.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"a": map[string]any{"type": "integer"},
					"b": map[string]any{"type": "integer"},
				},
				"required": []string{"a", "b"},
			},
		}

		result, err := client.Complete(t.Context(), Request{
			Model: "google/gemma-4-E4B-it-qat-q4_0-gguf",
			Messages: []Message{
				{Role: RoleUser, Content: "Use the add tool: what is 2 + 2?"},
			},
			Tools: []Tool{add},
		})
		require.NoError(t, err)

		t.Logf("finish %q, content %q", result.FinishReason, result.Message.Content)
	})
}
