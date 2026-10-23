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
		toolRoundTrip(t, func(req Request) *Response {
			resp, err := client.Complete(t.Context(), req)
			require.NoError(t, err)

			return resp
		})
	})

	t.Run("test stream with tool calling", func(t *testing.T) {
		toolRoundTrip(t, func(req Request) *Response {
			for ev := range client.Stream(t.Context(), req) {
				switch e := ev.(type) {
				case Done:
					return e.Response
				case Failed:
					require.NoError(t, e.Err)
				}
			}

			t.Fatal("the stream ended without a Done event")

			return nil
		})
	})
}

// toolRoundTrip offers the model an add tool, answers the calls it makes,
// and checks that it then replies in text. send makes one request.
func toolRoundTrip(t *testing.T, send func(Request) *Response) {
	t.Helper()

	const model = "google/gemma-4-E4B-it-qat-q4_0-gguf"

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

	history := []Message{{Role: RoleUser, Content: "Use the add tool: what is 2 + 2?"}}

	first := send(Request{Model: model, Messages: history, Tools: []Tool{add}})
	t.Logf("finish %q, content %q, calls %+v", first.FinishReason, first.Message.Content, first.Message.ToolCalls)

	// The finish reason is not reliable: some LLM backends say "stop" even
	// when they call a tool.
	if len(first.Message.ToolCalls) == 0 {
		t.Skip("the model answered without calling the tool; is the LLM backend running with --jinja?")
	}

	history = append(history, first.Message)
	for _, call := range first.Message.ToolCalls {
		history = append(history, Message{Role: RoleTool, ToolCallID: call.ID, Content: "4"})
	}

	second := send(Request{Model: model, Messages: history, Tools: []Tool{add}})
	t.Logf("finish %q, answer %q", second.FinishReason, second.Message.Content)
	assert.NotEmpty(t, second.Message.Content)
}
