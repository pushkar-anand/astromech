package llm

import (
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

	client, err := NewClient(
		baseURL,
		secret.New(apiKey),
	)
	require.NoError(t, err)

	t.Run("test models endpoint", func(t *testing.T) {
		models, err := client.Models(t.Context())
		assert.NoError(t, err)
		assert.NotEmpty(t, models)
	})

	t.Run("test completions endpoint", func(t *testing.T) {
		result, err := client.Complete(t.Context(), CompletionRequest{
			Model: "google/gemma-4-E4B-it-qat-q4_0-gguf",
			Messages: []Message{
				{
					Role:    "system",
					Content: "You are a helpful assistant.",
				},
				{
					Role:    "user",
					Content: "What is the capital of France?",
				},
			},
		})
		assert.NoError(t, err)
		assert.NotEmpty(t, result)
		assert.NotEmpty(t, result.Choices)

		t.Log(result.Choices[0].Message.Content)
	})

}
