package llm

import (
	"bytes"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"math"
	"net/http"
	"os"
	"strings"
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

	t.Run("test embeddings", func(t *testing.T) {
		// The debug dump would print every number of every vector.
		t.Setenv("DEBUG_ASTROMECH", "false")

		client, err := New(baseURL, secret.New(apiKey))
		require.NoError(t, err)

		// EmbeddingGemma expects a task prefix on queries and on the
		// documents searched; see its model card.
		resp, err := client.Embed(t.Context(), EmbedRequest{
			Model: "ggml-org/embeddinggemma-2-GGUF",
			Input: []string{
				"task: search result | query: is it going to rain today?",
				"title: none | text: get_weather: current weather and forecast for a city",
				"title: none | text: read_file: read a file from the project directory",
			},
		})
		require.NoError(t, err)
		require.Len(t, resp.Vectors, 3)

		query, weather, file := resp.Vectors[0], resp.Vectors[1], resp.Vectors[2]
		require.Len(t, weather, len(query))

		t.Logf("model %q, %d dimensions, usage %+v", resp.Model, len(query), resp.Usage)
		t.Logf("similarity to the weather tool %.3f, to the file tool %.3f", cosine(query, weather), cosine(query, file))

		assert.Greater(t, cosine(query, weather), cosine(query, file))
	})

	t.Run("test image input", func(t *testing.T) {
		// The base64 image would fill the debug dump.
		t.Setenv("DEBUG_ASTROMECH", "false")

		client, err := New(baseURL, secret.New(apiKey))
		require.NoError(t, err)

		result, err := client.Complete(t.Context(), Request{
			Model: "google/gemma-4-E4B-it-qat-q4_0-gguf",
			Messages: []Message{{
				Role:    RoleUser,
				Content: "What colour is this image? Answer in one word.",
				Media:   []Media{{MIMEType: "image/png", Data: redSquare(t)}},
			}},
		})

		if backendErr, ok := errors.AsType[*BackendError](err); ok && backendErr.StatusCode < http.StatusInternalServerError {
			t.Skipf("the LLM backend refused the image; is llama.cpp running with --mmproj? %v", err)
		}

		require.NoError(t, err)
		t.Logf("answer %q", result.Message.Content)
		assert.Contains(t, strings.ToLower(result.Message.Content), "red")
	})
}

// cosine is the cosine similarity of two vectors of the same length.
func cosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}

	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

// redSquare is a 64x64 PNG of solid red.
func redSquare(t *testing.T) []byte {
	t.Helper()

	img := image.NewRGBA(image.Rect(0, 0, 64, 64))
	draw.Draw(img, img.Bounds(), image.NewUniform(color.RGBA{R: 255, A: 255}), image.Point{}, draw.Src)

	var buf bytes.Buffer
	require.NoError(t, png.Encode(&buf, img))

	return buf.Bytes()
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
