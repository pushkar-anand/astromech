package llm_test

import (
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/pushkar-anand/astromech/llm"
)

func TestEmbed(t *testing.T) {
	b, c := newBackend(t, map[string]reply{
		// Out of order, as an LLM backend may answer.
		"POST /v1/embeddings": {body: `{"object":"list","model":"embed-a","data":[
			{"object":"embedding","index":1,"embedding":[0.3,0.4]},
			{"object":"embedding","index":0,"embedding":[0.1,0.2]}
		],"usage":{"prompt_tokens":6,"total_tokens":6}}`},
	})

	resp, err := c.Embed(t.Context(), llm.EmbedRequest{Model: "embed-a", Input: []string{"first", "second"}})
	require.NoError(t, err)

	assert.Equal(t, &llm.Embeddings{
		Vectors: [][]float32{{0.1, 0.2}, {0.3, 0.4}},
		Model:   "embed-a",
		Usage:   llm.Usage{InputTokens: 6, TotalTokens: 6},
	}, resp)

	assert.Equal(t, map[string]any{
		"model":           "embed-a",
		"input":           []any{"first", "second"},
		"encoding_format": "float",
	}, b.lastBody())
}

func TestEmbedDimensions(t *testing.T) {
	b, c := newBackend(t, map[string]reply{
		"POST /v1/embeddings": {body: `{"data":[{"index":0,"embedding":[0.1]}]}`},
	})

	_, err := c.Embed(t.Context(), llm.EmbedRequest{Model: "embed-a", Input: []string{"first"}, Dimensions: 256})
	require.NoError(t, err)

	assert.InDelta(t, 256, b.lastBody()["dimensions"], 0)
}

func TestEmbedNoInput(t *testing.T) {
	_, c := newBackend(t, nil)

	_, err := c.Embed(t.Context(), llm.EmbedRequest{Model: "embed-a"})
	assert.EqualError(t, err, "llm: embed: no input")
}

func TestEmbedWrongCount(t *testing.T) {
	_, c := newBackend(t, map[string]reply{
		"POST /v1/embeddings": {body: `{"data":[{"index":0,"embedding":[0.1]}]}`},
	})

	_, err := c.Embed(t.Context(), llm.EmbedRequest{Model: "embed-a", Input: []string{"first", "second"}})
	assert.EqualError(t, err, "llm: embed: the LLM backend sent 1 vectors for 2 inputs")
}

func TestEmbedBadIndex(t *testing.T) {
	for name, data := range map[string]string{
		"twice":        `[{"index":0,"embedding":[0.1]},{"index":0,"embedding":[0.2]}]`,
		"out of range": `[{"index":0,"embedding":[0.1]},{"index":2,"embedding":[0.2]}]`,
	} {
		t.Run(name, func(t *testing.T) {
			_, c := newBackend(t, map[string]reply{
				"POST /v1/embeddings": {body: `{"data":` + data + `}`},
			})

			_, err := c.Embed(t.Context(), llm.EmbedRequest{Model: "embed-a", Input: []string{"first", "second"}})
			assert.ErrorContains(t, err, "twice or out of range")
		})
	}
}

func TestEmbedError(t *testing.T) {
	_, c := newBackend(t, map[string]reply{
		"POST /v1/embeddings": {status: http.StatusNotFound, body: errorBody},
	})

	_, err := c.Embed(t.Context(), llm.EmbedRequest{Model: "embed-a", Input: []string{"first"}})
	assert.EqualError(t, err, "llm: embed: the LLM backend answered 404 Not Found: model not found")
	requireBackendError(t, err, http.StatusNotFound)
}
