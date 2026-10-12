package llm_test

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/pushkar-anand/astromech/llm"
	"github.com/pushkar-anand/astromech/llm/llmtest"
)

func TestEmbedReturnsAVectorPerInput(t *testing.T) {
	srv := llmtest.NewServer(t)

	reply := llmtest.Embeddings([]float32{0.5, -1}, []float32{0.25, 2})
	reply.Usage = llm.Usage{InputTokens: 6}
	srv.Enqueue(reply)

	got, err := newClient(t, srv).Embed(t.Context(), llm.EmbedRequest{
		Model:      "test-embedder",
		Input:      []string{"Netflix 649 on the credit card", "bedroom lamp"},
		Dimensions: 2,
	})
	require.NoError(t, err)

	assert.Equal(t, [][]float32{{0.5, -1}, {0.25, 2}}, got.Vectors)
	assert.Equal(t, "test-embedder", got.Model)
	assert.Equal(t, 6, got.Usage.InputTokens)

	req := srv.Requests()[0]
	assert.Equal(t, "float", req.Field("encoding_format"))
	assert.InDelta(t, 2, req.Field("dimensions"), 0)
}

func TestEmbedOrdersVectorsByInput(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"object": "list",
			"model":  "test-embedder",
			"data": []map[string]any{
				{"object": "embedding", "index": 1, "embedding": []float32{2}},
				{"object": "embedding", "index": 0, "embedding": []float32{1}},
			},
		})
	}))
	t.Cleanup(srv.Close)

	c, err := llm.New(srv.URL + "/v1")
	require.NoError(t, err)

	got, err := c.Embed(t.Context(), llm.EmbedRequest{Model: "test-embedder", Input: []string{"a", "b"}})
	require.NoError(t, err)
	assert.Equal(t, [][]float32{{1}, {2}}, got.Vectors)
}

func TestEmbedRefusesAMismatchedAnswer(t *testing.T) {
	srv := llmtest.NewServer(t)
	srv.Enqueue(llmtest.Embeddings([]float32{1}))

	_, err := newClient(t, srv).Embed(t.Context(), llm.EmbedRequest{Model: "test-embedder", Input: []string{"a", "b"}})
	assert.ErrorContains(t, err, "1 embeddings for 2 inputs")
}

func TestEmbedReturnsAnErrorSentInPlaceOfVectors(t *testing.T) {
	srv := llmtest.NewServer(t)
	srv.Enqueue(llmtest.Reply{Raw: `{"error":{"message":"upstream provider failed","code":502}}`})

	_, err := newClient(t, srv).Embed(t.Context(), llm.EmbedRequest{Model: "test-embedder", Input: []string{"a"}})

	apiErr, ok := errors.AsType[*llm.APIError](err)
	require.True(t, ok, "got %v", err)
	assert.Equal(t, http.StatusBadGateway, apiErr.StatusCode)
	assert.Equal(t, "upstream provider failed", apiErr.Message)
}

func TestEmbedWithNoInputSendsNothing(t *testing.T) {
	srv := llmtest.NewServer(t)

	got, err := newClient(t, srv).Embed(t.Context(), llm.EmbedRequest{Model: "test-embedder"})
	require.NoError(t, err)
	assert.Empty(t, got.Vectors)
	assert.Empty(t, srv.Requests())
}

func TestModelsReadsWhatTheServerReports(t *testing.T) {
	srv := llmtest.NewServer(t)
	srv.Enqueue(llmtest.Reply{Models: []map[string]any{
		{"id": "local-small"},
		{"id": "served-model", "max_model_len": 32768},
		{
			"id":                   "vendor/vision-model",
			"name":                 "Vision model",
			"context_length":       128000,
			"architecture":         map[string]any{"input_modalities": []string{"text", "image"}},
			"supported_parameters": []string{"tools", "reasoning", "temperature"},
		},
	}})

	models, err := newClient(t, srv).Models(t.Context())
	require.NoError(t, err)

	assert.Equal(t, []llm.Model{
		{ID: "local-small"},
		{ID: "served-model", ContextWindow: 32768},
		{
			ID:            "vendor/vision-model",
			Name:          "Vision model",
			ContextWindow: 128000,
			Capabilities:  &llm.Capabilities{Images: true, Tools: true, Reasoning: true},
		},
	}, models)
}
