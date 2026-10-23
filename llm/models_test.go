package llm_test

import (
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/pushkar-anand/astromech/llm"
)

func TestModels(t *testing.T) {
	_, c := newBackend(t, map[string]reply{
		"GET /v1/models": {body: `{"object":"list","data":[
			{"id":"model-a","name":"Model A","description":"A test model","created":1700000000,"context_length":32768},
			{"id":"model-b"}
		]}`},
	})

	models, err := c.Models(t.Context())
	require.NoError(t, err)

	assert.Equal(t, []llm.Model{
		{ID: "model-a", Name: "Model A", Description: "A test model", Created: 1700000000, ContextLength: 32768},
		{ID: "model-b"},
	}, models)
}

func TestModelsError(t *testing.T) {
	_, c := newBackend(t, map[string]reply{
		"GET /v1/models": {status: http.StatusInternalServerError, body: errorBody},
	})

	models, err := c.Models(t.Context())
	assert.Nil(t, models)
	assert.EqualError(t, err, "llm: list models: the LLM backend answered 500 Internal Server Error: model not found")
	requireBackendError(t, err, http.StatusInternalServerError)
}
