package llm_test

import (
	"net/http"
	"slices"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/pushkar-anand/astromech/llm"
)

func TestCollect(t *testing.T) {
	_, c := newBackend(t, map[string]reply{
		"POST /v1/chat/completions": {body: sse(chunkText, chunkText2, chunkStop, chunkUsage)},
	})

	var seen []llm.Event

	resp, err := llm.Collect(c.Stream(t.Context(), hello), func(ev llm.Event) {
		seen = append(seen, ev)
	})
	require.NoError(t, err)

	assert.Equal(t, "Hi there.", resp.Message.Content)
	assert.Equal(t, 13, resp.Usage.TotalTokens)
	assert.Equal(t, []llm.Event{
		llm.TextDelta{Text: "Hi "},
		llm.TextDelta{Text: "there."},
		llm.Done{Response: resp},
	}, seen)
}

func TestCollectWithoutCallback(t *testing.T) {
	_, c := newBackend(t, map[string]reply{
		"POST /v1/chat/completions": {body: sse(chunkText, chunkStop)},
	})

	resp, err := llm.Collect(c.Stream(t.Context(), hello), nil)
	require.NoError(t, err)

	assert.Equal(t, "Hi ", resp.Message.Content)
}

func TestCollectFailed(t *testing.T) {
	_, c := newBackend(t, map[string]reply{
		"POST /v1/chat/completions": {status: http.StatusNotFound, body: errorBody},
	})

	resp, err := llm.Collect(c.Stream(t.Context(), hello), nil)

	assert.Nil(t, resp)
	assert.EqualError(t, err, "llm: stream: the LLM backend answered 404 Not Found: model not found")
	requireBackendError(t, err, http.StatusNotFound)
}

func TestCollectWithoutEnd(t *testing.T) {
	stream := slices.Values([]llm.Event{llm.TextDelta{Text: "Hi "}})

	resp, err := llm.Collect(stream, nil)

	assert.Nil(t, resp)
	assert.ErrorIs(t, err, llm.ErrIncompleteStream)
}

func TestCollectStopsAtDone(t *testing.T) {
	done := llm.Done{Response: &llm.Response{Message: llm.Message{Content: "Hi"}}}
	stream := slices.Values([]llm.Event{done, llm.TextDelta{Text: "late"}})

	var seen []llm.Event

	resp, err := llm.Collect(stream, func(ev llm.Event) { seen = append(seen, ev) })
	require.NoError(t, err)

	assert.Same(t, done.Response, resp)
	assert.Equal(t, []llm.Event{done}, seen)
}
