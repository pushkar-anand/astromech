package agent_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/pushkar-anand/astromech/agent"
	"github.com/pushkar-anand/astromech/llm"
	"github.com/pushkar-anand/astromech/llm/llmtest"
)

func TestRunRejectsDuplicateTools(t *testing.T) {
	model := llmtest.New(t) // no turns: any request fails the test
	a := &agent.Agent{Model: model, ModelID: "model-a", Tools: []agent.Tool{addTool{}, addTool{}}}

	var events []agent.Event
	for ev := range a.Run(t.Context(), agent.Turn{Input: &llm.Message{Role: llm.RoleUser, Content: "hi"}}) {
		events = append(events, ev)
	}

	// Failed before anything was added, so the caller has nothing to store.
	require.Len(t, events, 1)
	failed, ok := events[0].(agent.Failed)
	require.True(t, ok, "got %T", events[0])
	assert.EqualError(t, failed.Err, `agent: two tools are named "add"`)
	assert.Empty(t, failed.Messages)
	assert.Empty(t, model.Requests())
}
