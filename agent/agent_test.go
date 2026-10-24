package agent_test

import (
	"context"
	"encoding/json/jsontext"
	"errors"
	"slices"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/pushkar-anand/astromech/agent"
	"github.com/pushkar-anand/astromech/llm"
	"github.com/pushkar-anand/astromech/llm/llmtest"
)

var (
	system  = llm.Message{Role: llm.RoleSystem, Content: "Be brief."}
	history = []llm.Message{
		{Role: llm.RoleUser, Content: "My name is Ada."},
		{Role: llm.RoleAssistant, Content: "Nice to meet you, Ada."},
	}
	input = llm.Message{Role: llm.RoleUser, Content: "What is 2 + 2?"}
)

// run collects every event of one turn.
func run(t *testing.T, a *agent.Agent, turn agent.Turn) []agent.Event {
	t.Helper()

	return slices.Collect(a.Run(t.Context(), turn))
}

// last is the turn's final event, which must be of type T.
func last[T agent.Event](t *testing.T, events []agent.Event) T {
	t.Helper()

	require.NotEmpty(t, events)
	ev, ok := events[len(events)-1].(T)
	require.True(t, ok, "last event is %T", events[len(events)-1])

	return ev
}

// added is the messages in the turn's MessageAdded events, in order.
func added(events []agent.Event) []llm.Message {
	var msgs []llm.Message
	for _, ev := range events {
		if e, ok := ev.(agent.MessageAdded); ok {
			msgs = append(msgs, e.Message)
		}
	}

	return msgs
}

// withUsage is turn with usage reported in its Done.
func withUsage(turn llmtest.Turn, usage llm.Usage) llmtest.Turn {
	for _, ev := range turn.Events {
		if done, ok := ev.(llm.Done); ok {
			done.Response.Usage = usage
		}
	}

	return turn
}

func addCall(id string) llm.ToolCall {
	return llm.ToolCall{ID: id, Name: "add", Arguments: `{"a":2,"b":2}`}
}

func reply(text string) llm.Message {
	return llm.Message{Role: llm.RoleAssistant, Content: text}
}

func callsReply(calls ...llm.ToolCall) llm.Message {
	return llm.Message{Role: llm.RoleAssistant, ToolCalls: calls}
}

func toolMessage(id, content string) llm.Message {
	return llm.Message{Role: llm.RoleTool, Content: content, ToolCallID: id}
}

func TestRunAnswers(t *testing.T) {
	model := llmtest.New(t, llmtest.Text("4"))
	a := &agent.Agent{Model: model, ModelID: "model-a", System: system.Content}

	events := run(t, a, agent.Turn{History: history, Input: &input})

	assert.Equal(t, []agent.Event{
		agent.MessageAdded{Message: input},
		agent.TextDelta{Text: "4"},
		agent.MessageAdded{Message: reply("4")},
		agent.Done{Stop: agent.StopAnswered, Messages: []llm.Message{input, reply("4")}},
	}, events)

	// The history is sent but not reported: the caller already has it.
	assert.Equal(t, []llm.Request{{
		Model:    "model-a",
		Messages: slices.Concat([]llm.Message{system}, history, []llm.Message{input}),
	}}, model.Requests())
}

func TestRunCallsTools(t *testing.T) {
	model := llmtest.New(t,
		llmtest.ToolCalls(addCall("call_1"), addCall("call_2")),
		llmtest.Text("4"),
	)
	a := &agent.Agent{Model: model, ModelID: "model-a", Tools: []agent.Tool{addTool{}}}

	events := run(t, a, agent.Turn{Input: &input})

	calls := callsReply(addCall("call_1"), addCall("call_2"))
	assert.Equal(t, []agent.Event{
		agent.MessageAdded{Message: input},
		agent.MessageAdded{Message: calls},
		agent.ToolStarted{Call: addCall("call_1")},
		agent.ToolFinished{Call: addCall("call_1"), Result: agent.Result{Content: "4"}},
		agent.MessageAdded{Message: toolMessage("call_1", "4")},
		agent.ToolStarted{Call: addCall("call_2")},
		agent.ToolFinished{Call: addCall("call_2"), Result: agent.Result{Content: "4"}},
		agent.MessageAdded{Message: toolMessage("call_2", "4")},
		agent.TextDelta{Text: "4"},
		agent.MessageAdded{Message: reply("4")},
		agent.Done{Stop: agent.StopAnswered, Messages: []llm.Message{
			input, calls, toolMessage("call_1", "4"), toolMessage("call_2", "4"), reply("4"),
		}},
	}, events)

	// The second round sends the first one's call and results, and the tools
	// both times.
	requests := model.Requests()
	require.Len(t, requests, 2)
	assert.Equal(t, []llm.Message{input}, requests[0].Messages)
	assert.Equal(t, []llm.Message{
		input, calls, toolMessage("call_1", "4"), toolMessage("call_2", "4"),
	}, requests[1].Messages)
	for _, req := range requests {
		assert.Equal(t, []llm.Tool{addTool{}.Def()}, req.Tools)
	}
}

func TestRunContinuesWithoutInput(t *testing.T) {
	// A turn that ended at its round limit, after a tool result.
	stopped := []llm.Message{input, callsReply(addCall("call_1")), toolMessage("call_1", "4")}

	model := llmtest.New(t, llmtest.Text("4"))
	a := &agent.Agent{Model: model, ModelID: "model-a"}

	events := run(t, a, agent.Turn{History: stopped})

	done := last[agent.Done](t, events)
	assert.Equal(t, []llm.Message{reply("4")}, done.Messages, "the history is not reported again")
	assert.Equal(t, stopped, model.Requests()[0].Messages)
}

func TestRunSumsUsage(t *testing.T) {
	model := llmtest.New(t,
		withUsage(llmtest.ToolCalls(addCall("call_1")), llm.Usage{InputTokens: 10, OutputTokens: 1, TotalTokens: 11}),
		withUsage(llmtest.Text("4"), llm.Usage{InputTokens: 20, OutputTokens: 2, TotalTokens: 22, CachedTokens: 10}),
	)
	a := &agent.Agent{Model: model, ModelID: "model-a", Tools: []agent.Tool{addTool{}}}

	done := last[agent.Done](t, run(t, a, agent.Turn{Input: &input}))

	assert.Equal(t, llm.Usage{InputTokens: 30, OutputTokens: 3, TotalTokens: 33, CachedTokens: 10}, done.Usage)
}

func TestRunStopsAtMaxRounds(t *testing.T) {
	for name, tc := range map[string]struct {
		maxRounds int
		want      int
	}{
		"set":     {maxRounds: 2, want: 2},
		"default": {maxRounds: 0, want: 25},
	} {
		t.Run(name, func(t *testing.T) {
			turns := make([]llmtest.Turn, tc.want)
			for i := range turns {
				turns[i] = llmtest.ToolCalls(addCall("call_1"))
			}

			model := llmtest.New(t, turns...)
			a := &agent.Agent{Model: model, ModelID: "model-a", Tools: []agent.Tool{addTool{}}, MaxRounds: tc.maxRounds}

			events := run(t, a, agent.Turn{Input: &input})

			done := last[agent.Done](t, events)
			assert.Equal(t, agent.StopMaxRounds, done.Stop)
			assert.Equal(t, added(events), done.Messages)
			assert.Len(t, done.Messages, 1+2*tc.want, "the input, then a call and a result each round")
			assert.Len(t, model.Requests(), tc.want)
		})
	}
}

func TestRunNegativeMaxRoundsHasNoLimit(t *testing.T) {
	// More rounds than the default limit, then an answer.
	turns := make([]llmtest.Turn, 30)
	for i := range turns {
		turns[i] = llmtest.ToolCalls(addCall("call_1"))
	}
	turns = append(turns, llmtest.Text("4"))

	model := llmtest.New(t, turns...)
	a := &agent.Agent{Model: model, ModelID: "model-a", Tools: []agent.Tool{addTool{}}, MaxRounds: -1}

	done := last[agent.Done](t, run(t, a, agent.Turn{Input: &input}))

	assert.Equal(t, agent.StopAnswered, done.Stop)
	assert.Len(t, model.Requests(), 31)
}

func TestRunFails(t *testing.T) {
	boom := errors.New("boom")
	usage := llm.Usage{InputTokens: 10, OutputTokens: 1, TotalTokens: 11}

	for name, tc := range map[string]struct {
		second  llmtest.Turn
		wantErr error
	}{
		"model fails":       {second: llmtest.Fail(boom), wantErr: boom},
		"stream ends early": {second: llmtest.Events(llm.TextDelta{Text: "4"}), wantErr: llm.ErrIncompleteStream},
	} {
		t.Run(name, func(t *testing.T) {
			// The first round succeeds, so there is something to report.
			model := llmtest.New(t, withUsage(llmtest.ToolCalls(addCall("call_1")), usage), tc.second)
			a := &agent.Agent{Model: model, ModelID: "model-a", Tools: []agent.Tool{addTool{}}}

			events := run(t, a, agent.Turn{Input: &input})

			failed := last[agent.Failed](t, events)
			require.ErrorIs(t, failed.Err, tc.wantErr)
			assert.Equal(t, []llm.Message{input, callsReply(addCall("call_1")), toolMessage("call_1", "4")}, failed.Messages)
			assert.Equal(t, added(events), failed.Messages)
			assert.Equal(t, usage, failed.Usage)
		})
	}
}

func TestRunFailsWhenContextIsDone(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	cancel()

	model := llmtest.New(t)
	a := &agent.Agent{Model: model, ModelID: "model-a"}

	events := slices.Collect(a.Run(ctx, agent.Turn{Input: &input}))

	failed := last[agent.Failed](t, events)
	require.ErrorIs(t, failed.Err, context.Canceled)
	assert.Equal(t, []llm.Message{input}, failed.Messages)
}

func TestRunStopsWhenCallerStops(t *testing.T) {
	// Only the first turn is used: the caller stops at the call.
	model := llmtest.New(t, llmtest.ToolCalls(addCall("call_1")))
	tool := &countingTool{}
	a := &agent.Agent{Model: model, ModelID: "model-a", Tools: []agent.Tool{tool}}

	var events []agent.Event
	for ev := range a.Run(t.Context(), agent.Turn{Input: &input}) {
		events = append(events, ev)
		if _, ok := ev.(agent.ToolStarted); ok {
			break
		}
	}

	assert.IsType(t, agent.ToolStarted{}, events[len(events)-1])
	assert.Zero(t, tool.calls, "the tool does not run after the caller stops")
	assert.Len(t, model.Requests(), 1)
}

func TestRunToolErrorResults(t *testing.T) {
	for name, tc := range map[string]struct {
		call llm.ToolCall
		want string
	}{
		"unknown tool": {
			call: llm.ToolCall{ID: "call_1", Name: "sub", Arguments: `{}`},
			want: `there is no tool named "sub"; the tools are: add, fail`,
		},
		"invalid JSON": {
			call: llm.ToolCall{ID: "call_1", Name: "add", Arguments: `{"a":2,`},
			want: `the arguments for add are not valid JSON: {"a":2,`,
		},
		"tool fails": {
			call: llm.ToolCall{ID: "call_1", Name: "fail", Arguments: `{}`},
			want: "fail failed: boom",
		},
	} {
		t.Run(name, func(t *testing.T) {
			model := llmtest.New(t, llmtest.ToolCalls(tc.call), llmtest.Text("sorry"))
			a := &agent.Agent{Model: model, ModelID: "model-a", Tools: []agent.Tool{addTool{}, failTool{}}}

			events := run(t, a, agent.Turn{Input: &input})

			assert.Contains(t, events, agent.ToolFinished{Call: tc.call, Result: agent.Result{Content: tc.want, IsError: true}})

			// The model reads the error and answers; the turn goes on.
			assert.Equal(t, agent.StopAnswered, last[agent.Done](t, events).Stop)
			assert.Equal(t, toolMessage("call_1", tc.want), model.Requests()[1].Messages[2])
		})
	}
}

func TestRunNoToolsNamesNone(t *testing.T) {
	call := llm.ToolCall{ID: "call_1", Name: "add", Arguments: `{}`}
	model := llmtest.New(t, llmtest.ToolCalls(call), llmtest.Text("sorry"))
	a := &agent.Agent{Model: model, ModelID: "model-a"}

	events := run(t, a, agent.Turn{Input: &input})

	assert.Contains(t, events, agent.ToolFinished{Call: call, Result: agent.Result{
		Content: `there is no tool named "add"; the tools are: none`,
		IsError: true,
	}})
}

func TestRunRejectsDuplicateTools(t *testing.T) {
	model := llmtest.New(t) // no turns: any request fails the test
	a := &agent.Agent{Model: model, ModelID: "model-a", Tools: []agent.Tool{addTool{}, addTool{}}}

	events := run(t, a, agent.Turn{Input: &llm.Message{Role: llm.RoleUser, Content: "hi"}})

	// Failed before anything was added, so the caller has nothing to store.
	require.Len(t, events, 1)
	failed := last[agent.Failed](t, events)
	assert.EqualError(t, failed.Err, `agent: two tools are named "add"`)
	assert.Empty(t, failed.Messages)
	assert.Empty(t, model.Requests())
}

// failTool always fails.
type failTool struct{}

func (failTool) Def() llm.Tool  { return llm.Tool{Name: "fail", Description: "Always fails."} }
func (failTool) ReadOnly() bool { return true }

func (failTool) Call(context.Context, jsontext.Value) (agent.Result, error) {
	return agent.Result{}, errors.New("boom")
}

// countingTool counts its calls.
type countingTool struct{ calls int }

func (*countingTool) Def() llm.Tool  { return llm.Tool{Name: "add", Description: "Counts its calls."} }
func (*countingTool) ReadOnly() bool { return true }

func (c *countingTool) Call(context.Context, jsontext.Value) (agent.Result, error) {
	c.calls++
	return agent.Result{Content: "4"}, nil
}
