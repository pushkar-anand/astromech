package agent_test

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/pushkar-anand/astromech/agent"
	"github.com/pushkar-anand/astromech/llm"
	"github.com/pushkar-anand/astromech/llm/llmtest"
)

type deviceArgs struct {
	ID int `json:"id" jsonschema:"the device's ID"`
}

type labelArgs struct {
	ID    int    `json:"id"`
	Label string `json:"label"`
}

// house is a fake inventory with one read-only tool and one that changes
// things.
type house struct {
	mu     sync.Mutex
	labels []labelArgs
}

func (h *house) tools() []agent.Tool {
	return []agent.Tool{
		agent.Func("get_device", "Returns a device.", func(_ context.Context, in deviceArgs) (map[string]any, error) {
			return map[string]any{"id": in.ID, "name": "printer"}, nil
		}, agent.ReadOnly()),
		agent.Func("label_device", "Labels a device.", func(_ context.Context, in labelArgs) (string, error) {
			h.mu.Lock()
			defer h.mu.Unlock()

			h.labels = append(h.labels, in)

			return "Labeled.", nil
		}),
	}
}

func (h *house) labeled() []labelArgs {
	h.mu.Lock()
	defer h.mu.Unlock()

	return h.labels
}

func newAgent(t *testing.T, srv *llmtest.Server, tools ...agent.Tool) *agent.Agent {
	t.Helper()

	c, err := llm.New(srv.URL())
	require.NoError(t, err)

	return &agent.Agent{Client: c, Model: llm.Model{ID: "test-model"}, System: "You keep the inventory.", Tools: tools}
}

func play(t *testing.T, a *agent.Agent, turn agent.Turn) []agent.Event {
	t.Helper()

	var events []agent.Event
	for e := range a.Run(t.Context(), turn) {
		events = append(events, e)
	}

	require.NotEmpty(t, events)

	return events
}

func done(t *testing.T, events []agent.Event) agent.Done {
	t.Helper()

	d, ok := events[len(events)-1].(agent.Done)
	require.True(t, ok, "the turn ended with %#v", events[len(events)-1])

	return d
}

func failed(t *testing.T, events []agent.Event) agent.Failed {
	t.Helper()

	f, ok := events[len(events)-1].(agent.Failed)
	require.True(t, ok, "the turn ended with %#v", events[len(events)-1])

	return f
}

func eventsOf[E agent.Event](events []agent.Event) []E {
	var out []E

	for _, e := range events {
		if e, ok := e.(E); ok {
			out = append(out, e)
		}
	}

	return out
}

func call(id, name, args string) llm.ToolCall {
	return llm.ToolCall{ID: id, Name: name, Arguments: args}
}

// sent returns the messages a request carried, as the server read them.
func sent(t *testing.T, req llmtest.Request) []map[string]any {
	t.Helper()

	var body struct {
		Messages []map[string]any `json:"messages"`
	}
	require.NoError(t, req.Decode(&body))

	return body.Messages
}

func toolNames(req llmtest.Request) []string {
	tools, _ := req.Field("tools").([]any)

	var names []string

	for _, tool := range tools {
		fn, _ := tool.(map[string]any)["function"].(map[string]any)
		name, _ := fn["name"].(string)
		names = append(names, name)
	}

	return names
}

func TestRunAnswersWithoutTools(t *testing.T) {
	srv := llmtest.NewServer(t)
	srv.Enqueue(llmtest.Text("Three devices joined."))

	events := play(t, newAgent(t, srv), agent.Turn{Input: new(llm.User("What joined this week?"))})

	var text strings.Builder
	for _, d := range eventsOf[agent.TextDelta](events) {
		text.WriteString(d.Text)
	}

	assert.Equal(t, "Three devices joined.", text.String())

	d := done(t, events)
	assert.Equal(t, agent.StopAnswered, d.Stop)
	require.Len(t, d.Messages, 2)
	assert.Equal(t, llm.User("What joined this week?"), d.Messages[0])
	assert.Equal(t, "Three devices joined.", d.Messages[1].Content)

	var added []llm.Message
	for _, e := range eventsOf[agent.MessageAdded](events) {
		added = append(added, e.Message)
	}

	assert.Equal(t, d.Messages, added)

	msgs := sent(t, srv.Requests()[0])
	require.Len(t, msgs, 2)
	assert.Equal(t, "system", msgs[0]["role"])
	assert.Equal(t, "You keep the inventory.", msgs[0]["content"])
}

func TestRunCallsAToolAndContinues(t *testing.T) {
	srv := llmtest.NewServer(t)
	srv.Enqueue(
		llmtest.ToolCalls(call("call_a", "get_device", `{"id":42}`)),
		llmtest.Text("Device 42 is the printer."),
	)

	h := &house{}
	events := play(t, newAgent(t, srv, h.tools()...), agent.Turn{Input: new(llm.User("What is device 42?"))})

	finished := eventsOf[agent.ToolFinished](events)
	require.Len(t, finished, 1)
	assert.JSONEq(t, `{"id":42,"name":"printer"}`, finished[0].Result.Content)
	assert.False(t, finished[0].Result.IsError)

	d := done(t, events)
	assert.Equal(t, agent.StopAnswered, d.Stop)
	require.Len(t, d.Messages, 4)
	assert.Equal(t, llm.RoleTool, d.Messages[2].Role)
	assert.Equal(t, "call_a", d.Messages[2].ToolCallID)
	assert.Equal(t, "Device 42 is the printer.", d.Messages[3].Content)

	assert.ElementsMatch(t, []string{"get_device", "label_device"}, toolNames(srv.Requests()[0]))

	second := sent(t, srv.Requests()[1])
	require.Len(t, second, 4)
	assert.Equal(t, "tool", second[3]["role"])
	assert.Equal(t, "call_a", second[3]["tool_call_id"])
}

func TestReadOnlyCallsRunAtOnce(t *testing.T) {
	var started sync.WaitGroup
	started.Add(2)

	ping := agent.Func("ping", "Pings a host.", func(ctx context.Context, in struct {
		Host string `json:"host"`
	}) (string, error) {
		started.Done()

		both := make(chan struct{})
		go func() { started.Wait(); close(both) }()

		select {
		case <-both:
			return in.Host + " is up", nil
		case <-time.After(5 * time.Second):
			return "", errors.New("the other call never started")
		case <-ctx.Done():
			return "", ctx.Err()
		}
	}, agent.ReadOnly())

	srv := llmtest.NewServer(t)
	srv.Enqueue(
		llmtest.ToolCalls(call("call_a", "ping", `{"host":"host-a"}`), call("call_b", "ping", `{"host":"host-b"}`)),
		llmtest.Text("Both are up."),
	)

	events := play(t, newAgent(t, srv, ping), agent.Turn{Input: new(llm.User("Are they up?"))})

	d := done(t, events)
	require.Len(t, d.Messages, 5)
	assert.Equal(t, "host-a is up", d.Messages[2].Content, "results follow the order of the calls")
	assert.Equal(t, "host-b is up", d.Messages[3].Content)
}

func TestManualModeWaitsForAPerson(t *testing.T) {
	label := call("call_l", "label_device", `{"id":42,"label":"printer"}`)

	t.Run("allowed", func(t *testing.T) {
		srv := llmtest.NewServer(t)
		srv.Enqueue(llmtest.ToolCalls(label), llmtest.Text("Labeled it."))

		h := &house{}
		a := newAgent(t, srv, h.tools()...)

		events := play(t, a, agent.Turn{Input: new(llm.User("Label device 42 as the printer."))})

		approvals := eventsOf[agent.ApprovalRequired](events)
		require.Len(t, approvals, 1)
		assert.Equal(t, label, approvals[0].Call)
		assert.Equal(t, "label_device", approvals[0].Tool.Name)

		first := done(t, events)
		assert.Equal(t, agent.StopAwaitingApproval, first.Stop)
		assert.Empty(t, h.labeled(), "nothing changes before a person allows it")
		assert.Equal(t, []llm.ToolCall{label}, agent.Pending(first.Messages))

		events = play(t, a, agent.Turn{
			History:   first.Messages,
			Decisions: map[string]agent.Verdict{"call_l": {Decision: agent.DecisionAllow}},
		})

		second := done(t, events)
		assert.Equal(t, agent.StopAnswered, second.Stop)
		assert.Equal(t, []labelArgs{{ID: 42, Label: "printer"}}, h.labeled())
		require.Len(t, second.Messages, 2)
		assert.Equal(t, "Labeled.", second.Messages[0].Content)
		assert.Empty(t, agent.Pending(append(first.Messages, second.Messages...)))
	})

	t.Run("declined", func(t *testing.T) {
		srv := llmtest.NewServer(t)
		srv.Enqueue(llmtest.ToolCalls(label), llmtest.Text("I left it as it is."))

		h := &house{}
		a := newAgent(t, srv, h.tools()...)

		first := done(t, play(t, a, agent.Turn{Input: new(llm.User("Label device 42 as the printer."))}))

		events := play(t, a, agent.Turn{
			History:   first.Messages,
			Decisions: map[string]agent.Verdict{"call_l": {Decision: agent.DecisionDeny, Reason: "It is the scanner."}},
		})

		assert.Empty(t, h.labeled())

		finished := eventsOf[agent.ToolFinished](events)
		require.Len(t, finished, 1)
		assert.True(t, finished[0].Result.IsError)

		second := done(t, events)
		assert.Equal(t, "The user declined this call, so it did not run. The reason given: It is the scanner.", second.Messages[0].Content)
	})

	t.Run("a new message instead", func(t *testing.T) {
		srv := llmtest.NewServer(t)
		srv.Enqueue(llmtest.ToolCalls(label), llmtest.Text("Here is device 7."))

		h := &house{}
		a := newAgent(t, srv, h.tools()...)

		first := done(t, play(t, a, agent.Turn{Input: new(llm.User("Label device 42 as the printer."))}))

		second := done(t, play(t, a, agent.Turn{History: first.Messages, Input: new(llm.User("Never mind, show device 7."))}))

		assert.Empty(t, h.labeled())
		require.Len(t, second.Messages, 3)
		assert.Equal(t, llm.RoleTool, second.Messages[0].Role, "the waiting call is answered before the new message")
		assert.Contains(t, second.Messages[0].Content, "did not run")
		assert.Equal(t, "Never mind, show device 7.", second.Messages[1].Content)
	})
}

func TestBypassModeRunsEveryCall(t *testing.T) {
	srv := llmtest.NewServer(t)
	srv.Enqueue(llmtest.ToolCalls(call("call_l", "label_device", `{"id":42,"label":"printer"}`)), llmtest.Text("Labeled it."))

	h := &house{}
	events := play(t, newAgent(t, srv, h.tools()...), agent.Turn{Input: new(llm.User("Label device 42.")), Mode: agent.ModeBypass})

	assert.Empty(t, eventsOf[agent.ApprovalRequired](events))
	assert.Equal(t, agent.StopAnswered, done(t, events).Stop)
	assert.Len(t, h.labeled(), 1)
}

func TestReadOnlyModeOffersReadOnlyTools(t *testing.T) {
	srv := llmtest.NewServer(t)
	srv.Enqueue(llmtest.ToolCalls(call("call_l", "label_device", `{"id":42,"label":"printer"}`)), llmtest.Text("I cannot change it."))

	h := &house{}
	events := play(t, newAgent(t, srv, h.tools()...), agent.Turn{Input: new(llm.User("Label device 42.")), Mode: agent.ModeReadOnly})

	assert.Equal(t, []string{"get_device"}, toolNames(srv.Requests()[0]))
	assert.Empty(t, h.labeled())

	finished := eventsOf[agent.ToolFinished](events)
	require.Len(t, finished, 1)
	assert.Equal(t, "label_device changes things, and this conversation is read-only, so the call did not run.", finished[0].Result.Content)
}

func TestAPersonsVerdictDoesNotOutliveTheTool(t *testing.T) {
	srv := llmtest.NewServer(t)
	srv.Enqueue(llmtest.ToolCalls(call("call_l", "label_device", `{"id":42,"label":"printer"}`)), llmtest.Text("I cannot change it."))

	h := &house{}
	a := newAgent(t, srv, h.tools()...)
	first := done(t, play(t, a, agent.Turn{Input: new(llm.User("Label device 42."))}))

	// The conversation became read-only before the person allowed the call.
	play(t, a, agent.Turn{
		History:   first.Messages,
		Decisions: map[string]agent.Verdict{"call_l": {Decision: agent.DecisionAllow}},
		Mode:      agent.ModeReadOnly,
	})

	assert.Empty(t, h.labeled())
}

type approverFunc func(context.Context, agent.ApprovalRequest) (agent.Verdict, error)

func (f approverFunc) Decide(ctx context.Context, req agent.ApprovalRequest) (agent.Verdict, error) {
	return f(ctx, req)
}

func TestAutoModeAsksTheApprover(t *testing.T) {
	label := call("call_l", "label_device", `{"id":42,"label":"printer"}`)

	tests := []struct {
		name    string
		verdict agent.Verdict
		err     error

		wantLabeled bool
		wantStop    agent.Stop
		wantDecided bool
		wantReason  string
	}{
		{name: "allow", verdict: agent.Verdict{Decision: agent.DecisionAllow, Reason: "The user asked for it."}, wantLabeled: true, wantStop: agent.StopAnswered, wantDecided: true},
		{name: "deny", verdict: agent.Verdict{Decision: agent.DecisionDeny, Reason: "The user asked about device 7."}, wantStop: agent.StopAnswered, wantDecided: true},
		{name: "ask", verdict: agent.Verdict{Decision: agent.DecisionAsk, Reason: "The request is unclear."}, wantStop: agent.StopAwaitingApproval, wantReason: "The request is unclear."},
		{name: "unknown decision", verdict: agent.Verdict{Decision: "maybe"}, wantStop: agent.StopAwaitingApproval},
		{name: "error", err: errors.New("the decision model is down"), wantStop: agent.StopAwaitingApproval, wantReason: "The approver failed: the decision model is down"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := llmtest.NewServer(t)
			srv.Enqueue(llmtest.ToolCalls(label))

			if tt.wantStop == agent.StopAnswered {
				srv.Enqueue(llmtest.Text("Done."))
			}

			h := &house{}
			a := newAgent(t, srv, h.tools()...)

			var asked agent.ApprovalRequest
			a.Approver = approverFunc(func(_ context.Context, req agent.ApprovalRequest) (agent.Verdict, error) {
				asked = req

				return tt.verdict, tt.err
			})

			events := play(t, a, agent.Turn{Input: new(llm.User("Label device 42 as the printer.")), Mode: agent.ModeAuto})

			assert.Equal(t, label, asked.Call)
			require.Len(t, asked.Conversation, 2, "the approver sees the conversation up to the call")
			assert.Equal(t, []llm.ToolCall{label}, asked.Conversation[1].ToolCalls)

			assert.Equal(t, tt.wantStop, done(t, events).Stop)
			assert.Equal(t, tt.wantLabeled, len(h.labeled()) == 1)

			decided := eventsOf[agent.Decided](events)
			if tt.wantDecided {
				require.Len(t, decided, 1)
				assert.Equal(t, tt.verdict, decided[0].Verdict)
			} else {
				assert.Empty(t, decided)

				approvals := eventsOf[agent.ApprovalRequired](events)
				require.Len(t, approvals, 1)
				assert.Equal(t, tt.wantReason, approvals[0].Reason)
			}
		})
	}
}

func TestAutoModeWithoutAnApproverAsksAPerson(t *testing.T) {
	srv := llmtest.NewServer(t)
	srv.Enqueue(llmtest.ToolCalls(call("call_l", "label_device", `{"id":42,"label":"printer"}`)))

	h := &house{}
	events := play(t, newAgent(t, srv, h.tools()...), agent.Turn{Input: new(llm.User("Label device 42.")), Mode: agent.ModeAuto})

	assert.Equal(t, agent.StopAwaitingApproval, done(t, events).Stop)
	assert.Empty(t, h.labeled())
}

func TestBadCallsAreAnsweredSoTheModelCanRetry(t *testing.T) {
	boom := agent.Func("boom", "Fails.", func(context.Context, struct{}) (string, error) {
		panic("wires crossed")
	}, agent.ReadOnly())

	h := &house{}
	tools := append(h.tools(), boom)

	tests := []struct {
		name string
		call llm.ToolCall
		want string
	}{
		{name: "unknown tool", call: call("call_x", "reboot", `{}`), want: `There is no tool named "reboot". The tools are boom, get_device, label_device.`},
		{name: "missing argument", call: call("call_x", "get_device", `{}`), want: "get_device failed: the arguments do not match the parameters"},
		{name: "wrong type", call: call("call_x", "get_device", `{"id":"forty-two"}`), want: "get_device failed: the arguments do not match the parameters"},
		{name: "not JSON", call: call("call_x", "get_device", `{"id":`), want: "get_device failed: the arguments are not JSON"},
		{name: "panic", call: call("call_x", "boom", ``), want: "boom failed: wires crossed"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := llmtest.NewServer(t)
			srv.Enqueue(llmtest.ToolCalls(tt.call), llmtest.Text("Let me try again."))

			events := play(t, newAgent(t, srv, tools...), agent.Turn{Input: new(llm.User("Go."))})

			finished := eventsOf[agent.ToolFinished](events)
			require.Len(t, finished, 1)
			assert.True(t, finished[0].Result.IsError)
			assert.Contains(t, finished[0].Result.Content, tt.want)
			assert.Equal(t, agent.StopAnswered, done(t, events).Stop, "the model reads the error and carries on")
		})
	}
}

func TestLongResultsAreCut(t *testing.T) {
	long := agent.Func("dump", "Dumps.", func(context.Context, struct{}) (string, error) {
		return strings.Repeat("é", 100), nil
	}, agent.ReadOnly())

	srv := llmtest.NewServer(t)
	srv.Enqueue(llmtest.ToolCalls(call("call_d", "dump", `{}`)), llmtest.Text("That is long."))

	a := newAgent(t, srv, long)
	a.MaxResultBytes = 11

	finished := eventsOf[agent.ToolFinished](play(t, a, agent.Turn{Input: new(llm.User("Dump."))}))
	require.Len(t, finished, 1)
	assert.Equal(t, strings.Repeat("é", 5)+"\n\n[The result was cut to its first 10 of 200 bytes.]", finished[0].Result.Content)
}

func TestMaxRoundsEndsATurnThatKeepsCallingTools(t *testing.T) {
	srv := llmtest.NewServer(t)
	srv.Enqueue(
		llmtest.ToolCalls(call("call_1", "get_device", `{"id":1}`)),
		llmtest.ToolCalls(call("call_2", "get_device", `{"id":2}`)),
	)

	h := &house{}
	a := newAgent(t, srv, h.tools()...)
	a.MaxRounds = 2

	d := done(t, play(t, a, agent.Turn{Input: new(llm.User("Look at everything."))}))
	assert.Equal(t, agent.StopMaxRounds, d.Stop)
	assert.Len(t, srv.Requests(), 2)
	assert.Equal(t, llm.RoleTool, d.Messages[len(d.Messages)-1].Role)
}

func TestTheModelsCapabilitiesShapeTheRequest(t *testing.T) {
	image := llm.UserWithImages("What is this?", llm.Image{URL: "https://example.com/a.png"})

	t.Run("unknown", func(t *testing.T) {
		srv := llmtest.NewServer(t)
		srv.Enqueue(llmtest.Text("A printer."))

		h := &house{}
		play(t, newAgent(t, srv, h.tools()...), agent.Turn{Input: &image, Effort: llm.EffortLow})

		req := srv.Requests()[0]
		assert.Equal(t, "low", req.Field("reasoning_effort"))
		assert.Len(t, toolNames(req), 2)
		assert.IsType(t, []any{}, sent(t, req)[1]["content"], "the image is sent")
	})

	t.Run("none", func(t *testing.T) {
		srv := llmtest.NewServer(t)
		srv.Enqueue(llmtest.Text("I cannot see it."))

		h := &house{}
		play(t, newAgent(t, srv, h.tools()...), agent.Turn{
			Input:  &image,
			Effort: llm.EffortLow,
			Model:  llm.Model{ID: "small-model", Capabilities: &llm.Capabilities{}},
		})

		req := srv.Requests()[0]
		assert.Equal(t, "small-model", req.Field("model"))
		assert.Nil(t, req.Field("reasoning_effort"))
		assert.Empty(t, toolNames(req))
		assert.Equal(t, "What is this?\n\n[1 image(s) left out: this model cannot see images.]", sent(t, req)[1]["content"])
	})
}

func TestAFailedReplyKeepsWhatArrived(t *testing.T) {
	srv := llmtest.NewServer(t)
	srv.Enqueue(llmtest.Reply{Raw: llmtest.SSE(map[string]any{
		"id":      "chatcmpl-test",
		"model":   "test-model",
		"choices": []map[string]any{{"index": 0, "delta": map[string]any{"content": "Device 42 is"}}},
	})})

	f := failed(t, play(t, newAgent(t, srv), agent.Turn{Input: new(llm.User("What is device 42?"))}))

	require.ErrorIs(t, f.Err, llm.ErrIncompleteStream)
	require.Len(t, f.Messages, 2)
	assert.Equal(t, "Device 42 is", f.Messages[1].Content)
}

func TestStoppingATurnEndsIt(t *testing.T) {
	srv := llmtest.NewServer(t)
	srv.Enqueue(llmtest.Reply{Text: "a long answer nobody reads", ChunkSize: 1})

	n := 0
	for e := range newAgent(t, srv).Run(t.Context(), agent.Turn{Input: new(llm.User("Hi."))}) {
		n++

		if _, ok := e.(agent.TextDelta); ok {
			break
		}
	}

	assert.Equal(t, 2, n, "the input, then one piece of text")
}

func TestATurnThatCannotStartFails(t *testing.T) {
	srv := llmtest.NewServer(t)
	h := &house{}
	a := newAgent(t, srv, h.tools()...)

	waiting := []llm.Message{
		llm.User("Label device 42."),
		{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{call("call_l", "label_device", `{"id":42,"label":"printer"}`)}},
	}

	tests := []struct {
		name string
		turn agent.Turn
		want string
	}{
		{name: "no model", turn: agent.Turn{Model: llm.Model{}}, want: "names no model"},
		{name: "unknown mode", turn: agent.Turn{Mode: "yolo"}, want: `mode "yolo"`},
		{name: "verdict for no call", turn: agent.Turn{History: waiting, Decisions: map[string]agent.Verdict{"call_x": {Decision: agent.DecisionAllow}}}, want: "call_x"},
		{name: "unknown decision", turn: agent.Turn{History: waiting, Decisions: map[string]agent.Verdict{"call_l": {Decision: "yes"}}}, want: `"yes"`},
		{name: "call followed by a message", turn: agent.Turn{History: append(waiting, llm.User("Hello?"))}, want: "has no result"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b := *a
			if tt.name == "no model" {
				b.Model = llm.Model{}
			}

			f := failed(t, play(t, &b, tt.turn))
			assert.ErrorContains(t, f.Err, tt.want)
			assert.Empty(t, f.Messages)
		})
	}

	assert.Empty(t, srv.Requests(), "nothing is sent")
	assert.Empty(t, h.labeled())
}

func TestTwoToolsWithOneNameAreRefused(t *testing.T) {
	h := &house{}
	a := newAgent(t, llmtest.NewServer(t), append(h.tools(), h.tools()[0])...)

	f := failed(t, play(t, a, agent.Turn{Input: new(llm.User("Hi."))}))
	assert.ErrorContains(t, f.Err, "two tools are named get_device")
}
