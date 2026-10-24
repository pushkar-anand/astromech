package agent_test

import (
	"context"
	"encoding/json/jsontext"
	"encoding/json/v2"
	"fmt"
	"iter"
	"os"
	"strconv"
	"testing"

	"github.com/pushkar-anand/build-with-go/secret"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/pushkar-anand/astromech/agent"
	"github.com/pushkar-anand/astromech/llm"
)

// liveModel is the chat model the live tests ask for.
const liveModel = "google/gemma-4-E4B-it-qat-q4_0-gguf"

func TestAgentLive(t *testing.T) {
	baseURL := os.Getenv("LLM_BASE_URL")
	if baseURL == "" {
		t.Skip("set LLM_BASE_URL, and LLM_API_KEY if the server needs one, to run live tests")
	}

	// The agent's own trace is printed instead of the client's debug dump.
	t.Setenv("DEBUG_ASTROMECH", "false")

	client, err := llm.New(baseURL, secret.New(os.Getenv("LLM_API_KEY")))
	require.NoError(t, err)

	model := &printingModel{model: client}
	a := &agent.Agent{
		Model:   model,
		ModelID: liveModel,
		System:  "You are a helpful assistant. Use the tools you are given.",
		Tools:   []agent.Tool{addTool{}},
	}

	history := []llm.Message{
		{Role: llm.RoleUser, Content: "My name is Ada."},
		{Role: llm.RoleAssistant, Content: "Nice to meet you, Ada."},
	}
	input := llm.Message{Role: llm.RoleUser, Content: "Use the add tool: what is 2 + 2? Then greet me by name."}

	var done *agent.Done

	for ev := range a.Run(t.Context(), agent.Turn{History: history, Input: &input}) {
		switch e := ev.(type) {
		case agent.TextDelta:
			fmt.Print(e.Text)
		case agent.ReasoningDelta:
			// Left out: it would drown the trace.
		case agent.MessageAdded:
			fmt.Printf("\n[message added] %s\n", describe(e.Message))
		case agent.ToolStarted:
			fmt.Printf("\n[tool started] %s(%s)\n", e.Call.Name, e.Call.Arguments)
		case agent.ToolFinished:
			fmt.Printf("[tool finished] %s: %+v\n", e.Call.Name, e.Result)
		case agent.Done:
			fmt.Printf("\n[done] stop %q, usage %+v\n", e.Stop, e.Usage)
			printMessages("done messages", e.Messages)
			done = &e
		case agent.Failed:
			fmt.Printf("\n[failed] %v\n", e.Err)
			printMessages("failed messages", e.Messages)
			require.NoError(t, e.Err)
		}
	}

	require.NotNil(t, done, "the turn ended without a Done event")
	require.NotEmpty(t, model.requests)

	system := llm.Message{Role: llm.RoleSystem, Content: a.System}

	// Every request starts with the system prompt, the history and the input.
	start := append([]llm.Message{system}, history...)
	start = append(start, input)

	for i, req := range model.requests {
		if assert.GreaterOrEqual(t, len(req.Messages), len(start), "request %d", i) {
			assert.Equal(t, start, req.Messages[:len(start)], "request %d does not start with system, history, input", i)
		}

		assert.Equal(t, model.requests[0].Tools, req.Tools, "request %d has different tools", i)
	}

	// Each request only appends to the one before, as the prompt cache needs.
	for i := 1; i < len(model.requests); i++ {
		prev, next := model.requests[i-1].Messages, model.requests[i].Messages
		if assert.Greater(t, len(next), len(prev), "request %d is not longer than request %d", i, i-1) {
			assert.Equal(t, prev, next[:len(prev)], "request %d rewrites request %d", i, i-1)
		}
	}

	// Done holds exactly what the turn added: the input, the replies and the
	// tool results, with no system prompt or history.
	assert.Equal(t, agent.StopAnswered, done.Stop)

	if assert.NotEmpty(t, done.Messages) {
		assert.Equal(t, input, done.Messages[0], "done messages do not start with the input")

		last := model.requests[len(model.requests)-1].Messages
		assert.Equal(t, last[len(start)-1:], done.Messages[:len(done.Messages)-1],
			"done messages are not the last request's input onwards, plus the final reply")
	}
}

// printingModel prints each request before passing it on, and keeps it.
type printingModel struct {
	model    agent.Model
	requests []llm.Request
}

func (m *printingModel) Stream(ctx context.Context, req llm.Request) iter.Seq[llm.Event] {
	m.requests = append(m.requests, req)
	printMessages(fmt.Sprintf("request %d, %d tools", len(m.requests)-1, len(req.Tools)), req.Messages)

	return m.model.Stream(ctx, req)
}

func printMessages(title string, messages []llm.Message) {
	fmt.Printf("--- %s ---\n", title)

	for i, msg := range messages {
		fmt.Printf("  %2d %s\n", i, describe(msg))
	}
}

// describe is one line about msg: its role, the start of its text, and any
// tool calls or tool call ID.
func describe(msg llm.Message) string {
	s := fmt.Sprintf("%-9s %q", msg.Role, short(msg.Content))

	for _, call := range msg.ToolCalls {
		s += fmt.Sprintf(" call %s %s(%s)", call.ID, call.Name, call.Arguments)
	}

	if msg.ToolCallID != "" {
		s += " for " + msg.ToolCallID
	}

	return s
}

func short(s string) string {
	const n = 60
	if r := []rune(s); len(r) > n {
		return string(r[:n]) + "..."
	}

	return s
}

// addTool adds two integers.
type addTool struct{}

func (addTool) Def() llm.Tool {
	return llm.Tool{
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
}

func (addTool) ReadOnly() bool { return true }

func (addTool) Call(_ context.Context, args jsontext.Value) (agent.Result, error) {
	var in struct {
		A int `json:"a"`
		B int `json:"b"`
	}
	if err := json.Unmarshal(args, &in); err != nil {
		return agent.Result{}, err
	}

	return agent.Result{Content: strconv.Itoa(in.A + in.B)}, nil
}
