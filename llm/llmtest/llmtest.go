// Package llmtest provides a scripted fake model, for testing code that
// drives one, such as an agent loop, without an LLM backend.
package llmtest

import (
	"context"
	"errors"
	"iter"
	"slices"
	"sync"
	"testing"

	"github.com/pushkar-anand/astromech/llm"
)

// ErrNoTurn is the error of a stream asked for after every scripted turn was
// used.
var ErrNoTurn = errors.New("llmtest: no scripted turn left")

// Turn is one scripted reply: the events its stream yields.
type Turn struct{ Events []llm.Event }

// Text is a reply in text, ending in Done with finish reason "stop".
func Text(text string) Turn {
	var events []llm.Event
	if text != "" {
		events = append(events, llm.TextDelta{Text: text})
	}

	return Turn{Events: append(events, llm.Done{Response: &llm.Response{
		Message:      llm.Message{Role: llm.RoleAssistant, Content: text},
		FinishReason: "stop",
	}})}
}

// ToolCalls is a reply that calls tools, ending in Done with finish reason
// "tool_calls".
func ToolCalls(calls ...llm.ToolCall) Turn {
	return Turn{Events: []llm.Event{llm.Done{Response: &llm.Response{
		Message:      llm.Message{Role: llm.RoleAssistant, ToolCalls: calls},
		FinishReason: "tool_calls",
	}}}}
}

// Fail is a reply that fails with err.
func Fail(err error) Turn {
	return Turn{Events: []llm.Event{llm.Failed{Err: err}}}
}

// Events is a reply of exactly these events, such as a stream that ends
// without Done or Failed.
func Events(events ...llm.Event) Turn {
	return Turn{Events: events}
}

// Model is a fake model that answers each request with the next scripted
// Turn and records what it was sent. It has the same Stream method as
// [llm.Client], and is safe for concurrent use. Make one with [New].
type Model struct {
	t testing.TB

	mu       sync.Mutex
	turns    []Turn
	requests []llm.Request
}

// New scripts a model. The test fails if the model is asked for more replies
// than there are turns, or, when the test ends, if a turn was not used.
func New(t testing.TB, turns ...Turn) *Model {
	t.Helper()

	m := &Model{t: t, turns: turns}

	t.Cleanup(func() {
		m.mu.Lock()
		defer m.mu.Unlock()

		if unused := len(m.turns) - len(m.requests); unused > 0 {
			t.Errorf("llmtest: %d of %d scripted turns were not used", unused, len(m.turns))
		}
	})

	return m
}

// Stream yields the next turn's events. Like [llm.Client.Stream], it sends
// nothing until ranged over, and fails with the context's error once ctx is
// done.
func (m *Model) Stream(ctx context.Context, req llm.Request) iter.Seq[llm.Event] {
	return func(yield func(llm.Event) bool) {
		if err := ctx.Err(); err != nil {
			yield(llm.Failed{Err: err})
			return
		}

		turn, ok := m.take(req)
		if !ok {
			yield(llm.Failed{Err: ErrNoTurn})
			return
		}

		for _, ev := range turn.Events {
			if err := ctx.Err(); err != nil {
				yield(llm.Failed{Err: err})
				return
			}

			if !yield(ev) {
				return
			}
		}
	}
}

// take records req and returns the turn that answers it.
func (m *Model) take(req llm.Request) (Turn, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()

	n := len(m.requests)
	m.requests = append(m.requests, snapshot(req))

	if n >= len(m.turns) {
		m.t.Errorf("llmtest: request %d has no scripted turn; only %d scripted", n+1, len(m.turns))
		return Turn{}, false
	}

	return m.turns[n], true
}

// Requests returns what the model was sent, in order.
func (m *Model) Requests() []llm.Request {
	m.mu.Lock()
	defer m.mu.Unlock()

	return slices.Clone(m.requests)
}

// snapshot copies req's slices, since a caller such as an agent keeps
// appending to its history after sending it.
func snapshot(req llm.Request) llm.Request {
	req.Messages = slices.Clone(req.Messages)
	for i := range req.Messages {
		req.Messages[i].ToolCalls = slices.Clone(req.Messages[i].ToolCalls)
		req.Messages[i].Media = slices.Clone(req.Messages[i].Media)
	}

	req.Tools = slices.Clone(req.Tools)

	return req
}
