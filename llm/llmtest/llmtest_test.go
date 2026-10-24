package llmtest_test

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"slices"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/pushkar-anand/astromech/llm"
	"github.com/pushkar-anand/astromech/llm/llmtest"
)

// streamer is what code under test sees; the fake must fit wherever the
// real client does.
type streamer interface {
	Stream(ctx context.Context, req llm.Request) iter.Seq[llm.Event]
}

var (
	_ streamer = (*llm.Client)(nil)
	_ streamer = (*llmtest.Model)(nil)
)

// recorder is a testing.TB that keeps errors and cleanups instead of acting
// on them, to test how the fake fails a test.
type recorder struct {
	testing.TB

	mu       sync.Mutex
	errors   []string
	cleanups []func()
}

func (r *recorder) Helper() {}

func (r *recorder) Errorf(format string, args ...any) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.errors = append(r.errors, fmt.Sprintf(format, args...))
}

func (r *recorder) Cleanup(f func()) { r.cleanups = append(r.cleanups, f) }

// finish runs the cleanups, as the end of a test does, and returns the
// errors reported.
func (r *recorder) finish() []string {
	for _, f := range slices.Backward(r.cleanups) {
		f()
	}

	return r.errors
}

func ask(text string) llm.Request {
	return llm.Request{Model: "model-a", Messages: []llm.Message{{Role: llm.RoleUser, Content: text}}}
}

func TestModelAnswersInOrder(t *testing.T) {
	call := llm.ToolCall{ID: "call_1", Name: "add", Arguments: `{"a":2,"b":2}`}
	m := llmtest.New(t, llmtest.ToolCalls(call), llmtest.Text("4"))

	first, err := llm.Collect(m.Stream(t.Context(), ask("2 + 2?")), nil)
	require.NoError(t, err)
	assert.Equal(t, []llm.ToolCall{call}, first.Message.ToolCalls)

	second, err := llm.Collect(m.Stream(t.Context(), ask("and now?")), nil)
	require.NoError(t, err)
	assert.Equal(t, "4", second.Message.Content)

	assert.Equal(t, []llm.Request{ask("2 + 2?"), ask("and now?")}, m.Requests())
}

func TestModelTurns(t *testing.T) {
	boom := errors.New("boom")
	call := llm.ToolCall{ID: "call_1", Name: "now"}

	for name, tc := range map[string]struct {
		turn llmtest.Turn
		want []llm.Event
	}{
		"text": {
			turn: llmtest.Text("Hi"),
			want: []llm.Event{
				llm.TextDelta{Text: "Hi"},
				llm.Done{Response: &llm.Response{
					Message:      llm.Message{Role: llm.RoleAssistant, Content: "Hi"},
					FinishReason: "stop",
				}},
			},
		},
		"empty text": {
			turn: llmtest.Text(""),
			want: []llm.Event{
				llm.Done{Response: &llm.Response{
					Message:      llm.Message{Role: llm.RoleAssistant},
					FinishReason: "stop",
				}},
			},
		},
		"tool calls": {
			turn: llmtest.ToolCalls(call),
			want: []llm.Event{
				llm.Done{Response: &llm.Response{
					Message:      llm.Message{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{call}},
					FinishReason: "tool_calls",
				}},
			},
		},
		"fail": {
			turn: llmtest.Fail(boom),
			want: []llm.Event{llm.Failed{Err: boom}},
		},
		"events": {
			turn: llmtest.Events(llm.TextDelta{Text: "cut "}),
			want: []llm.Event{llm.TextDelta{Text: "cut "}},
		},
	} {
		t.Run(name, func(t *testing.T) {
			m := llmtest.New(t, tc.turn)

			assert.Equal(t, tc.want, slices.Collect(m.Stream(t.Context(), ask("hi"))))
		})
	}
}

func TestModelKeepsWhatWasSent(t *testing.T) {
	m := llmtest.New(t, llmtest.Text("ok"))

	req := ask("hi")
	req.Messages = slices.Grow(req.Messages, 2)
	req.Tools = []llm.Tool{{Name: "now"}}

	for range m.Stream(t.Context(), req) {
	}

	// What an agent does next: reuse the backing arrays.
	req.Messages[0].Content = "changed"
	req.Tools[0].Name = "changed"
	_ = append(req.Messages, llm.Message{Role: llm.RoleAssistant, Content: "ok"})

	got := m.Requests()[0]
	assert.Equal(t, "hi", got.Messages[0].Content)
	assert.Equal(t, "now", got.Tools[0].Name)
	assert.Len(t, got.Messages, 1)
}

func TestModelSendsNothingUntilRanged(t *testing.T) {
	m := llmtest.New(t, llmtest.Text("ok"))

	stream := m.Stream(t.Context(), ask("hi"))
	assert.Empty(t, m.Requests())

	for range stream {
	}

	assert.Len(t, m.Requests(), 1)
}

func TestModelStopsEarly(t *testing.T) {
	m := llmtest.New(t, llmtest.Events(
		llm.TextDelta{Text: "one"},
		llm.TextDelta{Text: "two"},
	))

	var seen []llm.Event
	for ev := range m.Stream(t.Context(), ask("hi")) {
		seen = append(seen, ev)
		break
	}

	assert.Equal(t, []llm.Event{llm.TextDelta{Text: "one"}}, seen)
}

func TestModelCanceled(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	cancel()

	r := &recorder{TB: t}
	m := llmtest.New(r, llmtest.Text("ok"))

	_, err := llm.Collect(m.Stream(ctx, ask("hi")), nil)
	require.ErrorIs(t, err, context.Canceled)
	assert.Empty(t, m.Requests(), "a canceled request is not sent")

	assert.Equal(t, []string{"llmtest: 1 of 1 scripted turns were not used"}, r.finish())
}

func TestModelCanceledMidway(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()

	m := llmtest.New(t, llmtest.Text("Hi"))

	_, err := llm.Collect(m.Stream(ctx, ask("hi")), func(ev llm.Event) {
		if _, ok := ev.(llm.TextDelta); ok {
			cancel()
		}
	})
	assert.ErrorIs(t, err, context.Canceled)
}

func TestModelTooManyRequests(t *testing.T) {
	r := &recorder{TB: t}
	m := llmtest.New(r, llmtest.Text("ok"))

	_, err := llm.Collect(m.Stream(t.Context(), ask("hi")), nil)
	require.NoError(t, err)

	_, err = llm.Collect(m.Stream(t.Context(), ask("again")), nil)
	require.ErrorIs(t, err, llmtest.ErrNoTurn)

	assert.Len(t, m.Requests(), 2)
	assert.Equal(t, []string{"llmtest: request 2 has no scripted turn; only 1 scripted"}, r.finish())
}

func TestModelUnusedTurns(t *testing.T) {
	r := &recorder{TB: t}
	m := llmtest.New(r, llmtest.Text("one"), llmtest.Text("two"))

	_, err := llm.Collect(m.Stream(t.Context(), ask("hi")), nil)
	require.NoError(t, err)

	assert.Equal(t, []string{"llmtest: 1 of 2 scripted turns were not used"}, r.finish())
}

func TestModelConcurrent(t *testing.T) {
	const n = 8

	turns := make([]llmtest.Turn, n)
	for i := range turns {
		turns[i] = llmtest.Text("ok")
	}

	m := llmtest.New(t, turns...)

	var wg sync.WaitGroup
	for range n {
		wg.Go(func() {
			_, err := llm.Collect(m.Stream(t.Context(), ask("hi")), nil)
			assert.NoError(t, err)
		})
	}

	wg.Wait()
	assert.Len(t, m.Requests(), n)
}
