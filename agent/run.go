package agent

import (
	"context"
	"fmt"

	"github.com/pushkar-anand/astromech/llm"
)

// run is the state of one call to Run. It lives for one turn and is never
// shared, which is why it can hold the turn's ctx and yield.
type run struct {
	ctx   context.Context
	agent *Agent

	yield func(event Event) bool

	tools    []llm.Tool
	toolsMap map[string]Tool

	input   *llm.Message
	history []llm.Message
	current []llm.Message // what this turn added; Done and Failed report it

	usage llm.Usage

	roundLimit int

	repeats     map[repeat]int // how often each call returned each result
	repeatLimit int
	repeating   bool // a call went past repeatLimit; stop after this round
}

// repeat is one tool call and what it returned, as loop detection counts it.
type repeat struct{ name, args, content string }

// loop runs the turn: the input, then rounds until the model answers, a
// call keeps returning the same result, or the round limit is reached.
func (r *run) loop() {
	if !r.add(r.input) {
		return
	}

	for i := 0; r.roundLimit < 0 || i < r.roundLimit; i++ {
		response, ok := r.round(r.request())
		if !ok {
			return
		}

		if !r.add(&response.Message) {
			return
		}

		if len(response.Message.ToolCalls) == 0 {
			r.done(StopAnswered)
			return
		}

		if ok := r.callAllTools(response.Message.ToolCalls); !ok {
			return
		}

		if r.repeating {
			r.done(StopRepeating)
			return
		}
	}

	r.done(StopMaxRounds)
}

// request is what this round sends: the system prompt, the history and
// the turn so far. It is rebuilt every round, never stored.
func (r *run) request() llm.Request {
	var send []llm.Message

	if r.agent.System != "" {
		send = append(send, llm.Message{
			Role:    llm.RoleSystem,
			Content: r.agent.System,
		})
	}

	send = append(send, r.history...)
	send = append(send, r.current...)

	return llm.Request{
		Model:    r.agent.ModelID,
		Messages: send,
		Tools:    r.tools,
	}
}

// round streams one reply to the caller, adds its usage, and returns it.
// It is false when the turn is over: the caller stopped, or the model
// failed, which round has already reported.
func (r *run) round(request llm.Request) (*llm.Response, bool) {
	var response *llm.Response

	for event := range r.agent.Model.Stream(r.ctx, request) {
		switch e := event.(type) {
		case llm.TextDelta:
			if !r.yield(TextDelta{Text: e.Text}) {
				return nil, false
			}
		case llm.ReasoningDelta:
			if !r.yield(ReasoningDelta{Text: e.Text}) {
				return nil, false
			}
		case llm.Failed:
			r.fail(e.Err)
			return nil, false
		case llm.Done:
			r.usage = r.usage.Add(e.Response.Usage)
			response = e.Response
		}
	}

	// llm.Client always ends in Done or Failed; other models may not.
	if response == nil {
		r.fail(llm.ErrIncompleteStream)
		return nil, false
	}

	return response, true
}

// add appends msg to the turn and reports it. A nil msg adds nothing. It
// is false when the caller stopped.
func (r *run) add(msg *llm.Message) bool {
	if msg == nil {
		return true
	}

	r.current = append(r.current, *msg)

	return r.yield(MessageAdded{Message: *msg})
}

// done ends the turn with reason. The caller returns after it.
func (r *run) done(reason Stop) {
	r.yield(Done{
		Stop:     reason,
		Messages: r.current,
		Usage:    r.usage,
	})
}

// fail ends the turn with err. The caller returns after it.
func (r *run) fail(err error) {
	r.yield(Failed{Err: err, Messages: r.current, Usage: r.usage})
}

// callAllTools runs a round's tool calls in order, adding each result to
// the turn. It is false when the turn is over: the caller stopped, or ctx
// was done before a call, which callAllTools has already reported.
func (r *run) callAllTools(toolCalls []llm.ToolCall) bool {
	for i, toolCall := range toolCalls {
		if err := r.ctx.Err(); err != nil {
			r.skip(toolCalls[i:], err)
			return false
		}

		if !r.yield(ToolStarted{Call: toolCall}) {
			return false
		}

		result := r.noteRepeat(toolCall, callTool(r.ctx, r.toolsMap, toolCall))

		if !r.yield(ToolFinished{Call: toolCall, Result: result}) {
			return false
		}

		msg := llm.Message{
			Role:       llm.RoleTool,
			Content:    result.Content,
			ToolCallID: toolCall.ID,
		}

		if !r.add(&msg) {
			return false
		}
	}

	return true
}

// skip ends the turn with err, first giving each call that will not run a
// result: every call needs one, or the turn's messages are not a history
// the caller can store and resume.
func (r *run) skip(toolCalls []llm.ToolCall, err error) {
	for _, toolCall := range toolCalls {
		msg := llm.Message{
			Role:       llm.RoleTool,
			Content:    fmt.Sprintf("%s was not run: %v", toolCall.Name, err),
			ToolCallID: toolCall.ID,
		}

		if !r.add(&msg) {
			return
		}
	}

	r.fail(err)
}

// noteRepeat counts toolCall and its result. At repeatLimit it tells the
// model the result will not change; past it, the turn stops once every call
// this round has a result, so the history stays valid to resume.
func (r *run) noteRepeat(toolCall llm.ToolCall, result Result) Result {
	if r.repeatLimit < 0 {
		return result
	}

	key := repeat{name: toolCall.Name, args: toolCall.Arguments, content: result.Content}
	r.repeats[key]++

	switch n := r.repeats[key]; {
	case n == r.repeatLimit:
		result.Content += fmt.Sprintf("\n\n(%s has returned this %d times for these arguments, and will again: try something else, or answer.)", toolCall.Name, n)
	case n > r.repeatLimit:
		r.repeating = true
	}

	return result
}
