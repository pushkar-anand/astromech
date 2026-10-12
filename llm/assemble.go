package llm

import (
	"cmp"
	"context"
	"crypto/rand"
	"encoding/json"
	"strings"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/respjson"
)

// assembler builds a [Response] from the chunks of a stream, emitting an
// [Event] for each piece as it arrives.
type assembler struct {
	id, model string

	text      strings.Builder
	reasoning strings.Builder

	// calls holds the reply's tool calls in the order they started, and
	// slots maps the index a server gave a call to its place in calls.
	calls []*pendingCall
	slots map[int64]int

	finishReason string
	finished     bool
	usage        Usage
}

type pendingCall struct {
	id, name string
	args     strings.Builder
}

// add folds chunk into the reply. It returns false once emit does, which
// means the caller stopped reading.
func (a *assembler) add(chunk openai.ChatCompletionChunk, emit func(Event) bool) bool {
	a.id = cmp.Or(a.id, chunk.ID)
	a.model = cmp.Or(a.model, chunk.Model)

	// Usage arrives on its own chunk, with no choices, after the last one
	// that carries content.
	if chunk.JSON.Usage.Valid() {
		a.usage = usageOf(chunk.Usage)
	}

	if len(chunk.Choices) == 0 {
		return true
	}

	choice := chunk.Choices[0]
	delta := choice.Delta

	if r := reasoningOf(delta.JSON.ExtraFields); r != "" {
		a.reasoning.WriteString(r)

		if !emit(ReasoningDelta{Text: r}) {
			return false
		}
	}

	if delta.Content != "" {
		a.text.WriteString(delta.Content)

		if !emit(TextDelta{Text: delta.Content}) {
			return false
		}
	}

	for _, tc := range delta.ToolCalls {
		if !a.addCall(tc, emit) {
			return false
		}
	}

	if choice.FinishReason != "" {
		a.finishReason = choice.FinishReason
		a.finished = true
	}

	return true
}

func (a *assembler) addCall(tc openai.ChatCompletionChunkChoiceDeltaToolCall, emit func(Event) bool) bool {
	i, started := a.callIndex(tc)
	call := a.calls[i]

	if started {
		if !emit(ToolCallStarted{Index: i, ID: call.id, Name: call.name}) {
			return false
		}
	} else {
		call.id = cmp.Or(call.id, tc.ID)
		call.name = cmp.Or(call.name, tc.Function.Name)
	}

	args := argumentsOf(tc.Function.Arguments, tc.Function.JSON.Arguments)
	if args == "" {
		return true
	}

	call.args.WriteString(args)

	return emit(ToolCallDelta{Index: i, Arguments: args})
}

// callIndex places a tool call delta among the reply's calls, starting a
// new call when the delta begins one, and reports whether it did. Some
// local servers send every call at index 0: a new ID there starts a new
// call, and the pieces that follow without an ID belong to it. Without an
// index, a delta that names a tool starts a call and one that names
// neither a tool nor an ID continues the latest. The server's index only
// names a call, so an index far out of order or below 0 costs nothing.
func (a *assembler) callIndex(tc openai.ChatCompletionChunkChoiceDeltaToolCall) (int, bool) {
	if tc.JSON.Index.Valid() {
		if i, ok := a.slots[tc.Index]; ok {
			id := a.calls[i].id
			if tc.ID == "" || id == "" || id == tc.ID {
				return i, false
			}
		}
	} else if tc.ID == "" && tc.Function.Name == "" && len(a.calls) > 0 {
		return len(a.calls) - 1, false
	}

	i := len(a.calls)
	a.calls = append(a.calls, &pendingCall{id: tc.ID, name: tc.Function.Name})

	if tc.JSON.Index.Valid() {
		if a.slots == nil {
			a.slots = make(map[int64]int)
		}

		a.slots[tc.Index] = i
	}

	return i, true
}

// finish emits each tool call as done and returns the reply. It returns
// false once emit does.
func (a *assembler) finish(emit func(Event) bool) (*Response, bool) {
	resp := a.response(stopReason(a.finishReason, len(a.calls)))

	for i, call := range resp.Message.ToolCalls {
		if !emit(ToolCallDone{Index: i, Call: call}) {
			return nil, false
		}
	}

	return resp, true
}

// partial returns the text and reasoning that arrived before a stream
// failed, or nil when none did. Tool calls are left out: their arguments
// can be cut off, and a server refuses a conversation that holds a call
// with no result.
func (a *assembler) partial(ctx context.Context) *Response {
	if a.text.Len() == 0 && a.reasoning.Len() == 0 {
		return nil
	}

	stop := StopFailed
	if ctx.Err() != nil {
		stop = StopCanceled
	}

	resp := a.response(stop)
	resp.Message.ToolCalls = nil

	return resp
}

func (a *assembler) response(stop StopReason) *Response {
	msg := Message{
		Role:      RoleAssistant,
		Content:   a.text.String(),
		Reasoning: a.reasoning.String(),
	}

	for _, call := range a.calls {
		msg.ToolCalls = append(msg.ToolCalls, completeCall(call.id, call.name, call.args.String()))
	}

	return &Response{ID: a.id, Model: a.model, Message: msg, Stop: stop, Usage: a.usage}
}

// wholeReply maps a whole reply from the SDK. A gateway whose upstream
// model fails can answer 200 with an error object in place of choices; that
// error is returned.
func wholeReply(c *openai.ChatCompletion) (*Response, error) {
	if len(c.Choices) > 0 {
		return completionResponse(c), nil
	}

	if f, ok := c.JSON.ExtraFields["error"]; ok && f.Raw() != respjson.Null {
		return nil, errorInBody([]byte(c.RawJSON()))
	}

	return nil, ErrEmptyResponse
}

// replay emits the events a stream of resp would have carried. It returns
// false once emit does.
func replay(resp *Response, emit func(Event) bool) bool {
	m := resp.Message

	if m.Reasoning != "" && !emit(ReasoningDelta{Text: m.Reasoning}) {
		return false
	}

	if m.Content != "" && !emit(TextDelta{Text: m.Content}) {
		return false
	}

	for i, call := range m.ToolCalls {
		if !emit(ToolCallStarted{Index: i, ID: call.ID, Name: call.Name}) ||
			!emit(ToolCallDelta{Index: i, Arguments: call.Arguments}) ||
			!emit(ToolCallDone{Index: i, Call: call}) {
			return false
		}
	}

	return true
}

// completionResponse maps a whole reply from the SDK.
func completionResponse(c *openai.ChatCompletion) *Response {
	choice := c.Choices[0]
	m := choice.Message

	msg := Message{
		Role:      RoleAssistant,
		Content:   m.Content,
		Reasoning: reasoningOf(m.JSON.ExtraFields),
	}

	for _, tc := range m.ToolCalls {
		msg.ToolCalls = append(msg.ToolCalls, completeCall(tc.ID, tc.Function.Name, argumentsOf(tc.Function.Arguments, tc.Function.JSON.Arguments)))
	}

	var usage Usage
	if c.JSON.Usage.Valid() {
		usage = usageOf(c.Usage)
	}

	return &Response{
		ID:      c.ID,
		Model:   c.Model,
		Message: msg,
		Stop:    stopReason(choice.FinishReason, len(msg.ToolCalls)),
		Usage:   usage,
	}
}

// completeCall fills in what a server left out of a finished call: an ID,
// which the call's result has to name, and arguments for a tool that takes
// none.
func completeCall(id, name, args string) ToolCall {
	if id == "" {
		id = "call_" + rand.Text()
	}

	return ToolCall{ID: id, Name: name, Arguments: argumentsOrEmpty(args)}
}

// argumentsOrEmpty returns args, or an empty JSON object for a call with no
// arguments, which servers refuse to read back as an empty string.
func argumentsOrEmpty(args string) string {
	if strings.TrimSpace(args) == "" {
		return "{}"
	}

	return args
}

// argumentsOf returns a tool call's arguments as JSON text. The SDK decodes
// them as a string, which is what OpenAI sends; a server that sends the
// object itself leaves the string empty and the raw object in field.
func argumentsOf(decoded string, field respjson.Field) string {
	if field.Valid() {
		return decoded
	}

	if raw := field.Raw(); raw != "" && raw != respjson.Null {
		return raw
	}

	return decoded
}

// reasoningOf returns the reasoning a server sent alongside a reply, under
// whichever name it uses: reasoning_content (DeepSeek, vLLM) or reasoning
// (OpenRouter, Ollama).
func reasoningOf(fields map[string]respjson.Field) string {
	for _, name := range []string{"reasoning_content", "reasoning"} {
		var s string
		if json.Unmarshal([]byte(fields[name].Raw()), &s) == nil && s != "" {
			return s
		}
	}

	return ""
}

func usageOf(u openai.CompletionUsage) Usage {
	return Usage{
		InputTokens:     int(u.PromptTokens),
		OutputTokens:    int(u.CompletionTokens),
		CachedTokens:    int(u.PromptTokensDetails.CachedTokens),
		ReasoningTokens: int(u.CompletionTokensDetails.ReasoningTokens),
	}
}

// stopReason maps a server's finish reason. A reply that carries tool calls
// waits for their results whatever the server said, because some servers
// send "stop" there.
func stopReason(finish string, calls int) StopReason {
	if calls > 0 {
		return StopToolCalls
	}

	switch finish {
	case "length":
		return StopLength
	case "content_filter":
		return StopFiltered
	case "tool_calls", "function_call":
		return StopToolCalls
	}

	return StopEnd
}
