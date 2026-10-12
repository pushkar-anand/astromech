package agent

import "github.com/pushkar-anand/astromech/llm"

// Event is one step of a turn. A turn yields any number of the other
// events, then exactly one [Done] or [Failed], unless the caller stops
// reading first.
//
// The concrete types are [TextDelta], [ReasoningDelta], [Retrying],
// [ToolStarted], [ToolFinished], [ApprovalRequired], [Decided],
// [MessageAdded], [Done] and [Failed].
type Event interface {
	event()
}

// TextDelta is the next piece of the model's answer.
type TextDelta struct {
	Text string
}

// ReasoningDelta is the next piece of a reasoning model's thinking.
type ReasoningDelta struct {
	Text string
}

// Retrying says a request to the model failed before any of its reply
// arrived and is being sent again.
type Retrying struct {
	// Attempt counts retries from 1.
	Attempt int

	// Err is why the previous attempt failed.
	Err error
}

// ToolStarted says a tool call is running.
type ToolStarted struct {
	Call llm.ToolCall
}

// ToolFinished is the result of a tool call: what the tool returned, or why
// the call did not run. Every call the model makes ends with one, unless it
// waits for approval.
type ToolFinished struct {
	Call   llm.ToolCall
	Result Result
}

// ApprovalRequired says a call waits for a person to allow or deny it. The
// turn ends with [StopAwaitingApproval] once every other call has run, and
// the person's [Verdict] goes in the next turn's [Turn.Decisions].
type ApprovalRequired struct {
	Call llm.ToolCall
	Tool llm.ToolDef

	// Reason is why the approver asked, when one did.
	Reason string
}

// Decided says the approver allowed or denied a call without asking a
// person. Its reason is for showing beside the call.
type Decided struct {
	Call    llm.ToolCall
	Verdict Verdict
}

// MessageAdded is a message the turn added to the conversation. A caller
// that stores each one as it arrives keeps the conversation even if the
// program stops partway through the turn.
type MessageAdded struct {
	Message llm.Message
}

// Done ends a turn that finished.
type Done struct {
	Stop Stop

	// Messages is every message the turn added, in order, as its
	// [MessageAdded] events reported them.
	Messages []llm.Message

	// Usage adds up every request the turn sent.
	Usage llm.Usage
}

// Failed ends a turn that did not finish. Messages holds what the turn
// added before it failed: the user's input, finished replies and results,
// and the text of a reply the failure cut off. Storing them lets the
// conversation continue.
type Failed struct {
	Err      error
	Messages []llm.Message
	Usage    llm.Usage
}

// Stop says why a turn ended.
type Stop string

// Stop values.
const (
	// StopAnswered means the model answered.
	StopAnswered Stop = "answered"

	// StopAwaitingApproval means calls wait for a person's verdict.
	StopAwaitingApproval Stop = "awaiting_approval"

	// StopMaxRounds means the model was still calling tools when the turn
	// reached [Agent.MaxRounds]. A turn with no input continues it.
	StopMaxRounds Stop = "max_rounds"

	// StopLength means the answer was cut off at the model's token limit.
	StopLength Stop = "length"

	// StopFiltered means the server withheld the rest of the answer under
	// its content policy.
	StopFiltered Stop = "filtered"
)

func (TextDelta) event()        {}
func (ReasoningDelta) event()   {}
func (Retrying) event()         {}
func (ToolStarted) event()      {}
func (ToolFinished) event()     {}
func (ApprovalRequired) event() {}
func (Decided) event()          {}
func (MessageAdded) event()     {}
func (Done) event()             {}
func (Failed) event()           {}
