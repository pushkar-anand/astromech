package agent

import "github.com/pushkar-anand/astromech/llm"

type Event interface{ event() }

type Stop string

const (
	StopAnswered         Stop = "answered"
	StopMaxRounds        Stop = "max_rounds"
	StopRepeating        Stop = "repeating"         // a call kept returning the same result
	StopAwaitingApproval Stop = "awaiting_approval" // calls wait for a person's verdict
)

type (
	TextDelta struct{ Text string }

	ReasoningDelta struct{ Text string }

	ToolStarted struct{ Call llm.ToolCall }

	ToolFinished struct {
		Call   llm.ToolCall
		Result Result
	}

	MessageAdded struct{ Message llm.Message }

	Done struct {
		Stop     Stop
		Messages []llm.Message
		Usage    llm.Usage
	}

	Failed struct {
		Err      error
		Messages []llm.Message
		Usage    llm.Usage
	}

	// ApprovalRequired is a call waiting for a person. The turn ends with
	// StopAwaitingApproval, and the verdict goes in the next turn's
	// Decisions.
	ApprovalRequired struct {
		Call   llm.ToolCall
		Tool   llm.Tool
		Reason string // why an approver asked, when one did
	}
)

func (TextDelta) event()        {}
func (ReasoningDelta) event()   {}
func (ToolStarted) event()      {}
func (ToolFinished) event()     {}
func (MessageAdded) event()     {}
func (Done) event()             {}
func (Failed) event()           {}
func (ApprovalRequired) event() {}

var _ Event = TextDelta{}
var _ Event = ReasoningDelta{}
var _ Event = ToolStarted{}
var _ Event = ToolFinished{}
var _ Event = MessageAdded{}
var _ Event = Done{}
var _ Event = Failed{}
var _ Event = ApprovalRequired{}
