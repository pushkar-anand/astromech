package agent

import "github.com/pushkar-anand/astromech/llm"

type Event interface{ event() }

type Stop string

const (
	StopAnswered  Stop = "answered"
	StopMaxRounds Stop = "max_rounds"
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
)

func (TextDelta) event()      {}
func (ReasoningDelta) event() {}
func (ToolStarted) event()    {}
func (ToolFinished) event()   {}
func (MessageAdded) event()   {}
func (Done) event()           {}
func (Failed) event()         {}

var _ Event = TextDelta{}
var _ Event = ReasoningDelta{}
var _ Event = ToolStarted{}
var _ Event = ToolFinished{}
var _ Event = MessageAdded{}
var _ Event = Done{}
var _ Event = Failed{}
