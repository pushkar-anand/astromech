package llm

import (
	"errors"
	"iter"
)

// Event is one step of a streamed reply. A stream yields any number of
// deltas and [Retrying] events, then exactly one [Done] or [Failed], unless
// the caller stops reading first.
//
// The concrete types are [TextDelta], [ReasoningDelta], [ToolCallStarted],
// [ToolCallDelta], [ToolCallDone], [Retrying], [Done] and [Failed].
type Event interface {
	event()
}

// TextDelta is the next piece of the reply's text.
type TextDelta struct {
	Text string
}

// ReasoningDelta is the next piece of a reasoning model's thinking.
type ReasoningDelta struct {
	Text string
}

// ToolCallStarted is the first sight of a tool call. Index orders the calls
// of one reply from 0. ID is empty when the server sends none; the call's
// [ToolCallDone] carries the ID the call is answered by.
type ToolCallStarted struct {
	Index int
	ID    string
	Name  string
}

// ToolCallDelta is the next piece of a tool call's arguments. The pieces are
// fragments of JSON and parse only once the call is done.
type ToolCallDelta struct {
	Index     int
	Arguments string
}

// ToolCallDone is a tool call whose arguments are complete. Every call is
// done before the stream's [Done].
type ToolCallDone struct {
	Index int
	Call  ToolCall
}

// Retrying says the request failed before any of the reply arrived and is
// being sent again.
type Retrying struct {
	// Attempt counts retries from 1.
	Attempt int

	// Err is why the previous attempt failed.
	Err error
}

// Done ends a stream that finished.
type Done struct {
	Response *Response
}

// Failed ends a stream that did not finish. Partial holds the text and
// reasoning that arrived before the failure, with Stop set to
// [StopCanceled] or [StopFailed], so a caller can keep it in the
// conversation and ask the model to continue. Tool calls are left out of
// it, because their arguments can be cut off. It is nil when no text or
// reasoning arrived.
type Failed struct {
	Err     error
	Partial *Response
}

func (TextDelta) event()       {}
func (ReasoningDelta) event()  {}
func (ToolCallStarted) event() {}
func (ToolCallDelta) event()   {}
func (ToolCallDone) event()    {}
func (Retrying) event()        {}
func (Done) event()            {}
func (Failed) event()          {}

// Collect reads events to the end and returns the reply. A stream that
// failed returns its partial reply, which may be nil, with the error.
func Collect(events iter.Seq[Event]) (*Response, error) {
	for e := range events {
		switch e := e.(type) {
		case Done:
			return e.Response, nil
		case Failed:
			return e.Partial, e.Err
		}
	}

	return nil, errors.New("llm: the stream ended without a reply; the caller stopped reading or the stream is not from Client.Stream")
}
