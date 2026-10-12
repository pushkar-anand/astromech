package agent

import (
	"context"
	"fmt"
	"slices"

	"github.com/pushkar-anand/astromech/llm"
)

// Mode is how calls to tools that change things are approved in a
// conversation. Read-only tools run without approval in every mode. The
// zero Mode is [ModeManual].
type Mode string

// Mode values.
const (
	// ModeManual asks a person before every call that changes things.
	ModeManual Mode = "manual"

	// ModeAuto lets [Agent.Approver] decide each call, and asks a person
	// when it cannot. Without an approver it is ModeManual.
	ModeAuto Mode = "auto"

	// ModeBypass runs every call without asking.
	ModeBypass Mode = "bypass"

	// ModeReadOnly offers the model read-only tools only.
	ModeReadOnly Mode = "read_only"
)

// Approver decides calls in [ModeAuto].
type Approver interface {
	// Decide returns a verdict on req. An error, or a verdict to ask,
	// leaves the call to a person.
	Decide(ctx context.Context, req ApprovalRequest) (Verdict, error)
}

// ApprovalRequest is a call waiting for a verdict.
type ApprovalRequest struct {
	Call llm.ToolCall
	Tool llm.ToolDef

	// Conversation is everything before the call, the reply that made it
	// included. Its tool results come from outside the conversation and
	// can carry instructions planted for a model to follow, so an approver
	// that reads them can be talked into allowing a call.
	Conversation []llm.Message
}

// Decision is what a [Verdict] decides.
type Decision string

// Decision values.
const (
	DecisionAllow Decision = "allow"
	DecisionDeny  Decision = "deny"

	// DecisionAsk leaves the call to a person.
	DecisionAsk Decision = "ask"
)

// Verdict decides a call.
type Verdict struct {
	Decision Decision `json:"decision"`

	// Reason says why. The model reads the reason a call was denied, so it
	// can tell the user or try another way.
	Reason string `json:"reason,omitzero"`
}

// Pending returns the calls in history that wait for a verdict: the calls
// of its last reply that have no result yet. A program shows them after a
// restart, or when a person comes back to the conversation. It returns nil
// when no call waits.
func Pending(history []llm.Message) []llm.ToolCall {
	calls, _ := pending(history)

	return calls
}

// pending is [Pending], and also reports a history whose unanswered calls
// are followed by messages other than tool results. A server refuses such
// a history, so a turn refuses it too.
func pending(history []llm.Message) ([]llm.ToolCall, error) {
	at := -1

	for i, m := range slices.Backward(history) {
		if m.Role == llm.RoleAssistant && len(m.ToolCalls) > 0 {
			at = i

			break
		}
	}

	if at < 0 {
		return nil, nil
	}

	answered := make(map[string]bool)
	later := false

	for _, m := range history[at+1:] {
		if m.Role == llm.RoleTool {
			answered[m.ToolCallID] = true
		} else {
			later = true
		}
	}

	var calls []llm.ToolCall

	for _, call := range history[at].ToolCalls {
		if !answered[call.ID] {
			calls = append(calls, call)
		}
	}

	if len(calls) > 0 && later {
		return nil, fmt.Errorf("agent: call %s in the history has no result, but other messages follow it; a server refuses that history", calls[0].ID)
	}

	return calls, nil
}
