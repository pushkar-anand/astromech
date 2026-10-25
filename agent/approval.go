package agent

// Mode is how a turn decides the tool calls it may run without a person.
// The zero Mode is ModeAllowReadOnly.
type Mode string

const (
	// ModeAllowReadOnly runs read-only calls; the others wait for a person.
	ModeAllowReadOnly Mode = "allow_read_only"

	// ModeManual makes every call wait for a person, read-only ones too.
	ModeManual Mode = "manual"

	// ModeBypass runs every call without asking.
	ModeBypass Mode = "bypass"

	// ModeAuto runs read-only calls and lets an approver decide the
	// others. Without an approver it is ModeManual.
	ModeAuto Mode = "auto"
)

// Decision is what a Verdict decides about a call.
type Decision string

const (
	DecisionAllow Decision = "allow"
	DecisionDeny  Decision = "deny"
	DecisionAsk   Decision = "ask" // leave the call to a person
)

// Verdict decides one call. The model reads Reason when the call is
// denied, so it can tell the user or try another way.
type Verdict struct {
	Decision Decision `json:"decision"`
	Reason   string   `json:"reason,omitzero"`
}

// valid reports whether m is one of the modes.
func (m Mode) valid() bool {
	switch m {
	case ModeAllowReadOnly, ModeManual, ModeBypass, ModeAuto:
		return true
	default:
		return false
	}
}

// valid reports whether d is one of the decisions.
func (d Decision) valid() bool {
	switch d {
	case DecisionAllow, DecisionDeny, DecisionAsk:
		return true
	default:
		return false
	}
}
