package llm

import "errors"

var (
	// ErrNoReply is returned when the LLM backend answers with no choices
	// and no error to say why.
	ErrNoReply = errors.New("llm: the LLM backend sent no reply")

	// ErrIncompleteStream is returned when a stream ends before the LLM
	// backend said the reply was finished, such as when a proxy drops the
	// connection.
	ErrIncompleteStream = errors.New("llm: the stream ended before the reply finished")
)

// BackendError is an error the LLM backend reported partway through a
// stream, after it had already answered 200.
type BackendError struct {
	Message string `json:"message"`

	// Type is the kind of error, such as "server_error", when the LLM
	// backend says.
	Type string `json:"type"`
}

func (e *BackendError) Error() string {
	return "llm: the LLM backend failed: " + e.Message
}
