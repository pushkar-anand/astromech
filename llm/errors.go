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
