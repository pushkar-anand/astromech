package llm

import "errors"

var ErrIncompleteStream = errors.New("llm: the stream ended before the reply finished")
