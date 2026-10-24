package llm

import (
	json "encoding/json/v2"
	"errors"
	"fmt"
	"net/http"

	"github.com/pushkar-anand/build-with-go/http/client"
)

var (
	// ErrNoReply is returned when the LLM backend answers with no choices
	// and no error to say why.
	ErrNoReply = errors.New("llm: the LLM backend sent no reply")

	// ErrIncompleteStream is returned when a stream ends before the LLM
	// backend said the reply was finished, such as when a proxy drops the
	// connection.
	ErrIncompleteStream = errors.New("llm: the stream ended before the reply finished")
)

// BackendError is an error the LLM backend explained, either in the body of
// an error response or partway through a stream.
type BackendError struct {
	// StatusCode is the HTTP status of an error response, and 0 for an
	// error sent partway through a stream.
	StatusCode int `json:"-"`

	Message string `json:"message"`

	// Type is the kind of error, such as "server_error", when the LLM
	// backend says.
	Type string `json:"type"`

	// status is the error response, for its headers.
	status *client.StatusError
}

func (e *BackendError) Error() string {
	if e.StatusCode != 0 {
		return fmt.Sprintf("the LLM backend answered %d %s: %s", e.StatusCode, http.StatusText(e.StatusCode), e.Message)
	}

	return "the LLM backend failed: " + e.Message
}

// Unwrap returns the *client.StatusError of an error response, so its
// headers, such as Retry-After, stay reachable.
func (e *BackendError) Unwrap() error {
	if e.status == nil {
		return nil
	}

	return e.status
}

// explain replaces an error response with a BackendError when its body
// says what went wrong, and returns other errors as they are.
func explain(err error) error {
	status, ok := errors.AsType[*client.StatusError](err)
	if !ok {
		return err
	}

	var body struct {
		Error *BackendError `json:"error"`
	}

	if json.Unmarshal(status.Body, &body) != nil || body.Error == nil || body.Error.Message == "" {
		return err
	}

	body.Error.StatusCode = status.StatusCode
	body.Error.status = status

	return body.Error
}
