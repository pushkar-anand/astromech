package llm

import (
	"context"
	"fmt"
	"iter"
	"net/http"
	"strings"

	"github.com/pushkar-anand/build-with-go/http/client"
)

// streamEnd is the data of the event that ends a stream.
const streamEnd = "[DONE]"

// Wire format of a stream.
type (
	streamRequest struct {
		Request
		Stream        bool           `json:"stream"`
		StreamOptions *streamOptions `json:"stream_options,omitzero"`
	}

	streamOptions struct {
		IncludeUsage bool `json:"include_usage"`
	}

	streamChunk struct {
		ID      string `json:"id"`
		Model   string `json:"model"`
		Choices []struct {
			Delta        replyMessage `json:"delta"`
			FinishReason string       `json:"finish_reason"`
		} `json:"choices"`
		Usage *Usage `json:"usage,omitzero"`
		Stats *Stats `json:"timings,omitzero"`
	}
)

type (
	// Event is one step of a streamed reply. A stream yields any number of
	// deltas, then exactly one [Done] or [Failed], unless the caller stops
	// reading first.
	Event interface{ event() }

	// TextDelta is the next piece of the reply.
	TextDelta struct{ Text string }

	// ReasoningDelta is the next piece of the model's thinking.
	ReasoningDelta struct{ Text string }

	// Done ends a stream that finished, with the whole reply.
	Done struct{ Response *Response }

	// Failed ends a stream that did not finish.
	Failed struct{ Err error }
)

func (TextDelta) event()      {}
func (ReasoningDelta) event() {}
func (Done) event()           {}
func (Failed) event()         {}

// Stream sends req and yields the reply as it arrives. Stopping the loop
// early cancels the request.
func (c *Client) Stream(ctx context.Context, req Request) iter.Seq[Event] {
	return func(yield func(Event) bool) {
		sreq := streamRequest{
			Request:       req,
			Stream:        true,
			StreamOptions: &streamOptions{IncludeUsage: true},
		}

		stream := c.httpClient.Stream[streamChunk](ctx, http.MethodPost, chatCompletionPath, sreq, client.WithStreamEnd(streamEnd))

		var (
			text, reasoning strings.Builder
			resp            Response
		)

		for event, err := range stream {
			if err != nil {
				yield(Failed{Err: fmt.Errorf("llm: stream: %w", err)})
				return
			}

			chunk := event.Data

			// Recorded before the choices check: usage and timings come on
			// the last chunk, which has no choices.
			if resp.ID == "" {
				resp.ID = chunk.ID
			}

			if resp.Model == "" {
				resp.Model = chunk.Model
			}

			if chunk.Usage != nil {
				resp.Usage = *chunk.Usage
			}

			if chunk.Stats != nil {
				resp.Stats = chunk.Stats
			}

			if len(chunk.Choices) == 0 {
				continue
			}

			choice := chunk.Choices[0]

			if choice.Delta.ReasoningContent != "" {
				reasoning.WriteString(choice.Delta.ReasoningContent)

				if !yield(ReasoningDelta{Text: choice.Delta.ReasoningContent}) {
					return
				}
			}

			if choice.Delta.Content != "" {
				text.WriteString(choice.Delta.Content)

				if !yield(TextDelta{Text: choice.Delta.Content}) {
					return
				}
			}

			if choice.FinishReason != "" {
				resp.FinishReason = choice.FinishReason
			}
		}

		if resp.FinishReason == "" {
			yield(Failed{Err: ErrIncompleteStream})
			return
		}

		resp.Message = Message{Role: RoleAssistant, Content: text.String()}
		resp.Reasoning = reasoning.String()

		yield(Done{Response: &resp})
	}
}
