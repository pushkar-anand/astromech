package llm

import (
	"cmp"
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
		chatRequest
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
		Usage *usage        `json:"usage,omitzero"`
		Stats *Stats        `json:"timings,omitzero"`
		Error *BackendError `json:"error,omitzero"`
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

// pendingCall is a tool call being put back together from stream pieces.
type pendingCall struct {
	id, name string
	args     strings.Builder
}

// Stream sends req and yields the reply as it arrives. Stopping the loop
// early cancels the request.
func (c *Client) Stream(ctx context.Context, req Request) iter.Seq[Event] {
	return func(yield func(Event) bool) {
		sreq := streamRequest{
			chatRequest:   req.toWire(),
			Stream:        true,
			StreamOptions: &streamOptions{IncludeUsage: true},
		}

		stream := c.httpClient.Stream[streamChunk](ctx, http.MethodPost, chatCompletionPath, sreq, client.WithStreamEnd(streamEnd))

		var (
			text, reasoning strings.Builder
			resp            Response

			calls []*pendingCall           // in the order they started
			byIdx = map[int]*pendingCall{} // the latest call at each index
		)

		for event, err := range stream {
			if err != nil {
				yield(Failed{Err: fmt.Errorf("llm: stream: %w", explain(err))})
				return
			}

			chunk := event.Data

			if chunk.Error != nil {
				yield(Failed{Err: fmt.Errorf("llm: stream: %w", chunk.Error)})
				return
			}

			// Recorded before the choices check: usage and timings come on
			// the last chunk, which has no choices.
			resp.ID = cmp.Or(resp.ID, chunk.ID)
			resp.Model = cmp.Or(resp.Model, chunk.Model)

			if chunk.Usage != nil {
				resp.Usage = chunk.Usage.toUsage()
			}

			if chunk.Stats != nil {
				resp.Stats = chunk.Stats
			}

			if len(chunk.Choices) == 0 {
				continue
			}

			choice := chunk.Choices[0]

			if thought := choice.Delta.thinking(); thought != "" {
				reasoning.WriteString(thought)

				if !yield(ReasoningDelta{Text: thought}) {
					return
				}
			}

			if choice.Delta.Content != "" {
				text.WriteString(choice.Delta.Content)

				if !yield(TextDelta{Text: choice.Delta.Content}) {
					return
				}
			}

			for _, piece := range choice.Delta.ToolCalls {
				call := byIdx[piece.Index]

				// Some LLM backends send every call at index 0, told apart
				// only by id.
				if call == nil || (piece.ID != "" && call.id != "" && piece.ID != call.id) {
					call = &pendingCall{}
					calls = append(calls, call)
					byIdx[piece.Index] = call
				}

				call.id = cmp.Or(call.id, piece.ID)
				call.name = cmp.Or(call.name, piece.Function.Name)
				call.args.WriteString(piece.Function.Arguments)
			}

			if choice.FinishReason != "" {
				resp.FinishReason = choice.FinishReason
			}
		}

		if resp.FinishReason == "" {
			yield(Failed{Err: ErrIncompleteStream})
			return
		}

		wire := make([]replyToolCall, len(calls))
		for i, c := range calls {
			wire[i] = replyToolCall{ID: c.id, Function: wireFunction{Name: c.name, Arguments: c.args.String()}}
		}

		resp.Message = Message{
			Role:      RoleAssistant,
			Content:   text.String(),
			ToolCalls: toolCalls(wire),
		}
		resp.Reasoning = reasoning.String()

		yield(Done{Response: &resp})
	}
}
