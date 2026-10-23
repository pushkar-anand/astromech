package llm

import (
	"context"
	"fmt"
	"iter"
	"net/http"
	"strings"

	"github.com/pushkar-anand/build-with-go/http/client"
)

const streamEndKeyword = "[DONE]"

type (
	StreamRequest struct {
		CompletionRequest
		Stream  bool `json:"stream"`
		Options struct {
			IncludeUsage bool `json:"include_usage"`
		} `json:"stream_options"`
	}

	streamChunk struct {
		ID      string `json:"id"`
		Model   string `json:"model"`
		Choices []struct {
			Delta        ResponseMessage `json:"delta"`
			FinishReason string          `json:"finish_reason"`
		} `json:"choices"`
		Usage *Usage `json:"usage,omitzero"`
		Stats *Stats `json:"timings,omitzero"`
	}
)

type (
	Event interface{ event() }

	TextDelta      struct{ Text string }
	ReasoningDelta struct{ Text string }
	Done           struct{ Response *CompletionResponse } // the whole reply, assembled
	Failed         struct{ Err error }
)

func (TextDelta) event()      {}
func (ReasoningDelta) event() {}
func (Done) event()           {}
func (Failed) event()         {}

func (c *Client) Stream(ctx context.Context, req CompletionRequest) iter.Seq[Event] {
	return func(yield func(Event) bool) {
		sreq := StreamRequest{
			CompletionRequest: req,
			Stream:            true,
			Options: struct {
				IncludeUsage bool `json:"include_usage"`
			}{IncludeUsage: true},
		}

		stream := c.httpClient.Stream[streamChunk](ctx, http.MethodPost, chatCompletionPath, sreq, client.WithStreamEnd(streamEndKeyword))

		var (
			text, reasoning strings.Builder
			resp            CompletionResponse
			finish          string
		)

		for event, err := range stream {
			if err != nil {
				yield(Failed{Err: fmt.Errorf("llm: stream: %w", err)})
				return
			}

			chunk := event.Data

			if len(chunk.Choices) == 0 {
				continue
			}

			choice := chunk.Choices[0]

			if choice.Delta.ReasoningContent != "" {
				reasoning.WriteString(choice.Delta.ReasoningContent)
				yield(ReasoningDelta{Text: choice.Delta.ReasoningContent})
			}

			if choice.Delta.Content != "" {
				text.WriteString(choice.Delta.Content)
				yield(TextDelta{Text: choice.Delta.Content})
			}

			if choice.FinishReason != "" {
				finish = choice.FinishReason
			}
		}

		if finish == "" {
			yield(Failed{Err: ErrIncompleteStream})
			return
		}

		resp = CompletionResponse{
			ID:    "",
			Model: "",
			Choices: []Choice{
				{
					FinishReason: finish,
					Index:        0,
					Message: ResponseMessage{
						Role:             RoleAssistant,
						Content:          text.String(),
						ReasoningContent: reasoning.String(),
					},
				},
			},
			Stats: nil,
			Usage: Usage{},
		}
		yield(Done{Response: &resp})
	}
}
