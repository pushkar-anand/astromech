package agent

import (
	"context"
	"encoding/json/jsontext"
	"iter"
	"slices"

	"github.com/pushkar-anand/astromech/llm"
)

const defaultMaxRounds = 25

type (
	Result struct {
		Content string // what the model reads
		IsError bool   // for the UI; the model reads Content either way
	}

	Agent struct {
		Model     Model
		ModelID   string
		System    string
		Tools     []Tool
		MaxRounds int // Max rounds of calls, zero means defaultRounds, negative means no limit
	}

	Turn struct {
		History []llm.Message
		Input   *llm.Message
	}
)

type (
	Model interface {
		Stream(ctx context.Context, req llm.Request) iter.Seq[llm.Event]
	}

	Tool interface {
		Def() llm.Tool  // name, description, schema, as sent to the model
		ReadOnly() bool // changes nothing; used by approvals later
		Call(ctx context.Context, args jsontext.Value) (Result, error)
	}
)

func (a *Agent) Run(ctx context.Context, turn Turn) iter.Seq[Event] {
	return func(yield func(Event) bool) {
		history := slices.Clone(turn.History)
		input := turn.Input

		var (
			current []llm.Message
			usage   llm.Usage
		)

		if input != nil {
			current = append(current, *input)
			if !yield(MessageAdded{Message: *input}) {
				return
			}
		}

		for i := 0; !a.shouldExitLoop(turn, i); i++ {
			if a.System != "" {
				current = append(current, *a.systemPrompt())
			}

			current = append(current, history...)

			request := llm.Request{
				Model:    a.ModelID,
				Messages: current,
				Tools:    a.tools(),
			}

			var response *llm.Response

			for event := range a.Model.Stream(ctx, request) {
				switch e := event.(type) {
				case llm.TextDelta:
					if !yield(TextDelta{Text: e.Text}) {
						return
					}
				case llm.ReasoningDelta:
					if !yield(ReasoningDelta{Text: e.Text}) {
						return
					}
				case llm.Failed:
					if !yield(Failed{Err: e.Err, Messages: current, Usage: usage}) {
						return
					}
				case llm.Done:
					usage = e.Response.Usage
					response = e.Response
				}
			}

			current = append(current, response.Message)
			history = append(history, response.Message)

			if len(response.Message.ToolCalls) == 0 {
				yield(Done{
					Stop:     StopAnswered,
					Messages: current,
					Usage:    usage,
				})
				return
			}

			for _, toolCall := range response.Message.ToolCalls {
				if !yield(ToolStarted{Call: toolCall}) {
					return
				}

				// TODO: run tool

				if !yield(ToolFinished{Call: toolCall}) {
					return
				}

				current = append(current, llm.Message{
					Role:       llm.RoleTool,
					Content:    "",
					ToolCallID: toolCall.ID,
				})
			}
		}

		if !yield(Done{
			Stop:     StopMaxRounds,
			Messages: current,
			Usage:    usage,
		}) {
			return
		}
	}
}

func (a *Agent) shouldExitLoop(turn Turn, iteration int) bool {
	maxRounds := a.MaxRounds

	if maxRounds == 0 {
		maxRounds = defaultMaxRounds
	} else if maxRounds > 0 {
		return iteration >= maxRounds
	}

	// TODO: define a better exit as max rounds is unlimited
	return false
}

func (a *Agent) systemPrompt() *llm.Message {
	if a.System != "" {
		return &llm.Message{
			Role:    llm.RoleSystem,
			Content: a.System,
		}
	}

	return nil
}

func (a *Agent) tools() []llm.Tool {
	if len(a.Tools) == 0 {
		return nil
	}

	tools := make([]llm.Tool, len(a.Tools))

	for i, tool := range a.Tools {
		tools[i] = tool.Def()
	}

	return tools
}

func (a *Agent) callTool() {

}
