package agent

import (
	"cmp"
	"context"
	"encoding/json/jsontext"
	"fmt"
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

		tools, toolsMap := a.tools()

		limit := cmp.Or(a.MaxRounds, defaultMaxRounds)

		for i := 0; limit < 0 || i < limit; i++ {
			var send []llm.Message

			if a.System != "" {
				send = append(send, *a.systemPrompt())
			}

			send = append(send, history...)
			send = append(send, current...)

			request := llm.Request{
				Model:    a.ModelID,
				Messages: send,
				Tools:    tools,
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
					yield(Failed{Err: e.Err, Messages: current, Usage: usage})
					return
				case llm.Done:
					usage = usage.Add(e.Response.Usage)
					response = e.Response
				}
			}

			// llm.Client always ends in Done or Failed; other models may not.
			if response == nil {
				yield(Failed{Err: llm.ErrIncompleteStream, Messages: current, Usage: usage})
				return
			}

			current = append(current, response.Message)
			if !yield(MessageAdded{Message: response.Message}) {
				return
			}

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

				result := callTool(ctx, toolsMap, toolCall)

				if !yield(ToolFinished{Call: toolCall, Result: result}) {
					return
				}

				msg := llm.Message{
					Role:       llm.RoleTool,
					Content:    result.Content,
					ToolCallID: toolCall.ID,
				}

				current = append(current, msg)
				if !yield(MessageAdded{Message: msg}) {
					return
				}
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

func (a *Agent) systemPrompt() *llm.Message {
	if a.System != "" {
		return &llm.Message{
			Role:    llm.RoleSystem,
			Content: a.System,
		}
	}

	return nil
}

func (a *Agent) tools() ([]llm.Tool, map[string]Tool) {
	if len(a.Tools) == 0 {
		return nil, nil
	}

	tools := make([]llm.Tool, len(a.Tools))
	toolsMap := make(map[string]Tool)

	for i, tool := range a.Tools {
		def := tool.Def()
		tools[i] = def
		toolsMap[def.Name] = tool
	}

	return tools, toolsMap
}

func callTool(ctx context.Context, toolsMap map[string]Tool, toolCall llm.ToolCall) Result {
	t, ok := toolsMap[toolCall.Name]
	if !ok {
		return Result{
			Content: fmt.Sprintf("tool not found: %s", toolCall.Name),
			IsError: true,
		}
	}

	args := jsontext.Value(cmp.Or(toolCall.Arguments, "{}"))
	if !args.IsValid() {
		return Result{
			Content: fmt.Sprintf("invalid tool arguments: %s", toolCall.Name),
			IsError: true,
		}
	}

	result, err := t.Call(ctx, args)
	if err != nil {
		return Result{
			Content: fmt.Sprintf("tool call failed: %s", err),
			IsError: true,
		}
	}

	return result
}
