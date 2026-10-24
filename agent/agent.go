package agent

import (
	"cmp"
	"context"
	"encoding/json/jsontext"
	"fmt"
	"iter"
	"maps"
	"slices"
	"strings"

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
		tools, toolsMap, err := a.tools()
		if err != nil {
			yield(Failed{Err: err})
			return
		}

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

		limit := cmp.Or(a.MaxRounds, defaultMaxRounds)

		for i := 0; limit < 0 || i < limit; i++ {
			var send []llm.Message

			if a.System != "" {
				send = append(send, llm.Message{Role: llm.RoleSystem, Content: a.System})
			}

			// Read only: the caller owns the history.
			send = append(send, turn.History...)
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

		yield(Done{
			Stop:     StopMaxRounds,
			Messages: current,
			Usage:    usage,
		})
	}
}

// tools is the definitions sent to the model, and the tools by name. Two
// tools with one name are an error: the model could not tell them apart.
func (a *Agent) tools() ([]llm.Tool, map[string]Tool, error) {
	if len(a.Tools) == 0 {
		return nil, nil, nil
	}

	tools := make([]llm.Tool, len(a.Tools))
	toolsMap := make(map[string]Tool, len(a.Tools))

	for i, tool := range a.Tools {
		def := tool.Def()
		if _, ok := toolsMap[def.Name]; ok {
			return nil, nil, fmt.Errorf("agent: two tools are named %q", def.Name)
		}

		tools[i] = def
		toolsMap[def.Name] = tool
	}

	return tools, toolsMap, nil
}

// callTool runs the tool toolCall names. Whatever goes wrong becomes an
// error result, naming the tool and showing what the model sent, so the
// model can correct its next call.
func callTool(ctx context.Context, toolsMap map[string]Tool, toolCall llm.ToolCall) Result {
	t, ok := toolsMap[toolCall.Name]
	if !ok {
		names := cmp.Or(strings.Join(slices.Sorted(maps.Keys(toolsMap)), ", "), "none")

		return Result{
			Content: fmt.Sprintf("there is no tool named %q; the tools are: %s", toolCall.Name, names),
			IsError: true,
		}
	}

	args := jsontext.Value(cmp.Or(toolCall.Arguments, "{}"))
	if !args.IsValid() {
		return Result{
			Content: fmt.Sprintf("the arguments for %s are not valid JSON: %s", toolCall.Name, toolCall.Arguments),
			IsError: true,
		}
	}

	result, err := t.Call(ctx, args)
	if err != nil {
		return Result{
			Content: fmt.Sprintf("%s failed: %v", toolCall.Name, err),
			IsError: true,
		}
	}

	return result
}
