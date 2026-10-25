package agent

import (
	"cmp"
	"context"
	"encoding/json/jsontext"
	"fmt"
	"iter"
	"maps"
	"os"
	"runtime/debug"
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

		r := run{
			ctx:        ctx,
			agent:      a,
			yield:      yield,
			tools:      tools,
			toolsMap:   toolsMap,
			input:      turn.Input,
			history:    slices.Clone(turn.History), // stored for the turn; the caller may edit theirs meanwhile
			roundLimit: cmp.Or(a.MaxRounds, defaultMaxRounds),
		}

		r.loop()
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

// callTool runs the tool toolCall names. Whatever goes wrong, a panic
// included, becomes an error result, naming the tool and showing what the
// model sent, so the model can correct its next call. With
// DEBUG_ASTROMECH=true, a panic's stack is printed to stdout, since the
// model never sees it.
func callTool(ctx context.Context, toolsMap map[string]Tool, toolCall llm.ToolCall) (result Result) {
	defer func() {
		if r := recover(); r != nil {
			if os.Getenv("DEBUG_ASTROMECH") == "true" {
				fmt.Fprintf(os.Stdout, "agent: panic in tool %q: %v\n%s", toolCall.Name, r, debug.Stack())
			}

			result = Result{
				Content: fmt.Sprintf("panic in tool %q: %v", toolCall.Name, r),
				IsError: true,
			}
		}
	}()

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
