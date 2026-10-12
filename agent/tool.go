package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"

	"github.com/google/jsonschema-go/jsonschema"

	"github.com/pushkar-anand/astromech/llm"
)

// Tool is something the model can call.
type Tool interface {
	// Def describes the tool to the model.
	Def() llm.ToolDef

	// ReadOnly reports that the tool changes nothing, so it runs without
	// approval in every [Mode]. A tool that sends, writes, deletes or
	// switches anything returns false.
	ReadOnly() bool

	// Call runs the tool with the model's arguments, which are JSON. An
	// error is shown to the model as the call's result, so it can correct
	// its arguments or tell the user.
	Call(ctx context.Context, args json.RawMessage) (Result, error)
}

// Result is what a tool call returns to the model.
type Result struct {
	// Content is what the model reads.
	Content string `json:"content"`

	// IsError marks a call that failed, for showing the call. The model
	// reads Content either way.
	IsError bool `json:"is_error,omitzero"`
}

// FuncOption configures a tool built by [Func].
type FuncOption func(*funcTool)

// ReadOnly marks a [Func] tool as changing nothing, so it runs without
// approval.
func ReadOnly() FuncOption {
	return func(t *funcTool) { t.readOnly = true }
}

// Strict asks the model to keep to the tool's schema exactly, for servers
// that support strict tool schemas.
func Strict() FuncOption {
	return func(t *funcTool) { t.def.Strict = true }
}

type funcTool struct {
	def      llm.ToolDef
	readOnly bool
	schema   *jsonschema.Resolved
	call     func(ctx context.Context, args json.RawMessage) (Result, error)
}

// Func returns a tool that calls fn with the model's arguments decoded into
// an In, which is usually a struct. The tool's parameters are inferred from
// In: the json tag names each field, a field without omitempty or omitzero
// is required, and a jsonschema tag describes the field to the model.
//
// Arguments that do not match the parameters never reach fn; the model is
// told what is wrong. fn's result is sent to the model as JSON, or as it is
// when it is a string.
//
// A Func tool changes things unless it is marked [ReadOnly]. Func panics
// when In cannot be described as a JSON schema, which is a mistake in the
// program and shows the first time it runs.
func Func[In, Out any](name, description string, fn func(context.Context, In) (Out, error), opts ...FuncOption) Tool {
	schema, err := jsonschema.For[In](nil)
	if err != nil {
		panic(fmt.Sprintf("agent: the parameters of tool %s cannot be described: %v", name, err))
	}

	resolved, err := schema.Resolve(nil)
	if err != nil {
		panic(fmt.Sprintf("agent: the parameters of tool %s cannot be described: %v", name, err))
	}

	params, err := json.Marshal(schema)
	if err != nil {
		panic(fmt.Sprintf("agent: the parameters of tool %s cannot be described: %v", name, err))
	}

	t := &funcTool{
		def:    llm.ToolDef{Name: name, Description: description, Parameters: params},
		schema: resolved,
	}

	t.call = func(ctx context.Context, args json.RawMessage) (Result, error) {
		var in In
		if err := json.Unmarshal(args, &in); err != nil {
			return Result{}, fmt.Errorf("the arguments do not fit the parameters: %w", err)
		}

		out, err := fn(ctx, in)
		if err != nil {
			return Result{}, err
		}

		if s, ok := any(out).(string); ok {
			return Result{Content: s}, nil
		}

		content, err := json.Marshal(out)
		if err != nil {
			return Result{}, fmt.Errorf("the result cannot be sent as JSON: %w", err)
		}

		return Result{Content: string(content)}, nil
	}

	for _, opt := range opts {
		opt(t)
	}

	return t
}

func (t *funcTool) Def() llm.ToolDef { return t.def }

func (t *funcTool) ReadOnly() bool { return t.readOnly }

func (t *funcTool) Call(ctx context.Context, args json.RawMessage) (Result, error) {
	if len(bytes.TrimSpace(args)) == 0 {
		args = json.RawMessage("{}")
	}

	var v any
	if err := json.Unmarshal(args, &v); err != nil {
		return Result{}, fmt.Errorf("the arguments are not JSON: %w", err)
	}

	if err := t.schema.Validate(v); err != nil {
		return Result{}, fmt.Errorf("the arguments do not match the parameters: %w", err)
	}

	return t.call(ctx, args)
}
