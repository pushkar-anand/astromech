package agent

import (
	"context"
	"encoding/json/jsontext"
	"iter"

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
