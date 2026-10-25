package agent

import (
	"context"

	"github.com/pushkar-anand/astromech/llm"
)

type run struct {
	ctx   context.Context
	agent *Agent

	yield func(event Event) bool

	tools    []llm.Tool
	toolsMap map[string]Tool

	input   *llm.Message
	history []llm.Message
	current []llm.Message

	usage llm.Usage

	roundLimit int
}

func (r *run) loop() {
	if !r.add(r.input) {
		return
	}

	for i := 0; r.roundLimit < 0 || i < r.roundLimit; i++ {
		var send []llm.Message

		if r.agent.System != "" {
			send = append(send, r.system())
		}

		send = append(send, r.history...)
		send = append(send, r.current...)

		request := llm.Request{
			Model:    r.agent.ModelID,
			Messages: send,
			Tools:    r.tools,
		}

		var response *llm.Response

		for event := range r.agent.Model.Stream(r.ctx, request) {
			switch e := event.(type) {
			case llm.TextDelta:
				if !r.yield(TextDelta{Text: e.Text}) {
					return
				}
			case llm.ReasoningDelta:
				if !r.yield(ReasoningDelta{Text: e.Text}) {
					return
				}
			case llm.Failed:
				r.yield(Failed{Err: e.Err, Messages: r.current, Usage: r.usage})
				return
			case llm.Done:
				r.usage = r.usage.Add(e.Response.Usage)
				response = e.Response
			}
		}

		// llm.Client always ends in Done or Failed; other models may not.
		if response == nil {
			r.yield(Failed{Err: llm.ErrIncompleteStream, Messages: r.current, Usage: r.usage})
			return
		}

		if !r.add(&response.Message) {
			return
		}

		if len(response.Message.ToolCalls) == 0 {
			r.done(StopAnswered)
			return
		}

		if ok := r.callAllTools(response.Message.ToolCalls); !ok {
			return
		}
	}

	r.done(StopMaxRounds)
}

func (r *run) system() llm.Message {
	return llm.Message{
		Role:    llm.RoleSystem,
		Content: r.agent.System,
	}
}

func (r *run) add(msg *llm.Message) bool {
	if msg == nil {
		return true
	}

	r.current = append(r.current, *msg)

	return r.yield(MessageAdded{Message: *msg})
}

func (r *run) done(reason Stop) {
	r.yield(Done{
		Stop:     reason,
		Messages: r.current,
		Usage:    r.usage,
	})
}

func (r *run) callAllTools(toolCalls []llm.ToolCall) (ok bool) {
	for _, toolCall := range toolCalls {
		if !r.yield(ToolStarted{Call: toolCall}) {
			return false
		}

		result := callTool(r.ctx, r.toolsMap, toolCall)

		if !r.yield(ToolFinished{Call: toolCall, Result: result}) {
			return false
		}

		msg := llm.Message{
			Role:       llm.RoleTool,
			Content:    result.Content,
			ToolCallID: toolCall.ID,
		}

		if !r.add(&msg) {
			return false
		}
	}

	return true
}
