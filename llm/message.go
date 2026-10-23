package llm

// Message is one turn of a conversation.
type Message struct {
	Role    Role
	Content string

	// ToolCalls is set on an assistant message that calls tools.
	ToolCalls []ToolCall

	// ToolCallID is set on a RoleTool message: the call it answers.
	ToolCallID string
}
