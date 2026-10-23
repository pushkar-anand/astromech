package llm

// Role says who wrote a [Message].
type Role string

// Role values a conversation can carry.
const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"

	// RoleDeveloper is OpenAI-only; most other LLM backends reject it.
	RoleDeveloper Role = "developer"
)

// Message is one turn of a conversation.
type Message struct {
	Role    Role
	Content string

	// ToolCalls is set on an assistant message that calls tools.
	ToolCalls []ToolCall

	// ToolCallID is set on a RoleTool message: the call it answers.
	ToolCallID string
}
