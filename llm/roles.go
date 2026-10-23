package llm

// Role says who wrote a [Message].
type Role string

// Role values a conversation can carry.
const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"

	// RoleDeveloper is OpenAI-only; most other LLM backends reject it.
	RoleDeveloper Role = "developer"
)
