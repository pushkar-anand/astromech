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

	// Media is sent after Content. Only RoleUser messages may carry it.
	Media []Media
}

// Media is an image or audio clip sent with a [Message]. The model must
// accept that kind of input; on llama.cpp that means a server started with
// --mmproj.
type Media struct {
	// MIMEType is such as "image/png" or "audio/wav".
	MIMEType string

	// Data is the content itself. When it is set, URL is ignored.
	Data []byte

	// URL points at the content instead. Not every LLM backend fetches
	// URLs, and OpenAI accepts them only for images.
	URL string
}
