package llm

import (
	"encoding/json"
	"fmt"
)

// Role says who wrote a [Message].
type Role string

// Role values a conversation can carry.
const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"

	// RoleTool marks the result of a tool call, sent back so the model can
	// read it.
	RoleTool Role = "tool"
)

// Message is one turn of a conversation.
//
// The JSON tags are the storage form, which stays stable across releases.
// What goes on the wire to a server is built separately.
type Message struct {
	Role    Role   `json:"role"`
	Content string `json:"content,omitzero"`

	// Images go to the model with Content, on a user message. Only models
	// that accept image input can read them; others refuse the request.
	Images []Image `json:"images,omitzero"`

	// Reasoning is the thinking a reasoning model showed before it answered.
	// It is kept for display and is not sent back to the server, which
	// DeepSeek refuses.
	Reasoning string `json:"reasoning,omitzero"`

	// ToolCalls are the tools an assistant message asks to run.
	ToolCalls []ToolCall `json:"tool_calls,omitzero"`

	// ToolCallID and ToolName say which call a RoleTool message answers.
	ToolCallID string `json:"tool_call_id,omitzero"`
	ToolName   string `json:"tool_name,omitzero"`
}

// System returns a system message holding text.
func System(text string) Message {
	return Message{Role: RoleSystem, Content: text}
}

// User returns a user message holding text.
func User(text string) Message {
	return Message{Role: RoleUser, Content: text}
}

// Assistant returns an assistant message holding text.
func Assistant(text string) Message {
	return Message{Role: RoleAssistant, Content: text}
}

// ToolResult returns the message that answers call with content.
//
// A failed call is answered the same way, with content that says what went
// wrong, so the model can correct its arguments and try again.
func ToolResult(call ToolCall, content string) Message {
	return Message{
		Role:       RoleTool,
		Content:    content,
		ToolCallID: call.ID,
		ToolName:   call.Name,
	}
}

// UserWithImages returns a user message holding text and images.
func UserWithImages(text string, images ...Image) Message {
	return Message{Role: RoleUser, Content: text, Images: images}
}

// Image is a picture sent to the model, either by URL or inline.
type Image struct {
	// URL is where the server fetches the image from. A server on a private
	// network often cannot reach the public internet, so inline Data is the
	// safer choice for local models.
	URL string `json:"url,omitzero"`

	// Data is the encoded image, such as the bytes of a PNG or JPEG file,
	// and MediaType its type, such as "image/png". They are sent as a data
	// URL when URL is empty.
	Data      []byte `json:"data,omitzero"`
	MediaType string `json:"media_type,omitzero"`

	// Detail trades resolution for tokens on servers that support it: "low",
	// "high" or "auto". Empty leaves the server's default.
	Detail string `json:"detail,omitzero"`
}

// ToolCall is one tool an assistant message asks to run.
type ToolCall struct {
	ID   string `json:"id"`
	Name string `json:"name"`

	// Arguments is the JSON object the model wrote for the tool's
	// parameters. It is kept as text because a model can write JSON that does
	// not parse, and a conversation holding that call must still be stored
	// and sent back.
	Arguments string `json:"arguments"`
}

// DecodeArguments unmarshals the call's arguments into v.
func (c ToolCall) DecodeArguments(v any) error {
	if err := json.Unmarshal([]byte(c.Arguments), v); err != nil {
		return fmt.Errorf("llm: arguments for %s are not the JSON its schema asks for: %w", c.Name, err)
	}

	return nil
}

// ToolDef describes a tool the model may call.
type ToolDef struct {
	// Name is what the model calls the tool by. Servers accept letters,
	// digits, underscores and hyphens, up to 64 characters.
	Name string

	// Description tells the model what the tool does and when to use it.
	// Models choose tools mostly from this text.
	Description string

	// Parameters is a JSON Schema object for the arguments. Nil sends a
	// schema for an empty object, which some servers require even for a tool
	// that takes nothing.
	Parameters json.RawMessage

	// Strict asks the server to hold the arguments to Parameters exactly.
	// Servers that do not support it ignore it or refuse the request.
	Strict bool
}

// ToolChoice says whether the model may, must or must not call a tool.
type ToolChoice string

// ToolChoice values. The zero value leaves the choice to the server, which
// treats it as [ToolChoiceAuto] when tools are offered.
const (
	ToolChoiceAuto     ToolChoice = "auto"
	ToolChoiceNone     ToolChoice = "none"
	ToolChoiceRequired ToolChoice = "required"
)

// Effort is how much a reasoning model thinks before it answers. More
// effort gives better answers to hard questions and costs time and tokens.
type Effort string

// Effort values, from none to the most. A level is sent as it is, and a
// server that does not offer it refuses the request; many offer only low,
// medium and high.
const (
	EffortOff     Effort = "off"
	EffortMinimal Effort = "minimal"
	EffortLow     Effort = "low"
	EffortMedium  Effort = "medium"
	EffortHigh    Effort = "high"
)

// ResponseFormat asks the model for JSON in place of free text.
type ResponseFormat struct {
	// Name labels the schema. Servers that enforce a schema require one.
	Name string

	// Schema is the JSON Schema the reply must match. Nil asks only for
	// some JSON object.
	Schema json.RawMessage
}

// Request is one call to a model.
type Request struct {
	Model    string
	Messages []Message
	Tools    []ToolDef

	ToolChoice ToolChoice

	// Temperature and TopP are pointers so a zero can be sent; nil leaves
	// the server's default.
	Temperature *float64
	TopP        *float64

	// MaxTokens caps the reply's length; zero leaves the server's default.
	// On a reasoning model the cap includes the reasoning.
	MaxTokens int

	// Reasoning sets how hard a reasoning model thinks before it answers.
	// The zero value leaves the server's default. [Compat.Reasoning] says
	// how the level is sent.
	Reasoning Effort

	Stop []string
	Seed *int

	ResponseFormat *ResponseFormat

	// Extra holds request fields this package has no name for, such as
	// Ollama's options or reasoning_effort. Each is added to the request
	// body, and replaces any field of the same name the package wrote.
	Extra map[string]any
}

// StopReason says why a reply ended.
type StopReason string

// StopReason values.
const (
	// StopEnd means the model finished its answer.
	StopEnd StopReason = "end"

	// StopLength means the reply reached MaxTokens or the model's limit.
	StopLength StopReason = "length"

	// StopToolCalls means the model is waiting for the results of the tool
	// calls it made.
	StopToolCalls StopReason = "tool_calls"

	// StopFiltered means the server withheld the rest of the reply under
	// its content policy.
	StopFiltered StopReason = "filtered"

	// StopCanceled and StopFailed mark the partial reply of a stream that
	// did not finish: its context ended, or the request failed.
	StopCanceled StopReason = "canceled"
	StopFailed   StopReason = "failed"
)

// Usage counts the tokens a request used. A server that does not report
// usage leaves it zero.
type Usage struct {
	InputTokens  int `json:"input_tokens,omitzero"`
	OutputTokens int `json:"output_tokens,omitzero"`

	// CachedTokens is the part of InputTokens served from the server's
	// prompt cache.
	CachedTokens int `json:"cached_tokens,omitzero"`

	// ReasoningTokens is the part of OutputTokens spent on reasoning.
	ReasoningTokens int `json:"reasoning_tokens,omitzero"`
}

// Response is a model's reply.
type Response struct {
	// ID and Model are as the server reported them. Model can differ from
	// the requested one when a gateway routes the request.
	ID    string
	Model string

	Message Message
	Stop    StopReason
	Usage   Usage
}
