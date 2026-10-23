package llm

type (
	// Tool is a tool the model may ask to call.
	Tool struct {
		Name        string
		Description string

		// Parameters is a JSON Schema for the arguments, such as a
		// map[string]any. Nil if the tool takes none.
		Parameters any
	}

	// ToolCall is the model asking for a tool to be run.
	ToolCall struct {
		// ID pairs the call with its result; send it back as
		// Message.ToolCallID.
		ID   string
		Name string

		// Arguments is the JSON the model wrote for the tool's parameters.
		// It may not be valid JSON.
		Arguments string
	}
)
