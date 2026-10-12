// Package llm is a client for servers that speak the OpenAI Chat Completions
// API: hosted providers, gateways such as OpenRouter and LiteLLM, and local
// servers such as Ollama, vLLM and LM Studio.
//
// A [Client] sends a [Request] and returns the reply whole, with
// [Client.Complete], or as a stream of [Event] values, with [Client.Stream].
// Conversations are plain [Message] values with JSON tags, so a caller can
// store them and send them again later, to the same model or another.
//
// The client smooths over the ways these servers differ from OpenAI and from
// each other: tool calls without IDs, tool arguments sent as objects,
// reasoning under either of two field names, and a JSON reply to a streaming
// request. [Compat] covers the differences that cannot be detected from the
// reply.
//
// Failed requests are retried before any output reaches the caller. See
// [RetryPolicy].
package llm
