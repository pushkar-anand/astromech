// Package llm is a client for LLM backends that speak the OpenAI Chat
// Completions API.
//
// A [Client] sends a [Request] and returns the reply whole, with
// [Client.Complete], or as a stream of [Event] values, with [Client.Stream].
// [Client.Models] lists the models the backend offers.
//
// Setting DEBUG_ASTROMECH=true dumps every request and response, bodies
// included, to stdout.
package llm
