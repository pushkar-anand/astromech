// Package agent runs the loop between a user, a model and tools: it asks
// the model, runs the tools the model calls, hands back their results, and
// repeats until the model answers.
//
// An [Agent] holds what stays the same across conversations: the client,
// the system prompt and the tools. Each call to [Agent.Run] is one [Turn]
// of one conversation, and streams [Event] values for a program to render.
// The conversation itself belongs to the caller, who stores the messages a
// turn adds and passes them back as the next turn's history.
//
// Tools that change things wait for approval, depending on the turn's
// [Mode]. A turn that reaches such a call ends with [StopAwaitingApproval],
// and the next turn carries the person's [Verdict]. Nothing waits in memory
// between the two, so an approval can arrive on another request, or after
// a restart.
package agent
