// Package llmtest is a fake Chat Completions server for testing code that
// uses package llm. Queue the replies a test expects, point an [llm.Client]
// at [Server.URL], and read back what the client sent.
//
// Streamed replies are cut into small chunks, and tool call arguments into
// fragments, so a test exercises the same reassembly a real server's
// stream needs.
package llmtest

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"unicode/utf8"

	"github.com/pushkar-anand/astromech/llm"
)

// Reply is one scripted answer.
type Reply struct {
	// Status, when not zero or 200, makes the reply an error response with
	// Body as its body.
	Status int
	Header http.Header
	Body   string

	Text      string
	Reasoning string
	ToolCalls []llm.ToolCall

	// FinishReason defaults to "tool_calls" for a reply with tool calls and
	// "stop" otherwise.
	FinishReason string

	Usage llm.Usage

	// ChunkSize is how many bytes of text or arguments each streamed chunk
	// carries. Zero uses 3.
	ChunkSize int

	// Raw, when not empty, is sent as the body verbatim, for a server quirk
	// the other fields cannot describe. A streamed reply's Raw is a stream,
	// such as one [SSE] renders; a whole reply's is JSON.
	Raw string

	// Whole answers a streaming request with the whole reply as JSON, as
	// some servers do.
	Whole bool
}

// Text returns a reply that answers with text.
func Text(text string) Reply {
	return Reply{Text: text}
}

// ToolCalls returns a reply that calls tools.
func ToolCalls(calls ...llm.ToolCall) Reply {
	return Reply{ToolCalls: calls}
}

// Error returns a reply that refuses the request with status and an OpenAI
// error object holding message.
func Error(status int, message string) Reply {
	body, _ := json.Marshal(map[string]any{"error": map[string]string{"message": message}})

	return Reply{Status: status, Body: string(body)}
}

// Request is one request the server received.
type Request struct {
	Header http.Header

	// Body is the request body as sent.
	Body json.RawMessage
}

// Decode unmarshals the request body into v.
func (r Request) Decode(v any) error {
	return json.Unmarshal(r.Body, v)
}

// Field returns the request body's top-level field name, decoded, or nil
// when the body has none.
func (r Request) Field(name string) any {
	var obj map[string]any

	_ = json.Unmarshal(r.Body, &obj)

	return obj[name]
}

// Server is a fake server for one test. It is closed when the test ends.
type Server struct {
	t   testing.TB
	srv *httptest.Server

	mu       sync.Mutex
	replies  []Reply
	requests []Request
}

// NewServer starts a server. Every request it receives without a queued
// reply fails the test.
func NewServer(t testing.TB) *Server {
	t.Helper()

	s := &Server{t: t}
	s.srv = httptest.NewServer(http.HandlerFunc(s.serve))
	t.Cleanup(s.srv.Close)

	return s
}

// URL is the base URL to give [llm.New].
func (s *Server) URL() string {
	return s.srv.URL + "/v1"
}

// Enqueue adds replies, answered in order, one per request.
func (s *Server) Enqueue(replies ...Reply) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.replies = append(s.replies, replies...)
}

// Requests returns every request received so far.
func (s *Server) Requests() []Request {
	s.mu.Lock()
	defer s.mu.Unlock()

	return append([]Request(nil), s.requests...)
}

func (s *Server) serve(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)

	s.mu.Lock()
	s.requests = append(s.requests, Request{Header: r.Header.Clone(), Body: body})

	var (
		reply Reply
		ok    bool
	)

	if len(s.replies) > 0 {
		reply, s.replies, ok = s.replies[0], s.replies[1:], true
	}
	s.mu.Unlock()

	if !ok {
		s.t.Errorf("llmtest: %s %s arrived with no reply queued", r.Method, r.URL.Path)
		http.Error(w, `{"error":{"message":"llmtest: no reply queued"}}`, http.StatusInternalServerError)

		return
	}

	if r.URL.Path != "/v1/chat/completions" {
		s.t.Errorf("llmtest: %s %s is not an endpoint this server answers", r.Method, r.URL.Path)
		http.NotFound(w, r)

		return
	}

	for k, vs := range reply.Header {
		w.Header()[k] = vs
	}

	if reply.Status != 0 && reply.Status != http.StatusOK {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(reply.Status)
		_, _ = io.WriteString(w, reply.Body)

		return
	}

	var req struct {
		Stream        bool `json:"stream"`
		StreamOptions struct {
			IncludeUsage bool `json:"include_usage"`
		} `json:"stream_options"`
	}

	_ = json.Unmarshal(body, &req)

	if req.Stream && !reply.Whole {
		writeStream(w, reply, req.StreamOptions.IncludeUsage)

		return
	}

	writeCompletion(w, reply)
}

func (r Reply) finishReason() string {
	if r.FinishReason != "" {
		return r.FinishReason
	}

	if len(r.ToolCalls) > 0 {
		return "tool_calls"
	}

	return "stop"
}

func (r Reply) chunkSize() int {
	if r.ChunkSize > 0 {
		return r.ChunkSize
	}

	return 3
}

func usageJSON(u llm.Usage) map[string]any {
	return map[string]any{
		"prompt_tokens":             u.InputTokens,
		"completion_tokens":         u.OutputTokens,
		"total_tokens":              u.InputTokens + u.OutputTokens,
		"prompt_tokens_details":     map[string]any{"cached_tokens": u.CachedTokens},
		"completion_tokens_details": map[string]any{"reasoning_tokens": u.ReasoningTokens},
	}
}

func writeCompletion(w http.ResponseWriter, r Reply) {
	w.Header().Set("Content-Type", "application/json")

	if r.Raw != "" {
		_, _ = io.WriteString(w, r.Raw)

		return
	}

	msg := map[string]any{"role": "assistant", "content": r.Text}
	if r.Reasoning != "" {
		msg["reasoning_content"] = r.Reasoning
	}

	if len(r.ToolCalls) > 0 {
		calls := make([]map[string]any, 0, len(r.ToolCalls))
		for _, c := range r.ToolCalls {
			calls = append(calls, map[string]any{
				"id":       c.ID,
				"type":     "function",
				"function": map[string]any{"name": c.Name, "arguments": c.Arguments},
			})
		}

		msg["tool_calls"] = calls
	}

	_ = json.NewEncoder(w).Encode(map[string]any{
		"id":      "chatcmpl-test",
		"object":  "chat.completion",
		"created": 0,
		"model":   "test-model",
		"choices": []map[string]any{{"index": 0, "message": msg, "finish_reason": r.finishReason()}},
		"usage":   usageJSON(r.Usage),
	})
}

func writeStream(w http.ResponseWriter, r Reply, includeUsage bool) {
	w.Header().Set("Content-Type", "text/event-stream")

	if r.Raw != "" {
		_, _ = io.WriteString(w, r.Raw)

		return
	}

	send := func(delta map[string]any, finish any) {
		writeEvent(w, map[string]any{
			"id":      "chatcmpl-test",
			"object":  "chat.completion.chunk",
			"created": 0,
			"model":   "test-model",
			"choices": []map[string]any{{"index": 0, "delta": delta, "finish_reason": finish}},
		})
	}

	send(map[string]any{"role": "assistant", "content": ""}, nil)

	for _, piece := range split(r.Reasoning, r.chunkSize()) {
		send(map[string]any{"reasoning_content": piece}, nil)
	}

	for _, piece := range split(r.Text, r.chunkSize()) {
		send(map[string]any{"content": piece}, nil)
	}

	for i, c := range r.ToolCalls {
		send(map[string]any{"tool_calls": []map[string]any{{
			"index":    i,
			"id":       c.ID,
			"type":     "function",
			"function": map[string]any{"name": c.Name, "arguments": ""},
		}}}, nil)

		for _, piece := range split(c.Arguments, r.chunkSize()) {
			send(map[string]any{"tool_calls": []map[string]any{{
				"index":    i,
				"function": map[string]any{"arguments": piece},
			}}}, nil)
		}
	}

	send(map[string]any{}, r.finishReason())

	if includeUsage {
		writeEvent(w, map[string]any{
			"id":      "chatcmpl-test",
			"object":  "chat.completion.chunk",
			"created": 0,
			"model":   "test-model",
			"choices": []any{},
			"usage":   usageJSON(r.Usage),
		})
	}

	_, _ = io.WriteString(w, "data: [DONE]\n\n")
}

func writeEvent(w io.Writer, v any) {
	b, _ := json.Marshal(v)
	_, _ = fmt.Fprintf(w, "data: %s\n\n", b)
}

// split cuts s into pieces of about n bytes, on rune boundaries so each
// piece stays valid UTF-8.
func split(s string, n int) []string {
	var out []string

	for len(s) > n {
		cut := n
		for cut < len(s) && !utf8.RuneStart(s[cut]) {
			cut++
		}

		out = append(out, s[:cut])
		s = s[cut:]
	}

	if s != "" {
		out = append(out, s)
	}

	return out
}

// SSE renders events as a server-sent event stream body, for [Reply.Raw].
// Each event is marshalled to JSON unless it is already a string.
func SSE(events ...any) string {
	var b []byte

	for _, e := range events {
		data, ok := e.(string)
		if !ok {
			raw, _ := json.Marshal(e)
			data = string(raw)
		}

		b = append(b, "data: "...)
		b = append(b, data...)
		b = append(b, "\n\n"...)
	}

	return string(b)
}
