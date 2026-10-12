package llm

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"net/http"
	"net/url"
	"strings"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/ssestream"
)

// Compat covers the ways a server departs from OpenAI's API that cannot be
// detected from its replies. The zero value suits OpenAI and most servers
// that copy it.
type Compat struct {
	// MaxCompletionTokens sends [Request.MaxTokens] as max_completion_tokens,
	// which OpenAI's reasoning models require, in place of max_tokens.
	MaxCompletionTokens bool

	// NoStreamUsage leaves out stream_options, for servers that refuse a
	// streaming request carrying it. Their streams report no [Usage].
	NoStreamUsage bool

	// ToolResultNames sends the tool's name with each tool result, which
	// Mistral requires.
	ToolResultNames bool

	// Reasoning is the field [Request.Reasoning] is sent in.
	Reasoning ReasoningFormat
}

// ReasoningFormat is how a server takes the reasoning [Effort].
type ReasoningFormat string

// ReasoningFormat values, named by the field each one writes.
const (
	// ReasoningEffortField writes reasoning_effort, as OpenAI and most
	// gateways read it. Off is sent as "none".
	ReasoningEffortField ReasoningFormat = ""

	// ReasoningObject writes a reasoning object holding the effort, as
	// OpenRouter reads it. Off is sent as enabled: false.
	ReasoningObject ReasoningFormat = "object"

	// ReasoningThink writes Ollama's think flag. It is on or off, so every
	// level other than off turns it on.
	ReasoningThink ReasoningFormat = "think"

	// ReasoningTemplate writes chat_template_kwargs.enable_thinking, which
	// vLLM and SGLang pass to models such as Qwen3. It is on or off.
	ReasoningTemplate ReasoningFormat = "template"
)

// Client sends requests to one server. A Client is safe for concurrent use
// by multiple goroutines.
//
// Requests go through the official OpenAI Go SDK. Its services are built
// directly, which skips the OPENAI_* environment variables openai.NewClient
// reads into every client: a key, organization or custom header meant for
// OpenAI or a gateway never reaches the server the client talks to.
type Client struct {
	chat   openai.ChatCompletionService
	compat Compat
}

// Option configures a [Client].
type Option func(*config)

type config struct {
	apiKey     string
	httpClient *http.Client
	header     http.Header
	maxRetries int
	compat     Compat
}

// WithAPIKey sends key as a bearer token on every request. Without it no
// Authorization header is sent, which is what local servers expect.
func WithAPIKey(key string) Option {
	return func(c *config) { c.apiKey = key }
}

// WithHTTPClient sends requests through hc. The default client has no
// timeout, because a local model can take minutes to answer; bound a call
// with its context instead.
func WithHTTPClient(hc *http.Client) Option {
	return func(c *config) { c.httpClient = hc }
}

// WithHeader adds a header to every request, such as the HTTP-Referer
// OpenRouter uses to attribute traffic.
func WithHeader(key, value string) Option {
	return func(c *config) { c.header.Add(key, value) }
}

// WithMaxRetries sets how many times a failed request is sent again. The
// default is 2. A request is retried after a connection failure, a 408, 409
// or 429, or a 5xx, waiting as long as the server's Retry-After asks, up to
// a minute, or a short exponential backoff. Only the request is retried:
// once any of a reply has reached the caller, a failure ends the stream.
func WithMaxRetries(n int) Option {
	return func(c *config) { c.maxRetries = n }
}

// WithCompat sets how requests are shaped for a server that departs from
// OpenAI's API.
func WithCompat(compat Compat) Option {
	return func(c *config) { c.compat = compat }
}

// defaultMaxRetries matches the SDK's own default.
const defaultMaxRetries = 2

// New returns a client for the server at baseURL, the address the server's
// API paths hang off, such as "https://llm.example.com/v1". It returns an
// error when baseURL is not an http or https URL.
func New(baseURL string, opts ...Option) (*Client, error) {
	u, err := url.Parse(baseURL)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
		return nil, fmt.Errorf("llm: base URL %q needs to be an http or https address, such as https://llm.example.com/v1", baseURL)
	}

	cfg := config{header: make(http.Header), maxRetries: defaultMaxRetries}
	for _, opt := range opts {
		opt(&cfg)
	}

	// An empty key sends no Authorization header.
	sdkOpts := []option.RequestOption{
		option.WithBaseURL(strings.TrimRight(baseURL, "/") + "/"),
		option.WithAPIKey(cfg.apiKey),
		option.WithHTTPClient(cmp.Or(cfg.httpClient, &http.Client{})),
		option.WithMaxRetries(cfg.maxRetries),
		option.WithMiddleware(scrub),
	}

	for name, values := range cfg.header {
		for _, v := range values {
			sdkOpts = append(sdkOpts, option.WithHeaderAdd(name, v))
		}
	}

	return &Client{
		chat:   openai.NewChatCompletionService(sdkOpts...),
		compat: cfg.compat,
	}, nil
}

// Complete sends req and waits for the whole reply.
func (c *Client) Complete(ctx context.Context, req Request) (*Response, error) {
	p, err := c.params(req)
	if err != nil {
		return nil, err
	}

	var att attempts

	completion, err := c.chat.New(ctx, p, option.WithMiddleware(att.track))
	if err != nil {
		return nil, att.explain(ctx, err)
	}

	return wholeReply(completion)
}

// Stream sends req and yields the reply as it arrives. Stopping the loop
// early cancels the request.
func (c *Client) Stream(ctx context.Context, req Request) iter.Seq[Event] {
	return func(yield func(Event) bool) {
		p, err := c.params(req)
		if err != nil {
			yield(Failed{Err: err})

			return
		}

		if !c.compat.NoStreamUsage {
			p.StreamOptions.IncludeUsage = openai.Bool(true)
		}

		ctx, cancel := context.WithCancel(ctx)
		defer cancel()

		// The SDK calls the middleware on this goroutine, so a Retrying
		// event can be yielded from inside it.
		stopped := false
		emit := func(e Event) bool {
			if stopped {
				return false
			}

			if !yield(e) {
				stopped = true
				cancel()
			}

			return !stopped
		}

		att := attempts{onRetry: func(r Retrying) { emit(r) }, stream: true}

		stream := c.chat.NewStreaming(ctx, p, option.WithMiddleware(att.track))
		defer stream.Close()

		var a assembler
		for stream.Next() {
			if !a.add(stream.Current(), emit) {
				return
			}
		}

		if stopped {
			return
		}

		if err := stream.Err(); err != nil {
			yield(Failed{Err: att.explainStream(ctx, err), Partial: a.partial(ctx)})

			return
		}

		// Some servers answer a streaming request with the whole reply.
		if att.whole != nil {
			resp, err := decodeWhole(att.whole)
			if err != nil {
				yield(Failed{Err: err})

				return
			}

			if replay(resp, emit) {
				yield(Done{Response: resp})
			}

			return
		}

		if !a.finished {
			yield(Failed{Err: ErrIncompleteStream, Partial: a.partial(ctx)})

			return
		}

		resp, ok := a.finish(emit)
		if !ok {
			return
		}

		yield(Done{Response: resp})
	}
}

// explainStream is explain for an error a stream ended with, which can also
// be a failure the server reported inside the stream.
func (a *attempts) explainStream(ctx context.Context, err error) error {
	if se, ok := errors.AsType[*ssestream.StreamError](err); ok {
		return errorInBody(se.Event.Data)
	}

	return a.explain(ctx, err)
}

// decodeWhole reads the JSON reply a server sent to a streaming request.
func decodeWhole(body []byte) (*Response, error) {
	var completion openai.ChatCompletion
	if err := json.Unmarshal(body, &completion); err != nil {
		return nil, fmt.Errorf("llm: the server answered a streaming request with JSON that is not a chat completion: %w", err)
	}

	return wholeReply(&completion)
}
