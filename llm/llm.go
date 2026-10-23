package llm

import (
	"fmt"
	"os"

	"github.com/pushkar-anand/build-with-go/http/client"
	"github.com/pushkar-anand/build-with-go/http/transport"
	"github.com/pushkar-anand/build-with-go/secret"
)

type Client struct {
	httpClient *client.Client
}

func NewClient(
	baseURL string,
	apiKey secret.Secret[string],
) (*Client, error) {
	isDebug := os.Getenv("DEBUG_ASTROMECH") == "true"

	opts := []client.Option{
		client.WithBaseURL(baseURL),
		// A model can take minutes to answer; callers bound a call
		// with its context instead.
		client.WithTimeout(0),
	}

	// Only pass the API key if it's not empty
	if apiKey.Reveal() != "" {
		opts = append(opts, client.WithTokenSource(transport.StaticToken(apiKey)))
	}

	if isDebug {
		opts = append(opts, client.WithDebug(os.Stdout))
	}

	c, err := client.New(opts...)
	if err != nil {
		return nil, fmt.Errorf("llm: failed to create client: %w", err)
	}

	return &Client{
		httpClient: c,
	}, nil
}
