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
	c, err := client.New(
		client.WithBaseURL(baseURL),
		client.WithTokenSource(transport.StaticToken(apiKey)),
		client.WithDebug(os.Stdout),
	)
	if err != nil {
		return nil, fmt.Errorf("llm: failed to create client: %w", err)
	}

	return &Client{
		httpClient: c,
	}, nil
}
