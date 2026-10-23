package llm

import (
	"net/http"

	"github.com/pushkar-anand/build-with-go/http/client"
	"github.com/pushkar-anand/build-with-go/secret"
)

// NewWithTransport is New sending through rt, such as the in-memory network
// of an httptest server.
func NewWithTransport(baseURL string, apiKey secret.Secret[string], rt http.RoundTripper) (*Client, error) {
	return newClient(baseURL, apiKey, client.WithTransport(rt))
}
