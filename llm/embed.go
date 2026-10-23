package llm

import (
	"context"
	"errors"
	"fmt"
)

type (
	// EmbedRequest asks for the embedding of each input.
	EmbedRequest struct {
		Model string
		Input []string

		// Dimensions shortens each vector, on models that support it. Zero
		// keeps the model's own length.
		Dimensions int
	}

	// Embeddings are the vectors for an EmbedRequest.
	Embeddings struct {
		// Vectors holds one vector per input, in the order of the inputs.
		Vectors [][]float32

		// Model is as the LLM backend reported it.
		Model string

		// Usage is zero when the LLM backend does not report it. Only
		// InputTokens and TotalTokens are set.
		Usage Usage
	}
)

// Wire format of embeddings.
type (
	embedRequest struct {
		Model          string   `json:"model"`
		Input          []string `json:"input"`
		Dimensions     int      `json:"dimensions,omitzero"`
		EncodingFormat string   `json:"encoding_format"`
	}

	embedReply struct {
		Model string `json:"model"`
		Data  []struct {
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
		Usage usage `json:"usage"`
	}
)

// Embed returns the embedding of each input. Vectors from different models
// cannot be compared.
func (c *Client) Embed(ctx context.Context, req EmbedRequest) (*Embeddings, error) {
	if len(req.Input) == 0 {
		return nil, errors.New("llm: embed: no input")
	}

	reply, err := c.httpClient.Post[embedReply](ctx, embeddingsPath, embedRequest{
		Model:      req.Model,
		Input:      req.Input,
		Dimensions: req.Dimensions,
		// Some proxies default to base64, which needs decoding first.
		EncodingFormat: "float",
	})
	if err != nil {
		return nil, fmt.Errorf("llm: embed: %w", explain(err))
	}

	if len(reply.Data) != len(req.Input) {
		return nil, fmt.Errorf("llm: embed: the LLM backend sent %d vectors for %d inputs", len(reply.Data), len(req.Input))
	}

	// Each vector says which input it belongs to; LLM backends are free to
	// answer out of order.
	vectors := make([][]float32, len(req.Input))

	for _, d := range reply.Data {
		if d.Index < 0 || d.Index >= len(vectors) || vectors[d.Index] != nil {
			return nil, fmt.Errorf("llm: embed: the LLM backend sent a vector for input %d twice or out of range", d.Index)
		}

		vectors[d.Index] = d.Embedding
	}

	return &Embeddings{
		Vectors: vectors,
		Model:   reply.Model,
		Usage:   reply.Usage.toUsage(),
	}, nil
}
