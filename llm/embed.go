package llm

import (
	"cmp"
	"context"
	"errors"
	"fmt"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

// EmbedRequest asks for the embeddings of some texts.
type EmbedRequest struct {
	Model string
	Input []string

	// Dimensions shortens each vector, on models that support it. Zero
	// leaves the model's own length.
	Dimensions int
}

// Embeddings are the vectors for an [EmbedRequest].
type Embeddings struct {
	// Vectors holds one vector per input, in the order of the inputs.
	Vectors [][]float32

	Model string
	Usage Usage
}

// Embed returns the embedding of each input. Vectors from different models
// cannot be compared, so a program that stores them stores the model too.
func (c *Client) Embed(ctx context.Context, req EmbedRequest) (*Embeddings, error) {
	if req.Model == "" {
		return nil, errors.New("llm: the request names no model; set EmbedRequest.Model to an embedding model the server offers")
	}

	if len(req.Input) == 0 {
		return &Embeddings{Model: req.Model}, nil
	}

	p := openai.EmbeddingNewParams{
		Model: openai.EmbeddingModel(req.Model),
		Input: openai.EmbeddingNewParamsInputUnion{OfArrayOfStrings: req.Input},

		// The SDK decodes only floats; a server that defaults to base64
		// would otherwise answer with vectors it cannot read.
		EncodingFormat: openai.EmbeddingNewParamsEncodingFormatFloat,
	}

	if req.Dimensions > 0 {
		p.Dimensions = openai.Int(int64(req.Dimensions))
	}

	var att attempts

	resp, err := c.embeddings.New(ctx, p, option.WithMiddleware(att.track))
	if err != nil {
		return nil, att.explain(ctx, err)
	}

	if len(resp.Data) == 0 {
		if e := errorInReply(resp.JSON.ExtraFields, resp.RawJSON()); e != nil {
			return nil, e
		}
	}

	if len(resp.Data) != len(req.Input) {
		return nil, fmt.Errorf("llm: the server returned %d embeddings for %d inputs", len(resp.Data), len(req.Input))
	}

	out := &Embeddings{
		Vectors: make([][]float32, len(req.Input)),
		Model:   cmp.Or(resp.Model, req.Model),
		Usage:   Usage{InputTokens: int(resp.Usage.PromptTokens)},
	}

	// Each vector says which input it belongs to; servers are free to
	// answer out of order.
	for _, d := range resp.Data {
		i := int(d.Index)
		if i < 0 || i >= len(out.Vectors) || out.Vectors[i] != nil {
			return nil, fmt.Errorf("llm: the server returned an embedding for input %d, which is out of range or repeated", i)
		}

		v := make([]float32, len(d.Embedding))
		for j, x := range d.Embedding {
			v[j] = float32(x)
		}

		out.Vectors[i] = v
	}

	return out, nil
}
