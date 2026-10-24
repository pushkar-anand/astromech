package llm

import (
	"encoding/base64"
	"encoding/json/jsontext"
	"encoding/json/v2"
	"errors"
	"fmt"
	"strings"
)

// Wire format of a message's content.
type (
	// wireContent is a message's text and media. Text alone is sent as a
	// plain string, which every LLM backend and chat template understands.
	wireContent []contentPart

	contentPart struct {
		Type       string      `json:"type"`
		Text       string      `json:"text,omitzero"`
		ImageURL   *imageURL   `json:"image_url,omitzero"`
		InputAudio *inputAudio `json:"input_audio,omitzero"`
	}

	imageURL struct {
		URL string `json:"url"`
	}

	inputAudio struct {
		Data   string `json:"data"` // base64, without a "data:" prefix
		Format string `json:"format"`
	}
)

// MarshalJSONTo writes into the request's encoder, so media is not encoded
// and then copied in a second time.
func (c wireContent) MarshalJSONTo(enc *jsontext.Encoder) error {
	if len(c) == 1 && c[0].Type == "text" {
		return json.MarshalEncode(enc, c[0].Text)
	}

	return json.MarshalEncode(enc, []contentPart(c))
}

// base64String is prefix then data in base64, built in one allocation since
// media can run to megabytes.
func base64String(prefix string, data []byte) string {
	var b strings.Builder
	b.Grow(len(prefix) + base64.StdEncoding.EncodedLen(len(data)))
	b.WriteString(prefix)

	// Writes to a strings.Builder cannot fail.
	enc := base64.NewEncoder(base64.StdEncoding, &b)
	_, _ = enc.Write(data)
	_ = enc.Close()

	return b.String()
}

// audioFormats maps an audio MIME subtype to the format the LLM backend
// expects.
var audioFormats = map[string]string{
	"wav":   "wav",
	"wave":  "wav",
	"x-wav": "wav",
	"mpeg":  "mp3",
	"mp3":   "mp3",
}

// Media is an image or audio clip sent with a [Message]. It is encoded once,
// when it is made, rather than on every turn that resends it, and cannot
// change after that, so one Media can be shared between messages and
// goroutines. The model must accept that kind of input; on llama.cpp that
// means a server started with --mmproj.
type Media struct {
	mimeType string
	part     contentPart
}

// NewMedia is data of the given MIME type, such as "image/png" or
// "audio/wav".
func NewMedia(mimeType string, data []byte) (*Media, error) {
	if len(data) == 0 {
		return nil, errors.New("llm: media: no data")
	}

	kind, subtype, _ := strings.Cut(mimeType, "/")

	switch kind {
	case "image":
		return &Media{mimeType: mimeType, part: contentPart{
			Type:     "image_url",
			ImageURL: &imageURL{URL: base64String("data:"+mimeType+";base64,", data)},
		}}, nil

	case "audio":
		if format, ok := audioFormats[subtype]; ok {
			return &Media{mimeType: mimeType, part: contentPart{
				Type:       "input_audio",
				InputAudio: &inputAudio{Data: base64String("", data), Format: format},
			}}, nil
		}
	}

	return nil, fmt.Errorf("llm: media: type %q is not supported", mimeType)
}

// NewMediaURL is an image of the given MIME type that the LLM backend
// fetches from url. Not every LLM backend fetches URLs, and none take audio
// this way.
func NewMediaURL(mimeType, url string) (*Media, error) {
	if url == "" {
		return nil, errors.New("llm: media: no URL")
	}

	switch kind, _, _ := strings.Cut(mimeType, "/"); kind {
	case "image":
		return &Media{mimeType: mimeType, part: contentPart{Type: "image_url", ImageURL: &imageURL{URL: url}}}, nil
	case "audio":
		return nil, errors.New("llm: media: audio needs data; LLM backends do not take audio URLs")
	}

	return nil, fmt.Errorf("llm: media: type %q is not supported", mimeType)
}

// MIMEType is the type m was made with.
func (m *Media) MIMEType() string { return m.mimeType }

// content is m's text and media as the LLM backend expects them; nil when
// m has neither.
func (m Message) content() (wireContent, error) {
	var parts wireContent

	if m.Content != "" {
		parts = append(parts, contentPart{Type: "text", Text: m.Content})
	}

	for i, media := range m.Media {
		if media == nil || media.part.Type == "" {
			return nil, fmt.Errorf("media %d was not made by NewMedia or NewMediaURL", i)
		}

		parts = append(parts, media.part)
	}

	return parts, nil
}
