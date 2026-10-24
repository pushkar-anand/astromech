package llm

import (
	"encoding/base64"
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

func (c wireContent) MarshalJSON() ([]byte, error) {
	if len(c) == 1 && c[0].Type == "text" {
		return json.Marshal(c[0].Text)
	}

	return json.Marshal([]contentPart(c))
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

// content is m's text and media as the LLM backend expects them; nil when
// m has neither.
func (m Message) content() (wireContent, error) {
	var parts wireContent

	if m.Content != "" {
		parts = append(parts, contentPart{Type: "text", Text: m.Content})
	}

	for i, media := range m.Media {
		part, err := media.toWire()
		if err != nil {
			return nil, fmt.Errorf("media %d: %w", i, err)
		}

		parts = append(parts, part)
	}

	return parts, nil
}

func (m Media) toWire() (contentPart, error) {
	kind, subtype, _ := strings.Cut(m.MIMEType, "/")

	switch kind {
	case "image":
		url := m.URL
		if len(m.Data) > 0 {
			url = "data:" + m.MIMEType + ";base64," + base64.StdEncoding.EncodeToString(m.Data)
		}

		if url == "" {
			return contentPart{}, errors.New("no Data or URL")
		}

		return contentPart{Type: "image_url", ImageURL: &imageURL{URL: url}}, nil

	case "audio":
		format, ok := audioFormats[subtype]
		if !ok {
			return contentPart{}, fmt.Errorf("type %q is not supported", m.MIMEType)
		}

		if len(m.Data) == 0 {
			return contentPart{}, errors.New("audio needs Data; LLM backends do not take audio URLs")
		}

		return contentPart{Type: "input_audio", InputAudio: &inputAudio{
			Data:   base64.StdEncoding.EncodeToString(m.Data),
			Format: format,
		}}, nil
	}

	return contentPart{}, fmt.Errorf("type %q is not supported", m.MIMEType)
}
