package llm_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/pushkar-anand/astromech/llm"
)

func TestRequestSendsMedia(t *testing.T) {
	for name, tc := range map[string]struct {
		message llm.Message
		want    any
	}{
		"text only": {
			message: llm.Message{Role: llm.RoleUser, Content: "hi"},
			want:    "hi",
		},
		"image data": {
			message: llm.Message{Role: llm.RoleUser, Content: "What is this?", Media: []llm.Media{
				{MIMEType: "image/png", Data: []byte("png")},
			}},
			want: []any{
				map[string]any{"type": "text", "text": "What is this?"},
				map[string]any{"type": "image_url", "image_url": map[string]any{"url": "data:image/png;base64,cG5n"}},
			},
		},
		"image URL": {
			message: llm.Message{Role: llm.RoleUser, Content: "What is this?", Media: []llm.Media{
				{MIMEType: "image/png", URL: "https://example.com/a.png"},
			}},
			want: []any{
				map[string]any{"type": "text", "text": "What is this?"},
				map[string]any{"type": "image_url", "image_url": map[string]any{"url": "https://example.com/a.png"}},
			},
		},
		"data wins over URL": {
			message: llm.Message{Role: llm.RoleUser, Media: []llm.Media{
				{MIMEType: "image/png", Data: []byte("png"), URL: "https://example.com/a.png"},
			}},
			want: []any{
				map[string]any{"type": "image_url", "image_url": map[string]any{"url": "data:image/png;base64,cG5n"}},
			},
		},
		"media without text": {
			message: llm.Message{Role: llm.RoleUser, Media: []llm.Media{
				{MIMEType: "image/jpeg", Data: []byte("jpg")},
			}},
			want: []any{
				map[string]any{"type": "image_url", "image_url": map[string]any{"url": "data:image/jpeg;base64,anBn"}},
			},
		},
		"audio": {
			message: llm.Message{Role: llm.RoleUser, Content: "Transcribe this.", Media: []llm.Media{
				{MIMEType: "audio/x-wav", Data: []byte("wav")},
				{MIMEType: "audio/mpeg", Data: []byte("mp3")},
			}},
			want: []any{
				map[string]any{"type": "text", "text": "Transcribe this."},
				map[string]any{"type": "input_audio", "input_audio": map[string]any{"data": "d2F2", "format": "wav"}},
				map[string]any{"type": "input_audio", "input_audio": map[string]any{"data": "bXAz", "format": "mp3"}},
			},
		},
	} {
		t.Run(name, func(t *testing.T) {
			b, c := newBackend(t, map[string]reply{
				"POST /v1/chat/completions": {body: okReply},
			})

			_, err := c.Complete(t.Context(), llm.Request{Model: "model-a", Messages: []llm.Message{tc.message}})
			require.NoError(t, err)

			assert.Equal(t, []any{map[string]any{"role": "user", "content": tc.want}}, b.lastBody()["messages"])
		})
	}
}

func TestRequestRejectsMedia(t *testing.T) {
	for name, tc := range map[string]struct {
		media llm.Media
		want  string
	}{
		"unknown type":   {media: llm.Media{MIMEType: "application/pdf", Data: []byte("pdf")}, want: `type "application/pdf" is not supported`},
		"no type":        {media: llm.Media{Data: []byte("png")}, want: `type "" is not supported`},
		"unknown audio":  {media: llm.Media{MIMEType: "audio/flac", Data: []byte("flac")}, want: `type "audio/flac" is not supported`},
		"empty image":    {media: llm.Media{MIMEType: "image/png"}, want: "no Data or URL"},
		"audio from URL": {media: llm.Media{MIMEType: "audio/wav", URL: "https://example.com/a.wav"}, want: "audio needs Data"},
	} {
		t.Run(name, func(t *testing.T) {
			// No routes: a request that got through would fail differently.
			_, c := newBackend(t, nil)

			req := llm.Request{Model: "model-a", Messages: []llm.Message{
				{Role: llm.RoleUser, Content: "hi"},
				{Role: llm.RoleUser, Media: []llm.Media{tc.media}},
			}}

			_, err := c.Complete(t.Context(), req)
			require.ErrorContains(t, err, "llm: chat completion: message 1: media 0: "+tc.want)

			for ev := range c.Stream(t.Context(), req) {
				failed, ok := ev.(llm.Failed)
				require.True(t, ok, "got %T before the error", ev)
				assert.ErrorContains(t, failed.Err, "llm: stream: message 1: media 0: "+tc.want)
			}
		})
	}
}
