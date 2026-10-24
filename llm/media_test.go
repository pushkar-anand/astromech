package llm_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/pushkar-anand/astromech/llm"
)

// media is NewMedia, failing the test on an error.
func media(t *testing.T, mimeType, data string) *llm.Media {
	t.Helper()

	m, err := llm.NewMedia(mimeType, []byte(data))
	require.NoError(t, err)

	return m
}

func TestRequestSendsMedia(t *testing.T) {
	imageURL, err := llm.NewMediaURL("image/png", "https://example.com/a.png")
	require.NoError(t, err)

	for name, tc := range map[string]struct {
		message llm.Message
		want    any
	}{
		"text only": {
			message: llm.Message{Role: llm.RoleUser, Content: "hi"},
			want:    "hi",
		},
		"image data": {
			message: llm.Message{Role: llm.RoleUser, Content: "What is this?", Media: []*llm.Media{
				media(t, "image/png", "png"),
			}},
			want: []any{
				map[string]any{"type": "text", "text": "What is this?"},
				map[string]any{"type": "image_url", "image_url": map[string]any{"url": "data:image/png;base64,cG5n"}},
			},
		},
		"image URL": {
			message: llm.Message{Role: llm.RoleUser, Content: "What is this?", Media: []*llm.Media{imageURL}},
			want: []any{
				map[string]any{"type": "text", "text": "What is this?"},
				map[string]any{"type": "image_url", "image_url": map[string]any{"url": "https://example.com/a.png"}},
			},
		},
		"media without text": {
			message: llm.Message{Role: llm.RoleUser, Media: []*llm.Media{
				media(t, "image/jpeg", "jpg"),
			}},
			want: []any{
				map[string]any{"type": "image_url", "image_url": map[string]any{"url": "data:image/jpeg;base64,anBn"}},
			},
		},
		"audio": {
			message: llm.Message{Role: llm.RoleUser, Content: "Transcribe this.", Media: []*llm.Media{
				media(t, "audio/x-wav", "wav"),
				media(t, "audio/mpeg", "mp3"),
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

func TestRequestSharesMedia(t *testing.T) {
	b, c := newBackend(t, map[string]reply{
		"POST /v1/chat/completions": {body: okReply},
	})

	// One Media in two turns, as a conversation that resends its history.
	image := media(t, "image/png", "png")
	part := map[string]any{"type": "image_url", "image_url": map[string]any{"url": "data:image/png;base64,cG5n"}}

	_, err := c.Complete(t.Context(), llm.Request{Model: "model-a", Messages: []llm.Message{
		{Role: llm.RoleUser, Media: []*llm.Media{image}},
		{Role: llm.RoleAssistant, Content: "A picture."},
		{Role: llm.RoleUser, Media: []*llm.Media{image}},
	}})
	require.NoError(t, err)

	assert.Equal(t, []any{
		map[string]any{"role": "user", "content": []any{part}},
		map[string]any{"role": "assistant", "content": "A picture."},
		map[string]any{"role": "user", "content": []any{part}},
	}, b.lastBody()["messages"])
}

func TestMediaMIMEType(t *testing.T) {
	assert.Equal(t, "audio/x-wav", media(t, "audio/x-wav", "wav").MIMEType())
}

func TestNewMediaRejects(t *testing.T) {
	for name, tc := range map[string]struct {
		new  func() (*llm.Media, error)
		want string
	}{
		"unknown type": {
			new:  func() (*llm.Media, error) { return llm.NewMedia("application/pdf", []byte("pdf")) },
			want: `llm: media: type "application/pdf" is not supported`,
		},
		"no type": {
			new:  func() (*llm.Media, error) { return llm.NewMedia("", []byte("png")) },
			want: `llm: media: type "" is not supported`,
		},
		"unknown audio": {
			new:  func() (*llm.Media, error) { return llm.NewMedia("audio/flac", []byte("flac")) },
			want: `llm: media: type "audio/flac" is not supported`,
		},
		"no data": {
			new:  func() (*llm.Media, error) { return llm.NewMedia("image/png", nil) },
			want: "llm: media: no data",
		},
		"no URL": {
			new:  func() (*llm.Media, error) { return llm.NewMediaURL("image/png", "") },
			want: "llm: media: no URL",
		},
		"audio from URL": {
			new:  func() (*llm.Media, error) { return llm.NewMediaURL("audio/wav", "https://example.com/a.wav") },
			want: "llm: media: audio needs data; LLM backends do not take audio URLs",
		},
		"unknown type from URL": {
			new:  func() (*llm.Media, error) { return llm.NewMediaURL("text/csv", "https://example.com/a.csv") },
			want: `llm: media: type "text/csv" is not supported`,
		},
	} {
		t.Run(name, func(t *testing.T) {
			m, err := tc.new()

			assert.Nil(t, m)
			assert.EqualError(t, err, tc.want)
		})
	}
}

func TestRequestRejectsMedia(t *testing.T) {
	for name, m := range map[string]*llm.Media{
		"nil":  nil,
		"zero": {},
	} {
		t.Run(name, func(t *testing.T) {
			// No routes: a request that got through would fail differently.
			_, c := newBackend(t, nil)

			req := llm.Request{Model: "model-a", Messages: []llm.Message{
				{Role: llm.RoleUser, Content: "hi"},
				{Role: llm.RoleUser, Media: []*llm.Media{media(t, "image/png", "png"), m}},
			}}

			const want = "message 1: media 1 was not made by NewMedia or NewMediaURL"

			_, err := c.Complete(t.Context(), req)
			require.EqualError(t, err, "llm: chat completion: "+want)

			_, err = llm.Collect(c.Stream(t.Context(), req), nil)
			assert.EqualError(t, err, "llm: stream: "+want)
		})
	}
}
