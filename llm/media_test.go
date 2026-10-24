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
	imageURL, err := llm.NewMediaURL("https://example.com/a.png")
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

// Headers just long enough for http.DetectContentType.
const (
	pngHeader = "\x89PNG\r\n\x1a\n"
	wavHeader = "RIFF\x00\x00\x00\x00WAVE"
	mp3Header = "ID3"
)

func TestNewMediaType(t *testing.T) {
	for name, tc := range map[string]struct {
		mimeType, data string
		want           string
		wantPart       map[string]any
	}{
		"as given": {
			mimeType: "audio/x-wav", data: "wav", want: "audio/x-wav",
			wantPart: map[string]any{"type": "input_audio", "input_audio": map[string]any{"data": "d2F2", "format": "wav"}},
		},
		"normalized": {
			mimeType: "Image/PNG; charset=binary", data: "png", want: "image/png",
			wantPart: map[string]any{"type": "image_url", "image_url": map[string]any{"url": "data:image/png;base64,cG5n"}},
		},
		"given type wins": {
			mimeType: "image/webp", data: pngHeader, want: "image/webp",
			wantPart: map[string]any{"type": "image_url", "image_url": map[string]any{"url": "data:image/webp;base64,iVBORw0KGgo="}},
		},
		"detected PNG": {
			data: pngHeader, want: "image/png",
			wantPart: map[string]any{"type": "image_url", "image_url": map[string]any{"url": "data:image/png;base64,iVBORw0KGgo="}},
		},
		"detected WAV": {
			data: wavHeader, want: "audio/wave",
			wantPart: map[string]any{"type": "input_audio", "input_audio": map[string]any{"data": "UklGRgAAAABXQVZF", "format": "wav"}},
		},
		"detected MP3": {
			data: mp3Header, want: "audio/mpeg",
			wantPart: map[string]any{"type": "input_audio", "input_audio": map[string]any{"data": "SUQz", "format": "mp3"}},
		},
	} {
		t.Run(name, func(t *testing.T) {
			m := media(t, tc.mimeType, tc.data)
			assert.Equal(t, tc.want, m.MIMEType())

			b, c := newBackend(t, map[string]reply{
				"POST /v1/chat/completions": {body: okReply},
			})

			_, err := c.Complete(t.Context(), llm.Request{Model: "model-a", Messages: []llm.Message{
				{Role: llm.RoleUser, Media: []*llm.Media{m}},
			}})
			require.NoError(t, err)

			assert.Equal(t, []any{map[string]any{"role": "user", "content": []any{tc.wantPart}}}, b.lastBody()["messages"])
		})
	}
}

func TestNewMediaURLType(t *testing.T) {
	m, err := llm.NewMediaURL("https://example.com/a.png")
	require.NoError(t, err)

	assert.Empty(t, m.MIMEType())
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
		"unknown audio": {
			new:  func() (*llm.Media, error) { return llm.NewMedia("audio/flac", []byte("flac")) },
			want: `llm: media: type "audio/flac" is not supported`,
		},
		"no data": {
			new:  func() (*llm.Media, error) { return llm.NewMedia("image/png", nil) },
			want: "llm: media: no data",
		},
		"no URL": {
			new:  func() (*llm.Media, error) { return llm.NewMediaURL("") },
			want: "llm: media: no URL",
		},
		"bad type": {
			new:  func() (*llm.Media, error) { return llm.NewMedia("image/", []byte("png")) },
			want: "llm: media: mime: expected token after slash",
		},
		"undetectable": {
			new:  func() (*llm.Media, error) { return llm.NewMedia("", []byte("name,age\nAda,36\n")) },
			want: `llm: media: detected type "text/plain" is not supported; pass the MIME type if it is wrong`,
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
