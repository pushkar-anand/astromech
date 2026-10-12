package llm

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestThinkSplitterAcrossEveryCut(t *testing.T) {
	const reply = "<think>The hall light is on.</think>\n\nThe hall light is on."

	// Cut the reply at every pair of positions, so each tag is split
	// every way it can be.
	for i := range len(reply) {
		for j := i; j <= len(reply); j++ {
			var s thinkSplitter

			var answer, reasoning strings.Builder

			pieces := []string{reply[:i], reply[i:j], reply[j:]}
			segs := []segment{}

			for _, p := range pieces {
				segs = append(segs, s.feed(p)...)
			}

			segs = append(segs, s.flush()...)

			for _, seg := range segs {
				if seg.reasoning {
					reasoning.WriteString(seg.text)
				} else {
					answer.WriteString(seg.text)
				}
			}

			assert.Equal(t, "The hall light is on.", reasoning.String(), "cut at %d and %d", i, j)
			assert.Equal(t, "The hall light is on.", answer.String(), "cut at %d and %d", i, j)
		}
	}
}

func TestSplitThink(t *testing.T) {
	tests := []struct {
		name, in, answer, reasoning string
	}{
		{"no tags", "Just an answer.", "Just an answer.", ""},
		{"tags", "<think>hmm</think>Answer.", "Answer.", "hmm"},
		{"unclosed", "<think>still thinking", "", "still thinking"},
		{"a lone angle bracket", "a < b and <thin ice", "a < b and <thin ice", ""},
		{"text before the tag", "Sure. <think>check</think>Done.", "Sure. Done.", "check"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			answer, reasoning := splitThink(tt.in)
			assert.Equal(t, tt.answer, answer)
			assert.Equal(t, tt.reasoning, reasoning)
		})
	}
}
