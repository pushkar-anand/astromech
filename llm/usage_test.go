package llm_test

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/pushkar-anand/astromech/llm"
)

func TestUsageAdd(t *testing.T) {
	// Every field differs, so one left out of Add shows.
	a := llm.Usage{InputTokens: 1, OutputTokens: 2, TotalTokens: 3, CachedTokens: 4}
	b := llm.Usage{InputTokens: 10, OutputTokens: 20, TotalTokens: 30, CachedTokens: 40}

	assert.Equal(t, llm.Usage{InputTokens: 11, OutputTokens: 22, TotalTokens: 33, CachedTokens: 44}, a.Add(b))
	assert.Equal(t, a, llm.Usage{}.Add(a), "the zero Usage adds nothing")
}
