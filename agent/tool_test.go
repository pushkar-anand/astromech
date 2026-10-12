package agent_test

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/pushkar-anand/astromech/agent"
)

func TestFuncDescribesItsParameters(t *testing.T) {
	type args struct {
		ID    int    `json:"id" jsonschema:"the device's ID"`
		Label string `json:"label,omitzero" jsonschema:"the new label"`
	}

	tool := agent.Func("label_device", "Labels a device.", func(context.Context, args) (string, error) { return "", nil }, agent.Strict())

	def := tool.Def()
	assert.Equal(t, "label_device", def.Name)
	assert.Equal(t, "Labels a device.", def.Description)
	assert.True(t, def.Strict)
	assert.False(t, tool.ReadOnly(), "a Func tool changes things unless marked")

	var schema struct {
		Type       string `json:"type"`
		Required   []string
		Properties map[string]struct {
			Type        string
			Description string
		}
	}
	require.NoError(t, json.Unmarshal(def.Parameters, &schema))

	assert.Equal(t, "object", schema.Type)
	assert.Equal(t, []string{"id"}, schema.Required)
	assert.Equal(t, "integer", schema.Properties["id"].Type)
	assert.Equal(t, "the device's ID", schema.Properties["id"].Description)
	assert.Equal(t, "the new label", schema.Properties["label"].Description)
}

func TestFuncSendsTextAsItIsAndAnythingElseAsJSON(t *testing.T) {
	text := agent.Func("text", "", func(context.Context, struct{}) (string, error) { return `"quoted"`, nil })
	obj := agent.Func("obj", "", func(context.Context, struct{}) ([]int, error) { return []int{1, 2}, nil })

	res, err := text.Call(t.Context(), json.RawMessage(`{}`))
	require.NoError(t, err)
	assert.Equal(t, `"quoted"`, res.Content)

	res, err = obj.Call(t.Context(), nil)
	require.NoError(t, err, "no arguments read as an empty object")
	assert.Equal(t, `[1,2]`, res.Content)
}

func TestFuncPanicsOnParametersItCannotDescribe(t *testing.T) {
	defer func() {
		msg, _ := recover().(string)
		assert.True(t, strings.HasPrefix(msg, "agent: the parameters of tool bad cannot be described: "), msg)
		assert.Contains(t, msg, "chan int")
	}()

	agent.Func("bad", "", func(context.Context, struct {
		C chan int `json:"c"`
	}) (string, error) {
		return "", nil
	})
}
