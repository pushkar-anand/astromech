package agent

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"maps"
	"slices"
	"strings"
	"sync"
	"unicode/utf8"

	"github.com/pushkar-anand/astromech/llm"
)

// Agent runs the turns of conversations between a user, a model and tools.
// A turn only reads an Agent's fields, so one Agent serves many
// conversations at once.
type Agent struct {
	Client *llm.Client

	// Model is the model a turn uses unless the turn names its own.
	Model llm.Model

	// System is the system prompt. It is sent ahead of every request and
	// is not part of the conversation.
	System string

	Tools []Tool

	// Approver decides calls in [ModeAuto].
	Approver Approver

	// Request holds the fields every request starts from, such as
	// Temperature, MaxTokens or Extra. A turn sets its Model, Messages and
	// Tools, and its Reasoning when the turn names an effort.
	Request llm.Request

	// MaxRounds caps how many replies the model gives in one turn. Zero
	// means 10.
	MaxRounds int

	// MaxResultBytes caps a tool result; a longer one is cut, with a note
	// telling the model. One large result can fill a local model's context
	// window. Zero means 32 KiB, and a negative value means no cap.
	MaxResultBytes int
}

const (
	defaultMaxRounds      = 10
	defaultMaxResultBytes = 32 << 10
)

// Turn is one exchange in a conversation.
type Turn struct {
	// History is the conversation so far: the messages earlier turns added,
	// as the caller stored them. It holds no system prompt.
	History []llm.Message

	// Input is the user's new message. It is nil for a turn that settles
	// calls awaiting approval, or one that continues after
	// [StopMaxRounds].
	Input *llm.Message

	// Decisions holds a person's verdicts on calls awaiting approval, by
	// call ID. A call without one keeps waiting, unless Input is set: a
	// new message declines every call still waiting, because a server
	// refuses a conversation that moves on from a call with no result.
	Decisions map[string]Verdict

	// Model, when its ID is set, replaces [Agent.Model] for this turn.
	Model llm.Model

	// Effort is how hard a reasoning model thinks. It is sent only to a
	// model whose capabilities include reasoning or are not known.
	Effort llm.Effort

	Mode Mode
}

// Run plays one turn and streams what happens. Stopping the loop early
// cancels the turn, and the context of any tool still running ends.
func (a *Agent) Run(ctx context.Context, turn Turn) iter.Seq[Event] {
	return func(yield func(Event) bool) {
		r, waiting, err := a.start(turn)
		if err != nil {
			yield(Failed{Err: err})

			return
		}

		r.yield = yield
		r.play(ctx, waiting)
	}
}

// run is the state of one turn.
type run struct {
	agent     *Agent
	turn      Turn
	model     llm.Model
	mode      Mode
	maxRounds int
	maxResult int

	// tools are the tools the turn offers, and known every tool the agent
	// has, to tell the model why a tool it names is missing.
	tools map[string]Tool
	known map[string]Tool
	defs  []llm.ToolDef

	conv  []llm.Message
	added []llm.Message
	usage llm.Usage

	yield   func(Event) bool
	stopped bool
}

type job struct {
	call llm.ToolCall
	tool Tool
}

// start checks turn and returns its run and the calls in its history that
// wait for a verdict.
func (a *Agent) start(turn Turn) (*run, []llm.ToolCall, error) {
	if a.Client == nil {
		return nil, nil, errors.New("agent: Agent.Client is nil; set it to an llm.Client")
	}

	r := &run{
		agent:     a,
		turn:      turn,
		model:     a.Model,
		mode:      cmp.Or(turn.Mode, ModeManual),
		maxRounds: cmp.Or(a.MaxRounds, defaultMaxRounds),
		maxResult: cmp.Or(a.MaxResultBytes, defaultMaxResultBytes),
		tools:     make(map[string]Tool, len(a.Tools)),
		known:     make(map[string]Tool, len(a.Tools)),
		conv:      slices.Clone(turn.History),
	}

	if turn.Model.ID != "" {
		r.model = turn.Model
	}

	if r.model.ID == "" {
		return nil, nil, errors.New("agent: the turn names no model; set Agent.Model or Turn.Model")
	}

	switch r.mode {
	case ModeManual, ModeAuto, ModeBypass, ModeReadOnly:
	default:
		return nil, nil, fmt.Errorf("agent: mode %q is not one of manual, auto, bypass or read_only", r.mode)
	}

	for _, t := range a.Tools {
		def := t.Def()
		if _, dup := r.known[def.Name]; dup {
			return nil, nil, fmt.Errorf("agent: two tools are named %s; each tool needs a name of its own", def.Name)
		}

		r.known[def.Name] = t

		if r.mode == ModeReadOnly && !t.ReadOnly() {
			continue
		}

		r.tools[def.Name] = t
		r.defs = append(r.defs, def)
	}

	waiting, err := pending(turn.History)
	if err != nil {
		return nil, nil, err
	}

	for _, id := range slices.Sorted(maps.Keys(turn.Decisions)) {
		if !slices.ContainsFunc(waiting, func(c llm.ToolCall) bool { return c.ID == id }) {
			return nil, nil, fmt.Errorf("agent: there is a verdict for call %s, which is not waiting for one", id)
		}

		switch d := turn.Decisions[id].Decision; d {
		case DecisionAllow, DecisionDeny, DecisionAsk:
		default:
			return nil, nil, fmt.Errorf("agent: the verdict for call %s is %q, which is not one of allow, deny or ask", id, d)
		}
	}

	return r, waiting, nil
}

func (r *run) play(ctx context.Context, waiting []llm.ToolCall) {
	if len(waiting) > 0 {
		left, ok := r.settle(ctx, waiting, r.turn.Decisions, r.turn.Input != nil)
		if !ok {
			return
		}

		if left > 0 {
			r.done(StopAwaitingApproval)

			return
		}
	}

	if r.turn.Input != nil && !r.add(*r.turn.Input) {
		return
	}

	for range r.maxRounds {
		if ctx.Err() != nil {
			r.fail(context.Cause(ctx))

			return
		}

		resp, err := r.ask(ctx)
		if r.stopped {
			return
		}

		if err != nil {
			if resp != nil && !r.add(resp.Message) {
				return
			}

			r.fail(err)

			return
		}

		if !r.add(resp.Message) {
			return
		}

		calls := resp.Message.ToolCalls
		if len(calls) == 0 {
			r.done(stopOf(resp.Stop))

			return
		}

		left, ok := r.settle(ctx, calls, nil, false)
		if !ok {
			return
		}

		if ctx.Err() != nil {
			r.fail(context.Cause(ctx))

			return
		}

		if left > 0 {
			r.done(StopAwaitingApproval)

			return
		}
	}

	r.done(StopMaxRounds)
}

// ask sends the conversation to the model and streams its reply. On a
// failure it returns what arrived of the reply, which can be nil.
func (r *run) ask(ctx context.Context) (*llm.Response, error) {
	caps := r.model.Capabilities

	req := r.agent.Request
	req.Model = r.model.ID
	req.Messages = r.messages()
	req.Tools = nil

	if caps == nil || caps.Tools {
		req.Tools = r.defs
	}

	if effort := cmp.Or(r.turn.Effort, req.Reasoning); caps == nil || caps.Reasoning {
		req.Reasoning = effort
	} else {
		req.Reasoning = ""
	}

	for e := range r.agent.Client.Stream(ctx, req) {
		var out Event

		switch e := e.(type) {
		case llm.TextDelta:
			out = TextDelta{Text: e.Text}
		case llm.ReasoningDelta:
			out = ReasoningDelta{Text: e.Text}
		case llm.Retrying:
			out = Retrying{Attempt: e.Attempt, Err: e.Err}
		case llm.Done:
			r.usage = addUsage(r.usage, e.Response.Usage)

			return e.Response, nil
		case llm.Failed:
			if e.Partial != nil {
				r.usage = addUsage(r.usage, e.Partial.Usage)
			}

			return e.Partial, e.Err
		}

		if out != nil && !r.emit(out) {
			return nil, nil
		}
	}

	return nil, errors.New("agent: the model's reply ended without finishing or failing")
}

// messages returns the conversation as the model is sent it: after the
// system prompt, and without images for a model that cannot see them.
func (r *run) messages() []llm.Message {
	caps := r.model.Capabilities
	images := caps == nil || caps.Images

	msgs := make([]llm.Message, 0, len(r.conv)+1)
	if r.agent.System != "" {
		msgs = append(msgs, llm.System(r.agent.System))
	}

	for _, m := range r.conv {
		if len(m.Images) > 0 && !images {
			note := fmt.Sprintf("[%d image(s) left out: this model cannot see images.]", len(m.Images))
			m.Content = strings.TrimSpace(m.Content + "\n\n" + note)
			m.Images = nil
		}

		msgs = append(msgs, m)
	}

	return msgs
}

// settle answers calls: it runs those that may run, answers those that may
// not, and leaves the rest waiting for a person. verdicts are a person's,
// and declineWaiting answers the calls that would wait. It returns how many
// calls wait, and false once the caller stopped reading.
func (r *run) settle(ctx context.Context, calls []llm.ToolCall, verdicts map[string]Verdict, declineWaiting bool) (int, bool) {
	answers := make(map[string]Result, len(calls))
	waiting := 0

	var runs []job

	for _, call := range calls {
		tool, offered := r.tools[call.Name]
		if !offered {
			answers[call.ID] = r.unavailable(call)

			if !r.emit(ToolFinished{Call: call, Result: answers[call.ID]}) {
				return 0, false
			}

			continue
		}

		verdict, decided := verdicts[call.ID]
		byPerson := decided && verdict.Decision != DecisionAsk
		decided = byPerson

		switch {
		case decided:
		case tool.ReadOnly() || r.mode == ModeBypass:
			verdict, decided = Verdict{Decision: DecisionAllow}, true
		case r.mode == ModeAuto && r.agent.Approver != nil:
			verdict = r.approve(ctx, call, tool)
			decided = verdict.Decision != DecisionAsk

			if decided && !r.emit(Decided{Call: call, Verdict: verdict}) {
				return 0, false
			}
		}

		switch {
		case !decided && declineWaiting:
			answers[call.ID] = Result{Content: "The user sent a new message without deciding on this call, so it did not run.", IsError: true}
		case !decided:
			waiting++

			if !r.emit(ApprovalRequired{Call: call, Tool: tool.Def(), Reason: verdict.Reason}) {
				return 0, false
			}

			continue
		case verdict.Decision == DecisionAllow:
			runs = append(runs, job{call: call, tool: tool})

			continue
		default:
			answers[call.ID] = declined(verdict, byPerson)
		}

		if !r.emit(ToolFinished{Call: call, Result: answers[call.ID]}) {
			return 0, false
		}
	}

	if !r.execute(ctx, runs, answers) {
		return 0, false
	}

	for _, call := range calls {
		if res, ok := answers[call.ID]; ok && !r.add(llm.ToolResult(call, res.Content)) {
			return 0, false
		}
	}

	return waiting, true
}

// execute runs jobs and records each result in answers. Calls to read-only
// tools run at once, since they cannot get in each other's way; a batch
// with any call that changes things runs in the order the model gave.
func (r *run) execute(ctx context.Context, jobs []job, answers map[string]Result) bool {
	parallel := len(jobs) > 1 && !slices.ContainsFunc(jobs, func(j job) bool { return !j.tool.ReadOnly() })

	if !parallel {
		for _, j := range jobs {
			if !r.emit(ToolStarted{Call: j.call}) {
				return false
			}

			answers[j.call.ID] = r.call(ctx, j)

			if !r.emit(ToolFinished{Call: j.call, Result: answers[j.call.ID]}) {
				return false
			}
		}

		return true
	}

	for _, j := range jobs {
		if !r.emit(ToolStarted{Call: j.call}) {
			return false
		}
	}

	// The deferred cancel runs before the deferred Wait, so a caller that
	// stops reading ends the tools' context before waiting for them.
	var wg sync.WaitGroup
	defer wg.Wait()

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	results := make([]Result, len(jobs))
	finished := make(chan int, len(jobs))

	for i, j := range jobs {
		wg.Go(func() {
			results[i] = r.call(ctx, j)
			finished <- i
		})
	}

	for range jobs {
		i := <-finished
		answers[jobs[i].call.ID] = results[i]

		if !r.emit(ToolFinished{Call: jobs[i].call, Result: results[i]}) {
			return false
		}
	}

	return true
}

// call runs one tool. A tool that fails or panics answers with what went
// wrong, so the model can try again and the program keeps running.
func (r *run) call(ctx context.Context, j job) (res Result) {
	defer func() {
		if p := recover(); p != nil {
			res = Result{Content: fmt.Sprintf("%s failed: %v", j.call.Name, p), IsError: true}
		}

		res.Content = cut(res.Content, r.maxResult)
	}()

	if ctx.Err() != nil {
		return Result{Content: "The turn ended before this call ran.", IsError: true}
	}

	res, err := j.tool.Call(ctx, json.RawMessage(j.call.Arguments))
	if err != nil {
		return Result{Content: fmt.Sprintf("%s failed: %v", j.call.Name, err), IsError: true}
	}

	return res
}

func (r *run) approve(ctx context.Context, call llm.ToolCall, tool Tool) Verdict {
	v, err := r.agent.Approver.Decide(ctx, ApprovalRequest{Call: call, Tool: tool.Def(), Conversation: slices.Clip(r.conv)})
	if err != nil {
		return Verdict{Decision: DecisionAsk, Reason: "The approver failed: " + err.Error()}
	}

	if v.Decision != DecisionAllow && v.Decision != DecisionDeny {
		v.Decision = DecisionAsk
	}

	return v
}

// unavailable answers a call to a tool the turn does not offer.
func (r *run) unavailable(call llm.ToolCall) Result {
	if _, ok := r.known[call.Name]; ok {
		return Result{Content: fmt.Sprintf("%s changes things, and this conversation is read-only, so the call did not run.", call.Name), IsError: true}
	}

	if len(r.tools) == 0 {
		return Result{Content: fmt.Sprintf("There is no tool named %q, and no tools are available.", call.Name), IsError: true}
	}

	names := slices.Sorted(maps.Keys(r.tools))

	return Result{Content: fmt.Sprintf("There is no tool named %q. The tools are %s.", call.Name, strings.Join(names, ", ")), IsError: true}
}

func declined(v Verdict, byPerson bool) Result {
	text := "This call was declined automatically, so it did not run."
	if byPerson {
		text = "The user declined this call, so it did not run."
	}

	if v.Reason != "" {
		text += " The reason given: " + v.Reason
	}

	return Result{Content: text, IsError: true}
}

func (r *run) emit(e Event) bool {
	if r.stopped {
		return false
	}

	if !r.yield(e) {
		r.stopped = true
	}

	return !r.stopped
}

func (r *run) add(m llm.Message) bool {
	r.conv = append(r.conv, m)
	r.added = append(r.added, m)

	return r.emit(MessageAdded{Message: m})
}

func (r *run) done(stop Stop) {
	r.emit(Done{Stop: stop, Messages: r.added, Usage: r.usage})
}

func (r *run) fail(err error) {
	r.emit(Failed{Err: err, Messages: r.added, Usage: r.usage})
}

// cut shortens a tool result to limit bytes, on a character boundary, and
// says so at its end. A negative limit keeps the whole result.
func cut(s string, limit int) string {
	if limit < 0 || len(s) <= limit {
		return s
	}

	n := limit
	for n > 0 && !utf8.RuneStart(s[n]) {
		n--
	}

	return s[:n] + fmt.Sprintf("\n\n[The result was cut to its first %d of %d bytes.]", n, len(s))
}

func stopOf(s llm.StopReason) Stop {
	switch s {
	case llm.StopLength:
		return StopLength
	case llm.StopFiltered:
		return StopFiltered
	}

	return StopAnswered
}

func addUsage(a, b llm.Usage) llm.Usage {
	return llm.Usage{
		InputTokens:     a.InputTokens + b.InputTokens,
		OutputTokens:    a.OutputTokens + b.OutputTokens,
		CachedTokens:    a.CachedTokens + b.CachedTokens,
		ReasoningTokens: a.ReasoningTokens + b.ReasoningTokens,
	}
}
