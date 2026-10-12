package llm

import "strings"

const (
	thinkOpen  = "<think>"
	thinkClose = "</think>"
)

// segment is a run of reply text that is either reasoning or answer.
type segment struct {
	reasoning bool
	text      string
}

// thinkSplitter separates <think>…</think> reasoning from the answer in a
// reply's text, for servers that leave a reasoning model's thinking inline.
// Text arrives in pieces that can cut a tag in two, so the end of a piece
// that could be the start of a tag is held back until the next one.
type thinkSplitter struct {
	inside  bool
	pending string

	// trim drops the blank lines models put between </think> and the
	// answer.
	trim bool
}

// feed returns the segments of in that can be told apart so far.
func (s *thinkSplitter) feed(in string) []segment {
	buf := s.pending + in
	s.pending = ""

	var out []segment

	for buf != "" {
		tag := thinkOpen
		if s.inside {
			tag = thinkClose
		}

		i := strings.Index(buf, tag)
		if i < 0 {
			keep := partialTag(buf, tag)
			out = s.add(out, buf[:len(buf)-keep])
			s.pending = buf[len(buf)-keep:]

			break
		}

		out = s.add(out, buf[:i])
		buf = buf[i+len(tag):]

		if s.inside {
			s.trim = true
		}

		s.inside = !s.inside
	}

	return out
}

// flush returns what feed held back, once no more text is coming.
func (s *thinkSplitter) flush() []segment {
	rest := s.pending
	s.pending = ""

	return s.add(nil, rest)
}

func (s *thinkSplitter) add(out []segment, text string) []segment {
	if !s.inside && s.trim {
		text = strings.TrimLeft(text, "\r\n")
		if text == "" {
			return out
		}

		s.trim = false
	}

	if text == "" {
		return out
	}

	return append(out, segment{reasoning: s.inside, text: text})
}

// partialTag returns the length of the longest end of buf that tag starts
// with.
func partialTag(buf, tag string) int {
	for n := min(len(buf), len(tag)-1); n > 0; n-- {
		if strings.HasPrefix(tag, buf[len(buf)-n:]) {
			return n
		}
	}

	return 0
}

// splitThink separates a whole reply's text in one pass.
func splitThink(text string) (answer, reasoning string) {
	var s thinkSplitter

	var a, r strings.Builder

	for _, seg := range append(s.feed(text), s.flush()...) {
		if seg.reasoning {
			r.WriteString(seg.text)
		} else {
			a.WriteString(seg.text)
		}
	}

	return a.String(), r.String()
}
