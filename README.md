# astromech

An embeddable LLM agent harness for Go. Give it a system prompt and tools, and
it runs the tool-calling loop against any OpenAI-compatible API. It streams
what happens as events your program renders, in a web chat panel, a CLI, a bot
or a background job, and hands back each message for your program to store.

> Status: under construction. The API will change until `v0.1.0`.

## Packages

| Package     | What it is |
| ----------- | ---------- |
| `llm`       | Client for OpenAI-compatible servers: chat, streamed or whole, with reasoning levels and image input; embeddings; the model list with what each model accepts |
| `llm/llmtest` | A fake Chat Completions server for tests, with scripted replies |
| `agent`     | The loop: tools, permission modes, and approvals that wait across requests |
| `mcptool`   | Tools from any MCP client session, in-process or remote |
| `chathttp`  | Server-sent events framing for `net/http` handlers |

Only `chathttp` knows about HTTP serving. Everything else runs anywhere Go does.

## Compatible servers

Anything that serves the OpenAI Chat Completions API: hosted providers,
OpenRouter, LiteLLM, vLLM, Ollama and LM Studio among them.

## Make targets

| Target                   | What it does                                       |
| ------------------------ | -------------------------------------------------- |
| `make test`              | `go test -race -count=1 ./...`                     |
| `make cover`             | Tests with a coverage profile and summary          |
| `make vet` / `make lint` | `go vet` / `staticcheck`                           |
| `make fmt`               | `go fmt` + `goimports` (what the CI autoformatter runs) |
| `make tidy`              | Sync `go.mod`                                      |

## CI

Three workflows call the reusable ones in [pushkar-anand/.github][gh]: `ci.yml`
(tidy check, build, vet, gofmt, test, staticcheck), `cover.yml` (coverage, no
PR comment) and `autoformat.yml` (runs `make fmt` on PR branches).

[gh]: https://github.com/pushkar-anand/.github
