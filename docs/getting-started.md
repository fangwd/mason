# Getting started

## Configure a model vendor

Mason talks to OpenAI-compatible endpoints. Export your vendors, tried in
comma-separated order. Each vendor needs `AI_<UPPERCASE_NAME>_URL`, `_MODEL`
and `_API_KEY`:

```sh
export AI_VENDORS=openai
export AI_OPENAI_URL=https://api.openai.com/v1
export AI_OPENAI_MODEL=your-model-id
export AI_OPENAI_API_KEY=your-api-key
```

Vendor names use letters, digits and `_`. Optional per-vendor settings:

| Variable | Meaning |
| --- | --- |
| `AI_<NAME>_API_KEY` | May be omitted for servers without authentication (vLLM, Ollama); no `Authorization` header is sent then. |
| `AI_<NAME>_CONTEXT_TOKENS` | Context capacity (default 128000). |
| `AI_<NAME>_STREAM` | Streaming on/off (default true). |
| `AI_<NAME>_EXTRA_BODY` | JSON object merged into every request, e.g. `{"max_tokens": 32000, "temperature": 0.2}` or `{"enable_thinking": true}`. |
| `AI_<NAME>_HEADERS` | JSON object of extra headers, e.g. Azure's `{"api-key": "..."}` or OpenRouter's `{"HTTP-Referer": "..."}`. |

Optionally, `AI_EVALUATOR_VENDORS` names vendors, configured by the same
`AI_<NAME>_*` variables, for a cheaper model that grades every finished run
([architecture.md](architecture.md#run-evaluation)):

```sh
export AI_EVALUATOR_VENDORS=mini
export AI_MINI_URL=https://api.openai.com/v1
export AI_MINI_MODEL=your-cheap-model-id
export AI_MINI_API_KEY=your-api-key
```

The evaluator model must support tool calling: its verdict is a tool call, and
a model that answers in plain text grades every run `unclear`. Its context
must also fit a prompt of up to about 7k tokens. A local model through
Ollama's OpenAI-compatible endpoint:

```sh
export AI_EVALUATOR_VENDORS=ollama
export AI_OLLAMA_URL=http://localhost:11434/v1
export AI_OLLAMA_MODEL=qwen2.5:7b    # any model with tool calling
```

Ollama's default context is smaller than that prompt, and an overlong prompt
is cut silently; raise it on the server (`OLLAMA_CONTEXT_LENGTH=16384 ollama
serve`, or `num_ctx` in a Modelfile), since the OpenAI-compatible API cannot
set it. If streamed tool calls misbehave, set `AI_OLLAMA_STREAM=false`.

## Run

```sh
cargo run -- run "explain this repo"          # one prompt, answer streamed to stdout
cargo run -- serve --listen 127.0.0.1:3120    # session HTTP server; requires MASON_API_TOKEN
```

For `serve`, generate a secret with `openssl rand -hex 32` and set
`MASON_API_TOKEN`; see [api.md](api.md) for the endpoints and
[security.md](security.md) for the access model.

Shared flags (`run` and `serve`): `--cwd`, `--prompts` (YAML overrides of the
embedded prompts, re-read per run), `--skills-dir`, `--mcp` (opencode-format
server map). `serve` adds `--db` (SQLite, default `<cwd>/.mason/mason.db`).
Every flag has a `MASON_*` environment fallback (`mason --help`); `RUST_LOG`
controls logging. Full reference: [configuration.md](configuration.md).

Execution defaults to trusted — tools have the same host access as the user
running mason. `--execution-mode sandbox` opts into OS confinement; see
[security.md](security.md).

## MCP tools

MCP is off until granted. Create a server map and allow it explicitly:

```sh
cargo run -- mcp --mcp mcp.json --allow-mcp    # inspect connections/tools, no LLM call
cargo run -- run --mcp mcp.json --allow-mcp "Use Playwright to inspect my app"
```

Details in [mcp.md](mcp.md).

## Declarative app config

An application whose tools all come over MCP needs no Rust — `--config` reads a
manifest expressing the data-shaped subset of the builder:

```sh
mason serve --config examples/agent.yaml --db data/app.db
mason run --config examples/agent.yaml "explain this repo"
```

Manifest fields and precedence: [configuration.md](configuration.md#declarative-app-config-manifest).
