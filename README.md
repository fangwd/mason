# Mason

Mason — an AI agent that builds domain-specific tools.

Mason is the agent core: a model/tool loop, an LLM client, a tool framework,
context management, subagents, skills, sessions and an HTTP API. It is
domain-independent — domain-specific tools, profiles and prompts live in the
application that embeds it.

## Quickstart

Export an OpenAI-compatible vendor, then run:

```sh
export AI_VENDORS=openai
export AI_OPENAI_URL=https://api.openai.com/v1
export AI_OPENAI_MODEL=your-model-id
export AI_OPENAI_API_KEY=your-api-key

cargo run -- run "explain this repo"          # one prompt, answer streamed to stdout
cargo run -- serve --listen 127.0.0.1:3120    # session HTTP API; requires MASON_API_TOKEN
```

Vendor options, flags and environment fallbacks:
[docs/getting-started.md](docs/getting-started.md).

## Embed

```rust
let agent = Agent::builder()
    .vendors(Vendors::from_env()?)
    .cwd(repo_root)
    .instructions("You are an agent working on the Foo service.")
    .tools(builtin_tools())
    .build()?;

let answer = agent.run_once("explain this repo").await?;
```

Tools, profiles, prompts, stores and models are all extension points:
[docs/embedding.md](docs/embedding.md). An app whose tools all come over MCP
needs no Rust — a YAML manifest is enough
([docs/configuration.md](docs/configuration.md#declarative-app-config-manifest)).

## Documentation

| Doc | Contents |
| --- | --- |
| [getting-started.md](docs/getting-started.md) | Vendor setup, `run`/`serve`, common flags |
| [configuration.md](docs/configuration.md) | Flag/env reference, budgets, declarative app config |
| [embedding.md](docs/embedding.md) | Builder API and extension points |
| [mcp.md](docs/mcp.md) | MCP server map, lifecycle, sandbox interaction |
| [api.md](docs/api.md) | HTTP endpoints, events/SSE, session lifecycle |
| [security.md](docs/security.md) | Execution modes, sandbox, access and persistence |
| [architecture.md](docs/architecture.md) | Runtime limits, context management, source layout |

## Operating profile rules

Profile `shell`, `write_paths` and `hooks` ([configuration.md](docs/configuration.md))
hold only when they are operated this way:

- **Write `shell` patterns as narrowly as the commands.** A trailing `*` admits
  any flags: `git diff*` admits `git diff --output=FILE`, which writes a file.
- **Give a profile with `write_paths` a `shell` list too, and never allow `ln`.**
  `write_paths` is checked against where a write lands, symlinks followed, but a
  symlink created or swapped after the check is outside it; an unrestricted shell
  can write anywhere anyway.
- **Hook tools must report failure as an error.** `bash` reports a failing
  command's exit code in its output, so it is refused as a hook; wrap a shell
  check in an MCP tool that fails with an error.
- **Hooks are called like ordinary tool calls** (host policy and profile rules
  apply) and only on `write` and `edit`; a file changed another way is not checked.
- **Restart `mason serve` after changing `agent.yaml`.** Profiles, rules and hooks
  are read at startup. MCP *tools* need no restart: every run lists them again,
  and a session's server process whose tools differ from a run's listing is
  restarted as that run starts (never mid-run), losing the process's in-memory
  state.

## Development

```sh
./etc/check.sh              # fmt + clippy (both feature configs) + tests
./etc/check.sh --coverage   # the same with a cargo-llvm-cov report (optional)
```

Minimum supported Rust version: 1.94 (the `sqlx` floor), recorded in
`Cargo.toml`.
