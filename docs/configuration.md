# Configuration

Three layers configure mason, in precedence order:

1. **Explicit flags / `MASON_*` environment variables** — only values supplied
   explicitly take part, so a clap default never outranks a manifest, and a
   boolean grant can be revoked with `--allow-mcp=false`.
2. **A `--config` manifest** — see below. `--mcp <file>` replaces the
   manifest's MCP servers wholesale.
3. **Mason's defaults**.

## Flags and environment

Model vendors are configured with `AI_*` variables; see
[getting-started.md](getting-started.md). Every flag below has a `MASON_*`
environment fallback of the same name. Durations use `90s`, `15m`, `1h30m`.

Shared by `run`, `serve` and `mcp`:

| Flag | Env | Default | Meaning |
| --- | --- | --- | --- |
| `--config` | `MASON_CONFIG` | — | Declarative app config manifest. |
| `--execution-mode` | `MASON_EXECUTION_MODE` | `trusted` | `sandbox` confines subprocesses and file tools ([security.md](security.md)). |
| `--allow-network` | `MASON_ALLOW_NETWORK` | false | Grant network access to sandboxed subprocesses. |
| `--allow-mcp` | `MASON_ALLOW_MCP` | false | Grant configured MCP servers access to agent tool calls. |
| `--run-timeout` | `MASON_RUN_TIMEOUT` | 15m | Run budget. |
| `--model-request-timeout` | `MASON_MODEL_REQUEST_TIMEOUT` | 2m | Per-attempt model budget. |
| `--model-turn-timeout` | `MASON_MODEL_TURN_TIMEOUT` | 3m | Per model turn across all attempts and vendors. |
| `--model-idle-timeout` | `MASON_MODEL_IDLE_TIMEOUT` | 1m | Stream idle timeout. |
| `--shell-timeout` | `MASON_SHELL_TIMEOUT` | 2m | Default shell timeout. |
| `--shell-max-timeout` | `MASON_SHELL_MAX_TIMEOUT` | 10m | Maximum shell timeout. |
| `--max-steps` | `MASON_MAX_STEPS` | 150 | Tool-using steps per run; also caps delegated profiles. |
| `--max-concurrent-tools` | `MASON_MAX_CONCURRENT_TOOLS` | 4 | Tool calls running at once, shared with subagents. |
| `--max-tool-calls-per-step` | `MASON_MAX_TOOL_CALLS_PER_STEP` | 16 | Tool calls per model reply. |
| `--max-task-calls-per-step` | `MASON_MAX_TASK_CALLS_PER_STEP` | 4 | `task` calls per model reply. |
| `--max-total-tool-calls` | `MASON_MAX_TOTAL_TOOL_CALLS` | 1000 | Root + child tool-call budget. |
| `--max-total-tokens` | `MASON_MAX_TOTAL_TOKENS` | 20000000 | Root + child token budget (estimated). |
| `--max-response-bytes` | `MASON_MAX_RESPONSE_BYTES` | 8 MiB | Bound on one model turn. |
| `--spill-quota-mib` | `MASON_SPILL_QUOTA_MIB` | 256 | Total spill storage, including in-progress captures. |
| `--cwd` | `MASON_CWD` | `.` | Working directory. |
| `--prompts` | `MASON_PROMPTS` | embedded | YAML overrides of the embedded prompts, re-read per run. |
| `--skills-dir` | `MASON_SKILLS_DIR` | — | Directory of skills (`<name>/SKILL.md`), re-read per run. |
| `--memory-dir` | `MASON_MEMORY_DIR` | — | Persistent agent memory shared by every session ([architecture.md](architecture.md#memory)). Created when missing, so in a container it must be on a persistent volume or memory starts empty on every deploy. |
| `--mcp` | `MASON_MCP` | — | JSON file mapping server names to MCP server configs ([mcp.md](mcp.md)). |

`run` adds `--profile` (`MASON_PROFILE`, default `build`). `serve` adds:

| Flag | Env | Default | Meaning |
| --- | --- | --- | --- |
| `--listen` | `MASON_LISTEN` | `127.0.0.1:3120` | Address to listen on. |
| `--api-token` | `MASON_API_TOKEN` | required | Bearer token for all API clients (at least 16 characters). |
| `--max-active-runs` | `MASON_MAX_ACTIVE_RUNS` | 4 | Concurrently active root sessions. |
| `--max-pending-sessions` | `MASON_MAX_PENDING_SESSIONS` | 128 | Admitted running/queued root sessions. |
| `--db` | `MASON_DB` | `<cwd>/.mason/mason.db` | SQLite session database. |

What these budgets do at runtime: [architecture.md](architecture.md).

## Declarative app config (manifest)

`--config` reads a YAML manifest expressing the data-shaped subset of the
builder; an application whose tools all come over MCP needs no Rust. A
commented example lives in `examples/agent.yaml`.

```sh
mason serve --config examples/agent.yaml --db data/app.db
```

A manifest carries:

- `instructions` — domain system-prompt prose appended to the generic base.
  No secrets here; model credentials come from the `AI_*` environment.
- `profiles` — a builtin name *patches* the resolved builtin, keeping
  everything it does not mention; any other name registers a new profile
  (a description is required). Fields: `description`, `prompt`, `tools`,
  `allowed_subagents`, `subagent`, `max_steps`.
- `select_profile` — rules choosing a profile from a session's opaque
  `metadata`. Conditions are ANDed, the first matching rule wins, and
  `present` means the path exists and is not null.
- `mcp` servers (or an `mcp_file`), plus the execution grants
  `execution_mode` (default `trusted`), `allow_network` and `allow_mcp`
  (both fail closed).
- Optionally `prompts`, `skills_dir`, `memory_dir`, `limits`, `spill` and `cwd`. Relative
  paths resolve against the manifest's own directory.
- `version:` — the schema version. `GET /healthz` reports the binary's
  version, so an application can enforce a floor at startup (that route needs
  no token).

`limits` uses the same names as the flags and the same duration syntax, e.g.
`--shell-timeout 5m` or `limits: {shell_timeout: 5m}`.

### Growing beyond the manifest

`Manifest::load(path)?.into_builder().await?` returns an `AgentBuilder`, so an
application can start declarative and add code on top of the same file. Native
tools, a custom `ChatModel` or `SessionStore`, a `ToolPolicy` approval hook and
selector logic richer than metadata matching are the parts a manifest cannot
express; installing a code selector alongside a manifest's rules is a build
error, not a silent override. See [embedding.md](embedding.md).
