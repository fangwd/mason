# Embedding mason

Mason is the agent core: a model/tool loop, an LLM client, a tool framework,
context management, subagents, skills, sessions and an HTTP API. It is
domain-independent — domain-specific tools, profiles and prompts live in the
application that embeds it.

## Builder

```rust
let agent = Agent::builder()
    .vendors(Vendors::from_env()?)                     // or .model(custom_chat_model)
    .cwd(repo_root)
    .instructions("You are an agent working on the Foo service.")
    .tools(builtin_tools())                            // read/write/edit/bash/grep/glob/process
    .tool(Box::new(Typed(SqlTool::new(pool))))         // any domain tool
    .profiles([my_profile])                            // any domain profile
    .skills_dir("skills")
    .prompts(PromptSource::File("prompts/prompts.yaml".into()))
    .build()?;

let answer = agent.run_once("explain this repo").await?;
```

Sessions, the runner and the HTTP API:

```rust
let store = Arc::new(SqliteStore::open("data/mason.db").await?);
let runner = Runner::new(agent, store.clone());
runner.recover_and_drain().await?;                     // startup recovery
let security = mason::api::ApiSecurity::new(std::env::var("MASON_API_TOKEN")?)?;
axum::serve(listener, mason::api::router_with_security(
    runner, CancellationToken::new(), security,
)).await?;
```

A declarative starting point: `Manifest::load(path)?.into_builder().await?`
returns an `AgentBuilder` ([configuration.md](configuration.md#declarative-app-config-manifest)).

## Extension points

Extension points — and nothing else — are how a domain app customises mason:

- **`Tool` / `TypedTool`** — arbitrary tools, constructed with their own
  config. `TypedTool` gets serde-validated arguments (`deny_unknown_fields`
  for builtins) and a `schemars`-derived schema; restricted variants get their
  own name, truthful spec and execution checks.
- **`Profile`** — name, description, prompt, allowed tool names, allowed
  subagent profiles, subagent flag, `max_steps`. The builtins are `build`
  (the default), `general` and — when `read`/`grep`/`glob` are registered —
  `explore`. Subagent profiles cannot delegate further; the `task` tool lists
  only the profile's explicit child allowlist and re-checks it at execution.
- **`replace_profile`** — replaces a builtin profile by name, keeping the child
  allowlists that already name it, so an app can widen `explore` with its own
  read-only tools. Builtin prompt texts still come from the prompt snapshot.
- **Session metadata and `profile_selector`** — `POST /sessions` accepts opaque
  JSON `metadata`, stored unread and returned by `GET /sessions` and
  `GET /sessions/:id`. A selector maps a stored top-level session to the
  profile it runs, replacing its `agent`; an unknown choice fails that run
  with a diagnostic, and child sessions never pass through it.
- **`instructions`** — domain system-prompt prose appended to the generic
  base.
- **`PromptSource`** — substantial prompts as data (`prompts/prompts.yaml`,
  embedded by default). A `File` holds only the keys it changes, deep-merged
  over the embedded set (e.g. just `compaction: {template: ...}`); it is
  validated at build and re-read at the start of every run, so prompt
  iteration needs no restart or rebuild. Sections derived from opencode are
  marked in `prompts/prompts.yaml`; the full MIT notice is in `LICENSE` under
  "Third-party notices".
- **`memory_dir`** — persistent lessons the agent saves through the `memory`
  tool and sees indexed in every run's system prompt; shared by all sessions,
  so keep per-user data out of it ([architecture.md](architecture.md#memory)).
  One agent owns a memory directory; `memory` is a reserved tool name only
  when memory is on.
  Flagged runs get a post-run reflection on the agent's model, an extra
  request of up to 8 steps; `Agent::reflect` reviews runs the runner does not
  own. Consolidation periodically merges and prunes memories and promotes
  procedures to learned skills; `Agent::consolidate` runs it on demand.
- **`Evaluator`** — grades finished runs for analytics (`run.evaluated`) and
  to steer reflection. `LlmEvaluator` asks a cheaper model; an application
  can implement the trait with real user feedback instead
  ([architecture.md](architecture.md#run-evaluation)). Only
  `Manifest::into_builder` and `AgentSpec::into_builder` read
  `AI_EVALUATOR_VENDORS`; with `into_builder_with_vendors`,
  `into_builder_with_model` or a hand-built builder, set it explicitly:

  ```rust
  if let Some(vendors) = Vendors::evaluator_from_env()? {
      let judge = VendorModel::new(vendors);
      builder = builder.evaluator(Arc::new(LlmEvaluator::new(Arc::new(judge))));
  }
  ```
- **`SessionStore`** — SQLite by default (`SqliteStore`), any backend by
  trait.
- **`ChatModel`** — OpenAI-compatible vendor failover with streaming by
  default (`VendorModel`), any backend by trait. Context capacity comes from
  the model.
- **MCP** (feature `mcp`, on by default) — `McpServers::connect` discovers the
  tools of local (stdio) and remote (streamable HTTP, header auth) servers as
  `<server>_<tool>`, each connecting its server once per run scope; register
  them with `tools_shared`. See [mcp.md](mcp.md).
- **`execution_policy`** — an async `ToolPolicy` hook before every tool call,
  inherited by subagents; per-call approval lives there. See
  [security.md](security.md).
