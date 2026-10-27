# MCP

MCP support (feature `mcp`, on by default) connects local (stdio) and remote
(streamable HTTP, header auth) servers and mounts their tools as
`<server>_<tool>`. It is off until explicitly granted with `--allow-mcp`
(`allow_mcp: true` in a manifest, `MASON_ALLOW_MCP=true`); the grant fails
closed.

## Server map

Servers are configured in an opencode-format JSON file mapping names to
configs, passed with `--mcp` (which replaces a manifest's servers wholesale)
or inline in a manifest's `mcp:` key:

```json
{
    "playwright": {
        "type": "local",
        "command": ["npx", "-y", "@playwright/mcp@latest", "--headless"],
        "timeout": 120000
    }
}
```

`{env:NAME}` in a server's `command`, `environment`, `url` or `headers`
expands from mason's environment, so credentials stay out of config files; an
unset variable fails startup.

Inspect connections and discovered tools without an LLM call:

```sh
cargo run -- mcp --mcp mcp.json --allow-mcp
```

## Connection lifecycle

- Tools are discovered at startup, which validates profiles, and listed again
  at the start of every top-level run (its subagents share that list), so a
  server's new or changed tools reach the next run without a restart. A
  listing that fails, or that an interrupt ends, keeps the startup tools; each
  server's `timeout` bounds it, outside the run's deadline. A session's server
  process whose tools differ from the run's listing is restarted as the run
  starts, before any call; mid-run, only a server that exited is restarted.
- Each session connects its own server processes, kept across its messages
  until the session has been idle for 30 minutes or the server shuts down; a
  subagent's connections end with its run.
- Calls to one server run concurrently. A server that exits or crashes is
  started again on its next call.
- A server's `timeout` (default 30 s) bounds connecting and listing tools;
  calls run until they finish or the run is interrupted or times out.
- Configured connection failures stop CLI startup with diagnostics.
- Remote MCP servers need to provide their own state isolation.
- A cancelled or timed-out call may have already applied effects remotely —
  cancellation is not rollback.

## Sandbox interaction

Local MCP subprocesses use the selected execution mode
([security.md](security.md)). In sandbox mode:

- Installing packages or using a browser needs `--allow-network`.
- The server is best installed into the workspace.
- Servers needing a browser profile or a runtime under your home directory
  need trusted execution.
- For Playwright, include `--isolated` in its command when using independent
  sessions.
