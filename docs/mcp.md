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

- Tools are discovered at startup to validate profiles. Each top-level run
  refreshes servers whose namespaces its profile could use; its subagents share
  that listing. Exact tool lists and namespace-wide denials skip unrelated
  servers; globs conservatively include namespaces that could gain matching tools.
- Refresh lists tools over the session's existing connections, opening missing
  connections once for both listing and calls. Up to eight servers refresh
  concurrently within one budget: at most five seconds or 10% of the run timeout,
  whichever is smaller. Listing time counts toward the run timeout. Interrupts
  cancel discovery too.
- Each failed or unfinished server keeps its last successful listing; successful
  updates from other servers are retained. Failures and timeouts are logged.
  A cached tool can still fail when called if its server remains unavailable.
- Changes a running server reports through `tools/list` reach the next run.
  Updating a local server's executable requires restarting its process (for
  example by restarting `mason serve`); refresh does not reload executable code
  or discard a live server's in-memory state.
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
