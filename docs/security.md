# Security

The standalone CLI enforces workspace/process isolation and API protection;
embedders own the security boundary of their application.

## Execution modes

The CLI defaults to trusted execution: tools have the same host access as the
user running mason. `--execution-mode sandbox` opts into OS confinement:
macOS uses `sandbox-exec` and Linux needs `bubblewrap` (`bwrap`) with user
namespaces enabled. File operations also run in isolated workers.

In sandbox mode:

- Tools can read and write the workspace (including any secrets kept there)
  and read system directories (`/usr`, `/bin`, `/opt`, `/Library`, Xcode and
  similar); the rest of the host, including your home directory, is
  invisible.
- `.mason` state is hidden except spilled output under `.mason/output`, which
  is read-only.
- Subprocesses get a temporary HOME, a minimal environment (PATH, LANG and
  proxy/CA variables, never provider/API credentials), no controlling
  terminal, and on macOS no access to LaunchServices, Apple Events or the
  pasteboard.
- `--allow-network` grants network access to subprocesses; the model client
  connects to its configured endpoint independently.
- Unsupported or unavailable sandboxes fail closed.

Supported CLI platforms are macOS and Linux; Windows process-tree isolation
is not implemented.

SDK defaults remain trusted for existing custom tools; sandboxing arbitrary
domain tools requires an OS boundary supplied by the embedder.
`AgentBuilder::execution_policy` installs an async `ToolPolicy` hook before
every tool call, inherited by subagents — per-call approval lives there.

## HTTP access

`serve` requires a secret bearer token on every request except `/healthz`.
`router_with_security` provides the same layer for embedders; bare
`router`/`router_with_shutdown` require the embedder's own authentication.
One token grants all sessions; separate users need separate instances or
application-level authorization. See [api.md](api.md).

## Private persistence

New SQLite state directories use 0700 and the DB, WAL, SHM and ownership lock
use 0600 on Unix. Existing directories are not chmodded; existing DB/sidecar
permissions are repaired after ownership is acquired. Session history
persists until the owner removes or archives it.

Memory directories are created 0700 and memories written 0600 on Unix;
memories that are symlinks or not regular files are neither read nor
replaced.

## Spilled tool output

CLI outputs live in private `.mason/output`; SDK defaults remain
`<tmp>/mason-<uid>`. Directories use 0700, files use 0600, symlink
directories are rejected, and quota reservations serialize through an OS
lock. Retention cleanup is best effort; embedders may choose a private
location, quota and retention through `OutputSpill`.

## Known gaps

- **Process cleanup.** Unix cleanup targets process groups. Deliberately
  detached processes can escape group cancellation; OS sandbox reach still
  applies. Windows process-tree cleanup/isolation is not implemented.
- **Fetches may reach private networks.** Remote MCP servers and any
  domain-registered fetch tools resolve and connect from the host; there is
  no SSRF filtering in the core.
- **Memory is shared and can be poisoned.** Every session of an agent reads
  the same memories, so a lesson saved in one user's session reaches all
  others, and text the model was tricked into saving persists across runs.
  Reflection reads clipped tool output from the reviewed run. The prompts
  forbid saving personal data and copying tool output; that is model-level
  guidance, not enforcement.
- **Evaluation sends transcripts to the evaluator's provider.** With
  `AI_EVALUATOR_VENDORS` on another provider than the main model, every
  graded run's clipped transcript goes there too.
- **Prompt injection is a model-level defence only.** System prompts instruct
  the model to treat file contents, command output and web pages as untrusted
  evidence; that is mitigation, not isolation.
- **MCP cancellation is not rollback.** A cancelled or timed-out MCP call may
  have already applied effects remotely.
