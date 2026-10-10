# Architecture and runtime limits

What mason guarantees while a run executes. All budgets are configurable
([configuration.md](configuration.md)); defaults in parentheses. The HTTP
surface for interrupts, streaming and shutdown is in [api.md](api.md).

## Steps and fan-out

- 50 tool-using model calls per run by default (150 for the `build` profile,
  capped by explicit host limits) plus one final text-only call.
- 4 concurrent leaf tool calls, shared with children; 16 tool calls and 4
  `task` calls per model reply.
- A doom-loop detector warns on repeated identical calls (3 in the last 10)
  and stops the run after three warnings without a clean step in between. A
  warning adds two short lines on top of the output bound.
- Root and child runs share a 1,000-tool-call budget and a 20,000,000-token
  budget (estimated input plus reported/estimated completion usage; not a
  billing guarantee).
- A run takes its final text-only call when the tool-call budget is spent,
  when the remaining token budget is under twice the next request's estimate,
  or when a step starts in the last 10% of the run timeout; calls over the
  tool-call budget are skipped with an error result.
- Conversation loading queries only the current checkpoint and tail.

## Deadlines and retries

- 15-minute run budget; per model turn a 3-minute budget across all attempts
  and vendors, a 2-minute per-attempt budget and a 60-second stream idle
  timeout.
- 3 attempts per vendor with jittered exponential backoff honouring numeric
  `Retry-After` values; a backoff that would outlast the turn fails over or
  reports its error at once.
- Retries, vendor failover and the turn/attempt budgets apply only while no
  content has been streamed to the consumer (a usage report alone does not
  count). After that only the idle timeout, cancellation and the run budget
  bound a long generation — single-shot vendors deliver nothing early, so
  their whole response must fit the attempt budget.
- Transient provider error events (overloads, rate limits, server errors) and
  connections cut mid-event are retried like transport failures; a turn that
  runs out of time reports the last failed attempt.
- A context-overflow rejection is surfaced at once, without retry or
  failover, so the run can compact.
- A reply cut off at the output limit (`finish_reason=length`) is discarded
  and the step retried with a notice to write smaller pieces (the final
  text-only turn keeps its partial answer); `content_filter` fails the run.
- A connection lost after the finish reason arrived still completes the turn.

## Output bounds and spilling

- Model-visible tool output stays within 2,000 lines / 50 KiB including the
  truncation notice; previews clip lines at 2,000 characters. Failed tool
  output uses the same bounds. Model turns are bounded to 8 MiB across text,
  reasoning and tool-call fragments.
- Full output spills to uniquely named files eligible for cleanup after seven
  days (cleanup on startup/new spills, and hourly in `serve`). Streamed
  capture is capped at 10 MiB per call by default.
- CLI spill storage defaults to `.mason/output` with a 256 MiB quota
  including in-progress captures; a full quota evicts the oldest outputs
  untouched for an hour, and output that still cannot be saved returns
  clipped with a note instead of failing.
- Per tool: `read` pages hold about 2,000 lines or 48 KiB; `grep` skips files
  it cannot read and bounds its line buffer at 64 MiB; `edit` refuses files
  above 16 MiB; `write` and `edit` replace a file atomically (temp file plus
  rename, through a symlink to its target, keeping its mode); `read`, `edit`
  and `write` refuse FIFOs and devices.
- `bash` returns when the command exits. Surviving process groups are owned
  by the session, keep running between its messages, are listed/stopped
  through the `process` tool, and are killed after 30 idle minutes, at
  shutdown, or when a one-shot run (`mason run`, `run_once`) or a subagent's
  run ends. At most 32 background groups are retained; a group whose members
  all exited is forgotten within a second, so a reused id is never
  signalled. Later output is discarded by readers that end when their writers
  exit. Post-kill output draining remains bounded to five seconds.

## Context management

- Tokens are estimated at ~4 UTF-8 bytes/token. Requests are planned against
  context capacity minus a reply reserve (25%, capped at 20k).
- A request over budget first prunes old tool outputs from the sent view: the
  newest `min(40k, request budget / 3)` of tool output and results the model
  has not seen yet are protected, and pruning only applies when it frees more
  than `min(20k, request budget / 6)`. The newest result is protected even
  when it exceeds the scaled threshold.
- Only if pruning was not enough, the pre-tail conversation is compacted into
  an anchored summary. The conversation itself keeps every result, so
  summaries see original content, never placeholders; tool results and
  tool-call arguments are clipped to their first and last 1 KB, and history
  too large for one summarization request is folded in chunks.
- If recent tool results still exceed the request budget after compaction,
  only the sent view is shortened and originals are saved for paging with
  read/grep.
- Reasoning content is sent back only for the current tool-call round
  (assistant messages after the latest user message); earlier reasoning is
  stored but never resent.
- Provider usage is recorded when present but never replaces the estimate for
  the next, modified request.

## Sessions and storage

Sessions are single-process by contract:

- SQLite `BEGIN IMMEDIATE` serialises writers within one process;
  multi-process coordination needs a lease and is out of scope. An exclusive
  OS ownership lock prevents a second store/process from opening the same
  database.
- Startup recovery marks abandoned runs interrupted, keeps consumed inputs
  consumed and repairs unanswered tool calls — side effects are never
  replayed and no exactly-once execution is promised.
- Streaming deltas are transient and never replayed; committed messages are
  the source of truth.

## Memory

With a memory directory (`memory_dir`), the agent keeps lessons that make
future runs faster or more correct: tool quirks, costly environment facts,
user corrections. Every session of the agent shares them.

- Each memory is `<name>.md`: frontmatter with a one-line `description` and an
  `updated` date, then the lesson with **Why:** and **How to apply:** lines.
- Each top-level run lists the memories' descriptions in its system prompt,
  most recently changed first, snapshotted with the skills and shared with its
  subagents; there is no index file to drift. The listing is capped at 8 KB
  (about 2k tokens, resent every turn) and ends with a count of the older
  memories left out, which `view` still lists.
- The runtime-owned `memory` tool views, creates, edits (`str_replace`) and
  deletes (archives) memories. Subagents get a view-only variant. Profiles with an
  explicit tool list must name `memory` to get either.
- One process owns a memory directory: an OS lock on `<memory_dir>/.lock`
  makes a second agent or process fail at build, as the session store does.
  Writes are serialised and replace files atomically. At most 200 memories of
  4 KB each; descriptions are one line of up to 200 characters.
- Each memory records its `source`: `session N`, `reflection on session N`,
  `reflection` (one-shot runs), `consolidation`, or nothing for a one-shot
  run's own saves.

### Reflection

An agent rarely notices its own wasted steps, so finished runs are reviewed
afterwards. Deterministic signals decide whether a run is reviewed at all, so
most runs cost nothing extra:

- at least 3 distinct tool errors (a repeated error counts once);
- one tool was called at least 3 times with identical arguments;
- the run took at least 30 model turns;
- the user sent input while the run was working (stored sessions only);
- the run failed (stored sessions only);
- the evaluator judged it `failure` or `partial` (its reason is passed on). A
  `success` or `unclear` verdict is passed as context when other signals
  trigger a review, never as a trigger.

A run that saved, edited or deleted a memory itself is not reviewed: its own
lesson stands. The runner reflects on a session at most once per 30 minutes.

A flagged run gets one reflection: the reflection prompt, the current memory
index, the signals and the run's transcript (tool results and arguments
clipped to 2 KB, the whole capped at 48 KB with an oversized middle omitted;
the runner reads only that run's events), with the `memory` tool as its
only tool, at most 8 steps and 2 minutes. The runner reflects in the
background after recording `run.finished` or `run.failed`, one reflection at
a time, and shutdown cancels it; interrupted runs are never reviewed. `mason
run` reflects after printing its answer, before exiting. Subagents' own
transcripts are not reviewed.

### Consolidation

Reflection sees one run at a time, so duplicates, contradictions and stale
lessons accumulate. Consolidation reviews all of memory at once. It is due
when at least 30 memories exist and 10 of them changed since the last
consolidation (recorded in `.consolidated`), when 180 exist, or when the index
outgrows its 8 KB prompt budget. Every attempt, failed or not, is recorded in
`.consolidation-attempt` and defers the next by an hour, so a consolidation
that keeps failing is not retried after every run.

The runner checks after each finished or failed run, inside the same
one-at-a-time background task as reflection, and `mason run` checks before
exiting; `Agent::consolidate` runs one on the embedder's schedule. The model
sees every memory and learned skill in full (only the index when that exceeds
64 KB or half the context), with the `memory` and `learned_skill` tools, at most 10
steps and 5 minutes. It merges duplicates, resolves contradictions in favour
of the newer memory, deletes one-off or stale lessons, tightens descriptions
and promotes multi-step procedures to learned skills.

Learned skills live in `<memory_dir>/skills/<name>/SKILL.md` and are listed
and loaded by the `skill` tool like installed skills, whose names they cannot
take; an installed skill wins any clash. A learned skill holds up to 16 KB,
and there are at most 50. Only consolidation writes them.

Every delete, of a memory or a learned skill, moves the file to
`<memory_dir>/.archive/` with a timestamp, so consolidation mistakes can be
undone by hand.

## Run evaluation

Users rarely say whether a run worked, so an evaluator answers in their place.
With one configured (`AI_EVALUATOR_VENDORS`, or `AgentBuilder::evaluator`),
every finished or failed run is graded `success`, `partial`, `failure` or
`unclear` with a one-sentence reason. Interrupted runs are not graded.

- The default `LlmEvaluator` makes one request to its own, typically cheaper,
  model with the run's transcript clipped to 24 KB and a `verdict` tool. Its
  prompt treats the agent's claims as unproven until tool results back them;
  a reply without a valid verdict counts as `unclear`.
- The runner grades in the background right after the terminal event,
  concurrently across sessions, records `run.evaluated` and hands the verdict
  to reflection. `mason run` prints `> evaluated: <outcome> (<reason>)` on
  stderr.
- Evaluation does not need memory: without a memory directory verdicts are
  still recorded, which is enough for success-rate analytics.
- `Evaluator` is a trait: an application can answer with real user feedback
  instead of a model.

The evaluator sees only the run itself; it does not yet use the user's next
message as evidence.

## Source layout

| Path | Contents |
| --- | --- |
| `src/engine.rs` | Step loop |
| `src/llm/` | Wire protocol, SSE, retry/failover |
| `src/tool/` | Framework, bounds, builtins, replace matchers |
| `src/context.rs` | Context management |
| `src/memory.rs` | Persistent agent memory and learned skills |
| `src/reflect.rs` | Reflection signals and transcripts |
| `src/evaluate.rs` | Run evaluation |
| `src/profile.rs` | Profiles |
| `src/skill.rs` | Skills |
| `src/store/` | Session storage (SQLite) |
| `src/runner.rs` | Session runner |
| `src/api.rs` | HTTP API |
| `src/mcp.rs` | MCP client (feature `mcp`) |
| `src/cli.rs` | CLI |
| `prompts/prompts.yaml` | Embedded prompts |
| `migrations/` | SQLite migrations |
| `examples/` | App manifest and MCP fixture server |

## Development

```sh
./etc/check.sh              # fmt + clippy (both feature configs) + tests
./etc/check.sh --coverage   # the same with a cargo-llvm-cov report (optional)
```

Minimum supported Rust version: 1.94 (the `sqlx` floor), recorded in
`Cargo.toml`.
