# HTTP API

`mason serve` exposes a session API. Every request needs
`Authorization: Bearer <token>` — including SSE — except `GET /healthz`,
which reports the binary's version so applications can enforce a floor at
startup. Use TLS at a reverse proxy for remote access. This is a single-user
agent: sharing its token shares access to all sessions and the configured
workspace. Separate users need separate instances or application-level
authorization.

Embedders get the same layer with `router_with_security`; bare
`router`/`router_with_shutdown` require the embedder's own authentication.

## Endpoints

| Method & path | Purpose |
| --- | --- |
| `GET /healthz` | Version and liveness; no token. |
| `POST /sessions` | Create a session: `title`, `agent` (profile; must exist and not be a subagent profile), opaque JSON `metadata` (stored unread). Returns `{id}`. |
| `GET /sessions` | List sessions; filters `agent`, `parentId`, `limit`, `before` (ISO date). Each entry adds `busy` and `stalled`. |
| `GET /sessions/:id` | One session, including `metadata`, `busy`, `stalled`. |
| `PATCH /sessions/:id` | Sets `title` and/or `metadata` (`null` clears it). |
| `DELETE /sessions/:id` | Removes an idle session with its history and children; 409 while it runs. |
| `POST /sessions/:id/messages` | Submit an input; see below. |
| `GET /sessions/:id/inputs` | The inbox with keys and consumption times; paged with `after`/`limit`. |
| `DELETE /sessions/:id/inputs/:input` | Withdraw an input that has not entered the conversation. |
| `POST /sessions/:id/interrupt` | Interrupt the session's current run (204). |
| `POST /sessions/:id/resume` | Drain a stalled session's queued inputs again. |
| `GET /sessions/:id/events` | Event history as JSON, or SSE with `Accept: text/event-stream`; see below. |

## Messages

`POST /sessions/:id/messages` takes `content` (required, capped at 128 KiB),
plus:

- `delivery` — `queue` (default) starts a run after the current one ends;
  `steer` joins the current run at its next step.
- `source` — who sent it; defaults to `human`.
- `key` — idempotency key; a repeated `(session, key)` returns the original
  input, and idempotent retries remain accepted at capacity.

Accepted submissions return 202 with `{inputId}`. Submissions exceeding
admission capacity return 429 before storing a new input.

## Events

Committed events are the source of truth and carry monotonic IDs: `message`
(`{message, turn}`), `run.started`, `run.finished` (`{answer}`),
`run.interrupted`, `run.failed` (`{error}`), `subagent.started`
(`{sessionId, agent, description}`) and, with an evaluator configured,
`run.evaluated` (`{outcome, reason}`, outcome `success`, `partial`, `failure`
or `unclear`). `run.evaluated` follows its run's terminal event and carries
its `runId`; a client that stops reading at the terminal event misses it.

`GET /sessions/:id/events` returns a JSON page (`after` cursor, `limit`
default 500, max 1000) or, with `Accept: text/event-stream`, an SSE stream:

- Clients resume with `Last-Event-ID`; an explicit `after` overrides it.
- Live deltas stream as transient frames without IDs: `delta`, `committed`
  and `tool`. They are never replayed.
- A viewer that falls behind the broadcast window is replayed committed
  events from its cursor, then closed for a fresh subscription.
- A heartbeat comment is sent every 15 s.

## Session lifecycle

- **Admission**: the server defaults to 4 active roots and 128 admitted
  roots (`--max-active-runs`, `--max-pending-sessions`). SQLite inboxes cap
  pending inputs at 100 per session and 1,024 globally.
- **Interrupts**: an interrupt stops the current run only; queued inputs,
  including one posted while the interrupted run is still cleaning up, start
  the next run.
- **Stalls**: a failed run stops draining its session — remaining inputs wait
  for the next message, so a persistent failure (a bad prompt file, a
  provider outage) cannot consume the whole inbox. Such a session reports
  `"stalled": true`; `POST /sessions/:id/resume` drains it again, as does a
  server restart.
- **Child sessions**: stored subagents report busy state and can be
  interrupted independently. The parent's `subagent.started` event names the
  child so viewers can stream or interrupt it while it runs;
  `GET /sessions?parentId=:id` lists a session's children. Their work is
  owned by the parent; direct message submission to a child returns 409.
- **Shutdown**: SIGINT/SIGTERM cancel active runs and stop new work
  immediately. HTTP draining and run cleanup proceed concurrently, each with
  a five-second grace period; a slow SSE reader cannot delay tool
  cancellation.
