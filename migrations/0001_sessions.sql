-- Session, inbox, and event tables. Coordination is single-process.

CREATE TABLE session (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_id  INTEGER REFERENCES session(id) ON DELETE CASCADE,
    agent      TEXT NOT NULL DEFAULT 'build',
    title      TEXT,
    created_at INTEGER NOT NULL,              -- unix milliseconds, UTC
    metadata   TEXT                           -- opaque embedder metadata for its profile selector
);

-- Queued inputs start the next run; steering joins the current run's next step.
CREATE TABLE input (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES session(id) ON DELETE CASCADE,
    key         TEXT,                          -- idempotency key
    source      TEXT NOT NULL DEFAULT 'human',
    delivery    TEXT NOT NULL DEFAULT 'queue' CHECK (delivery IN ('queue', 'steer')),
    content     TEXT NOT NULL,
    created_at  INTEGER NOT NULL,
    consumed_at INTEGER,                       -- when it entered the conversation
    UNIQUE (session_id, key)                   -- NULL keys never conflict
);

CREATE INDEX idx_input_pending ON input (session_id, id) WHERE consumed_at IS NULL;

-- Committed messages, compactions, and run status; IDs are client replay cursors.
CREATE TABLE event (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES session(id) ON DELETE CASCADE,
    run_id     TEXT,                           -- links run.started to its terminal event
    type       TEXT NOT NULL,
    data       TEXT NOT NULL,
    created_at INTEGER NOT NULL
);

CREATE INDEX idx_event_session ON event (session_id, id);
CREATE INDEX idx_event_run ON event (run_id) WHERE run_id IS NOT NULL;

-- One terminal event per run, enforced by the database.
CREATE UNIQUE INDEX uq_event_terminal_run
    ON event (run_id)
    WHERE type IN ('run.finished', 'run.failed', 'run.interrupted');
