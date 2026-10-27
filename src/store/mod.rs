//! Session, inbox, and event persistence with a SQLite implementation.

pub mod sqlite;

pub use sqlite::SqliteStore;

use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::broadcast;

use crate::error::Result;
use crate::llm::Message;
use crate::prompt::PromptSet;

pub type SessionId = u64;
/// Event cursor.
pub type EventId = u64;

/// How an input joins the conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Delivery {
    /// Starts a run after the current one ends.
    #[default]
    Queue,
    /// Joins the current run at its next step.
    Steer,
}

impl Delivery {
    fn as_str(&self) -> &'static str {
        match self {
            Delivery::Queue => "queue",
            Delivery::Steer => "steer",
        }
    }

    pub(crate) fn parse(raw: &str) -> Result<Self> {
        match raw {
            "queue" => Ok(Self::Queue),
            "steer" => Ok(Self::Steer),
            other => Err(crate::Error::Run(format!("unknown delivery `{other}`"))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Session {
    pub id: SessionId,
    /// The parent session for subagent children.
    pub parent_id: Option<SessionId>,
    /// The profile this session runs.
    pub agent: String,
    pub title: Option<String>,
    pub created_at: DateTime<Utc>,
    /// Opaque embedder metadata; `Value::Null` when unset.
    pub metadata: Value,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct NewSession {
    pub title: Option<String>,
    /// The parent session for subagent children.
    pub parent_id: Option<SessionId>,
    /// The profile this session runs; `None` is `build`.
    pub agent: Option<String>,
    /// Opaque embedder metadata, [`Value::Null`] when unused. An embedder's
    /// profile selector may read it; the store and the API never do.
    pub metadata: Value,
}

/// Changes to a session; `None` leaves a field unchanged.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SessionUpdate {
    pub title: Option<String>,
    /// Replaces the metadata; [`Value::Null`] clears it.
    pub metadata: Option<Value>,
}

/// A new inbox input.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct NewInput {
    pub content: String,
    /// Queue (default) or steer.
    pub delivery: Delivery,
    /// Who sent it; defaults to `human`.
    pub source: String,
    /// Repeated `(session_id, key)` returns the original input.
    pub key: Option<String>,
}

impl NewInput {
    /// A plain queued human message.
    pub fn queued(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            source: "human".to_owned(),
            ..Default::default()
        }
    }
}

/// A claimed input.
#[derive(Debug, Clone, PartialEq)]
pub struct Input {
    pub id: u64,
    pub source: String,
    pub content: String,
}

/// One inbox row, pending or consumed.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StoredInput {
    pub id: u64,
    /// The idempotency key it was added with.
    pub key: Option<String>,
    pub source: String,
    pub delivery: Delivery,
    pub content: String,
    pub created_at: DateTime<Utc>,
    /// When it entered a conversation; `None` while it is still pending.
    pub consumed_at: Option<DateTime<Utc>>,
    /// The run it entered; `None` while it is still pending.
    pub run_id: Option<String>,
    /// That run's outcome: `running`, `finished`, `failed` or `interrupted`.
    pub outcome: Option<String>,
}

/// Committed event whose ID is the replay cursor.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Event {
    pub id: EventId,
    pub session_id: SessionId,
    /// The run this event belongs to, for run-scoped events.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    /// The event type (`message`, `compaction`, `run.started`,...).
    #[serde(rename = "type")]
    pub event_type: String,
    pub data: Value,
    /// When it was committed.
    pub created_at: DateTime<Utc>,
}

/// `list_sessions` filters.
#[derive(Debug, Clone, Default)]
pub struct SessionQuery {
    /// Only sessions of this profile.
    pub agent: Option<String>,
    /// Children of this session instead of top-level sessions.
    pub parent_id: Option<SessionId>,
    /// Default 50, clamped to 1..=200.
    pub limit: Option<u32>,
    /// Only sessions whose latest activity is before this time.
    pub before: Option<DateTime<Utc>>,
}

/// A session list row. `busy` is supplied by the runner, not the store.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionSummary {
    pub id: SessionId,
    /// The parent session for subagent children.
    pub parent_id: Option<SessionId>,
    pub agent: String,
    pub title: Option<String>,
    pub created_at: DateTime<Utc>,
    /// Latest event time, when there are events.
    pub last_event_at: Option<DateTime<Utc>>,
    /// Latest event type, when there are events.
    pub last_event_type: Option<String>,
    /// Opaque embedder metadata; `Value::Null` when unset.
    pub metadata: Value,
}

/// Atomically claimed inputs and a run ID, absent when no inputs were claimed.
#[derive(Debug, Clone, PartialEq)]
pub struct RunClaim {
    /// The claimed inputs, steers first then the queued one, by id.
    pub inputs: Vec<Input>,
    /// The run id, or `None` when nothing was claimed.
    pub run_id: Option<String>,
}

/// Conversation since the latest compaction.
#[derive(Debug, Clone, PartialEq)]
pub struct Conversation {
    /// The latest compaction summary, when there is one.
    pub summary: Option<String>,
    /// The kept message events: those at or after the compaction's tail,
    /// the latest user message wherever it sits, and the user message the
    /// compaction kept verbatim until a later one summarises it.
    pub events: Vec<Event>,
}

/// What startup recovery marked interrupted.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Recovery {
    /// `(session_id, run_id)` pairs repaired by this call.
    pub interrupted: Vec<(SessionId, String)>,
}

/// The error recorded on a run the process restarted out of.
pub const RESTART_INTERRUPTED: &str = "the process restarted while this run was active";

/// Session persistence backend. Coordination is single-process; multi-process use
/// requires external leasing.
#[async_trait::async_trait]
pub trait SessionStore: Send + Sync {
    /// Creates a session; `agent` defaults to `build`.
    async fn create_session(&self, new: NewSession) -> Result<SessionId>;

    /// One session, or `None`.
    async fn get_session(&self, id: SessionId) -> Result<Option<Session>>;

    /// Updates a session's title or metadata; `false` when it does not exist.
    async fn update_session(&self, id: SessionId, update: SessionUpdate) -> Result<bool>;

    /// Deletes a session with its inbox, events and child sessions; `false`
    /// when it does not exist.
    async fn delete_session(&self, id: SessionId) -> Result<bool>;

    /// Adds an input to the inbox; a repeated `(session_id, key)` returns
    /// the earlier input's id.
    async fn add_input(&self, session_id: SessionId, input: NewInput) -> Result<u64>;

    /// Removes an input nobody has consumed yet; `false` when there is no such
    /// pending input.
    async fn withdraw_input(&self, session_id: SessionId, input_id: u64) -> Result<bool>;

    /// The session's inbox in insertion order, pending and consumed alike.
    async fn list_inputs(&self, session_id: SessionId) -> Result<Vec<StoredInput>>;

    /// Bounded inbox page. Backends can override this to page in storage.
    async fn inputs_page(
        &self,
        session_id: SessionId,
        after: u64,
        limit: u32,
    ) -> Result<Vec<StoredInput>> {
        Ok(self
            .list_inputs(session_id)
            .await?
            .into_iter()
            .filter(|input| input.id > after)
            .take(limit as usize)
            .collect())
    }

    /// Claims and returns pending inputs of one delivery kind. When
    /// `run_id` is given, the claimed inputs' message events are stamped
    /// with it (steering joins the live run).
    async fn take_inputs(
        &self,
        session_id: SessionId,
        delivery: Delivery,
        limit: u32,
        run_id: Option<&str>,
    ) -> Result<Vec<Input>>;

    /// Atomically claims all steers and the oldest queued input with their message and
    /// `run.started` events.
    async fn start_run(&self, session_id: SessionId) -> Result<RunClaim>;

    /// Sessions with inputs nobody has consumed yet.
    async fn pending_sessions(&self) -> Result<Vec<SessionId>>;

    /// Top-level sessions (or one session's children), most recently active
    /// first.
    async fn list_sessions(&self, query: &SessionQuery) -> Result<Vec<SessionSummary>>;

    /// Appends and publishes one event.
    async fn append(&self, session_id: SessionId, event_type: &str, data: Value) -> Result<Event>;

    /// Appends a run-scoped event. Terminal types are enforced unique per
    /// run by the schema.
    async fn append_run_event(
        &self,
        session_id: SessionId,
        run_id: &str,
        event_type: &str,
        data: Value,
    ) -> Result<Event>;

    /// Committed events after the cursor, in cursor order.
    async fn list_events(&self, session_id: SessionId, after: EventId) -> Result<Vec<Event>>;

    /// At most `limit` committed events after the cursor, in cursor order.
    async fn events_page(
        &self,
        session_id: SessionId,
        after: EventId,
        limit: u32,
    ) -> Result<Vec<Event>>;

    /// One run's committed events in cursor order. Backends can override this to
    /// avoid reading the whole session.
    async fn run_events(&self, session_id: SessionId, run_id: &str) -> Result<Vec<Event>> {
        Ok(self
            .list_events(session_id, 0)
            .await?
            .into_iter()
            .filter(|event| event.run_id.as_deref() == Some(run_id))
            .collect())
    }

    /// Post-compaction history, always preserving the latest user message.
    async fn conversation(&self, session_id: SessionId) -> Result<Conversation>;

    /// Checkpoint and kept messages, with unanswered tool calls repaired by interrupted
    /// results.
    async fn history(&self, session_id: SessionId, prompts: &PromptSet) -> Result<Vec<Message>>;

    /// Committed events in cursor order, excluding events published before
    /// subscription.
    fn subscribe(&self, session_id: SessionId) -> broadcast::Receiver<Arc<Event>>;

    /// Idempotent startup recovery. Consumed inputs stay consumed; abandoned tool
    /// effects are never replayed.
    async fn recover(&self) -> Result<Recovery>;
}
