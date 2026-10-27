//! SQLite session persistence. Per-session append locks publish committed events in
//! cursor order.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde_json::{Value, json};
use sqlx::sqlite::{
    SqliteConnectOptions, SqliteJournalMode, SqlitePool, SqlitePoolOptions, SqliteRow,
};
use sqlx::{Row, Transaction};
use tokio::sync::broadcast;

use crate::Error;
use crate::error::Result;
use crate::llm::Message;
use crate::prompt::PromptSet;
use crate::store::{
    Conversation, Delivery, Event, EventId, Input, NewInput, NewSession, Recovery, RunClaim,
    Session, SessionId, SessionQuery, SessionStore, SessionSummary, SessionUpdate, StoredInput,
};

/// Result used to repair unanswered tool calls without replaying them.
pub const INTERRUPTED_RESULT: &str = "[Tool execution was interrupted]";
/// Broadcast buffer; lagging subscribers replay committed events from the store.
const EVENT_BUFFER: usize = 256;

/// Injects event-insert failures to test rollback.
type FailureHook = Arc<dyn Fn(&str) -> bool + Send + Sync>;

struct Channel {
    sender: broadcast::Sender<Arc<Event>>,
    /// Keeps insert and publication order aligned per session.
    append: tokio::sync::Mutex<()>,
}

/// The SQLite-backed [`SessionStore`].
pub struct SqliteStore {
    pool: SqlitePool,
    // Own the OS lock for the entire store lifetime, including recovery.
    _owner: std::fs::File,
    channels: StdMutex<HashMap<SessionId, Arc<Channel>>>,
    failure_hook: StdMutex<Option<FailureHook>>,
}

impl SqliteStore {
    /// Creates or opens the database and applies migrations.
    pub async fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            let mut builder = std::fs::DirBuilder::new();
            builder.recursive(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::DirBuilderExt;
                builder.mode(0o700);
            }
            builder.create(parent)?;
        }
        let mut options = std::fs::OpenOptions::new();
        options.read(true).write(true).create(true).truncate(false);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options
                .mode(0o600)
                .custom_flags(nix::fcntl::OFlag::O_NOFOLLOW.bits());
        }
        let database = options.open(path)?;
        if !database.metadata()?.is_file() {
            return Err(Error::Config(
                "session database must be a regular file".into(),
            ));
        }
        // Use a dedicated inode: SQLite owns its own byte-range locks on the DB.
        let mut lock_path = path.as_os_str().to_owned();
        lock_path.push(".mason-lock");
        let owner = options.open(Path::new(&lock_path))?;
        owner.try_lock().map_err(|_| {
            Error::Config("the session database is already owned by another Mason process".into())
        })?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            owner.set_permissions(std::fs::Permissions::from_mode(0o600))?;
            database.set_permissions(std::fs::Permissions::from_mode(0o600))?;
            // Repair sidecars from an older release before opening SQLite.
            for suffix in ["-wal", "-shm"] {
                let mut sidecar = path.as_os_str().to_owned();
                sidecar.push(suffix);
                let sidecar = Path::new(&sidecar);
                if let Ok(meta) = std::fs::symlink_metadata(sidecar) {
                    if !meta.is_file() || meta.file_type().is_symlink() {
                        return Err(Error::Config("unsafe SQLite sidecar path".into()));
                    }
                    std::fs::set_permissions(sidecar, std::fs::Permissions::from_mode(0o600))?;
                }
            }
        }
        let options = SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(true)
            .journal_mode(SqliteJournalMode::Wal)
            .busy_timeout(Duration::from_secs(5))
            .foreign_keys(true);
        let pool = SqlitePoolOptions::new()
            .max_connections(8)
            .connect_with(options)
            .await?;
        sqlx::migrate!("./migrations")
            .run(&pool)
            .await
            .map_err(|err| {
                Error::Run(format!("cannot apply the session-store migrations: {err}"))
            })?;
        Ok(Self {
            pool,
            _owner: owner,
            channels: StdMutex::new(HashMap::new()),
            failure_hook: StdMutex::new(None),
        })
    }

    /// Installs the failure hook (test seam).
    pub fn set_failure_hook(&self, hook: impl Fn(&str) -> bool + Send + Sync + 'static) {
        *self
            .failure_hook
            .lock()
            .expect("the hook mutex is poisoned") = Some(Arc::new(hook));
    }

    pub fn clear_failure_hook(&self) {
        *self
            .failure_hook
            .lock()
            .expect("the hook mutex is poisoned") = None;
    }

    fn check_hook(&self, event_type: &str) -> Result<()> {
        let hook = self
            .failure_hook
            .lock()
            .expect("the hook mutex is poisoned")
            .clone();
        if let Some(hook) = hook
            && hook(event_type)
        {
            return Err(Error::Run(format!(
                "injected failure for `{event_type}` events"
            )));
        }
        Ok(())
    }

    fn channel(&self, session_id: SessionId) -> Arc<Channel> {
        let mut channels = self.channels.lock().expect("the channel mutex is poisoned");
        if channels.len() > 64 {
            // Keep channels with viewers or in-flight appends (other Arc holders).
            channels.retain(|_, channel| {
                Arc::strong_count(channel) > 1 || channel.sender.receiver_count() > 0
            });
        }
        channels
            .entry(session_id)
            .or_insert_with(|| {
                Arc::new(Channel {
                    sender: broadcast::channel(EVENT_BUFFER).0,
                    append: tokio::sync::Mutex::new(()),
                })
            })
            .clone()
    }

    fn publish(channel: &Channel, event: &Event) {
        // A lagging or absent viewer never fails a committed append.
        let _ = channel.sender.send(Arc::new(event.clone()));
    }
}

fn now_millis() -> i64 {
    Utc::now().timestamp_millis()
}

fn timestamp(millis: i64) -> DateTime<Utc> {
    DateTime::from_timestamp_millis(millis).unwrap_or_default()
}

fn event_from_row(row: &SqliteRow) -> Result<Event> {
    let data: String = row.try_get("data")?;
    Ok(Event {
        id: row.try_get::<i64, _>("id")? as EventId,
        session_id: row.try_get::<i64, _>("session_id")? as SessionId,
        run_id: row.try_get("run_id")?,
        event_type: row.try_get("type")?,
        data: serde_json::from_str(&data)
            .map_err(|err| Error::Run(format!("an event payload is not valid JSON: {err}")))?,
        created_at: timestamp(row.try_get::<i64, _>("created_at")?),
    })
}

fn metadata(raw: Option<String>) -> Result<Value> {
    match raw {
        None => Ok(Value::Null),
        Some(text) => serde_json::from_str(&text)
            .map_err(|err| Error::Run(format!("session metadata is not valid JSON: {err}"))),
    }
}

fn session_from_row(row: SqliteRow) -> Result<Session> {
    Ok(Session {
        id: row.try_get::<i64, _>("id")? as SessionId,
        parent_id: row
            .try_get::<Option<i64>, _>("parent_id")?
            .map(|id| id as SessionId),
        agent: row.try_get("agent")?,
        title: row.try_get("title")?,
        created_at: timestamp(row.try_get::<i64, _>("created_at")?),
        metadata: metadata(row.try_get::<Option<String>, _>("metadata")?)?,
    })
}

async fn insert_event(
    tx: &mut Transaction<'static, sqlx::Sqlite>,
    session_id: SessionId,
    run_id: Option<&str>,
    event_type: &str,
    data: Value,
    created_at: i64,
) -> Result<Event> {
    let query = sqlx::query(
        "INSERT INTO event (session_id, run_id, type, data, created_at) VALUES (?, ?, ?, ?, ?)
         RETURNING id, session_id, run_id, type, data, created_at",
    )
    .bind(session_id as i64)
    .bind(run_id)
    .bind(event_type)
    .bind(data.to_string())
    .bind(created_at);
    let row = match query.fetch_one(&mut **tx).await {
        Ok(row) => row,
        Err(err) => {
            // Duplicate terminal events are rejected by the partial unique index.
            if matches!(err, sqlx::Error::Database(_))
                && err.to_string().contains("UNIQUE constraint failed")
            {
                return Err(Error::Run(format!(
                    "run {} already has its terminal event",
                    run_id.unwrap_or_default()
                )));
            }
            return Err(err.into());
        }
    };
    event_from_row(&row)
}

#[async_trait::async_trait]
impl SessionStore for SqliteStore {
    async fn create_session(&self, new: NewSession) -> Result<SessionId> {
        let metadata = (!new.metadata.is_null()).then(|| new.metadata.to_string());
        let row = sqlx::query(
            "INSERT INTO session (parent_id, agent, title, created_at, metadata)
             VALUES (?, ?, ?, ?, ?) RETURNING id",
        )
        .bind(new.parent_id.map(|id| id as i64))
        .bind(new.agent.unwrap_or_else(|| "build".to_owned()))
        .bind(new.title)
        .bind(now_millis())
        .bind(metadata)
        .fetch_one(&self.pool)
        .await?;
        Ok(row.try_get::<i64, _>("id")? as SessionId)
    }

    async fn get_session(&self, id: SessionId) -> Result<Option<Session>> {
        let row = sqlx::query(
            "SELECT id, parent_id, agent, title, created_at, metadata FROM session WHERE id = ?",
        )
        .bind(id as i64)
        .fetch_optional(&self.pool)
        .await?;
        row.map(session_from_row).transpose()
    }

    async fn update_session(&self, id: SessionId, update: SessionUpdate) -> Result<bool> {
        let metadata = update
            .metadata
            .as_ref()
            .map(|metadata| (!metadata.is_null()).then(|| metadata.to_string()));
        let result = sqlx::query(
            "UPDATE session
                SET title = CASE WHEN ? THEN ? ELSE title END,
                    metadata = CASE WHEN ? THEN ? ELSE metadata END
              WHERE id = ?",
        )
        .bind(update.title.is_some())
        .bind(update.title)
        .bind(metadata.is_some())
        .bind(metadata.flatten())
        .bind(id as i64)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected() > 0)
    }

    async fn delete_session(&self, id: SessionId) -> Result<bool> {
        let result = sqlx::query("DELETE FROM session WHERE id = ?")
            .bind(id as i64)
            .execute(&self.pool)
            .await?;
        Ok(result.rows_affected() > 0)
    }

    async fn withdraw_input(&self, session_id: SessionId, input_id: u64) -> Result<bool> {
        let result = sqlx::query(
            "DELETE FROM input WHERE id = ? AND session_id = ? AND consumed_at IS NULL",
        )
        .bind(input_id as i64)
        .bind(session_id as i64)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected() > 0)
    }

    async fn add_input(&self, session_id: SessionId, input: NewInput) -> Result<u64> {
        let source = if input.source.is_empty() {
            "human"
        } else {
            &input.source
        };
        // Return the original input ID for a repeated key without replacing its
        // content.
        let mut tx = self.pool.begin_with("BEGIN IMMEDIATE").await?;
        if let Some(key) = &input.key
            && let Some(id) = sqlx::query_scalar::<_, i64>(
                "SELECT id FROM input WHERE session_id = ? AND key = ?",
            )
            .bind(session_id as i64)
            .bind(key)
            .fetch_optional(&mut *tx)
            .await?
        {
            tx.commit().await?;
            return Ok(id as u64);
        }
        let pending = sqlx::query_scalar::<_, i64>(
            "SELECT COUNT(*) FROM input WHERE session_id = ? AND consumed_at IS NULL",
        )
        .bind(session_id as i64)
        .fetch_one(&mut *tx)
        .await?;
        let total =
            sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM input WHERE consumed_at IS NULL")
                .fetch_one(&mut *tx)
                .await?;
        if pending >= 100 || total >= 1024 {
            return Err(Error::Busy);
        }
        let row = sqlx::query(
            "INSERT INTO input (session_id, key, source, delivery, content, created_at)
             VALUES (?, ?, ?, ?, ?, ?)
             ON CONFLICT (session_id, key) DO UPDATE SET key = excluded.key
             RETURNING id",
        )
        .bind(session_id as i64)
        .bind(input.key)
        .bind(source)
        .bind(input.delivery.as_str())
        .bind(input.content)
        .bind(now_millis())
        .fetch_one(&mut *tx)
        .await?;
        let id = row.try_get::<i64, _>("id")? as u64;
        tx.commit().await?;
        Ok(id)
    }

    async fn list_inputs(&self, session_id: SessionId) -> Result<Vec<StoredInput>> {
        self.inputs_page(session_id, 0, u32::MAX).await
    }

    async fn inputs_page(
        &self,
        session_id: SessionId,
        after: u64,
        limit: u32,
    ) -> Result<Vec<StoredInput>> {
        if after > i64::MAX as u64 {
            return Ok(Vec::new());
        }
        // An input enters its run as a message event naming it; the run's
        // terminal event, if any, is its outcome.
        let rows = sqlx::query(
            "SELECT i.id, i.key, i.source, i.delivery, i.content, i.created_at, i.consumed_at, r.run_id,
                    CASE WHEN r.run_id IS NULL THEN NULL
                         ELSE COALESCE((SELECT substr(t.type, 5) FROM event t
                                         WHERE t.run_id = r.run_id
                                           AND t.type IN ('run.finished', 'run.failed', 'run.interrupted')
                                         ORDER BY t.id LIMIT 1), 'running') END AS outcome
               FROM input i
               LEFT JOIN (SELECT json_extract(e.data, '$.inputId') AS input_id, MIN(e.run_id) AS run_id
                            FROM event e
                           WHERE e.session_id = ? AND e.type = 'message' AND e.run_id IS NOT NULL
                             AND json_extract(e.data, '$.inputId') IS NOT NULL
                           GROUP BY input_id) r ON r.input_id = i.id
              WHERE i.session_id = ? AND i.id > ? ORDER BY i.id LIMIT ?",
        )
        .bind(session_id as i64)
        .bind(session_id as i64)
        .bind(after as i64)
        .bind(i64::from(limit))
        .fetch_all(&self.pool)
        .await?;
        rows.iter()
            .map(|row| -> Result<StoredInput> {
                Ok(StoredInput {
                    id: row.try_get::<i64, _>("id")? as u64,
                    key: row.try_get("key")?,
                    source: row.try_get("source")?,
                    delivery: Delivery::parse(&row.try_get::<String, _>("delivery")?)?,
                    content: row.try_get("content")?,
                    created_at: timestamp(row.try_get::<i64, _>("created_at")?),
                    consumed_at: row.try_get::<Option<i64>, _>("consumed_at")?.map(timestamp),
                    run_id: row.try_get("run_id")?,
                    outcome: row.try_get("outcome")?,
                })
            })
            .collect()
    }

    async fn take_inputs(
        &self,
        session_id: SessionId,
        delivery: Delivery,
        limit: u32,
        run_id: Option<&str>,
    ) -> Result<Vec<Input>> {
        let claim = self
            .deliver(session_id, &[(delivery, limit)], false, run_id)
            .await?;
        Ok(claim.inputs)
    }

    async fn start_run(&self, session_id: SessionId) -> Result<RunClaim> {
        // Claim steers and the oldest queued input atomically with their run events.
        self.deliver(
            session_id,
            &[(Delivery::Steer, 1000), (Delivery::Queue, 1)],
            true,
            None,
        )
        .await
    }

    async fn pending_sessions(&self) -> Result<Vec<SessionId>> {
        let rows = sqlx::query("SELECT DISTINCT session_id FROM input WHERE consumed_at IS NULL")
            .fetch_all(&self.pool)
            .await?;
        Ok(rows
            .iter()
            .map(|row| row.get::<i64, _>("session_id") as SessionId)
            .collect())
    }

    async fn list_sessions(&self, query: &SessionQuery) -> Result<Vec<SessionSummary>> {
        // Dynamic SQL uses only constant fragments; all values are bound.
        // Event ids are monotonic per session, so the newest id carries the latest
        // time and the (session_id, id) index answers it without a scan.
        const LAST_EVENT_AT: &str = "(SELECT e.created_at FROM event e WHERE e.session_id = s.id ORDER BY e.id DESC LIMIT 1)";
        let activity = format!("COALESCE({LAST_EVENT_AT}, s.created_at)");
        const SELECT: &str =
            "SELECT s.id, s.parent_id, s.agent, s.title, s.created_at, s.metadata,
                    (SELECT e.created_at FROM event e WHERE e.session_id = s.id ORDER BY e.id DESC LIMIT 1) AS last_event_at,
                    (SELECT e.type FROM event e WHERE e.session_id = s.id ORDER BY e.id DESC LIMIT 1) AS last_event_type
               FROM session s
              WHERE ";
        let mut sql = String::from(SELECT);
        sql.push_str(if query.parent_id.is_some() {
            "s.parent_id = ?"
        } else {
            "s.parent_id IS NULL"
        });
        if query.agent.is_some() {
            sql.push_str(" AND s.agent = ?");
        }
        if query.before.is_some() {
            sql.push_str(&format!(" AND {activity} < ?"));
        }
        sql.push_str(&format!(" ORDER BY {activity} DESC, s.id DESC LIMIT ?"));

        let mut statement = sqlx::query(sqlx::AssertSqlSafe(sql));
        if let Some(parent) = query.parent_id {
            statement = statement.bind(parent as i64);
        }
        if let Some(agent) = &query.agent {
            statement = statement.bind(agent.clone());
        }
        if let Some(before) = query.before {
            statement = statement.bind(before.timestamp_millis());
        }
        let limit = query.limit.unwrap_or(50).clamp(1, 200) as i64;
        statement = statement.bind(limit);

        let rows = statement.fetch_all(&self.pool).await?;
        rows.iter()
            .map(|row| -> Result<SessionSummary> {
                Ok(SessionSummary {
                    id: row.try_get::<i64, _>("id")? as SessionId,
                    parent_id: row
                        .try_get::<Option<i64>, _>("parent_id")?
                        .map(|id| id as SessionId),
                    agent: row.try_get("agent")?,
                    title: row.try_get("title")?,
                    created_at: timestamp(row.try_get::<i64, _>("created_at")?),
                    last_event_at: row
                        .try_get::<Option<i64>, _>("last_event_at")?
                        .map(timestamp),
                    last_event_type: row.try_get("last_event_type")?,
                    metadata: metadata(row.try_get::<Option<String>, _>("metadata")?)?,
                })
            })
            .collect()
    }

    async fn append(&self, session_id: SessionId, event_type: &str, data: Value) -> Result<Event> {
        let channel = self.channel(session_id);
        let _order = channel.append.lock().await;
        self.check_hook(event_type)?;
        let mut tx = self.pool.begin().await?;
        let event = insert_event(&mut tx, session_id, None, event_type, data, now_millis()).await?;
        tx.commit().await?;
        Self::publish(&channel, &event);
        Ok(event)
    }

    async fn append_run_event(
        &self,
        session_id: SessionId,
        run_id: &str,
        event_type: &str,
        data: Value,
    ) -> Result<Event> {
        let channel = self.channel(session_id);
        let _order = channel.append.lock().await;
        self.check_hook(event_type)?;
        let mut tx = self.pool.begin().await?;
        let event = insert_event(
            &mut tx,
            session_id,
            Some(run_id),
            event_type,
            data,
            now_millis(),
        )
        .await?;
        tx.commit().await?;
        Self::publish(&channel, &event);
        Ok(event)
    }

    async fn list_events(&self, session_id: SessionId, after: EventId) -> Result<Vec<Event>> {
        self.fetch_events(session_id, after, i64::MAX).await
    }

    async fn events_page(
        &self,
        session_id: SessionId,
        after: EventId,
        limit: u32,
    ) -> Result<Vec<Event>> {
        self.fetch_events(session_id, after, i64::from(limit)).await
    }

    async fn run_events(&self, session_id: SessionId, run_id: &str) -> Result<Vec<Event>> {
        let rows = sqlx::query(
            "SELECT id, session_id, run_id, type, data, created_at FROM event
              WHERE run_id = ? AND session_id = ? ORDER BY id",
        )
        .bind(run_id)
        .bind(session_id as i64)
        .fetch_all(&self.pool)
        .await?;
        rows.iter().map(event_from_row).collect()
    }

    async fn conversation(&self, session_id: SessionId) -> Result<Conversation> {
        let rows = sqlx::query(
            "WITH checkpoint AS (
                SELECT id, data FROM event WHERE session_id = ? AND type = 'compaction' ORDER BY id DESC LIMIT 1
             ), latest_user AS (
                SELECT id FROM event WHERE session_id = ? AND type = 'message'
                  AND json_extract(data, '$.message.role') = 'user' ORDER BY id DESC LIMIT 1
             )
             SELECT id, session_id, run_id, type, data, created_at FROM event
              WHERE session_id = ? AND (
                (type = 'compaction' AND id = (SELECT id FROM checkpoint)) OR
                (type = 'message' AND (
                    id >= COALESCE((SELECT json_extract(data, '$.tailFrom') FROM checkpoint), 0)
                    OR id = (SELECT id FROM latest_user)
                    OR id = (SELECT json_extract(data, '$.keptUser') FROM checkpoint)
                ))) ORDER BY id",
        )
        .bind(session_id as i64)
        .bind(session_id as i64)
        .bind(session_id as i64)
        .fetch_all(&self.pool)
        .await?;
        let events: Vec<Event> = rows.iter().map(event_from_row).collect::<Result<_>>()?;
        let compaction = events
            .iter()
            .rev()
            .find(|event| event.event_type == "compaction");
        let tail_from = compaction
            .and_then(|event| event.data.get("tailFrom"))
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let summary = compaction
            .and_then(|event| event.data.get("summary"))
            .and_then(Value::as_str)
            .map(str::to_owned);
        let kept_user = compaction
            .and_then(|event| event.data.get("keptUser"))
            .and_then(Value::as_u64);
        let last_user = events
            .iter()
            .rev()
            .find(|event| {
                event.event_type == "message"
                    && event.data.pointer("/message/role").and_then(Value::as_str) == Some("user")
            })
            .map(|event| event.id);
        let kept = events
            .into_iter()
            .filter(|event| {
                event.event_type == "message"
                    && (event.id >= tail_from
                        || Some(event.id) == last_user
                        || Some(event.id) == kept_user)
            })
            .collect();
        Ok(Conversation {
            summary,
            events: kept,
        })
    }

    async fn history(&self, session_id: SessionId, prompts: &PromptSet) -> Result<Vec<Message>> {
        let conversation = self.conversation(session_id).await?;
        let mut messages = Vec::with_capacity(conversation.events.len() + 1);
        if let Some(summary) = &conversation.summary {
            messages.push(Message::user(prompts.checkpoint(summary)));
        }
        let parsed: Vec<Message> = conversation
            .events
            .iter()
            .map(|event| {
                serde_json::from_value(event.data.get("message").cloned().unwrap_or(Value::Null))
                    .map_err(|err| {
                        Error::Run(format!("event {} holds no valid message: {err}", event.id))
                    })
            })
            .collect::<Result<_>>()?;
        let mut parsed = parsed.into_iter().peekable();
        while let Some(message) = parsed.next() {
            let mut results = Vec::new();
            if message.role == crate::llm::Role::Assistant && message.tool_calls.is_some() {
                while parsed
                    .peek()
                    .is_some_and(|message| message.role == crate::llm::Role::Tool)
                {
                    results.push(parsed.next().expect("the next message was checked"));
                }
            }
            let answered: HashSet<&str> = results
                .iter()
                .filter_map(|message| message.tool_call_id.as_deref())
                .collect();
            let unanswered: Vec<String> = message
                .tool_calls
                .iter()
                .flatten()
                .map(|call| call.id.clone())
                .filter(|id| !answered.contains(id.as_str()))
                .collect();
            messages.push(message);
            // Repair interrupted calls with a diagnostic; never replay their side
            // effects.
            for id in unanswered {
                messages.push(Message::tool(id, INTERRUPTED_RESULT));
            }
            messages.extend(results);
        }
        Ok(messages)
    }

    fn subscribe(&self, session_id: SessionId) -> broadcast::Receiver<Arc<Event>> {
        self.channel(session_id).sender.subscribe()
    }

    async fn recover(&self) -> Result<Recovery> {
        let data = json!({"error": super::RESTART_INTERRUPTED});
        let rows = sqlx::query(
            "INSERT INTO event (session_id, run_id, type, data, created_at)
             SELECT e.session_id, e.run_id, 'run.interrupted', ?, ?
               FROM event e
              WHERE e.type = 'run.started' AND e.run_id IS NOT NULL
                AND NOT EXISTS (
                    SELECT 1 FROM event t
                     WHERE t.run_id = e.run_id
                       AND t.type IN ('run.finished', 'run.failed', 'run.interrupted'))
             RETURNING session_id, run_id",
        )
        .bind(data.to_string())
        .bind(now_millis())
        .fetch_all(&self.pool)
        .await?;
        let interrupted = rows
            .iter()
            .map(|row| {
                (
                    row.get::<i64, _>("session_id") as SessionId,
                    row.get::<String, _>("run_id"),
                )
            })
            .collect::<Vec<_>>();
        // Publish the repairs for any live viewer, in cursor order.
        for (session_id, run_id) in &interrupted {
            let events = self.list_events(*session_id, 0).await?;
            if let Some(event) = events.iter().find(|event| {
                event.run_id.as_deref() == Some(run_id) && event.event_type == "run.interrupted"
            }) {
                Self::publish(&self.channel(*session_id), event);
            }
        }
        Ok(Recovery { interrupted })
    }
}

impl SqliteStore {
    async fn fetch_events(
        &self,
        session_id: SessionId,
        after: EventId,
        limit: i64,
    ) -> Result<Vec<Event>> {
        // A cursor past every possible id has nothing after it; cast truncation
        // would otherwise bind a negative cursor and replay everything.
        let Ok(after) = i64::try_from(after) else {
            return Ok(Vec::new());
        };
        let rows = sqlx::query(
            "SELECT id, session_id, run_id, type, data, created_at FROM event
              WHERE session_id = ? AND id > ? ORDER BY id LIMIT ?",
        )
        .bind(session_id as i64)
        .bind(after)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;
        rows.iter().map(event_from_row).collect()
    }

    /// Claims inputs and run-scoped message events in one transaction; publishes after
    /// commit.
    async fn deliver(
        &self,
        session_id: SessionId,
        batches: &[(Delivery, u32)],
        start: bool,
        run_id: Option<&str>,
    ) -> Result<RunClaim> {
        let channel = self.channel(session_id);
        let _order = channel.append.lock().await;
        let mut tx = self.pool.begin_with("BEGIN IMMEDIATE").await?;
        let now = now_millis();
        let mut inputs: Vec<Input> = Vec::new();
        let mut events: Vec<Event> = Vec::new();
        for (delivery, limit) in batches {
            let rows = sqlx::query(
                "UPDATE input SET consumed_at = ?
                  WHERE id IN (SELECT id FROM input
                                WHERE session_id = ? AND delivery = ? AND consumed_at IS NULL
                                ORDER BY id LIMIT ?)
                  RETURNING id, source, content",
            )
            .bind(now)
            .bind(session_id as i64)
            .bind(delivery.as_str())
            .bind(*limit as i64)
            .fetch_all(&mut *tx)
            .await?;
            let mut batch: Vec<Input> = rows
                .iter()
                .map(|row| Input {
                    id: row.get::<i64, _>("id") as u64,
                    source: row.get("source"),
                    content: row.get("content"),
                })
                .collect();
            batch.sort_by_key(|input| input.id);
            inputs.extend(batch);
        }
        let claimed_run = if inputs.is_empty() {
            None
        } else if start {
            Some(uuid::Uuid::new_v4().to_string())
        } else {
            run_id.map(str::to_owned)
        };
        for input in &inputs {
            self.check_hook("message")?;
            let data = json!({
                "message": {"role": "user", "content": input.content},
                "inputId": input.id,
                "source": input.source,
            });
            events.push(
                insert_event(
                    &mut tx,
                    session_id,
                    claimed_run.as_deref(),
                    "message",
                    data,
                    now,
                )
                .await?,
            );
        }
        if start && !inputs.is_empty() {
            self.check_hook("run.started")?;
            let run_id = claimed_run.as_deref().expect("a started run has an id");
            let data = json!({"inputIds": inputs.iter().map(|input| input.id).collect::<Vec<_>>()});
            events.push(
                insert_event(&mut tx, session_id, Some(run_id), "run.started", data, now).await?,
            );
        }
        tx.commit().await?;
        for event in &events {
            Self::publish(&channel, event);
        }
        Ok(RunClaim {
            inputs,
            run_id: claimed_run.filter(|_| start),
        })
    }
}
