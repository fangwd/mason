//! Per-session inbox draining, interruption, recovery, and transient event feeds.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

use serde_json::json;
use tokio::sync::{broadcast, mpsc};
use tokio::task::JoinHandle;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::agent::{Agent, SessionRun};
use crate::engine::RunEvent;
use crate::error::Result;
use crate::llm::Message;
use crate::reflect::{self, RunReview};
use crate::store::{Recovery, SessionId, SessionStore};

/// Buffer for the transient delta broadcast; lagging viewers re-sync from the store,
/// they do not block the run.
const DELTA_BUFFER: usize = 256;

/// Bounds root-session execution and admission; subagents retain their own fan-out limits.
#[derive(Debug, Clone, Copy)]
pub struct RunnerLimits {
    pub max_active_runs: usize,
    pub max_pending_sessions: usize,
}
impl Default for RunnerLimits {
    fn default() -> Self {
        Self {
            max_active_runs: 4,
            max_pending_sessions: 128,
        }
    }
}
impl RunnerLimits {
    pub fn validate(&self) -> Result<()> {
        if self.max_active_runs == 0
            || self.max_pending_sessions < self.max_active_runs
            || self.max_pending_sessions > tokio::sync::Semaphore::MAX_PERMITS
        {
            return Err(crate::Error::Config(
                "runner limits require 0 < active runs <= pending sessions".into(),
            ));
        }
        Ok(())
    }
}

struct SessionHandle {
    token: CancellationToken,
    task: Option<JoinHandle<()>>,
}

pub(crate) struct ChildSessionGuard {
    runner: Weak<Runner>,
    session_id: SessionId,
}

impl Drop for ChildSessionGuard {
    fn drop(&mut self) {
        if let Some(runner) = self.runner.upgrade() {
            detach(&runner, self.session_id);
        }
    }
}

/// Drains one inbox per session; different sessions run concurrently.
pub struct Runner {
    agent: Agent,
    slots: tokio::sync::Semaphore,
    suspended: Mutex<HashSet<SessionId>>,
    admission: Arc<tokio::sync::Semaphore>,
    store: Arc<dyn SessionStore>,
    sessions: Mutex<HashMap<SessionId, SessionHandle>>,
    deltas: Mutex<HashMap<SessionId, broadcast::Sender<Arc<RunEvent>>>>,
    /// Background reviews of finished runs.
    reviews: Mutex<Vec<JoinHandle<()>>>,
    /// Serializes reflections and consolidations.
    maintenance: tokio::sync::Mutex<()>,
    /// When each stored session was last reflected on, within the cooldown.
    reflected: Mutex<HashMap<SessionId, Instant>>,
    shutdown: CancellationToken,
}

impl Runner {
    pub fn new(agent: Agent, store: Arc<dyn SessionStore>) -> Arc<Self> {
        Self::with_limits(agent, store, RunnerLimits::default())
            .expect("valid default runner limits")
    }

    pub fn with_limits(
        agent: Agent,
        store: Arc<dyn SessionStore>,
        limits: RunnerLimits,
    ) -> Result<Arc<Self>> {
        limits.validate()?;
        Ok(Arc::new(Self {
            slots: tokio::sync::Semaphore::new(limits.max_active_runs),
            suspended: Mutex::new(HashSet::new()),
            admission: Arc::new(tokio::sync::Semaphore::new(limits.max_pending_sessions)),
            agent,
            store,
            sessions: Mutex::new(HashMap::new()),
            deltas: Mutex::new(HashMap::new()),
            reviews: Mutex::new(Vec::new()),
            maintenance: tokio::sync::Mutex::new(()),
            reflected: Mutex::new(HashMap::new()),
            shutdown: CancellationToken::new(),
        }))
    }

    pub fn store(&self) -> &Arc<dyn SessionStore> {
        &self.store
    }

    pub fn agent(&self) -> &Agent {
        &self.agent
    }

    /// Whether a drain task owns the session right now.
    pub fn is_busy(&self, session_id: SessionId) -> bool {
        self.sessions
            .lock()
            .expect("the runner mutex is poisoned")
            .contains_key(&session_id)
    }

    /// Whether a failed run stopped draining the session's inbox; its queued
    /// inputs wait for the next message or an explicit wake.
    pub fn is_stalled(&self, session_id: SessionId) -> bool {
        self.suspended
            .lock()
            .expect("suspended mutex")
            .contains(&session_id)
    }

    /// Starts draining the session's inbox unless it is already running.
    pub fn wake(self: &Arc<Self>, session_id: SessionId) {
        if let Err(err) = self.try_wake(session_id) {
            tracing::warn!("{err}");
        }
    }

    /// Reject admission instead of creating unbounded tasks.
    pub fn try_wake(self: &Arc<Self>, session_id: SessionId) -> Result<()> {
        self.suspended
            .lock()
            .expect("suspended mutex")
            .remove(&session_id);
        self.admit(session_id, None)
    }

    /// Reserve capacity before storing HTTP inputs, so rejected submissions have no side effects.
    pub async fn submit(
        self: &Arc<Self>,
        session_id: SessionId,
        input: crate::store::NewInput,
    ) -> Result<u64> {
        if self.shutdown.is_cancelled() {
            return Err(crate::Error::Cancelled);
        }
        let reserved = if self.is_busy(session_id) {
            None
        } else {
            Some(
                self.admission
                    .clone()
                    .try_acquire_owned()
                    .map_err(|_| crate::Error::Busy)?,
            )
        };
        let id = self.store.add_input(session_id, input).await?;
        self.suspended
            .lock()
            .expect("suspended mutex")
            .remove(&session_id);
        // A busy session can finish while storage is in flight. Its input is still
        // durable; the existing bounded refill loop will admit it when capacity opens.
        match self.admit(session_id, reserved) {
            Ok(()) | Err(crate::Error::Busy) => Ok(id),
            Err(err) => Err(err),
        }
    }

    fn admit(
        self: &Arc<Self>,
        session_id: SessionId,
        reserved: Option<tokio::sync::OwnedSemaphorePermit>,
    ) -> Result<()> {
        let mut sessions = self.sessions.lock().expect("the runner mutex is poisoned");
        if self.shutdown.is_cancelled() || sessions.contains_key(&session_id) {
            return Ok(());
        }
        let admission = match reserved {
            Some(permit) => permit,
            None => self
                .admission
                .clone()
                .try_acquire_owned()
                .map_err(|_| crate::Error::Busy)?,
        };
        let token = self.shutdown.child_token();
        let runner = self.clone();
        let task_token = token.clone();
        let task = tokio::spawn(async move {
            let _release = PanicDetach {
                runner: runner.clone(),
                session_id,
            };
            let permit = tokio::select! {
                _ = task_token.cancelled() => { finish(runner.clone(), session_id).await; return; },
                permit = runner.slots.acquire() => permit.expect("runner semaphore stays open"),
            };
            drain(runner.clone(), session_id, task_token).await;
            drop(permit);
            drop(admission);
            if !runner.shutdown.is_cancelled()
                && let Ok(pending) = runner.store.pending_sessions().await
            {
                for id in pending {
                    if runner
                        .suspended
                        .lock()
                        .expect("suspended mutex")
                        .contains(&id)
                    {
                        continue;
                    }
                    if runner.admit(id, None).is_err() {
                        break;
                    }
                }
            }
        });
        sessions.insert(
            session_id,
            SessionHandle {
                token,
                task: Some(task),
            },
        );
        Ok(())
    }

    pub(crate) fn register_child(
        self: &Arc<Self>,
        session_id: SessionId,
        token: CancellationToken,
    ) -> Result<ChildSessionGuard> {
        let mut sessions = self.sessions.lock().expect("the runner mutex is poisoned");
        if self.shutdown.is_cancelled() || token.is_cancelled() {
            return Err(crate::Error::Cancelled);
        }
        if sessions.contains_key(&session_id) {
            return Err(crate::Error::Run(format!(
                "session {session_id} is already running"
            )));
        }
        sessions.insert(session_id, SessionHandle { token, task: None });
        Ok(ChildSessionGuard {
            runner: Arc::downgrade(self),
            session_id,
        })
    }

    /// Stops the current run; queued inputs start the next run as usual.
    pub fn interrupt(&self, session_id: SessionId) {
        if let Some(handle) = self
            .sessions
            .lock()
            .expect("the runner mutex is poisoned")
            .get(&session_id)
        {
            handle.token.cancel();
        }
    }

    /// Subscribes to the transient delta feed of a session. Committed events come from
    /// the store; deltas cannot be replayed.
    pub fn subscribe_deltas(&self, session_id: SessionId) -> broadcast::Receiver<Arc<RunEvent>> {
        self.with_delta_channel(session_id, |sender| sender.subscribe())
    }

    /// Uses the session's channel under the registry lock, so cleanup cannot evict
    /// it between lookup and use.
    fn with_delta_channel<T>(
        &self,
        session_id: SessionId,
        use_channel: impl FnOnce(&broadcast::Sender<Arc<RunEvent>>) -> T,
    ) -> T {
        let sessions = self.sessions.lock().expect("the runner mutex is poisoned");
        let mut deltas = self.deltas.lock().expect("the runner mutex is poisoned");
        if deltas.len() > 64 {
            // Idle sessions without viewers can rebuild their channel on demand.
            deltas.retain(|id, sender| sender.receiver_count() > 0 || sessions.contains_key(id));
        }
        use_channel(
            deltas
                .entry(session_id)
                .or_insert_with(|| broadcast::channel(DELTA_BUFFER).0),
        )
    }

    /// Bridges the engine's bounded event queue into the session's broadcast.
    pub(crate) fn delta_sink(&self, session_id: SessionId) -> mpsc::Sender<RunEvent> {
        let (tx, mut rx) = mpsc::channel(DELTA_BUFFER);
        let broadcaster = self.with_delta_channel(session_id, Clone::clone);
        tokio::spawn(async move {
            while let Some(event) = rx.recv().await {
                let _ = broadcaster.send(Arc::new(event));
            }
        });
        tx
    }

    /// Recovers abandoned runs before draining pending inputs.
    pub async fn recover_and_drain(self: &Arc<Self>) -> Result<Recovery> {
        let recovery = self.store.recover().await?;
        for session_id in self.store.pending_sessions().await? {
            if self.try_wake(session_id).is_err() {
                break;
            }
        }
        Ok(recovery)
    }

    /// Grades a finished run, reflects on it and consolidates memory when due, in
    /// the background so the session's next run never waits; shutdown cancels it.
    /// Grading runs concurrently; memory maintenance runs one at a time.
    fn spawn_review(self: &Arc<Self>, session_id: SessionId, run_id: String, failed: bool) {
        let runner = self.clone();
        let token = self.shutdown.child_token();
        let task = tokio::spawn(async move {
            let messages = match run_messages(runner.store.as_ref(), session_id, &run_id).await {
                Ok(messages) => messages,
                Err(err) => {
                    tracing::warn!(
                        "agent session {session_id}: cannot load run {run_id} for review: {err}"
                    );
                    return;
                }
            };
            let mut review = RunReview {
                messages: &messages,
                failed,
                steered: reflect::steered(&messages),
                session: Some(session_id),
                verdict: None,
            };
            let verdict = match runner.agent.evaluate(&review, &token).await {
                Ok(verdict) => verdict,
                Err(err) => {
                    if !token.is_cancelled() {
                        tracing::warn!(
                            "agent session {session_id}: evaluating run {run_id} failed: {err}"
                        );
                    }
                    None
                }
            };
            if let Some(verdict) = &verdict
                && let Err(err) = runner
                    .store
                    .append_run_event(session_id, &run_id, "run.evaluated", json!(verdict))
                    .await
            {
                tracing::error!("agent session {session_id}: cannot record run.evaluated: {err}");
            }
            review.verdict = verdict.as_ref();
            let Some(memory) = runner.agent.memory() else {
                return;
            };
            let _turn = tokio::select! {
                _ = token.cancelled() => return,
                turn = runner.maintenance.lock() => turn,
            };
            if !runner.reflected_recently(session_id) {
                match runner.agent.reflect(review, token.clone()).await {
                    Ok(true) => runner.record_reflection(session_id),
                    Ok(false) => {}
                    Err(err) => {
                        if !token.is_cancelled() {
                            tracing::warn!(
                                "agent session {session_id}: reflection on run {run_id} failed: {err}"
                            );
                        }
                    }
                }
            }
            if !token.is_cancelled()
                && memory.consolidation_due().await
                && let Err(err) = runner.agent.consolidate(token.clone()).await
                && !token.is_cancelled()
            {
                tracing::warn!("memory consolidation failed: {err}");
            }
        });
        let mut reviews = self.reviews.lock().expect("reviews mutex");
        reviews.retain(|task| !task.is_finished());
        reviews.push(task);
    }

    /// Whether the session was reflected on within [`reflect::REFLECTION_COOLDOWN`].
    fn reflected_recently(&self, session_id: SessionId) -> bool {
        self.reflected
            .lock()
            .expect("reflected mutex")
            .get(&session_id)
            .is_some_and(|at| at.elapsed() < reflect::REFLECTION_COOLDOWN)
    }

    fn record_reflection(&self, session_id: SessionId) {
        let mut reflected = self.reflected.lock().expect("reflected mutex");
        reflected.retain(|_, at| at.elapsed() < reflect::REFLECTION_COOLDOWN);
        reflected.insert(session_id, Instant::now());
    }

    /// Stops new work, cancels runs and reviews, and allows bounded cleanup
    /// before aborting stragglers.
    pub async fn shutdown(&self, grace: Duration) {
        let handles: Vec<SessionHandle> = {
            let mut sessions = self.sessions.lock().expect("the runner mutex is poisoned");
            self.shutdown.cancel();
            sessions.drain().map(|(_, handle)| handle).collect()
        };
        let reviews: Vec<JoinHandle<()>> = self
            .reviews
            .lock()
            .expect("reviews mutex")
            .drain(..)
            .collect();
        let deadline = Instant::now() + grace;
        let tasks = handles
            .into_iter()
            .filter_map(|handle| handle.task)
            .chain(reviews);
        for mut task in tasks {
            if tokio::time::timeout_at(deadline, &mut task).await.is_err() {
                task.abort();
            }
        }
        self.agent.close_session_scopes().await;
    }
}

/// Releases the session handle if the drain task panics, so the session does not
/// stay busy forever.
struct PanicDetach {
    runner: Arc<Runner>,
    session_id: SessionId,
}

impl Drop for PanicDetach {
    fn drop(&mut self) {
        if std::thread::panicking() {
            tracing::error!(
                "agent session {}: the drain task panicked; releasing the session",
                self.session_id
            );
            detach(&self.runner, self.session_id);
        }
    }
}

/// Drains until empty, cancelled or failed; always releases the session handle.
async fn drain(runner: Arc<Runner>, session_id: SessionId, token: CancellationToken) {
    let profile = match runner.store.get_session(session_id).await {
        Ok(Some(session)) if session.parent_id.is_none() => runner.agent.select_profile(&session),
        Ok(Some(_)) => {
            detach(&runner, session_id);
            return;
        }
        Ok(None) => {
            tracing::warn!("agent session {session_id}: no such session");
            detach(&runner, session_id);
            return;
        }
        Err(err) => {
            tracing::error!("agent session {session_id}: {err}");
            detach(&runner, session_id);
            return;
        }
    };
    while !token.is_cancelled() {
        let claim = match runner.store.start_run(session_id).await {
            Ok(claim) => claim,
            Err(err) => {
                tracing::error!("agent session {session_id}: cannot claim inputs: {err}");
                runner
                    .suspended
                    .lock()
                    .expect("suspended mutex")
                    .insert(session_id);
                // Skip re-waking after a store failure to avoid a hot retry loop.
                detach(&runner, session_id);
                return;
            }
        };
        let Some(run_id) = claim.run_id else { break };
        let events = runner.delta_sink(session_id);
        let result = runner
            .agent
            .run_session_managed(
                SessionRun {
                    store: runner.store.clone(),
                    session_id,
                    run_id: run_id.clone(),
                    profile: profile.clone(),
                    token: token.clone(),
                    events: Some(events),
                },
                Arc::downgrade(&runner),
            )
            .await;
        let (event_type, data) = match result {
            Ok(answer) => ("run.finished", json!({"answer": answer})),
            Err(err) if token.is_cancelled() => {
                ("run.interrupted", json!({"error": err.to_string()}))
            }
            Err(err) => ("run.failed", json!({"error": err.to_string()})),
        };
        if let Err(err) = runner
            .store
            .append_run_event(session_id, &run_id, event_type, data)
            .await
        {
            tracing::error!("agent session {session_id}: cannot record {event_type}: {err}");
        }
        if event_type != "run.interrupted"
            && (runner.agent.memory().is_some() || runner.agent.has_evaluator())
        {
            runner.spawn_review(session_id, run_id.clone(), event_type == "run.failed");
        }
        if event_type == "run.failed" {
            runner
                .suspended
                .lock()
                .expect("suspended mutex")
                .insert(session_id);
            // Retrying queued inputs against the same failure would burn the whole
            // inbox: they wait for the next explicit wake.
            detach(&runner, session_id);
            return;
        }
    }
    finish(runner, session_id).await;
}

/// Rechecks the inbox after releasing ownership to avoid losing a concurrent input.
/// An interrupted run rechecks too: a wake during its cleanup was a no-op, and
/// its input would otherwise wait for an unrelated message.
async fn finish(runner: Arc<Runner>, session_id: SessionId) {
    detach(&runner, session_id);
    if runner.shutdown.is_cancelled() {
        return;
    }
    match runner.store.pending_sessions().await {
        Ok(ids) => {
            if ids.contains(&session_id) {
                runner.wake(session_id);
            }
        }
        Err(err) => tracing::error!("agent session {session_id} inbox: {err}"),
    }
}

/// The messages a run committed, its claimed inputs and steering included.
async fn run_messages(
    store: &dyn SessionStore,
    session_id: SessionId,
    run_id: &str,
) -> Result<Vec<Message>> {
    store
        .run_events(session_id, run_id)
        .await?
        .into_iter()
        .filter(|event| event.event_type == "message")
        .map(|event| {
            serde_json::from_value(event.data.get("message").cloned().unwrap_or_default()).map_err(
                |err| {
                    crate::Error::Run(format!("event {} holds no valid message: {err}", event.id))
                },
            )
        })
        .collect()
}

/// Removes a session's live handle without rechecking its inbox.
fn detach(runner: &Runner, session_id: SessionId) {
    runner
        .sessions
        .lock()
        .expect("the runner mutex is poisoned")
        .remove(&session_id);
}
