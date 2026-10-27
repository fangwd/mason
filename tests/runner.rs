//! Session runner ownership, interruption, and recovery tests.

mod support;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use mason::store::{
    Event as StoreEvent, NewInput, NewSession, Recovery, SessionStore, SqliteStore,
};
use mason::{
    Agent, OutputSpill, Profile, PromptSet, Role, Runner, SessionId, ToolCall, builtin_tools,
};
use serde_json::json;
use support::{
    EchoTool, FunctionModel, Reply, ScriptedModel, SleepTool, answer, call, is_summary_request,
};
use tokio::sync::Notify;

async fn fixture() -> (tempfile::TempDir, Arc<SqliteStore>) {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(
        SqliteStore::open(dir.path().join("mason.db"))
            .await
            .unwrap(),
    );
    (dir, store)
}

fn agent(dir: &std::path::Path, model: Arc<dyn mason::ChatModel>) -> Agent {
    Agent::builder()
        .model(model)
        .cwd(dir)
        .tools(builtin_tools())
        .spill(Arc::new(OutputSpill::new(dir.join("spill"))))
        .build()
        .unwrap()
}

async fn wait_for_events(
    store: &Arc<SqliteStore>,
    session: SessionId,
    event_type: &str,
    count: usize,
) -> Vec<StoreEvent> {
    let mut rx = store.subscribe(session);
    loop {
        let events = store.list_events(session, 0).await.unwrap();
        if events.iter().filter(|e| e.event_type == event_type).count() >= count {
            return events;
        }
        tokio::time::timeout(Duration::from_secs(15), rx.recv())
            .await
            .unwrap_or_else(|_| panic!("timed out waiting for {count} `{event_type}` events"))
            .expect("the store feed stays open");
    }
}

async fn wait_until_idle(runner: &Runner, session: SessionId) {
    for _ in 0..1_500 {
        if !runner.is_busy(session) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("session {session} stayed busy");
}

fn tool_call_json(command: &str) -> Reply {
    call("bash", &json!({"command": command}).to_string())
}

/// Injects a configurable number of failures into `pending_sessions` and `start_run`.
struct FlakyStore {
    inner: Arc<SqliteStore>,
    failures: AtomicUsize,
    pub calls: AtomicUsize,
    pub start_failures: AtomicUsize,
}

impl FlakyStore {
    fn new(inner: Arc<SqliteStore>, failures: usize) -> Arc<Self> {
        Arc::new(Self {
            inner,
            failures: AtomicUsize::new(failures),
            calls: AtomicUsize::new(0),
            start_failures: AtomicUsize::new(0),
        })
    }
}

#[async_trait::async_trait]
impl SessionStore for FlakyStore {
    async fn create_session(&self, new: NewSession) -> mason::Result<SessionId> {
        self.inner.create_session(new).await
    }
    async fn get_session(&self, id: SessionId) -> mason::Result<Option<mason::store::Session>> {
        self.inner.get_session(id).await
    }
    async fn update_session(
        &self,
        id: SessionId,
        update: mason::store::SessionUpdate,
    ) -> mason::Result<bool> {
        self.inner.update_session(id, update).await
    }
    async fn delete_session(&self, id: SessionId) -> mason::Result<bool> {
        self.inner.delete_session(id).await
    }
    async fn withdraw_input(&self, session_id: SessionId, input_id: u64) -> mason::Result<bool> {
        self.inner.withdraw_input(session_id, input_id).await
    }
    async fn add_input(&self, session_id: SessionId, input: NewInput) -> mason::Result<u64> {
        self.inner.add_input(session_id, input).await
    }
    async fn list_inputs(
        &self,
        session_id: SessionId,
    ) -> mason::Result<Vec<mason::store::StoredInput>> {
        self.inner.list_inputs(session_id).await
    }
    async fn take_inputs(
        &self,
        session_id: SessionId,
        delivery: mason::store::Delivery,
        limit: u32,
        run_id: Option<&str>,
    ) -> mason::Result<Vec<mason::store::Input>> {
        self.inner
            .take_inputs(session_id, delivery, limit, run_id)
            .await
    }
    async fn start_run(&self, session_id: SessionId) -> mason::Result<mason::store::RunClaim> {
        let remaining = self.start_failures.load(Ordering::SeqCst);
        if remaining > 0 {
            self.start_failures.store(remaining - 1, Ordering::SeqCst);
            return Err(mason::Error::Run("database temporarily unavailable".into()));
        }
        self.inner.start_run(session_id).await
    }
    async fn pending_sessions(&self) -> mason::Result<Vec<SessionId>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let remaining = self.failures.load(Ordering::SeqCst);
        if remaining > 0 {
            self.failures.store(remaining - 1, Ordering::SeqCst);
            return Err(mason::Error::Run("database temporarily unavailable".into()));
        }
        self.inner.pending_sessions().await
    }
    async fn list_sessions(
        &self,
        query: &mason::store::SessionQuery,
    ) -> mason::Result<Vec<mason::store::SessionSummary>> {
        self.inner.list_sessions(query).await
    }
    async fn append(
        &self,
        session_id: SessionId,
        event_type: &str,
        data: serde_json::Value,
    ) -> mason::Result<StoreEvent> {
        self.inner.append(session_id, event_type, data).await
    }
    async fn append_run_event(
        &self,
        session_id: SessionId,
        run_id: &str,
        event_type: &str,
        data: serde_json::Value,
    ) -> mason::Result<StoreEvent> {
        self.inner
            .append_run_event(session_id, run_id, event_type, data)
            .await
    }
    async fn list_events(
        &self,
        session_id: SessionId,
        after: u64,
    ) -> mason::Result<Vec<StoreEvent>> {
        self.inner.list_events(session_id, after).await
    }
    async fn events_page(
        &self,
        session_id: SessionId,
        after: u64,
        limit: u32,
    ) -> mason::Result<Vec<StoreEvent>> {
        self.inner.events_page(session_id, after, limit).await
    }
    async fn conversation(
        &self,
        session_id: SessionId,
    ) -> mason::Result<mason::store::Conversation> {
        self.inner.conversation(session_id).await
    }
    async fn history(
        &self,
        session_id: SessionId,
        prompts: &PromptSet,
    ) -> mason::Result<Vec<mason::Message>> {
        self.inner.history(session_id, prompts).await
    }
    fn subscribe(
        &self,
        session_id: SessionId,
    ) -> tokio::sync::broadcast::Receiver<Arc<StoreEvent>> {
        self.inner.subscribe(session_id)
    }
    async fn recover(&self) -> mason::Result<Recovery> {
        self.inner.recover().await
    }
}

#[tokio::test]
async fn a_queued_message_runs_the_agent_and_records_each_step() {
    let (dir, store) = fixture().await;
    let model = ScriptedModel::new(vec![tool_call_json("echo hi"), answer("said hi")]);
    let runner = Runner::new(agent(dir.path(), model.clone()), store.clone());
    let session = store
        .create_session(NewSession {
            title: Some("demo".into()),
            ..Default::default()
        })
        .await
        .unwrap();
    store
        .add_input(session, NewInput::queued("say hi"))
        .await
        .unwrap();

    runner.wake(session);
    let events = wait_for_events(&store, session, "run.finished", 1).await;
    let types: Vec<&str> = events.iter().map(|e| e.event_type.as_str()).collect();
    assert_eq!(
        types,
        [
            "message",
            "run.started",
            "message",
            "message",
            "message",
            "run.finished"
        ]
    );
    assert_eq!(
        events[3]
            .data
            .pointer("/message/content")
            .and_then(|v| v.as_str()),
        Some("hi\n\n[exit code 0]")
    );
    assert_eq!(events[5].data, json!({"answer": "said hi"}));

    wait_until_idle(&runner, session).await;
    assert!(!runner.is_busy(session));

    // The second model call saw the whole conversation.
    let requests = model.requests();
    let roles: Vec<Role> = requests[1].messages.iter().map(|m| m.role).collect();
    assert_eq!(
        roles,
        [Role::System, Role::User, Role::Assistant, Role::Tool]
    );

    let session_row = store.get_session(session).await.unwrap().unwrap();
    assert_eq!(session_row.title.as_deref(), Some("demo"));
}

#[tokio::test]
async fn steer_joins_the_running_run_and_queue_waits() {
    let (dir, store) = fixture().await;
    let gate = Arc::new(Notify::new());
    let model = ScriptedModel::new(vec![
        Reply::Gate(gate.clone(), Box::new(tool_call_json("true"))),
        answer("first run done"),
        answer("second run done"),
    ]);
    let runner = Runner::new(agent(dir.path(), model.clone()), store.clone());
    let session = store.create_session(NewSession::default()).await.unwrap();
    store
        .add_input(session, NewInput::queued("start"))
        .await
        .unwrap();
    runner.wake(session);
    wait_for_events(&store, session, "run.started", 1).await;
    // Wait until the first model call is in flight (parked on the gate):
    // the engine's step-1 steer has already run, so the steer input below
    // deterministically joins the second call.
    for _ in 0..1_500 {
        if model.request_count() == 1 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(
        model.request_count(),
        1,
        "the first call is parked on the gate"
    );

    store
        .add_input(session, NewInput::queued("later"))
        .await
        .unwrap();
    store
        .add_input(
            session,
            NewInput {
                content: "also check X".into(),
                delivery: mason::Delivery::Steer,
                source: "human".into(),
                key: None,
            },
        )
        .await
        .unwrap();
    gate.notify_one();

    let events = wait_for_events(&store, session, "run.finished", 2).await;
    let answers: Vec<&str> = events
        .iter()
        .filter(|e| e.event_type == "run.finished")
        .map(|e| e.data["answer"].as_str().unwrap())
        .collect();
    assert_eq!(answers, ["first run done", "second run done"]);

    // The steer message reached the first run's second model call; the
    // queued one waited for the next run.
    let requests = model.requests();
    assert_eq!(
        requests[1].messages.last().unwrap().content.as_deref(),
        Some("also check X")
    );
    assert_eq!(
        requests[2].messages.last().unwrap().content.as_deref(),
        Some("later")
    );
}

#[tokio::test]
async fn a_failed_run_stops_draining_and_keeps_the_queue() {
    let (dir, store) = fixture().await;
    let prompts_path = dir.path().join("prompts.yaml");
    std::fs::write(&prompts_path, mason::prompt::EMBEDDED_PROMPTS).unwrap();
    let model = ScriptedModel::new(vec![answer("two done"), answer("three done")]);
    let agent = Agent::builder()
        .model(model.clone())
        .cwd(dir.path())
        .tools(builtin_tools())
        .prompts(mason::PromptSource::File(prompts_path.clone()))
        .spill(Arc::new(OutputSpill::new(dir.path().join("spill"))))
        .build()
        .unwrap();
    let runner = Runner::new(agent, store.clone());
    let session = store.create_session(NewSession::default()).await.unwrap();
    for text in ["one", "two", "three"] {
        store
            .add_input(session, NewInput::queued(text))
            .await
            .unwrap();
    }
    // Every run reloads the prompt file; removing it fails the run before the
    // model is called.
    std::fs::remove_file(&prompts_path).unwrap();
    runner.wake(session);
    wait_for_events(&store, session, "run.failed", 1).await;
    wait_until_idle(&runner, session).await;
    let events = store.list_events(session, 0).await.unwrap();
    assert_eq!(
        events
            .iter()
            .filter(|e| e.event_type == "run.started")
            .count(),
        1,
        "one failure consumed one input, not the whole inbox"
    );
    assert!(
        store.pending_sessions().await.unwrap().contains(&session),
        "the rest stay queued"
    );
    assert_eq!(model.request_count(), 0);

    // Once the configuration is fixed, the next wake drains the rest.
    std::fs::write(&prompts_path, mason::prompt::EMBEDDED_PROMPTS).unwrap();
    runner.wake(session);
    let events = wait_for_events(&store, session, "run.finished", 2).await;
    let answers: Vec<&str> = events
        .iter()
        .filter(|e| e.event_type == "run.finished")
        .map(|e| e.data["answer"].as_str().unwrap())
        .collect();
    assert_eq!(answers, ["two done", "three done"]);
}

#[tokio::test]
async fn a_panicking_run_releases_the_session() {
    let (dir, store) = fixture().await;
    let model = FunctionModel::new(|_request: &mason::ChatRequest| panic!("the model exploded"));
    let runner = Runner::new(agent(dir.path(), model), store.clone());
    let session = store.create_session(NewSession::default()).await.unwrap();
    store
        .add_input(session, NewInput::queued("go"))
        .await
        .unwrap();
    runner.wake(session);
    wait_until_idle(&runner, session).await;
    // The run has no terminal event; startup recovery repairs it.
    assert_eq!(store.recover().await.unwrap().interrupted.len(), 1);
}

#[tokio::test]
async fn delta_subscriptions_survive_channel_cleanup() {
    let (dir, store) = fixture().await;
    let model = ScriptedModel::new(vec![answer("streamed answer")]);
    let runner = Runner::new(agent(dir.path(), model), store.clone());
    let session = store.create_session(NewSession::default()).await.unwrap();
    let mut deltas = runner.subscribe_deltas(session);
    // Many idle sessions without viewers push the registry past its cleanup
    // threshold; the live subscription above must survive.
    for other in 1..=80 {
        drop(runner.subscribe_deltas(session + 1_000 + other));
    }
    store
        .add_input(session, NewInput::queued("go"))
        .await
        .unwrap();
    runner.wake(session);
    wait_for_events(&store, session, "run.finished", 1).await;
    wait_until_idle(&runner, session).await;
    tokio::time::sleep(Duration::from_millis(100)).await;

    let mut text = String::new();
    while let Ok(event) = deltas.try_recv() {
        if let mason::RunEvent::Delta {
            delta: mason::ChatDelta::Text(fragment),
            ..
        } = &*event
        {
            text.push_str(fragment);
        }
    }
    assert_eq!(text, "streamed answer");
}

#[tokio::test]
async fn a_message_posted_during_interrupt_cleanup_still_runs() {
    let (dir, store) = fixture().await;
    // The tool ignores cancellation for a while, keeping the interrupted run
    // registered as busy during its cleanup.
    let slow = SleepTool::boxed("slow", Duration::from_secs(2), "slept");
    let model = ScriptedModel::new(vec![call("slow", "{}"), answer("second run done")]);
    let mut tools = builtin_tools();
    tools.push(slow);
    let agent = Agent::builder()
        .model(model.clone())
        .cwd(dir.path())
        .tools(tools)
        .spill(Arc::new(OutputSpill::new(dir.path().join("spill"))))
        .build()
        .unwrap();
    let runner = Runner::new(agent, store.clone());
    let session = store.create_session(NewSession::default()).await.unwrap();
    store
        .add_input(session, NewInput::queued("first"))
        .await
        .unwrap();
    runner.wake(session);
    for _ in 0..1_500 {
        if model.request_count() == 1 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(
        model.request_count(),
        1,
        "the first run is in its tool call"
    );

    runner.interrupt(session);
    assert!(
        runner.is_busy(session),
        "cleanup keeps the session registered"
    );
    store
        .add_input(session, NewInput::queued("next"))
        .await
        .unwrap();
    runner.wake(session); // a no-op while the interrupted run cleans up

    let events = wait_for_events(&store, session, "run.finished", 1).await;
    assert!(events.iter().any(|e| e.event_type == "run.interrupted"));
    assert_eq!(
        events.last().unwrap().data,
        json!({"answer": "second run done"})
    );
    wait_until_idle(&runner, session).await;
}

#[tokio::test]
async fn interrupting_stops_the_run_and_leaves_a_valid_history() {
    let (dir, store) = fixture().await;
    let model = ScriptedModel::new(vec![tool_call_json("sleep 30")]);
    let runner = Runner::new(agent(dir.path(), model.clone()), store.clone());
    let session = store.create_session(NewSession::default()).await.unwrap();
    store
        .add_input(session, NewInput::queued("wait"))
        .await
        .unwrap();

    let mut rx = store.subscribe(session);
    runner.wake(session);
    // Wait until the assistant tool-call message is committed.
    loop {
        let events = store.list_events(session, 0).await.unwrap();
        let has_calls = events
            .iter()
            .any(|e| e.data.pointer("/message/tool_calls").is_some());
        if has_calls {
            break;
        }
        tokio::time::timeout(Duration::from_secs(15), rx.recv())
            .await
            .expect("timed out waiting for the tool call")
            .expect("the store feed stays open");
    }
    runner.interrupt(session);
    wait_for_events(&store, session, "run.interrupted", 1).await;
    wait_until_idle(&runner, session).await;

    let prompts = PromptSet::parse(mason::prompt::EMBEDDED_PROMPTS).unwrap();
    let history = store.history(session, &prompts).await.unwrap();
    let rendered: Vec<(Role, Option<String>)> = history
        .iter()
        .map(|m| (m.role, m.content.clone()))
        .collect();
    assert_eq!(rendered[0], (Role::User, Some("wait".into())));
    assert_eq!(rendered[1].0, Role::Assistant);
    assert_eq!(rendered[1].1, None);
    assert_eq!(rendered[2].0, Role::Tool);
    assert_eq!(
        rendered[2].1.as_deref(),
        Some("(no output)\n\n[exit code killed]\nCommand aborted.")
    );
    assert!(!runner.is_busy(session));
}

#[tokio::test]
async fn a_failed_inbox_recheck_is_caught_after_a_successful_run() {
    let (dir, store) = fixture().await;
    let flaky = FlakyStore::new(store.clone(), 1);
    let model = ScriptedModel::new(vec![answer("ok")]);
    let runner = Runner::new(agent(dir.path(), model), flaky.clone());
    let session = store.create_session(NewSession::default()).await.unwrap();
    store
        .add_input(session, NewInput::queued("go"))
        .await
        .unwrap();

    runner.wake(session);
    wait_for_events(&store, session, "run.finished", 1).await;
    wait_until_idle(&runner, session).await;
    // Let the recheck run.
    for _ in 0..500 {
        if flaky.calls.load(Ordering::SeqCst) >= 1 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(
        flaky.calls.load(Ordering::SeqCst) >= 1,
        "the recheck happened"
    );
    assert!(
        !runner.is_busy(session),
        "the caught failure did not wedge the session"
    );
}

#[tokio::test]
async fn a_failed_claim_releases_the_session_for_the_next_wake() {
    let (dir, store) = fixture().await;
    let flaky = FlakyStore::new(store.clone(), 0);
    flaky.start_failures.store(1, Ordering::SeqCst);
    let model = ScriptedModel::new(vec![answer("done")]);
    let runner = Runner::new(agent(dir.path(), model.clone()), flaky.clone());
    let session = store.create_session(NewSession::default()).await.unwrap();
    store
        .add_input(session, NewInput::queued("go"))
        .await
        .unwrap();

    runner.wake(session);
    // The failed claim ends the drain task, which must release the session
    // instead of leaving it wedged busy.
    wait_until_idle(&runner, session).await;
    let events = store.list_events(session, 0).await.unwrap();
    assert!(
        events.is_empty(),
        "the failed claim recorded nothing: {events:?}"
    );
    assert_eq!(
        model.request_count(),
        0,
        "the failed claim ran no model call"
    );

    // The input is still pending: a fresh wake runs it to completion.
    runner.wake(session);
    wait_for_events(&store, session, "run.finished", 1).await;
    wait_until_idle(&runner, session).await;
    assert_eq!(model.request_count(), 1);
    runner.shutdown(Duration::from_secs(1)).await;
}

fn seeded_messages(turns: usize, tokens: usize) -> Vec<serde_json::Value> {
    let mut out = Vec::new();
    for n in 1..=turns {
        out.push(json!({"message": {"role": "user", "content": format!("q{n}")}}));
        out.push(json!({"message": {
            "role": "assistant",
            "content": null,
            "tool_calls": [{"id": format!("c{n}"), "type": "function", "function": {"name": "bash", "arguments": "{}"}}]
        }}));
        out.push(json!({"message": {"role": "tool", "tool_call_id": format!("c{n}"), "content": "x".repeat(tokens * 4)}}));
        out.push(json!({"message": {"role": "assistant", "content": format!("a{n}")}}));
    }
    out
}

#[tokio::test]
async fn the_current_request_survives_stored_compaction_verbatim() {
    let (dir, store) = fixture().await;
    let (big, _count) = EchoTool::boxed("big", &"y".repeat(16_000));
    let mains = Arc::new(AtomicUsize::new(0));
    let model = FunctionModel::with_context(
        {
            let mains = mains.clone();
            move |request: &mason::ChatRequest| {
                if is_summary_request(request) {
                    return answer("S");
                }
                let n = mains.fetch_add(1, Ordering::SeqCst) + 1;
                if n >= 5 {
                    return answer("done");
                }
                Reply::Calls(vec![(format!("t{n}"), "big".into(), "{}".into())])
            }
        },
        8_000,
    );
    let mut agent_builder_tools = builtin_tools();
    agent_builder_tools.push(big);
    let agent = Agent::builder()
        .model(model.clone())
        .cwd(dir.path())
        .tools(agent_builder_tools)
        .spill(Arc::new(OutputSpill::new(dir.path().join("spill"))))
        .build()
        .unwrap();
    let runner = Runner::new(agent, store.clone());

    let session = store.create_session(NewSession::default()).await.unwrap();
    for message in seeded_messages(1, 100) {
        store.append(session, "message", message).await.unwrap();
    }
    store
        .add_input(session, NewInput::queued("the actual request"))
        .await
        .unwrap();
    runner.wake(session);
    let events = wait_for_events(&store, session, "run.finished", 1).await;
    assert_eq!(events.last().unwrap().data, json!({"answer": "done"}));

    let requests = model.requests();
    let compacted: Vec<&mason::ChatRequest> = requests
        .iter()
        .filter(|request| {
            !is_summary_request(request)
                && request
                    .messages
                    .get(1)
                    .and_then(|m| m.content.as_deref())
                    .is_some_and(|c| c.contains("conversation-checkpoint"))
        })
        .collect();
    assert!(!compacted.is_empty(), "compaction happened during the run");
    for request in compacted {
        assert_eq!(
            request.messages[2],
            mason::Message::user("the actual request")
        );
    }
    for request in requests.iter().filter(|r| is_summary_request(r)) {
        let input = request.messages[1].content.clone().unwrap();
        assert!(!input.contains("the actual request"), "{input}");
    }
    assert!(events.iter().any(|e| e.event_type == "compaction"));

    // History respects the latest checkpoint.
    let prompts = PromptSet::parse(mason::prompt::EMBEDDED_PROMPTS).unwrap();
    let history = store.history(session, &prompts).await.unwrap();
    assert!(
        history[0]
            .content
            .as_deref()
            .unwrap()
            .contains("<conversation-checkpoint>")
    );
}

/// The user message a compaction kept verbatim ahead of its tail must reach a
/// later summary once a newer user message arrives; it used to vanish from
/// the stored history without ever being summarised.
#[tokio::test]
async fn a_kept_user_message_is_summarised_by_the_next_stored_compaction() {
    let (dir, store) = fixture().await;
    let (big, _count) = EchoTool::boxed("big", &"y".repeat(16_000));
    let mains = Arc::new(AtomicUsize::new(0));
    let model = FunctionModel::with_context(
        {
            let mains = mains.clone();
            move |request: &mason::ChatRequest| {
                if is_summary_request(request) {
                    return answer("S");
                }
                let n = mains.fetch_add(1, Ordering::SeqCst) + 1;
                // Four tool rounds per run, then an answer.
                if n.is_multiple_of(5) {
                    return answer("done");
                }
                Reply::Calls(vec![(format!("t{n}"), "big".into(), "{}".into())])
            }
        },
        8_000,
    );
    let mut tools = builtin_tools();
    tools.push(big);
    let agent = Agent::builder()
        .model(model.clone())
        .cwd(dir.path())
        .tools(tools)
        .spill(Arc::new(OutputSpill::new(dir.path().join("spill"))))
        .build()
        .unwrap();
    let runner = Runner::new(agent, store.clone());

    let session = store.create_session(NewSession::default()).await.unwrap();
    store
        .add_input(session, NewInput::queued("the original request"))
        .await
        .unwrap();
    runner.wake(session);
    wait_for_events(&store, session, "run.finished", 1).await;
    let first_run_summaries = model
        .requests()
        .iter()
        .filter(|request| is_summary_request(request))
        .count();
    assert!(first_run_summaries > 0, "the first run compacted");

    store
        .add_input(session, NewInput::queued("continue"))
        .await
        .unwrap();
    runner.wake(session);
    wait_for_events(&store, session, "run.finished", 2).await;

    let summarised = model
        .requests()
        .iter()
        .filter(|request| is_summary_request(request))
        .any(|request| {
            request.messages[1]
                .content
                .as_deref()
                .is_some_and(|input| input.contains("the original request"))
        });
    assert!(summarised, "the kept user message never reached a summary");
    let prompts = PromptSet::parse(mason::prompt::EMBEDDED_PROMPTS).unwrap();
    let history = store.history(session, &prompts).await.unwrap();
    assert_eq!(
        history.last().map(|m| m.content.as_deref()),
        Some(Some("done"))
    );
}

/// Port of the second half of: "compacts before a request that would not
/// fit" — the next compaction merges the previous summary and history
/// respects the latest checkpoint (stored).
#[tokio::test]
async fn stored_compactions_merge_previous_summaries() {
    let (dir, store) = fixture().await;
    let summaries = Arc::new(AtomicUsize::new(0));
    let model = FunctionModel::with_context(
        {
            let summaries = summaries.clone();
            move |request: &mason::ChatRequest| {
                if is_summary_request(request) {
                    answer(&format!(
                        "SUMMARY {}",
                        summaries.fetch_add(1, Ordering::SeqCst) + 1
                    ))
                } else {
                    answer("ok")
                }
            }
        },
        8_000,
    );
    let runner = Runner::new(agent(dir.path(), model.clone()), store.clone());
    let session = store.create_session(NewSession::default()).await.unwrap();
    for mut message in seeded_messages(4, 1_000) {
        if message["message"]["content"] == "q1" {
            message["message"]["content"] = json!(format!("q1 {}", "u".repeat(12_000)));
        }
        store.append(session, "message", message).await.unwrap();
    }
    store
        .add_input(session, NewInput::queued("next question"))
        .await
        .unwrap();
    runner.wake(session);
    wait_for_events(&store, session, "run.finished", 1).await;

    // Grow the conversation past the tail again and send one more input.
    store
        .append(
            session,
            "message",
            json!({"message": {"role": "assistant", "content": null, "tool_calls": [
                {"id": "c9", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]}}),
        )
        .await
        .unwrap();
    store
        .append(
            session,
            "message",
            json!({"message": {"role": "tool", "tool_call_id": "c9", "content": "x".repeat(24_000)}}),
        )
        .await
        .unwrap();
    store
        .add_input(session, NewInput::queued("one more"))
        .await
        .unwrap();
    // The recheck after the first run may have re-woken the session already;
    // wake is idempotent either way.
    runner.wake(session);
    wait_for_events(&store, session, "run.finished", 2).await;

    let requests = model.requests();
    let summary_requests: Vec<&mason::ChatRequest> =
        requests.iter().filter(|r| is_summary_request(r)).collect();
    assert!(summary_requests.len() >= 2, "two compactions happened");
    let second = summary_requests[1].messages[1].content.clone().unwrap();
    assert!(
        second.contains("<prior-summary>\nSUMMARY 1\n</prior-summary>"),
        "{second}"
    );

    let prompts = PromptSet::parse(mason::prompt::EMBEDDED_PROMPTS).unwrap();
    let history = store.history(session, &prompts).await.unwrap();
    let latest = format!("SUMMARY {}", summary_requests.len());
    assert!(
        history[0].content.as_deref().unwrap().contains(&latest),
        "history starts from the latest checkpoint: {}",
        history[0].content.as_deref().unwrap_or_default()
    );
}

#[tokio::test]
async fn subagents_run_in_stored_child_sessions() {
    let (dir, store) = fixture().await;
    let model = ScriptedModel::new(vec![
        call(
            "task",
            r#"{"description":"investigate","prompt":"look into it","agent":"general"}"#,
        ),
        answer("child done"),
        answer("parent done"),
    ]);
    let runner = Runner::new(agent(dir.path(), model.clone()), store.clone());
    let session = store.create_session(NewSession::default()).await.unwrap();
    store
        .add_input(session, NewInput::queued("go"))
        .await
        .unwrap();
    runner.wake(session);
    let events = wait_for_events(&store, session, "run.finished", 1).await;

    // The task result carries the child session id.
    let task_result = events
        .iter()
        .find(|e| {
            e.data.pointer("/message/role").and_then(|v| v.as_str()) == Some("tool")
                && e.data
                    .pointer("/message/content")
                    .and_then(|v| v.as_str())
                    .is_some_and(|c| c.contains("<task session="))
        })
        .expect("a task result with a session id");
    let content = task_result
        .data
        .pointer("/message/content")
        .unwrap()
        .as_str()
        .unwrap();
    let child_id: SessionId = content
        .split("session=\"")
        .nth(1)
        .unwrap()
        .split('"')
        .next()
        .unwrap()
        .parse()
        .unwrap();

    let child = store.get_session(child_id).await.unwrap().unwrap();
    assert_eq!(child.parent_id, Some(session));
    assert_eq!(child.agent, "general");
    assert_eq!(child.title.as_deref(), Some("investigate (@general)"));

    let child_events = store.list_events(child_id, 0).await.unwrap();
    let types: Vec<&str> = child_events.iter().map(|e| e.event_type.as_str()).collect();
    assert_eq!(types, ["message", "run.started", "message", "run.finished"]);
    assert_eq!(
        child_events[0]
            .data
            .pointer("/message/content")
            .and_then(|v| v.as_str()),
        Some("look into it")
    );
    assert_eq!(child_events[3].data, json!({"answer": "child done"}));
    assert!(
        content.contains("<task_result>\nchild done\n</task_result>"),
        "{content}"
    );
}

#[tokio::test]
async fn recovery_after_a_crash_does_not_replay_side_effects() {
    let (dir, store) = fixture().await;

    // Session A: crashed mid-run — inputs consumed, a tool call committed
    // but never answered, no terminal event.
    let abandoned = store.create_session(NewSession::default()).await.unwrap();
    store
        .add_input(abandoned, NewInput::queued("do the work"))
        .await
        .unwrap();
    store.start_run(abandoned).await.unwrap();
    store
        .append(
            abandoned,
            "message",
            json!({"message": {"role": "assistant", "content": null, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": "{\"command\":\"rm -rf /\"}"}}]}}),
        )
        .await
        .unwrap();

    // Session B: an input nobody consumed yet.
    let pending = store.create_session(NewSession::default()).await.unwrap();
    store
        .add_input(pending, NewInput::queued("hello"))
        .await
        .unwrap();

    let model = ScriptedModel::new(vec![answer("fresh run ok")]);
    let runner = Runner::new(agent(dir.path(), model.clone()), store.clone());
    let recovery = runner.recover_and_drain().await.unwrap();
    assert_eq!(recovery.interrupted.len(), 1);
    assert_eq!(recovery.interrupted[0].0, abandoned);

    // The abandoned run was closed; its consumed input never re-ran.
    let events = wait_for_events(&store, abandoned, "run.interrupted", 1).await;
    let started = events
        .iter()
        .filter(|e| e.event_type == "run.started")
        .count();
    assert_eq!(started, 1, "the abandoned inputs were not reclaimed");

    // The pending session drained normally.
    let events = wait_for_events(&store, pending, "run.finished", 1).await;
    assert_eq!(
        events.last().unwrap().data,
        json!({"answer": "fresh run ok"})
    );
    wait_until_idle(&runner, pending).await;

    // The repaired history answers the orphaned call.
    let prompts = PromptSet::parse(mason::prompt::EMBEDDED_PROMPTS).unwrap();
    let history = store.history(abandoned, &prompts).await.unwrap();
    assert_eq!(history.last().unwrap().role, Role::Tool);
    assert_eq!(
        history.last().unwrap().content.as_deref(),
        Some("[Tool execution was interrupted]")
    );
    assert_eq!(
        history[1].tool_calls,
        Some(vec![ToolCall::new(
            "c1",
            "bash",
            "{\"command\":\"rm -rf /\"}"
        )])
    );
}

#[tokio::test]
async fn shutdown_is_bounded_and_persists_terminal_events() {
    let (dir, store) = fixture().await;
    let model = ScriptedModel::new(vec![Reply::Hang]);
    let runner = Runner::new(agent(dir.path(), model), store.clone());
    let session = store.create_session(NewSession::default()).await.unwrap();
    store
        .add_input(session, NewInput::queued("go"))
        .await
        .unwrap();
    runner.wake(session);
    wait_for_events(&store, session, "run.started", 1).await;

    let started = std::time::Instant::now();
    runner.shutdown(Duration::from_secs(5)).await;
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "shutdown is bounded"
    );
    assert!(!runner.is_busy(session));

    let events = store.list_events(session, 0).await.unwrap();
    assert!(
        events.iter().any(|e| e.event_type == "run.interrupted"),
        "{events:?}"
    );
}

#[tokio::test]
async fn shutdown_prevents_new_wakes_from_starting_work() {
    let (dir, store) = fixture().await;
    let model = ScriptedModel::new(vec![answer("must stay queued")]);
    let runner = Runner::new(agent(dir.path(), model.clone()), store.clone());
    runner.shutdown(Duration::from_secs(1)).await;

    let session = store.create_session(NewSession::default()).await.unwrap();
    store
        .add_input(session, NewInput::queued("go"))
        .await
        .unwrap();
    runner.wake(session);

    assert!(!runner.is_busy(session));
    assert_eq!(model.request_count(), 0);
    assert!(store.list_events(session, 0).await.unwrap().is_empty());
    assert!(store.pending_sessions().await.unwrap().contains(&session));
}

#[tokio::test]
async fn wakes_deduplicate_and_deltas_are_broadcast() {
    let (dir, store) = fixture().await;
    let gate = Arc::new(Notify::new());
    let model = ScriptedModel::new(vec![Reply::Gate(
        gate.clone(),
        Box::new(answer("streamed answer")),
    )]);
    let runner = Runner::new(agent(dir.path(), model), store.clone());
    let session = store.create_session(NewSession::default()).await.unwrap();
    let mut deltas = runner.subscribe_deltas(session);
    store
        .add_input(session, NewInput::queued("go"))
        .await
        .unwrap();

    runner.wake(session);
    runner.wake(session); // deduplicated
    wait_for_events(&store, session, "run.started", 1).await;
    assert!(runner.is_busy(session));
    gate.notify_one();

    wait_for_events(&store, session, "run.finished", 1).await;
    wait_until_idle(&runner, session).await;
    // Let the delta relay finish forwarding before draining.
    tokio::time::sleep(Duration::from_millis(100)).await;

    let mut texts = Vec::new();
    let mut committed = 0;
    let mut stream_run_ids = std::collections::HashSet::new();
    while let Ok(event) = deltas.try_recv() {
        match &*event {
            mason::RunEvent::Delta {
                run_id,
                delta: mason::ChatDelta::Text(text),
                ..
            } => {
                texts.push(text.clone());
                stream_run_ids.insert(run_id.to_string());
            }
            mason::RunEvent::Committed { run_id, turn, .. } => {
                committed += 1;
                assert_eq!(*turn, 1, "the committed message names its turn");
                stream_run_ids.insert(run_id.to_string());
            }
            _ => {}
        }
    }
    assert_eq!(texts.concat(), "streamed answer");
    assert!(committed >= 1, "committed messages are observable too");
    let events = store.list_events(session, 0).await.unwrap();
    assert_eq!(
        events
            .iter()
            .filter(|e| e.event_type == "run.started")
            .count(),
        1
    );
    // Every streamed event carries the claimed run's id.
    let run_id = events
        .iter()
        .find(|e| e.event_type == "run.started")
        .and_then(|e| e.run_id.clone())
        .expect("run.started carries its run id");
    assert_eq!(
        stream_run_ids,
        [run_id.clone()].into_iter().collect(),
        "streamed events carry the claimed run id"
    );
    // Stored message events — the claimed user input and the persisted
    // assistant reply — are stamped with the same run id.
    for role in ["user", "assistant"] {
        let message = events
            .iter()
            .find(|e| {
                e.event_type == "message"
                    && e.data.pointer("/message/role").and_then(|v| v.as_str()) == Some(role)
            })
            .unwrap_or_else(|| panic!("a {role} message event"));
        assert_eq!(
            message.run_id.as_deref(),
            Some(run_id.as_str()),
            "the {role} message event carries the run id"
        );
        if role == "assistant" {
            assert_eq!(
                message.data.get("turn").and_then(|value| value.as_u64()),
                Some(1),
                "the durable assistant event carries its turn"
            );
        }
    }
}

/// A domain profile with one tool, standing in for an unattended triage agent.
fn triage_profile() -> Profile {
    Profile {
        name: "triage".into(),
        description: "Unattended investigation.".into(),
        prompt: Some("Investigate read-only and propose fixes.".into()),
        tools: Some(vec!["read".to_owned()]),
        ..Profile::new(String::new(), String::new())
    }
}

fn selector_agent(
    dir: &std::path::Path,
    model: Arc<dyn mason::ChatModel>,
    select: impl Fn(&mason::Session) -> Option<String> + Send + Sync + 'static,
) -> Agent {
    Agent::builder()
        .model(model)
        .cwd(dir)
        .tools(builtin_tools())
        .profiles([triage_profile()])
        .profile_selector(select)
        .spill(Arc::new(OutputSpill::new(dir.join("spill"))))
        .build()
        .unwrap()
}

#[tokio::test]
async fn a_profile_selector_overrides_the_stored_agent_from_metadata() {
    let (dir, store) = fixture().await;
    let model = ScriptedModel::new(vec![answer("triaged")]);
    let agent = selector_agent(dir.path(), model.clone(), |session| {
        session
            .metadata
            .get("site_id")
            .is_some_and(|value| !value.is_null())
            .then(|| "triage".to_owned())
    });
    let runner = Runner::new(agent, store.clone());
    let session = store
        .create_session(NewSession {
            metadata: json!({"site_id": 7}),
            ..Default::default()
        })
        .await
        .unwrap();
    store
        .add_input(session, NewInput::queued("look into it"))
        .await
        .unwrap();

    runner.wake(session);
    wait_for_events(&store, session, "run.finished", 1).await;

    // The stored row keeps its own agent; the run used the selected profile.
    assert_eq!(
        store.get_session(session).await.unwrap().unwrap().agent,
        "build"
    );
    let request = &model.requests()[0];
    let system = request.messages[0].content.as_deref().unwrap_or_default();
    assert!(
        system.contains("Investigate read-only and propose fixes."),
        "the selected profile's prompt reached the model: {system}"
    );
    let tools: Vec<&str> = request
        .tools
        .iter()
        .map(|tool| tool.function.name.as_str())
        .collect();
    assert_eq!(tools, ["read"], "the selected profile narrowed the tools");
}

#[tokio::test]
async fn a_session_without_selector_metadata_keeps_its_stored_agent() {
    let (dir, store) = fixture().await;
    let model = ScriptedModel::new(vec![answer("built")]);
    let agent = selector_agent(dir.path(), model.clone(), |session| {
        session
            .metadata
            .get("site_id")
            .is_some_and(|value| !value.is_null())
            .then(|| "triage".to_owned())
    });
    let runner = Runner::new(agent, store.clone());
    let session = store.create_session(NewSession::default()).await.unwrap();
    store
        .add_input(session, NewInput::queued("hello"))
        .await
        .unwrap();

    runner.wake(session);
    wait_for_events(&store, session, "run.finished", 1).await;

    let request = &model.requests()[0];
    let system = request.messages[0].content.as_deref().unwrap_or_default();
    assert!(
        !system.contains("Investigate read-only"),
        "the default profile ran: {system}"
    );
    assert!(
        request
            .tools
            .iter()
            .any(|tool| tool.function.name == "bash"),
        "the default profile kept the base tools"
    );
}

#[tokio::test]
async fn an_unknown_selected_profile_fails_the_run_with_a_diagnostic() {
    let (dir, store) = fixture().await;
    let model = ScriptedModel::new(vec![answer("unreachable")]);
    let agent = selector_agent(dir.path(), model.clone(), |_| Some("nope".to_owned()));
    let runner = Runner::new(agent, store.clone());
    let session = store.create_session(NewSession::default()).await.unwrap();
    store
        .add_input(session, NewInput::queued("hello"))
        .await
        .unwrap();

    runner.wake(session);
    let events = wait_for_events(&store, session, "run.failed", 1).await;
    let failure = events
        .iter()
        .find(|e| e.event_type == "run.failed")
        .unwrap();
    let error = failure.data["error"].as_str().unwrap_or_default();
    assert!(error.contains("Unknown agent \"nope\""), "{error}");
    assert_eq!(model.request_count(), 0, "no model call was made");
    wait_until_idle(&runner, session).await;
}

#[tokio::test]
async fn background_processes_outlive_a_sessions_run_until_shutdown() {
    let (dir, store) = fixture().await;
    let model = ScriptedModel::new(vec![
        tool_call_json("(sleep 1; touch survived; sleep 2; touch late) & echo started"),
        answer("started"),
        call("process", r#"{"action":"list"}"#),
        answer("listed"),
    ]);
    let runner = Runner::new(agent(dir.path(), model.clone()), store.clone());
    let session = store.create_session(NewSession::default()).await.unwrap();
    store
        .add_input(session, NewInput::queued("start it"))
        .await
        .unwrap();
    runner.wake(session);
    wait_for_events(&store, session, "run.finished", 1).await;
    wait_until_idle(&runner, session).await;

    // The run has ended, yet its background job keeps going.
    tokio::time::sleep(Duration::from_millis(1_500)).await;
    assert!(dir.path().join("survived").exists());

    // The next message's run still owns it.
    store
        .add_input(session, NewInput::queued("list them"))
        .await
        .unwrap();
    runner.wake(session);
    let events = wait_for_events(&store, session, "run.finished", 2).await;
    let listed = events
        .iter()
        .filter_map(|event| event.data.pointer("/message/content")?.as_str())
        .find(|content| content.starts_with('['))
        .unwrap();
    assert_ne!(listed, "[]", "the session still owns the process");

    // Shutdown stops it before it can finish.
    runner.shutdown(Duration::from_secs(1)).await;
    tokio::time::sleep(Duration::from_millis(2_500)).await;
    assert!(!dir.path().join("late").exists());
}

#[tokio::test]
async fn a_running_subagent_is_announced_listed_and_interruptible() {
    let (dir, store) = fixture().await;
    let model = ScriptedModel::new(vec![
        call(
            "task",
            r#"{"description":"investigate","prompt":"look into it","agent":"general"}"#,
        ),
        Reply::Hang,
        answer("parent done"),
    ]);
    let runner = Runner::new(agent(dir.path(), model.clone()), store.clone());
    let session = store.create_session(NewSession::default()).await.unwrap();
    store
        .add_input(session, NewInput::queued("go"))
        .await
        .unwrap();
    runner.wake(session);

    // The parent announces the child while it is still running.
    let events = wait_for_events(&store, session, "subagent.started", 1).await;
    let started = events
        .iter()
        .find(|event| event.event_type == "subagent.started")
        .unwrap();
    let child = started.data["sessionId"].as_u64().unwrap();
    assert_eq!(started.data["agent"], "general");
    let children = store
        .list_sessions(&mason::store::SessionQuery {
            parent_id: Some(session),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(children.iter().map(|s| s.id).collect::<Vec<_>>(), [child]);
    assert!(runner.is_busy(child));

    // Interrupting only the child, once it is waiting on the model, lets the parent carry on.
    for _ in 0..1_500 {
        if model.request_count() == 2 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    runner.interrupt(child);
    let events = wait_for_events(&store, session, "run.finished", 1).await;
    assert_eq!(
        events.last().unwrap().data,
        json!({"answer": "parent done"})
    );
}

type Deliveries = Arc<tokio::sync::Mutex<Vec<(String, serde_json::Value)>>>;

/// A webhook receiver: its url, and the (authorization, body) pairs it got.
async fn webhook_receiver() -> (String, Deliveries) {
    let received: Deliveries = Arc::default();
    let hook = {
        let received = received.clone();
        axum::Router::new().route(
            "/runs",
            axum::routing::post(
                move |headers: axum::http::HeaderMap,
                      axum::Json(body): axum::Json<serde_json::Value>| {
                    let received = received.clone();
                    async move {
                        let auth = headers
                            .get("authorization")
                            .and_then(|v| v.to_str().ok())
                            .unwrap_or_default()
                            .to_owned();
                        received.lock().await.push((auth, body));
                    }
                },
            ),
        )
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/runs", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, hook).await });
    (url, received)
}

async fn wait_for_deliveries(received: &Deliveries, count: usize) {
    for _ in 0..100 {
        if received.lock().await.len() >= count {
            return;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("the webhook got fewer than {count} deliveries");
}

#[tokio::test]
async fn a_finished_run_reports_its_outcome_on_its_inputs_and_to_the_webhook() {
    let (dir, store) = fixture().await;
    let (url, received) = webhook_receiver().await;

    let model = ScriptedModel::new(vec![answer("triaged")]);
    let runner = Runner::new(agent(dir.path(), model), store.clone());
    runner.notify_runs(mason::runner::RunWebhook {
        url,
        secret: Some("secret-token".into()),
    });
    let session = store
        .create_session(NewSession {
            metadata: json!({"site_id": 7}),
            ..Default::default()
        })
        .await
        .unwrap();
    store
        .add_input(
            session,
            NewInput {
                key: Some("issue-12".into()),
                ..NewInput::queued("look at issue 12")
            },
        )
        .await
        .unwrap();
    runner.wake(session);
    wait_for_events(&store, session, "run.finished", 1).await;

    let [input] = &store.list_inputs(session).await.unwrap()[..] else {
        panic!("one input")
    };
    assert_eq!(input.outcome.as_deref(), Some("finished"));
    let run_id = input.run_id.clone().expect("the run it entered");

    wait_for_deliveries(&received, 1).await;
    let received = received.lock().await;
    let (auth, body) = &received[0];
    assert_eq!(auth, "Bearer secret-token");
    assert_eq!(body["type"], "run.finished");
    assert_eq!(body["runId"], run_id.as_str());
    assert_eq!(body["metadata"], json!({"site_id": 7}));
    assert_eq!(body["inputs"], json!([{"id": input.id, "key": "issue-12"}]));
    assert_eq!(body["data"], json!({"answer": "triaged"}));
}

#[tokio::test]
async fn a_restart_reports_the_runs_it_found_abandoned_to_the_webhook() {
    let (dir, store) = fixture().await;
    let (url, received) = webhook_receiver().await;
    let abandoned = store.create_session(NewSession::default()).await.unwrap();
    let input_id = store
        .add_input(abandoned, NewInput::queued("do the work"))
        .await
        .unwrap();
    let claim = store.start_run(abandoned).await.unwrap();

    let model = ScriptedModel::new(vec![]);
    let runner = Runner::new(agent(dir.path(), model), store.clone());
    runner.notify_runs(mason::runner::RunWebhook { url, secret: None });
    let recovery = runner.recover_and_drain().await.unwrap();
    assert_eq!(recovery.interrupted.len(), 1);

    wait_for_deliveries(&received, 1).await;
    let received = received.lock().await;
    let (auth, body) = &received[0];
    assert_eq!(auth, "", "no secret, no authorization header");
    assert_eq!(body["type"], "run.interrupted");
    assert_eq!(body["sessionId"], abandoned);
    assert_eq!(body["runId"], claim.run_id.unwrap().as_str());
    assert_eq!(body["inputs"], json!([{"id": input_id, "key": null}]));
    assert_eq!(
        body["data"],
        json!({"error": "the process restarted while this run was active"})
    );
}
