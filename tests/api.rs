//! HTTP API integration tests using real sockets and bounded waits.

mod support;

use std::sync::Arc;
use std::time::Duration;

use futures_util::{Stream, StreamExt};
use mason::store::{Event as StoreEvent, NewSession, SessionId, SessionStore, SqliteStore};
use mason::{Agent, OutputSpill, Profile, Runner};
use serde_json::{Value, json};
use support::{Reply, ScriptedModel, answer, call};

struct Fixture {
    _dir: tempfile::TempDir,
    store: Arc<SqliteStore>,
    runner: Arc<Runner>,
    base: String,
    client: reqwest::Client,
}

async fn fixture(model: Arc<ScriptedModel>) -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(
        SqliteStore::open(dir.path().join("mason.db"))
            .await
            .unwrap(),
    );
    let agent = Agent::builder()
        .model(model)
        .cwd(dir.path())
        .tools(mason::builtin_tools())
        .profiles([Profile::new("research", "research-only agent")])
        .spill(Arc::new(OutputSpill::new(dir.path().join("spill"))))
        .build()
        .unwrap();
    let runner = Runner::new(agent, store.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let serve_runner = runner.clone();
    tokio::spawn(async move {
        axum::serve(listener, mason::api::router(serve_runner))
            .await
            .unwrap()
    });
    Fixture {
        _dir: dir,
        store,
        runner,
        base,
        client: reqwest::Client::new(),
    }
}

/// A minimal SSE frame reader over the response byte stream.
struct SseReader {
    buffer: String,
    stream: std::pin::Pin<Box<dyn Stream<Item = reqwest::Result<bytes::Bytes>> + Send>>,
}

#[derive(Debug, Default, Clone)]
struct Frame {
    id: Option<String>,
    event: Option<String>,
    data: Option<String>,
    comment: bool,
}

impl Frame {
    fn parse(text: &str) -> Self {
        let mut frame = Frame::default();
        for line in text.lines() {
            if let Some(value) = line.strip_prefix("id: ") {
                frame.id = Some(value.to_owned());
            } else if let Some(value) = line.strip_prefix("event: ") {
                frame.event = Some(value.to_owned());
            } else if let Some(value) = line.strip_prefix("data: ") {
                frame.data = Some(match frame.data.take() {
                    Some(previous) => format!("{previous}\n{value}"),
                    None => value.to_owned(),
                });
            } else if line.starts_with(':') {
                frame.comment = true;
            }
        }
        frame
    }

    fn json(&self) -> Value {
        serde_json::from_str(self.data.as_deref().unwrap_or("null")).unwrap_or(Value::Null)
    }
}

impl SseReader {
    async fn connect(client: &reqwest::Client, url: &str, last_event_id: Option<&str>) -> Self {
        let mut request = client
            .get(url)
            .header(reqwest::header::ACCEPT, "text/event-stream");
        if let Some(id) = last_event_id {
            request = request.header("last-event-id", id);
        }
        let response = request.send().await.unwrap();
        assert!(response.status().is_success(), "{}", response.status());
        Self {
            buffer: String::new(),
            stream: Box::pin(response.bytes_stream()),
        }
    }

    /// The next non-comment frame, bounded so a broken stream fails the test.
    async fn next_frame(&mut self) -> Frame {
        loop {
            if let Some(pos) = self.buffer.find("\n\n") {
                let text = self.buffer[..pos].to_owned();
                self.buffer.drain(..pos + 2);
                let frame = Frame::parse(&text);
                if frame.comment && frame.data.is_none() && frame.event.is_none() {
                    continue; // heartbeat
                }
                return frame;
            }
            let chunk = tokio::time::timeout(Duration::from_secs(15), self.stream.next())
                .await
                .expect("the SSE stream stalled")
                .expect("the SSE stream ended unexpectedly")
                .expect("the SSE stream failed");
            self.buffer.push_str(&String::from_utf8_lossy(&chunk));
        }
    }

    /// Reads until the server closes the stream; returns every frame.
    async fn until_closed(&mut self) -> Vec<Frame> {
        let mut frames = Vec::new();
        loop {
            if let Some(pos) = self.buffer.find("\n\n") {
                let text = self.buffer[..pos].to_owned();
                self.buffer.drain(..pos + 2);
                let frame = Frame::parse(&text);
                if frame.comment && frame.data.is_none() && frame.event.is_none() {
                    continue;
                }
                frames.push(frame);
                continue;
            }
            match tokio::time::timeout(Duration::from_secs(15), self.stream.next()).await {
                Ok(Some(Ok(chunk))) => self.buffer.push_str(&String::from_utf8_lossy(&chunk)),
                Ok(_) => return frames, // EOF: the server closed
                Err(_) => panic!("the SSE stream never closed"),
            }
        }
    }
}

async fn wait_for_event(
    store: &Arc<SqliteStore>,
    session: SessionId,
    event_type: &str,
    count: usize,
) -> Vec<StoreEvent> {
    for _ in 0..1_500 {
        let events = store.list_events(session, 0).await.unwrap();
        if events.iter().filter(|e| e.event_type == event_type).count() >= count {
            return events;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("timed out waiting for {count} `{event_type}` events");
}

#[tokio::test]
async fn session_crud_and_validation() {
    let fx = fixture(ScriptedModel::new(vec![])).await;

    // POST /sessions validates the profile and rejects subagent profiles.
    let response = fx
        .client
        .post(format!("{}/sessions", fx.base))
        .json(&json!({"agent": "nope"}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 400);
    let body: Value = response.json().await.unwrap();
    assert!(
        body["error"].as_str().unwrap().contains("Unknown agent"),
        "{body}"
    );

    let response = fx
        .client
        .post(format!("{}/sessions", fx.base))
        .json(&json!({"agent": "explore"}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 400);
    let body: Value = response.json().await.unwrap();
    assert!(
        body["error"].as_str().unwrap().contains("subagent profile"),
        "{body}"
    );

    let response = fx
        .client
        .post(format!("{}/sessions", fx.base))
        .json(&json!({"title": "demo", "agent": "research"}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 201);
    let body: Value = response.json().await.unwrap();
    let id = body["id"].as_u64().unwrap();

    let response = fx
        .client
        .get(format!("{}/sessions/{id}", fx.base))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["agent"], "research");
    assert_eq!(body["title"], "demo");
    assert_eq!(body["busy"], false);
    assert_eq!(body["parentId"], Value::Null);

    assert_eq!(
        fx.client
            .get(format!("{}/sessions/999999", fx.base))
            .send()
            .await
            .unwrap()
            .status(),
        404
    );
    assert_eq!(
        fx.client
            .post(format!("{}/sessions/999999/interrupt", fx.base))
            .send()
            .await
            .unwrap()
            .status(),
        404
    );

    // POST /sessions/:id/messages validates content and delivery.
    let response = fx
        .client
        .post(format!("{}/sessions/{id}/messages", fx.base))
        .json(&json!({"content": ""}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 400);
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["error"], "content is required");

    let response = fx
        .client
        .post(format!("{}/sessions/{id}/messages", fx.base))
        .json(&json!({"content": "x", "delivery": "sideways"}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 400);
    let body: Value = response.json().await.unwrap();
    let error = body["error"].as_str().unwrap();
    assert!(error.contains("queue") && error.contains("steer"), "{body}");

    let response = fx
        .client
        .post(format!("{}/sessions/{id}/messages", fx.base))
        .json(&json!({"content": "hello", "delivery": "steer", "key": "k1"}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 202);
    let first: Value = response.json().await.unwrap();
    let response = fx
        .client
        .post(format!("{}/sessions/{id}/messages", fx.base))
        .json(&json!({"content": "hello", "delivery": "steer", "key": "k1"}))
        .send()
        .await
        .unwrap();
    let again: Value = response.json().await.unwrap();
    assert_eq!(
        again["inputId"], first["inputId"],
        "the repeated key is accepted once"
    );

    // Interrupt answers 204.
    assert_eq!(
        fx.client
            .post(format!("{}/sessions/{id}/interrupt", fx.base))
            .send()
            .await
            .unwrap()
            .status(),
        204
    );
}

#[tokio::test]
async fn list_sessions_filters_and_reflects_busy() {
    let gate = Arc::new(tokio::sync::Notify::new());
    let model = ScriptedModel::new(vec![Reply::Gate(gate.clone(), Box::new(answer("ok")))]);
    let fx = fixture(model).await;

    let a = fx
        .client
        .post(format!("{}/sessions", fx.base))
        .json(&json!({"title": "first"}))
        .send()
        .await
        .unwrap();
    let a: u64 = a.json::<Value>().await.unwrap()["id"].as_u64().unwrap();
    fx.store
        .append(
            a,
            "message",
            json!({"message": {"role": "assistant", "content": "hi"}}),
        )
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(5)).await;
    let b = fx
        .client
        .post(format!("{}/sessions", fx.base))
        .json(&json!({"title": "second", "agent": "research"}))
        .send()
        .await
        .unwrap();
    let b: u64 = b.json::<Value>().await.unwrap()["id"].as_u64().unwrap();
    fx.store
        .create_session(NewSession {
            title: Some("child".into()),
            parent_id: Some(a),
            agent: Some("explore".into()),
            ..Default::default()
        })
        .await
        .unwrap();

    let sessions: Value = fx
        .client
        .get(format!("{}/sessions", fx.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let rows: Vec<(u64, String, String)> = sessions["sessions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| {
            (
                s["id"].as_u64().unwrap(),
                s["title"].as_str().unwrap().to_owned(),
                s["agent"].as_str().unwrap().to_owned(),
            )
        })
        .collect();
    assert_eq!(
        rows,
        vec![
            (b, "second".into(), "research".into()),
            (a, "first".into(), "build".into())
        ],
        "children are not listed"
    );
    assert_eq!(sessions["sessions"][0]["busy"], false);
    assert_eq!(sessions["sessions"][0]["lastEventAt"], Value::Null);
    assert_eq!(sessions["sessions"][1]["lastEventType"], "message");

    let filtered: Value = fx
        .client
        .get(format!("{}/sessions?agent=research", fx.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let ids: Vec<u64> = filtered["sessions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["id"].as_u64().unwrap())
        .collect();
    assert_eq!(ids, [b]);

    let none: Value = fx
        .client
        .get(format!("{}/sessions?before=1970-01-01T00:00:00Z", fx.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(none["sessions"].as_array().unwrap().is_empty());

    let limited: Value = fx
        .client
        .get(format!("{}/sessions?limit=1", fx.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let ids: Vec<u64> = limited["sessions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["id"].as_u64().unwrap())
        .collect();
    assert_eq!(ids, [b]);

    assert_eq!(
        fx.client
            .get(format!("{}/sessions?before=nope", fx.base))
            .send()
            .await
            .unwrap()
            .status(),
        400
    );

    // Busy: a gated run in flight.
    fx.client
        .post(format!("{}/sessions/{a}/messages", fx.base))
        .json(&json!({"content": "go"}))
        .send()
        .await
        .unwrap();
    wait_for_event(&fx.store, a, "run.started", 1).await;
    let one: Value = fx
        .client
        .get(format!("{}/sessions/{a}", fx.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(one["busy"], true);
    gate.notify_one();
    wait_for_event(&fx.store, a, "run.finished", 1).await;
    for _ in 0..1_500 {
        if !fx.runner.is_busy(a) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let one: Value = fx
        .client
        .get(format!("{}/sessions/{a}", fx.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(one["busy"], false);
}

#[tokio::test]
async fn the_event_stream_replays_from_a_cursor_then_follows_live() {
    let gate = Arc::new(tokio::sync::Notify::new());
    let model = ScriptedModel::new(vec![Reply::Gate(
        gate.clone(),
        Box::new(answer("hi there")),
    )]);
    let fx = fixture(model).await;
    let session = fx
        .store
        .create_session(NewSession::default())
        .await
        .unwrap();

    fx.client
        .post(format!("{}/sessions/{session}/messages", fx.base))
        .json(&json!({"content": "hello"}))
        .send()
        .await
        .unwrap();
    let events = wait_for_event(&fx.store, session, "run.started", 1).await;
    let user_message_id = events[0].id;

    let mut reader = SseReader::connect(
        &fx.client,
        &format!(
            "{}/sessions/{session}/events?after={user_message_id}",
            fx.base
        ),
        None,
    )
    .await;
    // Let the replay happen before the run continues.
    tokio::time::sleep(Duration::from_millis(100)).await;
    gate.notify_one();

    let mut frames = Vec::new();
    loop {
        let frame = reader.next_frame().await;
        let is_finished = frame.event.as_deref() == Some("run.finished");
        frames.push(frame);
        if is_finished {
            break;
        }
    }
    // Filter transient frames before asserting the committed event sequence.
    let committed: Vec<&Frame> = frames
        .iter()
        .filter(|f| {
            !matches!(
                f.event.as_deref(),
                Some("delta") | Some("committed") | Some("tool")
            )
        })
        .collect();
    let types: Vec<&str> = committed
        .iter()
        .map(|f| f.event.as_deref().unwrap())
        .collect();
    assert_eq!(types, ["run.started", "message", "run.finished"]);
    assert!(
        committed[0].id.as_deref().unwrap().parse::<u64>().unwrap() > user_message_id,
        "the replay starts after the cursor"
    );
    assert_eq!(
        committed[2]
            .json()
            .pointer("/data/answer")
            .and_then(Value::as_str),
        Some("hi there")
    );
    // Committed store frames carry `id:` for Last-Event-ID reconnects.
    assert!(committed.iter().all(|f| f.id.is_some()));
}

#[tokio::test]
async fn reconnect_honours_last_event_id_and_explicit_after() {
    let fx = fixture(ScriptedModel::new(vec![])).await;
    let session = fx
        .store
        .create_session(NewSession::default())
        .await
        .unwrap();
    for n in 0..5 {
        fx.store
            .append(
                session,
                "message",
                json!({"message": {"role": "user", "content": n.to_string()}}),
            )
            .await
            .unwrap();
    }

    let mut reader = SseReader::connect(
        &fx.client,
        &format!("{}/sessions/{session}/events", fx.base),
        Some("3"),
    )
    .await;
    let mut ids = Vec::new();
    for _ in 0..2 {
        ids.push(
            reader
                .next_frame()
                .await
                .id
                .unwrap()
                .parse::<u64>()
                .unwrap(),
        );
    }
    assert_eq!(ids, [4, 5], "Last-Event-ID resumes after event 3");
    drop(reader);

    let mut reader = SseReader::connect(
        &fx.client,
        &format!("{}/sessions/{session}/events?after=2", fx.base),
        Some("4"),
    )
    .await;
    let mut ids = Vec::new();
    for _ in 0..3 {
        ids.push(
            reader
                .next_frame()
                .await
                .id
                .unwrap()
                .parse::<u64>()
                .unwrap(),
        );
    }
    assert_eq!(ids, [3, 4, 5], "an explicit after overrides Last-Event-ID");
}

#[tokio::test]
async fn a_slow_viewer_replays_persisted_events_then_closes() {
    let fx = fixture(ScriptedModel::new(vec![])).await;
    let session = fx
        .store
        .create_session(NewSession::default())
        .await
        .unwrap();
    let mut reader = SseReader::connect(
        &fx.client,
        &format!("{}/sessions/{session}/events", fx.base),
        None,
    )
    .await;

    // Overwhelm every buffer between the store and the viewer (broadcast
    // window 256, SSE queue 256, socket buffers) while it reads nothing.
    const EVENTS: usize = 5_000;
    for n in 0..EVENTS {
        fx.store
            .append(
                session,
                "message",
                json!({"message": {"role": "user", "content": format!("{n} {}", "x".repeat(4096))}}),
            )
            .await
            .unwrap();
    }

    let frames = reader.until_closed().await;
    let ids: Vec<u64> = frames
        .iter()
        .filter(|f| f.event.as_deref() == Some("message"))
        .map(|f| f.id.as_deref().unwrap().parse::<u64>().unwrap())
        .collect();
    let mut sorted = ids.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        EVENTS,
        "every persisted event arrives exactly once"
    );
    assert_eq!(
        ids, sorted,
        "in strictly ascending cursor order, no gaps or duplicates"
    );
    // The stream then closed (until_closed returned), inviting a fresh
    // subscription.
}

#[tokio::test]
async fn deltas_are_transient_and_committed_messages_carry_cursors() {
    let model = ScriptedModel::new(vec![Reply::Deltas(vec![
        mason::ChatDelta::Text("streamed ".into()),
        mason::ChatDelta::Text("answer".into()),
    ])]);
    let fx = fixture(model).await;
    let session = fx
        .store
        .create_session(NewSession::default())
        .await
        .unwrap();
    let mut reader = SseReader::connect(
        &fx.client,
        &format!("{}/sessions/{session}/events", fx.base),
        None,
    )
    .await;
    fx.client
        .post(format!("{}/sessions/{session}/messages", fx.base))
        .json(&json!({"content": "go"}))
        .send()
        .await
        .unwrap();

    let mut frames = Vec::new();
    loop {
        let frame = reader.next_frame().await;
        let finished = frame.event.as_deref() == Some("run.finished");
        frames.push(frame);
        if finished {
            break;
        }
    }
    let deltas: Vec<&Frame> = frames
        .iter()
        .filter(|f| f.event.as_deref() == Some("delta"))
        .collect();
    assert!(!deltas.is_empty(), "the run streamed deltas");
    for delta in &deltas {
        assert!(delta.id.is_none(), "deltas carry no id: {delta:?}");
        let body = delta.json();
        assert_eq!(body["kind"], "delta");
        assert!(body["content"]["turn"].is_number(), "{body}");
        assert!(body["content"]["seq"].is_number(), "{body}");
        assert!(
            body["content"]["runId"]
                .as_str()
                .is_some_and(|id| !id.is_empty()),
            "deltas carry their run id: {body}"
        );
    }
    // The streamed run id is the stored run's id: reconnecting viewers and committed
    // events reconcile through it.
    let stored = fx.store.list_events(session, 0).await.unwrap();
    let run_id = stored
        .iter()
        .find(|event| event.event_type == "run.started")
        .and_then(|event| event.run_id.clone())
        .expect("run.started carries its run id");
    for delta in &deltas {
        assert_eq!(
            delta.json()["content"]["runId"].as_str(),
            Some(run_id.as_str()),
            "every delta belongs to the claimed run"
        );
    }
    let texts: Vec<String> = deltas
        .iter()
        .map(|d| d.json()["content"]["delta"].clone())
        .filter_map(|delta: Value| {
            if delta["kind"] == "text" {
                delta["content"].as_str().map(str::to_owned)
            } else {
                None
            }
        })
        .collect();
    assert_eq!(texts.concat(), "streamed answer");
    // The committed assistant message supersedes the provisional text.
    let committed = frames
        .iter()
        .find(|f| {
            f.event.as_deref() == Some("message")
                && f.json()
                    .pointer("/data/message/role")
                    .and_then(Value::as_str)
                    == Some("assistant")
                && f.json().pointer("/data/message/content").is_some()
        })
        .expect("a committed assistant message");
    assert!(committed.id.is_some());
    assert_eq!(
        committed
            .json()
            .pointer("/data/message/content")
            .and_then(Value::as_str),
        Some("streamed answer")
    );
}

#[tokio::test]
async fn interrupting_over_http_ends_the_run() {
    let model = ScriptedModel::new(vec![call("bash", r#"{"command":"sleep 30"}"#)]);
    let fx = fixture(model).await;
    let session = fx
        .store
        .create_session(NewSession::default())
        .await
        .unwrap();
    let mut reader = SseReader::connect(
        &fx.client,
        &format!("{}/sessions/{session}/events", fx.base),
        None,
    )
    .await;
    fx.client
        .post(format!("{}/sessions/{session}/messages", fx.base))
        .json(&json!({"content": "wait"}))
        .send()
        .await
        .unwrap();

    // Wait for the assistant tool-call message, then interrupt.
    loop {
        let events = fx.store.list_events(session, 0).await.unwrap();
        if events
            .iter()
            .any(|e| e.data.pointer("/message/tool_calls").is_some())
        {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(
        fx.client
            .post(format!("{}/sessions/{session}/interrupt", fx.base))
            .send()
            .await
            .unwrap()
            .status(),
        204
    );

    let mut saw_interrupted = false;
    for _ in 0..40 {
        let frame = reader.next_frame().await;
        if frame.event.as_deref() == Some("run.interrupted") {
            saw_interrupted = true;
            break;
        }
    }
    assert!(saw_interrupted, "the SSE feed shows the terminal event");
    let prompts = mason::PromptSet::parse(mason::prompt::EMBEDDED_PROMPTS).unwrap();
    let history = fx.store.history(session, &prompts).await.unwrap();
    assert_eq!(
        history.last().unwrap().role,
        mason::Role::Tool,
        "the interrupted call is answered"
    );
}

#[tokio::test]
async fn events_without_sse_accept_returns_a_json_list() {
    let fx = fixture(ScriptedModel::new(vec![])).await;
    let session = fx
        .store
        .create_session(NewSession::default())
        .await
        .unwrap();
    for n in 0..3 {
        fx.store
            .append(
                session,
                "message",
                json!({"message": {"role": "user", "content": n.to_string()}}),
            )
            .await
            .unwrap();
    }
    let list: Value = fx
        .client
        .get(format!("{}/sessions/{session}/events?after=1", fx.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let array = list.as_array().unwrap();
    assert_eq!(array.len(), 2);
    assert_eq!(array[0]["type"], "message");
    assert!(array[0]["id"].as_u64().unwrap() > 1);
    assert_eq!(array[0]["sessionId"], session);

    let page: Value = fx
        .client
        .get(format!("{}/sessions/{session}/events?limit=2", fx.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let page = page.as_array().unwrap();
    assert_eq!(page.len(), 2, "limit pages the list");
    assert_eq!(page[0]["data"]["message"]["content"], "0");
}

#[tokio::test]
async fn session_metadata_and_the_inbox_are_readable_over_http() {
    let fx = fixture(ScriptedModel::new(vec![answer("done")])).await;

    let id: u64 = fx
        .client
        .post(format!("{}/sessions", fx.base))
        .json(&json!({
            "title": "shop triage",
            "metadata": {"site_id": 7},
        }))
        .send()
        .await
        .unwrap()
        .json::<Value>()
        .await
        .unwrap()["id"]
        .as_u64()
        .unwrap();
    let plain: u64 = fx
        .client
        .post(format!("{}/sessions", fx.base))
        .json(&json!({}))
        .send()
        .await
        .unwrap()
        .json::<Value>()
        .await
        .unwrap()["id"]
        .as_u64()
        .unwrap();

    let session: Value = fx
        .client
        .get(format!("{}/sessions/{id}", fx.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(session["metadata"], json!({"site_id": 7}));
    let session: Value = fx
        .client
        .get(format!("{}/sessions/{plain}", fx.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(session["metadata"], Value::Null);

    let listed: Value = fx
        .client
        .get(format!("{}/sessions", fx.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let rows = listed["sessions"].as_array().unwrap();
    let row = rows.iter().find(|row| row["id"] == json!(id)).unwrap();
    assert_eq!(
        row["metadata"],
        json!({"site_id": 7}),
        "the list carries it"
    );

    // The inbox reports keys, sources and consumption state.
    assert_eq!(
        fx.client
            .post(format!("{}/sessions/{id}/messages", fx.base))
            .json(&json!({"content": "issue 1", "key": "issue-1", "source": "jscrawl"}))
            .send()
            .await
            .unwrap()
            .status(),
        202
    );
    wait_for_event(&fx.store, id, "run.finished", 1).await;

    let inputs: Value = fx
        .client
        .get(format!("{}/sessions/{id}/inputs", fx.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let inputs = inputs["inputs"].as_array().unwrap();
    assert_eq!(inputs.len(), 1);
    assert_eq!(inputs[0]["key"], "issue-1");
    assert_eq!(inputs[0]["source"], "jscrawl");
    assert_eq!(inputs[0]["delivery"], "queue");
    assert_eq!(inputs[0]["content"], "issue 1");
    assert!(
        inputs[0]["consumedAt"].is_string(),
        "the claimed input reports when it was consumed: {}",
        inputs[0]
    );

    // A pending input has no consumption time yet.
    fx.client
        .post(format!("{}/sessions/{id}/interrupt", fx.base))
        .send()
        .await
        .unwrap();
    let empty: Value = fx
        .client
        .get(format!("{}/sessions/{plain}/inputs", fx.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(empty["inputs"], json!([]));

    assert_eq!(
        fx.client
            .get(format!("{}/sessions/999999/inputs", fx.base))
            .send()
            .await
            .unwrap()
            .status(),
        404
    );
}

#[tokio::test]
async fn sessions_can_be_renamed_deleted_and_queued_inputs_withdrawn() {
    let fx = fixture(ScriptedModel::new(vec![Reply::Hang])).await;
    let created: Value = fx
        .client
        .post(format!("{}/sessions", fx.base))
        .json(&json!({"title": "draft", "metadata": {"ticket": 7}}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let id = created["id"].as_u64().unwrap();
    let url = format!("{}/sessions/{id}", fx.base);

    let patched: Value = fx
        .client
        .patch(&url)
        .json(&json!({"title": "final", "metadata": null}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(patched["title"], "final");
    assert_eq!(patched["metadata"], Value::Null);
    let rejected = fx
        .client
        .patch(&url)
        .json(&json!({"agent": "research"}))
        .send()
        .await
        .unwrap();
    assert_eq!(rejected.status(), 400);

    // A queued input nobody has claimed can be withdrawn, once.
    let input = fx
        .store
        .add_input(id, mason::store::NewInput::queued("never mind"))
        .await
        .unwrap();
    let withdraw = format!("{url}/inputs/{input}");
    assert_eq!(
        fx.client.delete(&withdraw).send().await.unwrap().status(),
        204
    );
    assert_eq!(
        fx.client.delete(&withdraw).send().await.unwrap().status(),
        404
    );
    assert!(fx.store.list_inputs(id).await.unwrap().is_empty());

    // A running session refuses deletion until it is idle.
    fx.client
        .post(format!("{url}/messages"))
        .json(&json!({"content": "hang"}))
        .send()
        .await
        .unwrap();
    wait_for_event(&fx.store, id, "run.started", 1).await;
    assert_eq!(fx.client.delete(&url).send().await.unwrap().status(), 409);
    fx.runner.interrupt(id);
    for _ in 0..500 {
        if !fx.runner.is_busy(id) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(fx.client.delete(&url).send().await.unwrap().status(), 204);
    assert_eq!(fx.client.get(&url).send().await.unwrap().status(), 404);
    assert!(fx.store.list_events(id, 0).await.unwrap().is_empty());
}

#[tokio::test]
async fn a_failed_run_reports_a_stall_that_resume_clears() {
    let fx = fixture(ScriptedModel::new(vec![
        Reply::Error("provider outage".into()),
        answer("recovered"),
    ]))
    .await;
    let id = fx
        .store
        .create_session(NewSession::default())
        .await
        .unwrap();
    let url = format!("{}/sessions/{id}", fx.base);
    fx.client
        .post(format!("{url}/messages"))
        .json(&json!({"content": "first"}))
        .send()
        .await
        .unwrap();
    wait_for_event(&fx.store, id, "run.failed", 1).await;
    // An input queued without a wake waits behind the failure.
    fx.store
        .add_input(id, mason::store::NewInput::queued("second"))
        .await
        .unwrap();
    for _ in 0..500 {
        if !fx.runner.is_busy(id) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let session: Value = fx
        .client
        .get(&url)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(session["stalled"], true);

    let resumed = fx
        .client
        .post(format!("{url}/resume"))
        .send()
        .await
        .unwrap();
    assert_eq!(resumed.status(), 202);
    let events = wait_for_event(&fx.store, id, "run.finished", 1).await;
    assert_eq!(events.last().unwrap().data, json!({"answer": "recovered"}));
    let session: Value = fx
        .client
        .get(&url)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(session["stalled"], false);
}
