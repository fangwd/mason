//! CLI server startup, recovery, and shutdown tests.

use std::time::Duration;

use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn answer_sse(text: &str) -> String {
    format!(
        "data: {}\n\ndata: {{\"choices\":[{{\"delta\":{{}},\"finish_reason\":\"stop\"}}]}}\n\ndata: [DONE]\n\n",
        json!({"choices": [{"delta": {"role": "assistant", "content": text}}]})
    )
}

fn vendor_env(command: &mut Command, server: &MockServer) {
    command
        .env("MASON_API_TOKEN", "mason-test-api-token")
        .env("AI_VENDORS", "mock")
        .env("AI_MOCK_URL", server.uri())
        .env("AI_MOCK_MODEL", "mock-model")
        .env("AI_MOCK_API_KEY", "mock-key")
        .env("RUST_LOG", "warn")
        .kill_on_drop(true);
}

#[tokio::test]
async fn duplicate_servers_fail_before_recovering_the_original_servers_run() {
    let dir = tempfile::tempdir().unwrap();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_secs(30)))
        .mount(&server)
        .await;
    let (mut child, base) = spawn_serve(dir.path(), &server).await;
    let client = reqwest::Client::builder()
        .default_headers({
            let mut headers = reqwest::header::HeaderMap::new();
            headers.insert(
                "authorization",
                "Bearer mason-test-api-token".parse().unwrap(),
            );
            headers
        })
        .build()
        .unwrap();
    let session: Value = client
        .post(format!("{base}/sessions"))
        .json(&json!({}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let id = session["id"].as_u64().unwrap();
    client
        .post(format!("{base}/sessions/{id}/messages"))
        .json(&json!({"content":"wait"}))
        .send()
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let events: Vec<Value> = client
                .get(format!("{base}/sessions/{id}/events"))
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
            if events.iter().any(|event| event["type"] == "run.started") {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    for address in [base.strip_prefix("http://").unwrap(), "127.0.0.1:0"] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_mason"));
        command.current_dir(dir.path()).args([
            "serve",
            "--listen",
            address,
            "--db",
            dir.path().join("mason.db").to_str().unwrap(),
        ]);
        vendor_env(&mut command, &server);
        let output = command.output().await.unwrap();
        assert_eq!(output.status.code(), Some(1));
        let events: Vec<Value> = client
            .get(format!("{base}/sessions/{id}/events"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert!(
            !events
                .iter()
                .any(|event| event["type"] == "run.interrupted")
        );
    }
    terminate(&mut child);
    tokio::time::timeout(Duration::from_secs(10), child.wait())
        .await
        .unwrap()
        .unwrap();
}

async fn spawn_serve(dir: &std::path::Path, server: &MockServer) -> (Child, String) {
    spawn_serve_with(dir, server, ["--execution-mode", "trusted"]).await
}

/// Spawns `serve` with extra arguments replacing the default execution-mode flag.
async fn spawn_serve_with(
    dir: &std::path::Path,
    server: &MockServer,
    extra: impl IntoIterator<Item = impl AsRef<std::ffi::OsStr>>,
) -> (Child, String) {
    let mut command = Command::new(env!("CARGO_BIN_EXE_mason"));
    command
        .current_dir(dir)
        .arg("serve")
        .args(extra)
        .args([
            "--listen",
            "127.0.0.1:0",
            "--db",
            dir.join("mason.db").to_str().unwrap(),
            "--cwd",
            dir.to_str().unwrap(),
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    vendor_env(&mut command, server);
    let mut child = command.spawn().unwrap();
    let stdout = child.stdout.take().unwrap();
    let mut lines = BufReader::new(stdout).lines();
    let announced = tokio::time::timeout(Duration::from_secs(15), lines.next_line())
        .await
        .expect("mason announces its address")
        .expect("the announce line reads")
        .expect("mason starts");
    let base = announced
        .strip_prefix("mason serving on ")
        .unwrap_or_else(|| panic!("unexpected announce line: {announced}"))
        .to_owned();
    // Keep draining stdout so the pipe never fills.
    tokio::spawn(async move { while let Ok(Some(_)) = lines.next_line().await {} });
    (child, base)
}

#[tokio::test]
async fn serve_answers_sessions_and_streams_events_then_exits_cleanly() {
    let dir = tempfile::tempdir().unwrap();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(answer_sse("Hello from serve")),
        )
        .mount(&server)
        .await;

    let (mut child, base) = spawn_serve(dir.path(), &server).await;
    let client = reqwest::Client::builder()
        .default_headers({
            let mut headers = reqwest::header::HeaderMap::new();
            headers.insert(
                "authorization",
                "Bearer mason-test-api-token".parse().unwrap(),
            );
            headers
        })
        .build()
        .unwrap();

    let created: Value = client
        .post(format!("{base}/sessions"))
        .json(&json!({"title": "smoke"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let session = created["id"].as_u64().unwrap();
    let accepted = client
        .post(format!("{base}/sessions/{session}/messages"))
        .json(&json!({"content": "hello"}))
        .send()
        .await
        .unwrap();
    assert_eq!(accepted.status(), 202);

    // Stream the events over SSE until the run finishes.
    let mut response = client
        .get(format!("{base}/sessions/{session}/events"))
        .header(reqwest::header::ACCEPT, "text/event-stream")
        .send()
        .await
        .unwrap();
    assert!(response.status().is_success());
    let mut saw_finished = false;
    let mut buffer = String::new();
    while let Some(chunk) = response.chunk().await.unwrap() {
        buffer.push_str(&String::from_utf8_lossy(&chunk));
        while let Some(pos) = buffer.find("\n\n") {
            let frame: String = buffer.drain(..pos + 2).collect();
            if frame.contains("event: run.finished") {
                assert!(frame.contains("Hello from serve"), "{frame}");
                saw_finished = true;
            }
        }
        if saw_finished {
            break;
        }
    }
    assert!(saw_finished, "the SSE stream carried the finished run");
    // Keep the SSE connection open across the shutdown: live viewers must
    // not hold the server up — the shutdown token ends their streams.
    let held = tokio::spawn(async move { while let Ok(Some(_)) = response.chunk().await {} });

    // SIGTERM: graceful shutdown, clean exit.
    terminate(&mut child);
    let status = tokio::time::timeout(Duration::from_secs(15), child.wait())
        .await
        .expect("mason exits after SIGTERM despite the open SSE stream")
        .unwrap();
    assert!(status.success(), "clean exit: {status}");
    // The open stream was closed by the shutdown, not left hanging.
    tokio::time::timeout(Duration::from_secs(5), held)
        .await
        .expect("the open SSE stream ends with the server")
        .unwrap();
}

#[cfg(unix)]
fn terminate(child: &mut Child) {
    nix::sys::signal::kill(
        nix::unistd::Pid::from_raw(child.id().unwrap() as i32),
        nix::sys::signal::Signal::SIGTERM,
    )
    .unwrap();
}

#[cfg(not(unix))]
fn terminate(child: &mut Child) {
    child.start_kill().unwrap();
}

/// SIGINT during an active bash call: the run is interrupted, the shell
/// descendant never finishes its delayed work, and the terminal event is
/// persisted before exit.
#[cfg(unix)]
#[tokio::test]
async fn sigint_interrupts_active_runs_and_stops_shell_descendants() {
    let dir = tempfile::tempdir().unwrap();
    let marker = dir.path().join("marker");
    let server = MockServer::start().await;
    let command = format!("(sleep 3; touch {}) & sleep 30", marker.display());
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(format!(
                    "data: {}\n\ndata: {{\"choices\":[{{\"delta\":{{}},\"finish_reason\":\"tool_calls\"}}]}}\n\ndata: [DONE]\n\n",
                    json!({"choices": [{"delta": {"tool_calls": [{
                        "index": 0,
                        "id": "c1",
                        "function": {"name": "bash", "arguments": json!({"command": command}).to_string()}
                    }]}}]})
                )),
        )
        .mount(&server)
        .await;

    let (mut child, base) = spawn_serve(dir.path(), &server).await;
    let client = reqwest::Client::builder()
        .default_headers({
            let mut headers = reqwest::header::HeaderMap::new();
            headers.insert(
                "authorization",
                "Bearer mason-test-api-token".parse().unwrap(),
            );
            headers
        })
        .build()
        .unwrap();
    let created: Value = client
        .post(format!("{base}/sessions"))
        .json(&json!({}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let session = created["id"].as_u64().unwrap();
    client
        .post(format!("{base}/sessions/{session}/messages"))
        .json(&json!({"content": "run it"}))
        .send()
        .await
        .unwrap();

    // Wait until the assistant tool-call message is committed (the first
    // `message` event is the user input), then give bash a moment to spawn
    // its background descendant.
    let mut saw_tool_call = false;
    for _ in 0..1_500 {
        let events: Vec<Value> = client
            .get(format!("{base}/sessions/{session}/events"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        if events
            .iter()
            .any(|event| event.pointer("/data/message/tool_calls").is_some())
        {
            saw_tool_call = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(saw_tool_call, "the assistant tool call was committed");
    tokio::time::sleep(Duration::from_millis(500)).await;

    nix::sys::signal::kill(
        nix::unistd::Pid::from_raw(child.id().unwrap() as i32),
        nix::sys::signal::Signal::SIGINT,
    )
    .unwrap();
    let status = tokio::time::timeout(Duration::from_secs(15), child.wait())
        .await
        .expect("mason exits after SIGINT")
        .unwrap();
    assert!(status.success(), "clean exit: {status}");

    // The terminal event was persisted during the graceful shutdown.
    let store = mason::SqliteStore::open(dir.path().join("mason.db"))
        .await
        .unwrap();
    let events = mason::store::SessionStore::list_events(&store, session, 0)
        .await
        .unwrap();
    assert!(
        events
            .iter()
            .any(|event| event.event_type == "run.interrupted"),
        "the interrupted run has its terminal event: {events:?}"
    );

    // The process group died with the run: the marker never appears.
    tokio::time::sleep(Duration::from_millis(3_600)).await;
    assert!(!marker.exists(), "a shell descendant outlived the shutdown");
}

#[cfg(unix)]
#[tokio::test]
async fn a_non_reading_sse_viewer_does_not_hold_shutdown() {
    let dir = tempfile::tempdir().unwrap();
    let started = dir.path().join("started");
    let marker = dir.path().join("marker");
    let shell = format!(
        "touch '{}'; sleep 2; touch '{}'; sleep 30",
        started.display(),
        marker.display()
    );
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(format!(
                    "data: {}\n\ndata: [DONE]\n\n",
                    json!({"choices": [{
                        "delta": {"tool_calls": [{
                            "index": 0,
                            "id": "command",
                            "function": {
                                "name": "bash",
                                "arguments": json!({"command": shell}).to_string()
                            }
                        }]},
                        "finish_reason": "tool_calls"
                    }]})
                )),
        )
        .mount(&server)
        .await;
    // Seed the backlog before the server obtains exclusive database ownership.
    let store = mason::SqliteStore::open(dir.path().join("mason.db"))
        .await
        .unwrap();
    let session =
        mason::store::SessionStore::create_session(&store, mason::store::NewSession::default())
            .await
            .unwrap();
    let payload = "e".repeat(65_536);
    for _ in 0..512 {
        mason::store::SessionStore::append(
            &store,
            session,
            "message",
            json!({"message": {"role": "user", "content": payload}}),
        )
        .await
        .unwrap();
    }
    drop(store);
    let (mut child, base) = spawn_serve(dir.path(), &server).await;

    // Connect and never read a byte: the replay wedges on backpressure.
    let addr = base.strip_prefix("http://").unwrap();
    let mut socket = tokio::net::TcpStream::connect(addr).await.unwrap();
    use tokio::io::AsyncWriteExt;
    socket
        .write_all(
            format!(
                "GET /sessions/{session}/events HTTP/1.1\r\n\
                 Host: {addr}\r\n\
                 Authorization: Bearer mason-test-api-token\r\n\
                 Accept: text/event-stream\r\n\
                 \r\n"
            )
            .as_bytes(),
        )
        .await
        .unwrap();
    socket.flush().await.unwrap();
    // Give the replay time to fill the channel and stall.
    tokio::time::sleep(Duration::from_millis(500)).await;

    let client = reqwest::Client::builder()
        .default_headers({
            let mut headers = reqwest::header::HeaderMap::new();
            headers.insert(
                "authorization",
                "Bearer mason-test-api-token".parse().unwrap(),
            );
            headers
        })
        .build()
        .unwrap();
    let created: Value = client
        .post(format!("{base}/sessions"))
        .json(&json!({}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let active_session = created["id"].as_u64().unwrap();
    let accepted = client
        .post(format!("{base}/sessions/{active_session}/messages"))
        .json(&json!({"content": "run it"}))
        .send()
        .await
        .unwrap();
    assert_eq!(accepted.status(), 202);
    tokio::time::timeout(Duration::from_secs(15), async {
        while !started.exists() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("the shell starts before SIGTERM");

    terminate(&mut child);
    let status = tokio::time::timeout(Duration::from_secs(15), child.wait())
        .await
        .expect("mason exits after SIGTERM despite the stuck viewer")
        .unwrap();
    assert!(status.success(), "clean exit: {status}");
    assert!(
        !marker.exists(),
        "the shell kept running while HTTP connections drained"
    );
    let store = mason::SqliteStore::open(dir.path().join("mason.db"))
        .await
        .unwrap();
    let events = mason::store::SessionStore::list_events(&store, active_session, 0)
        .await
        .unwrap();
    assert!(
        events
            .iter()
            .any(|event| event.event_type == "run.interrupted"),
        "the active run persists its terminal event during shutdown"
    );
}

/// The manifest a configuration test serves with: domain prose, a narrowed
/// unattended profile, and a metadata rule that selects it.
const MANIFEST: &str = r#"version: 1
execution_mode: trusted
instructions: |
  DOMAIN-INSTRUCTIONS-MARKER
profiles:
  unattended:
    description: Unattended investigation.
    prompt: |
      UNATTENDED-PROMPT-MARKER
    tools: [read]
    max_steps: 7
select_profile:
  - profile: unattended
    when_metadata:
      tenant_id:
        present: true
"#;

/// The first chat completion the mock vendor saw, as JSON.
async fn first_request(server: &MockServer) -> Value {
    let requests = server.received_requests().await.expect("requests recorded");
    let body = &requests[0].body;
    serde_json::from_slice(body).expect("the request body is JSON")
}

fn system_of(request: &Value) -> String {
    request["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|message| message["role"] == "system")
        .and_then(|message| message["content"].as_str())
        .unwrap_or_default()
        .to_owned()
}

fn tool_names(request: &Value) -> Vec<String> {
    request["tools"]
        .as_array()
        .map(|tools| {
            tools
                .iter()
                .map(|tool| {
                    tool["function"]["name"]
                        .as_str()
                        .unwrap_or_default()
                        .to_owned()
                })
                .collect()
        })
        .unwrap_or_default()
}

#[tokio::test]
async fn a_manifest_supplies_instructions_profiles_and_metadata_selection() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("agent.yaml"), MANIFEST).unwrap();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_string(answer_sse("done")))
        .mount(&server)
        .await;
    // No --execution-mode flag: the manifest must supply it.
    let (_child, base) = spawn_serve_with(dir.path(), &server, ["--config", "agent.yaml"]).await;
    let client = reqwest::Client::builder()
        .default_headers({
            let mut headers = reqwest::header::HeaderMap::new();
            headers.insert(
                "authorization",
                "Bearer mason-test-api-token".parse().unwrap(),
            );
            headers
        })
        .build()
        .unwrap();

    // A session whose metadata matches the rule runs the declared profile.
    let session: Value = client
        .post(format!("{base}/sessions"))
        .json(&json!({"title": "tenant", "metadata": {"tenant_id": 3}}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let id = session["id"].as_u64().unwrap();
    client
        .post(format!("{base}/sessions/{id}/messages"))
        .json(&json!({"content": "investigate"}))
        .send()
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(20), async {
        loop {
            let events: Vec<Value> = client
                .get(format!("{base}/sessions/{id}/events"))
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
            if events.iter().any(|event| event["type"] == "run.finished") {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .expect("the selected profile finishes its run");

    let request = first_request(&server).await;
    let system = system_of(&request);
    assert!(
        system.contains("DOMAIN-INSTRUCTIONS-MARKER"),
        "the manifest's instructions reached the model: {system}"
    );
    assert!(
        system.contains("UNATTENDED-PROMPT-MARKER"),
        "the rule selected the declared profile: {system}"
    );
    assert_eq!(
        tool_names(&request),
        ["read"],
        "its allowlist narrowed the tools"
    );
    // The stored row keeps its own agent; only the run's profile changed.
    let stored: Value = client
        .get(format!("{base}/sessions/{id}"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(stored["agent"], "build");
    assert_eq!(stored["metadata"], json!({"tenant_id": 3}));
}

#[tokio::test]
async fn a_session_without_matching_metadata_keeps_its_stored_profile() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("agent.yaml"), MANIFEST).unwrap();
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_string(answer_sse("done")))
        .mount(&server)
        .await;
    let (_child, base) = spawn_serve_with(dir.path(), &server, ["--config", "agent.yaml"]).await;
    let client = reqwest::Client::builder()
        .default_headers({
            let mut headers = reqwest::header::HeaderMap::new();
            headers.insert(
                "authorization",
                "Bearer mason-test-api-token".parse().unwrap(),
            );
            headers
        })
        .build()
        .unwrap();
    let session: Value = client
        .post(format!("{base}/sessions"))
        .json(&json!({"metadata": {"tenant_id": null}}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let id = session["id"].as_u64().unwrap();
    client
        .post(format!("{base}/sessions/{id}/messages"))
        .json(&json!({"content": "hello"}))
        .send()
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(20), async {
        loop {
            let events: Vec<Value> = client
                .get(format!("{base}/sessions/{id}/events"))
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
            if events.iter().any(|event| event["type"] == "run.finished") {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .expect("the run finishes");

    let request = first_request(&server).await;
    let system = system_of(&request);
    assert!(system.contains("DOMAIN-INSTRUCTIONS-MARKER"));
    assert!(
        !system.contains("UNATTENDED-PROMPT-MARKER"),
        "a null value does not match `present`: {system}"
    );
    assert!(
        tool_names(&request).iter().any(|name| name == "bash"),
        "the stored profile kept the base tools"
    );
}

#[tokio::test]
async fn healthz_answers_without_a_token_and_everything_else_refuses() {
    let dir = tempfile::tempdir().unwrap();
    let server = MockServer::start().await;
    let (mut child, base) = spawn_serve(dir.path(), &server).await;
    let anonymous = reqwest::Client::new();

    let health: Value = anonymous
        .get(format!("{base}/healthz"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(health["status"], "ok");
    assert_eq!(health["version"], env!("CARGO_PKG_VERSION"));

    for path in [
        "/sessions",
        "/sessions/1",
        "/sessions/1/inputs",
        "/sessions/1/events",
    ] {
        let response = anonymous.get(format!("{base}{path}")).send().await.unwrap();
        assert_eq!(response.status(), 401, "{path} needs the token");
    }
    let wrong = anonymous
        .get(format!("{base}/sessions"))
        .bearer_auth("not-the-token-at-all")
        .send()
        .await
        .unwrap();
    assert_eq!(wrong.status(), 401);

    child.kill().await.unwrap();
}

#[tokio::test]
async fn the_default_database_lives_under_the_agent_working_directory() {
    let dir = tempfile::tempdir().unwrap();
    let (launch, work) = (dir.path().join("launch"), dir.path().join("work"));
    std::fs::create_dir(&launch).unwrap();
    std::fs::create_dir(&work).unwrap();
    let server = MockServer::start().await;
    let mut command = Command::new(env!("CARGO_BIN_EXE_mason"));
    command
        .current_dir(&launch)
        .args(["serve", "--listen", "127.0.0.1:0", "--cwd"])
        .arg(&work)
        .stdout(std::process::Stdio::piped());
    vendor_env(&mut command, &server);
    let mut child = command.spawn().unwrap();
    let mut lines = BufReader::new(child.stdout.take().unwrap()).lines();
    tokio::time::timeout(Duration::from_secs(15), lines.next_line())
        .await
        .expect("mason announces its address")
        .unwrap()
        .expect("mason starts");
    assert!(work.join(".mason/mason.db").exists());
    assert!(!launch.join(".mason").exists());
    child.kill().await.unwrap();
}
