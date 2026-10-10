//! MCP tool discovery, validation, timeout, and transport tests.

#![cfg(feature = "mcp")]

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::extract::State as AxumState;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use mason::mcp::{McpServerConfig, McpServers, connect_config, connect_transport};
use mason::tool::policy::ExecutionMode;
use mason::{Tool, ToolCtx};
use rmcp::model::{
    CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock, ErrorData,
    ListToolsResult, PaginatedRequestParams, ProgressNotificationParam, ServerCapabilities,
    ServerConfig, Tool as McpToolModel,
};
use rmcp::service::RequestContext;
use rmcp::{RoleServer, ServerHandler, ServiceExt};
use serde_json::{Value, json};
use tokio::net::TcpListener;

#[derive(Clone)]
struct FakeServer {
    fail_listing: bool,
    progress_sent: Arc<AtomicUsize>,
}

impl FakeServer {
    fn new() -> Self {
        Self {
            fail_listing: false,
            progress_sent: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn failing_listing() -> Self {
        Self {
            fail_listing: true,
            progress_sent: Arc::new(AtomicUsize::new(0)),
        }
    }
}

fn schema() -> Arc<serde_json::Map<String, Value>> {
    Arc::new(
        json!({
            "type": "object",
            "properties": {"text": {"type": "string"}, "times": {"type": "integer"}},
            "required": ["text"],
            "additionalProperties": false
        })
        .as_object()
        .unwrap()
        .clone(),
    )
}

fn empty_schema() -> Arc<serde_json::Map<String, Value>> {
    Arc::new(json!({"type": "object"}).as_object().unwrap().clone())
}

impl ServerHandler for FakeServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        if self.fail_listing {
            return Err(ErrorData::internal_error("discovery is broken", None));
        }
        Ok(ListToolsResult {
            tools: vec![
                McpToolModel::new("echo", "Echoes the text", schema()),
                McpToolModel::new("fail", "Always fails", empty_schema()),
                McpToolModel::new("structured", "Structured content", empty_schema()),
                McpToolModel::new("progress", "Keeps reporting progress", empty_schema()),
                McpToolModel::new("t", "Collision bait", empty_schema()),
            ],
            ..Default::default()
        })
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, ErrorData> {
        let arguments = request.arguments.clone().unwrap_or_default();
        match request.name.as_ref() {
            "echo" => {
                let text = arguments
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let times = arguments
                    .get("times")
                    .and_then(Value::as_u64)
                    .unwrap_or(1)
                    .max(1);
                Ok(
                    CallToolResult::success(vec![ContentBlock::text(text.repeat(times as usize))])
                        .into(),
                )
            }
            "fail" => Ok(CallToolResult::error(vec![ContentBlock::text("it broke")]).into()),
            "structured" => {
                let mut result = CallToolResult::success(Vec::new());
                result.structured_content = Some(json!({"ok": true}));
                Ok(result.into())
            }
            "progress" => {
                // Reports progress every 10 ms for up to ten seconds: only
                // the client's total deadline ends the call earlier.
                let token = context.meta.get_progress_token();
                let deadline = Instant::now() + Duration::from_secs(10);
                loop {
                    tokio::select! {
                        _ = context.ct.cancelled() => break,
                        _ = tokio::time::sleep(Duration::from_millis(10)) => {}
                    }
                    if Instant::now() >= deadline {
                        break;
                    }
                    let sent = self.progress_sent.fetch_add(1, Ordering::SeqCst) + 1;
                    if let Some(token) = &token {
                        let _ = context
                            .peer
                            .notify_progress(ProgressNotificationParam::new(
                                token.clone(),
                                sent as f64,
                            ))
                            .await;
                    }
                }
                Ok(CallToolResult::success(vec![ContentBlock::text("finished")]).into())
            }
            other => Err(ErrorData::internal_error(
                format!("unknown tool {other}"),
                None,
            )),
        }
    }
}

/// Serves `server` on one half of a duplex pipe and connects a client on the
/// other; `closed` fires when the server side shuts down.
async fn duplex_pair(
    name: &str,
    server: FakeServer,
    timeout: Duration,
) -> (
    std::result::Result<mason::mcp::McpConnection, String>,
    Arc<tokio::sync::Notify>,
) {
    let (client_io, server_io) = tokio::io::duplex(64 * 1024);
    let closed = Arc::new(tokio::sync::Notify::new());
    let signal = closed.clone();
    tokio::spawn(async move {
        if let Ok(service) = server.serve(server_io).await {
            let _ = service.waiting().await;
        }
        signal.notify_one();
    });
    (connect_transport(name, client_io, timeout).await, closed)
}

fn ctx() -> ToolCtx {
    ToolCtx::new(std::env::temp_dir())
}

/// The built `mcp_fixture` example. etc/check.sh builds it before the tests
/// run; examples get no `CARGO_BIN_EXE_` variable, so resolve its location.
fn fixture_path() -> String {
    let target = std::env::var("CARGO_TARGET_DIR")
        .unwrap_or_else(|_| format!("{}/target", env!("CARGO_MANIFEST_DIR")));
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    format!("{target}/{profile}/examples/mcp_fixture")
}

async fn call(tool: &Arc<dyn Tool>, args: Value) -> Result<String, mason::ToolError> {
    tool.call(args, &ctx()).await
}

#[tokio::test]
async fn tools_of_a_local_stdio_server_become_agent_tools() {
    let config = McpServerConfig::Local {
        command: vec![fixture_path()],
        cwd: None,
        environment: None,
        enabled: None,
        timeout: Some(15_000),
    };
    let connection = connect_config("my files", &config, ExecutionMode::Trusted, true)
        .await
        .expect("the fixture connects");
    let names: Vec<String> = connection
        .tools()
        .iter()
        .map(|tool| tool.spec().function.name)
        .collect();
    assert_eq!(
        names,
        ["my_files_echo", "my_files_fail", "my_files_structured"]
    );

    let [echo, fail, structured] = &connection.tools()[..3] else {
        panic!("three tools")
    };

    // The advertised schema matches the old argument contract.
    let spec = echo.spec();
    assert_eq!(spec.function.parameters["type"], "object");
    assert_eq!(spec.function.parameters["required"], json!(["text"]));
    assert_eq!(
        spec.function.parameters["additionalProperties"],
        json!(false)
    );

    assert_eq!(
        call(echo, json!({"text": "hi", "times": 2})).await.unwrap(),
        "hihi"
    );
    let err = call(echo, json!({"text": "hi", "extra": 1}))
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("unknown argument \"extra\""),
        "{err}"
    );
    let err = call(fail, json!({})).await.unwrap_err();
    assert!(err.to_string().contains("it broke"), "{err}");
    assert_eq!(call(structured, json!({})).await.unwrap(), "{\"ok\":true}");

    connection.close().await;
}

#[tokio::test]
async fn cli_style_mcp_connections_are_independent_per_run_scope() {
    let configs = [(
        "fixture".into(),
        McpServerConfig::Local {
            command: vec![fixture_path()],
            cwd: None,
            environment: Some([("MASON_FIXTURE_PID".into(), "true".into())].into()),
            enabled: None,
            timeout: Some(15_000),
        },
    )]
    .into();
    let servers = McpServers::connect(&configs, ExecutionMode::Trusted, true).await;
    assert!(servers.failures().is_empty(), "{:?}", servers.failures());
    let tool = servers
        .tools()
        .into_iter()
        .find(|tool| tool.spec().function.name == "fixture_echo")
        .unwrap();
    let first = ctx();
    let second = ctx();
    let first_pid = tool.call(json!({"text":"pid"}), &first).await.unwrap();
    let repeated = tool.call(json!({"text":"pid"}), &first).await.unwrap();
    let second_pid = tool.call(json!({"text":"pid"}), &second).await.unwrap();
    assert_eq!(first_pid, repeated);
    assert_ne!(first_pid, second_pid);
}

#[cfg(unix)]
#[tokio::test]
async fn a_crashed_local_server_is_restarted_on_the_next_call() {
    let configs = [(
        "fixture".into(),
        McpServerConfig::Local {
            command: vec![fixture_path()],
            cwd: None,
            environment: Some([("MASON_FIXTURE_PID".into(), "true".into())].into()),
            enabled: None,
            timeout: Some(15_000),
        },
    )]
    .into();
    let servers = McpServers::connect(&configs, ExecutionMode::Trusted, true).await;
    assert!(servers.failures().is_empty(), "{:?}", servers.failures());
    let tool = servers
        .tools()
        .into_iter()
        .find(|tool| tool.spec().function.name == "fixture_echo")
        .unwrap();
    let scope = ctx();
    let first = tool.call(json!({"text":"pid"}), &scope).await.unwrap();
    std::process::Command::new("kill")
        .args(["-9", &first])
        .status()
        .unwrap();
    // The transport notices the exit shortly after the kill.
    let mut second = String::new();
    for _ in 0..50 {
        tokio::time::sleep(Duration::from_millis(100)).await;
        if let Ok(pid) = tool.call(json!({"text":"pid"}), &scope).await {
            second = pid;
            break;
        }
    }
    assert!(!second.is_empty(), "the server was restarted");
    assert_ne!(first, second);
}

#[cfg(unix)]
#[tokio::test]
async fn failed_mcp_startup_preserves_bounded_redacted_diagnostics() {
    let config = McpServerConfig::Local {
        command: vec![
            "/bin/sh".into(),
            "-c".into(),
            "echo \"failure: $PRIVATE_KEY\" >&2; exit 1".into(),
        ],
        cwd: None,
        environment: Some([("PRIVATE_KEY".into(), "sensitive-fixture-secret".into())].into()),
        enabled: None,
        timeout: Some(2000),
    };
    let error = connect_config("broken", &config, ExecutionMode::Trusted, true)
        .await
        .err()
        .unwrap();
    assert!(error.contains("failure:"), "{error}");
    assert!(error.contains("[redacted]"), "{error}");
    assert!(!error.contains("sensitive-fixture-secret"));
}

#[tokio::test]
async fn remote_tools_are_called_over_http_with_the_configured_headers() {
    let (url, auth_seen) = remote_server().await;
    let config = McpServerConfig::Remote {
        url,
        headers: Some([("Authorization".to_owned(), "Bearer t0ken".to_owned())].into()),
        enabled: None,
        timeout: Some(10_000),
    };
    let connection = connect_config("remote", &config, ExecutionMode::Trusted, true)
        .await
        .expect("the remote connects");
    let echo = connection
        .tools()
        .iter()
        .find(|tool| tool.spec().function.name == "remote_echo")
        .expect("the echo tool");
    assert_eq!(
        call(echo, json!({"text": "over http"})).await.unwrap(),
        "over http"
    );
    assert_eq!(auth_seen.lock().unwrap().as_deref(), Some("Bearer t0ken"));
    connection.close().await;
}

#[tokio::test]
async fn servers_that_cannot_connect_are_reported() {
    let mut configs = BTreeMap::new();
    configs.insert(
        "missing".to_owned(),
        McpServerConfig::Local {
            command: vec!["definitely-not-a-command".to_owned()],
            cwd: None,
            environment: None,
            enabled: None,
            timeout: Some(2_000),
        },
    );
    configs.insert(
        "down".to_owned(),
        McpServerConfig::Remote {
            url: "http://127.0.0.1:1/mcp".to_owned(),
            headers: None,
            enabled: None,
            timeout: Some(2_000),
        },
    );
    configs.insert(
        "off".to_owned(),
        McpServerConfig::Local {
            command: vec![fixture_path()],
            cwd: None,
            environment: None,
            enabled: Some(false),
            timeout: None,
        },
    );

    let servers = McpServers::connect(&configs, ExecutionMode::Trusted, true).await;
    assert!(servers.tools().is_empty());
    let failures = servers.failures();
    assert_eq!(failures.len(), 2, "{failures:?}");
    // BTreeMap order: down, missing; the disabled server is silent.
    assert!(
        failures[0].starts_with("MCP server \"down\" unavailable"),
        "{failures:?}"
    );
    assert!(
        failures[1].starts_with("MCP server \"missing\" unavailable"),
        "{failures:?}"
    );
}

#[tokio::test]
async fn a_long_call_outlives_the_connect_timeout_and_ends_with_the_run() {
    let server = FakeServer::new();
    let progress_sent = server.progress_sent.clone();
    let (connection, _closed) = duplex_pair("remote", server, Duration::from_millis(200)).await;
    let connection = connection.expect("the duplex fake connects");
    let progress = connection
        .tools()
        .iter()
        .find(|tool| tool.spec().function.name == "remote_progress")
        .expect("the progress tool")
        .clone();

    // The 200 ms timeout covers connecting only; the run's cancellation ends the call.
    let ctx = ctx();
    let token = ctx.token.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(800)).await;
        token.cancel();
    });
    let started = Instant::now();
    let err = progress.call(json!({}), &ctx).await.unwrap_err();
    assert!(
        matches!(err, mason::ToolError::Fatal(mason::Error::Cancelled)),
        "{err}"
    );
    assert!(started.elapsed() >= Duration::from_millis(800));
    assert!(
        progress_sent.load(Ordering::SeqCst) > 0,
        "progress was flowing"
    );
    connection.close().await;
}

#[tokio::test]
async fn sanitised_names_cannot_collide_silently() {
    let fixture = || McpServerConfig::Local {
        command: vec![fixture_path()],
        cwd: None,
        environment: None,
        enabled: None,
        timeout: Some(15_000),
    };
    let configs = [("a.b".to_owned(), fixture()), ("a_b".to_owned(), fixture())].into();
    let servers = McpServers::connect(&configs, ExecutionMode::Trusted, true).await;
    let names: Vec<String> = servers
        .tools()
        .iter()
        .map(|tool| tool.spec().function.name)
        .collect();
    let unique: std::collections::HashSet<&String> = names.iter().collect();
    assert_eq!(
        names.len(),
        unique.len(),
        "no duplicate tool names: {names:?}"
    );
    assert_eq!(names.iter().filter(|n| *n == "a_b_echo").count(), 1);
    assert!(
        servers
            .failures()
            .iter()
            .any(|line| line.contains("collision: a_b_echo")),
        "the collision is reported, not silent: {:?}",
        servers.failures()
    );
}

#[tokio::test]
async fn failed_discovery_closes_its_client() {
    let (connection, closed) = duplex_pair(
        "broken",
        FakeServer::failing_listing(),
        Duration::from_secs(5),
    )
    .await;
    let err = match connection {
        Ok(_) => panic!("discovery should fail"),
        Err(err) => err,
    };
    assert!(err.contains("discovery is broken"), "{err}");
    // The failed client was closed: the server side observes the shutdown.
    tokio::time::timeout(Duration::from_secs(5), closed.notified())
        .await
        .expect("the server side shut down after the failed discovery");
}

#[derive(Clone)]
struct RemoteState {
    auth: Arc<Mutex<Option<String>>>,
}

async fn remote_server() -> (String, Arc<Mutex<Option<String>>>) {
    let state = RemoteState {
        auth: Arc::new(Mutex::new(None)),
    };
    let app = Router::new()
        .route("/mcp", post(remote_handler))
        .route("/mcp", get(|| async { StatusCode::METHOD_NOT_ALLOWED }))
        .with_state(state.clone());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/mcp", listener.local_addr().unwrap());
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (url, state.auth)
}

async fn remote_handler(
    AxumState(state): AxumState<RemoteState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    if let Some(auth) = headers
        .get("authorization")
        .and_then(|value| value.to_str().ok())
    {
        *state.auth.lock().unwrap() = Some(auth.to_owned());
    }
    let method = body
        .get("method")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let id = body.get("id").cloned();
    let respond_id = id.clone();
    let respond = move |result: Value| -> Response {
        match respond_id {
            Some(id) => Json(json!({"jsonrpc": "2.0", "id": id, "result": result})).into_response(),
            None => StatusCode::ACCEPTED.into_response(),
        }
    };
    match method {
        "initialize" => {
            let version = body
                .pointer("/params/protocolVersion")
                .cloned()
                .unwrap_or(Value::Null);
            respond(json!({
                "protocolVersion": version,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "remote-fixture", "version": "1.0.0"}
            }))
        }
        "tools/list" => respond(json!({"tools": [{
            "name": "echo",
            "description": "Echoes the text",
            "inputSchema": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"]
            }
        }]})),
        "tools/call" => {
            let text = body
                .pointer("/params/arguments/text")
                .and_then(Value::as_str)
                .unwrap_or_default();
            respond(json!({
                "content": [{"type": "text", "text": text}],
                "isError": false
            }))
        }
        "ping" => respond(json!({})),
        _ if id.is_some() => Json(json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": {"code": -32601, "message": format!("unknown method {method}")}
        }))
        .into_response(),
        _ => StatusCode::ACCEPTED.into_response(),
    }
}
