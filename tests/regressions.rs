mod support;

use std::sync::Arc;
use std::time::Duration;

use mason::{
    Agent, Message, NewSession, OutputSpill, PromptSource, Role, Runner, SessionStore, SqliteStore,
    Tool, ToolCall, ToolCtx, Typed,
};
use serde_json::json;
use support::{EchoTool, FunctionModel, Reply, ScriptedModel, answer, call, config};

#[tokio::test]
async fn child_sessions_are_busy_interruptible_and_parent_owned() {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(
        SqliteStore::open(dir.path().join("db.sqlite"))
            .await
            .unwrap(),
    );
    let child_started = Arc::new(tokio::sync::Notify::new());
    let signal = child_started.clone();
    let model = FunctionModel::new(move |request| {
        if request.messages.iter().any(|message| {
            message.role == Role::User && message.content.as_deref() == Some("child")
        }) {
            signal.notify_one();
            Reply::Hang
        } else if request
            .messages
            .last()
            .is_some_and(|message| message.role == Role::Tool)
        {
            answer("parent done")
        } else {
            call(
                "task",
                r#"{"agent":"general","description":"child","prompt":"child"}"#,
            )
        }
    });
    let agent = Agent::builder()
        .model(model.clone())
        .cwd(dir.path())
        .build()
        .unwrap();
    let runner = Runner::new(agent, store.clone());
    let parent = store.create_session(NewSession::default()).await.unwrap();
    store
        .add_input(parent, mason::NewInput::queued("parent"))
        .await
        .unwrap();
    runner.wake(parent);
    tokio::time::timeout(Duration::from_secs(5), child_started.notified())
        .await
        .unwrap();
    let child = parent + 1;
    assert_eq!(
        store.get_session(child).await.unwrap().unwrap().parent_id,
        Some(parent)
    );
    assert!(runner.is_busy(child));
    let mut events = store.subscribe(child);
    store
        .add_input(
            child,
            mason::NewInput::queued("must not start independently"),
        )
        .await
        .unwrap();
    runner.wake(child);
    assert_eq!(model.request_count(), 2);
    runner.interrupt(child);
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if events.recv().await.unwrap().event_type == "run.interrupted" {
                break;
            }
        }
    })
    .await
    .unwrap();
    let mut parent_events = store.subscribe(parent);
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if store
                .list_events(parent, 0)
                .await
                .unwrap()
                .iter()
                .any(|event| event.event_type == "run.finished")
            {
                break;
            }
            parent_events.recv().await.unwrap();
        }
    })
    .await
    .unwrap();
    runner.wake(child);
    runner.shutdown(Duration::from_secs(1)).await;
    let log = store.list_events(child, 0).await.unwrap();
    assert_eq!(
        log.iter()
            .filter(|event| event.event_type == "run.started")
            .count(),
        1
    );
    assert!(!runner.is_busy(child));
}

#[tokio::test]
async fn reused_call_ids_are_repaired_within_each_assistant_turn() {
    let dir = tempfile::tempdir().unwrap();
    let store = SqliteStore::open(dir.path().join("db.sqlite"))
        .await
        .unwrap();
    let session = store.create_session(NewSession::default()).await.unwrap();
    let mut assistant = Message::user("");
    assistant.role = Role::Assistant;
    assistant.content = None;
    assistant.tool_calls = Some(vec![ToolCall::new("same", "echo", "{}")]);
    for message in [
        Message::user("first"),
        assistant.clone(),
        Message::tool("same", "old result"),
        Message::user("second"),
        assistant,
        Message::user("resume"),
    ] {
        store
            .append(session, "message", json!({"message": message}))
            .await
            .unwrap();
    }
    let prompts = PromptSource::Embedded.load().unwrap();
    let history = store.history(session, &prompts).await.unwrap();
    assert_eq!(history[2], Message::tool("same", "old result"));
    assert_eq!(
        history[5],
        Message::tool("same", "[Tool execution was interrupted]")
    );
    assert_eq!(history[6], Message::user("resume"));
}

#[test]
fn reasoning_payload_contributes_to_context_and_tail_budgets() {
    let mut message = Message::user("");
    message.role = Role::Assistant;
    message.content = None;
    message.reasoning_content = Some("r".repeat(60_000));
    assert_eq!(mason::context::estimate_tokens(&[message.clone()]), 15_000);
    assert_eq!(
        mason::context::tail_start(&[message, Message::user("latest")], 2_000),
        1
    );
}

#[tokio::test]
async fn forced_summary_instructions_survive_overflow_retry() {
    let dir = tempfile::tempdir().unwrap();
    let (echo, _) = EchoTool::boxed("echo", "ok");
    let (mut run, model) = config(
        vec![
            call("echo", "{}"),
            Reply::Error("maximum context length is 8192 tokens".into()),
            answer("done"),
        ],
        vec![echo],
        dir.path(),
    )
    .await;
    run.limits.max_steps = 1;
    let instructions = run.prompts.max_steps.clone();
    run.compact = Some(Arc::new(|messages, _| {
        Box::pin(async move { Ok(Some(messages)) })
    }));
    mason::engine::run(run).await.unwrap();
    let requests = model.requests();
    assert_eq!(requests.len(), 3);
    for request in &requests[1..] {
        assert!(request.tools.is_empty());
        assert_eq!(
            request.messages.last().unwrap().content.as_deref(),
            Some(instructions.as_str())
        );
    }
}

#[tokio::test]
async fn filesystem_search_leaves_the_runtime_responsive() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("big.txt"), "a".repeat(8 * 1024 * 1024)).unwrap();
    let (signal, progress) = tokio::sync::oneshot::channel();
    let ctx = ToolCtx::new(dir.path());
    let searching = tokio::spawn(async move {
        let _ = signal.send(());
        Typed(mason::tool::builtin::GrepTool)
            .call(json!({"pattern": "^a+$", "path": "big.txt"}), &ctx)
            .await
    });
    progress.await.unwrap();
    tokio::task::yield_now().await;
    assert!(
        !searching.is_finished(),
        "the search finished before another runtime task could run"
    );
    let output = searching.await.unwrap().unwrap();
    assert!(output.starts_with("big.txt:1:"));
}

#[tokio::test]
async fn cancelled_searches_stop_instead_of_scanning() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("file.txt"), "needle").unwrap();
    let ctx = ToolCtx::new(dir.path());
    ctx.token.cancel();
    for tool in [
        Box::new(Typed(mason::tool::builtin::GrepTool)) as Box<dyn Tool>,
        Box::new(Typed(mason::tool::builtin::GlobTool)),
    ] {
        let result = tool.call(json!({"pattern": "needle"}), &ctx).await;
        assert!(matches!(
            result,
            Err(mason::ToolError::Fatal(mason::Error::Cancelled))
        ));
    }
}

#[tokio::test]
async fn streamed_capture_slices_chunks_to_the_remaining_budget() {
    let dir = tempfile::tempdir().unwrap();
    let spill = Arc::new(OutputSpill::new(dir.path()).with_capture_cap(60_000));
    let mut capture = spill.capture();
    assert!(!capture.append(&vec![b'a'; 50_000]).await.unwrap());
    assert!(capture.append(&vec![b'b'; 100_000]).await.unwrap());
    assert!(capture.append(b"discarded").await.unwrap());
    let captured = capture.finish("").await.unwrap();
    let saved = tokio::fs::read(captured.path.unwrap()).await.unwrap();
    assert_eq!(saved.len(), 60_000);
    assert!(saved[..50_000].iter().all(|byte| *byte == b'a'));
    assert!(saved[50_000..].iter().all(|byte| *byte == b'b'));
}

#[tokio::test]
async fn api_rejects_wrong_field_types_and_direct_child_messages() {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(
        SqliteStore::open(dir.path().join("db.sqlite"))
            .await
            .unwrap(),
    );
    let agent = Agent::builder()
        .model(ScriptedModel::new(vec![]))
        .cwd(dir.path())
        .build()
        .unwrap();
    let runner = Runner::new(agent, store.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let app = mason::api::router(runner.clone());
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let client = reqwest::Client::new();
    for body in [
        json!({"agent": 42}),
        json!({"agent": []}),
        json!({"title": false}),
    ] {
        let response = client
            .post(format!("http://{address}/sessions"))
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status().as_u16(), 400);
    }
    let session = store.create_session(NewSession::default()).await.unwrap();
    for body in [
        json!({"content": "go", "delivery": 42}),
        json!({"content": "go", "key": 42}),
        json!({"content": "go", "source": []}),
        json!({"content": 42}),
    ] {
        let response = client
            .post(format!("http://{address}/sessions/{session}/messages"))
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status().as_u16(), 400);
    }
    assert!(store.pending_sessions().await.unwrap().is_empty());
    let child = store
        .create_session(NewSession {
            parent_id: Some(session),
            agent: Some("general".into()),
            ..Default::default()
        })
        .await
        .unwrap();
    let response = client
        .post(format!("http://{address}/sessions/{child}/messages"))
        .json(&json!({"content": "go"}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status().as_u16(), 409);
    assert!(store.pending_sessions().await.unwrap().is_empty());
    server.abort();
    runner.shutdown(Duration::from_secs(1)).await;
}

#[cfg(feature = "mcp")]
#[tokio::test]
async fn mcp_keeps_the_servers_dictionary_and_pattern_schemas() {
    use rmcp::model::{
        CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock, ErrorData,
        ListToolsResult, PaginatedRequestParams, ServerCapabilities, ServerConfig,
    };
    use rmcp::service::RequestContext;
    use rmcp::{ServerHandler, ServiceExt};
    use serde_json::Value;
    #[derive(Clone)]
    struct DictionaryServer(Value);
    impl ServerHandler for DictionaryServer {
        fn get_info(&self) -> ServerConfig {
            ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
        }
        async fn list_tools(
            &self,
            _: Option<PaginatedRequestParams>,
            _: RequestContext<rmcp::RoleServer>,
        ) -> Result<ListToolsResult, ErrorData> {
            Ok(ListToolsResult {
                tools: vec![rmcp::model::Tool::new(
                    "dictionary",
                    "dictionary",
                    Arc::new(self.0.as_object().unwrap().clone()),
                )],
                ..Default::default()
            })
        }
        async fn call_tool(
            &self,
            request: CallToolRequestParams,
            _: RequestContext<rmcp::RoleServer>,
        ) -> Result<CallToolResponse, ErrorData> {
            Ok(CallToolResult::success(vec![ContentBlock::text(
                serde_json::to_string(&request.arguments).unwrap(),
            )])
            .into())
        }
    }
    for schema in [
        json!({"type": "object", "additionalProperties": {"type": "string"}}),
        json!({"type": "object", "patternProperties": {"^arbitrary$": {"type": "string"}}, "additionalProperties": false}),
    ] {
        let (client_io, server_io) = tokio::io::duplex(65_536);
        let remote = DictionaryServer(schema.clone());
        let server =
            tokio::spawn(async move { remote.serve(server_io).await.unwrap().waiting().await });
        let connection = mason::mcp::connect_transport("test", client_io, Duration::from_secs(2))
            .await
            .unwrap();
        assert_eq!(connection.tools()[0].spec().function.parameters, schema);
        let result = connection.tools()[0]
            .call(json!({"arbitrary": "valid"}), &ToolCtx::new("."))
            .await
            .unwrap();
        assert!(result.contains("valid"));
        connection.close().await;
        server.abort();
    }
}
