//! Public CLI/SDK safety boundaries and regression coverage.
mod support;

use mason::store::{NewInput, NewSession, SessionStore, SqliteStore};
use mason::tool::policy::{ExecutionMode, ToolPolicy, workspace_path};
use mason::{Agent, ChatDelta, OutputSpill, Runner, ToolCtx, ToolError, TurnAccumulator};
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::Duration;
use support::{Reply, ScriptedModel, answer, call, config, tool_results};

#[tokio::test]
async fn a_second_store_cannot_recover_an_owned_database_and_the_lock_releases() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sessions.db");
    let store = SqliteStore::open(&path).await.unwrap();
    let session = store.create_session(NewSession::default()).await.unwrap();
    store
        .add_input(session, NewInput::queued("hi"))
        .await
        .unwrap();
    let claim = store.start_run(session).await.unwrap();
    assert!(
        SqliteStore::open(&path)
            .await
            .err()
            .unwrap()
            .to_string()
            .contains("already owned")
    );
    assert!(
        !store
            .list_events(session, 0)
            .await
            .unwrap()
            .iter()
            .any(|event| event.event_type == "run.interrupted")
    );
    drop(store);
    let reopened = SqliteStore::open(&path).await.unwrap();
    assert_eq!(
        reopened.recover().await.unwrap().interrupted,
        vec![(session, claim.run_id.unwrap())]
    );
}

#[cfg(unix)]
#[tokio::test]
async fn database_and_sidecars_are_private_in_a_new_state_directory() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("state/sessions.db");
    let store = SqliteStore::open(&path).await.unwrap();
    store.create_session(NewSession::default()).await.unwrap();
    assert_eq!(
        std::fs::metadata(path.parent().unwrap())
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o700
    );
    for suffix in ["", "-wal", "-shm", ".mason-lock"] {
        let file = std::path::PathBuf::from(format!("{}{suffix}", path.display()));
        assert_eq!(
            std::fs::metadata(file).unwrap().permissions().mode() & 0o777,
            0o600,
            "{suffix}"
        );
    }
}

#[cfg(unix)]
#[test]
fn workspace_checks_reject_escape_symlinks_traversal_and_new_paths() {
    let dir = tempfile::tempdir().unwrap();
    let workspace = dir.path().join("workspace");
    std::fs::create_dir(&workspace).unwrap();
    std::os::unix::fs::symlink(dir.path(), workspace.join("escape")).unwrap();
    assert!(workspace_path(&workspace, "../secret").is_err());
    assert!(workspace_path(&workspace, "escape/new/file").is_err());
    assert!(workspace_path(&workspace, "missing/../../secret").is_err());
    assert!(
        workspace_path(&workspace, "new/nested/file")
            .unwrap()
            .starts_with(workspace.canonicalize().unwrap())
    );
}

struct DenyTools;
#[async_trait::async_trait]
impl ToolPolicy for DenyTools {
    async fn authorize(&self, name: &str, _: &Value, _: &ToolCtx) -> Result<(), ToolError> {
        if name == "task" {
            Ok(())
        } else {
            Err(ToolError::failure("approval declined"))
        }
    }
}

#[tokio::test]
async fn policy_denial_prevents_side_effects_and_is_inherited_by_children() {
    let dir = tempfile::tempdir().unwrap();
    let model = ScriptedModel::new(vec![
        call(
            "task",
            r#"{"agent":"general","description":"d","prompt":"create a file"}"#,
        ),
        call(
            "write",
            r#"{"filePath":"should-not-exist","content":"bad"}"#,
        ),
        answer("child denied"),
        answer("parent done"),
    ]);
    let agent = Agent::builder()
        .model(model.clone())
        .cwd(dir.path())
        .tools(mason::builtin_tools())
        .execution_policy(ExecutionMode::Trusted, false, Arc::new(DenyTools))
        .build()
        .unwrap();
    assert_eq!(agent.run_once("delegate").await.unwrap(), "parent done");
    assert!(!dir.path().join("should-not-exist").exists());
    assert!(model.requests().iter().flat_map(|r| &r.messages).any(|m| {
        m.content
            .as_deref()
            .is_some_and(|s| s.contains("approval declined"))
    }));
}

#[tokio::test]
async fn small_local_models_can_build_and_run() {
    for capacity in [4096, 8192, 16384] {
        let model = ScriptedModel::with_context(vec![answer("ok")], capacity);
        let agent = Agent::builder().model(model).build().unwrap();
        assert_eq!(agent.run_once("hello").await.unwrap(), "ok");
    }
}

#[tokio::test]
async fn oversized_recent_results_fit_the_sent_context_and_preserve_stored_content() {
    let dir = tempfile::tempdir().unwrap();
    let (big, _) = support::EchoTool::boxed("big", &"y".repeat(24_000));
    let (mut config, model) = config(
        vec![call("big", "{}"), answer("done")],
        vec![big],
        dir.path(),
    )
    .await;
    config.max_tokens = Some(4_000);
    let outcome = mason::engine::run(config).await.unwrap();
    let requests = model.requests();
    let second = &requests[1];
    let specs = serde_json::to_string(&second.tools).unwrap();
    assert!(
        mason::context::estimate_tokens(&second.messages) + mason::context::estimate_str(&specs)
            <= 4_000
    );
    let result = second
        .messages
        .iter()
        .find(|m| m.role == mason::Role::Tool)
        .unwrap()
        .content
        .as_ref()
        .unwrap();
    assert!(result.contains("Shortened for context"));
    assert_eq!(tool_results(&outcome.messages)[0], "y".repeat(24_000));
}

#[tokio::test]
async fn an_oversized_result_is_spilled_once_across_steps() {
    let dir = tempfile::tempdir().unwrap();
    let (big, _) = support::EchoTool::boxed("big", &"y".repeat(20_000));
    let (small, _) = support::EchoTool::boxed("small", "x");
    let (mut config, model) = config(
        vec![call("big", "{}"), call("small", "{}"), answer("done")],
        vec![big, small],
        dir.path(),
    )
    .await;
    // A large prompt keeps the view over budget while both results stay inside
    // the pruning protection window, so the big result is shortened twice.
    config.messages = vec![mason::Message::user("p".repeat(56_000))];
    config.max_tokens = Some(18_000);
    mason::engine::run(config).await.unwrap();
    let notices: Vec<String> = model.requests()[1..]
        .iter()
        .map(|request| {
            request
                .messages
                .iter()
                .find(|m| m.role == mason::Role::Tool)
                .unwrap()
                .content
                .clone()
                .unwrap()
                .split_once("[Shortened for context; full result: ")
                .expect("the big result is shortened")
                .1
                .to_owned()
        })
        .collect();
    assert_eq!(notices.len(), 2);
    assert_eq!(notices[0], notices[1], "the same spill file is reused");
    let spilled = std::fs::read_dir(dir.path().join("spill"))
        .unwrap()
        .filter(|entry| {
            entry
                .as_ref()
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with("output-")
        })
        .count();
    assert_eq!(spilled, 1);
}

#[test]
fn turn_byte_budget_bounds_text_reasoning_and_tool_arguments() {
    for fragment in [
        ChatDelta::Text("x".repeat(65)),
        ChatDelta::Reasoning("x".repeat(65)),
        ChatDelta::ToolCall {
            index: 0,
            id: Some("c".into()),
            name: Some("tool".into()),
            arguments: "x".repeat(65),
        },
    ] {
        let mut accumulator = TurnAccumulator::new().with_max_bytes(64);
        assert!(accumulator.push(fragment));
        assert!(
            accumulator
                .finish()
                .unwrap_err()
                .to_string()
                .contains("byte limit")
        );
    }
    let mut accumulator = TurnAccumulator::new().with_max_bytes(64);
    assert!(!accumulator.push(ChatDelta::Text("x".repeat(32))));
    assert!(accumulator.push(ChatDelta::Reasoning("x".repeat(33))));
    assert!(accumulator.finish().is_err());
}

struct LargeFailure;
#[async_trait::async_trait]
impl mason::Tool for LargeFailure {
    fn spec(&self) -> mason::ToolSpec {
        mason::ToolSpec::function("fail", "large error", json!({"type":"object"}))
    }
    async fn call(&self, _: Value, _: &ToolCtx) -> Result<String, ToolError> {
        Err(ToolError::failure("x".repeat(100_000)))
    }
}

#[tokio::test]
async fn failed_tool_output_uses_the_same_preview_and_spill_bounds() {
    let dir = tempfile::tempdir().unwrap();
    let (config, _) = config(
        vec![call("fail", "{}"), answer("ok")],
        vec![Box::new(LargeFailure)],
        dir.path(),
    )
    .await;
    let outcome = mason::engine::run(config).await.unwrap();
    let outputs = tool_results(&outcome.messages);
    assert!(outputs[0].len() <= mason::tool::output::MAX_BYTES);
    assert!(outputs[0].contains("output-"));
}

#[tokio::test]
async fn root_and_child_share_the_tool_call_budget() {
    let dir = tempfile::tempdir().unwrap();
    let model = ScriptedModel::new(vec![
        call(
            "task",
            r#"{"agent":"general","description":"d","prompt":"create"}"#,
        ),
        answer("child wrapped up"),
        answer("parent wrapped up"),
    ]);
    let limits = mason::Limits {
        max_total_tool_calls: 1,
        ..mason::Limits::default()
    };
    let agent = Agent::builder()
        .model(model.clone())
        .cwd(dir.path())
        .tools(mason::builtin_tools())
        .limits(limits)
        .build()
        .unwrap();
    // The parent's task call spends the shared budget, so both runs answer without tools.
    assert_eq!(agent.run_once("go").await.unwrap(), "parent wrapped up");
    let requests = model.requests();
    assert!(requests[1].tools.is_empty() && requests[2].tools.is_empty());
}

#[tokio::test]
async fn spill_quota_accounts_for_concurrent_reservations_from_separate_instances() {
    let dir = tempfile::tempdir().unwrap();
    let spill = || {
        Arc::new(
            OutputSpill::new(dir.path())
                .with_capture_cap(60_000)
                .with_storage_limit(100_000),
        )
    };
    let first = spill();
    let second = spill();
    let mut capture = first.capture();
    capture.append(&vec![b'x'; 55_000]).await.unwrap();
    assert!(
        second
            .save(&"y".repeat(55_000))
            .await
            .unwrap_err()
            .to_string()
            .contains("quota")
    );
    let finished = capture.finish("").await.unwrap();
    assert_eq!(
        std::fs::metadata(finished.path.unwrap()).unwrap().len(),
        55_000
    );
}

#[tokio::test]
async fn server_admission_bounds_active_work_and_queued_tasks() {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(SqliteStore::open(dir.path().join("db")).await.unwrap());
    let gate = Arc::new(tokio::sync::Notify::new());
    let model = ScriptedModel::new(vec![
        Reply::Gate(gate.clone(), Box::new(answer("first"))),
        answer("second"),
        answer("third"),
    ]);
    let agent = Agent::builder()
        .model(model.clone())
        .cwd(dir.path())
        .build()
        .unwrap();
    let runner = Runner::with_limits(
        agent,
        store.clone(),
        mason::runner::RunnerLimits {
            max_active_runs: 1,
            max_pending_sessions: 2,
        },
    )
    .unwrap();
    let mut ids = Vec::new();
    for _ in 0..3 {
        let id = store.create_session(NewSession::default()).await.unwrap();
        store.add_input(id, NewInput::queued("go")).await.unwrap();
        ids.push(id);
    }
    runner.try_wake(ids[0]).unwrap();
    runner.try_wake(ids[1]).unwrap();
    assert!(runner.try_wake(ids[2]).is_err());
    let rejected = store.create_session(NewSession::default()).await.unwrap();
    assert!(matches!(
        runner.submit(rejected, NewInput::queued("rejected")).await,
        Err(mason::Error::Busy)
    ));
    assert!(store.list_inputs(rejected).await.unwrap().is_empty());
    tokio::time::timeout(Duration::from_secs(5), async {
        while model.request_count() == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(model.request_count(), 1);
    gate.notify_one();
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if store
                .list_events(ids[2], 0)
                .await
                .unwrap()
                .iter()
                .any(|event| event.event_type == "run.finished")
            {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(model.request_count(), 3);
    runner.shutdown(Duration::from_secs(1)).await;
}

#[tokio::test]
async fn token_budget_stops_before_a_model_request_and_inbox_pages_have_cursors() {
    let dir = tempfile::tempdir().unwrap();
    let (mut config, model) = config(vec![answer("unreachable")], vec![], dir.path()).await;
    config.limits.max_total_tokens = 1;
    config.messages = vec![mason::Message::user("a deliberately oversized prompt")];
    assert!(matches!(
        mason::engine::run(config).await,
        Err(mason::Error::BudgetExceeded("token"))
    ));
    assert_eq!(model.request_count(), 0);

    let store = SqliteStore::open(dir.path().join("db")).await.unwrap();
    let session = store.create_session(NewSession::default()).await.unwrap();
    let first = store
        .add_input(session, NewInput::queued("first"))
        .await
        .unwrap();
    let second = store
        .add_input(session, NewInput::queued("second"))
        .await
        .unwrap();
    assert_eq!(store.inputs_page(session, 0, 1).await.unwrap()[0].id, first);
    assert_eq!(
        store.inputs_page(session, first, 1).await.unwrap()[0].id,
        second
    );
    assert!(
        store
            .inputs_page(session, u64::MAX, 1)
            .await
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn inbox_capacity_is_atomic_and_idempotent_retries_still_work() {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(SqliteStore::open(dir.path().join("db")).await.unwrap());
    let session = store.create_session(NewSession::default()).await.unwrap();
    let mut input = NewInput::queued("first");
    input.key = Some("original".into());
    let original = store.add_input(session, input.clone()).await.unwrap();
    for _ in 1..95 {
        store
            .add_input(session, NewInput::queued("queued"))
            .await
            .unwrap();
    }
    let mut tasks = tokio::task::JoinSet::new();
    for _ in 0..10 {
        let store = store.clone();
        tasks.spawn(async move { store.add_input(session, NewInput::queued("race")).await });
    }
    let mut accepted = 0;
    while let Some(result) = tasks.join_next().await {
        match result.unwrap() {
            Ok(_) => accepted += 1,
            Err(mason::Error::Busy) => {}
            Err(error) => panic!("{error}"),
        }
    }
    assert_eq!(accepted, 5);
    assert_eq!(store.list_inputs(session).await.unwrap().len(), 100);
    assert_eq!(store.add_input(session, input).await.unwrap(), original);
}

#[tokio::test]
async fn authenticated_api_rejects_missing_and_bad_tokens() {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(SqliteStore::open(dir.path().join("db")).await.unwrap());
    let agent = Agent::builder()
        .model(ScriptedModel::new(vec![]))
        .build()
        .unwrap();
    let runner = Runner::new(agent, store);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let security = mason::api::ApiSecurity::new("test-token-long-enough".into()).unwrap();
    let shutdown = mason::CancellationToken::new();
    let app = mason::api::router_with_security(runner, shutdown.clone(), security);
    let task = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let base = format!("http://{address}");
    let client = reqwest::Client::new();
    let url = format!("{base}/sessions");
    assert_eq!(
        client
            .post(&url)
            .json(&json!({}))
            .send()
            .await
            .unwrap()
            .status(),
        401
    );
    assert_eq!(
        client
            .post(&url)
            .bearer_auth("wrong-token")
            .json(&json!({}))
            .send()
            .await
            .unwrap()
            .status(),
        401
    );
    // Any Host and Origin are accepted with the token: a browser app or proxy needs no allow-list.
    let created = client
        .post(&url)
        .bearer_auth("test-token-long-enough")
        .header("host", "mason.internal:3120")
        .header("origin", "http://app.internal")
        .json(&json!({}))
        .send()
        .await
        .unwrap();
    assert_eq!(created.status(), 201);
    // SSE endpoints receive the same authentication layer.
    assert_eq!(
        client
            .get(format!("{base}/sessions/1/events"))
            .header("accept", "text/event-stream")
            .send()
            .await
            .unwrap()
            .status(),
        401
    );
    shutdown.cancel();
    task.abort();
}

#[cfg(unix)]
#[tokio::test]
async fn background_processes_are_owned_stoppable_and_cleaned_up_after_the_run() {
    let dir = tempfile::tempdir().unwrap();
    let (config, _) = config(
        vec![
            call(
                "bash",
                r#"{"command":"(sleep 2; touch leaked) & echo started"}"#,
            ),
            answer("done"),
        ],
        mason::builtin_tools(),
        dir.path(),
    )
    .await;
    let registry = config.ctx.processes.clone();
    mason::engine::run(config).await.unwrap();
    assert!(registry.list().is_empty());
    tokio::time::sleep(Duration::from_millis(1300)).await;
    assert!(!dir.path().join("leaked").exists());
    assert!(
        !registry.stop(std::process::id()),
        "cannot stop an unowned process"
    );
}
