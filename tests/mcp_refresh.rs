//! Run-time MCP discovery shares session connections and isolates server failures.
#![cfg(feature = "mcp")]
mod support;

use mason::store::{NewInput, NewSession, SessionStore, SqliteStore};
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::{Duration, Instant};
use support::{ScriptedModel, answer, call};

fn server(dir: &std::path::Path, name: &str) -> Value {
    let target = std::env::var("CARGO_TARGET_DIR")
        .unwrap_or_else(|_| format!("{}/target", env!("CARGO_MANIFEST_DIR")));
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    json!({"type":"local", "command":[format!("{target}/{profile}/examples/mcp_fixture")], "environment":{
        "MASON_FIXTURE_STARTS":dir.join(format!("{name}.starts")),
        "MASON_FIXTURE_LATE_TOOL":dir.join(format!("{name}.late")),
        "MASON_FIXTURE_FAIL_LISTING":dir.join(format!("{name}.fail")),
        "MASON_FIXTURE_SLOW_LISTING":dir.join(format!("{name}.slow")),
        "MASON_FIXTURE_PID":"true"
    }})
}

async fn agent(
    dir: &std::path::Path,
    model: Arc<ScriptedModel>,
    servers: &[&str],
    tools: &[&str],
) -> mason::Agent {
    let configs: serde_json::Map<_, _> = servers
        .iter()
        .map(|name| (name.to_string(), server(dir, name)))
        .collect();
    let manifest = mason::config::Manifest::parse(
        &json!({
            "version":1, "cwd":dir, "allow_mcp":true, "mcp":configs,
            "profiles":{"build":{"tools":tools}}
        })
        .to_string(),
        dir.to_path_buf(),
    )
    .unwrap();
    manifest
        .into_builder_with_model(model)
        .await
        .unwrap()
        .limits(mason::Limits {
            timeout: Duration::from_secs(10),
            ..Default::default()
        })
        .build()
        .unwrap()
}

async fn session(dir: &std::path::Path) -> (Arc<SqliteStore>, u64) {
    let store = Arc::new(SqliteStore::open(dir.join("sessions.db")).await.unwrap());
    let id = store.create_session(NewSession::default()).await.unwrap();
    (store, id)
}

async fn run(agent: &mason::Agent, store: &Arc<SqliteStore>, id: u64) {
    store.add_input(id, NewInput::queued("go")).await.unwrap();
    let claim = store.start_run(id).await.unwrap();
    let answer = agent
        .run_session(mason::agent::SessionRun {
            store: store.clone(),
            session_id: id,
            run_id: claim.run_id.clone().unwrap(),
            profile: "build".into(),
            token: Default::default(),
            events: None,
        })
        .await
        .unwrap();
    store
        .append_run_event(
            id,
            claim.run_id.as_deref().unwrap(),
            "run.finished",
            json!({"answer":answer}),
        )
        .await
        .unwrap();
}

fn starts(dir: &std::path::Path, server: &str) -> usize {
    std::fs::read_to_string(dir.join(format!("{server}.starts")))
        .unwrap()
        .lines()
        .count()
}

fn has_tool(model: &ScriptedModel, turn: usize, name: &str) -> bool {
    model.requests()[turn]
        .tools
        .iter()
        .any(|tool| tool.function.name == name)
}

#[tokio::test]
async fn refresh_and_calls_reuse_one_session_process_and_skip_unused_servers() {
    let dir = tempfile::tempdir().unwrap();
    let model = ScriptedModel::new(vec![
        call("used_echo", r#"{"text":"pid"}"#),
        answer("one"),
        call("used_late", "{}"),
        answer("two"),
    ]);
    let agent = agent(dir.path(), model.clone(), &["used", "unused"], &["used_*"]).await;
    assert_eq!(starts(dir.path(), "used"), 1, "startup discovery");
    let (store, id) = session(dir.path()).await;
    run(&agent, &store, id).await;
    assert_eq!(
        starts(dir.path(), "used"),
        2,
        "listing and call share the process"
    );
    assert_eq!(
        starts(dir.path(), "unused"),
        1,
        "unused servers are not refreshed"
    );
    std::fs::write(dir.path().join("used.late"), "").unwrap();
    run(&agent, &store, id).await;
    assert!(has_tool(&model, 2, "used_late"));
    assert_eq!(
        starts(dir.path(), "used"),
        2,
        "the next message reuses the session"
    );
    assert_eq!(starts(dir.path(), "unused"), 1);
    agent.close_session_scope(id).await;
}

#[tokio::test]
async fn failed_servers_keep_their_last_good_listing_without_discarding_other_updates() {
    let dir = tempfile::tempdir().unwrap();
    let model = ScriptedModel::new(vec![answer("one"), answer("two"), answer("three")]);
    let agent = agent(dir.path(), model.clone(), &["a", "b"], &["a_*", "b_*"]).await;
    let (store, id) = session(dir.path()).await;
    run(&agent, &store, id).await;
    std::fs::write(dir.path().join("a.fail"), "").unwrap();
    std::fs::write(dir.path().join("b.late"), "").unwrap();
    run(&agent, &store, id).await;
    assert!(has_tool(&model, 1, "a_echo"));
    assert!(has_tool(&model, 1, "b_late"));
    std::fs::write(dir.path().join("b.fail"), "").unwrap();
    run(&agent, &store, id).await;
    assert!(
        has_tool(&model, 2, "b_late"),
        "fallback keeps the last success, not startup tools"
    );
    assert_eq!(starts(dir.path(), "a"), 2);
    assert_eq!(starts(dir.path(), "b"), 2);
    agent.close_session_scope(id).await;
}

#[tokio::test]
async fn slow_servers_share_one_refresh_deadline_and_do_not_block_healthy_updates() {
    let dir = tempfile::tempdir().unwrap();
    let model = ScriptedModel::new(vec![answer("one"), answer("two")]);
    let agent = agent(dir.path(), model.clone(), &["a", "b", "z"], &["*"]).await;
    let (store, id) = session(dir.path()).await;
    run(&agent, &store, id).await;
    for marker in ["a.slow", "b.slow", "z.late"] {
        std::fs::write(dir.path().join(marker), "").unwrap();
    }
    let started = Instant::now();
    run(&agent, &store, id).await;
    assert!(
        started.elapsed() < Duration::from_secs(3),
        "refresh has one 1s budget, not sequential 30s timeouts"
    );
    assert!(
        has_tool(&model, 1, "z_late"),
        "healthy updates survive the shared deadline"
    );
    assert!(has_tool(&model, 1, "a_echo") && has_tool(&model, 1, "b_echo"));
    agent.close_session_scope(id).await;
}
