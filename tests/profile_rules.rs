//! Profile restrictions must survive delegation and filesystem path spelling.
mod support;

use mason::profile::ProfilePatch;
use mason::store::{NewInput, NewSession, SessionStore, SqliteStore};
use mason::{Agent, OutputSpill, Tool, ToolCtx, ToolError, ToolSpec, builtin_tools};
use serde_json::{Value, json};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use support::{ScriptedModel, answer, call};

fn task() -> support::Reply {
    call(
        "task",
        r#"{"description":"edit","prompt":"edit files","agent":"general"}"#,
    )
}

fn builder(dir: &std::path::Path, model: Arc<ScriptedModel>, patch: Value) -> mason::AgentBuilder {
    Agent::builder()
        .model(model)
        .cwd(dir)
        .tools(builtin_tools())
        .spill(Arc::new(OutputSpill::new(dir.join("spill"))))
        .patch_profile(
            "build",
            serde_json::from_value::<ProfilePatch>(patch).unwrap(),
        )
}

async fn run(agent: &Agent, dir: &std::path::Path, stored: bool) -> mason::Result<String> {
    if !stored {
        return agent.run_once("go").await;
    }
    let store = Arc::new(SqliteStore::open(dir.join("sessions.db")).await.unwrap());
    let id = store.create_session(NewSession::default()).await.unwrap();
    store.add_input(id, NewInput::queued("go")).await.unwrap();
    let claim = store.start_run(id).await.unwrap();
    let result = agent
        .run_session(mason::agent::SessionRun {
            store,
            session_id: id,
            run_id: claim.run_id.unwrap(),
            profile: "build".into(),
            token: Default::default(),
            events: None,
        })
        .await;
    agent.close_session_scope(id).await;
    result
}

#[tokio::test]
async fn children_must_obey_both_parent_and_child_path_and_shell_rules() {
    for stored in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let model = ScriptedModel::new(vec![
            task(),
            call("write", r#"{"filePath":"protected.txt","content":"bad"}"#),
            call(
                "write",
                r#"{"filePath":"allowed/child-denied.txt","content":"bad"}"#,
            ),
            call("bash", r#"{"command":"touch shell-escape"}"#),
            call("write", r#"{"filePath":"allowed/ok.txt","content":"ok"}"#),
            answer("child done"),
            answer("parent done"),
        ]);
        let agent = builder(
            dir.path(),
            model.clone(),
            json!({"shell": [], "write_paths": ["allowed/*"]}),
        )
        .patch_profile(
            "general",
            serde_json::from_value(json!({"write_paths": ["allowed/ok.txt", "protected.txt"]}))
                .unwrap(),
        )
        .build()
        .unwrap();
        assert_eq!(
            run(&agent, dir.path(), stored).await.unwrap(),
            "parent done"
        );
        assert!(!dir.path().join("protected.txt").exists());
        assert!(!dir.path().join("allowed/child-denied.txt").exists());
        assert!(!dir.path().join("shell-escape").exists());
        assert_eq!(
            std::fs::read_to_string(dir.path().join("allowed/ok.txt")).unwrap(),
            "ok"
        );
        let requests = model.requests();
        let results = support::tool_results(&requests[5].messages);
        assert_eq!(
            results
                .iter()
                .filter(|text| text.contains("permission denied"))
                .count(),
            3
        );
    }
}

struct CheckFile(Arc<AtomicUsize>);

#[async_trait::async_trait]
impl Tool for CheckFile {
    fn spec(&self) -> ToolSpec {
        ToolSpec::function("check", "check edited file", json!({"type":"object"}))
    }
    async fn call(&self, args: Value, ctx: &ToolCtx) -> Result<String, ToolError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        let contents = std::fs::read_to_string(ctx.cwd.join(args["path"].as_str().unwrap()))
            .unwrap_or_default();
        if contents == "ok" {
            Ok("ok".into())
        } else {
            Err(ToolError::failure("broken file"))
        }
    }
}

#[tokio::test]
async fn parent_hooks_check_child_edits_even_after_a_failed_child() {
    for stored in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let model = ScriptedModel::new(vec![
            task(),
            call("write", r#"{"filePath":"file.txt","content":"bad"}"#),
            support::Reply::Error("child failed".into()),
            answer("premature finish"),
            call("write", r#"{"filePath":"file.txt","content":"ok"}"#),
            answer("fixed"),
        ]);
        let count = Arc::new(AtomicUsize::new(0));
        let hook = json!({"match":"*", "tool":"check", "args":{"path":"{path}"}});
        let agent = builder(
            dir.path(),
            model.clone(),
            json!({
                "hooks": {"after_edit": [hook.clone()], "before_finish": [hook]}
            }),
        )
        .tools([Box::new(CheckFile(count.clone())) as Box<dyn Tool>])
        // The parent hook is runtime-owned, even if the child cannot call it.
        .patch_profile(
            "general",
            serde_json::from_value(json!({"tools":["write"]})).unwrap(),
        )
        .build()
        .unwrap();
        assert_eq!(run(&agent, dir.path(), stored).await.unwrap(), "fixed");
        assert_eq!(
            count.load(Ordering::SeqCst),
            4,
            "child after-edit, parent finish, repair after-edit, final finish"
        );
        assert!(model.requests()[4].messages.iter().any(|m| {
            m.content
                .as_deref()
                .unwrap_or("")
                .contains("[Mason runtime check]")
        }));
    }
}

#[tokio::test]
async fn delegation_cannot_reintroduce_denied_or_read_only_filtered_tools() {
    for patch in [
        json!({"deny":["write", "bash"]}),
        json!({"read_only":true, "tools":["*", "task"]}),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let model = ScriptedModel::new(vec![
            task(),
            call("write", r#"{"filePath":"escape","content":"bad"}"#),
            answer("child done"),
            answer("done"),
        ]);
        let agent = builder(dir.path(), model.clone(), patch).build().unwrap();
        agent.run_once("go").await.unwrap();
        assert!(!dir.path().join("escape").exists());
        assert!(
            !model.requests()[1]
                .tools
                .iter()
                .any(|tool| matches!(tool.function.name.as_str(), "write" | "bash"))
        );
    }
}

#[cfg(unix)]
#[tokio::test]
async fn backslashes_cannot_masquerade_as_allowed_directory_separators() {
    let dir = tempfile::tempdir().unwrap();
    let model = ScriptedModel::new(vec![
        call(
            "write",
            r#"{"filePath":"allowed\\outside.txt","content":"bad"}"#,
        ),
        call(
            "write",
            r#"{"filePath":"allowed/inside.txt","content":"ok"}"#,
        ),
        answer("done"),
    ]);
    let agent = builder(
        dir.path(),
        model,
        json!({"shell":[], "write_paths":["allowed/*"]}),
    )
    .build()
    .unwrap();
    agent.run_once("go").await.unwrap();
    assert!(!dir.path().join("allowed\\outside.txt").exists());
    assert!(dir.path().join("allowed/inside.txt").is_file());
}

#[cfg(unix)]
#[tokio::test]
async fn hooks_receive_the_literal_backslash_filename() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir(dir.path().join("allowed")).unwrap();
    std::fs::write(dir.path().join("allowed/file.txt"), "ok").unwrap();
    let model = ScriptedModel::new(vec![
        call(
            "write",
            r#"{"filePath":"allowed\\file.txt","content":"bad"}"#,
        ),
        answer("done"),
        call(
            "write",
            r#"{"filePath":"allowed\\file.txt","content":"ok"}"#,
        ),
        answer("fixed"),
    ]);
    let count = Arc::new(AtomicUsize::new(0));
    let agent = builder(
        dir.path(),
        model,
        json!({"hooks":{"before_finish":[
            {"match":"*", "tool":"check", "args":{"path":"{path}"}}
        ]}}),
    )
    .tools([Box::new(CheckFile(count.clone())) as Box<dyn Tool>])
    .build()
    .unwrap();
    assert_eq!(agent.run_once("go").await.unwrap(), "fixed");
    assert_eq!(count.load(Ordering::SeqCst), 2);
}
