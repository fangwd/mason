//! Agent learning: memory, reflection, consolidation and run evaluation.

mod support;

use std::sync::Arc;
use std::time::{Duration, Instant};

use mason::evaluate::LlmEvaluator;
use mason::reflect::RunReview;
use mason::store::{NewInput, NewSession, SessionStore, SqliteStore};
use mason::{Agent, CancellationToken, ChatRequest, Message, OutputSpill, Runner, builtin_tools};
use support::{FailTool, Reply, ScriptedModel, answer, call};

fn agent(dir: &std::path::Path, model: Arc<ScriptedModel>) -> mason::AgentBuilder {
    Agent::builder()
        .model(model)
        .cwd(dir)
        .tools(builtin_tools())
        .spill(Arc::new(OutputSpill::new(dir.join("spill"))))
}

fn system(request: &ChatRequest) -> String {
    request.messages[0].content.clone().unwrap()
}

fn memory_spec(request: &ChatRequest) -> Option<serde_json::Value> {
    request
        .tools
        .iter()
        .find(|spec| spec.function.name == "memory")
        .map(|spec| spec.function.parameters.clone())
}

#[tokio::test]
async fn a_memory_saved_in_one_run_is_indexed_in_the_next() {
    let dir = tempfile::tempdir().unwrap();
    let model = ScriptedModel::new(vec![
        call(
            "memory",
            r#"{"command":"create","name":"orders-schema","description":"orders uses placed_on","body":"Use placed_on.\n**Why:** no created_at.\n**How to apply:** time filters."}"#,
        ),
        answer("saved"),
        answer("done"),
    ]);
    let built = agent(dir.path(), model.clone())
        .memory_dir(dir.path().join("memory"))
        .build()
        .unwrap();

    built.run_once("first").await.unwrap();
    let first = &model.requests()[0];
    assert!(
        system(first).contains("<memory_index>\n(no memories yet)\n</memory_index>"),
        "{}",
        system(first)
    );
    assert!(memory_spec(first).is_some());
    assert!(dir.path().join("memory/orders-schema.md").is_file());

    built.run_once("second").await.unwrap();
    let second = model.requests().pop().unwrap();
    assert!(
        system(&second).contains("- orders-schema: orders uses placed_on"),
        "{}",
        system(&second)
    );
}

#[tokio::test]
async fn without_a_memory_dir_there_is_no_memory() {
    let dir = tempfile::tempdir().unwrap();
    let model = ScriptedModel::new(vec![answer("done")]);
    agent(dir.path(), model.clone())
        .build()
        .unwrap()
        .run_once("hi")
        .await
        .unwrap();
    let request = &model.requests()[0];
    assert!(!system(request).contains("<memory_index>"));
    assert!(memory_spec(request).is_none());
}

#[tokio::test]
async fn subagents_get_read_only_memory() {
    let dir = tempfile::tempdir().unwrap();
    let model = ScriptedModel::new(vec![
        call(
            "task",
            r#"{"description":"d","prompt":"look","agent":"general"}"#,
        ),
        answer("child"),
        answer("done"),
    ]);
    agent(dir.path(), model.clone())
        .memory_dir(dir.path().join("memory"))
        .build()
        .unwrap()
        .run_once("go")
        .await
        .unwrap();
    let requests = model.requests();
    let parent = memory_spec(&requests[0]).unwrap();
    assert_eq!(
        parent.pointer("/properties/command/enum").unwrap(),
        &serde_json::json!(["view", "create", "str_replace", "delete"])
    );
    let child = memory_spec(&requests[1]).unwrap();
    assert_eq!(
        child.pointer("/properties/command/enum").unwrap(),
        &serde_json::json!(["view"])
    );
}

const SAVE: &str = r#"{"command":"create","name":"flaky-retries","description":"flaky fails twice","body":"Avoid flaky.\n**Why:** it failed.\n**How to apply:** use another tool."}"#;

fn failing_run() -> Vec<Reply> {
    vec![
        call("flaky", "{}"),
        call("broken", "{}"),
        call("offline", "{}"),
        answer("done"),
    ]
}

/// Three tools failing with distinct errors, enough to flag a run.
fn failing_tools(builder: mason::AgentBuilder) -> mason::AgentBuilder {
    builder
        .tool(FailTool::boxed("flaky", "flaky is down"))
        .tool(FailTool::boxed("broken", "broken tool"))
        .tool(FailTool::boxed("offline", "offline service"))
}

async fn wait_for_file(path: &std::path::Path) -> String {
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        if let Ok(text) = std::fs::read_to_string(path) {
            return text;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("{} was never written", path.display());
}

#[tokio::test]
async fn reflection_reviews_a_flagged_run_and_saves_with_provenance() {
    let dir = tempfile::tempdir().unwrap();
    let model = ScriptedModel::new(
        failing_run()
            .into_iter()
            .chain([call("memory", SAVE), answer("saved one lesson")])
            .collect(),
    );
    let built = failing_tools(agent(dir.path(), model.clone()))
        .memory_dir(dir.path().join("memory"))
        .build()
        .unwrap();
    let outcome = built.run_prompt("go", None).await.unwrap();
    let review = RunReview {
        session: Some(7),
        ..RunReview::new(&outcome.messages)
    };
    assert!(
        built
            .reflect(review, CancellationToken::new())
            .await
            .unwrap()
    );

    let request = &model.requests()[4];
    let system = system(request);
    assert!(
        system.contains("You review a finished agent run"),
        "{system}"
    );
    assert!(system.contains("(no memories yet)"), "{system}");
    let prompt = request.messages[1].content.clone().unwrap();
    assert!(prompt.contains("- 3 distinct tool errors"), "{prompt}");
    assert!(
        prompt.contains("[Tool result]: Error: flaky is down"),
        "{prompt}"
    );
    assert!(!prompt.contains("You are an agent working"), "{prompt}");
    let names: Vec<&str> = request
        .tools
        .iter()
        .map(|spec| spec.function.name.as_str())
        .collect();
    assert_eq!(names, ["memory"]);

    let saved = std::fs::read_to_string(dir.path().join("memory/flaky-retries.md")).unwrap();
    assert!(
        saved.contains("source: \"reflection on session 7\""),
        "{saved}"
    );
}

#[tokio::test]
async fn unflagged_runs_and_memoryless_agents_are_not_reviewed() {
    let dir = tempfile::tempdir().unwrap();
    let messages = vec![
        Message::user("go"),
        Message::tool("a", "Error: x"),
        Message::tool("b", "Error: y"),
        Message::tool("c", "Error: z"),
    ];

    let model = ScriptedModel::new(vec![]);
    let without_memory = agent(dir.path(), model.clone()).build().unwrap();
    let review = RunReview {
        failed: true,
        ..RunReview::new(&messages)
    };
    assert!(
        !without_memory
            .reflect(review, CancellationToken::new())
            .await
            .unwrap()
    );

    let with_memory = agent(dir.path(), model.clone())
        .memory_dir(dir.path().join("memory"))
        .build()
        .unwrap();
    let clean = RunReview::new(&messages[..2]);
    assert!(
        !with_memory
            .reflect(clean, CancellationToken::new())
            .await
            .unwrap()
    );
    assert_eq!(model.request_count(), 0);
}

async fn served(
    dir: &std::path::Path,
    model: Arc<ScriptedModel>,
) -> (Arc<Runner>, Arc<SqliteStore>, mason::SessionId) {
    serve(
        dir,
        failing_tools(agent(dir, model)).memory_dir(dir.join("memory")),
    )
    .await
}

async fn serve(
    dir: &std::path::Path,
    builder: mason::AgentBuilder,
) -> (Arc<Runner>, Arc<SqliteStore>, mason::SessionId) {
    let store = Arc::new(SqliteStore::open(dir.join("mason.db")).await.unwrap());
    let runner = Runner::new(builder.build().unwrap(), store.clone());
    let session = store.create_session(NewSession::default()).await.unwrap();
    runner
        .submit(session, NewInput::queued("go"))
        .await
        .unwrap();
    (runner, store, session)
}

#[tokio::test]
async fn the_runner_reflects_on_flagged_runs_in_the_background() {
    let dir = tempfile::tempdir().unwrap();
    let model = ScriptedModel::new(
        failing_run()
            .into_iter()
            .chain([call("memory", SAVE), answer("saved")])
            .collect(),
    );
    let (runner, store, session) = served(dir.path(), model.clone()).await;
    let saved = wait_for_file(&dir.path().join("memory/flaky-retries.md")).await;
    assert!(
        saved.contains(&format!("source: \"reflection on session {session}\"")),
        "{saved}"
    );
    let prompt = model.requests()[4].messages[1].content.clone().unwrap();
    assert!(prompt.contains("[User]: go"), "{prompt}");
    let events = store.list_events(session, 0).await.unwrap();
    assert!(
        events
            .iter()
            .any(|event| event.event_type == "run.finished")
    );
    assert!(
        !events
            .iter()
            .any(|event| event.event_type == "run.evaluated"),
        "no evaluator, no verdict"
    );
    runner.shutdown(Duration::from_secs(5)).await;
}

#[tokio::test]
async fn shutdown_cancels_a_running_reflection() {
    let dir = tempfile::tempdir().unwrap();
    let model = ScriptedModel::new(failing_run().into_iter().chain([Reply::Hang]).collect());
    let (runner, _store, _session) = served(dir.path(), model.clone()).await;
    let deadline = Instant::now() + Duration::from_secs(10);
    while model.request_count() < 5 {
        assert!(Instant::now() < deadline, "the reflection never started");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let started = Instant::now();
    runner.shutdown(Duration::from_secs(5)).await;
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "{:?}",
        started.elapsed()
    );
    assert!(!dir.path().join("memory/flaky-retries.md").exists());
}

fn seed(dir: &std::path::Path, count: usize) {
    std::fs::create_dir_all(dir).unwrap();
    for i in 0..count {
        std::fs::write(
            dir.join(format!("lesson-{i}.md")),
            format!("---\ndescription: \"lesson {i}\"\n---\nBody {i}.\n"),
        )
        .unwrap();
    }
}

#[tokio::test]
async fn consolidation_merges_duplicates_and_archives_the_originals() {
    let dir = tempfile::tempdir().unwrap();
    let memory = dir.path().join("memory");
    seed(&memory, 2);
    let model = ScriptedModel::new(vec![
        call(
            "memory",
            r#"{"command":"create","name":"lessons","description":"both lessons","body":"Body 0 and 1."}"#,
        ),
        call("memory", r#"{"command":"delete","name":"lesson-0"}"#),
        call("memory", r#"{"command":"delete","name":"lesson-1"}"#),
        answer("merged two lessons"),
    ]);
    let built = agent(dir.path(), model.clone())
        .memory_dir(&memory)
        .build()
        .unwrap();
    assert!(built.consolidate(CancellationToken::new()).await.unwrap());

    let request = &model.requests()[0];
    assert!(system(request).contains("maintain an agent's persistent memory"));
    let prompt = request.messages[1].content.clone().unwrap();
    assert!(prompt.contains("<memory name=\"lesson-0\">"), "{prompt}");
    assert!(prompt.contains("Body 1."), "{prompt}");
    assert!(
        prompt.contains("<learned_skills>\n\n</learned_skills>"),
        "{prompt}"
    );
    let names: Vec<&str> = request
        .tools
        .iter()
        .map(|spec| spec.function.name.as_str())
        .collect();
    assert_eq!(names, ["memory", "learned_skill"]);

    let merged = std::fs::read_to_string(memory.join("lessons.md")).unwrap();
    assert!(merged.contains("source: \"consolidation\""), "{merged}");
    assert!(!memory.join("lesson-0.md").exists());
    assert_eq!(
        std::fs::read_dir(memory.join(".archive")).unwrap().count(),
        2
    );
    assert!(memory.join(".consolidated").is_file());
}

#[tokio::test]
async fn a_promoted_skill_is_offered_to_later_runs() {
    let dir = tempfile::tempdir().unwrap();
    let model = ScriptedModel::new(vec![
        call(
            "learned_skill",
            r#"{"command":"create","name":"release-notes","description":"Drafts release notes","body":"1. List merged PRs."}"#,
        ),
        answer("promoted"),
        call("skill", r#"{"name":"release-notes"}"#),
        answer("done"),
    ]);
    let built = agent(dir.path(), model.clone())
        .memory_dir(dir.path().join("memory"))
        .build()
        .unwrap();
    built.consolidate(CancellationToken::new()).await.unwrap();

    let outcome = built.run_prompt("write notes", None).await.unwrap();
    let run = &model.requests()[2];
    assert!(
        system(run).contains("<name>release-notes</name>"),
        "{}",
        system(run)
    );
    let loaded = outcome
        .messages
        .iter()
        .find(|message| message.role == mason::Role::Tool)
        .and_then(|message| message.content.clone())
        .unwrap();
    assert!(loaded.contains("1. List merged PRs."), "{loaded}");
}

#[tokio::test]
async fn installed_skills_win_over_learned_ones() {
    let dir = tempfile::tempdir().unwrap();
    let installed = dir.path().join("skills/deploy");
    std::fs::create_dir_all(&installed).unwrap();
    std::fs::write(
        installed.join("SKILL.md"),
        "---\nname: deploy\ndescription: Installed deploy\n---\nInstalled.\n",
    )
    .unwrap();
    let learned = dir.path().join("memory/skills/deploy");
    std::fs::create_dir_all(&learned).unwrap();
    std::fs::write(
        learned.join("SKILL.md"),
        "---\nname: deploy\ndescription: Learned deploy\n---\nLearned.\n",
    )
    .unwrap();
    let model = ScriptedModel::new(vec![answer("done")]);
    agent(dir.path(), model.clone())
        .skills_dir(dir.path().join("skills"))
        .memory_dir(dir.path().join("memory"))
        .build()
        .unwrap()
        .run_once("go")
        .await
        .unwrap();
    let system = system(&model.requests()[0]);
    assert!(system.contains("Installed deploy"), "{system}");
    assert!(!system.contains("Learned deploy"), "{system}");
}

#[tokio::test]
async fn the_runner_consolidates_when_due() {
    let dir = tempfile::tempdir().unwrap();
    seed(&dir.path().join("memory"), mason::memory::CONSOLIDATE_AT);
    let model = ScriptedModel::new(vec![answer("done"), answer("already tidy")]);
    let (runner, _store, _session) = served(dir.path(), model.clone()).await;
    wait_for_file(&dir.path().join("memory/.consolidated")).await;
    assert_eq!(model.request_count(), 2);
    assert!(system(&model.requests()[1]).contains("maintain an agent's persistent memory"));
    runner.shutdown(Duration::from_secs(5)).await;
}

#[tokio::test]
async fn shutdown_cancels_a_running_consolidation() {
    let dir = tempfile::tempdir().unwrap();
    seed(&dir.path().join("memory"), mason::memory::CONSOLIDATE_AT);
    let model = ScriptedModel::new(vec![answer("done"), Reply::Hang]);
    let (runner, _store, _session) = served(dir.path(), model.clone()).await;
    let deadline = Instant::now() + Duration::from_secs(10);
    while model.request_count() < 2 {
        assert!(Instant::now() < deadline, "the consolidation never started");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let started = Instant::now();
    runner.shutdown(Duration::from_secs(5)).await;
    assert!(started.elapsed() < Duration::from_secs(5));
    assert!(!dir.path().join("memory/.consolidated").exists());
}

fn verdict(outcome: &str, reason: &str) -> Reply {
    call(
        "verdict",
        &serde_json::json!({"outcome": outcome, "reason": reason}).to_string(),
    )
}

async fn wait_for_event(
    store: &SqliteStore,
    session: mason::SessionId,
    event_type: &str,
) -> mason::StoreEvent {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let events = store.list_events(session, 0).await.unwrap();
        if let Some(event) = events
            .into_iter()
            .find(|event| event.event_type == event_type)
        {
            return event;
        }
        assert!(Instant::now() < deadline, "no `{event_type}` event");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

#[tokio::test]
async fn the_runner_records_a_verdict_from_a_separate_model() {
    let dir = tempfile::tempdir().unwrap();
    let main = ScriptedModel::new(vec![answer("done")]);
    let judge = ScriptedModel::new(vec![verdict("success", "it answered")]);
    let (runner, store, session) = serve(
        dir.path(),
        agent(dir.path(), main.clone()).evaluator(Arc::new(LlmEvaluator::new(judge.clone()))),
    )
    .await;
    let event = wait_for_event(&store, session, "run.evaluated").await;
    assert_eq!(
        event.data,
        serde_json::json!({"outcome": "success", "reason": "it answered"})
    );
    let finished = wait_for_event(&store, session, "run.finished").await;
    assert_eq!(event.run_id, finished.run_id);
    assert!(event.id > finished.id, "graded after the terminal event");

    let request = &judge.requests()[0];
    assert!(system(request).contains("You grade whether an agent run"));
    assert!(
        request.messages[1]
            .content
            .as_deref()
            .unwrap()
            .contains("[User]: go")
    );
    assert_eq!(main.request_count(), 1, "the main model is not asked");
    assert!(!dir.path().join("memory").exists(), "memory stays off");
    runner.shutdown(Duration::from_secs(5)).await;
}

#[tokio::test]
async fn a_failure_verdict_triggers_reflection_with_its_reason() {
    let dir = tempfile::tempdir().unwrap();
    let main = ScriptedModel::new(vec![answer("done"), call("memory", SAVE), answer("saved")]);
    let judge = ScriptedModel::new(vec![verdict("failure", "the tests never ran")]);
    let (runner, _store, _session) = serve(
        dir.path(),
        agent(dir.path(), main.clone())
            .memory_dir(dir.path().join("memory"))
            .evaluator(Arc::new(LlmEvaluator::new(judge))),
    )
    .await;
    wait_for_file(&dir.path().join("memory/flaky-retries.md")).await;
    let prompt = main.requests()[1].messages[1].content.clone().unwrap();
    assert!(
        prompt.contains("- the evaluator judged it failure: the tests never ran"),
        "{prompt}"
    );
    runner.shutdown(Duration::from_secs(5)).await;
}

#[tokio::test]
async fn a_success_verdict_is_context_not_a_trigger() {
    let dir = tempfile::tempdir().unwrap();
    let model = ScriptedModel::new(vec![answer("recovered"), answer("noted")]);
    let built = agent(dir.path(), model.clone())
        .memory_dir(dir.path().join("memory"))
        .build()
        .unwrap();
    let messages = vec![
        Message::user("go"),
        Message::tool("a", "Error: x"),
        Message::tool("b", "Error: y"),
        Message::tool("c", "Error: z"),
    ];
    let success = mason::evaluate::Verdict {
        outcome: mason::evaluate::Outcome::Success,
        reason: "it worked in the end".into(),
    };
    let clean = RunReview {
        verdict: Some(&success),
        ..RunReview::new(&messages[..1])
    };
    assert!(
        !built
            .reflect(clean, CancellationToken::new())
            .await
            .unwrap()
    );
    let flagged = RunReview {
        verdict: Some(&success),
        ..RunReview::new(&messages)
    };
    assert!(
        built
            .reflect(flagged, CancellationToken::new())
            .await
            .unwrap()
    );
    let prompt = model.requests()[0].messages[1].content.clone().unwrap();
    assert!(
        prompt.contains("- (context) the evaluator judged it success: it worked in the end"),
        "{prompt}"
    );
}

#[tokio::test]
async fn a_session_is_reflected_on_at_most_once_per_cooldown() {
    let dir = tempfile::tempdir().unwrap();
    let model = ScriptedModel::new(
        failing_run()
            .into_iter()
            .chain([answer("no lesson")])
            .chain(failing_run())
            .chain([answer("unexpected second reflection")])
            .collect(),
    );
    let (runner, store, session) = served(dir.path(), model.clone()).await;
    let deadline = Instant::now() + Duration::from_secs(10);
    while model.request_count() < 5 {
        assert!(Instant::now() < deadline, "the first reflection never ran");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    runner
        .submit(session, NewInput::queued("again"))
        .await
        .unwrap();
    while store
        .list_events(session, 0)
        .await
        .unwrap()
        .iter()
        .filter(|event| event.event_type == "run.finished")
        .count()
        < 2
    {
        assert!(Instant::now() < deadline, "the second run never finished");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(model.request_count(), 9, "no second reflection");
    runner.shutdown(Duration::from_secs(5)).await;
}
