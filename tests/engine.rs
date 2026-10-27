//! Engine concurrency, cancellation, limits, and persistence tests.

mod support;

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use mason::engine::{self, PersistFn, SteerFn};
use mason::tool::{Typed, builtin::EditTool};
use mason::{ChatDelta, Error, Limits, Message, PromptSource, Role, RunEvent, builtin_tools};
use support::{
    CancelProbeTool, EchoTool, FailTool, HeldTaskTool, LateFailTool, Reply, ScriptedModel,
    SleepTool, answer, call, call_id, calls, config, recorder, tool_results,
};
use tokio::sync::{Notify, mpsc};
use tokio_util::sync::CancellationToken;

/// Runs scripted replies and returns the answer, tool results, and model.
async fn script(
    replies: Vec<Reply>,
    cwd: &std::path::Path,
) -> (String, Vec<String>, Arc<ScriptedModel>) {
    let (config, model) = config(replies, builtin_tools(), cwd).await;
    let outcome = engine::run(config).await.expect("the run should succeed");
    (outcome.answer, tool_results(&outcome.messages), model)
}

#[tokio::test]
async fn tool_results_go_back_until_the_model_answers() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.txt"), "alpha\nbeta").unwrap();
    let (result, tools, _) = script(
        vec![call("read", r#"{"filePath":"a.txt"}"#), answer("done")],
        dir.path(),
    )
    .await;
    assert_eq!(result, "done");
    assert_eq!(tools[0], "1: alpha\n2: beta\n\n(End of file - 2 lines)");
}

#[tokio::test]
async fn large_bash_output_survives_engine_truncation_with_its_exit_code() {
    let dir = tempfile::tempdir().unwrap();
    let (_, tools, _) = script(
        vec![
            call("bash", r#"{"command":"seq 1 3000; exit 7"}"#),
            answer("ok"),
        ],
        dir.path(),
    )
    .await;
    let result = &tools[0];
    assert!(result.ends_with("[exit code 7]"), "{result}");
    assert_eq!(
        result.matches("Full output saved to").count(),
        1,
        "the engine did not spill the bounded result again: {result}"
    );
    assert!(result.lines().count() <= 2_000);
}

#[tokio::test]
async fn bad_calls_become_fixable_errors() {
    let dir = tempfile::tempdir().unwrap();
    let (_, tools, _) = script(
        vec![
            call("nope", "{}"),
            call("read", "{oops"),
            call("read", r#"{"path":"a.txt"}"#),
            call("read", r#"{"filePath":"missing.txt"}"#),
            answer("ok"),
        ],
        dir.path(),
    )
    .await;
    assert!(tools[0].contains(r#"unknown tool "nope""#), "{tools:?}");
    assert!(
        tools[0].contains("Available tools: read, write, edit, bash"),
        "{tools:?}"
    );
    assert!(tools[1].contains("not valid JSON"), "{tools:?}");
    assert!(tools[2].contains("invalid arguments"), "{tools:?}");
    assert!(tools[2].contains("filePath"), "{tools:?}");
    assert!(tools[3].starts_with("Error:"), "{tools:?}");
    assert!(tools[3].contains("No such file or directory"), "{tools:?}");
}

#[tokio::test]
async fn a_repeated_identical_call_gets_a_warning() {
    let dir = tempfile::tempdir().unwrap();
    let same = || call("bash", r#"{"command":"echo hi"}"#);
    let (_, tools, _) = script(vec![same(), same(), same(), answer("ok")], dir.path()).await;
    assert!(!tools[1].contains("Stop repeating"), "{tools:?}");
    assert!(
        tools[2].contains("exact bash call 3 times in your last 10 calls"),
        "{tools:?}"
    );
}

#[tokio::test]
async fn an_alternating_loop_gets_the_warning_too() {
    let dir = tempfile::tempdir().unwrap();
    let a = || call("bash", r#"{"command":"echo a"}"#);
    let b = || call("bash", r#"{"command":"echo b"}"#);
    let (_, tools, _) = script(vec![a(), b(), a(), b(), a(), answer("ok")], dir.path()).await;
    assert!(
        tools[..4].iter().all(|t| !t.contains("Stop repeating")),
        "{tools:?}"
    );
    assert!(
        tools[4].contains("exact bash call 3 times in your last 10 calls"),
        "{tools:?}"
    );
}

#[tokio::test]
async fn the_step_limit_forces_a_text_answer_without_tools() {
    let dir = tempfile::tempdir().unwrap();
    let (mut config, model) = config(
        vec![call("bash", r#"{"command":"true"}"#), answer("summary")],
        builtin_tools(),
        dir.path(),
    )
    .await;
    config.limits.max_steps = 1;
    let outcome = engine::run(config).await.unwrap();
    assert_eq!(outcome.answer, "summary");

    let requests = model.requests();
    assert_eq!(requests.len(), 2);
    let prompts = PromptSource::Embedded.load().unwrap();
    let last = requests[1].messages.last().unwrap();
    assert_eq!(last.content.as_deref(), Some(prompts.max_steps.as_str()));
    assert!(
        requests[1].tools.is_empty(),
        "tools are disabled on the final turn"
    );
}

#[tokio::test]
async fn identical_outputs_of_different_calls_do_not_trip_the_detector() {
    // Silent commands all print the same text; that is not a loop.
    let dir = tempfile::tempdir().unwrap();
    let (shell, executed) = EchoTool::boxed("bash", "(no output)\n\n[exit code 0]");
    let mut replies: Vec<Reply> = (0..6)
        .map(|n| call("bash", &format!("{{\"n\":{n}}}")))
        .collect();
    replies.push(answer("done"));
    let (config, _model) = config(replies, vec![shell], dir.path()).await;
    let outcome = engine::run(config).await.unwrap();
    assert_eq!(outcome.answer, "done");
    assert_eq!(executed.load(Ordering::SeqCst), 6);
}

#[tokio::test]
async fn scattered_repeats_do_not_stop_a_long_run() {
    let dir = tempfile::tempdir().unwrap();
    let same = || call("bash", r#"{"command":"echo same"}"#);
    let other = |n: usize| call("bash", &format!(r#"{{"command":"echo other{n}"}}"#));
    // Three warnings in total, each followed by a clean step: the streak
    // never reaches three, so the run completes.
    let (answer_text, tools, _) = script(
        vec![
            same(),
            same(),
            same(),
            other(1),
            same(),
            other(2),
            same(),
            other(3),
            answer("ok"),
        ],
        dir.path(),
    )
    .await;
    assert_eq!(answer_text, "ok");
    let warned: Vec<bool> = tools.iter().map(|t| t.contains("Stop repeating")).collect();
    assert_eq!(
        warned,
        [false, false, true, false, true, false, true, false],
        "{tools:?}"
    );
}

#[tokio::test]
async fn a_repeat_during_recovery_stops_without_executing_the_disabled_tool() {
    let dir = tempfile::tempdir().unwrap();
    let (echo, executed) = EchoTool::boxed("echo", "unchanged");
    let replies: Vec<Reply> = (0..20).map(|_| call("echo", r#"{"n":0}"#)).collect();
    let (mut config, model) = config(replies, vec![echo], dir.path()).await;
    let (persist, saved) = recorder();
    config.persist = Some(persist);
    let err = engine::run(config).await.unwrap_err();
    assert!(err.to_string().contains("3 doom-loop warnings"), "{err}");
    // Repeats 3, 4 and 5 warn. One recovery request follows, but its
    // attempt to call the withheld tool never executes.
    assert_eq!(executed.load(Ordering::SeqCst), 5);
    assert_eq!(model.request_count(), 6);
    assert!(
        err.to_string().contains("disabled tool during recovery"),
        "{err}"
    );
    let recovery = &model.requests()[5];
    assert!(recovery.tools.is_empty());
    assert!(
        recovery
            .messages
            .last()
            .unwrap()
            .content
            .as_deref()
            .unwrap()
            .contains("No tools are available for this turn")
    );
    let tool_messages = saved
        .lock()
        .unwrap()
        .iter()
        .filter(|message| message.role == mason::Role::Tool)
        .count();
    assert_eq!(tool_messages, 6);
}

#[tokio::test]
async fn a_doom_loop_can_recover_with_a_different_tool() {
    let dir = tempfile::tempdir().unwrap();
    let (inspect, inspected) = EchoTool::boxed("inspect", "unchanged evidence");
    let (validate, validated) = EchoTool::boxed("validate", "all fixtures passed");
    let mut replies: Vec<Reply> = (0..5).map(|_| Reply::Message(
        serde_json::from_value(serde_json::json!({
            "role": "assistant", "content": null, "reasoning_content": "inspect again",
            "tool_calls": [{"id": "inspect", "type": "function", "function": {"name": "inspect", "arguments": "{}"}}]
        })).unwrap()
    )).collect();
    replies.extend([call("validate", "{}"), answer("validated")]);
    let (config, model) = config(replies, vec![inspect, validate], dir.path()).await;
    let outcome = engine::run(config).await.unwrap();
    assert_eq!(outcome.answer, "validated");
    assert_eq!(inspected.load(Ordering::SeqCst), 5);
    assert_eq!(validated.load(Ordering::SeqCst), 1);
    let requests = model.requests();
    assert!(
        requests[4]
            .messages
            .iter()
            .any(|message| message.reasoning_content.is_some())
    );
    assert!(
        requests[5]
            .messages
            .iter()
            .all(|message| message.reasoning_content.is_none()),
        "the recovery instruction starts a new reasoning round"
    );
    assert_eq!(
        requests[5]
            .tools
            .iter()
            .map(|tool| tool.function.name.as_str())
            .collect::<Vec<_>>(),
        ["validate"]
    );
    assert!(
        requests[5]
            .messages
            .last()
            .unwrap()
            .content
            .as_deref()
            .unwrap()
            .contains("Mason runtime recovery")
    );
    assert_eq!(
        requests[6].tools.len(),
        2,
        "normal tools return after the recovery turn"
    );
    assert_eq!(tool_results(&outcome.messages).len(), 6);
}

#[tokio::test]
async fn doom_recovery_does_not_extend_the_step_budget() {
    let dir = tempfile::tempdir().unwrap();
    let (inspect, inspected) = EchoTool::boxed("inspect", "unchanged evidence");
    let (validate, validated) = EchoTool::boxed("validate", "should not run");
    let mut replies: Vec<Reply> = (0..5).map(|_| call("inspect", "{}")).collect();
    replies.push(answer("The work is unfinished; the step budget is spent."));
    let (mut config, model) = config(replies, vec![inspect, validate], dir.path()).await;
    config.limits.max_steps = 5;
    let outcome = engine::run(config).await.unwrap();
    assert!(outcome.answer.contains("unfinished"));
    assert_eq!(inspected.load(Ordering::SeqCst), 5);
    assert_eq!(validated.load(Ordering::SeqCst), 0);
    let requests = model.requests();
    assert_eq!(requests.len(), 6);
    assert!(
        requests[5].tools.is_empty(),
        "the wrap-up turn stays text-only"
    );
    let prompts = PromptSource::Embedded.load().unwrap();
    assert!(
        requests[5]
            .messages
            .iter()
            .any(|message| message.content.as_deref() == Some(prompts.max_steps.as_str()))
    );
}

#[tokio::test]
async fn a_recovery_on_the_final_turn_keeps_the_answer_despite_a_withheld_call() {
    let dir = tempfile::tempdir().unwrap();
    let (inspect, inspected) = EchoTool::boxed("inspect", "unchanged evidence");
    let mut replies: Vec<Reply> = (0..5).map(|_| call("inspect", "{}")).collect();
    replies.push(Reply::Message(
        serde_json::from_value(serde_json::json!({
            "role": "assistant", "content": "Partial summary of the work.",
            "tool_calls": [{"id": "late", "type": "function", "function": {"name": "inspect", "arguments": "{}"}}]
        }))
        .unwrap(),
    ));
    let (mut config, model) = config(replies, vec![inspect], dir.path()).await;
    config.limits.max_steps = 5;
    let outcome = engine::run(config).await.unwrap();
    assert_eq!(outcome.answer, "Partial summary of the work.");
    assert_eq!(inspected.load(Ordering::SeqCst), 5);
    let requests = model.requests();
    assert_eq!(requests.len(), 6);
    assert!(requests[5].tools.is_empty());
    assert!(
        requests[5].messages.iter().all(|message| !message
            .content
            .as_deref()
            .unwrap_or_default()
            .contains("Mason runtime recovery")),
        "the final turn sends only the wrap-up prompt"
    );
}

#[tokio::test]
async fn a_second_doom_loop_stops_instead_of_recovering_forever() {
    let dir = tempfile::tempdir().unwrap();
    let (inspect, inspected) = EchoTool::boxed("inspect", "unchanged evidence");
    let (other, used_other) = EchoTool::boxed("other", "unchanged alternative");
    let mut replies: Vec<Reply> = (0..5).map(|_| call("inspect", "{}")).collect();
    replies.extend((0..10).map(|_| call("other", "{}")));
    let (config, model) = config(replies, vec![inspect, other], dir.path()).await;
    let err = engine::run(config).await.unwrap_err();
    assert!(
        err.to_string().contains("following a recovery attempt"),
        "{err}"
    );
    assert_eq!(inspected.load(Ordering::SeqCst), 5);
    assert_eq!(used_other.load(Ordering::SeqCst), 5);
    assert_eq!(model.request_count(), 10);
}

#[tokio::test]
async fn a_blocked_recovery_reply_executes_no_sibling_calls_and_persists_all_results() {
    let dir = tempfile::tempdir().unwrap();
    let (inspect, inspected) = EchoTool::boxed("inspect", "unchanged evidence");
    let (other, used_other) = EchoTool::boxed("other", "should not run");
    let mut replies: Vec<Reply> = (0..5).map(|_| call("inspect", "{}")).collect();
    replies.push(calls(vec![
        ("allowed".into(), "other".into(), "{}".into()),
        ("blocked".into(), "inspect".into(), "{}".into()),
    ]));
    let (mut config, _) = config(replies, vec![inspect, other], dir.path()).await;
    let (persist, saved) = recorder();
    config.persist = Some(persist);
    let err = engine::run(config).await.unwrap_err();
    assert!(
        err.to_string().contains("disabled tool during recovery"),
        "{err}"
    );
    assert_eq!(inspected.load(Ordering::SeqCst), 5);
    assert_eq!(used_other.load(Ordering::SeqCst), 0);
    let saved = saved.lock().unwrap();
    let results: Vec<_> = saved
        .iter()
        .filter(|message| message.role == Role::Tool)
        .collect();
    assert_eq!(results.len(), 7);
    assert_eq!(results[5].tool_call_id.as_deref(), Some("allowed"));
    assert_eq!(results[6].tool_call_id.as_deref(), Some("blocked"));
}

#[tokio::test]
async fn bounds_fan_out_and_saves_fast_results_while_a_slow_sibling_is_pending() {
    let dir = tempfile::tempdir().unwrap();
    let (task_tool, gauge, release) = HeldTaskTool::boxed();
    let items = (0..8)
        .map(|n| (n.to_string(), "task".to_owned(), format!(r#"{{"n":{n}}}"#)))
        .collect();
    let (mut config, _model) = config(
        vec![calls(items), answer("done")],
        vec![task_tool],
        dir.path(),
    )
    .await;
    config.limits.max_concurrent_tools = 2;
    config.limits.max_task_calls_per_step = 3;

    let saved: Arc<Mutex<Vec<Message>>> = Arc::new(Mutex::new(Vec::new()));
    let fast_saved = Arc::new(Notify::new());
    let persist: PersistFn = Arc::new({
        let saved = saved.clone();
        let fast_saved = fast_saved.clone();
        move |_turn, message| {
            let saved = saved.clone();
            let fast_saved = fast_saved.clone();
            Box::pin(async move {
                let is_fast = message.content.as_deref() == Some("result 1");
                saved.lock().unwrap().push(message);
                if is_fast {
                    fast_saved.notify_one();
                }
                Ok(())
            })
        }
    });
    config.persist = Some(persist);

    let completed = tokio::spawn(engine::run(config));
    fast_saved.notified().await;
    let has_slow = saved
        .lock()
        .unwrap()
        .iter()
        .any(|message| message.content.as_deref() == Some("result 0"));
    assert!(
        !has_slow,
        "the slow sibling must not be saved before it completes"
    );

    release.notify_one();
    let outcome = completed.await.unwrap().expect("the run should succeed");
    assert_eq!(outcome.answer, "done");
    assert_eq!(gauge.peak.load(Ordering::SeqCst), 2, "fan-out is bounded");
    assert_eq!(*gauge.executed.lock().unwrap(), ["0", "1", "2"]);
    let messages = saved.lock().unwrap();
    let tool_messages: Vec<&Message> = messages
        .iter()
        .filter(|message| message.role == mason::Role::Tool)
        .collect();
    assert_eq!(tool_messages.len(), 8);
    let capped = tool_messages
        .iter()
        .filter(|message| {
            message
                .content
                .as_deref()
                .is_some_and(|c| c.contains("call limit"))
        })
        .count();
    assert_eq!(capped, 5);
}

#[tokio::test]
async fn steer_inputs_are_not_consumed_during_the_final_summary() {
    let dir = tempfile::tempdir().unwrap();
    let (mut config, _model) = config(
        vec![call("read", r#"{"filePath":"a.txt"}"#), answer("summary")],
        builtin_tools(),
        dir.path(),
    )
    .await;
    std::fs::write(dir.path().join("a.txt"), "x").unwrap();
    config.limits.max_steps = 1;
    let steer_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let steer: SteerFn = Arc::new({
        let steer_count = steer_count.clone();
        move || {
            let steer_count = steer_count.clone();
            Box::pin(async move {
                steer_count.fetch_add(1, Ordering::SeqCst);
                Ok(Vec::new())
            })
        }
    });
    config.steer = Some(steer);
    let outcome = engine::run(config).await.unwrap();
    assert_eq!(outcome.answer, "summary");
    assert_eq!(
        steer_count.load(Ordering::SeqCst),
        1,
        "skipped on the final turn"
    );
}

#[tokio::test(start_paused = true)]
async fn a_step_starting_near_the_deadline_wraps_up_without_tools() {
    let dir = tempfile::tempdir().unwrap();
    let (mut config, model) = config(
        vec![call("slow", "{}"), answer("summary")],
        vec![SleepTool::boxed("slow", Duration::from_millis(950), "ok")],
        dir.path(),
    )
    .await;
    config.limits.timeout = Duration::from_secs(1);
    let outcome = engine::run(config).await.unwrap();
    assert_eq!(outcome.answer, "summary");
    assert!(model.requests()[1].tools.is_empty());
}

#[tokio::test]
async fn a_nearly_spent_token_budget_wraps_up_without_tools() {
    let dir = tempfile::tempdir().unwrap();
    let big = "x".repeat(40_000);
    let tools = || vec![EchoTool::boxed("big", &big).0];
    let (probe, probe_model) = config(vec![answer("probe")], tools(), dir.path()).await;
    engine::run(probe).await.unwrap();
    let request = &probe_model.requests()[0];
    let first = mason::context::estimate_tokens(&request.messages)
        + mason::context::estimate_str(&serde_json::to_string(&request.tools).unwrap());

    // The first request fits twice; after a 10k-token result, the next does not.
    let (mut config, model) = config(
        vec![call("big", "{}"), answer("summary")],
        tools(),
        dir.path(),
    )
    .await;
    config.limits.max_total_tokens = 2 * first + 15_000;
    let outcome = engine::run(config).await.unwrap();
    assert_eq!(outcome.answer, "summary");
    let requests = model.requests();
    assert!(!requests[0].tools.is_empty());
    assert!(requests[1].tools.is_empty());
}

#[tokio::test]
async fn calls_over_the_tool_call_budget_are_skipped_then_the_run_wraps_up() {
    let dir = tempfile::tempdir().unwrap();
    let (echo, executed) = EchoTool::boxed("echo", "hi");
    let items = (0..2)
        .map(|n| (n.to_string(), "echo".to_owned(), format!(r#"{{"n":{n}}}"#)))
        .collect();
    let (mut config, model) = config(
        vec![calls(items), answer("summary")],
        vec![echo],
        dir.path(),
    )
    .await;
    config.limits.max_total_tool_calls = 1;
    config.limits.max_concurrent_tools = 1;
    let outcome = engine::run(config).await.unwrap();
    assert_eq!(outcome.answer, "summary");
    assert_eq!(executed.load(Ordering::SeqCst), 1);
    let tools = tool_results(&outcome.messages);
    assert!(
        tools[1].contains("tool-call budget is exhausted"),
        "{tools:?}"
    );
    assert!(model.requests()[1].tools.is_empty());
}

#[tokio::test(start_paused = true)]
async fn the_run_deadline_aborts_an_in_flight_call() {
    let dir = tempfile::tempdir().unwrap();
    let (mut config, _model) = config(vec![Reply::Hang], vec![], dir.path()).await;
    config.limits.timeout = Duration::from_millis(20);
    let err = engine::run(config).await.unwrap_err();
    assert!(err.to_string().contains("deadline"), "{err}");
    assert!(matches!(err, Error::RunDeadline(_)), "{err}");
}

/// Port of the deadline test's tool-signal half: an in-flight call observes
/// the cancellation through its context token.
#[tokio::test]
async fn the_run_deadline_reaches_tool_signals() {
    let dir = tempfile::tempdir().unwrap();
    let (probe, flag) = CancelProbeTool::boxed();
    let (mut config, _model) = config(
        vec![call_id("p", "probe", "{}"), answer("never")],
        vec![probe],
        dir.path(),
    )
    .await;
    config.limits.timeout = Duration::from_millis(200);
    let err = engine::run(config).await.unwrap_err();
    assert!(matches!(err, Error::RunDeadline(_)), "{err}");
    assert_eq!(
        flag.load(Ordering::SeqCst),
        1,
        "the cancellation reached the tool"
    );
}

#[tokio::test]
async fn delta_ordering_matches_the_folded_message() {
    let dir = tempfile::tempdir().unwrap();
    let (mut config, _model) = config(
        vec![Reply::Deltas(vec![
            ChatDelta::Text("a".into()),
            ChatDelta::Text("b".into()),
        ])],
        vec![],
        dir.path(),
    )
    .await;
    let (tx, mut rx) = mpsc::channel(64);
    config.events = Some(tx);
    let outcome = engine::run(config).await.unwrap();
    assert_eq!(outcome.answer, "ab");
    let mut events = Vec::new();
    while let Ok(event) = rx.try_recv() {
        events.push(event);
    }
    let deltas: Vec<(u32, u32, String)> = events
        .iter()
        .filter_map(|event| match event {
            RunEvent::Delta {
                turn,
                seq,
                delta: ChatDelta::Text(text),
                ..
            } => Some((*turn, *seq, text.clone())),
            _ => None,
        })
        .collect();
    assert_eq!(deltas[0], (1, 0, "a".into()));
    assert_eq!(deltas[1], (1, 1, "b".into()));
    // The committed assistant message follows its deltas, matches them and carries the
    // run and turn it belongs to; every event of the run shares one run id.
    let committed = events.iter().find_map(|event| match event {
        RunEvent::Committed { message, turn, .. } if message.role == mason::Role::Assistant => {
            Some((message.content.clone(), *turn))
        }
        _ => None,
    });
    assert_eq!(
        committed,
        Some((Some("ab".to_owned()), 1)),
        "the committed message names its turn"
    );
    let run_ids: std::collections::HashSet<String> = events
        .iter()
        .map(|event| match event {
            RunEvent::Delta { run_id, .. }
            | RunEvent::Committed { run_id, .. }
            | RunEvent::ToolStarted { run_id, .. } => run_id.to_string(),
        })
        .collect();
    assert_eq!(run_ids.len(), 1, "one run id per run: {run_ids:?}");
    assert!(
        run_ids.iter().all(|id| !id.is_empty()),
        "the ephemeral run id is present: {run_ids:?}"
    );
}

#[tokio::test]
async fn tool_errors_let_siblings_finish() {
    let dir = tempfile::tempdir().unwrap();
    let tools = vec![
        FailTool::boxed("flaky", "boom"),
        SleepTool::boxed("slow", Duration::from_millis(150), "slow ok"),
    ];
    let (config, _model) = config(
        vec![
            calls(vec![
                ("f".into(), "flaky".into(), "{}".into()),
                ("s".into(), "slow".into(), "{}".into()),
            ]),
            answer("done"),
        ],
        tools,
        dir.path(),
    )
    .await;
    let outcome = engine::run(config).await.unwrap();
    let results = tool_results(&outcome.messages);
    assert!(results.iter().any(|r| r == "Error: boom"), "{results:?}");
    assert!(results.iter().any(|r| r == "slow ok"), "{results:?}");
    assert_eq!(outcome.answer, "done");
}

#[tokio::test]
async fn fatal_errors_cancel_and_drain_siblings() {
    let dir = tempfile::tempdir().unwrap();
    let (probe, flag) = CancelProbeTool::boxed();
    let tools = vec![FailTool::boxed_fatal("killer", "fatal boom"), probe];
    let (config, _model) = config(
        vec![
            calls(vec![
                ("k".into(), "killer".into(), "{}".into()),
                ("p".into(), "probe".into(), "{}".into()),
            ]),
            answer("never"),
        ],
        tools,
        dir.path(),
    )
    .await;
    let started = std::time::Instant::now();
    let err = engine::run(config).await.unwrap_err();
    assert!(err.to_string().contains("fatal boom"), "{err}");
    assert_eq!(
        flag.load(Ordering::SeqCst),
        1,
        "the sibling saw the cancellation"
    );
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "draining is bounded"
    );
}

#[tokio::test]
async fn observer_failure_cannot_corrupt_history() {
    let dir = tempfile::tempdir().unwrap();
    // A closed receiver...
    let (mut closed_config, _model) = config(
        vec![call("bash", r#"{"command":"echo hi"}"#), answer("done")],
        builtin_tools(),
        dir.path(),
    )
    .await;
    let (tx, rx) = mpsc::channel(4);
    drop(rx);
    closed_config.events = Some(tx);
    let outcome = engine::run(closed_config)
        .await
        .expect("a dead observer must not fail the run");
    assert_eq!(outcome.answer, "done");

    // ...and a full one that nobody reads.
    let (mut stalled_config, _model) = config(
        vec![call("bash", r#"{"command":"echo hi"}"#), answer("done")],
        builtin_tools(),
        dir.path(),
    )
    .await;
    let (tx, _stalled) = mpsc::channel(1);
    stalled_config.events = Some(tx);
    let outcome = engine::run(stalled_config)
        .await
        .expect("a slow observer must not fail the run");
    assert_eq!(outcome.answer, "done");
    assert!(tool_results(&outcome.messages)[0].contains("hi"));
}

#[tokio::test]
async fn a_cut_off_reply_is_discarded_and_retried_with_a_notice() {
    let dir = tempfile::tempdir().unwrap();
    let (mut config, model) = config(
        vec![
            Reply::Deltas(vec![
                ChatDelta::Text("partial".into()),
                ChatDelta::Done {
                    finish_reason: Some("length".into()),
                },
            ]),
            answer("done"),
        ],
        builtin_tools(),
        dir.path(),
    )
    .await;
    let prompts = config.prompts.clone();
    let (persist, saved) = recorder();
    config.persist = Some(persist);
    let outcome = engine::run(config).await.unwrap();
    assert_eq!(outcome.answer, "done");
    let requests = model.requests();
    let notice = requests[1].messages.last().unwrap();
    assert_eq!(
        notice.content.as_deref(),
        Some(prompts.output_limit.as_str())
    );
    assert!(
        requests[2..].is_empty() && !requests[1].tools.is_empty(),
        "one retry with tools"
    );
    let saved = saved.lock().unwrap();
    assert!(
        saved
            .iter()
            .all(|message| message.content.as_deref() != Some("partial")),
        "the cut-off reply was not persisted"
    );
}

#[tokio::test]
async fn required_persistence_failure_stops_the_run() {
    let dir = tempfile::tempdir().unwrap();
    let (mut config, model) = config(
        vec![call("bash", r#"{"command":"echo hi"}"#), answer("done")],
        builtin_tools(),
        dir.path(),
    )
    .await;
    config.persist = Some(Arc::new(|_turn, _message| {
        Box::pin(async { Err(Error::Run("store down".to_owned())) })
    }));
    let err = engine::run(config).await.unwrap_err();
    assert!(err.to_string().contains("store down"), "{err}");
    assert_eq!(
        model.request_count(),
        1,
        "the run stopped at the first persistence failure"
    );
}

#[tokio::test]
async fn invalid_limits_are_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let (mut zero_tools, _model) = config(vec![answer("done")], vec![], dir.path()).await;
    zero_tools.limits = Limits {
        max_concurrent_tools: 0,
        ..Limits::default()
    };
    let err = engine::run(zero_tools).await.unwrap_err();
    assert!(err.to_string().contains("positive integers"), "{err}");

    let (mut zero_timeout, _model) = config(vec![answer("done")], vec![], dir.path()).await;
    zero_timeout.limits.timeout = Duration::ZERO;
    let err = engine::run(zero_timeout).await.unwrap_err();
    assert!(err.to_string().contains("timeout"), "{err}");
}

#[tokio::test(start_paused = true)]
async fn cancelling_a_run_with_an_uncooperative_tool_still_ends_it() {
    let dir = tempfile::tempdir().unwrap();
    let (config, _model) = config(
        vec![
            calls(vec![("h".into(), "hang".into(), "{}".into())]),
            answer("never"),
        ],
        // SleepTool never looks at the cancellation token.
        vec![SleepTool::boxed("hang", Duration::from_secs(60), "never")],
        dir.path(),
    )
    .await;
    let external = CancellationToken::new();
    let canceller = external.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(100)).await;
        canceller.cancel();
    });
    let started = tokio::time::Instant::now();
    let err = engine::run_with_token(config, external).await.unwrap_err();
    assert!(matches!(err, Error::Cancelled), "{err}");
    let elapsed = started.elapsed();
    assert!(
        elapsed >= Duration::from_secs(5),
        "the cleanup grace ran: {elapsed:?}"
    );
    assert!(
        elapsed < Duration::from_secs(10),
        "the uncooperative tool was aborted: {elapsed:?}"
    );
}

#[tokio::test(start_paused = true)]
async fn the_cleanup_grace_starts_when_the_failure_happens() {
    let dir = tempfile::tempdir().unwrap();
    let tools = vec![
        LateFailTool::boxed("latefail", Duration::from_secs(10), "late boom"),
        SleepTool::boxed("hang", Duration::from_secs(60), "never"),
    ];
    let (config, _model) = config(
        vec![
            calls(vec![
                ("f".into(), "latefail".into(), "{}".into()),
                ("h".into(), "hang".into(), "{}".into()),
            ]),
            answer("never"),
        ],
        tools,
        dir.path(),
    )
    .await;
    let started = tokio::time::Instant::now();
    let err = engine::run(config).await.unwrap_err();
    assert!(err.to_string().contains("late boom"), "{err}");
    let elapsed = started.elapsed();
    // Grace measured from launch would abort immediately at t=10s (the
    // deadline having passed); the sibling must get until t≈15s.
    assert!(
        elapsed >= Duration::from_secs(14),
        "the sibling got its grace after the failure: {elapsed:?}"
    );
    assert!(
        elapsed < Duration::from_secs(20),
        "cleanup stays bounded: {elapsed:?}"
    );
}

#[tokio::test]
async fn concurrent_edits_of_one_file_all_survive() {
    for iteration in 0..8 {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("f.txt");
        std::fs::write(&file, "one\ntwo\nthree\nfour\n").unwrap();
        let edit_call = |id: &str, old: &str, new: &str| {
            (
                id.to_owned(),
                "edit".to_owned(),
                serde_json::json!({
                    "filePath": "f.txt",
                    "oldString": old,
                    "newString": new,
                })
                .to_string(),
            )
        };
        let (config, _model) = config(
            vec![
                calls(vec![
                    edit_call("e1", "one", "1"),
                    edit_call("e2", "two", "2"),
                    edit_call("e3", "three", "3"),
                    edit_call("e4", "four", "4"),
                ]),
                answer("done"),
            ],
            vec![Box::new(Typed(EditTool))],
            dir.path(),
        )
        .await;
        let outcome = engine::run(config).await.unwrap();
        assert_eq!(outcome.answer, "done");
        assert_eq!(
            std::fs::read_to_string(&file).unwrap(),
            "1\n2\n3\n4\n",
            "iteration {iteration}: every concurrent edit survives"
        );
    }
}

#[tokio::test]
async fn persistence_keeps_conversation_order_under_a_slow_callback() {
    let dir = tempfile::tempdir().unwrap();
    let tools = vec![
        SleepTool::boxed("first", Duration::from_millis(10), "A"),
        SleepTool::boxed("second", Duration::from_millis(40), "B"),
    ];
    let (mut config, _model) = config(
        vec![
            calls(vec![
                ("a".into(), "first".into(), "{}".into()),
                ("b".into(), "second".into(), "{}".into()),
            ]),
            answer("done"),
        ],
        tools,
        dir.path(),
    )
    .await;
    let recorded: Arc<Mutex<Vec<Message>>> = Arc::new(Mutex::new(Vec::new()));
    config.persist = Some(Arc::new({
        let recorded = recorded.clone();
        move |_turn: u32, message: Message| {
            let recorded = recorded.clone();
            Box::pin(async move {
                // The first result to complete persists slowly: without the
                // commit lock, B's persistence would overtake A's and the
                // durable order would be [B, A].
                if message.content.as_deref() == Some("A") {
                    tokio::time::sleep(Duration::from_millis(150)).await;
                }
                recorded.lock().unwrap().push(message);
                Ok(())
            })
        }
    }));
    let outcome = engine::run(config).await.unwrap();
    assert_eq!(outcome.answer, "done");
    let tool_order: Vec<String> = recorded
        .lock()
        .unwrap()
        .iter()
        .filter(|message| message.role == Role::Tool)
        .map(|message| message.content.clone().unwrap_or_default())
        .collect();
    assert_eq!(
        tool_order,
        ["A".to_owned(), "B".to_owned()],
        "the durable order equals the conversation order"
    );
    // The conversation itself holds the same order.
    let conversation: Vec<String> = outcome
        .messages
        .iter()
        .filter(|message| message.role == Role::Tool)
        .map(|message| message.content.clone().unwrap_or_default())
        .collect();
    assert_eq!(conversation, ["A".to_owned(), "B".to_owned()]);
}

#[cfg(unix)]
#[tokio::test]
async fn concurrent_edits_through_a_symlink_alias_all_survive() {
    for iteration in 0..8 {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("target.txt");
        std::fs::write(&file, "x\ny\n").unwrap();
        std::os::unix::fs::symlink(&file, dir.path().join("alias.txt")).unwrap();
        let (config, _model) = config(
            vec![
                calls(vec![
                    (
                        "e1".into(),
                        "edit".into(),
                        serde_json::json!({
                            "filePath": "target.txt",
                            "oldString": "x",
                            "newString": "1",
                        })
                        .to_string(),
                    ),
                    (
                        "e2".into(),
                        "edit".into(),
                        serde_json::json!({
                            "filePath": "alias.txt",
                            "oldString": "y",
                            "newString": "2",
                        })
                        .to_string(),
                    ),
                ]),
                answer("done"),
            ],
            vec![Box::new(Typed(EditTool))],
            dir.path(),
        )
        .await;
        engine::run(config).await.unwrap();
        assert_eq!(
            std::fs::read_to_string(&file).unwrap(),
            "1\n2\n",
            "iteration {iteration}: edits through the alias share the lock"
        );
    }
}

#[tokio::test(start_paused = true)]
async fn a_hung_persist_callback_cannot_outlive_the_run_deadline() {
    let dir = tempfile::tempdir().unwrap();
    let (echo, _gauge) = EchoTool::boxed("echo", "hi");
    let (mut config, _model) = config(
        vec![call_id("e1", "echo", "{}"), answer("never")],
        vec![echo],
        dir.path(),
    )
    .await;
    config.limits.timeout = Duration::from_millis(300);
    config.persist = Some(Arc::new(|_turn, _message| {
        Box::pin(async move { std::future::pending::<mason::Result<()>>().await })
    }));
    let started = tokio::time::Instant::now();
    let err = engine::run(config).await.unwrap_err();
    assert!(matches!(err, Error::RunDeadline(_)), "{err}");
    let elapsed = started.elapsed();
    assert!(
        elapsed >= Duration::from_millis(300),
        "the deadline fired first: {elapsed:?}"
    );
    // Bounded by the deadline plus the cleanup grace, never the hang.
    assert!(
        elapsed < Duration::from_secs(7),
        "the hung persist was abandoned after the grace: {elapsed:?}"
    );
}

#[tokio::test(start_paused = true)]
async fn a_hung_steer_callback_cannot_outlive_the_run_deadline() {
    let dir = tempfile::tempdir().unwrap();
    let (mut config, _model) = config(vec![answer("never")], vec![], dir.path()).await;
    config.limits.timeout = Duration::from_millis(300);
    config.steer = Some(Arc::new(|| {
        Box::pin(async move { std::future::pending::<mason::Result<Vec<Message>>>().await })
    }));
    let started = tokio::time::Instant::now();
    let err = engine::run(config).await.unwrap_err();
    assert!(matches!(err, Error::RunDeadline(_)), "{err}");
    let elapsed = started.elapsed();
    assert!(
        elapsed < Duration::from_secs(3),
        "the deadline bounded the hung steer pull: {elapsed:?}"
    );
}

/// Passes when the file at `path` says "ok", as a crawler check would on a fixed file.
struct CheckTool;

#[async_trait::async_trait]
impl mason::Tool for CheckTool {
    fn spec(&self) -> mason::ToolSpec {
        mason::ToolSpec::function(
            "check",
            "checks a file",
            serde_json::json!({"type": "object"}),
        )
    }
    async fn call(
        &self,
        args: serde_json::Value,
        ctx: &mason::ToolCtx,
    ) -> std::result::Result<String, mason::ToolError> {
        let path = args["path"].as_str().unwrap_or_default();
        let text = std::fs::read_to_string(ctx.cwd.join(path)).unwrap_or_default();
        if text.contains("ok") {
            Ok(format!("{path} is ok"))
        } else {
            Err(mason::ToolError::Failure(format!("{path} is broken")))
        }
    }
}

fn rules(shell: Option<&[&str]>, write_paths: Option<&[&str]>) -> mason::Rules {
    let hook = mason::Hook {
        pattern: "crawlers/*.ts".into(),
        tool: "check".into(),
        args: serde_json::from_value(serde_json::json!({"path": "{path}"})).unwrap(),
    };
    mason::Rules {
        shell: shell.map(|list| list.iter().map(|s| s.to_string()).collect()),
        write_paths: write_paths.map(|list| list.iter().map(|s| s.to_string()).collect()),
        hooks: mason::Hooks {
            after_edit: vec![hook.clone()],
            before_finish: vec![hook],
        },
    }
}

#[tokio::test]
async fn hooks_check_each_edit_and_hold_the_finish_until_checks_pass() {
    let dir = tempfile::tempdir().unwrap();
    let mut tools = builtin_tools();
    tools.push(Box::new(CheckTool));
    let (mut config, model) = config(
        vec![
            call(
                "write",
                r#"{"filePath":"crawlers/shop.ts","content":"bad"}"#,
            ),
            answer("done"),
            call(
                "write",
                r#"{"filePath":"crawlers/shop.ts","content":"ok now"}"#,
            ),
            answer("done"),
        ],
        tools,
        dir.path(),
    )
    .await;
    config.ctx = config.ctx.with_rules(Arc::new(rules(None, None)));
    let outcome = engine::run(config).await.unwrap();
    assert_eq!(outcome.answer, "done");
    let results = tool_results(&outcome.messages);
    assert!(
        results[0].contains(
            "[check on crawlers/shop.ts after this edit: FAILED]\ncrawlers/shop.ts is broken"
        ),
        "{}",
        results[0]
    );
    assert!(
        results[1].contains("after this edit: passed"),
        "{}",
        results[1]
    );
    // the first "done" was sent back with the failed check
    let requests = model.requests();
    let sent_back = requests[2]
        .messages
        .last()
        .unwrap()
        .content
        .clone()
        .unwrap_or_default();
    assert!(
        sent_back.contains("[Mason runtime check]")
            && sent_back.contains("check on crawlers/shop.ts:\ncrawlers/shop.ts is broken"),
        "{sent_back}"
    );
    assert_eq!(model.request_count(), 4);
}

#[tokio::test]
async fn profile_rules_keep_bash_and_writes_to_what_they_allow() {
    let dir = tempfile::tempdir().unwrap();
    let mut tools = builtin_tools();
    tools.push(Box::new(CheckTool));
    let (mut config, _) = config(
        vec![
            calls(vec![
                (
                    "1".into(),
                    "bash".into(),
                    r#"{"command":"git status"}"#.into(),
                ),
                (
                    "2".into(),
                    "bash".into(),
                    r#"{"command":"git status; rm -rf x"}"#.into(),
                ),
                (
                    "3".into(),
                    "bash".into(),
                    r#"{"command":"cat secrets"}"#.into(),
                ),
                (
                    "4".into(),
                    "write".into(),
                    r#"{"filePath":"src/core.ts","content":"x"}"#.into(),
                ),
                (
                    "5".into(),
                    "write".into(),
                    r#"{"filePath":"crawlers/../src/core.ts","content":"x"}"#.into(),
                ),
                (
                    "6".into(),
                    "write".into(),
                    r#"{"filePath":"notes/a.md","content":"x"}"#.into(),
                ),
            ]),
            answer("done"),
        ],
        tools,
        dir.path(),
    )
    .await;
    std::process::Command::new("git")
        .arg("init")
        .arg("-q")
        .current_dir(dir.path())
        .status()
        .unwrap();
    config.ctx = config.ctx.with_rules(Arc::new(mason::Rules {
        hooks: Default::default(),
        ..rules(Some(&["git status*"]), Some(&["crawlers/*", "notes/*"]))
    }));
    let outcome = engine::run(config).await.unwrap();
    // parallel calls commit as they finish: read each result by its call id
    let by_id = |id: &str| {
        outcome
            .messages
            .iter()
            .find(|message| message.tool_call_id.as_deref() == Some(id))
            .and_then(|message| message.content.clone())
            .unwrap_or_default()
    };
    let results: Vec<String> = ["1", "2", "3", "4", "5", "6"]
        .iter()
        .map(|id| by_id(id))
        .collect();
    assert!(!results[0].starts_with("Error"), "{}", results[0]);
    assert!(
        results[1].contains("without chaining or redirection"),
        "{}",
        results[1]
    );
    assert!(
        results[2].contains("this profile's shell allows only: git status*"),
        "{}",
        results[2]
    );
    assert!(
        results[3].contains("this profile may change only: crawlers/*, notes/*"),
        "{}",
        results[3]
    );
    assert!(results[4].contains("may change only"), "{}", results[4]);
    assert!(!results[5].starts_with("Error"), "{}", results[5]);
    assert!(!dir.path().join("src/core.ts").exists());
}

#[tokio::test]
async fn failing_checks_fail_the_run_once_repairs_are_spent() {
    let dir = tempfile::tempdir().unwrap();
    let mut tools = builtin_tools();
    tools.push(Box::new(CheckTool));
    let mut replies = vec![call(
        "write",
        r#"{"filePath":"crawlers/shop.ts","content":"bad"}"#,
    )];
    replies.extend((0..4).map(|_| answer("done")));
    let (mut config, model) = config(replies, tools, dir.path()).await;
    config.ctx = config.ctx.with_rules(Arc::new(rules(None, None)));
    let err = engine::run(config).await.unwrap_err();
    assert!(
        err.to_string().contains("before-finish checks still fail"),
        "{err}"
    );
    assert_eq!(
        model.request_count(),
        5,
        "the edit, then three repairs and a final refusal"
    );
}

#[tokio::test]
async fn the_final_turn_cannot_skip_failing_checks() {
    let dir = tempfile::tempdir().unwrap();
    let mut tools = builtin_tools();
    tools.push(Box::new(CheckTool));
    let (mut config, _) = config(
        vec![
            call(
                "write",
                r#"{"filePath":"crawlers/shop.ts","content":"bad"}"#,
            ),
            answer("out of steps"),
        ],
        tools,
        dir.path(),
    )
    .await;
    config.limits.max_steps = 1;
    config.ctx = config.ctx.with_rules(Arc::new(rules(None, None)));
    let err = engine::run(config).await.unwrap_err();
    assert!(
        err.to_string().contains("before-finish checks still fail"),
        "{err}"
    );
}

#[tokio::test]
async fn hooks_get_the_authorization_of_an_ordinary_call() {
    let dir = tempfile::tempdir().unwrap();
    let hook = mason::Hook {
        pattern: "notes/*".into(),
        tool: "write".into(),
        args: serde_json::from_value(serde_json::json!({"filePath": "secret/x", "content": "x"}))
            .unwrap(),
    };
    let (mut config, _) = config(
        vec![
            call("write", r#"{"filePath":"notes/a.md","content":"x"}"#),
            answer("done"),
        ],
        builtin_tools(),
        dir.path(),
    )
    .await;
    config.ctx = config.ctx.with_rules(Arc::new(mason::Rules {
        write_paths: Some(vec!["notes/*".into()]),
        hooks: mason::Hooks {
            after_edit: vec![hook],
            before_finish: vec![],
        },
        ..Default::default()
    }));
    let outcome = engine::run(config).await.unwrap();
    let results = tool_results(&outcome.messages);
    assert!(results[0].contains("[write on notes/a.md after this edit: FAILED]\npermission denied: this profile may change only: notes/*"), "{}", results[0]);
    assert!(!dir.path().join("secret/x").exists());
}

#[cfg(unix)]
#[tokio::test]
async fn write_paths_hold_for_where_a_symlink_points() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("allowed")).unwrap();
    std::fs::create_dir_all(dir.path().join("protected")).unwrap();
    std::fs::write(dir.path().join("protected/target.txt"), "keep").unwrap();
    std::os::unix::fs::symlink(
        dir.path().join("protected/target.txt"),
        dir.path().join("allowed/link.txt"),
    )
    .unwrap();
    let (mut config, _) = config(
        vec![
            call(
                "write",
                r#"{"filePath":"allowed/link.txt","content":"changed"}"#,
            ),
            answer("done"),
        ],
        builtin_tools(),
        dir.path(),
    )
    .await;
    config.ctx = config.ctx.with_rules(Arc::new(mason::Rules {
        write_paths: Some(vec!["allowed/*".into()]),
        ..Default::default()
    }));
    let outcome = engine::run(config).await.unwrap();
    assert!(tool_results(&outcome.messages)[0].contains("may change only: allowed/*"));
    assert_eq!(
        std::fs::read_to_string(dir.path().join("protected/target.txt")).unwrap(),
        "keep"
    );
}
#[cfg(unix)]
#[tokio::test]
async fn a_parent_step_after_a_symlink_cannot_leave_write_paths() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("allowed")).unwrap();
    std::fs::create_dir_all(dir.path().join("protected/sub")).unwrap();
    std::fs::write(dir.path().join("protected/target.txt"), "keep").unwrap();
    // allowed/link/.. is protected/, though it reads as allowed/
    std::os::unix::fs::symlink("../protected/sub", dir.path().join("allowed/link")).unwrap();
    let (mut config, _) = config(
        vec![
            call(
                "write",
                r#"{"filePath":"allowed/link/../target.txt","content":"changed"}"#,
            ),
            answer("done"),
        ],
        builtin_tools(),
        dir.path(),
    )
    .await;
    config.ctx = config.ctx.with_rules(Arc::new(mason::Rules {
        write_paths: Some(vec!["allowed/*".into()]),
        ..Default::default()
    }));
    let outcome = engine::run(config).await.unwrap();
    assert!(tool_results(&outcome.messages)[0].contains("may change only: allowed/*"));
    assert_eq!(
        std::fs::read_to_string(dir.path().join("protected/target.txt")).unwrap(),
        "keep"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn hooks_check_the_file_a_symlink_alias_writes() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("crawlers")).unwrap();
    std::fs::create_dir_all(dir.path().join("notes")).unwrap();
    std::fs::write(dir.path().join("crawlers/shop.ts"), "").unwrap();
    std::os::unix::fs::symlink("../crawlers/shop.ts", dir.path().join("notes/link")).unwrap();
    let mut tools = builtin_tools();
    tools.push(Box::new(CheckTool));
    let (mut config, _) = config(
        vec![
            call("write", r#"{"filePath":"notes/link","content":"bad"}"#),
            call("write", r#"{"filePath":"notes/link","content":"ok"}"#),
            answer("done"),
        ],
        tools,
        dir.path(),
    )
    .await;
    config.ctx = config.ctx.with_rules(Arc::new(rules(None, None)));
    let outcome = engine::run(config).await.unwrap();
    let results = tool_results(&outcome.messages);
    assert!(
        results[0].contains("[check on crawlers/shop.ts after this edit: FAILED]"),
        "{}",
        results[0]
    );
    assert!(
        results[1].contains("[check on crawlers/shop.ts after this edit: passed]"),
        "{}",
        results[1]
    );
}

#[tokio::test]
async fn a_hanging_finish_check_ends_at_the_run_deadline() {
    let dir = tempfile::tempdir().unwrap();
    let mut tools = builtin_tools();
    tools.push(SleepTool::boxed("check", Duration::from_secs(600), "never"));
    let (mut config, _) = config(
        vec![
            call("write", r#"{"filePath":"crawlers/shop.ts","content":"x"}"#),
            answer("done"),
        ],
        tools,
        dir.path(),
    )
    .await;
    config.limits.timeout = Duration::from_secs(1);
    let hook = mason::Hook {
        pattern: "crawlers/*".into(),
        tool: "check".into(),
        args: Default::default(),
    };
    config.ctx = config.ctx.with_rules(Arc::new(mason::Rules {
        hooks: mason::Hooks {
            after_edit: vec![],
            before_finish: vec![hook],
        },
        ..Default::default()
    }));
    let started = std::time::Instant::now();
    let err = tokio::time::timeout(Duration::from_secs(20), engine::run(config))
        .await
        .expect("the deadline ends the hook")
        .unwrap_err();
    assert!(matches!(err, Error::RunDeadline(_)), "{err}");
    // the deadline, then the cleanup grace a call that ignores cancellation gets
    assert!(
        started.elapsed() < Duration::from_secs(1) + engine::CLEANUP_GRACE + Duration::from_secs(2)
    );
}

#[tokio::test]
async fn hooks_check_a_file_written_through_a_parent_step() {
    let dir = tempfile::tempdir().unwrap();
    let mut tools = builtin_tools();
    tools.push(Box::new(CheckTool));
    let (mut config, _) = config(
        vec![
            call(
                "write",
                r#"{"filePath":"crawlers/../crawlers/shop.ts","content":"bad"}"#,
            ),
            call("write", r#"{"filePath":"crawlers/shop.ts","content":"ok"}"#),
            answer("done"),
        ],
        tools,
        dir.path(),
    )
    .await;
    config.ctx = config.ctx.with_rules(Arc::new(rules(None, None)));
    let outcome = engine::run(config).await.unwrap();
    let results = tool_results(&outcome.messages);
    assert!(
        results[0].contains("[check on crawlers/shop.ts after this edit: FAILED]"),
        "{}",
        results[0]
    );
}

/// A check that, interrupted, cleans up asynchronously before it ends.
struct CleaningCheck(Arc<std::sync::atomic::AtomicBool>);

#[async_trait::async_trait]
impl mason::Tool for CleaningCheck {
    fn spec(&self) -> mason::ToolSpec {
        mason::ToolSpec::function(
            "check",
            "checks slowly",
            serde_json::json!({"type": "object"}),
        )
    }
    async fn call(
        &self,
        _args: serde_json::Value,
        ctx: &mason::ToolCtx,
    ) -> std::result::Result<String, mason::ToolError> {
        ctx.token.cancelled().await;
        tokio::time::sleep(Duration::from_millis(100)).await;
        self.0.store(true, Ordering::SeqCst);
        Err(mason::ToolError::Failure("interrupted".into()))
    }
}

#[tokio::test]
async fn an_interrupted_hook_gets_the_cleanup_grace_of_a_call() {
    let dir = tempfile::tempdir().unwrap();
    let cleaned = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mut tools = builtin_tools();
    tools.push(Box::new(CleaningCheck(cleaned.clone())));
    let (mut config, _) = config(
        vec![
            call("write", r#"{"filePath":"crawlers/shop.ts","content":"x"}"#),
            answer("done"),
        ],
        tools,
        dir.path(),
    )
    .await;
    config.limits.timeout = Duration::from_secs(1);
    let hook = mason::Hook {
        pattern: "crawlers/*".into(),
        tool: "check".into(),
        args: Default::default(),
    };
    config.ctx = config.ctx.with_rules(Arc::new(mason::Rules {
        hooks: mason::Hooks {
            after_edit: vec![],
            before_finish: vec![hook],
        },
        ..Default::default()
    }));
    let err = engine::run(config).await.unwrap_err();
    assert!(matches!(err, Error::RunDeadline(_)), "{err}");
    assert!(
        cleaned.load(Ordering::SeqCst),
        "the hook finished its cleanup"
    );
}

#[tokio::test]
async fn hooks_run_outside_the_tool_call_budget() {
    let dir = tempfile::tempdir().unwrap();
    let mut tools = builtin_tools();
    tools.push(Box::new(CheckTool));
    let (mut config, _) = config(
        vec![
            call("write", r#"{"filePath":"crawlers/shop.ts","content":"ok"}"#),
            answer("done"),
        ],
        tools,
        dir.path(),
    )
    .await;
    config.limits.max_total_tool_calls = 1;
    config.ctx = config.ctx.with_rules(Arc::new(rules(None, None)));
    // the write took the one call; its after-edit and finish checks still run
    let outcome = engine::run(config).await.unwrap();
    assert_eq!(outcome.answer, "done");
    let edit_result = outcome.messages[2].content.as_deref().unwrap();
    assert!(
        edit_result.contains("after this edit: passed"),
        "{edit_result}"
    );
}

/// Approves `check` slowly, like a person at an approval prompt.
struct SlowApproval;

#[async_trait::async_trait]
impl mason::tool::policy::ToolPolicy for SlowApproval {
    async fn authorize(
        &self,
        name: &str,
        _args: &serde_json::Value,
        _ctx: &mason::ToolCtx,
    ) -> std::result::Result<(), mason::ToolError> {
        if name == "check" {
            tokio::time::sleep(Duration::from_secs(2)).await;
        }
        Ok(())
    }
}

#[tokio::test]
async fn approval_granted_after_cancellation_starts_no_hook() {
    let dir = tempfile::tempdir().unwrap();
    let (checker, calls) = EchoTool::boxed("check", "ok");
    let mut tools = builtin_tools();
    tools.push(checker);
    let (mut config, _) = config(
        vec![
            call("write", r#"{"filePath":"crawlers/shop.ts","content":"x"}"#),
            answer("done"),
        ],
        tools,
        dir.path(),
    )
    .await;
    config.limits.timeout = Duration::from_secs(1);
    let hook = mason::Hook {
        pattern: "crawlers/*".into(),
        tool: "check".into(),
        args: Default::default(),
    };
    config.ctx = config
        .ctx
        .with_execution(
            mason::tool::policy::ExecutionMode::Trusted,
            false,
            Arc::new(SlowApproval),
        )
        .with_rules(Arc::new(mason::Rules {
            hooks: mason::Hooks {
                after_edit: vec![],
                before_finish: vec![hook],
            },
            ..Default::default()
        }));
    let err = engine::run(config).await.unwrap_err();
    assert!(matches!(err, Error::RunDeadline(_)), "{err}");
    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "the approval came during the grace: no call"
    );
}

#[tokio::test]
async fn a_finish_check_takes_the_one_tool_slot_without_deadlock() {
    let dir = tempfile::tempdir().unwrap();
    let mut tools = builtin_tools();
    tools.push(Box::new(CheckTool));
    let (mut config, _) = config(
        vec![
            call("write", r#"{"filePath":"crawlers/shop.ts","content":"ok"}"#),
            answer("done"),
        ],
        tools,
        dir.path(),
    )
    .await;
    config.limits.max_concurrent_tools = 1;
    config.ctx = config.ctx.with_rules(Arc::new(rules(None, None)));
    let outcome = tokio::time::timeout(Duration::from_secs(10), engine::run(config)).await;
    assert_eq!(outcome.expect("no deadlock").unwrap().answer, "done");
}
