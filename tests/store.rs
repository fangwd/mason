//! SQLite transactions, event publication, and recovery tests.

use std::sync::Arc;
use std::time::Duration;

use mason::store::{Delivery, NewInput, NewSession, SessionQuery, SessionStore, SqliteStore};
use mason::{PromptSet, Role, ToolCall};
use serde_json::json;

async fn store() -> (tempfile::TempDir, SqliteStore) {
    let dir = tempfile::tempdir().unwrap();
    let store = SqliteStore::open(dir.path().join("mason.db"))
        .await
        .unwrap();
    (dir, store)
}

fn prompts() -> Arc<PromptSet> {
    PromptSet::parse(mason::prompt::EMBEDDED_PROMPTS)
        .unwrap()
        .into()
}

fn tool_call_message(id: &str) -> serde_json::Value {
    json!({
        "role": "assistant",
        "content": null,
        "tool_calls": [{"id": id, "type": "function", "function": {"name": "bash", "arguments": "{}"}}]
    })
}

#[tokio::test]
async fn a_repeated_key_is_accepted_once() {
    let (_dir, store) = store().await;
    let id = store.create_session(NewSession::default()).await.unwrap();
    let input = NewInput {
        content: "url 7 failed".into(),
        source: "jscrawl".into(),
        key: Some("fail-7".into()),
        delivery: Delivery::Steer,
    };
    let first = store.add_input(id, input.clone()).await.unwrap();
    let again = store.add_input(id, input).await.unwrap();
    assert_eq!(again, first, "the repeated key returns the earlier input");

    // Key-less inputs are never deduplicated.
    let a = store.add_input(id, NewInput::queued("one")).await.unwrap();
    let b = store.add_input(id, NewInput::queued("one")).await.unwrap();
    assert_ne!(a, b);

    let claim = store.start_run(id).await.unwrap();
    let events = store.list_events(id, 0).await.unwrap();
    let users: Vec<&serde_json::Value> = events
        .iter()
        .filter(|event| {
            event.data.pointer("/message/role").and_then(|v| v.as_str()) == Some("user")
        })
        .map(|event| event.data.get("source").unwrap())
        .collect();
    // Only the oldest queued input joins the run; the steer is claimed too.
    assert_eq!(claim.inputs.len(), 2);
    assert!(
        users
            .iter()
            .any(|source| source.as_str() == Some("jscrawl"))
    );
}

#[tokio::test]
async fn history_answers_tool_calls_whose_results_were_never_saved() {
    let (_dir, store) = store().await;
    let id = store.create_session(NewSession::default()).await.unwrap();
    store
        .append(
            id,
            "message",
            json!({"message": {"role": "user", "content": "go"}}),
        )
        .await
        .unwrap();
    store
        .append(id, "message", json!({"message": tool_call_message("c1")}))
        .await
        .unwrap();

    let history = store.history(id, &prompts()).await.unwrap();
    assert_eq!(history.len(), 3);
    assert_eq!(history[0], mason::Message::user("go"));
    assert_eq!(history[1].role, Role::Assistant);
    assert_eq!(
        history[1].tool_calls,
        Some(vec![ToolCall::new("c1", "bash", "{}")])
    );
    assert_eq!(
        history[2],
        mason::Message::tool("c1", "[Tool execution was interrupted]")
    );
}

#[tokio::test]
async fn a_failed_run_started_rolls_back_consumption_and_user_messages() {
    let (_dir, store) = store().await;
    let id = store.create_session(NewSession::default()).await.unwrap();
    store
        .add_input(
            id,
            NewInput {
                content: "steer".into(),
                delivery: Delivery::Steer,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    store
        .add_input(id, NewInput::queued("queue"))
        .await
        .unwrap();

    store.set_failure_hook(|event_type| event_type == "run.started");
    let err = store.start_run(id).await.unwrap_err();
    assert!(err.to_string().contains("injected failure"), "{err}");
    assert!(
        store.list_events(id, 0).await.unwrap().is_empty(),
        "nothing committed"
    );

    // The inputs were not consumed: the next start_run claims both.
    store.clear_failure_hook();
    let claim = store.start_run(id).await.unwrap();
    let contents: Vec<&str> = claim
        .inputs
        .iter()
        .map(|input| input.content.as_str())
        .collect();
    assert_eq!(contents, ["steer", "queue"]);
    let run_id = claim.run_id.expect("a claimed run has an id");
    let events = store.list_events(id, 0).await.unwrap();
    let types: Vec<&str> = events
        .iter()
        .map(|event| event.event_type.as_str())
        .collect();
    assert_eq!(types, ["message", "message", "run.started"]);

    // Terminal events are unique per run and link back by run id.
    store
        .append_run_event(id, &run_id, "run.finished", json!({"answer": "ok"}))
        .await
        .unwrap();
    let err = store
        .append_run_event(id, &run_id, "run.failed", json!({"error": "late"}))
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("already has its terminal event"),
        "{err}"
    );
}

#[tokio::test]
async fn dead_or_lagging_viewers_do_not_fail_appends() {
    let (_dir, store) = store().await;
    let id = store.create_session(NewSession::default()).await.unwrap();
    let dropped = store.subscribe(id);
    drop(dropped); // a viewer that went away
    let stalled = store.subscribe(id); // a viewer that never reads
    let mut live = store.subscribe(id);

    store
        .append(
            id,
            "message",
            json!({"message": {"role": "assistant", "content": "one"}}),
        )
        .await
        .unwrap();
    store
        .append(
            id,
            "message",
            json!({"message": {"role": "assistant", "content": "two"}}),
        )
        .await
        .unwrap();

    let mut received = Vec::new();
    while let Ok(event) = live.try_recv() {
        received.push(
            event
                .data
                .pointer("/message/content")
                .and_then(|v| v.as_str())
                .unwrap()
                .to_owned(),
        );
    }
    assert_eq!(received, ["one", "two"], "other viewers are unaffected");
    assert_eq!(store.list_events(id, 0).await.unwrap().len(), 2);
    drop(stalled);
}

#[tokio::test]
async fn list_sessions_orders_by_latest_activity() {
    let (_dir, store) = store().await;
    let a = store
        .create_session(NewSession {
            title: Some("first".into()),
            ..Default::default()
        })
        .await
        .unwrap();
    store
        .append(
            a,
            "message",
            json!({"message": {"role": "assistant", "content": "hi"}}),
        )
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(5)).await;
    let b = store
        .create_session(NewSession {
            title: Some("second".into()),
            agent: Some("triage".into()),
            ..Default::default()
        })
        .await
        .unwrap();
    store
        .create_session(NewSession {
            title: Some("child".into()),
            parent_id: Some(a),
            agent: Some("explore".into()),
            ..Default::default()
        })
        .await
        .unwrap();

    let all = store.list_sessions(&SessionQuery::default()).await.unwrap();
    let rows: Vec<(u64, Option<&str>, &str)> = all
        .iter()
        .map(|session| (session.id, session.title.as_deref(), session.agent.as_str()))
        .collect();
    assert_eq!(
        rows,
        vec![(b, Some("second"), "triage"), (a, Some("first"), "build")],
        "children are not listed"
    );
    assert_eq!(all[0].last_event_at, None);
    assert_eq!(all[1].last_event_type.as_deref(), Some("message"));

    let triage = store
        .list_sessions(&SessionQuery {
            agent: Some("triage".into()),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(
        triage.iter().map(|session| session.id).collect::<Vec<_>>(),
        [b]
    );

    let epoch = chrono::DateTime::UNIX_EPOCH;
    let none = store
        .list_sessions(&SessionQuery {
            before: Some(epoch),
            ..Default::default()
        })
        .await
        .unwrap();
    assert!(none.is_empty());

    let limited = store
        .list_sessions(&SessionQuery {
            limit: Some(1),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(
        limited.iter().map(|session| session.id).collect::<Vec<_>>(),
        [b]
    );
}

#[tokio::test]
async fn concurrent_start_runs_claim_inputs_exactly_once() {
    let (_dir, store) = store().await;
    let store = Arc::new(store);
    let id = store.create_session(NewSession::default()).await.unwrap();
    store
        .add_input(id, NewInput::queued("the only queued one"))
        .await
        .unwrap();
    for n in 0..3 {
        store
            .add_input(
                id,
                NewInput {
                    content: format!("steer {n}"),
                    delivery: Delivery::Steer,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
    }

    let mut join = tokio::task::JoinSet::new();
    for _ in 0..4 {
        let store = store.clone();
        join.spawn(async move { store.start_run(id).await });
    }
    let mut claims = Vec::new();
    while let Some(outcome) = join.join_next().await {
        claims.push(outcome.unwrap().unwrap());
    }
    let claimed: Vec<String> = claims
        .iter()
        .flat_map(|claim| claim.inputs.iter().map(|input| input.content.clone()))
        .collect();
    assert_eq!(
        claimed.len(),
        4,
        "every input is claimed exactly once across the races"
    );
    assert_eq!(
        claims.iter().filter(|claim| claim.run_id.is_some()).count(),
        1,
        "one run started"
    );

    // Nothing is left pending.
    assert!(store.pending_sessions().await.unwrap().is_empty());
    assert!(store.start_run(id).await.unwrap().inputs.is_empty());
}

#[tokio::test]
async fn failed_transactions_publish_nothing() {
    let (_dir, store) = store().await;
    let id = store.create_session(NewSession::default()).await.unwrap();
    let mut events = store.subscribe(id);
    store.add_input(id, NewInput::queued("go")).await.unwrap();
    store.set_failure_hook(|event_type| event_type == "message");
    let err = store.start_run(id).await.unwrap_err();
    assert!(err.to_string().contains("injected failure"), "{err}");
    store.clear_failure_hook();
    assert!(
        events.try_recv().is_err(),
        "the rolled-back transaction published nothing"
    );
}

#[tokio::test]
async fn concurrent_appends_publish_in_cursor_order() {
    let (_dir, store) = store().await;
    let id = store.create_session(NewSession::default()).await.unwrap();
    let mut events = store.subscribe(id);
    let store = Arc::new(store);

    let mut join = tokio::task::JoinSet::new();
    for n in 0..20 {
        let store = store.clone();
        join.spawn(async move {
            store
                .append(
                    id,
                    "message",
                    json!({"message": {"role": "user", "content": n.to_string()}}),
                )
                .await
        });
    }
    while join.join_next().await.is_some() {}

    let mut seen: Vec<u64> = Vec::new();
    while let Ok(event) = events.try_recv() {
        seen.push(event.id);
    }
    assert_eq!(seen.len(), 20);
    let mut sorted = seen.clone();
    sorted.sort_unstable();
    assert_eq!(seen, sorted, "subscribers see strictly ascending cursors");

    let listed: Vec<u64> = store
        .list_events(id, 0)
        .await
        .unwrap()
        .iter()
        .map(|event| event.id)
        .collect();
    assert_eq!(listed, sorted, "the log matches the publication order");
    // Cursor replay works from the middle.
    let middle = sorted[9];
    let tail = store.list_events(id, middle).await.unwrap();
    assert_eq!(tail.len(), 10);
    assert!(tail.iter().all(|event| event.id > middle));
}

#[tokio::test]
async fn recovery_marks_abandoned_runs_interrupted_once() {
    let (_dir, store) = store().await;

    // An abandoned run: claimed input, an assistant tool call, no results,
    // no terminal event.
    let abandoned = store.create_session(NewSession::default()).await.unwrap();
    store
        .add_input(abandoned, NewInput::queued("do the work"))
        .await
        .unwrap();
    let claim = store.start_run(abandoned).await.unwrap();
    let run_id = claim.run_id.clone().unwrap();
    store
        .append(
            abandoned,
            "message",
            json!({"message": tool_call_message("c1")}),
        )
        .await
        .unwrap();

    // A finished run, untouched by recovery.
    let finished = store.create_session(NewSession::default()).await.unwrap();
    store
        .add_input(finished, NewInput::queued("done deal"))
        .await
        .unwrap();
    let done = store.start_run(finished).await.unwrap();
    store
        .append_run_event(
            finished,
            done.run_id.as_deref().unwrap(),
            "run.finished",
            json!({"answer": "ok"}),
        )
        .await
        .unwrap();

    let recovery = store.recover().await.unwrap();
    assert_eq!(recovery.interrupted, vec![(abandoned, run_id.clone())]);

    // Idempotent: the second pass repairs nothing.
    assert_eq!(store.recover().await.unwrap().interrupted, vec![]);

    let events = store.list_events(abandoned, 0).await.unwrap();
    let terminal: Vec<&str> = events
        .iter()
        .filter(|event| event.event_type.starts_with("run.") && event.event_type != "run.started")
        .map(|event| event.event_type.as_str())
        .collect();
    assert_eq!(terminal, ["run.interrupted"], "exactly one terminal event");
    assert!(
        events
            .iter()
            .any(|event| event.run_id.as_deref() == Some(run_id.as_str()))
    );

    // The consumed input stays consumed: nothing replays.
    assert!(!store.pending_sessions().await.unwrap().contains(&abandoned));
    assert!(store.start_run(abandoned).await.unwrap().inputs.is_empty());

    // History repairs the unanswered call.
    let history = store.history(abandoned, &prompts()).await.unwrap();
    assert_eq!(history.last().unwrap().role, Role::Tool);
    assert_eq!(
        history.last().unwrap().content.as_deref(),
        Some("[Tool execution was interrupted]")
    );
}

#[tokio::test]
async fn conversation_honors_compaction_and_keeps_the_newest_user_message() {
    let (_dir, store) = store().await;
    let id = store.create_session(NewSession::default()).await.unwrap();

    let mut ids = Vec::new();
    for content in ["old question", "old answer"] {
        let role = if content.ends_with("question") {
            "user"
        } else {
            "assistant"
        };
        let event = store
            .append(
                id,
                "message",
                json!({"message": {"role": role, "content": content}}),
            )
            .await
            .unwrap();
        ids.push(event.id);
    }
    // The current request lands before the compaction, like a long run.
    let current = store
        .append(
            id,
            "message",
            json!({"message": {"role": "user", "content": "the actual request"}}),
        )
        .await
        .unwrap();
    let kept = store
        .append(
            id,
            "message",
            json!({"message": {"role": "assistant", "content": "working"}}),
        )
        .await
        .unwrap();
    store
        .append(
            id,
            "compaction",
            json!({"summary": "S1", "tailFrom": kept.id}),
        )
        .await
        .unwrap();
    let after = store
        .append(
            id,
            "message",
            json!({"message": {"role": "assistant", "content": "done"}}),
        )
        .await
        .unwrap();

    let conversation = store.conversation(id).await.unwrap();
    assert_eq!(conversation.summary.as_deref(), Some("S1"));
    let kept_ids: Vec<u64> = conversation.events.iter().map(|event| event.id).collect();
    assert_eq!(
        kept_ids,
        vec![current.id, kept.id, after.id],
        "the tail plus the newest user message, in cursor order"
    );
    assert!(!kept_ids.contains(&ids[0]) && !kept_ids.contains(&ids[1]));

    // The history prepends the checkpoint.
    let history = store.history(id, &prompts()).await.unwrap();
    assert_eq!(history[0].role, Role::User);
    assert!(
        history[0]
            .content
            .as_deref()
            .unwrap()
            .contains("<conversation-checkpoint>")
    );
    assert!(history[0].content.as_deref().unwrap().contains("S1"));
    assert_eq!(history.len(), 4);
}

#[tokio::test]
async fn a_cursor_past_every_id_returns_nothing() {
    let (_dir, store) = store().await;
    let session = store.create_session(NewSession::default()).await.unwrap();
    store
        .append(
            session,
            "message",
            json!({"message": {"role": "assistant", "content": "hi"}}),
        )
        .await
        .unwrap();
    // A cursor above i64::MAX used to bind negative and replay everything.
    assert!(
        store
            .events_page(session, u64::MAX, 10)
            .await
            .unwrap()
            .is_empty()
    );
    assert!(
        store
            .list_events(session, u64::MAX)
            .await
            .unwrap()
            .is_empty()
    );
    assert_eq!(store.events_page(session, 0, 10).await.unwrap().len(), 1);
}

#[tokio::test]
async fn session_metadata_round_trips_and_the_inbox_is_readable() {
    let (_dir, store) = store().await;
    let with_metadata = store
        .create_session(NewSession {
            title: Some("shop triage".into()),
            metadata: json!({"site_id": 7, "labels": ["unattended"]}),
            ..Default::default()
        })
        .await
        .unwrap();
    let plain = store.create_session(NewSession::default()).await.unwrap();

    let session = store.get_session(with_metadata).await.unwrap().unwrap();
    assert_eq!(
        session.metadata,
        json!({"site_id": 7, "labels": ["unattended"]}),
        "the store returns the metadata it was given, unread"
    );
    assert_eq!(
        store.get_session(plain).await.unwrap().unwrap().metadata,
        serde_json::Value::Null,
        "a session without metadata reports null"
    );

    let rows = store.list_sessions(&SessionQuery::default()).await.unwrap();
    let listed = rows.iter().find(|row| row.id == with_metadata).unwrap();
    assert_eq!(
        listed.metadata,
        json!({"site_id": 7, "labels": ["unattended"]})
    );
    assert_eq!(
        rows.iter().find(|row| row.id == plain).unwrap().metadata,
        serde_json::Value::Null
    );

    // The inbox lists keys, sources and consumption state in insertion order.
    store
        .add_input(
            with_metadata,
            NewInput {
                content: "issue 1".into(),
                source: "jscrawl".into(),
                key: Some("issue-1".into()),
                delivery: Delivery::Queue,
            },
        )
        .await
        .unwrap();
    store
        .add_input(
            with_metadata,
            NewInput {
                content: "and another thing".into(),
                delivery: Delivery::Steer,
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let inputs = store.list_inputs(with_metadata).await.unwrap();
    assert_eq!(inputs.len(), 2);
    assert_eq!(inputs[0].key.as_deref(), Some("issue-1"));
    assert_eq!(inputs[0].source, "jscrawl");
    assert_eq!(inputs[0].delivery, Delivery::Queue);
    assert_eq!(inputs[0].content, "issue 1");
    assert_eq!(
        inputs[0].consumed_at, None,
        "an unclaimed input stays pending"
    );
    assert_eq!(inputs[1].key, None);
    assert_eq!(inputs[1].source, "human");
    assert_eq!(inputs[1].delivery, Delivery::Steer);

    store.start_run(with_metadata).await.unwrap();
    let consumed = store.list_inputs(with_metadata).await.unwrap();
    assert!(
        consumed
            .iter()
            .all(|input| input.consumed_at.is_some_and(|at| at >= input.created_at)),
        "a claimed run marks every input it took as consumed"
    );
    assert_eq!(store.list_inputs(plain).await.unwrap().len(), 0);
}

#[tokio::test]
async fn run_events_returns_only_that_runs_events() {
    let (_dir, store) = store().await;
    let id = store.create_session(NewSession::default()).await.unwrap();
    let other = store.create_session(NewSession::default()).await.unwrap();
    store.add_input(id, NewInput::queued("one")).await.unwrap();
    let first = store.start_run(id).await.unwrap().run_id.unwrap();
    store
        .append_run_event(
            id,
            &first,
            "message",
            json!({"message": {"role": "assistant", "content": "a"}}),
        )
        .await
        .unwrap();
    store.append(id, "note", json!({})).await.unwrap();
    store.add_input(id, NewInput::queued("two")).await.unwrap();
    let second = store.start_run(id).await.unwrap().run_id.unwrap();
    store
        .append_run_event(other, &first, "message", json!({}))
        .await
        .unwrap();

    let events = store.run_events(id, &first).await.unwrap();
    let types: Vec<&str> = events.iter().map(|e| e.event_type.as_str()).collect();
    assert_eq!(types, ["message", "run.started", "message"]);
    assert!(events.windows(2).all(|pair| pair[0].id < pair[1].id));
    assert!(
        events
            .iter()
            .all(|e| e.session_id == id && e.run_id.as_deref() == Some(first.as_str()))
    );
    assert_eq!(store.run_events(id, &second).await.unwrap().len(), 2);
}
