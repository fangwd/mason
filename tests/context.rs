//! Context pruning and compaction tests.

mod support;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use mason::context::{compact_conversation, in_memory_compactor, summarize};
use mason::{ChatDelta, ChatRequest, Message, PromptSource, Role, ToolCall, Usage, engine};
use support::{EchoTool, FunctionModel, Reply, ScriptedModel, answer, config, is_summary_request};
use tokio_util::sync::CancellationToken;

/// Creates a tool result estimated at `tokens` tokens.
fn result(id: &str, tokens: usize) -> Message {
    Message::tool(id, "x".repeat(tokens * 4))
}

fn call_message(id: &str) -> Message {
    Message {
        role: Role::Assistant,
        content: None,
        tool_calls: Some(vec![ToolCall::new(id, "bash", "{}")]),
        tool_call_id: None,
        reasoning_content: None,
    }
}

/// Seeds a system prompt followed by completed turns with tool results of about
/// `tokens` tokens each.
fn seeded(turns: usize, tokens: usize) -> Vec<Message> {
    let mut messages = vec![Message::system("sys")];
    for n in 1..=turns {
        messages.push(Message::user(format!("q{n}")));
        messages.push(call_message(&format!("c{n}")));
        messages.push(result(&format!("c{n}"), tokens));
        messages.push(Message {
            role: Role::Assistant,
            content: Some(format!("a{n}")),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        });
    }
    messages
}

#[tokio::test]
async fn compacts_before_an_oversized_request_keeping_the_recent_tail() {
    let dir = tempfile::tempdir().unwrap();
    let prompts = PromptSource::Embedded.load().unwrap();
    // Exercise a 2k request budget; long user history forces compaction after pruning.
    let summaries = Arc::new(AtomicUsize::new(0));
    let model = FunctionModel::with_context(
        {
            let summaries = summaries.clone();
            move |request: &ChatRequest| {
                if is_summary_request(request) {
                    answer(&format!(
                        "SUMMARY {}",
                        summaries.fetch_add(1, Ordering::SeqCst) + 1
                    ))
                } else {
                    answer("answer")
                }
            }
        },
        22_000,
    );

    let mut messages = seeded(4, 1_000);
    messages[1].content = Some(format!("q1 {}", "u".repeat(12_000)));
    messages.push(Message::user("next question"));
    let (mut run_config, _scripted) = config(vec![], vec![], dir.path()).await;
    run_config.model = model.clone();
    run_config.messages = messages;
    run_config.max_tokens = Some(2_000);
    run_config.compact = Some(in_memory_compactor(model.clone(), prompts.clone(), 2_000));

    let outcome = engine::run(run_config).await.unwrap();
    assert_eq!(outcome.answer, "answer");

    let requests = model.requests();
    assert!(requests.len() >= 2);
    // The summary calls came first and saw the seeded conversation, in chunks
    // that fit the budget, with long parts clipped.
    assert!(is_summary_request(&requests[0]));
    let summaries: Vec<&ChatRequest> = requests
        .iter()
        .take_while(|request| is_summary_request(request))
        .collect();
    for request in &summaries {
        assert!(mason::context::estimate_tokens(&request.messages) <= 2_000);
    }
    let summary_input: String = summaries
        .iter()
        .map(|request| request.messages[1].content.clone().unwrap())
        .collect();
    assert!(summary_input.contains("[User]: q1"), "{summary_input}");
    assert!(
        summary_input.contains("[Tool result]: xxx"),
        "{summary_input}"
    );
    assert!(
        summary_input.contains("\n[... truncated ...]\n"),
        "{summary_input}"
    );

    // The main request keeps the system prompt, gains the checkpoint with
    // the summary, and ends with the current user message verbatim.
    let request = &requests[requests.len() - 1];
    assert!(!is_summary_request(request));
    assert_eq!(request.messages[0], Message::system("sys"));
    let checkpoint = request.messages[1].content.clone().unwrap();
    assert!(
        checkpoint.contains("<conversation-checkpoint>"),
        "{checkpoint}"
    );
    assert!(
        checkpoint.contains(&format!("SUMMARY {}", summaries.len())),
        "{checkpoint}"
    );
    assert_eq!(
        request.messages.last().unwrap(),
        &Message::user("next question")
    );
    assert!(
        mason::context::estimate_tokens(&request.messages) <= 2_500,
        "the compacted request fits the usable budget (plus checkpoint slack)"
    );
}

#[tokio::test]
async fn compaction_summarizes_original_results_not_pruned_placeholders() {
    let dir = tempfile::tempdir().unwrap();
    let prompts = PromptSource::Embedded.load().unwrap();
    let model = FunctionModel::with_context(
        |request: &ChatRequest| {
            if is_summary_request(request) {
                answer("S")
            } else {
                answer("answer")
            }
        },
        22_000,
    );

    // 72k tokens of tool output: pruning clears the oldest results but cannot
    // reach the 2k budget, so the run compacts. The summary must see the
    // original results, not the pruned placeholders.
    let mut messages = seeded(6, 12_000);
    messages.push(Message::user("next"));
    let (mut run_config, _scripted) = config(vec![], vec![], dir.path()).await;
    run_config.model = model.clone();
    run_config.messages = messages;
    run_config.max_tokens = Some(2_000);
    run_config.compact = Some(in_memory_compactor(model.clone(), prompts, 2_000));

    let outcome = engine::run(run_config).await.unwrap();
    assert_eq!(outcome.answer, "answer");
    let requests = model.requests();
    let summary = requests
        .iter()
        .find(|request| is_summary_request(request))
        .expect("the run compacted");
    let input = summary.messages[1].content.clone().unwrap();
    assert!(input.contains("[User]: q1"), "{input}");
    assert!(input.contains("[Tool result]: xxx"), "{input}");
    assert!(!input.contains(mason::context::PRUNED), "{input}");
    let main = requests.last().unwrap();
    assert!(!is_summary_request(main));
    assert!(mason::context::estimate_tokens(&main.messages) <= 2_500);
}

#[tokio::test]
async fn the_next_compaction_merges_the_previous_summary() {
    let prompts = PromptSource::Embedded.load().unwrap();
    let summaries = Arc::new(AtomicUsize::new(0));
    let model = FunctionModel::new({
        let summaries = summaries.clone();
        move |request: &ChatRequest| {
            assert!(is_summary_request(request));
            answer(&format!(
                "SUMMARY {}",
                summaries.fetch_add(1, Ordering::SeqCst) + 1
            ))
        }
    });
    let token = CancellationToken::new();

    // First compaction of a conversation without a checkpoint; a tight budget
    // folds it in several chunks.
    let first = compact_conversation(&seeded(4, 1_000), 2_000, &*model, &prompts, &token)
        .await
        .unwrap()
        .expect("there is a head to summarize");
    let first_summary = format!("SUMMARY {}", summaries.load(Ordering::SeqCst));
    assert!(
        first[1]
            .content
            .as_deref()
            .is_some_and(|c| c.contains(&first_summary))
    );
    let first_count = model.requests().len();

    // Grow past the tail again; the second compaction merges the first.
    let mut grown = first;
    grown.push(call_message("c9"));
    grown.push(result("c9", 1_500));
    let second = compact_conversation(&grown, 2_000, &*model, &prompts, &token)
        .await
        .unwrap()
        .expect("the grown head is summarizable");
    let requests = model.requests();
    let input = requests[first_count].messages[1].content.clone().unwrap();
    assert!(
        input.contains(&format!(
            "<prior-summary>\n{first_summary}\n</prior-summary>"
        )),
        "{input}"
    );
    let last_summary = format!("SUMMARY {}", requests.len());
    assert!(
        second[1]
            .content
            .as_deref()
            .is_some_and(|c| c.contains(&last_summary))
    );
}

#[tokio::test]
async fn the_current_request_survives_compaction_verbatim() {
    let dir = tempfile::tempdir().unwrap();
    let prompts = PromptSource::Embedded.load().unwrap();
    // Every "big" call adds ~3k tokens, exceeding the 2k request budget.
    let (big, _executed) = EchoTool::boxed("big", &"y".repeat(12_000));
    let mains = Arc::new(AtomicUsize::new(0));
    let model = FunctionModel::with_context(
        {
            let mains = mains.clone();
            move |request: &ChatRequest| {
                if is_summary_request(request) {
                    return answer("S");
                }
                let n = mains.fetch_add(1, Ordering::SeqCst) + 1;
                if n >= 5 {
                    return answer("done");
                }
                Reply::Calls(vec![(format!("t{n}"), "big".into(), "{}".into())])
            }
        },
        22_000,
    );

    let mut messages = seeded(1, 100);
    messages.push(Message::user("the actual request"));
    let (mut run_config, _scripted) = config(vec![], vec![big], dir.path()).await;
    run_config.model = model.clone();
    run_config.messages = messages;
    run_config.max_tokens = Some(2_000);
    run_config.compact = Some(in_memory_compactor(model.clone(), prompts, 2_000));

    let outcome = engine::run(run_config).await.unwrap();
    assert_eq!(outcome.answer, "done");

    let requests = model.requests();
    let compacted: Vec<&ChatRequest> = requests
        .iter()
        .filter(|request| {
            !is_summary_request(request)
                && request
                    .messages
                    .get(1)
                    .and_then(|m| m.content.as_deref())
                    .is_some_and(|c| c.contains("conversation-checkpoint"))
        })
        .collect();
    assert!(!compacted.is_empty(), "compaction happened during the run");
    for request in compacted {
        assert_eq!(request.messages[2], Message::user("the actual request"));
    }
    for request in requests.iter().filter(|r| is_summary_request(r)) {
        let input = request.messages[1].content.clone().unwrap();
        assert!(!input.contains("the actual request"), "{input}");
    }
}

#[tokio::test]
async fn compacts_and_retries_once_when_the_provider_reports_an_overflow() {
    let dir = tempfile::tempdir().unwrap();
    let prompts = PromptSource::Embedded.load().unwrap();
    let first_main = Arc::new(AtomicUsize::new(0));
    let model = FunctionModel::with_context(
        {
            let first_main = first_main.clone();
            move |request: &ChatRequest| {
                if is_summary_request(request) {
                    return answer("S");
                }
                if first_main.fetch_add(1, Ordering::SeqCst) == 0 {
                    return Reply::Error(
                        "This model's maximum context length is 8192 tokens".to_owned(),
                    );
                }
                answer("recovered")
            }
        },
        1_000_000,
    );

    // Larger than the 15k-token tail kept verbatim, so there is something
    // to summarize.
    let mut messages = seeded(3, 6_000);
    messages.push(Message::user("go"));
    let (mut run_config, _scripted) = config(vec![], vec![], dir.path()).await;
    run_config.model = model.clone();
    run_config.messages = messages;
    run_config.max_tokens = Some(mason::context::usable_tokens(1_000_000).unwrap());
    run_config.compact = Some(in_memory_compactor(
        model.clone(),
        prompts,
        mason::context::usable_tokens(1_000_000).unwrap(),
    ));

    let outcome = engine::run(run_config).await.unwrap();
    assert_eq!(outcome.answer, "recovered");
    let requests = model.requests();
    let summaries = requests.iter().filter(|r| is_summary_request(r)).count();
    assert_eq!(summaries, 1, "one compaction for the single overflow");
    assert_eq!(requests.len(), 3, "failed main + summary + retried main");
}

#[tokio::test]
async fn an_uncompactable_run_fails_with_the_provider_error() {
    let dir = tempfile::tempdir().unwrap();
    let prompts = PromptSource::Embedded.load().unwrap();
    let model = FunctionModel::with_context(
        |_request: &ChatRequest| Reply::Error("prompt is too long".to_owned()),
        1_000_000,
    );
    let (mut run_config, _scripted) = config(vec![], vec![], dir.path()).await;
    run_config.model = model.clone();
    run_config.messages = vec![Message::user("hi")];
    run_config.max_tokens = Some(980_000);
    run_config.compact = Some(in_memory_compactor(model.clone(), prompts, 980_000));

    let err = engine::run(run_config).await.unwrap_err();
    assert!(err.to_string().contains("prompt is too long"), "{err}");
    assert_eq!(model.request_count(), 1, "nothing to compact, so no retry");
}

#[tokio::test]
async fn usage_is_recorded_and_estimates_still_drive_compaction() {
    let dir = tempfile::tempdir().unwrap();
    let prompts = PromptSource::Embedded.load().unwrap();
    let (big, _executed) = EchoTool::boxed("big", &"y".repeat(4_000));
    // Exercise a deliberately tight 500-token request budget. History triggers a
    // compaction before the first main call; the turn then reports a tiny
    // usage, yet the next step compacts again on the estimate.
    let main_turns = Arc::new(AtomicUsize::new(0));
    let model = FunctionModel::with_context(
        {
            let main_turns = main_turns.clone();
            move |request: &ChatRequest| {
                if is_summary_request(request) {
                    return answer("S");
                }
                if main_turns.fetch_add(1, Ordering::SeqCst) == 0 {
                    Reply::Deltas(vec![
                        ChatDelta::Text("partial".into()),
                        ChatDelta::Usage {
                            prompt_tokens: 1,
                            completion_tokens: 1,
                        },
                        ChatDelta::ToolCall {
                            index: 0,
                            id: Some("b1".into()),
                            name: Some("big".into()),
                            arguments: "{}".into(),
                        },
                    ])
                } else {
                    answer("done")
                }
            }
        },
        20_500,
    );
    let (mut run_config, _handle) = config(vec![], vec![big], dir.path()).await;
    let mut messages = seeded(2, 1_500);
    messages.push(Message::user("go"));
    run_config.model = model.clone();
    run_config.messages = messages;
    run_config.max_tokens = Some(500);
    run_config.compact = Some(in_memory_compactor(model.clone(), prompts, 500));

    let outcome = engine::run(run_config).await.unwrap();
    assert_eq!(outcome.answer, "done");
    assert_eq!(
        outcome.usage,
        Some(Usage {
            prompt_tokens: 1,
            completion_tokens: 1
        })
    );

    let kinds: Vec<bool> = model.requests().iter().map(is_summary_request).collect();
    let mains: Vec<usize> = (0..kinds.len()).filter(|&i| !kinds[i]).collect();
    assert_eq!(mains.len(), 2, "{kinds:?}");
    assert!(
        mains[0] > 0,
        "estimate-triggered compaction first: {kinds:?}"
    );
    assert!(
        mains[1] > mains[0] + 1,
        "the tiny reported usage did not replace the estimate: step 2 compacted again: {kinds:?}"
    );
}

#[tokio::test]
async fn an_empty_summary_fails() {
    let prompts = PromptSource::Embedded.load().unwrap();
    let model = ScriptedModel::new(vec![answer("   ")]);
    let err = summarize(
        &*model,
        &prompts,
        &[Message::user("q1"), call_message("c1"), result("c1", 10)],
        None,
        100_000,
        &CancellationToken::new(),
    )
    .await
    .unwrap_err();
    assert!(err.to_string().contains("summary came back empty"), "{err}");
}

#[tokio::test]
async fn history_larger_than_the_context_is_summarized_in_chunks() {
    let prompts = PromptSource::Embedded.load().unwrap();
    let model = ScriptedModel::new((1..=10).map(|n| answer(&format!("summary {n}"))).collect());
    // Four 2k-token messages and a 30k-token one against a 6k-token context.
    let mut history: Vec<Message> = (0..4)
        .map(|n| Message::user(format!("{n}{}", "x".repeat(8_000))))
        .collect();
    history.push(Message::user("y".repeat(120_000)));
    let usable = 6_000;
    let summary = summarize(
        &*model,
        &prompts,
        &history,
        None,
        usable,
        &CancellationToken::new(),
    )
    .await
    .unwrap();
    let requests = model.requests();
    assert!(requests.len() > 2, "{}", requests.len());
    assert_eq!(summary, format!("summary {}", requests.len()));
    for (index, request) in requests.iter().enumerate() {
        assert!(mason::context::estimate_tokens(&request.messages) <= usable);
        if index > 0 {
            let prompt = request.messages[1].content.as_deref().unwrap();
            assert!(
                prompt.contains(&format!("summary {index}")),
                "each chunk extends the summary"
            );
        }
    }
}
