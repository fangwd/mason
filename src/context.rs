//! Context pruning and summary-based compaction. Request estimates, not prior provider
//! usage, drive context limits.

use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use crate::Error;
use crate::engine::CompactFn;
use crate::error::Result;
use crate::llm::{ChatModel, ChatRequest, ChatTurn, Message, Role};
use crate::prompt::PromptSet;

/// Room a reply may need: requests are planned against
/// `context capacity - OUTPUT_RESERVE`.
pub const OUTPUT_RESERVE: usize = 20_000;
/// Tool outputs newer than this many tokens are never pruned.
pub const PRUNE_PROTECT: usize = 40_000;
/// Pruning only happens when it frees more than this many tokens.
pub const PRUNE_MINIMUM: usize = 20_000;
/// The placeholder an old tool output is replaced with.
pub const PRUNED: &str = "[Old tool result content cleared]";
/// Tool results and tool-call arguments inside a summarization request are
/// clipped to this many UTF-8 bytes.
pub const SUMMARY_CLIP_BYTES: usize = 2_000;

/// Estimates tokens from content, reasoning, and serialized tool calls at four
/// UTF-8 bytes per token.
pub fn estimate_tokens(messages: &[Message]) -> usize {
    let chars: usize = messages.iter().map(estimate_message).sum();
    (chars as f64 / 4.0).round() as usize
}

fn estimate_message(message: &Message) -> usize {
    let content = message.content.as_ref().map_or(0, |content| content.len());
    let reasoning = message
        .reasoning_content
        .as_ref()
        .map_or(0, |text| text.len());
    let calls = message
        .tool_calls
        .as_ref()
        .and_then(|calls| serde_json::to_string(calls).ok())
        .map_or(0, |json| json.len());
    content + reasoning + calls
}

/// The same estimator for a serialized payload (tool specs, prompt text).
pub fn estimate_str(text: &str) -> usize {
    (text.len() as f64 / 4.0).round() as usize
}

/// Returns context capacity minus the reply reserve, rejecting insufficient capacity.
pub fn usable_tokens(context_tokens: usize) -> Result<usize> {
    if context_tokens < 1_024 {
        return Err(Error::Config(
            "context capacity must be at least 1024 tokens".into(),
        ));
    }
    Ok(context_tokens - output_reserve(context_tokens))
}

/// Scale the reply reserve for smaller local models, capped at 20k.
pub fn output_reserve(context_tokens: usize) -> usize {
    (context_tokens / 4).min(OUTPUT_RESERVE)
}

/// Tokens of recent conversation kept verbatim when compacting.
pub fn tail_budget(usable: usize) -> usize {
    (usable / 4).clamp(128, 15_000)
}

/// Prunes tool outputs older than the newest [`PRUNE_PROTECT`] tokens of tool output,
/// never the results the model has not seen yet (those after the latest assistant
/// message). Returns unchanged history unless more than [`PRUNE_MINIMUM`] tokens are
/// freed.
pub fn prune(messages: &[Message]) -> Vec<Message> {
    prune_with_budget(messages, usize::MAX)
}

/// Scale pruning thresholds to the request budget.
pub fn prune_with_budget(messages: &[Message], budget: usize) -> Vec<Message> {
    let protect = PRUNE_PROTECT.min(budget / 3);
    let minimum = PRUNE_MINIMUM.min(budget / 6);
    let unconsumed_from = messages
        .iter()
        .rposition(|message| message.role == Role::Assistant)
        .map_or(0, |index| index + 1);
    let mut cleared = Vec::new();
    let mut total = 0usize;
    let mut freed = 0usize;
    for index in (0..messages.len()).rev() {
        let message = &messages[index];
        if message.role != Role::Tool || message.content.as_deref() == Some(PRUNED) {
            continue;
        }
        let tokens = estimate_tokens(&messages[index..index + 1]);
        let newest = total == 0;
        total += tokens;
        if !newest && total > protect && index < unconsumed_from {
            cleared.push(index);
            freed += tokens;
        }
    }
    if freed <= minimum {
        return messages.to_vec();
    }
    messages
        .iter()
        .enumerate()
        .map(|(index, message)| {
            if cleared.contains(&index) {
                Message {
                    content: Some(PRUNED.to_owned()),
                    ..message.clone()
                }
            } else {
                message.clone()
            }
        })
        .collect()
}

/// Drops reasoning from assistant messages before the latest user message. Thinking
/// providers want reasoning back only within the current tool-call round; earlier
/// rounds are stale context that some providers reject.
pub fn strip_stale_reasoning(messages: &mut [Message]) {
    let Some(last_user) = messages
        .iter()
        .rposition(|message| message.role == Role::User)
    else {
        return;
    };
    for message in &mut messages[..last_user] {
        message.reasoning_content = None;
    }
}

/// Where the verbatim tail starts: the earliest message whose suffix fits
/// the budget. Never a tool result, which must follow its tool call.
pub fn tail_start(messages: &[Message], budget: usize) -> usize {
    let mut tokens = 0usize;
    let mut start: Option<usize> = None;
    for index in (0..messages.len()).rev() {
        tokens += estimate_tokens(&messages[index..index + 1]);
        if tokens > budget && start.is_some() {
            break;
        }
        if messages[index].role != Role::Tool {
            start = Some(index);
        }
    }
    start.unwrap_or(0)
}

/// Keeps the first and last `max / 2` bytes, since errors and exit codes
/// usually sit at the end.
fn clip(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.to_owned();
    }
    let mut head = max / 2;
    while !text.is_char_boundary(head) {
        head -= 1;
    }
    let mut tail = text.len() - max / 2;
    while !text.is_char_boundary(tail) {
        tail += 1;
    }
    format!("{}\n[... truncated ...]\n{}", &text[..head], &text[tail..])
}

/// Renders a summary input message, clipping tool results and tool-call
/// arguments to [`SUMMARY_CLIP_BYTES`].
pub fn serialize(message: &Message) -> String {
    match message.role {
        Role::Tool => format!(
            "[Tool result]: {}",
            clip(message.content.as_deref().unwrap_or(""), SUMMARY_CLIP_BYTES)
        ),
        Role::User => format!("[User]: {}", message.content.as_deref().unwrap_or("")),
        _ => {
            let mut parts: Vec<String> = message
                .content
                .as_ref()
                .filter(|content| !content.is_empty())
                .map(|content| format!("[Assistant]: {content}"))
                .into_iter()
                .collect();
            parts.extend(message.tool_calls.iter().flatten().map(|call| {
                format!(
                    "[Assistant tool call]: {}({})",
                    call.function.name,
                    clip(&call.function.arguments, SUMMARY_CLIP_BYTES)
                )
            }));
            parts.join("\n")
        }
    }
}

/// Renders the summarization request, including a previous summary when present.
pub fn compaction_prompt(
    prompts: &PromptSet,
    previous_summary: Option<&str>,
    context: &[String],
) -> String {
    let conversation = format!(
        "Here is the conversation so far:\n\n<conversation>\n{}\n</conversation>",
        context.join("\n\n")
    );
    match previous_summary {
        None => [conversation, prompts.compaction.new_summary.clone(), prompts.compaction.template.clone()]
            .join("\n\n"),
        Some(previous) => [
            conversation,
            format!(
                "Here is the summary of the conversation before the <conversation> above:\n\n<prior-summary>\n{previous}\n</prior-summary>"
            ),
            prompts.compaction.update_instructions.clone(),
            prompts.compaction.template.clone(),
        ]
        .join("\n\n"),
    }
}

/// Summarizes messages with any prior summary, folding the stream without forwarding
/// deltas. History too large for one `usable`-token request is folded in chunks,
/// each updating the summary of the chunks before it.
pub async fn summarize(
    model: &dyn ChatModel,
    prompts: &PromptSet,
    messages: &[Message],
    previous_summary: Option<&str>,
    usable: usize,
    token: &CancellationToken,
) -> Result<String> {
    let system = Message::system(prompts.compaction.system.clone());
    let mut summary = previous_summary.map(str::to_owned);
    let mut context = messages.iter().map(serialize).peekable();
    while context.peek().is_some() {
        // The fixed prompt cannot shrink, so tiny contexts still give the
        // conversation half the budget.
        let room = usable
            .saturating_sub(
                estimate_tokens(std::slice::from_ref(&system))
                    + estimate_str(&compaction_prompt(prompts, summary.as_deref(), &[])),
            )
            .max(usable / 2);
        let mut chunk: Vec<String> = Vec::new();
        let mut used = 0;
        while let Some(part) = context.peek() {
            // One more token for the separator between parts.
            let tokens = estimate_str(part) + 1;
            if used + tokens <= room {
                used += tokens;
                chunk.extend(context.next());
            } else if chunk.is_empty() {
                let max = room.saturating_mul(4).saturating_sub(64);
                chunk.extend(context.next().map(|part| clip(&part, max)));
                break;
            } else {
                break;
            }
        }
        summary = Some(
            summarize_chunk(model, &system, prompts, &chunk, summary.as_deref(), token).await?,
        );
    }
    summary.ok_or_else(|| Error::Run("there was nothing to summarize".to_owned()))
}

async fn summarize_chunk(
    model: &dyn ChatModel,
    system: &Message,
    prompts: &PromptSet,
    context: &[String],
    previous_summary: Option<&str>,
    token: &CancellationToken,
) -> Result<String> {
    let request = ChatRequest {
        messages: vec![
            system.clone(),
            Message::user(compaction_prompt(prompts, previous_summary, context)),
        ],
        tools: Vec::new(),
    };
    let turn = ChatTurn::fold(model.stream(request, token).await?).await?;
    let summary = turn.message.content.unwrap_or_default();
    let summary = summary.trim();
    if summary.is_empty() {
        return Err(Error::Run("the summary came back empty".to_owned()));
    }
    Ok(summary.to_owned())
}

/// Compaction callback for in-memory runs: replaces older history with a summary
/// checkpoint.
pub fn in_memory_compactor(
    model: Arc<dyn ChatModel>,
    prompts: Arc<PromptSet>,
    usable: usize,
) -> CompactFn {
    Arc::new(move |messages: Vec<Message>, token: CancellationToken| {
        let model = model.clone();
        let prompts = prompts.clone();
        Box::pin(
            async move { compact_conversation(&messages, usable, &*model, &prompts, &token).await },
        )
    })
}

/// Replaces history before the kept tail with a summary checkpoint, preserving the
/// latest user message. Returns `None` when there is nothing to summarize.
pub async fn compact_conversation(
    messages: &[Message],
    usable: usize,
    model: &dyn ChatModel,
    prompts: &PromptSet,
    token: &CancellationToken,
) -> Result<Option<Vec<Message>>> {
    let (system, previous, rest) = split_checkpointed(messages);
    let start = tail_start(rest, tail_budget(usable));
    let last_user = rest.iter().rposition(|message| message.role == Role::User);
    let head: Vec<Message> = rest[..start]
        .iter()
        .enumerate()
        .filter(|(index, _)| Some(*index) != last_user)
        .map(|(_, message)| message.clone())
        .collect();
    if head.is_empty() {
        return Ok(None);
    }
    let summary = summarize(model, prompts, &head, previous.as_deref(), usable, token).await?;
    let mut compacted: Vec<Message> = Vec::new();
    if let Some(system) = system {
        compacted.push(system.clone());
    }
    compacted.push(Message::user(prompts.checkpoint(&summary)));
    // Preserve the latest user message even when it precedes the kept tail.
    if let Some(index) = last_user
        && index < start
    {
        compacted.push(rest[index].clone());
    }
    compacted.extend_from_slice(&rest[start..]);
    Ok(Some(compacted))
}

/// Splits `[system?, checkpoint?,...rest]`.
fn split_checkpointed(messages: &[Message]) -> (Option<&Message>, Option<String>, &[Message]) {
    let mut index = 0;
    let system = messages
        .first()
        .filter(|message| message.role == Role::System);
    if system.is_some() {
        index = 1;
    }
    let previous = messages.get(index).and_then(|message| {
        if message.role == Role::User {
            extract_checkpoint(message.content.as_deref()?)
        } else {
            None
        }
    });
    if previous.is_some() {
        index += 1;
    }
    (system, previous, &messages[index..])
}

/// The summary inside a checkpoint message, when it is one.
fn extract_checkpoint(content: &str) -> Option<String> {
    if !content.contains("<conversation-checkpoint>") {
        return None;
    }
    let start = content.find("<summary>")? + "<summary>".len();
    let end = start + content[start..].find("</summary>")?;
    Some(content[start..end].trim().to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::ToolCall;

    fn user(content: &str) -> Message {
        Message::user(content)
    }

    fn assistant(content: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: Some(content.to_owned()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    fn call(id: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: None,
            tool_calls: Some(vec![ToolCall::new(id, "bash", "{}")]),
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    fn result(id: &str, tokens: usize) -> Message {
        Message::tool(id, "x".repeat(tokens * 4))
    }

    fn turn(n: usize, tokens: usize) -> Vec<Message> {
        vec![
            user(&format!("q{n}")),
            call(&format!("c{n}")),
            result(&format!("c{n}"), tokens),
        ]
    }

    #[test]
    fn estimates_four_chars_per_token_over_content_and_calls() {
        assert_eq!(estimate_tokens(&[]), 0);
        assert_eq!(estimate_tokens(&[user("abcdefgh")]), 2);
        let with_call = call("id1");
        let json_len = serde_json::to_string(with_call.tool_calls.as_ref().unwrap())
            .unwrap()
            .len();
        assert_eq!(
            estimate_tokens(&[with_call]),
            (json_len as f64 / 4.0).round() as usize
        );
    }

    #[test]
    fn usable_tokens_keeps_the_output_reserve() {
        assert_eq!(usable_tokens(22_000).unwrap(), 16_500);
        assert_eq!(usable_tokens(128_000).unwrap(), 108_000);
        assert_eq!(usable_tokens(8_192).unwrap(), 6_144);
        assert_eq!(usable_tokens(16_384).unwrap(), 12_288);
        assert!(usable_tokens(1_023).is_err());
        assert!(usable_tokens(0).is_err());
    }

    #[test]
    fn tail_budget_is_quarter_clamped() {
        assert_eq!(tail_budget(2_000), 500);
        assert_eq!(tail_budget(512), 128);
        assert_eq!(tail_budget(10_000), 2_500);
        assert_eq!(tail_budget(100_000), 15_000);
    }

    #[test]
    fn prune_clears_outputs_older_than_the_protected_budget() {
        let messages: Vec<Message> = (1..=5).flat_map(|n| turn(n, 30_000)).collect();
        let pruned = prune(&messages);
        let outputs: Vec<bool> = pruned
            .iter()
            .filter(|message| message.role == Role::Tool)
            .map(|message| message.content.as_deref() == Some(PRUNED))
            .collect();
        // Only the newest 40k tokens of tool output survive.
        assert_eq!(outputs, [true, true, true, true, false]);
        assert_ne!(
            messages[2].content.as_deref(),
            Some(PRUNED),
            "the input is not mutated"
        );
    }

    #[test]
    fn prune_works_within_a_single_user_turn() {
        let mut messages = vec![user("q")];
        for n in 1..=5 {
            messages.push(call(&format!("c{n}")));
            messages.push(result(&format!("c{n}"), 30_000));
        }
        let pruned = prune(&messages);
        let outputs: Vec<bool> = pruned
            .iter()
            .filter(|message| message.role == Role::Tool)
            .map(|message| message.content.as_deref() == Some(PRUNED))
            .collect();
        assert_eq!(outputs, [true, true, true, true, false]);
    }

    #[test]
    fn prune_spares_results_the_model_has_not_seen() {
        let mut messages = turn(1, 30_000);
        messages.push(Message {
            role: Role::Assistant,
            content: None,
            tool_calls: Some(
                ["c2", "c3", "c4"]
                    .map(|id| ToolCall::new(id, "bash", "{}"))
                    .to_vec(),
            ),
            tool_call_id: None,
            reasoning_content: None,
        });
        for id in ["c2", "c3", "c4"] {
            messages.push(result(id, 30_000));
        }
        let pruned = prune(&messages);
        let outputs: Vec<bool> = pruned
            .iter()
            .filter(|message| message.role == Role::Tool)
            .map(|message| message.content.as_deref() == Some(PRUNED))
            .collect();
        // The 90k-token batch exceeds the protected budget but is unseen: only
        // the consumed result goes.
        assert_eq!(outputs, [true, false, false, false]);
    }

    #[test]
    fn prune_leaves_everything_when_it_would_free_too_little() {
        let mut messages = turn(1, 15_000);
        messages.extend(turn(2, 30_000));
        messages.extend(turn(3, 1));
        messages.extend(turn(4, 1));
        assert_eq!(prune(&messages), messages);
    }

    #[test]
    fn stale_reasoning_is_stripped_before_the_latest_user_message() {
        let reasoned = |text: &str| Message {
            reasoning_content: Some("why".into()),
            ..call(text)
        };
        let mut messages = vec![
            Message::system("sys"),
            user("q1"),
            reasoned("c1"),
            result("c1", 1),
            user("q2"),
            reasoned("c2"),
            result("c2", 1),
            reasoned("c3"),
        ];
        strip_stale_reasoning(&mut messages);
        let kept: Vec<bool> = messages
            .iter()
            .filter(|message| message.role == Role::Assistant)
            .map(|message| message.reasoning_content.is_some())
            .collect();
        assert_eq!(
            kept,
            [false, true, true],
            "the current round keeps its reasoning"
        );

        let mut no_user = vec![reasoned("c1")];
        strip_stale_reasoning(&mut no_user);
        assert!(no_user[0].reasoning_content.is_some());
    }

    #[test]
    fn tail_start_keeps_complete_call_result_groups() {
        let messages = vec![
            user("a"),
            call("c1"),
            result("c1", 500),
            user("b"),
            call("c2"),
            result("c2", 900),
            assistant("done"),
        ];
        assert_eq!(tail_start(&messages, 1_000), 3);
        // The 900-token result does not fit, so only the final answer is kept.
        assert_eq!(tail_start(&messages, 600), 6);
        // A huge last result still keeps the call that produced it.
        assert_eq!(
            tail_start(&[user("a"), call("c1"), result("c1", 5_000)], 1_000),
            1
        );
        assert_eq!(tail_start(&[user("only")], 10), 0);
    }

    #[test]
    fn serialization_clips_tool_results() {
        assert_eq!(serialize(&user("q1")), "[User]: q1");
        assert_eq!(
            serialize(&result("c1", 10)),
            format!("[Tool result]: {}", "x".repeat(40))
        );
        let long = Message::tool("c1", format!("start{}end", "x".repeat(SUMMARY_CLIP_BYTES)));
        let rendered = serialize(&long);
        assert!(rendered.starts_with("[Tool result]: start"), "{rendered}");
        assert!(rendered.contains("\n[... truncated ...]\n"), "{rendered}");
        assert!(rendered.ends_with("end"), "{rendered}");
        assert!(rendered.len() < SUMMARY_CLIP_BYTES + 64);
        assert_eq!(serialize(&call("c1")), "[Assistant tool call]: bash({})");
        let mut write = call("c2");
        write.tool_calls.as_mut().unwrap()[0].function.arguments = "y".repeat(50_000);
        assert!(serialize(&write).len() < SUMMARY_CLIP_BYTES + 64);
        assert_eq!(serialize(&assistant("hi")), "[Assistant]: hi");
    }

    #[test]
    fn compaction_prompts_merge_previous_summaries() {
        let prompts = PromptSet::parse(crate::prompt::EMBEDDED_PROMPTS).unwrap();
        let fresh = compaction_prompt(&prompts, None, &["[User]: q1".to_owned()]);
        assert!(fresh.contains("<conversation>\n[User]: q1\n</conversation>"));
        assert!(fresh.contains("Create a new anchored summary"));
        assert!(!fresh.contains("<prior-summary>"));

        let merged = compaction_prompt(&prompts, Some("EARLIER"), &["[User]: q2".to_owned()]);
        assert!(
            merged.contains("<prior-summary>\nEARLIER\n</prior-summary>"),
            "{merged}"
        );
        assert!(merged.contains("Carry forward objectives"));
    }

    #[test]
    fn checkpoints_round_trip() {
        let prompts = PromptSet::parse(crate::prompt::EMBEDDED_PROMPTS).unwrap();
        let checkpoint = Message::user(prompts.checkpoint("S1"));
        let messages = vec![Message::system("sys"), checkpoint, user("q")];
        let (system, previous, rest) = split_checkpointed(&messages);
        assert_eq!(system.unwrap().content.as_deref(), Some("sys"));
        assert_eq!(previous.as_deref(), Some("S1"));
        assert_eq!(rest.len(), 1);
        assert_eq!(extract_checkpoint("plain user text"), None);
    }
}
