//! Post-run reflection: deterministic signals decide whether a finished run is
//! worth reviewing, and a clipped transcript is what the review sees. Also the
//! budgets of reflection and memory consolidation.

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use crate::context::serialize;
use crate::evaluate::{Outcome, Verdict};
use crate::llm::{Message, Role};
use crate::store::SessionId;

/// Distinct tool errors that flag a run.
pub const TOOL_ERRORS: usize = 3;
/// Identical tool calls that flag a run.
pub const REPEATED_CALLS: usize = 3;
/// Assistant turns that flag a run.
pub const MANY_STEPS: usize = 30;
/// Steps a reflection may take.
pub const REFLECTION_STEPS: usize = 8;
/// Wall-clock budget of one reflection.
pub const REFLECTION_TIMEOUT: Duration = Duration::from_secs(120);
/// Transcript bytes a reflection sees, at most.
pub const REFLECTION_TRANSCRIPT_BYTES: usize = 48_000;
/// The shortest gap between two reflections on one stored session.
pub const REFLECTION_COOLDOWN: Duration = Duration::from_secs(30 * 60);
/// Steps a consolidation may take.
pub const CONSOLIDATION_STEPS: usize = 10;
/// Wall-clock budget of one consolidation.
pub const CONSOLIDATION_TIMEOUT: Duration = Duration::from_secs(300);
/// Memory and learned-skill bytes a consolidation sees in full, at most.
pub const CONSOLIDATION_INPUT_BYTES: usize = 64_000;

/// A finished run to review.
#[derive(Debug, Clone, Copy)]
pub struct RunReview<'a> {
    /// The run's conversation; system messages are ignored.
    pub messages: &'a [Message],
    pub failed: bool,
    /// The user sent input while the run was working (see [`steered`]).
    pub steered: bool,
    /// The stored session, recorded as the provenance of saved memories.
    pub session: Option<SessionId>,
    /// The evaluator's verdict, when one graded the run.
    pub verdict: Option<&'a Verdict>,
}

impl<'a> RunReview<'a> {
    /// A run with nothing known beyond its messages.
    pub fn new(messages: &'a [Message]) -> Self {
        Self {
            messages,
            failed: false,
            steered: false,
            session: None,
            verdict: None,
        }
    }
}

/// Why a run deserves review.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Signal {
    ToolErrors(usize),
    /// The most often repeated identical tool call and its count.
    RepeatedCall(String, usize),
    ManySteps(usize),
    /// The user sent input while the run was working.
    Steered,
    Failed,
    /// The evaluator judged the run a failure or partial success.
    Judged(Outcome, String),
}

impl std::fmt::Display for Signal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Signal::ToolErrors(count) => write!(f, "{count} distinct tool errors"),
            Signal::RepeatedCall(name, count) => {
                write!(
                    f,
                    "`{name}` was called {count} times with identical arguments"
                )
            }
            Signal::ManySteps(count) => write!(f, "the run took {count} model turns"),
            Signal::Steered => write!(f, "the user redirected the run while it was working"),
            Signal::Failed => write!(f, "the run failed"),
            Signal::Judged(outcome, reason) => {
                write!(f, "the evaluator judged it {outcome}: {reason}")
            }
        }
    }
}

/// The signals in one run: its messages, plus the facts only the caller knows.
/// Only a failure or partial verdict is a signal, and a run that already saved
/// a memory has none: its own lesson stands.
pub fn signals(run: &RunReview<'_>) -> Vec<Signal> {
    let messages = run.messages;
    if saved_memory(messages) {
        return Vec::new();
    }
    let mut signals = Vec::new();
    let errors = messages
        .iter()
        .filter(|message| message.role == Role::Tool)
        .filter_map(|message| message.content.as_deref())
        .filter(|content| content.starts_with("Error:"))
        .collect::<HashSet<_>>()
        .len();
    if errors >= TOOL_ERRORS {
        signals.push(Signal::ToolErrors(errors));
    }

    let mut calls: HashMap<(&str, &str), usize> = HashMap::new();
    for call in messages
        .iter()
        .flat_map(|message| message.tool_calls.iter().flatten())
    {
        *calls
            .entry((&call.function.name, &call.function.arguments))
            .or_default() += 1;
    }
    if let Some(((name, _), count)) = calls
        .into_iter()
        .filter(|(_, count)| *count >= REPEATED_CALLS)
        .max_by_key(|((name, _), count)| (*count, std::cmp::Reverse(*name)))
    {
        signals.push(Signal::RepeatedCall(name.to_owned(), count));
    }

    let turns = messages
        .iter()
        .filter(|message| message.role == Role::Assistant)
        .count();
    if turns >= MANY_STEPS {
        signals.push(Signal::ManySteps(turns));
    }

    if run.steered {
        signals.push(Signal::Steered);
    }
    if run.failed {
        signals.push(Signal::Failed);
    }
    if let Some(verdict) = run.verdict
        && matches!(verdict.outcome, Outcome::Failure | Outcome::Partial)
    {
        signals.push(Signal::Judged(verdict.outcome, verdict.reason.clone()));
    }
    signals
}

/// Whether the run changed memory itself.
fn saved_memory(messages: &[Message]) -> bool {
    messages
        .iter()
        .flat_map(|message| message.tool_calls.iter().flatten())
        .filter(|call| call.function.name == "memory")
        .filter_map(|call| serde_json::from_str::<serde_json::Value>(&call.function.arguments).ok())
        .any(|args| {
            args.get("command")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|command| command != "view")
        })
}

/// Whether a stored run's messages hold user input after its first model turn.
/// In-memory runs cannot be steered, and their compaction checkpoints would read
/// as user input, so only stored transcripts are checked.
pub fn steered(messages: &[Message]) -> bool {
    messages
        .iter()
        .skip_while(|message| message.role != Role::Assistant)
        .any(|message| message.role == Role::User)
}

/// The run as text within `budget` bytes; an oversized run keeps its start and
/// end around an omission marker.
pub fn transcript(messages: &[Message], budget: usize) -> String {
    let rendered: Vec<String> = messages
        .iter()
        .filter(|message| message.role != Role::System)
        .map(serialize)
        .collect();
    let total: usize = rendered.iter().map(|part| part.len() + 2).sum();
    if total <= budget {
        return rendered.join("\n\n");
    }
    let half = budget / 2;
    let mut head = Vec::new();
    let mut used = 0;
    for part in &rendered {
        if used + part.len() + 2 > half {
            break;
        }
        used += part.len() + 2;
        head.push(part.as_str());
    }
    let mut tail = Vec::new();
    used = 0;
    for part in rendered[head.len()..].iter().rev() {
        if used + part.len() + 2 > half {
            break;
        }
        used += part.len() + 2;
        tail.push(part.as_str());
    }
    tail.reverse();
    let omitted = rendered.len() - head.len() - tail.len();
    let marker = format!("[... {omitted} messages omitted ...]");
    head.into_iter()
        .chain(std::iter::once(marker.as_str()))
        .chain(tail)
        .collect::<Vec<_>>()
        .join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::ToolCall;

    fn assistant(content: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: Some(content.to_owned()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    fn assistant_call(name: &str, arguments: &str) -> Message {
        Message {
            content: None,
            tool_calls: Some(vec![ToolCall::new("id", name, arguments)]),
            ..assistant("")
        }
    }

    fn turns(count: usize) -> Vec<Message> {
        let mut messages = vec![Message::user("go")];
        for i in 0..count {
            messages.push(assistant_call("read", &format!("{{\"path\":\"{i}\"}}")));
            messages.push(Message::tool(format!("read-{i}"), "ok"));
        }
        messages
    }

    #[test]
    fn a_clean_run_has_no_signals() {
        let mut messages = turns(MANY_STEPS - 2);
        messages.push(Message::tool("x", "Error: one failure is normal"));
        messages.push(assistant("done"));
        assert!(signals(&RunReview::new(&messages)).is_empty());
    }

    #[test]
    fn distinct_tool_errors_flag_at_the_threshold() {
        let mut messages = turns(1);
        for _ in 0..4 {
            messages.push(Message::tool("a", "Error: no such file"));
        }
        messages.push(Message::tool("b", "Error: old text not found"));
        assert!(
            signals(&RunReview::new(&messages)).is_empty(),
            "repeats count once"
        );
        messages.push(Message::tool("c", "Error: permission denied"));
        assert_eq!(
            signals(&RunReview::new(&messages)),
            vec![Signal::ToolErrors(3)]
        );
    }

    #[test]
    fn a_run_that_saved_a_memory_is_not_reviewed() {
        let mut messages = turns(MANY_STEPS);
        messages.push(assistant_call("memory", r#"{"command":"view"}"#));
        assert!(
            !signals(&RunReview::new(&messages)).is_empty(),
            "viewing saves nothing"
        );
        messages.push(assistant_call(
            "memory",
            r#"{"command":"create","name":"a","description":"b","body":"c"}"#,
        ));
        assert!(
            signals(&RunReview {
                failed: true,
                ..RunReview::new(&messages)
            })
            .is_empty()
        );
    }

    #[test]
    fn identical_calls_flag_and_name_the_most_repeated() {
        let mut messages = turns(1);
        for _ in 0..2 {
            messages.push(assistant_call("glob", r#"{"pattern":"*.rs"}"#));
        }
        for _ in 0..3 {
            messages.push(assistant_call("grep", r#"{"pattern":"x"}"#));
        }
        assert_eq!(
            signals(&RunReview::new(&messages)),
            vec![Signal::RepeatedCall("grep".into(), 3)]
        );
    }

    #[test]
    fn many_steps_steering_and_failure_flag() {
        let messages = turns(MANY_STEPS);
        assert_eq!(
            signals(&RunReview {
                failed: true,
                steered: true,
                ..RunReview::new(&messages)
            }),
            vec![
                Signal::ManySteps(MANY_STEPS),
                Signal::Steered,
                Signal::Failed
            ]
        );
    }

    #[test]
    fn only_failure_and_partial_verdicts_flag() {
        let messages = turns(1);
        let judged = |outcome| {
            let verdict = Verdict {
                outcome,
                reason: "tests never ran".into(),
            };
            signals(&RunReview {
                verdict: Some(&verdict),
                ..RunReview::new(&messages)
            })
        };
        assert_eq!(
            judged(Outcome::Partial),
            vec![Signal::Judged(Outcome::Partial, "tests never ran".into())]
        );
        assert_eq!(judged(Outcome::Failure).len(), 1);
        assert!(judged(Outcome::Success).is_empty());
        assert!(judged(Outcome::Unclear).is_empty());
    }

    #[test]
    fn steering_is_user_input_after_the_first_turn() {
        let mut messages = vec![Message::user("go"), Message::user("and also this")];
        assert!(!steered(&messages));
        messages.push(assistant("working"));
        assert!(!steered(&messages));
        messages.push(Message::user("no, use the staging database"));
        assert!(steered(&messages));
    }

    #[test]
    fn an_oversized_transcript_keeps_its_ends() {
        let mut messages = vec![Message::system("hidden"), Message::user("first")];
        for i in 0..50 {
            messages.push(assistant(&format!("step {i} {}", "x".repeat(100))));
        }
        messages.push(assistant("last"));
        let full = transcript(&messages, usize::MAX);
        assert!(!full.contains("hidden"));
        assert!(full.starts_with("[User]: first"));

        let clipped = transcript(&messages, 1_000);
        assert!(clipped.len() <= 1_000 + 64, "{}", clipped.len());
        assert!(clipped.starts_with("[User]: first"), "{clipped}");
        assert!(clipped.ends_with("[Assistant]: last"), "{clipped}");
        assert!(clipped.contains("messages omitted"), "{clipped}");
    }
}
