//! Run evaluation: an answerer stands in for a user who will not say whether a
//! run succeeded. The default asks a (typically cheaper) model.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio_util::sync::CancellationToken;

use crate::error::Result;
use crate::llm::{ChatModel, ChatRequest, ChatTurn, Message, ToolSpec};
use crate::prompt::PromptSet;
use crate::reflect::{RunReview, transcript};

/// Transcript bytes a model evaluator sees.
pub const EVALUATION_BYTES: usize = 24_000;

/// Whether a run achieved what the user asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Outcome {
    Success,
    Partial,
    Failure,
    Unclear,
}

impl std::fmt::Display for Outcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Outcome::Success => "success",
            Outcome::Partial => "partial",
            Outcome::Failure => "failure",
            Outcome::Unclear => "unclear",
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Verdict {
    pub outcome: Outcome,
    pub reason: String,
}

impl Verdict {
    fn unclear(reason: impl Into<String>) -> Self {
        Self {
            outcome: Outcome::Unclear,
            reason: reason.into(),
        }
    }
}

/// Answers "did this run succeed?" for a finished run.
#[async_trait::async_trait]
pub trait Evaluator: Send + Sync {
    /// `prompts` is the agent's snapshot; answerers that are not models may ignore it.
    async fn evaluate(
        &self,
        run: &RunReview<'_>,
        prompts: &PromptSet,
        token: &CancellationToken,
    ) -> Result<Verdict>;
}

/// Grades runs with one model request and a `verdict` tool call.
pub struct LlmEvaluator {
    model: Arc<dyn ChatModel>,
}

impl LlmEvaluator {
    pub fn new(model: Arc<dyn ChatModel>) -> Self {
        Self { model }
    }
}

#[async_trait::async_trait]
impl Evaluator for LlmEvaluator {
    async fn evaluate(
        &self,
        run: &RunReview<'_>,
        prompts: &PromptSet,
        token: &CancellationToken,
    ) -> Result<Verdict> {
        let request = ChatRequest {
            messages: vec![
                Message::system(prompts.evaluation.system.clone()),
                Message::user(
                    prompts.evaluation_prompt(&transcript(run.messages, EVALUATION_BYTES)),
                ),
            ],
            tools: vec![verdict_spec(&prompts.evaluation.tool)],
        };
        let turn = ChatTurn::fold(self.model.stream(request, token).await?).await?;
        Ok(parse_verdict(&turn.message))
    }
}

fn verdict_spec(description: &str) -> ToolSpec {
    ToolSpec::function(
        "verdict",
        description,
        json!({
            "type": "object",
            "properties": {
                "outcome": {"type": "string", "enum": ["success", "partial", "failure", "unclear"]},
                "reason": {"type": "string", "description": "One sentence a developer can act on"}
            },
            "required": ["outcome", "reason"],
            "additionalProperties": false
        }),
    )
}

/// The verdict a reply's `verdict` call carries; anything else is unclear.
fn parse_verdict(reply: &Message) -> Verdict {
    let Some(call) = reply
        .tool_calls
        .iter()
        .flatten()
        .find(|call| call.function.name == "verdict")
    else {
        return Verdict::unclear("the evaluator gave no verdict");
    };
    match serde_json::from_str::<Verdict>(&call.function.arguments) {
        Ok(verdict) => Verdict {
            reason: verdict.reason.trim().to_owned(),
            ..verdict
        },
        Err(err) => Verdict::unclear(format!("the evaluator's verdict was invalid: {err}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{Role, ToolCall};

    fn reply(calls: Vec<ToolCall>) -> Message {
        Message {
            role: Role::Assistant,
            content: Some("thinking".into()),
            tool_calls: Some(calls),
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    #[test]
    fn a_verdict_call_is_parsed() {
        let verdict = parse_verdict(&reply(vec![ToolCall::new(
            "c",
            "verdict",
            r#"{"outcome":"partial","reason":" tests never ran "}"#,
        )]));
        assert_eq!(
            verdict,
            Verdict {
                outcome: Outcome::Partial,
                reason: "tests never ran".into()
            }
        );
    }

    #[test]
    fn anything_else_is_unclear() {
        for (calls, reason) in [
            (vec![], "gave no verdict"),
            (vec![ToolCall::new("c", "other", "{}")], "gave no verdict"),
            (
                vec![ToolCall::new(
                    "c",
                    "verdict",
                    r#"{"outcome":"great","reason":"x"}"#,
                )],
                "invalid",
            ),
            (vec![ToolCall::new("c", "verdict", "not json")], "invalid"),
        ] {
            let verdict = parse_verdict(&reply(calls));
            assert_eq!(verdict.outcome, Outcome::Unclear);
            assert!(verdict.reason.contains(reason), "{}", verdict.reason);
        }
    }
}
