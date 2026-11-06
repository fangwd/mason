//! The `look` tool: a main model that takes no images asks a vision model about
//! the images a tool call returned.

use std::sync::Arc;

use schemars::{JsonSchema, schema_for};
use serde::Deserialize;
use serde_json::Value;

use crate::Error;
use crate::context::estimate_tokens;
use crate::llm::message::{IMAGES_NOT_SENT, TurnAccumulator};
use crate::llm::{ChatModel, ChatRequest, Message, Role, ToolSpec};
use crate::prompt::LookPrompts;
use crate::tool::{Tool, ToolCtx, ToolError, ToolOutput};

const CUT_OFF: &str = "\n\n[the vision model's answer was cut off at its output limit; ask a narrower question for the rest]";

/// What a tool result says in place of the images it returned.
pub fn hint(call_id: &str) -> String {
    let note = IMAGES_NOT_SENT.trim_end_matches(']');
    format!("{note}; ask about them with look, call_id \"{call_id}\"]")
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct Args {
    /// The call_id of the tool result whose images to look at.
    call_id: String,
    /// What to find out from the images.
    question: String,
}

pub struct LookTool {
    description: String,
    system: String,
    model: Arc<dyn ChatModel>,
    max_response_bytes: usize,
}

impl LookTool {
    pub fn new(
        prompts: &LookPrompts,
        model: Arc<dyn ChatModel>,
        max_response_bytes: usize,
    ) -> Self {
        Self {
            description: prompts.description.clone(),
            system: prompts.system.clone(),
            model,
            max_response_bytes,
        }
    }
}

#[async_trait::async_trait]
impl Tool for LookTool {
    fn spec(&self) -> ToolSpec {
        let parameters = serde_json::to_value(schema_for!(Args))
            .unwrap_or_else(|_| serde_json::json!({"type": "object"}));
        ToolSpec::function("look", self.description.clone(), parameters)
    }

    async fn call(&self, args: Value, ctx: &ToolCtx) -> Result<ToolOutput, ToolError> {
        let args: Args = serde_json::from_value(args)
            .map_err(|err| ToolError::failure(format!("invalid arguments: {err}")))?;
        let images = match &ctx.conversation {
            Some(conversation) => conversation
                .lock()
                .await
                .iter()
                .rev()
                .find(|message| {
                    message.role == Role::Tool
                        && message.tool_call_id.as_deref() == Some(args.call_id.as_str())
                })
                .map(|message| message.images.clone())
                .unwrap_or_default(),
            None => Vec::new(),
        };
        if images.is_empty() {
            return Err(ToolError::failure(format!(
                "no images from tool call {} to look at",
                args.call_id
            )));
        }
        let request = ChatRequest {
            messages: vec![
                Message::system(self.system.clone()),
                Message {
                    images,
                    ..Message::user(args.question)
                },
            ],
            tools: Vec::new(),
        };
        let estimated = estimate_tokens(&request.messages);
        if let Some(budget) = &ctx.budget {
            budget.charge_tokens(estimated).map_err(ToolError::Fatal)?;
        }
        let turn = match self.model.stream(request, &ctx.token).await {
            Ok(stream) => {
                TurnAccumulator::new(self.max_response_bytes)
                    .fold(stream)
                    .await
            }
            Err(err) => Err(err),
        };
        let turn = match turn {
            Ok(turn) => turn,
            Err(Error::Cancelled) => return Err(ToolError::Fatal(Error::Cancelled)),
            Err(err) => {
                return Err(ToolError::failure(format!(
                    "the vision model failed: {err}"
                )));
            }
        };
        if let Some(budget) = &ctx.budget {
            budget
                .charge_reply(estimated, &turn)
                .map_err(ToolError::Fatal)?;
        }
        let mut answer = turn.message.content.unwrap_or_default();
        if answer.trim().is_empty() {
            return Err(ToolError::failure("the vision model gave no answer"));
        }
        if turn.finish_reason.as_deref() == Some("length") {
            answer.push_str(CUT_OFF);
        }
        Ok(ToolOutput {
            text: answer,
            images: Vec::new(),
            usage: turn.usage,
        })
    }

    fn read_only(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{ChatDelta, ChatStream, Image};
    use crate::prompt::PromptSource;
    use tokio::sync::Mutex;
    use tokio_util::sync::CancellationToken;

    struct Unused;

    #[async_trait::async_trait]
    impl ChatModel for Unused {
        fn context_tokens(&self) -> usize {
            128_000
        }

        async fn stream(
            &self,
            _request: ChatRequest,
            _token: &CancellationToken,
        ) -> crate::Result<ChatStream> {
            unreachable!("a look without images asks no model")
        }
    }

    #[tokio::test]
    async fn a_call_without_images_is_a_failure_the_model_sees() {
        let prompts = PromptSource::Embedded.load().unwrap();
        let tool = LookTool::new(&prompts.look, Arc::new(Unused), 1024);
        let mut ctx = ToolCtx::new(".");
        ctx.conversation = Some(Arc::new(Mutex::new(vec![
            Message::tool("c1", "plain text"),
            Message {
                images: vec![Image {
                    mime_type: "image/png".into(),
                    data: "aW1n".into(),
                }],
                ..Message::user("an image from the user, not a tool")
            },
        ])));
        for call_id in ["c1", "c9"] {
            let args = serde_json::json!({"call_id": call_id, "question": "what?"});
            let err = tool.call(args, &ctx).await.unwrap_err();
            assert!(
                matches!(&err, ToolError::Failure(message) if message.contains(call_id)),
                "{err:?}"
            );
        }
    }

    struct Replies {
        text: String,
        finish_reason: &'static str,
    }

    #[async_trait::async_trait]
    impl ChatModel for Replies {
        fn context_tokens(&self) -> usize {
            128_000
        }

        async fn stream(
            &self,
            _request: ChatRequest,
            _token: &CancellationToken,
        ) -> crate::Result<ChatStream> {
            let deltas = vec![
                Ok(ChatDelta::Text(self.text.clone())),
                Ok(ChatDelta::Done {
                    finish_reason: Some(self.finish_reason.to_owned()),
                }),
            ];
            Ok(Box::pin(futures_util::stream::iter(deltas)))
        }
    }

    async fn look(model: Replies, max_response_bytes: usize) -> Result<ToolOutput, ToolError> {
        let prompts = PromptSource::Embedded.load().unwrap();
        let tool = LookTool::new(&prompts.look, Arc::new(model), max_response_bytes);
        let mut ctx = ToolCtx::new(".");
        ctx.conversation = Some(Arc::new(Mutex::new(vec![Message {
            images: vec![Image {
                mime_type: "image/png".into(),
                data: "aW1n".into(),
            }],
            ..Message::tool("c1", "a screenshot")
        }])));
        let args = serde_json::json!({"call_id": "c1", "question": "what?"});
        tool.call(args, &ctx).await
    }

    #[tokio::test]
    async fn an_answer_over_the_response_byte_limit_fails() {
        let model = Replies {
            text: "x".repeat(300),
            finish_reason: "stop",
        };
        let err = look(model, 256).await.unwrap_err();
        assert!(
            matches!(&err, ToolError::Failure(message) if message.contains("byte limit")),
            "{err:?}"
        );
    }

    #[tokio::test]
    async fn a_cut_off_answer_says_so() {
        let model = Replies {
            text: "a red button and".into(),
            finish_reason: "length",
        };
        let output = look(model, 1024).await.unwrap();
        assert_eq!(output.text, format!("a red button and{CUT_OFF}"));
    }
}
