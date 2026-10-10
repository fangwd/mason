//! OpenAI-compatible messages, tool calls, and streaming turn accumulation.

use std::collections::BTreeMap;

use futures_util::{Stream, StreamExt};
use serde::{Deserialize, Serialize};

use crate::Error;
use crate::error::Result;
use crate::llm::error::LlmError;

/// The chat message roles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// System instructions.
    System,
    /// User (and steering) input.
    User,
    /// Model replies, including tool-call turns.
    Assistant,
    /// A tool result, keyed by `tool_call_id`.
    Tool,
}

/// A tool call on the wire: `arguments` is the raw JSON text, parsed by the
/// engine so that invalid JSON becomes a fixable tool error, not a turn
/// failure.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// The provider-assigned call id, echoed back on the tool result.
    pub id: String,
    /// Always `"function"` on the OpenAI wire.
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

impl ToolCall {
    /// A `function` tool call with the given arguments JSON text.
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            kind: "function".to_owned(),
            function: FunctionCall {
                name: name.into(),
                arguments: arguments.into(),
            },
        }
    }
}

/// A chat message on the exact OpenAI wire shape.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    /// The text; `null` on assistant turns that only carry tool calls.
    /// Always serialized, as providers expect the key.
    #[serde(default)]
    pub content: Option<String>,
    /// Tool calls requested by an assistant message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// The call this tool result answers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Thinking models expect their reasoning back on their own messages.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self::plain(Role::System, content)
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::plain(Role::User, content)
    }

    /// A tool result answering `tool_call_id`.
    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            tool_call_id: Some(tool_call_id.into()),
            ..Self::plain(Role::Tool, content)
        }
    }

    fn plain(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }
}

/// A tool definition on the OpenAI wire shape.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    /// Always `"function"` on the OpenAI wire.
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ToolFunction,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolFunction {
    /// The tool name the model calls.
    pub name: String,
    /// What the tool does, for the model.
    pub description: String,
    /// The JSON Schema of the arguments object.
    pub parameters: serde_json::Value,
}

impl ToolSpec {
    pub fn function(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Self {
            kind: "function".to_owned(),
            function: ToolFunction {
                name: name.into(),
                description: description.into(),
                parameters,
            },
        }
    }
}

/// What one model turn sends: the conversation and the callable tools.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ChatRequest {
    /// The conversation, a system message first.
    pub messages: Vec<Message>,
    /// Available tools; omitted from requests when empty.
    pub tools: Vec<ToolSpec>,
}

/// The provider's token accounting for one turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

/// One incremental piece of a model turn, in arrival order.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(tag = "kind", content = "content", rename_all = "camelCase")]
pub enum ChatDelta {
    /// A fragment of assistant text.
    Text(String),
    /// A fragment of reasoning (thinking models).
    Reasoning(String),
    /// An incremental tool-call argument fragment, keyed by call index.
    /// `id` and `name` may arrive in a later fragment than the first one
    /// for that index.
    ToolCall {
        /// The provider's call index; fragments of one call share it.
        index: u32,
        id: Option<String>,
        name: Option<String>,
        /// An arguments fragment; concatenated per index, parsed only by the
        /// engine.
        arguments: String,
    },
    /// The provider's token usage, typically just before completion.
    Usage {
        prompt_tokens: u32,
        completion_tokens: u32,
    },
    /// The turn completed; the stream ends here.
    Done {
        /// Why the model stopped, when the provider said.
        finish_reason: Option<String>,
    },
}

/// Completed assistant message and provider usage, independent of transport mode.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatTurn {
    pub message: Message,
    /// Real provider usage, when it reported any.
    pub usage: Option<Usage>,
    /// Why the model stopped, when the provider said.
    pub finish_reason: Option<String>,
}

/// Tool-call fragments accumulated for one call.
#[derive(Debug, Default)]
struct CallFragments {
    index: u32,
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

/// Accumulates deltas while the engine forwards them; also used by [`ChatTurn::fold`].
#[derive(Debug)]
pub struct TurnAccumulator {
    text: String,
    reasoning: String,
    max_bytes: usize,
    received_bytes: usize,
    exceeded: bool,
    /// Calls in arrival order; fragments for an index append to its newest call.
    calls: Vec<CallFragments>,
    newest: BTreeMap<u32, usize>,
    usage: Option<Usage>,
    completed: bool,
    finish_reason: Option<String>,
}

impl Default for TurnAccumulator {
    fn default() -> Self {
        Self {
            text: String::new(),
            reasoning: String::new(),
            calls: Vec::new(),
            newest: BTreeMap::new(),
            usage: None,
            completed: false,
            finish_reason: None,
            max_bytes: 8 * 1024 * 1024,
            received_bytes: 0,
            exceeded: false,
        }
    }
}

impl TurnAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Bound text, reasoning and tool-call fragments across the whole turn.
    pub fn with_max_bytes(mut self, max_bytes: usize) -> Self {
        self.max_bytes = max_bytes;
        self
    }

    /// Accumulates one delta; returns `true` on [`ChatDelta::Done`], after
    /// which the stream should be dropped and [`finish`](Self::finish)
    /// called.
    pub fn push(&mut self, delta: ChatDelta) -> bool {
        let bytes = match &delta {
            ChatDelta::Text(text) | ChatDelta::Reasoning(text) => text.len(),
            ChatDelta::ToolCall {
                id,
                name,
                arguments,
                ..
            } => {
                arguments.len()
                    + id.as_ref().map_or(0, String::len)
                    + name.as_ref().map_or(0, String::len)
                    + 64
            }
            _ => 0,
        };
        self.received_bytes = self.received_bytes.saturating_add(bytes);
        if self.exceeded || self.received_bytes > self.max_bytes {
            self.exceeded = true;
            return true;
        }
        match delta {
            ChatDelta::Text(fragment) => self.text.push_str(&fragment),
            ChatDelta::Reasoning(fragment) => self.reasoning.push_str(&fragment),
            ChatDelta::ToolCall {
                index,
                id,
                name,
                arguments,
            } => {
                // Some endpoints reuse one index for distinct calls: a new id starts
                // a new call rather than merging arguments. An empty id is no id.
                let id = id.filter(|id| !id.is_empty());
                let position = match self.newest.get(&index) {
                    Some(&position)
                        if id.is_none()
                            || self.calls[position].id.is_none()
                            || self.calls[position].id == id =>
                    {
                        position
                    }
                    _ => {
                        self.calls.push(CallFragments {
                            index,
                            ..CallFragments::default()
                        });
                        self.newest.insert(index, self.calls.len() - 1);
                        self.calls.len() - 1
                    }
                };
                let call = &mut self.calls[position];
                if call.id.is_none() {
                    call.id = id;
                }
                if call.name.is_none() {
                    call.name = name;
                }
                call.arguments.push_str(&arguments);
            }
            ChatDelta::Usage {
                prompt_tokens,
                completion_tokens,
            } => {
                self.usage = Some(Usage {
                    prompt_tokens,
                    completion_tokens,
                });
            }
            ChatDelta::Done { finish_reason } => {
                self.completed = true;
                self.finish_reason = finish_reason;
                return true;
            }
        }
        false
    }

    /// Rejects incomplete turns and missing tool IDs/names; argument JSON is validated
    /// by the engine.
    pub fn finish(mut self) -> Result<ChatTurn> {
        if self.exceeded {
            return Err(Error::Llm(LlmError::Protocol(
                "model turn exceeded its response byte limit".into(),
            )));
        }
        if !self.completed {
            return Err(Error::Llm(LlmError::Protocol(
                "the stream ended before the turn completed".to_owned(),
            )));
        }
        self.calls.sort_by_key(|call| call.index);
        let tool_calls = if self.calls.is_empty() {
            None
        } else {
            Some(
                self.calls
                    .into_iter()
                    .map(|call| {
                        let (Some(id), Some(name)) = (call.id, call.name) else {
                            return Err(Error::Llm(LlmError::Protocol(format!(
                                "tool call {} never received its id and name",
                                call.index
                            ))));
                        };
                        Ok(ToolCall::new(id, name, call.arguments))
                    })
                    .collect::<Result<Vec<_>>>()?,
            )
        };
        // Providers reject a null content without tool calls when it is sent
        // back, so an empty text turn keeps an empty string.
        let content = if self.text.is_empty() && tool_calls.is_some() {
            None
        } else {
            Some(self.text)
        };
        let message = Message {
            role: Role::Assistant,
            content,
            tool_calls,
            tool_call_id: None,
            reasoning_content: (!self.reasoning.is_empty()).then_some(self.reasoning),
        };
        Ok(ChatTurn {
            message,
            usage: self.usage,
            finish_reason: self.finish_reason,
        })
    }
}

impl ChatTurn {
    /// Folds a stream to completion without forwarding deltas.
    pub async fn fold(stream: impl Stream<Item = Result<ChatDelta>>) -> Result<Self> {
        futures_util::pin_mut!(stream);
        let mut accumulator = TurnAccumulator::new();
        while let Some(item) = stream.next().await {
            if accumulator.push(item?) {
                break;
            }
        }
        accumulator.finish()
    }
}

#[cfg(test)]
mod tests {
    use futures_util::stream;

    use super::*;

    fn stream_of(deltas: Vec<Result<ChatDelta>>) -> impl Stream<Item = Result<ChatDelta>> {
        stream::iter(deltas)
    }

    fn ok(deltas: Vec<ChatDelta>) -> Vec<Result<ChatDelta>> {
        deltas.into_iter().map(Ok).collect()
    }

    fn call(index: u32, id: Option<&str>, name: Option<&str>, arguments: &str) -> ChatDelta {
        ChatDelta::ToolCall {
            index,
            id: id.map(str::to_owned),
            name: name.map(str::to_owned),
            arguments: arguments.to_owned(),
        }
    }

    #[test]
    fn messages_serialize_on_the_exact_wire_shape() {
        let user = serde_json::to_value(Message::user("hi")).unwrap();
        assert_eq!(user, serde_json::json!({"role": "user", "content": "hi"}));

        let assistant = Message {
            role: Role::Assistant,
            content: None,
            tool_calls: Some(vec![ToolCall::new("c1", "read", "{\"path\":\"a\"}")]),
            tool_call_id: None,
            reasoning_content: Some("thinking".into()),
        };
        assert_eq!(
            serde_json::to_value(&assistant).unwrap(),
            serde_json::json!({
                "role": "assistant",
                "content": null,
                "reasoning_content": "thinking",
                "tool_calls": [{
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "read", "arguments": "{\"path\":\"a\"}"}
                }]
            })
        );

        let tool = serde_json::to_value(Message::tool("c1", "done")).unwrap();
        assert_eq!(
            tool,
            serde_json::json!({"role": "tool", "content": "done", "tool_call_id": "c1"})
        );
    }

    #[test]
    fn tool_spec_serializes_on_the_wire_shape() {
        let spec = ToolSpec::function("read", "Read a file", serde_json::json!({"type": "object"}));
        assert_eq!(
            serde_json::to_value(&spec).unwrap(),
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "Read a file",
                    "parameters": {"type": "object"}
                }
            })
        );
    }

    #[tokio::test]
    async fn fold_concatenates_text_and_reasoning_fragments() {
        let turn = ChatTurn::fold(stream_of(ok(vec![
            ChatDelta::Text("Hel".into()),
            ChatDelta::Reasoning("hmm".into()),
            ChatDelta::Text("lo".into()),
            ChatDelta::Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
            },
            ChatDelta::Done {
                finish_reason: Some("stop".into()),
            },
        ])))
        .await
        .unwrap();
        assert_eq!(turn.message.content.as_deref(), Some("Hello"));
        assert_eq!(turn.message.reasoning_content.as_deref(), Some("hmm"));
        assert_eq!(turn.message.tool_calls, None);
        assert_eq!(
            turn.usage,
            Some(Usage {
                prompt_tokens: 10,
                completion_tokens: 5
            })
        );
    }

    #[tokio::test]
    async fn fold_joins_tool_calls_per_index_in_index_order() {
        // Fragments arrive out of index order and id/name lag behind the
        // first arguments fragment.
        let turn = ChatTurn::fold(stream_of(ok(vec![
            call(1, None, None, "{\"b\":"),
            call(0, Some("c0"), Some("read"), "{\"a\""),
            call(1, Some("c1"), Some("write"), "2}"),
            call(0, None, None, ":1}"),
            ChatDelta::Done {
                finish_reason: Some("tool_calls".into()),
            },
        ])))
        .await
        .unwrap();
        assert_eq!(turn.message.content, None, "a tool turn has null content");
        assert_eq!(
            turn.message.tool_calls.unwrap(),
            vec![
                ToolCall::new("c0", "read", "{\"a\":1}"),
                ToolCall::new("c1", "write", "{\"b\":2}"),
            ]
        );
    }

    #[tokio::test]
    async fn fold_starts_a_new_call_when_an_index_carries_a_new_id() {
        let turn = ChatTurn::fold(stream_of(ok(vec![
            call(0, Some("c0"), Some("read"), "{\"a\""),
            call(0, None, None, ":1}"),
            call(0, Some("c1"), Some("write"), "{\"b\""),
            call(0, None, None, ":2}"),
            ChatDelta::Done {
                finish_reason: Some("tool_calls".into()),
            },
        ])))
        .await
        .unwrap();
        assert_eq!(
            turn.message.tool_calls.unwrap(),
            vec![
                ToolCall::new("c0", "read", "{\"a\":1}"),
                ToolCall::new("c1", "write", "{\"b\":2}"),
            ]
        );
    }

    #[tokio::test]
    async fn an_empty_id_on_a_continuation_fragment_does_not_split_the_call() {
        let turn = ChatTurn::fold(stream_of(ok(vec![
            call(0, Some("c0"), Some("read"), "{\"a\""),
            call(0, Some(""), None, ":1}"),
            ChatDelta::Done {
                finish_reason: Some("tool_calls".into()),
            },
        ])))
        .await
        .unwrap();
        assert_eq!(
            turn.message.tool_calls.unwrap(),
            vec![ToolCall::new("c0", "read", "{\"a\":1}")]
        );
    }

    #[tokio::test]
    async fn fold_carries_the_finish_reason() {
        let turn = ChatTurn::fold(stream_of(ok(vec![
            ChatDelta::Text("partial".into()),
            ChatDelta::Done {
                finish_reason: Some("length".into()),
            },
        ])))
        .await
        .unwrap();
        assert_eq!(turn.finish_reason.as_deref(), Some("length"));
    }

    #[tokio::test]
    async fn an_empty_text_turn_keeps_an_empty_string_not_null() {
        for deltas in [
            vec![ChatDelta::Done {
                finish_reason: Some("stop".into()),
            }],
            vec![
                ChatDelta::Reasoning("hmm".into()),
                ChatDelta::Done {
                    finish_reason: Some("stop".into()),
                },
            ],
        ] {
            let turn = ChatTurn::fold(stream_of(ok(deltas))).await.unwrap();
            assert_eq!(turn.message.content.as_deref(), Some(""));
            let wire = serde_json::to_value(&turn.message).unwrap();
            assert_eq!(wire["content"], serde_json::json!(""), "{wire}");
        }
    }

    #[tokio::test]
    async fn fold_keeps_invalid_argument_json_for_the_engine() {
        // Not parsed here: the engine turns it into a fixable tool error.
        let turn = ChatTurn::fold(stream_of(ok(vec![
            call(0, Some("c0"), Some("read"), "{oops"),
            ChatDelta::Done {
                finish_reason: None,
            },
        ])))
        .await
        .unwrap();
        assert_eq!(
            turn.message.tool_calls.unwrap()[0].function.arguments,
            "{oops"
        );
    }

    #[tokio::test]
    async fn fold_rejects_a_call_that_never_received_id_or_name() {
        for broken in [
            vec![
                call(0, None, Some("read"), "{}"),
                ChatDelta::Done {
                    finish_reason: None,
                },
            ],
            vec![
                call(0, Some("c0"), None, "{}"),
                ChatDelta::Done {
                    finish_reason: None,
                },
            ],
        ] {
            let err = ChatTurn::fold(stream_of(ok(broken))).await.unwrap_err();
            assert!(err.to_string().contains("tool call 0"), "{err}");
        }
    }

    #[tokio::test]
    async fn fold_rejects_a_stream_that_ends_without_done() {
        let err = ChatTurn::fold(stream_of(ok(vec![ChatDelta::Text("hi".into())])))
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Llm(LlmError::Protocol(_))), "{err}");
        assert!(
            err.to_string().contains("before the turn completed"),
            "{err}"
        );
    }

    #[tokio::test]
    async fn fold_propagates_stream_errors() {
        let err = ChatTurn::fold(stream_of(vec![
            Ok(ChatDelta::Text("partial".into())),
            Err(Error::Llm(LlmError::Transport("reset".into()))),
        ]))
        .await
        .unwrap_err();
        assert!(err.to_string().contains("reset"), "{err}");
    }
}
