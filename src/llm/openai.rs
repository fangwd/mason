//! OpenAI-compatible request encoding, response decoding, and HTTP transport.

use std::pin::Pin;
use std::result::Result as StdResult;

use async_trait::async_trait;
use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::Error;
use crate::error::Result;
use crate::llm::Vendor;
use crate::llm::error::{LlmError, retry_after_ms};
use crate::llm::message::{ChatDelta, ChatRequest, Message, ToolSpec};

/// A response body as a byte stream, normalized to [`LlmError`].
pub(crate) type BodyStream = Pin<Box<dyn Stream<Item = StdResult<Bytes, LlmError>> + Send>>;

/// One HTTP attempt's response, transport-independent.
pub(crate) struct WireResponse {
    pub status: u16,
    /// A parsed `Retry-After` header in milliseconds, when present.
    pub retry_after_ms: Option<u64>,
    pub body: BodyStream,
}

/// Sends request bodies to vendors. Implemented by reqwest in production and
/// by scripted doubles in tests.
#[async_trait]
pub(crate) trait Transport: Send + Sync {
    /// POSTs `body` to the vendor's `chat/completions` endpoint.
    async fn send(&self, vendor: &Vendor, body: Vec<u8>) -> StdResult<WireResponse, LlmError>;
}

/// The production transport over reqwest with rustls.
pub(crate) struct ReqwestTransport {
    client: reqwest::Client,
}

impl ReqwestTransport {
    pub(crate) fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl Transport for ReqwestTransport {
    async fn send(&self, vendor: &Vendor, body: Vec<u8>) -> StdResult<WireResponse, LlmError> {
        let mut request = self
            .client
            .post(format!("{}/chat/completions", vendor.url))
            .headers(vendor.headers.clone())
            .header(reqwest::header::CONTENT_TYPE, "application/json");
        if let Some(key) = &vendor.api_key {
            request = request.bearer_auth(key);
        }
        let response = request
            .body(body)
            .send()
            .await
            .map_err(|err| LlmError::Transport(err.to_string()))?;
        let status = response.status().as_u16();
        let retry_after_ms = response
            .headers()
            .get(reqwest::header::RETRY_AFTER)
            .and_then(|value| value.to_str().ok())
            .and_then(retry_after_ms);
        let body = response
            .bytes_stream()
            .map(|chunk| chunk.map_err(|err| LlmError::Transport(err.to_string())))
            .boxed();
        Ok(WireResponse {
            status,
            retry_after_ms,
            body,
        })
    }
}

/// A complete response body is bounded; longer ones are a broken provider.
pub(crate) const MAX_COMPLETE_BODY: usize = 32 * 1024 * 1024;

/// Encodes a chat request; only streaming vendors request streamed usage reports.
/// Request fields mason sets itself, which `extra_body` cannot override.
pub(crate) const RESERVED_FIELDS: [&str; 5] =
    ["model", "messages", "tools", "stream", "stream_options"];

pub(crate) fn request_body(vendor: &Vendor, request: &ChatRequest) -> Result<Vec<u8>> {
    #[derive(Serialize)]
    struct StreamOptions {
        include_usage: bool,
    }
    #[derive(Serialize)]
    struct WireRequest<'a> {
        model: &'a str,
        messages: &'a [Message],
        #[serde(skip_serializing_if = "<[_]>::is_empty")]
        tools: &'a [ToolSpec],
        #[serde(skip_serializing_if = "Option::is_none")]
        stream: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        stream_options: Option<StreamOptions>,
    }
    let wire = WireRequest {
        model: &vendor.model,
        messages: &request.messages,
        tools: &request.tools,
        stream: vendor.stream.then_some(true),
        stream_options: vendor.stream.then_some(StreamOptions {
            include_usage: true,
        }),
    };
    let serialize_error = |err: serde_json::Error| {
        Error::Llm(LlmError::Protocol(format!(
            "cannot serialize the request: {err}"
        )))
    };
    let mut body = serde_json::to_value(&wire).map_err(serialize_error)?;
    if let serde_json::Value::Object(fields) = &mut body {
        fields.extend(vendor.extra_body.clone());
    }
    serde_json::to_vec(&body).map_err(serialize_error)
}

/// Reads bounded error text under the turn's cancellation and deadline.
pub(crate) async fn read_error_body(
    mut body: BodyStream,
    token: &CancellationToken,
    turn_deadline: tokio::time::Instant,
    request_deadline: tokio::time::Instant,
    idle_timeout: std::time::Duration,
) -> String {
    const CAP: usize = 64 * 1024;
    let mut buf: Vec<u8> = Vec::new();
    while buf.len() < CAP {
        let chunk = tokio::select! {
            _ = token.cancelled() => break,
            _ = tokio::time::sleep_until(turn_deadline) => break,
            _ = tokio::time::sleep_until(request_deadline) => break,
            _ = tokio::time::sleep(idle_timeout) => break,
            item = body.next() => match item {
                Some(Ok(chunk)) => chunk,
                _ => break,
            },
        };
        buf.extend_from_slice(&chunk[..chunk.len().min(CAP - buf.len())]);
    }
    String::from_utf8_lossy(&buf).chars().take(500).collect()
}

/// What one SSE `data` payload contributed to the turn.
#[derive(Debug)]
pub(crate) enum ChunkEvent {
    /// Decoded deltas plus the choice's finish reason, when this chunk
    /// completed the choice.
    Decoded {
        deltas: Vec<ChatDelta>,
        finish_reason: Option<String>,
    },
    /// The `data: [DONE]` sentinel.
    StreamDone,
}

/// Decodes OpenAI chunk data; malformed payloads and provider errors are protocol
/// failures.
pub(crate) fn decode_chunk(data: &str) -> Result<ChunkEvent> {
    if data.trim() == "[DONE]" {
        return Ok(ChunkEvent::StreamDone);
    }
    let chunk: WireChunk = serde_json::from_str(data).map_err(|err| {
        Error::Llm(LlmError::Protocol(format!(
            "cannot decode a response chunk ({err}): {}",
            preview(data)
        )))
    })?;
    if let Some(error) = &chunk.error {
        return Err(Error::Llm(
            error.classify("the provider reported an error mid-stream"),
        ));
    }
    let mut deltas = Vec::new();
    // n > 1 is not supported; the first choice is the turn.
    if let Some(choice) = chunk.choices.first()
        && let Some(delta) = &choice.delta
    {
        if let Some(text) = &delta.content
            && !text.is_empty()
        {
            deltas.push(ChatDelta::Text(text.clone()));
        }
        if let Some(reasoning) = delta
            .reasoning_content
            .as_ref()
            .or(delta.reasoning.as_ref())
            && !reasoning.is_empty()
        {
            deltas.push(ChatDelta::Reasoning(reasoning.clone()));
        }
        for (position, call) in delta.tool_calls.iter().flatten().enumerate() {
            deltas.push(ChatDelta::ToolCall {
                index: call.index.unwrap_or(position as u32),
                id: call.id.clone(),
                name: call.function.as_ref().and_then(|f| f.name.clone()),
                arguments: call
                    .function
                    .as_ref()
                    .and_then(|f| f.arguments.clone())
                    .unwrap_or_default(),
            });
        }
    }
    if let Some(usage) = chunk.usage {
        deltas.push(usage.delta());
    }
    let finish_reason = chunk.choices.first().and_then(|c| c.finish_reason.clone());
    Ok(ChunkEvent::Decoded {
        deltas,
        finish_reason,
    })
}

/// Converts a single-shot response to the same delta sequence as a streamed turn.
pub(crate) fn adapt_complete(body: &[u8]) -> Result<Vec<ChatDelta>> {
    let wire: WireComplete = serde_json::from_slice(body).map_err(|err| {
        Error::Llm(LlmError::Protocol(format!(
            "cannot decode the response ({err}): {}",
            String::from_utf8_lossy(body)
                .chars()
                .take(500)
                .collect::<String>()
        )))
    })?;
    if let Some(error) = &wire.error {
        return Err(Error::Llm(error.classify("the provider reported an error")));
    }
    let choice = wire.choices.first();
    let Some(message) = choice.and_then(|choice| choice.message.as_ref()) else {
        return Err(Error::Llm(LlmError::Protocol(format!(
            "no message in the response: {}",
            String::from_utf8_lossy(body)
                .chars()
                .take(500)
                .collect::<String>()
        ))));
    };
    let mut deltas = Vec::new();
    if let Some(text) = &message.content
        && !text.is_empty()
    {
        deltas.push(ChatDelta::Text(text.clone()));
    }
    if let Some(reasoning) = message
        .reasoning_content
        .as_ref()
        .or(message.reasoning.as_ref())
        && !reasoning.is_empty()
    {
        deltas.push(ChatDelta::Reasoning(reasoning.clone()));
    }
    for (index, call) in message.tool_calls.iter().flatten().enumerate() {
        deltas.push(ChatDelta::ToolCall {
            index: index as u32,
            id: Some(call.id.clone()),
            name: Some(call.function.name.clone()),
            arguments: call.function.arguments.clone(),
        });
    }
    // A missing finish reason is an incomplete turn, not success.
    let Some(finish_reason) = choice.and_then(|choice| choice.finish_reason.clone()) else {
        return Err(Error::Llm(LlmError::Protocol(
            "the response did not include a finish reason".to_owned(),
        )));
    };
    if let Some(usage) = wire.usage {
        deltas.push(usage.delta());
    }
    deltas.push(ChatDelta::Done {
        finish_reason: Some(finish_reason),
    });
    Ok(deltas)
}

/// Returns a bounded payload excerpt for diagnostics.
fn preview(data: &str) -> String {
    data.chars().take(200).collect()
}

#[derive(Debug, Deserialize)]
struct WireChunk {
    #[serde(default)]
    choices: Vec<WireChunkChoice>,
    #[serde(default)]
    usage: Option<WireUsage>,
    #[serde(default)]
    error: Option<WireErrorBody>,
}

#[derive(Debug, Deserialize)]
struct WireChunkChoice {
    #[serde(default)]
    delta: Option<WireDelta>,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct WireDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    /// OpenRouter's and newer vLLM's name for `reasoning_content`.
    #[serde(default)]
    reasoning: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<WireDeltaToolCall>>,
}

#[derive(Debug, Deserialize)]
struct WireDeltaToolCall {
    /// Some providers omit it; the call's position in the chunk stands in.
    #[serde(default)]
    index: Option<u32>,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<WireDeltaFunction>,
}

#[derive(Debug, Deserialize)]
struct WireDeltaFunction {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct WireComplete {
    #[serde(default)]
    choices: Vec<WireCompleteChoice>,
    #[serde(default)]
    usage: Option<WireUsage>,
    #[serde(default)]
    error: Option<WireErrorBody>,
}

#[derive(Debug, Deserialize)]
struct WireCompleteChoice {
    #[serde(default)]
    message: Option<WireMessage>,
    #[serde(default)]
    finish_reason: Option<String>,
}

/// A single-shot reply message, more lenient than [`Message`]: `type` may be
/// missing and reasoning may arrive as `reasoning`.
#[derive(Debug, Deserialize)]
struct WireMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    reasoning: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<WireToolCall>>,
}

#[derive(Debug, Deserialize)]
struct WireToolCall {
    id: String,
    function: crate::llm::message::FunctionCall,
}

/// Counts may be absent or `null`.
#[derive(Debug, Clone, Copy, Deserialize)]
struct WireUsage {
    #[serde(default)]
    prompt_tokens: Option<u32>,
    #[serde(default)]
    completion_tokens: Option<u32>,
}

impl WireUsage {
    fn delta(self) -> ChatDelta {
        ChatDelta::Usage {
            prompt_tokens: self.prompt_tokens.unwrap_or(0),
            completion_tokens: self.completion_tokens.unwrap_or(0),
        }
    }
}

#[derive(Debug, Deserialize)]
struct WireErrorBody {
    #[serde(default)]
    message: Option<String>,
    #[serde(default)]
    code: Option<String>,
    #[serde(default)]
    r#type: Option<String>,
}

impl WireErrorBody {
    /// Renders every field the overflow patterns may need to see.
    fn render(&self) -> String {
        [
            self.message.as_deref(),
            self.code.as_deref(),
            self.r#type.as_deref(),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join(" ")
    }

    /// A transient provider error (overload, rate limit, server error) is a
    /// retryable transport failure; anything else, including a context
    /// overflow, stays a protocol error.
    fn classify(&self, context: &str) -> LlmError {
        const TRANSIENT: [&str; 9] = [
            "server_error",
            "server error",
            "overloaded",
            "rate_limit",
            "rate limit",
            "timeout",
            "timed out",
            "unavailable",
            "try again",
        ];
        let rendered = self.render();
        let message = format!("{context}: {rendered}");
        let lower = rendered.to_ascii_lowercase();
        let transient_code = self
            .code
            .as_deref()
            .and_then(|code| code.parse::<u16>().ok())
            .is_some_and(|code| code == 429 || code >= 500);
        if !crate::llm::error::is_context_overflow(&rendered)
            && (transient_code || TRANSIENT.iter().any(|word| lower.contains(word)))
        {
            LlmError::Transport(message)
        } else {
            LlmError::Protocol(message)
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::llm::message::{FunctionCall, ToolCall};

    fn vendor(stream: bool) -> Vendor {
        Vendor {
            name: "test".into(),
            url: "https://llm.test".into(),
            model: "test-model".into(),
            api_key: Some("key".into()),
            context_tokens: 128_000,
            extra_body: Default::default(),
            headers: Default::default(),
            stream,
        }
    }

    fn deltas_of(event: ChunkEvent) -> (Vec<ChatDelta>, Option<String>) {
        match event {
            ChunkEvent::Decoded {
                deltas,
                finish_reason,
            } => (deltas, finish_reason),
            ChunkEvent::StreamDone => panic!("expected a decoded chunk"),
        }
    }

    #[test]
    fn request_body_omits_tools_when_empty_and_streams_by_default() {
        let request = ChatRequest {
            messages: vec![Message::user("hi")],
            tools: vec![],
        };
        let body: serde_json::Value =
            serde_json::from_slice(&request_body(&vendor(true), &request).unwrap()).unwrap();
        assert_eq!(body["model"], "test-model");
        assert_eq!(body["messages"], json!([{"role": "user", "content": "hi"}]));
        assert_eq!(body["stream"], true);
        assert_eq!(body["stream_options"], json!({"include_usage": true}));
        assert!(body.get("tools").is_none());
    }

    #[test]
    fn request_body_honours_single_shot_and_extra_body() {
        let mut single = vendor(false);
        single.extra_body = json!({"enable_thinking": false, "max_tokens": 32_000})
            .as_object()
            .unwrap()
            .clone();
        let request = ChatRequest {
            messages: vec![],
            tools: vec![ToolSpec::function(
                "read",
                "Read",
                json!({"type": "object"}),
            )],
        };
        let body: serde_json::Value =
            serde_json::from_slice(&request_body(&single, &request).unwrap()).unwrap();
        assert!(body.get("stream").is_none());
        assert!(body.get("stream_options").is_none());
        assert_eq!(body["enable_thinking"], false);
        assert_eq!(body["max_tokens"], 32_000);
        assert_eq!(body["tools"][0]["function"]["name"], "read");
    }

    #[test]
    fn decode_chunk_extracts_text_reasoning_and_tool_fragments() {
        let data = json!({
            "choices": [{
                "delta": {
                    "content": "Hel",
                    "reasoning_content": "hmm",
                    "tool_calls": [
                        {"index": 0, "id": "c0", "function": {"name": "read", "arguments": ""}},
                        {"index": 1, "function": {"arguments": "{\"a\""}}
                    ]
                },
                "finish_reason": null
            }]
        })
        .to_string();
        let (deltas, finish) = deltas_of(decode_chunk(&data).unwrap());
        assert_eq!(finish, None);
        assert_eq!(
            deltas,
            vec![
                ChatDelta::Text("Hel".into()),
                ChatDelta::Reasoning("hmm".into()),
                ChatDelta::ToolCall {
                    index: 0,
                    id: Some("c0".into()),
                    name: Some("read".into()),
                    arguments: String::new()
                },
                ChatDelta::ToolCall {
                    index: 1,
                    id: None,
                    name: None,
                    arguments: "{\"a\"".into()
                },
            ]
        );
    }

    #[test]
    fn decode_chunk_skips_empty_fragments_and_records_finish_and_usage() {
        let role_only =
            json!({"choices": [{"delta": {"role": "assistant", "content": ""}}]}).to_string();
        let (deltas, finish) = deltas_of(decode_chunk(&role_only).unwrap());
        assert!(deltas.is_empty());
        assert_eq!(finish, None);

        let finished = json!({"choices": [{"delta": {}, "finish_reason": "stop"}]}).to_string();
        let (_, finish) = deltas_of(decode_chunk(&finished).unwrap());
        assert_eq!(finish.as_deref(), Some("stop"));

        // Usage typically arrives after the finish chunk, on an empty choice list.
        let usage = json!({"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 7}})
            .to_string();
        let (deltas, _) = deltas_of(decode_chunk(&usage).unwrap());
        assert_eq!(
            deltas,
            vec![ChatDelta::Usage {
                prompt_tokens: 11,
                completion_tokens: 7
            }]
        );
    }

    #[test]
    fn decoding_tolerates_common_provider_variations() {
        // No tool-call index, reasoning under `reasoning`, null usage counts.
        let data = json!({
            "choices": [{"delta": {
                "reasoning": "think",
                "tool_calls": [
                    {"id": "a", "function": {"name": "read", "arguments": "{}"}},
                    {"id": "b", "function": {"name": "glob", "arguments": "{}"}}
                ]
            }}],
            "usage": {"prompt_tokens": 5, "completion_tokens": null}
        })
        .to_string();
        let (deltas, _) = deltas_of(decode_chunk(&data).unwrap());
        assert_eq!(deltas[0], ChatDelta::Reasoning("think".into()));
        let indices: Vec<u32> = deltas
            .iter()
            .filter_map(|delta| match delta {
                ChatDelta::ToolCall { index, .. } => Some(*index),
                _ => None,
            })
            .collect();
        assert_eq!(indices, [0, 1]);
        assert_eq!(
            deltas.last(),
            Some(&ChatDelta::Usage {
                prompt_tokens: 5,
                completion_tokens: 0
            })
        );

        // A single-shot tool call without `type`.
        let body = json!({
            "choices": [{
                "message": {"content": null, "tool_calls": [
                    {"id": "c", "function": {"name": "read", "arguments": "{}"}}
                ]},
                "finish_reason": "tool_calls"
            }]
        })
        .to_string();
        let deltas = adapt_complete(body.as_bytes()).unwrap();
        assert!(
            matches!(&deltas[0], ChatDelta::ToolCall { name: Some(name), .. } if name == "read")
        );

        assert!(crate::llm::error::is_context_overflow(
            "InvalidParameter: Range of input length should be [1, 98304]"
        ));
    }

    #[test]
    fn error_events_are_transient_unless_they_report_an_overflow() {
        let classify =
            |error: serde_json::Value| match decode_chunk(&json!({"error": error}).to_string()) {
                Err(Error::Llm(err)) => err,
                other => panic!("expected an error, got {other:?}"),
            };
        assert!(classify(json!({"type": "overloaded_error"})).is_retryable());
        assert!(classify(json!({"code": "503", "message": "busy"})).is_retryable());
        assert!(!classify(json!({"type": "invalid_request_error"})).is_retryable());
        let overflow = classify(json!({
            "message": "maximum context length is 8192 tokens; try again with less",
            "type": "invalid_request_error"
        }));
        assert!(!overflow.is_retryable() && overflow.is_context_overflow());
    }

    #[test]
    fn decode_chunk_handles_the_done_sentinel() {
        assert!(decode_chunk("data placeholder").is_err());
        assert!(matches!(decode_chunk("[DONE]"), Ok(ChunkEvent::StreamDone)));
        assert!(matches!(
            decode_chunk(" [DONE] "),
            Ok(ChunkEvent::StreamDone)
        ));
    }

    #[test]
    fn decode_chunk_reports_provider_errors_and_malformed_data_without_panic() {
        let err = decode_chunk(
            &json!({"error": {"message": "boom", "code": "server_error"}}).to_string(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("boom"), "{err}");
        assert!(err.to_string().contains("server_error"), "{err}");

        for bad in ["not json at all", "{\"choices\": 5}"] {
            let err = decode_chunk(bad).unwrap_err();
            assert!(
                err.to_string().contains("cannot decode a response chunk"),
                "{bad}: {err}"
            );
        }
    }

    #[test]
    fn adapt_complete_produces_the_streaming_delta_sequence() {
        let body = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "reasoning_content": "hmm",
                    "tool_calls": [{
                        "id": "c0",
                        "type": "function",
                        "function": {"name": "read", "arguments": "{\"a\":1}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7}
        })
        .to_string();
        assert_eq!(
            adapt_complete(body.as_bytes()).unwrap(),
            vec![
                ChatDelta::Reasoning("hmm".into()),
                ChatDelta::ToolCall {
                    index: 0,
                    id: Some("c0".into()),
                    name: Some("read".into()),
                    arguments: "{\"a\":1}".into()
                },
                ChatDelta::Usage {
                    prompt_tokens: 11,
                    completion_tokens: 7
                },
                ChatDelta::Done {
                    finish_reason: Some("tool_calls".into())
                },
            ]
        );
    }

    #[test]
    fn adapt_complete_reports_missing_messages_and_errors() {
        let err = adapt_complete(b"{}").unwrap_err();
        assert!(
            err.to_string().contains("no message in the response"),
            "{err}"
        );

        let err = adapt_complete(b"<html>gateway</html>").unwrap_err();
        assert!(
            err.to_string().contains("cannot decode the response"),
            "{err}"
        );

        let err = adapt_complete(
            &json!({"error": {"message": "bad key"}})
                .to_string()
                .into_bytes(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("bad key"), "{err}");
    }

    #[test]
    fn adapt_complete_keeps_null_content_tool_turns() {
        let body = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{"id": "c0", "type": "function", "function": {"name": "t", "arguments": "{}"}}]
                },
                "finish_reason": "tool_calls"
            }]
        })
        .to_string();
        let deltas = adapt_complete(body.as_bytes()).unwrap();
        assert!(
            !deltas.iter().any(|d| matches!(d, ChatDelta::Text(_))),
            "no text delta for null content"
        );
        assert!(matches!(deltas[0], ChatDelta::ToolCall { .. }));
        assert_eq!(
            deltas[0],
            ChatDelta::ToolCall {
                index: 0,
                id: Some("c0".into()),
                name: Some("t".into()),
                arguments: "{}".into()
            }
        );
        let call = ToolCall {
            id: "c0".into(),
            kind: "function".into(),
            function: FunctionCall {
                name: "t".into(),
                arguments: "{}".into(),
            },
        };
        assert_eq!(call, ToolCall::new("c0", "t", "{}"));
    }

    #[test]
    fn adapt_complete_requires_a_finish_reason() {
        let body = json!({
            "choices": [{"message": {"role": "assistant", "content": "hi"}}]
        })
        .to_string();
        let err = adapt_complete(body.as_bytes()).unwrap_err();
        assert!(
            err.to_string().contains("did not include a finish reason"),
            "{err}"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn stalled_error_bodies_obey_the_idle_timeout() {
        let started = tokio::time::Instant::now();
        let body: BodyStream = Box::pin(futures_util::stream::pending());
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(61),
            read_error_body(
                body,
                &CancellationToken::new(),
                started + super::super::TURN_BUDGET,
                started + super::super::REQUEST_BUDGET,
                super::super::IDLE_TIMEOUT,
            ),
        )
        .await
        .expect("the error body has an idle deadline");
        assert!(result.is_empty());
        assert_eq!(started.elapsed(), super::super::IDLE_TIMEOUT);
    }

    #[tokio::test(start_paused = true)]
    async fn error_bodies_obey_the_remaining_request_budget() {
        let started = tokio::time::Instant::now();
        let body: BodyStream = Box::pin(futures_util::stream::pending());
        let result = read_error_body(
            body,
            &CancellationToken::new(),
            started + super::super::TURN_BUDGET,
            started + std::time::Duration::from_secs(5),
            super::super::IDLE_TIMEOUT,
        )
        .await;
        assert!(result.is_empty());
        assert_eq!(started.elapsed(), std::time::Duration::from_secs(5));
    }
}
