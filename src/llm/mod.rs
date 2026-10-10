//! Streaming model interface and OpenAI-compatible vendor failover. Retry, failover
//! and the turn and attempt deadlines stop after the first forwarded delta; from
//! then on the idle timeout and cancellation bound the stream.

use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use futures_util::{Stream, StreamExt, stream};
use tokio::time::{Instant, Sleep, sleep, sleep_until};

use crate::Error;
use crate::error::Result;
use crate::llm::error::retry_delay;
use crate::llm::openai::{BodyStream, ChunkEvent, ReqwestTransport, Transport};
use crate::llm::sse::SseDecoder;

pub mod error;
pub mod message;
mod openai;
mod sse;

pub use error::{LlmError, is_context_overflow};
pub use message::{
    ChatDelta, ChatRequest, ChatTurn, FunctionCall, Message, Role, ToolCall, ToolSpec,
    TurnAccumulator, Usage,
};
pub use tokio_util::sync::CancellationToken;

/// Attempts per vendor before failing over.
const MAX_ATTEMPTS: usize = 3;
/// Budget of a single HTTP attempt until its first delta is forwarded.
const REQUEST_BUDGET: Duration = Duration::from_secs(120);
/// Budget spanning all attempts and all vendors until the first forwarded delta.
const TURN_BUDGET: Duration = Duration::from_secs(180);
/// A stream that produced no chunk for this long is considered stalled.
const IDLE_TIMEOUT: Duration = Duration::from_secs(60);

/// Configurable timeouts for slow local inference and remote endpoints.
#[derive(Debug, Clone, Copy)]
pub struct ModelTimeouts {
    pub request: Duration,
    pub turn: Duration,
    pub idle: Duration,
}

impl Default for ModelTimeouts {
    fn default() -> Self {
        Self {
            request: REQUEST_BUDGET,
            turn: TURN_BUDGET,
            idle: IDLE_TIMEOUT,
        }
    }
}

impl ModelTimeouts {
    pub fn validate(self) -> Result<Self> {
        if [self.request, self.turn, self.idle].contains(&Duration::ZERO) {
            return Err(Error::Config("model timeouts must be positive".into()));
        }
        Ok(self)
    }
}

/// One OpenAI-compatible endpoint. Vendors are tried in order.
#[derive(Debug, Clone, PartialEq)]
pub struct Vendor {
    /// The name from `AI_VENDORS`, used in diagnostics.
    pub name: String,
    /// Base URL without trailing slashes; requests go to
    /// `{url}/chat/completions`.
    pub url: String,
    /// The model id to request.
    pub model: String,
    /// Bearer token; without one no `Authorization` header is sent (local
    /// servers).
    pub api_key: Option<String>,
    /// Context capacity in tokens (`AI_<NAME>_CONTEXT_TOKENS`, default
    /// 128_000).
    pub context_tokens: usize,
    /// Fields merged into every request (`AI_<NAME>_EXTRA_BODY`, a JSON
    /// object), e.g. `max_tokens`, `temperature` or `enable_thinking`.
    pub extra_body: serde_json::Map<String, serde_json::Value>,
    /// Headers sent with every request (`AI_<NAME>_HEADERS`, a JSON object of
    /// strings), e.g. Azure's `api-key`.
    pub headers: reqwest::header::HeaderMap,
    /// `false` (`AI_<NAME>_STREAM=false`) falls back to single-shot
    /// responses adapted into the same delta stream.
    pub stream: bool,
}

/// The configured vendors, in failover order.
#[derive(Debug, Clone, Default)]
pub struct Vendors(Vec<Vendor>);

impl Vendors {
    /// Reads `AI_VENDORS` and the `AI_<NAME>_*` variables from the process environment.
    pub fn from_env() -> Result<Self> {
        Self::from_lookup(|key| std::env::var(key).ok())
    }

    /// The evaluator's vendors: `AI_EVALUATOR_VENDORS` names vendors configured by
    /// the same `AI_<NAME>_*` variables; `None` when it is unset.
    pub fn evaluator_from_env() -> Result<Option<Self>> {
        Self::evaluator_from_lookup(|key| std::env::var(key).ok())
    }

    /// The [`evaluator_from_env`](Self::evaluator_from_env) contract against an
    /// injectable lookup.
    pub fn evaluator_from_lookup(get: impl Fn(&str) -> Option<String>) -> Result<Option<Self>> {
        if get("AI_EVALUATOR_VENDORS").is_none_or(|names| names.trim().is_empty()) {
            return Ok(None);
        }
        Self::from_lookup(|key| match key {
            "AI_VENDORS" => get("AI_EVALUATOR_VENDORS"),
            _ => get(key),
        })
        .map(Some)
    }

    /// The [`from_env`](Self::from_env) contract against an injectable
    /// lookup, for tests and custom configuration sources.
    pub fn from_lookup(get: impl Fn(&str) -> Option<String>) -> Result<Self> {
        const DEFAULT_CONTEXT_TOKENS: usize = 128_000;
        let raw = get("AI_VENDORS").unwrap_or_default();
        let names: Vec<String> = raw
            .split(',')
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .map(str::to_owned)
            .collect();
        if names.is_empty() {
            return Err(Error::Config("AI_VENDORS is not set".to_owned()));
        }
        let mut vendors = Vec::with_capacity(names.len());
        for name in names {
            if !name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
            {
                return Err(Error::Config(format!(
                    "AI_VENDORS name `{name}` must use only letters, digits and _, since it names AI_<NAME>_* variables"
                )));
            }
            let prefix = format!("AI_{}_", name.to_uppercase());
            let required = |suffix: &str| -> Result<String> {
                let key = format!("{prefix}{suffix}");
                get(&key)
                    .filter(|value| !value.trim().is_empty())
                    .ok_or_else(|| Error::Config(format!("{key} is required")))
            };
            let url = required("URL")?.trim_end_matches('/').to_owned();
            if url.is_empty() {
                return Err(Error::Config(format!("{prefix}URL must not be empty")));
            }
            let model = required("MODEL")?;
            let api_key = get(&format!("{prefix}API_KEY")).filter(|key| !key.trim().is_empty());
            let context_tokens = match get(&format!("{prefix}CONTEXT_TOKENS")) {
                Some(raw) if !raw.trim().is_empty() => raw
                    .trim()
                    .parse::<usize>()
                    .ok()
                    .filter(|tokens| *tokens > 0)
                    .ok_or_else(|| {
                        Error::Config(format!("{prefix}CONTEXT_TOKENS must be a positive integer"))
                    })?,
                _ => DEFAULT_CONTEXT_TOKENS,
            };
            let stream = optional_bool(&get, &prefix, "STREAM")?.unwrap_or(true);
            let extra_body = json_object(&get, &prefix, "EXTRA_BODY")?;
            if let Some(reserved) = extra_body
                .keys()
                .find(|key| openai::RESERVED_FIELDS.contains(&key.as_str()))
            {
                return Err(Error::Config(format!(
                    "{prefix}EXTRA_BODY cannot set `{reserved}`, which mason sets itself"
                )));
            }
            let mut headers = reqwest::header::HeaderMap::new();
            for (key, value) in json_object(&get, &prefix, "HEADERS")? {
                let invalid =
                    || Error::Config(format!("{prefix}HEADERS has an invalid header `{key}`"));
                let serde_json::Value::String(value) = value else {
                    return Err(invalid());
                };
                let name = reqwest::header::HeaderName::from_bytes(key.as_bytes())
                    .map_err(|_| invalid())?;
                let mut value =
                    reqwest::header::HeaderValue::from_str(&value).map_err(|_| invalid())?;
                value.set_sensitive(true);
                headers.insert(name, value);
            }
            vendors.push(Vendor {
                name,
                url,
                model,
                api_key,
                context_tokens,
                extra_body,
                headers,
                stream,
            });
        }
        Ok(Self(vendors))
    }

    /// The vendors in failover order.
    pub fn as_slice(&self) -> &[Vendor] {
        &self.0
    }

    /// Smallest vendor context capacity; requests must fit every failover target.
    pub fn context_tokens(&self) -> usize {
        self.0
            .iter()
            .map(|vendor| vendor.context_tokens)
            .min()
            .unwrap_or(0)
    }
}

/// An optional JSON object variable; its parse error never echoes the value.
fn json_object(
    get: &impl Fn(&str) -> Option<String>,
    prefix: &str,
    suffix: &str,
) -> Result<serde_json::Map<String, serde_json::Value>> {
    let key = format!("{prefix}{suffix}");
    match get(&key).filter(|raw| !raw.trim().is_empty()) {
        None => Ok(serde_json::Map::new()),
        Some(raw) => serde_json::from_str(&raw)
            .map_err(|_| Error::Config(format!("{key} must be a JSON object"))),
    }
}

fn optional_bool(
    get: &impl Fn(&str) -> Option<String>,
    prefix: &str,
    suffix: &str,
) -> Result<Option<bool>> {
    let key = format!("{prefix}{suffix}");
    match get(&key) {
        None => Ok(None),
        Some(raw) => match raw.trim().to_ascii_lowercase().as_str() {
            "" => Ok(None),
            "true" => Ok(Some(true)),
            "false" => Ok(Some(false)),
            _ => Err(Error::Config(format!("{key} must be true or false"))),
        },
    }
}

/// A model turn as a delta stream: items in arrival order, ending with
/// [`ChatDelta::Done`]. Errors are terminal items; the stream ends after
/// them.
pub type ChatStream = Pin<Box<dyn Stream<Item = Result<ChatDelta>> + Send>>;

/// Streaming model backend with its own context capacity; [`VendorModel`] provides
/// OpenAI-compatible failover.
#[async_trait]
pub trait ChatModel: Send + Sync {
    /// The context capacity in tokens that memory management plans against.
    fn context_tokens(&self) -> usize;

    /// Runs one model turn. The returned stream owns a clone of `token` and
    /// retains its deadlines after this call returns; cancelling or dropping
    /// the stream closes the underlying response body.
    async fn stream(&self, request: ChatRequest, token: &CancellationToken) -> Result<ChatStream>;
}

/// The default [`ChatModel`]: OpenAI-compatible vendors with retry and failover.
pub struct VendorModel {
    vendors: Arc<[Vendor]>,
    transport: Arc<dyn Transport>,
    timeouts: ModelTimeouts,
}

impl VendorModel {
    /// A model over the given vendors, speaking real HTTP.
    pub fn new(vendors: Vendors) -> Self {
        Self::with_transport(vendors, Arc::new(ReqwestTransport::new()))
    }

    pub fn with_timeouts(mut self, timeouts: ModelTimeouts) -> Result<Self> {
        self.timeouts = timeouts.validate()?;
        Ok(self)
    }

    /// A model over a test transport.
    pub(crate) fn with_transport(vendors: Vendors, transport: Arc<dyn Transport>) -> Self {
        Self {
            vendors: vendors.0.into(),
            transport,
            timeouts: ModelTimeouts::default(),
        }
    }
}

impl std::fmt::Debug for VendorModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VendorModel")
            .field(
                "vendors",
                &self.vendors.iter().map(|v| &v.name).collect::<Vec<_>>(),
            )
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl ChatModel for VendorModel {
    fn context_tokens(&self) -> usize {
        self.vendors
            .iter()
            .map(|vendor| vendor.context_tokens)
            .min()
            .unwrap_or(0)
    }

    async fn stream(&self, request: ChatRequest, token: &CancellationToken) -> Result<ChatStream> {
        if self.vendors.is_empty() {
            return Err(Error::Config("no LLM vendors are configured".to_owned()));
        }
        let state = TurnState {
            vendors: self.vendors.clone(),
            request,
            transport: self.transport.clone(),
            token: token.clone(),
            turn_deadline: Instant::now() + self.timeouts.turn,
            timeouts: self.timeouts,
            request_deadline: Instant::now(),
            vendor: 0,
            attempt: 1,
            phase: Phase::Request,
            pending: VecDeque::new(),
            forwarded: false,
            finish_reason: None,
            completed: false,
            done_queued: false,
            premature_done: false,
            last_error: None,
            sse: SseDecoder::default(),
        };
        Ok(stream::unfold(state, |mut state| async move {
            state.step().await.map(|item| (item, state))
        })
        .boxed())
    }
}

enum Phase {
    Request,
    /// Wait out a backoff, then retry the given vendor and attempt.
    Backoff {
        until: Pin<Box<Sleep>>,
        vendor: usize,
        attempt: usize,
    },
    Streaming {
        body: BodyStream,
    },
    Collecting {
        body: BodyStream,
        buf: Vec<u8>,
    },
    /// Drain pending deltas before ending the stream.
    Finished,
}

enum WaitOutcome {
    Chunk(Bytes),
    Eof,
    /// The attempt failed in a class that is retryable while nothing has
    /// been forwarded.
    Attempt(LlmError),
    /// Cancellation or the turn deadline: the turn is over regardless.
    Terminal(Result<ChatDelta>),
}

/// One turn's state machine: attempts, deadlines, framing and completion.
struct TurnState {
    vendors: Arc<[Vendor]>,
    request: ChatRequest,
    transport: Arc<dyn Transport>,
    token: CancellationToken,
    turn_deadline: Instant,
    timeouts: ModelTimeouts,
    request_deadline: Instant,
    vendor: usize,
    attempt: usize,
    phase: Phase,
    /// Decoded deltas waiting to be yielded, in wire order.
    pending: VecDeque<ChatDelta>,
    /// Whether any item reached the consumer: gates retry and failover.
    forwarded: bool,
    finish_reason: Option<String>,
    /// The choice received a non-null finish reason.
    completed: bool,
    /// `Done` was queued; guards emitting it exactly once.
    done_queued: bool,
    /// A sentinel without a finish reason is an incomplete turn.
    premature_done: bool,
    /// The latest failed attempt, reported if the turn deadline ends the retries.
    last_error: Option<String>,
    sse: SseDecoder,
}

impl TurnState {
    /// Produces the next stream item, driving attempts and decoding until
    /// one is ready. `None` ends the stream.
    async fn step(&mut self) -> Option<Result<ChatDelta>> {
        let item = self.advance().await;
        if matches!(item, Some(Err(_))) {
            // Errors are terminal: deltas decoded before one never follow it.
            self.pending.clear();
            self.phase = Phase::Finished;
        }
        item
    }

    async fn advance(&mut self) -> Option<Result<ChatDelta>> {
        loop {
            if let Some(delta) = self.pending.pop_front() {
                // Usage is bookkeeping: a retry after it duplicates no output.
                if !matches!(delta, ChatDelta::Usage { .. }) {
                    self.forwarded = true;
                }
                if matches!(delta, ChatDelta::Done { .. }) {
                    self.phase = Phase::Finished;
                }
                return Some(Ok(delta));
            }
            match std::mem::replace(&mut self.phase, Phase::Finished) {
                Phase::Finished => return None,
                Phase::Request => {
                    if let Some(item) = self.request().await {
                        return Some(item);
                    }
                }
                Phase::Backoff {
                    until,
                    vendor,
                    attempt,
                } => {
                    tokio::select! {
                        _ = self.token.cancelled() => {
                            return Some(Err(Error::Llm(LlmError::Cancelled)));
                        }
                        _ = sleep_until(self.turn_deadline) => {
                            return Some(Err(Error::Llm(self.deadline())));
                        }
                        _ = until => {
                            self.vendor = vendor;
                            self.attempt = attempt;
                            self.phase = Phase::Request;
                        }
                    }
                }
                Phase::Streaming { mut body } => {
                    match self.wait_chunk(&mut body).await {
                        WaitOutcome::Chunk(bytes) => {
                            let events = match self.sse.push(&bytes) {
                                Ok(events) => events,
                                Err(err) => {
                                    if let Some(item) = self.fail_with(err) {
                                        return Some(item);
                                    }
                                    continue;
                                }
                            };
                            if let Some(err) = self.ingest(events) {
                                if let Some(item) = self.fail_with(err) {
                                    return Some(item);
                                }
                                continue;
                            }
                            if self.premature_done {
                                // A sentinel without a finish reason fails like a
                                // truncated stream.
                                let err = LlmError::Transport(
                                    "the provider sent [DONE] before a finish reason".to_owned(),
                                );
                                if let Some(item) = self.fail_attempt(err) {
                                    return Some(item);
                                }
                            } else {
                                self.phase = if self.done_queued {
                                    Phase::Finished
                                } else {
                                    Phase::Streaming { body }
                                };
                            }
                        }
                        WaitOutcome::Eof => {
                            // Events a trailing lone-CR line may still dispatch.
                            let trailing = match self.sse.finish() {
                                Ok(events) => events,
                                Err(err) => {
                                    if let Some(item) = self.fail_with(err) {
                                        return Some(item);
                                    }
                                    continue;
                                }
                            };
                            if let Some(err) = self.ingest(trailing) {
                                if let Some(item) = self.fail_with(err) {
                                    return Some(item);
                                }
                                continue;
                            }
                            if self.done_queued || self.completed {
                                // `[DONE]` or a completed choice plus clean
                                // EOF complete the turn exactly once.
                                self.queue_done();
                                self.phase = Phase::Finished;
                            } else {
                                let err = LlmError::Transport(
                                    "the stream ended before the turn completed".to_owned(),
                                );
                                if let Some(item) = self.fail_attempt(err) {
                                    return Some(item);
                                }
                            }
                        }
                        WaitOutcome::Attempt(_) if self.completed => {
                            // Only the trailing usage chunk or `[DONE]` was lost.
                            self.queue_done();
                            self.phase = Phase::Finished;
                        }
                        WaitOutcome::Attempt(err) => {
                            if let Some(item) = self.fail_attempt(err) {
                                return Some(item);
                            }
                        }
                        WaitOutcome::Terminal(item) => return Some(item),
                    }
                }
                Phase::Collecting { mut body, mut buf } => match self.wait_chunk(&mut body).await {
                    WaitOutcome::Chunk(bytes) => {
                        buf.extend_from_slice(&bytes);
                        if buf.len() > openai::MAX_COMPLETE_BODY {
                            let err = LlmError::Protocol(format!(
                                "the complete response exceeded {} bytes",
                                openai::MAX_COMPLETE_BODY
                            ));
                            if let Some(item) = self.fail_attempt(err) {
                                return Some(item);
                            }
                            continue;
                        }
                        self.phase = Phase::Collecting { body, buf };
                    }
                    WaitOutcome::Eof => match openai::adapt_complete(&buf) {
                        Ok(deltas) => {
                            self.done_queued = true;
                            self.pending.extend(deltas);
                            self.phase = Phase::Finished;
                        }
                        Err(err) => {
                            if let Some(item) = self.fail_with(err) {
                                return Some(item);
                            }
                        }
                    },
                    WaitOutcome::Attempt(err) => {
                        if let Some(item) = self.fail_attempt(err) {
                            return Some(item);
                        }
                    }
                    WaitOutcome::Terminal(item) => return Some(item),
                },
            }
        }
    }

    /// Issues one attempt. Returns `Some(item)` when the turn is over or an
    /// item is ready, `None` when the machine should continue with the
    /// updated phase.
    async fn request(&mut self) -> Option<Result<ChatDelta>> {
        let vendor = self.vendors[self.vendor].clone();
        let body = match openai::request_body(&vendor, &self.request) {
            Ok(body) => body,
            Err(err) => {
                self.phase = Phase::Finished;
                return Some(Err(err));
            }
        };
        self.request_deadline = Instant::now() + self.timeouts.request;
        let sent = tokio::select! {
            _ = self.token.cancelled() => return Some(Err(Error::Llm(LlmError::Cancelled))),
            _ = sleep_until(self.turn_deadline) => return Some(Err(Error::Llm(self.deadline()))),
            _ = sleep_until(self.request_deadline) => Err(request_deadline_error(self.timeouts.request)),
            result = self.transport.send(&vendor, body) => result,
        };
        let response = match sent {
            Ok(response) => response,
            Err(err) => return self.fail_attempt(err),
        };
        if !(200..300).contains(&response.status) {
            // HTTP status failures surface before any stream parsing: the error body is
            // text, never SSE.
            let body = openai::read_error_body(
                response.body,
                &self.token,
                self.turn_deadline,
                self.request_deadline,
                self.timeouts.idle,
            )
            .await;
            let err = LlmError::Status {
                status: response.status,
                body,
                retry_after_ms: response.retry_after_ms,
            };
            return self.fail_attempt(err);
        }
        self.sse = SseDecoder::default();
        self.phase = if vendor.stream {
            Phase::Streaming {
                body: response.body,
            }
        } else {
            Phase::Collecting {
                body: response.body,
                buf: Vec::new(),
            }
        };
        None
    }

    /// Waits for the next body chunk under the cancellation token and the idle
    /// timeout. The turn and attempt budgets apply only while nothing has been
    /// forwarded: a long generation is bounded by idle time, not total time.
    async fn wait_chunk(&mut self, body: &mut BodyStream) -> WaitOutcome {
        let budgeted = !self.forwarded;
        tokio::select! {
            _ = self.token.cancelled() => {
                WaitOutcome::Terminal(Err(Error::Llm(LlmError::Cancelled)))
            }
            _ = sleep_until(self.turn_deadline), if budgeted => {
                WaitOutcome::Terminal(Err(Error::Llm(self.deadline())))
            }
            _ = sleep_until(self.request_deadline), if budgeted => {
                WaitOutcome::Attempt(request_deadline_error(self.timeouts.request))
            }
            _ = sleep(self.timeouts.idle) => WaitOutcome::Attempt(LlmError::Transport(format!(
                "the stream was idle for {} ms",
                self.timeouts.idle.as_millis()
            ))),
            item = body.next() => match item {
                Some(Ok(bytes)) => WaitOutcome::Chunk(bytes),
                Some(Err(err)) => WaitOutcome::Attempt(err),
                None => WaitOutcome::Eof,
            },
        }
    }

    /// Decodes SSE events into the pending queue. Returns a terminal error
    /// for protocol violations and provider error events.
    fn ingest(&mut self, events: Vec<String>) -> Option<Error> {
        for event in events {
            match openai::decode_chunk(&event) {
                Ok(ChunkEvent::StreamDone) => {
                    if self.completed {
                        self.queue_done();
                    } else {
                        // A sentinel without a finish reason fails like a truncated
                        // stream.
                        self.premature_done = true;
                    }
                    return None;
                }
                Ok(ChunkEvent::Decoded {
                    deltas,
                    finish_reason,
                }) => {
                    self.pending.extend(deltas);
                    if finish_reason.is_some() {
                        self.completed = true;
                        self.finish_reason = finish_reason;
                    }
                }
                Err(err) => return Some(err),
            }
        }
        None
    }

    /// Queues the single terminal `Done` delta.
    fn queue_done(&mut self) {
        if !self.done_queued {
            self.done_queued = true;
            self.pending.push_back(ChatDelta::Done {
                finish_reason: self.finish_reason.clone(),
            });
        }
    }

    /// Routes a decode or protocol failure through the attempt logic: not retried on
    /// the same vendor, but failed over while nothing has been forwarded.
    fn deadline(&self) -> LlmError {
        LlmError::TurnDeadline {
            last: self.last_error.clone(),
        }
    }

    fn fail_with(&mut self, err: Error) -> Option<Result<ChatDelta>> {
        match err {
            Error::Llm(err) => self.fail_attempt(err),
            other => {
                self.phase = Phase::Finished;
                Some(Err(other))
            }
        }
    }

    /// Retries or fails over only before forwarding a delta, preventing duplicated
    /// output. A context overflow surfaces at once: requests are planned against
    /// the smallest vendor capacity, so the next vendor would overflow too, and
    /// the engine needs the classification to compact.
    fn fail_attempt(&mut self, err: LlmError) -> Option<Result<ChatDelta>> {
        if !self.forwarded && !err.is_context_overflow() {
            self.last_error = Some(err.to_string());
            let delay = retry_delay(self.attempt, &err);
            // A backoff that would outlast the turn fails over or reports this error now.
            if err.is_retryable()
                && self.attempt < MAX_ATTEMPTS
                && Instant::now() + delay < self.turn_deadline
            {
                let (vendor, attempt) = (self.vendor, self.attempt + 1);
                self.reset_attempt();
                self.phase = Phase::Backoff {
                    until: Box::pin(sleep(delay)),
                    vendor,
                    attempt,
                };
                return None;
            }
            if self.vendor + 1 < self.vendors.len() {
                self.vendor += 1;
                self.attempt = 1;
                self.reset_attempt();
                self.phase = Phase::Request;
                return None;
            }
        }
        self.phase = Phase::Finished;
        Some(Err(Error::Llm(err)))
    }

    /// Drops the failed attempt's decode state: a retry re-requests the
    /// whole turn, so its deltas must never be appended to undelivered
    /// leftovers of the previous attempt.
    fn reset_attempt(&mut self) {
        self.pending.clear();
        self.finish_reason = None;
        self.completed = false;
        self.done_queued = false;
        self.premature_done = false;
    }
}

fn request_deadline_error(timeout: Duration) -> LlmError {
    LlmError::Transport(format!(
        "the request exceeded its {} ms deadline",
        timeout.as_millis()
    ))
}

#[cfg(test)]
mod tests {
    use std::result::Result as StdResult;
    use std::sync::Mutex;

    use serde_json::json;

    use super::*;
    use crate::llm::openai::WireResponse;

    enum BodyScript {
        /// Emit these bytes as one body chunk.
        Chunk(&'static [u8]),
        /// Emit nothing for this long (exercises the idle timeout).
        Delay(Duration),
        /// Fail the body stream.
        Fail(&'static str),
    }

    enum Outcome {
        /// Respond with a status and a scripted body.
        Response {
            status: u16,
            retry_after_ms: Option<u64>,
            steps: Vec<BodyScript>,
        },
        /// Never resolve: the attempt hangs until a deadline fires.
        Hang,
        /// Fail at the transport level.
        Fail(&'static str),
    }

    struct ScriptedTransport {
        outcomes: Mutex<VecDeque<Outcome>>,
        requests: Mutex<Vec<(String, serde_json::Value)>>,
    }

    impl ScriptedTransport {
        fn new(outcomes: Vec<Outcome>) -> Arc<Self> {
            Arc::new(Self {
                outcomes: Mutex::new(outcomes.into()),
                requests: Mutex::new(Vec::new()),
            })
        }

        fn requests(&self) -> Vec<(String, serde_json::Value)> {
            self.requests
                .lock()
                .expect("the transport mutex is poisoned")
                .clone()
        }

        fn request_count(&self) -> usize {
            self.requests
                .lock()
                .expect("the transport mutex is poisoned")
                .len()
        }
    }

    #[async_trait]
    impl Transport for ScriptedTransport {
        async fn send(&self, vendor: &Vendor, body: Vec<u8>) -> StdResult<WireResponse, LlmError> {
            let parsed = serde_json::from_slice(&body).unwrap_or(serde_json::Value::Null);
            self.requests
                .lock()
                .expect("the transport mutex is poisoned")
                .push((vendor.name.clone(), parsed));
            let outcome = self
                .outcomes
                .lock()
                .expect("the transport mutex is poisoned")
                .pop_front()
                .unwrap_or(Outcome::Fail("the script ran out of outcomes"));
            match outcome {
                Outcome::Hang => {
                    std::future::pending::<()>().await;
                    unreachable!("a hung send is only ended by a deadline")
                }
                Outcome::Fail(message) => Err(LlmError::Transport(message.to_owned())),
                Outcome::Response {
                    status,
                    retry_after_ms,
                    steps,
                } => Ok(WireResponse {
                    status,
                    retry_after_ms,
                    body: script_body(steps),
                }),
            }
        }
    }

    fn script_body(steps: Vec<BodyScript>) -> BodyStream {
        stream::unfold(steps.into_iter(), |mut steps| async move {
            loop {
                match steps.next()? {
                    BodyScript::Chunk(bytes) => {
                        return Some((Ok(Bytes::copy_from_slice(bytes)), steps));
                    }
                    BodyScript::Delay(duration) => sleep(duration).await,
                    BodyScript::Fail(message) => {
                        return Some((Err(LlmError::Transport(message.to_owned())), steps));
                    }
                }
            }
        })
        .boxed()
    }

    /// A body chunk, leaked into 'static (tests only).
    fn chunk(text: impl Into<String>) -> BodyScript {
        BodyScript::Chunk(Box::leak(text.into().into_bytes().into_boxed_slice()))
    }

    fn event(payload: serde_json::Value) -> BodyScript {
        chunk(format!("data: {payload}\n\n"))
    }

    fn text_event(text: &str) -> BodyScript {
        event(json!({"choices": [{"delta": {"content": text}}]}))
    }

    fn finish_event(reason: &str) -> BodyScript {
        event(json!({"choices": [{"delta": {}, "finish_reason": reason}]}))
    }

    fn usage_event(prompt: u32, completion: u32) -> BodyScript {
        event(
            json!({"choices": [], "usage": {"prompt_tokens": prompt, "completion_tokens": completion}}),
        )
    }

    fn done_event() -> BodyScript {
        chunk("data: [DONE]\n\n")
    }

    fn ok_stream(steps: Vec<BodyScript>) -> Outcome {
        Outcome::Response {
            status: 200,
            retry_after_ms: None,
            steps,
        }
    }

    fn vendors(count: usize, stream_mode: bool) -> Vendors {
        Vendors(
            (0..count)
                .map(|i| Vendor {
                    name: format!("v{i}"),
                    url: format!("https://v{i}.test"),
                    model: "m".into(),
                    api_key: Some("k".into()),
                    context_tokens: 128_000,
                    extra_body: Default::default(),
                    headers: Default::default(),
                    stream: stream_mode,
                })
                .collect(),
        )
    }

    async fn fold_with(vendors: Vendors, outcomes: Vec<Outcome>) -> ChatTurn {
        let model = VendorModel::with_transport(vendors, ScriptedTransport::new(outcomes));
        let stream = model
            .stream(ChatRequest::default(), &CancellationToken::new())
            .await
            .expect("the turn should start");
        ChatTurn::fold(stream).await.expect("the turn should fold")
    }

    /// The next item, which must be a delta.
    async fn next_delta(stream: &mut ChatStream) -> ChatDelta {
        stream
            .next()
            .await
            .expect("the stream should not end")
            .expect("the item should not be an error")
    }

    #[tokio::test(start_paused = true)]
    async fn interrupt_cancels_a_retry_after_backoff_without_another_request() {
        let transport = ScriptedTransport::new(vec![Outcome::Response {
            status: 429,
            retry_after_ms: Some(60_000),
            steps: vec![chunk("rate limited")],
        }]);
        let model = VendorModel::with_transport(vendors(1, true), transport.clone());
        let token = CancellationToken::new();
        let canceller = token.clone();
        tokio::spawn(async move {
            sleep(Duration::from_secs(5)).await;
            canceller.cancel();
        });
        let stream = model.stream(ChatRequest::default(), &token).await.unwrap();
        let err = ChatTurn::fold(stream).await.unwrap_err();
        assert!(matches!(err, Error::Llm(LlmError::Cancelled)), "{err}");
        assert_eq!(
            transport.request_count(),
            1,
            "the backoff must not issue another request"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn the_turn_deadline_bounds_all_vendor_attempts_together() {
        let transport = ScriptedTransport::new(vec![
            Outcome::Response {
                status: 401,
                retry_after_ms: None,
                steps: vec![chunk("unauthorized")],
            },
            Outcome::Hang,
            Outcome::Hang,
        ]);
        let model = VendorModel::with_transport(vendors(3, true), transport.clone());
        let stream = model
            .stream(ChatRequest::default(), &CancellationToken::new())
            .await
            .unwrap();
        let err = ChatTurn::fold(stream).await.unwrap_err();
        assert!(
            matches!(err, Error::Llm(LlmError::TurnDeadline { .. })),
            "{err}"
        );
        // The deadline names the failure that led to it, not just the timeout.
        assert!(err.to_string().contains("after: "), "{err}");
        // v1 answers 401 (not retryable: immediate failover). v2 attempt 1
        // stalls until its 120 s request deadline; attempt 2 starts around
        // t=122 s and is cut off by the 180 s turn deadline. v3 is never
        // reached: the budget spans all vendors together.
        assert_eq!(transport.request_count(), 3);
    }

    #[tokio::test(start_paused = true)]
    async fn a_long_generation_outlives_the_turn_and_attempt_budgets() {
        // Four minutes of slow but steady output: longer than both budgets,
        // never idle long enough to stall.
        let transport = ScriptedTransport::new(vec![ok_stream(vec![
            text_event("start"),
            BodyScript::Delay(Duration::from_secs(50)),
            text_event(" a"),
            BodyScript::Delay(Duration::from_secs(50)),
            text_event(" b"),
            BodyScript::Delay(Duration::from_secs(50)),
            text_event(" c"),
            BodyScript::Delay(Duration::from_secs(50)),
            text_event(" d"),
            BodyScript::Delay(Duration::from_secs(50)),
            finish_event("stop"),
            done_event(),
        ])]);
        let model = VendorModel::with_transport(vendors(2, true), transport.clone());
        let started = Instant::now();
        let turn = ChatTurn::fold(
            model
                .stream(ChatRequest::default(), &CancellationToken::new())
                .await
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(turn.message.content.as_deref(), Some("start a b c d"));
        assert!(started.elapsed() > TURN_BUDGET, "{:?}", started.elapsed());
        assert_eq!(transport.request_count(), 1, "nothing was retried");
    }

    #[tokio::test(start_paused = true)]
    async fn configurable_model_timeouts_replace_the_defaults() {
        let transport = ScriptedTransport::new(vec![Outcome::Hang]);
        let model = VendorModel::with_transport(vendors(1, true), transport)
            .with_timeouts(ModelTimeouts {
                request: Duration::from_secs(20),
                turn: Duration::from_secs(7),
                idle: Duration::from_secs(3),
            })
            .unwrap();
        let started = Instant::now();
        let error = ChatTurn::fold(
            model
                .stream(ChatRequest::default(), &CancellationToken::new())
                .await
                .unwrap(),
        )
        .await
        .unwrap_err();
        assert!(matches!(error, Error::Llm(LlmError::TurnDeadline { .. })));
        assert_eq!(started.elapsed(), Duration::from_secs(7));
        assert!(
            ModelTimeouts {
                idle: Duration::ZERO,
                ..ModelTimeouts::default()
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn budgets_match_the_source_values() {
        assert_eq!(TURN_BUDGET, Duration::from_secs(180));
        assert_eq!(REQUEST_BUDGET, Duration::from_secs(120));
        assert_eq!(IDLE_TIMEOUT, Duration::from_secs(60));
        assert_eq!(MAX_ATTEMPTS, 3);
    }

    #[tokio::test(start_paused = true)]
    async fn stalled_error_bodies_do_not_exhaust_the_turn_before_failover() {
        let transport = ScriptedTransport::new(vec![
            Outcome::Response {
                status: 401,
                retry_after_ms: None,
                steps: vec![BodyScript::Delay(Duration::from_secs(180))],
            },
            ok_stream(vec![
                text_event("healthy"),
                finish_event("stop"),
                done_event(),
            ]),
        ]);
        let model = VendorModel::with_transport(vendors(2, true), transport.clone());
        let started = Instant::now();
        let turn = ChatTurn::fold(
            model
                .stream(ChatRequest::default(), &CancellationToken::new())
                .await
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(turn.message.content.as_deref(), Some("healthy"));
        assert_eq!(transport.request_count(), 2);
        assert_eq!(started.elapsed(), IDLE_TIMEOUT);
    }

    #[tokio::test(start_paused = true)]
    async fn a_retryable_status_retries_then_fails_over() {
        let transport = ScriptedTransport::new(vec![
            Outcome::Response {
                status: 500,
                retry_after_ms: None,
                steps: vec![],
            },
            Outcome::Response {
                status: 503,
                retry_after_ms: None,
                steps: vec![],
            },
            Outcome::Response {
                status: 500,
                retry_after_ms: None,
                steps: vec![],
            },
            ok_stream(vec![
                text_event("hello"),
                finish_event("stop"),
                done_event(),
            ]),
        ]);
        let model = VendorModel::with_transport(vendors(2, true), transport.clone());
        let turn = ChatTurn::fold(
            model
                .stream(ChatRequest::default(), &CancellationToken::new())
                .await
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(turn.message.content.as_deref(), Some("hello"));
        let requests = transport.requests();
        let names: Vec<&str> = requests.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(
            names,
            ["v0", "v0", "v0", "v1"],
            "three attempts, then the next vendor"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_non_retryable_status_fails_over_immediately() {
        let transport = ScriptedTransport::new(vec![
            Outcome::Response {
                status: 401,
                retry_after_ms: None,
                steps: vec![chunk("nope")],
            },
            ok_stream(vec![text_event("hi"), finish_event("stop"), done_event()]),
        ]);
        let model = VendorModel::with_transport(vendors(2, true), transport.clone());
        let turn = ChatTurn::fold(
            model
                .stream(ChatRequest::default(), &CancellationToken::new())
                .await
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(turn.message.content.as_deref(), Some("hi"));
        let requests = transport.requests();
        let names: Vec<&str> = requests.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(names, ["v0", "v1"], "no retries of a 401");
    }

    #[tokio::test(start_paused = true)]
    async fn a_context_overflow_surfaces_without_retry_or_failover() {
        let transport = ScriptedTransport::new(vec![
            Outcome::Response {
                status: 400,
                retry_after_ms: None,
                steps: vec![chunk("This model's maximum context length is 8192 tokens")],
            },
            ok_stream(vec![
                text_event("never"),
                finish_event("stop"),
                done_event(),
            ]),
        ]);
        let model = VendorModel::with_transport(vendors(2, true), transport.clone());
        let err = ChatTurn::fold(
            model
                .stream(ChatRequest::default(), &CancellationToken::new())
                .await
                .unwrap(),
        )
        .await
        .unwrap_err();
        let Error::Llm(llm_err) = &err else {
            panic!("{err}");
        };
        assert!(llm_err.is_context_overflow(), "{err}");
        assert_eq!(
            transport.request_count(),
            1,
            "the next vendor would overflow too; the engine must see the overflow"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_usage_only_delta_does_not_disarm_the_retry() {
        let transport = ScriptedTransport::new(vec![
            ok_stream(vec![
                event(json!({
                    "choices": [{"delta": {"role": "assistant", "content": ""}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 0}
                })),
                BodyScript::Fail("connection reset"),
            ]),
            ok_stream(vec![text_event("hi"), finish_event("stop"), done_event()]),
        ]);
        let model = VendorModel::with_transport(vendors(1, true), transport.clone());
        let turn = ChatTurn::fold(
            model
                .stream(ChatRequest::default(), &CancellationToken::new())
                .await
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(turn.message.content.as_deref(), Some("hi"));
        assert_eq!(
            transport.request_count(),
            2,
            "usage alone is not forwarded output"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_stream_failure_before_any_delta_retries_the_attempt() {
        let transport = ScriptedTransport::new(vec![
            ok_stream(vec![
                chunk("data: {\"cho"),
                BodyScript::Fail("connection reset"),
            ]),
            ok_stream(vec![text_event("hi"), finish_event("stop"), done_event()]),
        ]);
        let model = VendorModel::with_transport(vendors(1, true), transport.clone());
        let turn = ChatTurn::fold(
            model
                .stream(ChatRequest::default(), &CancellationToken::new())
                .await
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(turn.message.content.as_deref(), Some("hi"));
        assert_eq!(transport.request_count(), 2);
    }

    #[tokio::test(start_paused = true)]
    async fn a_disconnect_after_forwarded_output_is_terminal() {
        let transport = ScriptedTransport::new(vec![
            ok_stream(vec![
                text_event("partial"),
                BodyScript::Fail("connection reset"),
            ]),
            ok_stream(vec![
                text_event("never used"),
                finish_event("stop"),
                done_event(),
            ]),
        ]);
        let model = VendorModel::with_transport(vendors(2, true), transport.clone());
        let mut stream = model
            .stream(ChatRequest::default(), &CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(
            next_delta(&mut stream).await,
            ChatDelta::Text("partial".into())
        );
        let err = stream.next().await.unwrap().unwrap_err();
        assert!(err.to_string().contains("connection reset"), "{err}");
        assert!(stream.next().await.is_none(), "the turn ends terminally");
        assert_eq!(
            transport.request_count(),
            1,
            "forwarded output is never retried"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_provider_error_event_after_forwarded_output_is_terminal() {
        let transport = ScriptedTransport::new(vec![ok_stream(vec![
            text_event("partial"),
            chunk(format!(
                "data: {}\n\n",
                json!({"error": {"message": "boom", "code": "server_error"}})
            )),
        ])]);
        let model = VendorModel::with_transport(vendors(1, true), transport.clone());
        let mut stream = model
            .stream(ChatRequest::default(), &CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(
            next_delta(&mut stream).await,
            ChatDelta::Text("partial".into())
        );
        let err = stream.next().await.unwrap().unwrap_err();
        assert!(err.to_string().contains("boom"), "{err}");
        assert_eq!(transport.request_count(), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn provider_error_events_retry_when_transient_and_fail_over_otherwise() {
        // A 200 whose body is an error event (common behind proxies) must not
        // kill the turn: a transient one is retried on the same vendor, any
        // other fails over to the next.
        for (error, expected) in [
            (
                json!({"message": "upstream exploded", "type": "server_error"}),
                ["v0", "v0"],
            ),
            (
                json!({"message": "unsupported parameter", "type": "invalid_request_error"}),
                ["v0", "v1"],
            ),
        ] {
            let transport = ScriptedTransport::new(vec![
                ok_stream(vec![chunk(format!(
                    "data: {}\n\n",
                    json!({"error": error})
                ))]),
                ok_stream(vec![text_event("hi"), finish_event("stop"), done_event()]),
            ]);
            let model = VendorModel::with_transport(vendors(2, true), transport.clone());
            let turn = ChatTurn::fold(
                model
                    .stream(ChatRequest::default(), &CancellationToken::new())
                    .await
                    .unwrap(),
            )
            .await
            .unwrap();
            assert_eq!(turn.message.content.as_deref(), Some("hi"));
            let requests = transport.requests();
            let names: Vec<&str> = requests.iter().map(|(name, _)| name.as_str()).collect();
            assert_eq!(names, expected, "{error}");
        }
    }

    #[tokio::test(start_paused = true)]
    async fn nothing_follows_a_terminal_error() {
        // The text is decoded in the same chunk as the malformed event.
        let model = VendorModel::with_transport(
            vendors(1, true),
            ScriptedTransport::new(vec![ok_stream(vec![chunk(concat!(
                "data: {\"choices\":[{\"delta\":{\"content\":\"lost\"}}]}\n\n",
                "data: not json\n\n"
            ))])]),
        );
        let mut stream = model
            .stream(ChatRequest::default(), &CancellationToken::new())
            .await
            .unwrap();
        let err = stream.next().await.unwrap().unwrap_err();
        assert!(err.to_string().contains("cannot decode"), "{err}");
        assert!(
            stream.next().await.is_none(),
            "decoded deltas never follow the error"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_stalled_stream_hits_the_idle_timeout_and_retries() {
        let transport = ScriptedTransport::new(vec![
            ok_stream(vec![BodyScript::Delay(Duration::from_secs(61))]),
            ok_stream(vec![text_event("hi"), finish_event("stop"), done_event()]),
        ]);
        let model = VendorModel::with_transport(vendors(1, true), transport.clone());
        let turn = ChatTurn::fold(
            model
                .stream(ChatRequest::default(), &CancellationToken::new())
                .await
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(turn.message.content.as_deref(), Some("hi"));
        assert_eq!(
            transport.request_count(),
            2,
            "the idle timeout freed the turn budget"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn chunks_within_the_idle_timeout_keep_the_stream_alive() {
        let turn = fold_with(
            vendors(1, true),
            vec![ok_stream(vec![
                BodyScript::Delay(Duration::from_secs(59)),
                text_event("slow but alive"),
                finish_event("stop"),
                done_event(),
            ])],
        )
        .await;
        assert_eq!(turn.message.content.as_deref(), Some("slow but alive"));
    }

    #[tokio::test(start_paused = true)]
    async fn cancellation_mid_stream_ends_the_turn() {
        let transport = ScriptedTransport::new(vec![ok_stream(vec![
            text_event("a"),
            BodyScript::Delay(Duration::from_secs(30)),
            text_event("b"),
        ])]);
        let model = VendorModel::with_transport(vendors(1, true), transport.clone());
        let token = CancellationToken::new();
        let mut stream = model.stream(ChatRequest::default(), &token).await.unwrap();
        assert_eq!(next_delta(&mut stream).await, ChatDelta::Text("a".into()));
        let canceller = token.clone();
        tokio::spawn(async move {
            sleep(Duration::from_secs(1)).await;
            canceller.cancel();
        });
        let err = stream.next().await.unwrap().unwrap_err();
        assert!(matches!(err, Error::Llm(LlmError::Cancelled)), "{err}");
    }

    #[tokio::test(start_paused = true)]
    async fn done_completes_the_turn_exactly_once() {
        let model = VendorModel::with_transport(
            vendors(1, true),
            ScriptedTransport::new(vec![ok_stream(vec![
                text_event("hi"),
                finish_event("stop"),
                done_event(),
                // Anything after [DONE] must never reach the consumer.
                text_event("late"),
            ])]),
        );
        let mut stream = model
            .stream(ChatRequest::default(), &CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(next_delta(&mut stream).await, ChatDelta::Text("hi".into()));
        assert_eq!(
            next_delta(&mut stream).await,
            ChatDelta::Done {
                finish_reason: Some("stop".into())
            }
        );
        assert!(stream.next().await.is_none(), "the stream ends after Done");
    }

    #[tokio::test(start_paused = true)]
    async fn done_before_a_finish_reason_fails_the_turn() {
        let model = VendorModel::with_transport(
            vendors(1, true),
            ScriptedTransport::new(vec![ok_stream(vec![text_event("partial"), done_event()])]),
        );
        let mut stream = model
            .stream(ChatRequest::default(), &CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(
            next_delta(&mut stream).await,
            ChatDelta::Text("partial".into())
        );
        let err = stream
            .next()
            .await
            .expect("the turn ends with an error")
            .unwrap_err();
        assert!(
            err.to_string().contains("[DONE] before a finish reason"),
            "{err}"
        );
        assert!(stream.next().await.is_none(), "the stream ends");
    }

    #[tokio::test(start_paused = true)]
    async fn a_premature_done_retries_without_duplicating_deltas() {
        let transport = ScriptedTransport::new(vec![
            // Both events arrive in one body chunk: the partial text is
            // decoded but never forwarded before `[DONE]` fails the attempt.
            ok_stream(vec![chunk(concat!(
                "data: {\"choices\":[{\"delta\":{\"content\":\"lost\"}}]}\n\n",
                "data: [DONE]\n\n"
            ))]),
            ok_stream(vec![
                text_event("hello"),
                finish_event("stop"),
                done_event(),
            ]),
        ]);
        let model = VendorModel::with_transport(vendors(1, true), transport.clone());
        let turn = ChatTurn::fold(
            model
                .stream(ChatRequest::default(), &CancellationToken::new())
                .await
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(
            turn.message.content.as_deref(),
            Some("hello"),
            "the failed attempt's partial text is discarded, not duplicated"
        );
        assert_eq!(transport.request_count(), 2);
    }

    #[tokio::test(start_paused = true)]
    async fn a_completed_choice_plus_clean_eof_completes_without_done_sentinel() {
        let turn = fold_with(
            vendors(1, true),
            vec![ok_stream(vec![text_event("hi"), finish_event("stop")])],
        )
        .await;
        assert_eq!(turn.message.content.as_deref(), Some("hi"));
    }

    #[tokio::test(start_paused = true)]
    async fn a_disconnect_after_the_finish_reason_completes_the_turn() {
        let turn = fold_with(
            vendors(1, true),
            vec![ok_stream(vec![
                text_event("hi"),
                finish_event("stop"),
                BodyScript::Fail("connection reset"),
            ])],
        )
        .await;
        assert_eq!(turn.message.content.as_deref(), Some("hi"));
        assert_eq!(turn.finish_reason.as_deref(), Some("stop"));
    }

    #[tokio::test(start_paused = true)]
    async fn eof_before_completion_fails() {
        // Nothing forwarded yet: the EOF is a retryable dropped connection,
        // so every attempt of the vendor is used before the turn fails.
        let transport = ScriptedTransport::new(vec![
            ok_stream(vec![event(
                json!({"choices": [{"delta": {"role": "assistant"}}]}),
            )]),
            ok_stream(vec![]),
            ok_stream(vec![]),
        ]);
        let model = VendorModel::with_transport(vendors(1, true), transport.clone());
        let err = ChatTurn::fold(
            model
                .stream(ChatRequest::default(), &CancellationToken::new())
                .await
                .unwrap(),
        )
        .await
        .unwrap_err();
        assert!(
            err.to_string().contains("ended before the turn completed"),
            "{err}"
        );
        assert_eq!(transport.request_count(), 3);

        // After output was forwarded: terminal with a single attempt.
        let transport = ScriptedTransport::new(vec![ok_stream(vec![text_event("hi")])]);
        let model = VendorModel::with_transport(vendors(1, true), transport.clone());
        let err = ChatTurn::fold(
            model
                .stream(ChatRequest::default(), &CancellationToken::new())
                .await
                .unwrap(),
        )
        .await
        .unwrap_err();
        assert!(
            err.to_string().contains("ended before the turn completed"),
            "{err}"
        );
        assert_eq!(transport.request_count(), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn a_partial_event_at_eof_is_terminal_after_output() {
        let transport = ScriptedTransport::new(vec![ok_stream(vec![
            text_event("hi"),
            chunk("data: {\"choices\""),
        ])]);
        let model = VendorModel::with_transport(vendors(1, true), transport.clone());
        let err = ChatTurn::fold(
            model
                .stream(ChatRequest::default(), &CancellationToken::new())
                .await
                .unwrap(),
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("middle of an SSE line"), "{err}");
        assert_eq!(
            transport.request_count(),
            1,
            "forwarded output is never retried"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_backoff_longer_than_the_turn_reports_the_real_error_at_once() {
        let transport = ScriptedTransport::new(vec![Outcome::Response {
            status: 429,
            retry_after_ms: Some(60_000),
            steps: vec![chunk("slow down")],
        }]);
        let model = VendorModel::with_transport(vendors(1, true), transport.clone())
            .with_timeouts(ModelTimeouts {
                turn: Duration::from_secs(30),
                ..ModelTimeouts::default()
            })
            .unwrap();
        let started = Instant::now();
        let err = ChatTurn::fold(
            model
                .stream(ChatRequest::default(), &CancellationToken::new())
                .await
                .unwrap(),
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("HTTP 429"), "{err}");
        assert_eq!(transport.request_count(), 1);
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "it did not sleep 60 s"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_connection_cut_mid_line_before_output_is_retried() {
        let transport = ScriptedTransport::new(vec![
            ok_stream(vec![chunk("data: {\"choices\"")]),
            ok_stream(vec![text_event("hi"), finish_event("stop"), done_event()]),
        ]);
        let model = VendorModel::with_transport(vendors(1, true), transport.clone());
        let turn = ChatTurn::fold(
            model
                .stream(ChatRequest::default(), &CancellationToken::new())
                .await
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(turn.message.content.as_deref(), Some("hi"));
        assert_eq!(transport.request_count(), 2);
    }

    #[tokio::test(start_paused = true)]
    async fn usage_after_the_finish_chunk_is_retained() {
        let turn = fold_with(
            vendors(1, true),
            vec![ok_stream(vec![
                text_event("hi"),
                finish_event("stop"),
                usage_event(11, 7),
                done_event(),
            ])],
        )
        .await;
        assert_eq!(turn.message.content.as_deref(), Some("hi"));
        assert_eq!(
            turn.usage,
            Some(Usage {
                prompt_tokens: 11,
                completion_tokens: 7
            })
        );
    }

    #[tokio::test(start_paused = true)]
    async fn malformed_chunk_data_errors_without_panic() {
        let transport = ScriptedTransport::new(vec![ok_stream(vec![chunk("data: not json\n\n")])]);
        let model = VendorModel::with_transport(vendors(1, true), transport.clone());
        let err = ChatTurn::fold(
            model
                .stream(ChatRequest::default(), &CancellationToken::new())
                .await
                .unwrap(),
        )
        .await
        .unwrap_err();
        assert!(
            err.to_string().contains("cannot decode a response chunk"),
            "{err}"
        );
        assert_eq!(transport.request_count(), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn streaming_and_single_shot_fold_identically() {
        let streaming = ok_stream(vec![
            event(json!({"choices": [{"delta": {"role": "assistant", "content": ""}}]})),
            event(json!({"choices": [{"delta": {"reasoning_content": "hmm"}}]})),
            event(json!({"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "c0", "function": {"name": "read", "arguments": ""}}
            ]}}]})),
            event(json!({"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": "{\"a\":"}}
            ]}}]})),
            event(json!({"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": "1}"}}
            ]}}]})),
            event(
                json!({"choices": [{"delta": {}, "finish_reason": "tool_calls"}],
                         "usage": {"prompt_tokens": 11, "completion_tokens": 7}}),
            ),
            done_event(),
        ]);
        let complete = json!({
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
        let single_shot = Outcome::Response {
            status: 200,
            retry_after_ms: None,
            steps: vec![chunk(complete)],
        };

        let streamed = fold_with(vendors(1, true), vec![streaming]).await;
        let shot = fold_with(vendors(1, false), vec![single_shot]).await;
        assert_eq!(
            streamed, shot,
            "both transports must fold into the same turn"
        );
        assert_eq!(
            streamed.message.content, None,
            "a tool turn has null content"
        );
        assert_eq!(
            streamed.message.tool_calls,
            Some(vec![ToolCall::new("c0", "read", "{\"a\":1}")])
        );
        assert_eq!(streamed.message.reasoning_content.as_deref(), Some("hmm"));
        assert_eq!(
            streamed.usage,
            Some(Usage {
                prompt_tokens: 11,
                completion_tokens: 7
            })
        );
    }

    #[tokio::test(start_paused = true)]
    async fn fragmented_utf8_and_framing_reassemble() {
        // One event, split across body chunks every three bytes: mid-JSON,
        // mid-frame and mid-character.
        let event = format!(
            "data: {}\n\n",
            json!({"choices": [{"delta": {"content": "café ☕"}}]})
        );
        let bytes = event.into_bytes();
        let steps: Vec<BodyScript> = bytes
            .chunks(3)
            .map(|piece| BodyScript::Chunk(Box::leak(piece.to_vec().into_boxed_slice()) as &[u8]))
            .chain([finish_event("stop"), done_event()])
            .collect();
        let turn = fold_with(vendors(1, true), vec![ok_stream(steps)]).await;
        assert_eq!(turn.message.content.as_deref(), Some("café ☕"));
    }

    #[tokio::test(start_paused = true)]
    async fn status_failures_surface_before_stream_parsing() {
        // A 400 whose body looks like SSE must be reported as the status
        // error, never parsed as a stream — and its text must stay usable
        // for context-overflow detection.
        let transport = ScriptedTransport::new(vec![Outcome::Response {
            status: 400,
            retry_after_ms: None,
            steps: vec![chunk(format!(
                "data: {}\n\n",
                json!({"error": {"message": "This model's maximum context length is 8192 tokens."}})
            ))],
        }]);
        let model = VendorModel::with_transport(vendors(1, true), transport.clone());
        let err = ChatTurn::fold(
            model
                .stream(ChatRequest::default(), &CancellationToken::new())
                .await
                .unwrap(),
        )
        .await
        .unwrap_err();
        let rendered = err.to_string();
        assert!(rendered.starts_with("HTTP 400: data:"), "{rendered}");
        assert!(is_context_overflow(&rendered), "{rendered}");
        assert_eq!(transport.request_count(), 1, "a 400 is not retryable");
    }

    #[test]
    fn vendors_from_lookup_parses_the_environment_contract() {
        use std::collections::HashMap;
        let vars: HashMap<String, String> = [
            ("AI_VENDORS", "openai, deepseek ,"),
            ("AI_OPENAI_URL", "https://api.openai.test/v1///"),
            ("AI_OPENAI_MODEL", "gpt-test"),
            ("AI_OPENAI_API_KEY", "sk-1"),
            ("AI_DEEPSEEK_URL", "https://ds.test"),
            ("AI_DEEPSEEK_MODEL", "ds"),
            ("AI_DEEPSEEK_CONTEXT_TOKENS", "64000"),
            ("AI_DEEPSEEK_EXTRA_BODY", r#"{"enable_thinking": true}"#),
            ("AI_DEEPSEEK_HEADERS", r#"{"api-key": "secret-header"}"#),
            ("AI_DEEPSEEK_STREAM", "false"),
        ]
        .into_iter()
        .map(|(key, value)| (key.to_owned(), value.to_owned()))
        .collect();
        let vendors = Vendors::from_lookup(|key| vars.get(key).cloned()).unwrap();
        assert_eq!(vendors.as_slice().len(), 2);

        let openai = &vendors.as_slice()[0];
        assert_eq!(openai.name, "openai");
        assert_eq!(
            openai.url, "https://api.openai.test/v1",
            "trailing slashes are stripped"
        );
        assert_eq!(openai.context_tokens, 128_000, "the default context");
        assert!(openai.stream, "streaming is the default");
        assert_eq!(openai.api_key.as_deref(), Some("sk-1"));
        assert!(openai.extra_body.is_empty() && openai.headers.is_empty());

        let deepseek = &vendors.as_slice()[1];
        assert_eq!(deepseek.context_tokens, 64_000);
        assert_eq!(deepseek.api_key, None, "the API key is optional");
        assert_eq!(deepseek.extra_body["enable_thinking"], true);
        assert_eq!(deepseek.headers["api-key"], "secret-header");
        assert!(
            !format!("{deepseek:?}").contains("secret-header"),
            "header values are redacted"
        );
        assert!(
            !deepseek.stream,
            "AI_<NAME>_STREAM=false falls back to single-shot"
        );

        assert_eq!(
            vendors.context_tokens(),
            64_000,
            "the smallest vendor context"
        );
    }

    #[test]
    fn evaluator_vendors_share_the_vendor_variables() {
        use std::collections::HashMap;
        let vars: HashMap<&str, &str> = [
            ("AI_VENDORS", "main"),
            ("AI_MAIN_URL", "https://main.test"),
            ("AI_MAIN_MODEL", "big"),
            ("AI_EVALUATOR_VENDORS", "mini"),
            ("AI_MINI_URL", "https://mini.test"),
            ("AI_MINI_MODEL", "small"),
        ]
        .into_iter()
        .collect();
        let lookup = |key: &str| vars.get(key).map(|value| value.to_string());
        let evaluator = Vendors::evaluator_from_lookup(lookup).unwrap().unwrap();
        assert_eq!(evaluator.as_slice().len(), 1);
        assert_eq!(evaluator.as_slice()[0].name, "mini");
        assert_eq!(evaluator.as_slice()[0].model, "small");

        let unset = |key: &str| {
            (key != "AI_EVALUATOR_VENDORS")
                .then(|| lookup(key))
                .flatten()
        };
        assert!(Vendors::evaluator_from_lookup(unset).unwrap().is_none());
        let incomplete = |key: &str| (key == "AI_EVALUATOR_VENDORS").then(|| "mini".to_owned());
        let err = Vendors::evaluator_from_lookup(incomplete).unwrap_err();
        assert!(err.to_string().contains("AI_MINI_URL"), "{err}");
    }

    #[test]
    fn vendors_from_lookup_rejects_incomplete_configuration() {
        use std::collections::HashMap;
        let err = Vendors::from_lookup(|_| None).unwrap_err();
        assert!(err.to_string().contains("AI_VENDORS"), "{err}");

        let lookup = |vars: HashMap<String, String>| move |key: &str| vars.get(key).cloned();
        let base: HashMap<String, String> = [
            ("AI_VENDORS", "x"),
            ("AI_X_URL", "https://x.test"),
            ("AI_X_MODEL", "m"),
        ]
        .into_iter()
        .map(|(key, value)| (key.to_owned(), value.to_owned()))
        .collect();
        assert!(Vendors::from_lookup(lookup(base)).is_ok());
        let named = |name: &str| {
            let name = name.to_owned();
            move |key: &str| (key == "AI_VENDORS").then(|| name.clone())
        };
        let err = Vendors::from_lookup(named("my-llm")).unwrap_err();
        assert!(err.to_string().contains("`my-llm`"), "{err}");

        for (key, value) in [
            ("AI_X_CONTEXT_TOKENS", "many"),
            ("AI_X_CONTEXT_TOKENS", "0"),
            ("AI_X_STREAM", "yes"),
            ("AI_X_URL", "   "),
            ("AI_X_EXTRA_BODY", "[1]"),
            ("AI_X_EXTRA_BODY", r#"{"messages": []}"#),
            ("AI_X_HEADERS", r#"{"bad header": "v"}"#),
            ("AI_X_HEADERS", r#"{"x-n": 1}"#),
        ] {
            let mut vars: HashMap<String, String> = [
                ("AI_VENDORS", "x"),
                ("AI_X_URL", "https://x.test"),
                ("AI_X_MODEL", "m"),
                ("AI_X_API_KEY", "k"),
            ]
            .into_iter()
            .map(|(k, v)| (k.to_owned(), v.to_owned()))
            .collect();
            vars.insert(key.to_owned(), value.to_owned());
            let err = Vendors::from_lookup(lookup(vars)).unwrap_err();
            assert!(err.to_string().contains(key), "{key}={value}: {err}");
        }
    }

    #[tokio::test]
    async fn an_empty_vendor_list_is_a_configuration_error() {
        let model = VendorModel::with_transport(Vendors(vec![]), ScriptedTransport::new(vec![]));
        let Err(err) = model
            .stream(ChatRequest::default(), &CancellationToken::new())
            .await
        else {
            panic!("an empty vendor list should not start a turn");
        };
        assert!(matches!(err, Error::Config(_)), "{err}");
    }
}
