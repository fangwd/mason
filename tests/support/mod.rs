//! Shared model and tool doubles for runtime tests.
#![allow(dead_code)]

use std::collections::VecDeque;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures_util::{StreamExt, stream};
use mason::{
    CancellationToken, ChatDelta, ChatModel, ChatRequest, ChatStream, Error, LlmError, Message,
    OutputSpill, PromptSource, Result, Role, RunConfig, Tool, ToolCtx, ToolError, ToolSpec,
};

/// One scripted model turn.
pub enum Reply {
    /// A plain assistant answer.
    Answer(String),
    /// An assistant turn calling tools: (id, name, arguments JSON).
    Calls(Vec<(String, String, String)>),
    /// A whole message (reasoning, null content,...).
    Message(Message),
    /// Exact deltas; a trailing `Done` is appended when missing.
    Deltas(Vec<ChatDelta>),
    /// The turn fails with an HTTP 400 whose body is this text.
    Error(String),
    /// The turn produces nothing until cancelled.
    Hang,
    /// Wait, then behave like the inner reply.
    After(Duration, Box<Reply>),
    /// Wait for notification or cancellation, then produce the inner reply.
    Gate(Arc<tokio::sync::Notify>, Box<Reply>),
}

pub fn answer(text: &str) -> Reply {
    Reply::Answer(text.to_owned())
}

pub fn call(name: &str, args: &str) -> Reply {
    call_id(name, name, args)
}

/// A single tool call with an explicit id.
pub fn call_id(id: &str, name: &str, args: &str) -> Reply {
    Reply::Calls(vec![(id.to_owned(), name.to_owned(), args.to_owned())])
}

/// Many tool calls in one reply.
pub fn calls(items: Vec<(String, String, String)>) -> Reply {
    Reply::Calls(items)
}

pub struct ScriptedModel {
    replies: Mutex<VecDeque<Reply>>,
    requests: Mutex<Vec<ChatRequest>>,
    context: usize,
}

pub type ScriptedHandle = Arc<ScriptedModel>;

impl ScriptedModel {
    pub fn new(replies: Vec<Reply>) -> ScriptedHandle {
        Self::with_context(replies, 128_000)
    }

    pub fn with_context(replies: Vec<Reply>, context: usize) -> ScriptedHandle {
        Arc::new(Self {
            replies: Mutex::new(replies.into()),
            requests: Mutex::new(Vec::new()),
            context,
        })
    }

    /// Every request the model was sent, in order.
    pub fn requests(&self) -> Vec<ChatRequest> {
        self.requests
            .lock()
            .expect("the script mutex is poisoned")
            .clone()
    }

    pub fn request_count(&self) -> usize {
        self.requests
            .lock()
            .expect("the script mutex is poisoned")
            .len()
    }
}

#[async_trait::async_trait]
impl ChatModel for ScriptedModel {
    fn context_tokens(&self) -> usize {
        self.context
    }

    async fn stream(&self, request: ChatRequest, token: &CancellationToken) -> Result<ChatStream> {
        self.requests
            .lock()
            .expect("the script mutex is poisoned")
            .push(request);
        let reply = self
            .replies
            .lock()
            .expect("the script mutex is poisoned")
            .pop_front()
            .unwrap_or_else(|| Reply::Error("the script ran out of replies".to_owned()));
        Ok(reply_stream(reply, token.clone()))
    }
}

/// Responds from the request, allowing summary and main calls to interleave.
pub struct FunctionModel {
    responder: Box<dyn Fn(&ChatRequest) -> Reply + Send + Sync>,
    requests: Mutex<Vec<ChatRequest>>,
    context: usize,
}

impl FunctionModel {
    pub fn new(responder: impl Fn(&ChatRequest) -> Reply + Send + Sync + 'static) -> Arc<Self> {
        Self::with_context(responder, 128_000)
    }

    pub fn with_context(
        responder: impl Fn(&ChatRequest) -> Reply + Send + Sync + 'static,
        context: usize,
    ) -> Arc<Self> {
        Arc::new(Self {
            responder: Box::new(responder),
            requests: Mutex::new(Vec::new()),
            context,
        })
    }

    pub fn requests(&self) -> Vec<ChatRequest> {
        self.requests
            .lock()
            .expect("the script mutex is poisoned")
            .clone()
    }

    pub fn request_count(&self) -> usize {
        self.requests
            .lock()
            .expect("the script mutex is poisoned")
            .len()
    }
}

#[async_trait::async_trait]
impl ChatModel for FunctionModel {
    fn context_tokens(&self) -> usize {
        self.context
    }

    async fn stream(&self, request: ChatRequest, token: &CancellationToken) -> Result<ChatStream> {
        let reply = (self.responder)(&request);
        self.requests
            .lock()
            .expect("the script mutex is poisoned")
            .push(request);
        Ok(reply_stream(reply, token.clone()))
    }
}

/// Identifies a compaction summary request.
pub fn is_summary_request(request: &ChatRequest) -> bool {
    request
        .messages
        .first()
        .and_then(|message| message.content.as_deref())
        .is_some_and(|content| content.contains("context summarization agent"))
}

fn reply_stream(reply: Reply, token: CancellationToken) -> ChatStream {
    match reply {
        Reply::After(delay, inner) => Box::pin(
            stream::once(async move {
                tokio::select! {
                    _ = tokio::time::sleep(delay) => {}
                    _ = token.cancelled() => {}
                }
                reply_stream(*inner, token)
            })
            .flatten(),
        ),
        Reply::Gate(notify, inner) => Box::pin(
            stream::once(async move {
                tokio::select! {
                    _ = notify.notified() => {}
                    _ = token.cancelled() => {}
                }
                reply_stream(*inner, token)
            })
            .flatten(),
        ),
        Reply::Hang => Box::pin(stream::once(async move {
            token.cancelled().await;
            Err(Error::Llm(LlmError::Cancelled))
        })),
        other => Box::pin(stream::iter(deltas_for(other))),
    }
}

fn deltas_for(reply: Reply) -> Vec<Result<ChatDelta>> {
    match reply {
        Reply::Answer(text) => {
            vec![
                Ok(ChatDelta::Text(text)),
                Ok(ChatDelta::Done {
                    finish_reason: Some("stop".into()),
                }),
            ]
        }
        Reply::Calls(items) => items
            .into_iter()
            .enumerate()
            .map(|(index, (id, name, arguments))| {
                Ok(ChatDelta::ToolCall {
                    index: index as u32,
                    id: Some(id),
                    name: Some(name),
                    arguments,
                })
            })
            .chain([Ok(ChatDelta::Done {
                finish_reason: Some("tool_calls".into()),
            })])
            .collect(),
        Reply::Message(message) => {
            let mut deltas = Vec::new();
            if let Some(text) = message.content.filter(|text| !text.is_empty()) {
                deltas.push(Ok(ChatDelta::Text(text)));
            }
            if let Some(reasoning) = message.reasoning_content.filter(|text| !text.is_empty()) {
                deltas.push(Ok(ChatDelta::Reasoning(reasoning)));
            }
            let has_calls = message.tool_calls.is_some();
            for (index, call) in message.tool_calls.into_iter().flatten().enumerate() {
                deltas.push(Ok(ChatDelta::ToolCall {
                    index: index as u32,
                    id: Some(call.id),
                    name: Some(call.function.name),
                    arguments: call.function.arguments,
                }));
            }
            deltas.push(Ok(ChatDelta::Done {
                finish_reason: Some(if has_calls { "tool_calls" } else { "stop" }.to_owned()),
            }));
            deltas
        }
        Reply::Deltas(mut deltas) => {
            if !matches!(deltas.last(), Some(ChatDelta::Done { .. })) {
                // Auto-appended completion includes a finish reason.
                deltas.push(ChatDelta::Done {
                    finish_reason: Some("stop".into()),
                });
            }
            deltas.into_iter().map(Ok).collect()
        }
        Reply::Error(body) => {
            vec![Err(Error::Llm(LlmError::Status {
                status: 400,
                body,
                retry_after_ms: None,
            }))]
        }
        Reply::Hang | Reply::After(..) | Reply::Gate(..) => unreachable!("handled as streams"),
    }
}

fn plain_spec(name: &str) -> ToolSpec {
    ToolSpec::function(name, "a test tool", serde_json::json!({"type": "object"}))
}

/// Returns a constant output and counts executions.
pub struct EchoTool {
    name: String,
    output: String,
    pub calls: Arc<AtomicUsize>,
}

impl EchoTool {
    pub fn boxed(name: &str, output: &str) -> (Box<dyn Tool>, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        let tool = Box::new(Self {
            name: name.to_owned(),
            output: output.to_owned(),
            calls: calls.clone(),
        });
        (tool, calls)
    }
}

#[async_trait::async_trait]
impl Tool for EchoTool {
    fn spec(&self) -> ToolSpec {
        plain_spec(&self.name)
    }
    async fn call(
        &self,
        _args: serde_json::Value,
        _ctx: &ToolCtx,
    ) -> std::result::Result<String, ToolError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(self.output.clone())
    }
}

/// Concurrency gauge for the fan-out tests.
#[derive(Default)]
pub struct Gauge {
    pub current: AtomicUsize,
    pub peak: AtomicUsize,
    pub executed: Mutex<Vec<String>>,
}

/// Tracks concurrency and holds call `n == 0` until notified.
pub struct HeldTaskTool {
    pub gauge: Arc<Gauge>,
    pub release: Arc<tokio::sync::Notify>,
}

impl HeldTaskTool {
    pub fn boxed() -> (Box<dyn Tool>, Arc<Gauge>, Arc<tokio::sync::Notify>) {
        let gauge = Arc::new(Gauge::default());
        let release = Arc::new(tokio::sync::Notify::new());
        let tool = Box::new(Self {
            gauge: gauge.clone(),
            release: release.clone(),
        });
        (tool, gauge, release)
    }
}

#[async_trait::async_trait]
impl Tool for HeldTaskTool {
    fn spec(&self) -> ToolSpec {
        plain_spec("task")
    }
    async fn call(
        &self,
        args: serde_json::Value,
        _ctx: &ToolCtx,
    ) -> std::result::Result<String, ToolError> {
        let n = args.get("n").and_then(|n| n.as_i64()).unwrap_or(-1);
        self.gauge
            .executed
            .lock()
            .expect("unpoisoned")
            .push(n.to_string());
        let current = self.gauge.current.fetch_add(1, Ordering::SeqCst) + 1;
        self.gauge.peak.fetch_max(current, Ordering::SeqCst);
        let result = async {
            if n == 0 {
                self.release.notified().await;
            }
            format!("result {n}")
        }
        .await;
        self.gauge.current.fetch_sub(1, Ordering::SeqCst);
        Ok(result)
    }
}

/// Fails the way the test asks: model-visible or fatal.
pub struct FailTool {
    name: String,
    message: String,
    fatal: bool,
}

impl FailTool {
    pub fn boxed(name: &str, message: &str) -> Box<dyn Tool> {
        Box::new(Self {
            name: name.to_owned(),
            message: message.to_owned(),
            fatal: false,
        })
    }
    pub fn boxed_fatal(name: &str, message: &str) -> Box<dyn Tool> {
        Box::new(Self {
            name: name.to_owned(),
            message: message.to_owned(),
            fatal: true,
        })
    }
}

#[async_trait::async_trait]
impl Tool for FailTool {
    fn spec(&self) -> ToolSpec {
        plain_spec(&self.name)
    }
    async fn call(
        &self,
        _args: serde_json::Value,
        _ctx: &ToolCtx,
    ) -> std::result::Result<String, ToolError> {
        if self.fatal {
            Err(ToolError::Fatal(Error::Run(self.message.clone())))
        } else {
            Err(ToolError::Failure(self.message.clone()))
        }
    }
}

/// Sleeps, then returns its output; used as the slow sibling.
pub struct SleepTool {
    name: String,
    duration: Duration,
    output: String,
}

impl SleepTool {
    pub fn boxed(name: &str, duration: Duration, output: &str) -> Box<dyn Tool> {
        Box::new(Self {
            name: name.to_owned(),
            duration,
            output: output.to_owned(),
        })
    }
}

#[async_trait::async_trait]
impl Tool for SleepTool {
    fn spec(&self) -> ToolSpec {
        plain_spec(&self.name)
    }
    async fn call(
        &self,
        _args: serde_json::Value,
        _ctx: &ToolCtx,
    ) -> std::result::Result<String, ToolError> {
        tokio::time::sleep(self.duration).await;
        Ok(self.output.clone())
    }
}

/// Sleeps, then fails fatally: models a failure landing mid-run, well after
/// the tools launched (cleanup-grace tests).
pub struct LateFailTool {
    name: String,
    delay: Duration,
    message: String,
}

impl LateFailTool {
    pub fn boxed(name: &str, delay: Duration, message: &str) -> Box<dyn Tool> {
        Box::new(Self {
            name: name.to_owned(),
            delay,
            message: message.to_owned(),
        })
    }
}

#[async_trait::async_trait]
impl Tool for LateFailTool {
    fn spec(&self) -> ToolSpec {
        plain_spec(&self.name)
    }
    async fn call(
        &self,
        _args: serde_json::Value,
        _ctx: &ToolCtx,
    ) -> std::result::Result<String, ToolError> {
        tokio::time::sleep(self.delay).await;
        Err(ToolError::Fatal(Error::Run(self.message.clone())))
    }
}

/// Records cancellation reaching an in-flight tool.
pub struct CancelProbeTool {
    pub flag: Arc<AtomicUsize>,
}

impl CancelProbeTool {
    pub fn boxed() -> (Box<dyn Tool>, Arc<AtomicUsize>) {
        let flag = Arc::new(AtomicUsize::new(0));
        let tool = Box::new(Self { flag: flag.clone() });
        (tool, flag)
    }
}

#[async_trait::async_trait]
impl Tool for CancelProbeTool {
    fn spec(&self) -> ToolSpec {
        plain_spec("probe")
    }
    async fn call(
        &self,
        _args: serde_json::Value,
        ctx: &ToolCtx,
    ) -> std::result::Result<String, ToolError> {
        tokio::select! {
            _ = ctx.token.cancelled() => {
                self.flag.store(1, Ordering::SeqCst);
                Ok("probe cancelled".to_owned())
            }
            _ = tokio::time::sleep(Duration::from_secs(30)) => Ok("probe timed out".to_owned()),
        }
    }
}

/// Run fixture with temporary working and spill directories, seeded with a user
/// message.
pub async fn config(
    replies: Vec<Reply>,
    tools: Vec<Box<dyn Tool>>,
    cwd: &Path,
) -> (RunConfig, ScriptedHandle) {
    let mut toolset = mason::Toolset::new();
    toolset.extend(tools).expect("the fixture tools are unique");
    let model = ScriptedModel::new(replies);
    let config = RunConfig {
        model: model.clone(),
        tools: Arc::new(toolset),
        prompts: PromptSource::Embedded
            .load()
            .expect("the embedded prompts are valid"),
        ctx: ToolCtx::new(cwd).with_spill(Arc::new(OutputSpill::new(cwd.join("spill")))),
        limits: mason::Limits::default(),
        messages: vec![Message::user("go")],
        run_id: None,
        events: None,
        persist: None,
        steer: None,
        compact: None,
        max_tokens: None,
        close_scope: true,
    };
    (config, model)
}

/// Tool result contents in conversation order.
pub fn tool_results(messages: &[Message]) -> Vec<String> {
    messages
        .iter()
        .filter(|message| message.role == Role::Tool)
        .map(|message| message.content.clone().unwrap_or_default())
        .collect()
}

/// A persistence callback that records every committed message.
pub fn recorder() -> (mason::engine::PersistFn, Arc<Mutex<Vec<Message>>>) {
    let saved: Arc<Mutex<Vec<Message>>> = Arc::new(Mutex::new(Vec::new()));
    let callback: mason::engine::PersistFn = Arc::new({
        let saved = saved.clone();
        move |_turn, message| {
            let saved = saved.clone();
            Box::pin(async move {
                saved.lock().expect("unpoisoned").push(message);
                Ok(())
            })
        }
    });
    (callback, saved)
}
