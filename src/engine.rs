//! Streaming model/tool loop with bounded concurrency, persistence, and cancellation.

use std::collections::{HashMap, HashSet, VecDeque};
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use futures_util::StreamExt;
use serde_json::Value;
use tokio::sync::{Mutex, mpsc};
use tokio::task::JoinSet;
use tokio::time::{Instant, sleep, sleep_until};
use tokio_util::sync::CancellationToken;

use crate::Error;
use crate::context::{estimate_str, estimate_tokens, prune_with_budget, strip_stale_reasoning};
use crate::error::Result;
use crate::llm::message::TurnAccumulator;
use crate::llm::{ChatDelta, ChatModel, ChatRequest, ChatTurn, LlmError, Message, ToolCall, Usage};
use crate::prompt::PromptSet;
use crate::tool::{Keep, ToolCtx, ToolError, Toolset};

/// Repeated identical calls within the sliding window trigger a warning.
const DOOM_LOOP_THRESHOLD: usize = 3;
const DOOM_LOOP_WINDOW: usize = 10;
const MAX_DOOM_LOOP_WARNINGS: usize = 3;
pub const DEFAULT_RUN_TIMEOUT: Duration = Duration::from_secs(15 * 60);
/// Cleanup grace after cancellation, before remaining tool calls are aborted.
pub const CLEANUP_GRACE: Duration = Duration::from_secs(5);

/// Per-run limits.
#[derive(Debug, Clone, PartialEq)]
pub struct Limits {
    /// Model calls before the run must wrap up without tools.
    pub max_steps: usize,
    /// Tool calls executing at the same time.
    pub max_concurrent_tools: usize,
    /// Tool calls per model reply.
    pub max_tool_calls_per_step: usize,
    /// `task` (subagent) calls per model reply.
    pub max_task_calls_per_step: usize,
    /// Wall-clock budget for the whole run, shared with children through
    /// the cancellation token.
    pub timeout: Duration,
    pub max_total_tool_calls: usize,
    pub max_total_tokens: usize,
    pub max_response_bytes: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_steps: 50,
            max_concurrent_tools: 4,
            max_tool_calls_per_step: 16,
            max_task_calls_per_step: 4,
            timeout: DEFAULT_RUN_TIMEOUT,
            max_total_tool_calls: 1000,
            max_total_tokens: 20_000_000,
            max_response_bytes: 8 * 1024 * 1024,
        }
    }
}

/// Budget shared by the root and its descendants. Leaf tools share permits;
/// delegation does not hold a permit while waiting for child work.
#[derive(Debug)]
pub struct RunBudget {
    calls: AtomicUsize,
    tokens: AtomicUsize,
    max_calls: usize,
    max_tokens: usize,
    tool_slots: tokio::sync::Semaphore,
}
impl RunBudget {
    fn new(limits: &Limits) -> Self {
        Self {
            calls: AtomicUsize::new(0),
            tokens: AtomicUsize::new(0),
            max_calls: limits.max_total_tool_calls,
            max_tokens: limits.max_total_tokens,
            tool_slots: tokio::sync::Semaphore::new(limits.max_concurrent_tools),
        }
    }
    fn charge_tokens(&self, tokens: usize) -> Result<()> {
        if self
            .tokens
            .fetch_add(tokens, Ordering::SeqCst)
            .saturating_add(tokens)
            > self.max_tokens
        {
            return Err(Error::BudgetExceeded("token"));
        }
        Ok(())
    }
    fn tokens_left(&self) -> usize {
        self.max_tokens
            .saturating_sub(self.tokens.load(Ordering::SeqCst))
    }
    fn calls_exhausted(&self) -> bool {
        self.calls.load(Ordering::SeqCst) >= self.max_calls
    }
}

/// Lossy observer events. Slow or disconnected consumers never block the run.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "kind", content = "content", rename_all = "camelCase")]
pub enum RunEvent {
    /// Transient delta keyed by run, turn, and sequence; committed messages supersede
    /// it.
    #[serde(rename_all = "camelCase")]
    Delta {
        run_id: Arc<str>,
        turn: u32,
        seq: u32,
        delta: ChatDelta,
    },
    /// Committed message replacing provisional text for its run and turn.
    #[serde(rename_all = "camelCase")]
    Committed {
        run_id: Arc<str>,
        turn: u32,
        message: Message,
    },
    /// A tool call is about to execute.
    #[serde(rename_all = "camelCase")]
    ToolStarted {
        run_id: Arc<str>,
        turn: u32,
        call_id: String,
        name: String,
        arguments: String,
    },
}

/// Bounded event queue handle for [`RunConfig::events`].
pub type EventSink = mpsc::Sender<RunEvent>;

/// Persists a completed message; failure stops the run.
pub type PersistFn =
    Arc<dyn Fn(u32, Message) -> Pin<Box<dyn Future<Output = Result<()>> + Send>> + Send + Sync>;
/// Pulls already-saved steering messages between steps.
pub type SteerFn =
    Arc<dyn Fn() -> Pin<Box<dyn Future<Output = Result<Vec<Message>>> + Send>> + Send + Sync>;
/// Returns compacted history, or `None`; stored compactors may reload their source.
pub type CompactFn = Arc<
    dyn Fn(
            Vec<Message>,
            CancellationToken,
        ) -> Pin<Box<dyn Future<Output = Result<Option<Vec<Message>>>> + Send>>
        + Send
        + Sync,
>;

/// Everything one engine run needs.
pub struct RunConfig {
    pub model: Arc<dyn ChatModel>,
    /// The tool registry (specs already computed at registration).
    pub tools: Arc<Toolset>,
    /// This run's immutable prompt snapshot.
    pub prompts: Arc<PromptSet>,
    /// Prototype context for tool calls; the run installs its own token.
    pub ctx: ToolCtx,
    /// Limits for this run.
    pub limits: Limits,
    /// The conversation so far: a system message, then user/assistant turns.
    pub messages: Vec<Message>,
    /// Run ID for event correlation; generated when absent.
    pub run_id: Option<String>,
    /// Optional lossy observer queue.
    pub events: Option<EventSink>,
    /// Required message persistence when a store is configured.
    pub persist: Option<PersistFn>,
    /// Steering between steps.
    pub steer: Option<SteerFn>,
    /// Compaction callback.
    pub compact: Option<CompactFn>,
    /// Compact before a request estimated above this many tokens.
    pub max_tokens: Option<usize>,
    /// Whether the run owns its tool scope and closes it at the end; a stored
    /// session keeps background processes and MCP connections between runs.
    pub close_scope: bool,
}

/// What a finished run leaves behind.
#[derive(Debug)]
pub struct RunOutcome {
    /// The final assistant text.
    pub answer: String,
    /// The full conversation, including every tool result.
    pub messages: Vec<Message>,
    /// Latest provider usage report; context management still uses request estimates.
    pub usage: Option<Usage>,
}

/// Runs the loop with a fresh cancellation token.
pub async fn run(config: RunConfig) -> Result<RunOutcome> {
    run_with_token(config, CancellationToken::new()).await
}

/// Runs the loop under an external token (the runner's per-session token).
pub async fn run_with_token(
    mut config: RunConfig,
    external: CancellationToken,
) -> Result<RunOutcome> {
    let limits = config.limits.clone();
    for value in [
        limits.max_concurrent_tools,
        limits.max_tool_calls_per_step,
        limits.max_task_calls_per_step,
        limits.max_total_tool_calls,
        limits.max_total_tokens,
        limits.max_response_bytes,
    ] {
        if value < 1 {
            return Err(Error::Config(
                "tool limits must be positive integers".to_owned(),
            ));
        }
    }
    if limits.timeout <= Duration::ZERO {
        return Err(Error::Config("the run timeout must be positive".to_owned()));
    }
    if config.ctx.budget.is_none() {
        config.ctx.budget = Some(Arc::new(RunBudget::new(&limits)));
    }
    let scope = config.close_scope.then(|| config.ctx.scope());
    let token = external.child_token();
    let deadline_fired = Arc::new(AtomicBool::new(false));
    let timer = {
        let token = token.clone();
        let flag = deadline_fired.clone();
        let timeout = limits.timeout;
        tokio::spawn(async move {
            sleep(timeout).await;
            flag.store(true, Ordering::SeqCst);
            token.cancel();
        })
    };
    let result = run_loop(config, &limits, token.clone()).await;
    timer.abort();
    token.cancel();
    if let Some(scope) = scope {
        scope.close().await;
    }
    result.map_err(|err| {
        normalize_cancellation(err, &external, &token, &deadline_fired, limits.timeout)
    })
}

/// Maps cancellation-class errors onto their run-level cause: an external
/// interrupt stays `Cancelled`, the deadline becomes `RunDeadline`.
fn normalize_cancellation(
    err: Error,
    external: &CancellationToken,
    token: &CancellationToken,
    deadline_fired: &AtomicBool,
    timeout: Duration,
) -> Error {
    let cancellation_class = matches!(
        err,
        Error::Cancelled
            | Error::Llm(LlmError::Cancelled)
            | Error::Llm(LlmError::TurnDeadline { .. })
    );
    if cancellation_class
        && token.is_cancelled()
        && !external.is_cancelled()
        && deadline_fired.load(Ordering::SeqCst)
    {
        return Error::RunDeadline(timeout);
    }
    if matches!(err, Error::Llm(LlmError::Cancelled)) {
        return Error::Cancelled;
    }
    err
}

/// Shared state across a run's workers.
struct RunShared {
    max_response_bytes: usize,
    conversation: Mutex<Vec<Message>>,
    doom: DoomState,
    tools: Arc<Toolset>,
    ctx: ToolCtx,
    events: Option<EventSink>,
    persist: Option<PersistFn>,
    /// Stored or generated run ID used by every event.
    run_id: Arc<str>,
    /// Spill files of oversized tool results, by tool call id; the view is
    /// rebuilt every step but each result is saved once.
    spilled: StdMutex<HashMap<String, PathBuf>>,
    /// Files this run wrote or edited, relative to the working directory, for
    /// the profile's before-finish checks.
    edited: StdMutex<std::collections::BTreeSet<String>>,
}

impl RunShared {
    /// Persists messages in conversation order, allowing bounded cleanup after
    /// cancellation.
    async fn commit(&self, turn: u32, message: Message) -> Result<()> {
        // Hold the lock through persistence to preserve durable conversation order.
        let mut conversation = self.conversation.lock().await;
        conversation.push(message.clone());
        if let Some(persist) = &self.persist {
            let mut future = persist(turn, message.clone());
            tokio::select! {
                result = &mut future => result?,
                () = self.ctx.token.cancelled() => {
                    tokio::select! {
                        result = &mut future => result?,
                        () = sleep(CLEANUP_GRACE) => return Err(Error::Cancelled),
                    }
                }
            }
        }
        self.emit(RunEvent::Committed {
            run_id: self.run_id.clone(),
            turn,
            message,
        });
        Ok(())
    }

    /// Drops events on full or closed queues without blocking the run.
    fn emit(&self, event: RunEvent) {
        if let Some(events) = &self.events
            && let Err(err) = events.try_send(event)
        {
            tracing::debug!(%err, "dropping a run event for a slow or gone observer");
        }
    }
}

/// Races a run callback against cancellation.
async fn guarded<T>(
    token: &CancellationToken,
    future: impl Future<Output = Result<T>>,
) -> Result<T> {
    tokio::select! {
        result = future => result,
        () = token.cancelled() => Err(Error::Cancelled),
    }
}

/// Doom-loop detection state.
#[derive(Default)]
struct DoomState {
    calls: StdMutex<VecDeque<String>>,
    warnings: AtomicUsize,
    warned_tools: StdMutex<HashSet<String>>,
}

impl DoomState {
    fn record_call(&self, key: &str) -> usize {
        let mut calls = self.calls.lock().expect("the doom state is not poisoned");
        calls.push_back(key.to_owned());
        if calls.len() > DOOM_LOOP_WINDOW {
            calls.pop_front();
        }
        calls.iter().filter(|k| *k == key).count()
    }

    fn warn(&self, name: &str) -> usize {
        self.warned_tools
            .lock()
            .expect("the doom state is not poisoned")
            .insert(name.to_owned());
        self.warnings.fetch_add(1, Ordering::SeqCst) + 1
    }

    fn warned_tools(&self) -> HashSet<String> {
        self.warned_tools
            .lock()
            .expect("the doom state is not poisoned")
            .clone()
    }

    fn warnings(&self) -> usize {
        self.warnings.load(Ordering::SeqCst)
    }

    /// A step without a warning ends the streak: only sustained repetition
    /// stops a run, not three scattered repeats over a long one.
    fn clear_warnings(&self) {
        self.warnings.store(0, Ordering::SeqCst);
        self.warned_tools
            .lock()
            .expect("the doom state is not poisoned")
            .clear();
    }
}

async fn run_loop(
    config: RunConfig,
    limits: &Limits,
    token: CancellationToken,
) -> Result<RunOutcome> {
    let RunConfig {
        model,
        tools,
        prompts,
        ctx,
        messages,
        run_id,
        events,
        persist,
        steer,
        compact,
        max_tokens,
        ..
    } = config;
    let specs = tools.specs();
    let spec_json = serde_json::to_string(&specs).unwrap_or_default();
    let spec_tokens = estimate_str(&spec_json);
    let run_id: Arc<str> = match run_id {
        Some(run_id) => Arc::from(run_id.as_str()),
        None => Arc::from(uuid::Uuid::new_v4().to_string().as_str()),
    };
    let shared = Arc::new(RunShared {
        conversation: Mutex::new(messages),
        max_response_bytes: limits.max_response_bytes,
        doom: DoomState::default(),
        tools,
        ctx: ctx.with_token(token.clone()),
        events,
        persist,
        run_id,
        spilled: StdMutex::new(HashMap::new()),
        edited: StdMutex::new(Default::default()),
    });
    let mut finish_checks = 0;
    let mut usage: Option<Usage> = None;
    let run_budget = shared.ctx.budget.clone().expect("run budget");
    // Leaves the final text-only turn time to answer before the hard deadline.
    let wrap_up_at = Instant::now() + limits.timeout - limits.timeout / 10;
    let mut output_limit_hit = false;
    // One recovery attempt per run, within the existing step/time/token budgets.
    let mut recovery_used = false;
    let mut recovery_pending = false;
    let mut recovery_tools = HashSet::new();

    for step in 1usize.. {
        if token.is_cancelled() {
            return Err(Error::Cancelled);
        }
        let mut last_step =
            step > limits.max_steps || run_budget.calls_exhausted() || Instant::now() >= wrap_up_at;
        // Leave late steering queued for a new run with a full tool budget.
        if !last_step && let Some(steer) = &steer {
            let steered = guarded(&token, steer()).await?;
            shared.conversation.lock().await.extend(steered);
        }
        let budget = max_tokens.map(|max_tokens| max_tokens.saturating_sub(spec_tokens));
        let mut view = {
            let view = request_view(&shared.conversation.lock().await, budget);
            // Pruning was not enough: compact, then rebuild the view.
            if budget.is_some_and(|budget| estimate_tokens(&view) > budget)
                && let Some(compact) = &compact
                && compact_in_place(compact, &shared, &token).await?
            {
                request_view(&shared.conversation.lock().await, budget)
            } else {
                view
            }
        };
        if let Some(budget) = budget {
            fit_tool_view(&mut view, budget, &shared).await?;
        }
        // Wrap up while one more request of this size still fits the token budget.
        if (estimate_tokens(&view) + spec_tokens).saturating_mul(2) > run_budget.tokens_left() {
            last_step = true;
        }

        // The final text-only turn already withholds every tool.
        let recovering = recovery_pending && !last_step;
        let send = |mut messages: Vec<Message>| {
            let tools: Vec<_> = if last_step {
                Vec::new()
            } else {
                specs
                    .iter()
                    .filter(|spec| !recovering || !recovery_tools.contains(&spec.function.name))
                    .cloned()
                    .collect()
            };
            if output_limit_hit {
                messages.push(Message::user(prompts.output_limit.clone()));
            }
            if last_step {
                messages.push(Message::user(prompts.max_steps.clone()));
            }
            if recovering {
                let mut names: Vec<_> = recovery_tools.iter().cloned().collect();
                names.sort();
                let next = if tools.is_empty() {
                    "No tools are available for this turn. If the task is complete, answer now; \
                     otherwise report the blocker and unfinished work."
                } else {
                    "Use the evidence already collected to make the next concrete change or validation \
                     with another available tool. If the task is complete, answer now. If you cannot \
                     make progress, report the blocker and unfinished work. Do not repeat the inspection."
                };
                messages.push(Message::user(format!(
                    "[Mason runtime recovery] Repeated tool calls exhausted the doom-loop warning budget. \
                     The following tools are unavailable for this turn: {}. {next} \
                     This is the only recovery attempt; run budgets still apply.",
                    names.join(", ")
                )));
                strip_stale_reasoning(&mut messages);
            }
            ChatRequest { messages, tools }
        };

        let mut turn = run_turn(&model, send(view), &token, &shared, step as u32).await;
        if let Err(Error::Llm(llm_err)) = &turn
            && llm_err.is_context_overflow()
            && let Some(compact) = &compact
        {
            // Compact once and retry if the provider rejects the context estimate.
            let err = turn.unwrap_err();
            if compact_in_place(compact, &shared, &token).await? {
                let mut view = request_view(&shared.conversation.lock().await, budget);
                if let Some(budget) = budget {
                    fit_tool_view(&mut view, budget, &shared).await?;
                }
                turn = run_turn(&model, send(view), &token, &shared, step as u32).await;
            } else {
                return Err(err);
            }
        }
        let chat_turn = turn?;
        // A cut-off reply's tool calls are not trustworthy: discard it and ask
        // for smaller pieces. The final text-only turn keeps its partial answer.
        output_limit_hit = chat_turn.finish_reason.as_deref() == Some("length") && !last_step;
        if output_limit_hit {
            continue;
        }
        if chat_turn.finish_reason.as_deref() == Some("content_filter") {
            return Err(Error::Run(
                "the model stopped early (finish_reason=content_filter)".to_owned(),
            ));
        }
        if chat_turn.usage.is_some() {
            usage = chat_turn.usage;
        }
        let assistant = chat_turn.message;

        shared.commit(step as u32, assistant.clone()).await?;

        let calls = assistant.tool_calls.clone().unwrap_or_default();
        if recovering
            && calls
                .iter()
                .any(|call| recovery_tools.contains(&call.function.name))
        {
            // Enforce the recovery restriction even if the model calls an omitted
            // tool. Save a result for every call; execute none of this reply.
            for call in &calls {
                shared.commit(step as u32, Message::tool(
                    call.id.clone(),
                    "Error: no tools executed because this reply called a tool disabled during doom-loop recovery.",
                )).await?;
            }
            return Err(Error::Run(format!(
                "Stopped after {MAX_DOOM_LOOP_WARNINGS} doom-loop warnings: the model repeated a disabled tool during recovery"
            )));
        }
        if (calls.is_empty() || last_step)
            && let Some(failures) = before_finish(&shared).await?
        {
            // no turn left to repair in, or the repairs are spent: the run did not succeed
            if last_step || finish_checks == MAX_FINISH_CHECKS {
                return Err(Error::Run(format!(
                    "the profile's before-finish checks still fail:\n\n{failures}"
                )));
            }
            finish_checks += 1;
            shared
                .commit(
                    step as u32,
                    Message::user(format!(
                        "[Mason runtime check] This profile checks edited files before a run \
                         may finish, and some checks failed:\n\n{failures}\n\nFix them, then finish."
                    )),
                )
                .await?;
            continue;
        }
        if calls.is_empty() || last_step {
            let messages = std::mem::take(&mut *shared.conversation.lock().await);
            return Ok(RunOutcome {
                answer: assistant.content.unwrap_or_default(),
                messages,
                usage,
            });
        }
        recovery_pending = false;

        let warnings_before = shared.doom.warnings();
        execute_calls(&shared, calls, limits, &token, step as u32).await?;

        if shared.doom.warnings() == warnings_before {
            shared.doom.clear_warnings();
        } else if shared.doom.warnings() >= MAX_DOOM_LOOP_WARNINGS {
            if recovery_used {
                return Err(Error::Run(format!(
                    "Stopped after {MAX_DOOM_LOOP_WARNINGS} doom-loop warnings following a recovery attempt"
                )));
            }
            recovery_tools = shared.doom.warned_tools();
            shared.doom.clear_warnings();
            recovery_used = true;
            recovery_pending = true;
            tracing::warn!(tools = ?recovery_tools, "doom-loop warning budget exhausted; attempting one recovery turn");
        }
    }
    unreachable!("the loop returns through the answer or an error")
}

/// How often a run that tries to finish is sent back to fix failed checks.
const MAX_FINISH_CHECKS: usize = 3;

/// Runs a profile hook on one edited file: whether it passed, and its output.
async fn run_hook(
    shared: &RunShared,
    hook: &crate::profile::Hook,
    path: &str,
) -> Result<(bool, String)> {
    // an interrupt or the run deadline ends a hook like any step: after the
    // cleanup grace an ordinary call gets
    let call = hook_call(shared, hook, path);
    tokio::pin!(call);
    tokio::select! {
        result = &mut call => result,
        () = shared.ctx.token.cancelled() => {
            let _ = tokio::time::timeout(CLEANUP_GRACE, &mut call).await;
            Err(Error::Cancelled)
        }
    }
}

async fn hook_call(
    shared: &RunShared,
    hook: &crate::profile::Hook,
    path: &str,
) -> Result<(bool, String)> {
    let Some(tool) = shared.tools.get(&hook.tool) else {
        return Ok((
            false,
            format!("the {} tool is not available to this run", hook.tool),
        ));
    };
    // the authorization an ordinary call of the tool would get; the call budget
    // bounds the model, not the checks the profile makes of its edits
    let args = hook.args_for(path);
    let ctx = &shared.ctx;
    if let Err(err) = ctx.policy.authorize(&hook.tool, &args, ctx).await {
        return match err {
            ToolError::Failure(message) => Ok((false, format!("permission denied: {message}"))),
            ToolError::Fatal(err) => Err(err),
        };
    }
    if let Some(denied) = ctx
        .rules
        .as_ref()
        .and_then(|rules| rules.check(&hook.tool, &args, &ctx.cwd))
    {
        return Ok((false, format!("permission denied: {denied}")));
    }
    // a run cancelled while the call waited for approval starts no tool in its grace
    if ctx.token.is_cancelled() {
        return Err(Error::Cancelled);
    }
    match tool.call(args, ctx).await {
        Ok(output) => Ok((true, output)),
        Err(ToolError::Failure(message)) => Ok((false, message)),
        Err(ToolError::Fatal(err)) => Err(err),
    }
}

/// The failed before-finish checks of the files this run edited, if any failed.
async fn before_finish(shared: &RunShared) -> Result<Option<String>> {
    let Some(rules) = shared.ctx.rules.clone() else {
        return Ok(None);
    };
    let edited: Vec<String> = shared
        .edited
        .lock()
        .expect("edited files")
        .iter()
        .cloned()
        .collect();
    let mut failures = Vec::new();
    for hook in &rules.hooks.before_finish {
        for path in edited
            .iter()
            .filter(|path| crate::profile::glob_match(&hook.pattern, path))
        {
            // a finish check is a leaf call: it takes a slot of the shared concurrency
            // limit, as an after-edit check runs in the slot of its edit
            let _permit = match &shared.ctx.budget {
                Some(budget) => Some(tokio::select! {
                    permit = budget.tool_slots.acquire() => permit.expect("tool semaphore stays open"),
                    () = shared.ctx.token.cancelled() => return Err(Error::Cancelled),
                }),
                None => None,
            };
            let (passed, output) = run_hook(shared, hook, path).await?;
            if !passed {
                failures.push(format!("{} on {path}:\n{output}", hook.tool));
            }
        }
    }
    let failures = shared
        .ctx
        .spill
        .truncate(&failures.join("\n\n"), Keep::Head)
        .await
        .map_err(|err| match err {
            ToolError::Fatal(err) => err,
            ToolError::Failure(message) => Error::Run(message),
        })?;
    Ok((!failures.is_empty()).then_some(failures))
}

/// The after-edit hook results for a file a `write` or `edit` just changed.
async fn after_edit(shared: &RunShared, path: &str) -> Result<String> {
    let Some(rules) = shared.ctx.rules.clone() else {
        return Ok(String::new());
    };
    let mut notes = String::new();
    for hook in rules
        .hooks
        .after_edit
        .iter()
        .filter(|hook| crate::profile::glob_match(&hook.pattern, path))
    {
        let (passed, output) = run_hook(shared, hook, path).await?;
        let verdict = if passed { "passed" } else { "FAILED" };
        notes.push_str(&format!(
            "\n\n[{} on {path} after this edit: {verdict}]\n{output}",
            hook.tool
        ));
    }
    Ok(notes)
}

/// The messages to send: stale reasoning stripped and, when over budget, old tool
/// outputs pruned. The conversation itself keeps everything, so compaction
/// summarizes original results rather than placeholders.
fn request_view(conversation: &[Message], budget: Option<usize>) -> Vec<Message> {
    let mut view = conversation.to_vec();
    strip_stale_reasoning(&mut view);
    if budget.is_some_and(|budget| estimate_tokens(&view) > budget) {
        view = prune_with_budget(&view, budget.unwrap_or(usize::MAX));
    }
    view
}

/// Only the sent view is shortened. Conversation/compaction retain the original
/// results, and the model can page the saved content through read/grep.
async fn fit_tool_view(view: &mut [Message], budget: usize, shared: &RunShared) -> Result<()> {
    if estimate_tokens(view) <= budget {
        return Ok(());
    }
    let mut baseline = view.to_vec();
    let count = baseline
        .iter_mut()
        .filter(|message| message.role == crate::llm::Role::Tool)
        .map(|message| {
            message.content = None;
        })
        .count();
    let base = estimate_tokens(&baseline);
    if count == 0 || base >= budget {
        return Err(Error::Config("prompt/tool definitions cannot fit the context; shorten the prompt or use a larger model".into()));
    }
    let bytes = budget
        .saturating_sub(base)
        .saturating_sub(32)
        .saturating_mul(4)
        / count;
    for message in view
        .iter_mut()
        .filter(|message| message.role == crate::llm::Role::Tool)
    {
        let Some(content) = &mut message.content else {
            continue;
        };
        if content.len() <= bytes {
            continue;
        }
        let cached = message.tool_call_id.as_deref().and_then(|id| {
            shared
                .spilled
                .lock()
                .expect("spill cache mutex")
                .get(id)
                .cloned()
        });
        let path = match cached {
            Some(path) => path,
            None => {
                let path = shared
                    .ctx
                    .spill
                    .save(content)
                    .await
                    .map_err(|error| match error {
                        ToolError::Fatal(error) => error,
                        ToolError::Failure(error) => Error::Run(error),
                    })?;
                if let Some(id) = &message.tool_call_id {
                    shared
                        .spilled
                        .lock()
                        .expect("spill cache mutex")
                        .insert(id.clone(), path.clone());
                }
                path
            }
        };
        let notice = format!("\n[Shortened for context; full result: {}]", path.display());
        if notice.len() > bytes {
            return Err(Error::Config(
                "insufficient context for tool results; use a larger model or fewer tools".into(),
            ));
        }
        let mut end = bytes - notice.len();
        while !content.is_char_boundary(end) {
            end -= 1;
        }
        content.truncate(end);
        content.push_str(&notice);
    }
    Ok(())
}

async fn compact_in_place(
    compact: &CompactFn,
    shared: &RunShared,
    token: &CancellationToken,
) -> Result<bool> {
    let messages = shared.conversation.lock().await.clone();
    let compacted = guarded(token, compact(messages, token.clone())).await?;
    if let Some(messages) = compacted {
        *shared.conversation.lock().await = messages;
        Ok(true)
    } else {
        Ok(false)
    }
}

/// One model turn: stream, forward deltas lossily, fold into the message.
async fn run_turn(
    model: &Arc<dyn ChatModel>,
    request: ChatRequest,
    token: &CancellationToken,
    shared: &RunShared,
    turn: u32,
) -> Result<ChatTurn> {
    let budget = shared.ctx.budget.as_ref().expect("run budget");
    let input_tokens = estimate_tokens(&request.messages)
        + estimate_str(&serde_json::to_string(&request.tools).unwrap_or_default());
    budget.charge_tokens(input_tokens)?;
    let stream = model.stream(request, token).await?;
    futures_util::pin_mut!(stream);
    let mut accumulator = TurnAccumulator::new().with_max_bytes(shared.max_response_bytes);
    let mut seq = 0u32;
    while let Some(item) = stream.next().await {
        let delta = item?;
        shared.emit(RunEvent::Delta {
            run_id: shared.run_id.clone(),
            turn,
            seq,
            delta: delta.clone(),
        });
        seq += 1;
        if accumulator.push(delta) {
            break;
        }
    }
    let turn = accumulator.finish()?;
    let reported_prompt = turn
        .usage
        .as_ref()
        .map_or(0, |usage| usage.prompt_tokens as usize);
    let completion = turn.usage.as_ref().map_or_else(
        || estimate_tokens(std::slice::from_ref(&turn.message)),
        |usage| usage.completion_tokens as usize,
    );
    budget.charge_tokens(
        reported_prompt
            .saturating_sub(input_tokens)
            .saturating_add(completion),
    )?;
    Ok(turn)
}

/// Executes calls with bounded concurrency; fatal errors cancel siblings with cleanup
/// grace.
async fn execute_calls(
    shared: &Arc<RunShared>,
    calls: Vec<ToolCall>,
    limits: &Limits,
    token: &CancellationToken,
    turn: u32,
) -> Result<()> {
    let calls = Arc::new(calls);
    let next = Arc::new(AtomicUsize::new(0));
    let tasks_used = Arc::new(AtomicUsize::new(0));
    let mut join = JoinSet::new();
    let workers = limits.max_concurrent_tools.min(calls.len());
    for _ in 0..workers {
        let shared = shared.clone();
        let calls = calls.clone();
        let next = next.clone();
        let tasks_used = tasks_used.clone();
        let token = token.clone();
        let limits = limits.clone();
        join.spawn(async move {
            loop {
                if token.is_cancelled() {
                    return Err(Error::Cancelled);
                }
                let index = next.fetch_add(1, Ordering::SeqCst);
                if index >= calls.len() {
                    return Ok(());
                }
                let output = handle_call(
                    &shared,
                    index,
                    &calls[index],
                    &limits,
                    &tasks_used,
                    &token,
                    turn,
                )
                .await?;
                shared
                    .commit(turn, Message::tool(calls[index].id.clone(), output))
                    .await?;
                if token.is_cancelled() {
                    return Err(Error::Cancelled);
                }
            }
        });
    }

    let mut fatal: Option<Error> = None;
    // Start cleanup grace at failure or cancellation, not at tool launch.
    let mut cleanup_deadline: Option<Instant> = None;
    loop {
        tokio::select! {
            outcome = join.join_next(), if !join.is_empty() => {
                let Some(outcome) = outcome else { break };
                match outcome {
                    Ok(Ok(())) => {}
                    Ok(Err(err)) => {
                        if fatal.is_none() {
                            tracing::error!(error = %err, "a tool call failed fatally; cancelling the run");
                            fatal = Some(err);
                            token.cancel();
                            cleanup_deadline = Some(Instant::now() + CLEANUP_GRACE);
                        }
                    }
                    Err(join_err) if join_err.is_panic() => {
                        if fatal.is_none() {
                            fatal = Some(Error::Run(format!("a tool call panicked: {join_err}")));
                            token.cancel();
                            cleanup_deadline = Some(Instant::now() + CLEANUP_GRACE);
                        }
                    }
                    Err(_) => {} // aborted during cleanup: expected
                }
            }
            // Abort tools that ignore cancellation once cleanup grace expires.
            _ = token.cancelled(), if fatal.is_none() && !join.is_empty() => {
                fatal = Some(Error::Cancelled);
                cleanup_deadline = Some(Instant::now() + CLEANUP_GRACE);
            }
            _ = sleep_until(cleanup_deadline.unwrap_or_else(Instant::now)),
                if cleanup_deadline.is_some() && !join.is_empty() =>
            {
                join.abort_all();
            }
            else => break,
        }
    }
    match fatal {
        Some(err) => Err(err),
        None => Ok(()),
    }
}

/// Executes one call with reply-limit and doom-loop checks.
async fn handle_call(
    shared: &RunShared,
    index: usize,
    call: &ToolCall,
    limits: &Limits,
    tasks_used: &AtomicUsize,
    token: &CancellationToken,
    turn: u32,
) -> Result<String> {
    let name = &call.function.name;
    if shared.doom.warnings() >= MAX_DOOM_LOOP_WARNINGS {
        return Ok(
            "Error: tool skipped because repeated calls exhausted the doom-loop warning budget."
                .to_owned(),
        );
    }
    const OVER_LIMIT: &str = "Error: tool skipped because this reply exceeded its tool/task call limit. Narrow the work.";
    if index >= limits.max_tool_calls_per_step {
        return Ok(OVER_LIMIT.to_owned());
    }
    // Count task calls only once they passed the per-reply index check, so which
    // ones run does not depend on worker scheduling.
    if name == "task"
        && tasks_used.fetch_add(1, Ordering::SeqCst) + 1 > limits.max_task_calls_per_step
    {
        return Ok(OVER_LIMIT.to_owned());
    }

    let key = call_key(call);
    let repeats = shared.doom.record_call(&key);
    shared.emit(RunEvent::ToolStarted {
        run_id: shared.run_id.clone(),
        turn,
        call_id: call.id.clone(),
        name: name.clone(),
        arguments: call.function.arguments.clone(),
    });
    let mut output = call_tool(shared, call, token).await?;
    if repeats >= DOOM_LOOP_THRESHOLD {
        shared.doom.warn(name);
        output.push_str(&format!(
            "\n\nYou have made this exact {name} call {repeats} times in your last {DOOM_LOOP_WINDOW} calls. Stop repeating it and try a different approach."
        ));
    }
    Ok(output)
}

/// Returns recoverable failures as tool text; fatal errors and cancellation propagate.
async fn call_tool(
    shared: &RunShared,
    call: &ToolCall,
    token: &CancellationToken,
) -> Result<String> {
    if token.is_cancelled() {
        return Err(Error::Cancelled);
    }
    let name = &call.function.name;
    let Some(tool) = shared.tools.get(name) else {
        return Ok(format!(
            "Error: unknown tool \"{name}\". Available tools: {}",
            shared.tools.names().join(", ")
        ));
    };
    let raw = if call.function.arguments.trim().is_empty() {
        "{}"
    } else {
        &call.function.arguments
    };
    let args: Value = match serde_json::from_str(raw) {
        Ok(args) => args,
        Err(err) => {
            return Ok(format!(
                "Error: the {name} arguments are not valid JSON ({err}). Rewrite the call."
            ));
        }
    };
    let ctx = shared.ctx.clone().with_token(token.clone());
    let budget = ctx.budget.as_ref().expect("run budget");
    if budget.calls.fetch_add(1, Ordering::SeqCst) >= budget.max_calls {
        return Ok(
            "Error: tool skipped because the run's tool-call budget is exhausted.".to_owned(),
        );
    }
    let _permit = if name == "task" {
        None
    } else {
        Some(tokio::select! {
            _ = token.cancelled() => return Err(Error::Cancelled),
            permit = budget.tool_slots.acquire() => permit.expect("tool semaphore stays open"),
        })
    };
    tokio::select! {
        _ = token.cancelled() => return Err(Error::Cancelled),
        authorized = ctx.policy.authorize(name, &args, &ctx) => {
            if let Err(err) = authorized {
                return match err {
                    ToolError::Failure(message) => Ok(format!("Error: permission denied: {message}")),
                    ToolError::Fatal(err) => Err(err),
                };
            }
        }
    }
    if let Some(rules) = &ctx.rules
        && let Some(denied) = rules.check(name, &args, &ctx.cwd)
    {
        return Ok(format!("Error: permission denied: {denied}"));
    }
    let edited = matches!(name.as_str(), "write" | "edit")
        .then(|| args.get("filePath").and_then(Value::as_str))
        .flatten()
        .map(str::to_owned);
    match tool.call(args, &ctx).await {
        Ok(mut output) => {
            // the file the write changed, so hooks match it whatever alias reached it
            if let Some(path) =
                edited.and_then(|path| crate::profile::written_file(&ctx.cwd, &path))
            {
                shared
                    .edited
                    .lock()
                    .expect("edited files")
                    .insert(path.clone());
                output.push_str(&after_edit(shared, &path).await?);
            }
            match shared.ctx.spill.truncate(&output, Keep::Head).await {
                Ok(text) => Ok(text),
                Err(ToolError::Failure(message)) => Ok(format!("Error: {message}")),
                Err(ToolError::Fatal(err)) => Err(err),
            }
        }
        Err(ToolError::Failure(message)) => {
            if token.is_cancelled() {
                // Cancellation wins over the tool's own error report.
                return Err(Error::Cancelled);
            }
            shared
                .ctx
                .spill
                .truncate(&format!("Error: {message}"), Keep::Head)
                .await
                .map_err(|err| match err {
                    ToolError::Fatal(err) => err,
                    ToolError::Failure(_) => Error::Run("cannot bound failed tool output".into()),
                })
        }
        Err(ToolError::Fatal(err)) => Err(err),
    }
}

/// Sorts JSON keys recursively so argument ordering cannot evade doom-loop detection.
fn call_key(call: &ToolCall) -> String {
    fn canonical(value: &Value) -> Value {
        match value {
            Value::Object(map) => {
                let mut keys: Vec<&String> = map.keys().collect();
                keys.sort();
                let mut sorted = serde_json::Map::new();
                for key in keys {
                    sorted.insert(key.clone(), canonical(&map[key]));
                }
                Value::Object(sorted)
            }
            Value::Array(items) => Value::Array(items.iter().map(canonical).collect()),
            other => other.clone(),
        }
    }
    match serde_json::from_str::<Value>(&call.function.arguments) {
        Ok(args) => format!(
            "{}{}",
            call.function.name,
            serde_json::to_string(&canonical(&args)).unwrap_or_default()
        ),
        Err(_) => format!("{}{}", call.function.name, call.function.arguments),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ToolCall;

    fn call(name: &str, arguments: &str) -> ToolCall {
        ToolCall::new("id", name, arguments)
    }

    #[test]
    fn call_key_normalizes_json_key_order() {
        let a = call_key(&call("read", r#"{"path":"x","limit":2}"#));
        let b = call_key(&call("read", r#"{"limit":2,"path":"x"}"#));
        assert_eq!(a, b);
        let c = call_key(&call("t", r#"{"o":{"a":1,"b":[2,{"d":3,"c":4}]}}"#));
        let d = call_key(&call("t", r#"{"o":{"b":[2,{"c":4,"d":3}],"a":1}}"#));
        assert_eq!(c, d);
        assert_ne!(a, call_key(&call("grep", r#"{"path":"x","limit":2}"#)));
        let broken = call_key(&call("read", "{oops"));
        assert_eq!(broken, "read{oops");
    }

    #[test]
    fn doom_state_counts_within_the_window() {
        let doom = DoomState::default();
        assert_eq!(doom.record_call("k"), 1);
        assert_eq!(doom.record_call("other"), 1);
        assert_eq!(doom.record_call("k"), 2);
        for _ in 0..DOOM_LOOP_WINDOW {
            doom.record_call("filler");
        }
        assert_eq!(doom.record_call("k"), 1, "the window slides");
        assert_eq!(doom.warn("echo"), 1);
        assert_eq!(doom.warnings(), 1);
        assert!(doom.warned_tools().contains("echo"));
        doom.clear_warnings();
        assert_eq!(doom.warnings(), 0);
        assert!(doom.warned_tools().is_empty());
    }

    #[test]
    fn limits_defaults_match_the_source() {
        let limits = Limits::default();
        assert_eq!(limits.max_steps, 50);
        assert_eq!(limits.max_concurrent_tools, 4);
        assert_eq!(limits.max_tool_calls_per_step, 16);
        assert_eq!(limits.max_task_calls_per_step, 4);
        assert_eq!(limits.timeout, Duration::from_secs(15 * 60));
    }
}
