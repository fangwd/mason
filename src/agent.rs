//! Configurable agents, run assembly, and embedding interfaces.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::{Arc, Mutex as StdMutex, Weak};
use std::time::Duration;

use serde_json::json;
use tokio_util::sync::CancellationToken;

use crate::Error;
use crate::context::{summarize, tail_budget, tail_start, usable_tokens};
use crate::engine::{self, EventSink, Limits, PersistFn, RunConfig, RunOutcome, SteerFn};
use crate::error::Result;
use crate::evaluate::{Evaluator, Outcome, Verdict};
use crate::llm::{ChatModel, Message, Role, VendorModel, Vendors};
use crate::memory::{
    Kind as MemoryKind, Memory, MemoryEntry, MemoryTool, index_entries, prompt_index,
};
use crate::profile::{
    LaunchSubagent, Profile, ProfilePatch, ProfileSelector, ProfileSet, SubagentRequest,
    SubagentResult, TaskTool, resolve_children, resolve_profile_text, task_available,
};
use crate::prompt::{Environment, PromptSet, PromptSource};
use crate::reflect::{self, RunReview};
use crate::skill::{Skill, SkillTool, load_skills, skills_entries};
use crate::store::{Delivery, NewSession, Session, SessionId, SessionStore};
use crate::tool::{OutputSpill, Tool, ToolCtx, ToolScope, Toolset};

/// Lists a refreshable tool source's tools again (MCP servers): each top-level
/// run asks, so a server's new or changed tools reach the next run without a
/// restart. An error keeps the tools the source had at build.
pub type ToolRefresh = Arc<
    dyn Fn() -> Pin<
            Box<
                dyn std::future::Future<Output = std::result::Result<Vec<Arc<dyn Tool>>, String>>
                    + Send,
            >,
        > + Send
        + Sync,
>;

/// A stored session's background processes and MCP connections are closed
/// after this long without a run.
pub const SESSION_SCOPE_IDLE: Duration = Duration::from_secs(30 * 60);

/// Cloneable agent with shared model, tools, profiles, prompts, and limits.
#[derive(Clone)]
pub struct Agent {
    inner: Arc<AgentInner>,
}

struct AgentInner {
    model: Arc<dyn ChatModel>,
    tools: Arc<Toolset>,
    instructions: Option<String>,
    prompts: PromptSource,
    cwd: PathBuf,
    limits: Limits,
    cap_profile_steps: bool,
    shell_timeouts: Option<(u64, u64)>,
    spill: Arc<OutputSpill>,
    profiles: Arc<ProfileSet>,
    selector: Option<ProfileSelector>,
    skills_dir: Option<PathBuf>,
    memory: Option<Arc<Memory>>,
    evaluator: Option<Arc<dyn Evaluator>>,
    execution_mode: crate::tool::policy::ExecutionMode,
    allow_network: bool,
    policy: Option<Arc<dyn crate::tool::policy::ToolPolicy>>,
    scopes: Arc<SessionScopes>,
    /// The refreshable source and the names its tools had at build.
    refresh: Option<(ToolRefresh, Vec<String>)>,
}

impl AgentInner {
    /// The installed skills, then the learned skills whose names are free.
    async fn skills(&self) -> Vec<Skill> {
        let mut skills = match &self.skills_dir {
            Some(dir) => load_skills(dir).await,
            None => Vec::new(),
        };
        if let Some(memory) = &self.memory {
            for learned in load_skills(&memory.skills_dir()).await {
                if skills.iter().any(|skill| skill.name == learned.name) {
                    tracing::warn!(
                        skill = learned.name,
                        "skipping a learned skill whose name an installed skill takes"
                    );
                } else {
                    skills.push(learned);
                }
            }
        }
        skills
    }

    /// Loads the skills, the memory index and the refreshed tools a top-level run
    /// and its children share.
    async fn snapshot(&self, token: &CancellationToken) -> Arc<RunSnapshot> {
        // each server's connect timeout bounds the listing; an interrupt ends it
        let refreshed = match &self.refresh {
            Some((refresh, _)) => match tokio::select! {
                listed = refresh() => listed,
                () = token.cancelled() => Err("the run was cancelled".to_owned()),
            } {
                Ok(tools) => Some(tools),
                Err(error) => {
                    tracing::warn!("keeping the tools listed at startup: {error}");
                    None
                }
            },
            None => None,
        };
        Arc::new(RunSnapshot {
            skills: self.skills().await,
            memory: match &self.memory {
                Some(memory) => memory.index().await,
                None => Vec::new(),
            },
            refreshed,
        })
    }
}

/// Skills, the memory index and refreshed tools, snapshotted per top-level run.
#[derive(Default)]
struct RunSnapshot {
    skills: Vec<Skill>,
    memory: Vec<MemoryEntry>,
    /// The refreshable source's tools as listed for this run; `None` keeps the
    /// build-time ones.
    refreshed: Option<Vec<Arc<dyn Tool>>>,
}

/// Tool scopes of stored sessions, kept between their runs until idle.
#[derive(Default)]
struct SessionScopes(StdMutex<HashMap<SessionId, ScopeEntry>>);

struct ScopeEntry {
    scope: ToolScope,
    active: usize,
    epoch: u64,
}

impl SessionScopes {
    fn acquire(self: &Arc<Self>, session: SessionId) -> ScopeLease {
        let mut scopes = self.0.lock().expect("session scope mutex");
        let entry = scopes.entry(session).or_insert_with(|| ScopeEntry {
            scope: ToolScope::default(),
            active: 0,
            epoch: 0,
        });
        entry.active += 1;
        entry.epoch += 1;
        ScopeLease {
            scopes: Arc::downgrade(self),
            session,
            scope: entry.scope.clone(),
        }
    }

    fn release(self: &Arc<Self>, session: SessionId) {
        let epoch = {
            let mut scopes = self.0.lock().expect("session scope mutex");
            let Some(entry) = scopes.get_mut(&session) else {
                return;
            };
            entry.active -= 1;
            entry.epoch
        };
        let scopes = Arc::downgrade(self);
        tokio::spawn(async move {
            tokio::time::sleep(SESSION_SCOPE_IDLE).await;
            let Some(scopes) = scopes.upgrade() else {
                return;
            };
            let idle = {
                let mut map = scopes.0.lock().expect("session scope mutex");
                match map.get(&session) {
                    Some(entry) if entry.active == 0 && entry.epoch == epoch => {
                        map.remove(&session).map(|entry| entry.scope)
                    }
                    _ => None,
                }
            };
            if let Some(scope) = idle {
                scope.close().await;
            }
        });
    }

    async fn close(&self, session: SessionId) {
        let scope = self
            .0
            .lock()
            .expect("session scope mutex")
            .remove(&session)
            .map(|entry| entry.scope);
        if let Some(scope) = scope {
            scope.close().await;
        }
    }

    async fn close_all(&self) {
        let scopes: Vec<ToolScope> = self
            .0
            .lock()
            .expect("session scope mutex")
            .drain()
            .map(|(_, entry)| entry.scope)
            .collect();
        for scope in scopes {
            scope.close().await;
        }
    }
}

/// Holds a session's scope for one run; dropping it starts the idle timer, even
/// when the run is cancelled or aborted.
struct ScopeLease {
    scopes: Weak<SessionScopes>,
    session: SessionId,
    scope: ToolScope,
}

impl Drop for ScopeLease {
    fn drop(&mut self) {
        if let Some(scopes) = self.scopes.upgrade() {
            scopes.release(self.session);
        }
    }
}

impl std::fmt::Debug for Agent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Agent")
            .field("cwd", &self.inner.cwd)
            .field("tools", &self.inner.tools.names())
            .field("profiles", &self.inner.profiles.names())
            .field("limits", &self.inner.limits)
            .finish_non_exhaustive()
    }
}

/// One run's assembled toolset, system prompt and limits.
struct Assembled {
    tools: Arc<Toolset>,
    system: String,
    limits: Limits,
    rules: Arc<crate::profile::Rules>,
}

impl Agent {
    pub fn builder() -> AgentBuilder {
        AgentBuilder::default()
    }

    /// Runs one prompt as `build` without persistence or steering.
    pub async fn run_once(&self, prompt: &str) -> Result<String> {
        self.run_prompt(prompt, None)
            .await
            .map(|outcome| outcome.answer)
    }

    /// [`run_once`](Self::run_once) with a lossy event queue: the CLI streams
    /// text deltas from it while the answer accumulates.
    pub async fn run_once_with_events(&self, prompt: &str, events: EventSink) -> Result<String> {
        self.run_prompt(prompt, Some(events))
            .await
            .map(|outcome| outcome.answer)
    }

    /// The full run outcome (answer plus the conversation) for the `build`
    /// profile.
    pub async fn run_prompt(&self, prompt: &str, events: Option<EventSink>) -> Result<RunOutcome> {
        self.run_prompt_as("build", prompt, events).await
    }

    pub async fn run_prompt_as(
        &self,
        profile: &str,
        prompt: &str,
        events: Option<EventSink>,
    ) -> Result<RunOutcome> {
        self.run_prompt_as_with_token(profile, prompt, events, CancellationToken::new())
            .await
    }

    pub async fn run_prompt_as_with_token(
        &self,
        profile_name: &str,
        prompt: &str,
        events: Option<EventSink>,
        token: CancellationToken,
    ) -> Result<RunOutcome> {
        let inner = self.inner.clone();
        let Some(profile) = inner.profiles.get(profile_name) else {
            return Err(Error::Config(format!(
                "Unknown agent \"{profile_name}\". Agents: {}",
                inner.profiles.names().join(", ")
            )));
        };
        if profile.subagent {
            return Err(Error::Config(format!(
                "\"{profile_name}\" is a subagent profile"
            )));
        }
        let profile = profile.clone();

        // Share prompt, skill and memory snapshots with child runs.
        let prompts = inner.prompts.load()?;
        let snapshot = inner.snapshot(&token).await;
        let usable = crate::context::usable_tokens(inner.model.context_tokens())?;

        let launch: LaunchSubagent = Arc::new({
            let inner = inner.clone();
            let prompts = prompts.clone();
            let snapshot = snapshot.clone();
            move |request: SubagentRequest| {
                let inner = inner.clone();
                let prompts = prompts.clone();
                let snapshot = snapshot.clone();
                Box::pin(async move { run_child(inner, prompts, snapshot, usable, request).await })
            }
        });
        let assembled = assemble(&inner, &prompts, &profile, &snapshot, None, Some(launch))?;

        let messages = vec![Message::system(assembled.system), Message::user(prompt)];
        let ctx = tool_context(&inner)
            .with_spill(inner.spill.clone())
            .with_execution(
                inner.execution_mode,
                inner.allow_network,
                inner
                    .policy
                    .clone()
                    .unwrap_or_else(|| Arc::new(crate::tool::policy::AllowTools)),
            );
        let compact =
            crate::context::in_memory_compactor(inner.model.clone(), prompts.clone(), usable);
        engine::run_with_token(
            RunConfig {
                model: inner.model.clone(),
                tools: assembled.tools,
                prompts,
                ctx: ctx.with_rules(assembled.rules.clone()),
                limits: assembled.limits,
                messages,
                run_id: None,
                events,
                persist: None,
                steer: None,
                compact: Some(compact),
                max_tokens: Some(usable),
                close_scope: true,
            },
            token,
        )
        .await
    }

    /// Runs a claimed session with persisted messages, inbox steering, compaction, and
    /// stored child sessions.
    pub async fn run_session(&self, run: SessionRun) -> Result<String> {
        self.run_session_inner(run, None).await
    }

    pub(crate) async fn run_session_managed(
        &self,
        run: SessionRun,
        runner: Weak<crate::runner::Runner>,
    ) -> Result<String> {
        self.run_session_inner(run, Some(runner)).await
    }

    async fn run_session_inner(
        &self,
        run: SessionRun,
        runner: Option<Weak<crate::runner::Runner>>,
    ) -> Result<String> {
        let inner = self.inner.clone();
        let Some(profile) = inner.profiles.get(&run.profile).cloned() else {
            return Err(Error::Config(format!(
                "Unknown agent \"{}\". Agents: {}",
                run.profile,
                inner.profiles.names().join(", ")
            )));
        };
        if profile.subagent {
            return Err(Error::Config(format!(
                "\"{}\" is a subagent profile",
                run.profile
            )));
        }
        // Share prompt, skill and memory snapshots with child runs.
        let prompts = inner.prompts.load()?;
        let snapshot = inner.snapshot(&run.token).await;
        let usable = usable_tokens(inner.model.context_tokens())?;
        run_saved(
            &inner,
            &prompts,
            &snapshot,
            usable,
            &profile,
            run.store,
            run.session_id,
            &run.run_id,
            run.token,
            run.events,
            true,
            runner,
            None,
        )
        .await
    }

    /// Base system prompt with domain instructions; profiles and skills are added per
    /// run.
    pub fn system_prompt(&self, prompts: &PromptSet) -> String {
        let env = Environment::capture(self.inner.cwd.to_string_lossy().into_owned());
        let base = prompts.system_prompt(&env);
        match &self.inner.instructions {
            Some(instructions) => format!("{base}\n\n{instructions}"),
            None => base,
        }
    }

    /// The working directory tools resolve against.
    pub fn cwd(&self) -> &Path {
        &self.inner.cwd
    }

    pub fn limits(&self) -> &Limits {
        &self.inner.limits
    }

    pub fn tools(&self) -> &Toolset {
        &self.inner.tools
    }

    pub fn model(&self) -> Arc<dyn ChatModel> {
        self.inner.model.clone()
    }

    pub fn prompts(&self) -> &PromptSource {
        &self.inner.prompts
    }

    pub fn spill(&self) -> Arc<OutputSpill> {
        self.inner.spill.clone()
    }

    pub fn profiles(&self) -> &ProfileSet {
        &self.inner.profiles
    }

    /// The profile a stored top-level session runs: the selector's choice when one is
    /// registered and returns a name, otherwise the session's own `agent`.
    pub fn select_profile(&self, session: &Session) -> String {
        self.inner
            .selector
            .as_ref()
            .and_then(|selector| selector.select(session))
            .unwrap_or_else(|| session.agent.clone())
    }

    pub fn skills_dir(&self) -> Option<&Path> {
        self.inner.skills_dir.as_deref()
    }

    pub fn instructions(&self) -> Option<&str> {
        self.inner.instructions.as_deref()
    }

    /// The memory directory, when memory is on.
    pub fn memory(&self) -> Option<&Memory> {
        self.inner.memory.as_deref()
    }

    /// Reviews a finished run and updates memory when its signals flag it;
    /// `Ok(false)` when memory is off or nothing flagged the run.
    pub async fn reflect(&self, run: RunReview<'_>, token: CancellationToken) -> Result<bool> {
        let inner = &self.inner;
        let Some(memory) = &inner.memory else {
            return Ok(false);
        };
        let signals = reflect::signals(&run);
        if signals.is_empty() {
            return Ok(false);
        }
        let prompts = inner.prompts.load()?;
        let usable = usable_tokens(inner.model.context_tokens())?;
        let index = memory.index().await;
        let system = format!(
            "{}\n\n{}",
            prompts.reflection.system,
            prompts.memory_section(&prompt_index(&index))
        );
        let mut signals: Vec<String> = signals.iter().map(|signal| format!("- {signal}")).collect();
        // A verdict that is not itself a signal is still context for the review.
        if let Some(verdict) = run.verdict
            && !matches!(verdict.outcome, Outcome::Failure | Outcome::Partial)
        {
            signals.push(format!(
                "- (context) the evaluator judged it {}: {}",
                verdict.outcome, verdict.reason
            ));
        }
        let signals = signals.join("\n");
        let transcript = reflect::transcript(
            run.messages,
            inline_budget(usable).min(reflect::REFLECTION_TRANSCRIPT_BYTES),
        );
        let prompt = prompts.reflection_prompt(&signals, &transcript);
        let tool = MemoryTool::new(prompts.memory.description.clone(), memory.clone(), false);
        let mut tools = Toolset::new();
        tools.push(Box::new(match run.session {
            Some(id) => tool.with_source(format!("reflection on session {id}")),
            None => tool.with_source("reflection"),
        }))?;
        let limits = Limits {
            max_steps: reflect::REFLECTION_STEPS,
            timeout: reflect::REFLECTION_TIMEOUT,
            ..inner.limits.clone()
        };
        memory_run(inner, prompts, system, prompt, tools, limits, token).await?;
        Ok(true)
    }

    /// Grades a finished run with the configured evaluator; `Ok(None)` when there
    /// is none.
    pub async fn evaluate(
        &self,
        run: &RunReview<'_>,
        token: &CancellationToken,
    ) -> Result<Option<Verdict>> {
        let Some(evaluator) = &self.inner.evaluator else {
            return Ok(None);
        };
        let prompts = self.inner.prompts.load()?;
        evaluator.evaluate(run, &prompts, token).await.map(Some)
    }

    /// Whether an evaluator grades finished runs.
    pub fn has_evaluator(&self) -> bool {
        self.inner.evaluator.is_some()
    }

    /// Merges, corrects and prunes memories and maintains learned skills;
    /// `Ok(false)` when memory is off. Callers decide when, e.g. with
    /// [`Memory::consolidation_due`], which an attempt defers by an hour.
    pub async fn consolidate(&self, token: CancellationToken) -> Result<bool> {
        let inner = &self.inner;
        let Some(memory) = &inner.memory else {
            return Ok(false);
        };
        // Stamped first, so a consolidation that keeps failing is not retried
        // after every run.
        memory.mark_attempt().await?;
        let prompts = inner.prompts.load()?;
        let usable = usable_tokens(inner.model.context_tokens())?;
        let render = |entries: Vec<(String, String)>, tag: &str| {
            entries
                .into_iter()
                .map(|(name, text)| format!("<{tag} name=\"{name}\">\n{}\n</{tag}>", text.trim()))
                .collect::<Vec<_>>()
                .join("\n\n")
        };
        let mut memories = render(memory.contents(MemoryKind::Memory).await, "memory");
        let mut skills = render(memory.contents(MemoryKind::Skill).await, "skill");
        if memories.len() + skills.len()
            > inline_budget(usable).min(reflect::CONSOLIDATION_INPUT_BYTES)
        {
            let listing = |entries: Vec<MemoryEntry>| {
                format!(
                    "Too large to show in full; view entries by name.\n{}",
                    index_entries(&entries)
                )
            };
            memories = listing(memory.list(MemoryKind::Memory).await);
            skills = listing(memory.list(MemoryKind::Skill).await);
        }
        let prompt = prompts.consolidation_prompt(&memories, &skills);
        let installed = match &inner.skills_dir {
            Some(dir) => load_skills(dir).await,
            None => Vec::new(),
        };
        let mut tools = Toolset::new();
        tools.push(Box::new(
            MemoryTool::new(prompts.memory.description.clone(), memory.clone(), false)
                .with_source("consolidation"),
        ))?;
        tools.push(Box::new(
            MemoryTool::learned_skills(
                prompts.consolidation.skill_tool.clone(),
                memory.clone(),
                installed.into_iter().map(|skill| skill.name).collect(),
            )
            .with_source("consolidation"),
        ))?;
        let limits = Limits {
            max_steps: reflect::CONSOLIDATION_STEPS,
            timeout: reflect::CONSOLIDATION_TIMEOUT,
            ..inner.limits.clone()
        };
        let system = prompts.consolidation.system.clone();
        memory_run(inner, prompts, system, prompt, tools, limits, token).await?;
        memory.mark_consolidated().await?;
        Ok(true)
    }

    /// Stops every stored session's background processes and closes its MCP
    /// connections; [`Runner::shutdown`](crate::Runner::shutdown) calls this.
    pub async fn close_session_scopes(&self) {
        self.inner.scopes.close_all().await;
    }

    /// Stops one session's background processes and closes its MCP connections,
    /// e.g. when the session is deleted.
    pub async fn close_session_scope(&self, session: SessionId) {
        self.inner.scopes.close(session).await;
    }
}

/// Session-run inputs beyond the shared agent configuration.
pub struct SessionRun {
    pub store: Arc<dyn SessionStore>,
    pub session_id: SessionId,
    /// The claimed run id (terminal events are appended by the runner).
    pub run_id: String,
    /// The profile name from the session row.
    pub profile: String,
    /// The runner's per-session cancellation token.
    pub token: CancellationToken,
    /// Optional lossy delta/event queue.
    pub events: Option<EventSink>,
}

/// Half the usable context in bytes, at four bytes per token: what reflection and
/// consolidation inline.
fn inline_budget(usable: usize) -> usize {
    usable / 2 * 4
}

/// Runs a memory-maintenance conversation: no persistence, events or compaction.
async fn memory_run(
    inner: &AgentInner,
    prompts: Arc<PromptSet>,
    system: String,
    prompt: String,
    tools: Toolset,
    limits: Limits,
    token: CancellationToken,
) -> Result<()> {
    let usable = usable_tokens(inner.model.context_tokens())?;
    let ctx = tool_context(inner)
        .with_spill(inner.spill.clone())
        .with_execution(
            inner.execution_mode,
            inner.allow_network,
            inner
                .policy
                .clone()
                .unwrap_or_else(|| Arc::new(crate::tool::policy::AllowTools)),
        );
    engine::run_with_token(
        RunConfig {
            model: inner.model.clone(),
            tools: Arc::new(tools),
            prompts,
            ctx,
            limits,
            messages: vec![Message::system(system), Message::user(prompt)],
            run_id: None,
            events: None,
            persist: None,
            steer: None,
            compact: None,
            max_tokens: Some(usable),
            close_scope: true,
        },
        token,
    )
    .await?;
    Ok(())
}

fn tool_context(inner: &AgentInner) -> ToolCtx {
    let mut ctx = ToolCtx::new(inner.cwd.clone());
    if let Some((default, maximum)) = inner.shell_timeouts {
        ctx.shell_timeout_ms = default;
        ctx.shell_max_timeout_ms = maximum;
    }
    ctx
}

/// Shared stored-run core. Children cannot steer; persisted events carry `run_id`.
#[allow(clippy::too_many_arguments)]
async fn run_saved(
    inner: &Arc<AgentInner>,
    prompts: &Arc<PromptSet>,
    snapshot: &Arc<RunSnapshot>,
    usable: usize,
    profile: &Profile,
    store: Arc<dyn SessionStore>,
    session_id: SessionId,
    run_id: &str,
    token: CancellationToken,
    events: Option<EventSink>,
    steer: bool,
    runner: Option<Weak<crate::runner::Runner>>,
    budget: Option<Arc<crate::engine::RunBudget>>,
) -> Result<String> {
    let launch = saved_launcher(inner, &store, prompts, snapshot, usable, session_id, runner);
    let assembled = assemble(
        inner,
        prompts,
        profile,
        snapshot,
        Some(session_id),
        Some(launch),
    )?;
    let history = store.history(session_id, prompts).await?;
    let messages = std::iter::once(Message::system(assembled.system.clone()))
        .chain(history)
        .collect();
    // A root run keeps the session's processes and MCP connections for its next
    // message; a child session never gets one, so its run owns its scope.
    let lease = steer.then(|| inner.scopes.acquire(session_id));
    // the session's servers older than this run's tools start again, before any call
    #[cfg(feature = "mcp")]
    if let (Some(lease), Some(listed)) = (&lease, &snapshot.refreshed) {
        crate::mcp::close_outdated(&lease.scope, listed).await;
    }
    let mut ctx = tool_context(inner)
        .with_spill(inner.spill.clone())
        .with_execution(
            inner.execution_mode,
            inner.allow_network,
            inner
                .policy
                .clone()
                .unwrap_or_else(|| Arc::new(crate::tool::policy::AllowTools)),
        );
    if let Some(lease) = &lease {
        ctx = ctx.with_scope(&lease.scope);
    }
    ctx.budget = budget;
    let persist: PersistFn = Arc::new({
        let store = store.clone();
        let run_id = run_id.to_owned();
        move |turn, message| {
            let store = store.clone();
            let run_id = run_id.clone();
            Box::pin(async move {
                store
                    .append_run_event(
                        session_id,
                        &run_id,
                        "message",
                        json!({ "message": message, "turn": turn }),
                    )
                    .await
                    .map(|_| ())
            })
        }
    });
    let steer_fn: Option<SteerFn> = steer.then(|| {
        Arc::new({
            let store = store.clone();
            let run_id = run_id.to_owned();
            move || {
                let store = store.clone();
                let run_id = run_id.clone();
                Box::pin(async move {
                    let inputs = store
                        .take_inputs(session_id, Delivery::Steer, 1000, Some(&run_id))
                        .await?;
                    let messages: Vec<Message> = inputs
                        .into_iter()
                        .map(|input| Message::user(input.content))
                        .collect();
                    Ok(messages)
                })
                    as Pin<Box<dyn std::future::Future<Output = Result<Vec<Message>>> + Send>>
            }
        }) as SteerFn
    });
    let compact = store_compactor(
        inner,
        &store,
        session_id,
        run_id,
        prompts,
        usable,
        assembled.system,
    );

    let outcome = engine::run_with_token(
        RunConfig {
            model: inner.model.clone(),
            tools: assembled.tools,
            prompts: prompts.clone(),
            ctx: ctx.with_rules(assembled.rules.clone()),
            limits: assembled.limits,
            messages,
            run_id: Some(run_id.to_owned()),
            events,
            persist: Some(persist),
            steer: steer_fn,
            compact: Some(compact),
            max_tokens: Some(usable),
            close_scope: lease.is_none(),
        },
        token,
    )
    .await?;
    Ok(outcome.answer)
}

/// Launches stored child sessions under the parent's token and snapshots.
fn saved_launcher(
    inner: &Arc<AgentInner>,
    store: &Arc<dyn SessionStore>,
    prompts: &Arc<PromptSet>,
    snapshot: &Arc<RunSnapshot>,
    usable: usize,
    parent_session: SessionId,
    runner: Option<Weak<crate::runner::Runner>>,
) -> LaunchSubagent {
    let inner = inner.clone();
    let store = store.clone();
    let prompts = prompts.clone();
    let snapshot = snapshot.clone();
    Arc::new(move |request: SubagentRequest| {
        let inner = inner.clone();
        let store = store.clone();
        let prompts = prompts.clone();
        let snapshot = snapshot.clone();
        let runner = runner.clone();
        Box::pin(async move {
            let token = request.token.child_token();
            let title = format!("{} (@{})", request.description, request.profile.name);
            let child = store
                .create_session(NewSession {
                    title: Some(title),
                    parent_id: Some(parent_session),
                    agent: Some(request.profile.name.clone()),
                    ..Default::default()
                })
                .await?;
            // Lets viewers of the parent find, stream and interrupt the child while it runs.
            store
                .append(
                    parent_session,
                    "subagent.started",
                    json!({
                        "sessionId": child,
                        "agent": request.profile.name,
                        "description": request.description,
                    }),
                )
                .await?;
            let owner = runner.as_ref().and_then(Weak::upgrade);
            let _guard = owner
                .as_ref()
                .map(|runner| runner.register_child(child, token.clone()))
                .transpose()?;
            let events = owner.as_ref().map(|runner| runner.delta_sink(child));
            let run_id = uuid::Uuid::new_v4().to_string();
            store
                .append_run_event(
                    child,
                    &run_id,
                    "message",
                    json!({"message": Message::user(request.prompt.clone())}),
                )
                .await?;
            store
                .append_run_event(child, &run_id, "run.started", json!({}))
                .await?;
            let result = run_saved(
                &inner,
                &prompts,
                &snapshot,
                usable,
                &request.profile,
                store.clone(),
                child,
                &run_id,
                token.clone(),
                events,
                false,
                runner,
                request.budget,
            )
            .await;
            match result {
                Ok(answer) => {
                    store
                        .append_run_event(child, &run_id, "run.finished", json!({"answer": answer}))
                        .await?;
                    Ok(SubagentResult {
                        session_id: Some(child),
                        answer,
                    })
                }
                Err(err) => {
                    let event_type = if token.is_cancelled() {
                        "run.interrupted"
                    } else {
                        "run.failed"
                    };
                    let _ = store
                        .append_run_event(
                            child,
                            &run_id,
                            event_type,
                            json!({"error": err.to_string()}),
                        )
                        .await;
                    Err(err)
                }
            }
        })
    })
}

/// Summarizes stored history, persists a compaction event, and rebuilds the model view.
fn store_compactor(
    inner: &Arc<AgentInner>,
    store: &Arc<dyn SessionStore>,
    session_id: SessionId,
    run_id: &str,
    prompts: &Arc<PromptSet>,
    usable: usize,
    system: String,
) -> crate::engine::CompactFn {
    let inner = inner.clone();
    let store = store.clone();
    let prompts = prompts.clone();
    let run_id = run_id.to_owned();
    Arc::new(move |_messages: Vec<Message>, token: CancellationToken| {
        let inner = inner.clone();
        let store = store.clone();
        let prompts = prompts.clone();
        let system = system.clone();
        let run_id = run_id.clone();
        Box::pin(async move {
            let conversation = store.conversation(session_id).await?;
            let messages: Vec<Message> = conversation
                .events
                .iter()
                .map(|event| {
                    serde_json::from_value(event.data.get("message").cloned().unwrap_or_default())
                        .map_err(|err| {
                            Error::Run(format!("event {} holds no valid message: {err}", event.id))
                        })
                })
                .collect::<Result<_>>()?;
            let start = tail_start(&messages, tail_budget(usable));
            let last_user = messages
                .iter()
                .rposition(|message| message.role == Role::User);
            let head: Vec<Message> = messages[..start]
                .iter()
                .enumerate()
                .filter(|(index, _)| Some(*index) != last_user)
                .map(|(_, message)| message.clone())
                .collect();
            if head.is_empty() {
                return Ok(None);
            }
            let summary = summarize(
                &*inner.model,
                &prompts,
                &head,
                conversation.summary.as_deref(),
                usable,
                &token,
            )
            .await?;
            let Some(tail_event) = conversation.events.get(start) else {
                return Ok(None);
            };
            // A user message kept verbatim ahead of the tail is summarised by the
            // next compaction; `conversation()` keeps it until then.
            let kept_user = last_user
                .filter(|index| *index < start)
                .and_then(|index| conversation.events.get(index))
                .map(|event| event.id);
            store
                .append_run_event(
                    session_id,
                    &run_id,
                    "compaction",
                    json!({"summary": summary, "tailFrom": tail_event.id, "keptUser": kept_user}),
                )
                .await?;
            let history = store.history(session_id, &prompts).await?;
            Ok(Some(
                std::iter::once(Message::system(system))
                    .chain(history)
                    .collect(),
            ))
        })
    })
}

/// Assembles profile-specific tools, prompts, and limits. Child runs never receive
/// `task`.
fn assemble(
    inner: &AgentInner,
    prompts: &PromptSet,
    profile: &Profile,
    snapshot: &RunSnapshot,
    session: Option<SessionId>,
    launch: Option<LaunchSubagent>,
) -> Result<Assembled> {
    let mut tools = Toolset::new();
    // build-time tools, minus the refreshed source's, which this run lists afresh
    let replaced: &[String] = match (&inner.refresh, &snapshot.refreshed) {
        (Some((_, names)), Some(_)) => names,
        _ => &[],
    };
    for name in inner.tools.names() {
        if !replaced.iter().any(|replaced| replaced == name)
            && let Some(tool) = inner.tools.get(name)
            && profile.admits(name, tool.read_only())
        {
            tools.push_shared(tool)?;
        }
    }
    for tool in snapshot.refreshed.iter().flatten() {
        let name = tool.spec().function.name;
        if profile.admits(&name, tool.read_only()) && tools.get(&name).is_none() {
            tools.push_shared(tool.clone())?;
        }
    }
    if !snapshot.skills.is_empty() && profile.allows("skill") {
        tools.push(Box::new(SkillTool::new(
            prompts.skill.description.clone(),
            Arc::new(snapshot.skills.clone()),
        )))?;
    }
    if let Some(memory) = &inner.memory
        && profile.allows("memory")
    {
        let tool = MemoryTool::new(
            prompts.memory.description.clone(),
            memory.clone(),
            profile.subagent,
        );
        tools.push(Box::new(match session {
            Some(id) => tool.with_source(format!("session {id}")),
            None => tool,
        }))?;
    }
    if let Some(launch) = launch
        && task_available(profile)
    {
        let children = resolve_children(prompts, &inner.profiles, &profile.allowed_subagents);
        tools.push(Box::new(TaskTool::new(
            prompts,
            inner.limits.max_task_calls_per_step,
            children,
            launch,
        )))?;
    }

    let mut sections = vec![{
        let env = Environment::capture(inner.cwd.to_string_lossy().into_owned());
        let base = prompts.system_prompt(&env);
        match &inner.instructions {
            Some(instructions) => format!("{base}\n\n{instructions}"),
            None => base,
        }
    }];
    let (_, profile_prompt) = resolve_profile_text(prompts, profile);
    sections.extend(profile_prompt);
    if !snapshot.skills.is_empty() && profile.allows("skill") {
        sections.push(prompts.skill_section(&skills_entries(&snapshot.skills)));
    }
    if inner.memory.is_some() && profile.allows("memory") {
        sections.push(prompts.memory_section(&prompt_index(&snapshot.memory)));
    }

    let limits = Limits {
        max_steps: if inner.cap_profile_steps {
            profile.max_steps.min(inner.limits.max_steps)
        } else {
            profile.max_steps
        },
        ..inner.limits.clone()
    };
    Ok(Assembled {
        tools: Arc::new(tools),
        system: sections.join("\n\n"),
        limits,
        rules: Arc::new(profile.rules.clone()),
    })
}

/// Runs an in-memory child under the parent's token and prompt/skill snapshots.
async fn run_child(
    inner: Arc<AgentInner>,
    prompts: Arc<PromptSet>,
    snapshot: Arc<RunSnapshot>,
    usable: usize,
    request: SubagentRequest,
) -> Result<SubagentResult> {
    let assembled = assemble(&inner, &prompts, &request.profile, &snapshot, None, None)?;
    let messages = vec![
        Message::system(assembled.system),
        Message::user(request.prompt),
    ];
    let mut ctx = tool_context(&inner)
        .with_spill(inner.spill.clone())
        .with_execution(
            inner.execution_mode,
            inner.allow_network,
            inner
                .policy
                .clone()
                .unwrap_or_else(|| Arc::new(crate::tool::policy::AllowTools)),
        );
    ctx.budget = request.budget;
    let compact = crate::context::in_memory_compactor(inner.model.clone(), prompts.clone(), usable);
    let outcome = engine::run_with_token(
        RunConfig {
            model: inner.model.clone(),
            tools: assembled.tools,
            prompts,
            ctx: ctx.with_rules(assembled.rules.clone()),
            limits: assembled.limits,
            messages,
            run_id: None,
            events: None,
            persist: None,
            steer: None,
            compact: Some(compact),
            max_tokens: Some(usable),
            close_scope: true,
        },
        request.token,
    )
    .await?;
    Ok(SubagentResult {
        session_id: None,
        answer: outcome.answer,
    })
}

/// Builds an [`Agent`]. Supplying both `vendors` and `model` is an error;
/// so is supplying neither.
#[derive(Default)]
pub struct AgentBuilder {
    vendors: Option<Vendors>,
    model: Option<Arc<dyn ChatModel>>,
    tools: Vec<Box<dyn Tool>>,
    shared_tools: Vec<Arc<dyn Tool>>,
    profiles: Vec<Profile>,
    replaced_profiles: Vec<Profile>,
    profile_patches: Vec<(String, ProfilePatch)>,
    selector_conflict: bool,
    selector: Option<ProfileSelector>,
    instructions: Option<String>,
    prompts: Option<PromptSource>,
    cwd: Option<PathBuf>,
    limits: Option<Limits>,
    spill: Option<Arc<OutputSpill>>,
    skills_dir: Option<PathBuf>,
    memory_dir: Option<PathBuf>,
    evaluator: Option<Arc<dyn Evaluator>>,
    model_timeouts: Option<crate::llm::ModelTimeouts>,
    shell_timeouts: Option<(u64, u64)>,
    execution_mode: crate::tool::policy::ExecutionMode,
    allow_network: bool,
    policy: Option<Arc<dyn crate::tool::policy::ToolPolicy>>,
    refresh: Option<(ToolRefresh, Vec<String>)>,
}

impl AgentBuilder {
    /// Default model backend with OpenAI-compatible vendor failover.
    pub fn vendors(mut self, vendors: Vendors) -> Self {
        self.vendors = Some(vendors);
        self
    }

    pub fn model_timeouts(mut self, timeouts: crate::llm::ModelTimeouts) -> Self {
        self.model_timeouts = Some(timeouts);
        self
    }

    /// A custom `ChatModel` instead of the vendor default.
    pub fn model(mut self, model: Arc<dyn ChatModel>) -> Self {
        self.model = Some(model);
        self
    }

    /// The working directory (default: the process's).
    pub fn cwd(mut self, cwd: impl Into<PathBuf>) -> Self {
        self.cwd = Some(cwd.into());
        self
    }

    /// Domain system-prompt prose appended to the generic base.
    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    /// Registers tools; duplicates are rejected at build time.
    pub fn tools(mut self, tools: impl IntoIterator<Item = Box<dyn Tool>>) -> Self {
        self.tools.extend(tools);
        self
    }

    pub fn tool(mut self, tool: Box<dyn Tool>) -> Self {
        self.tools.push(tool);
        self
    }

    /// Registers shared tools (e.g. the MCP connections' adapters).
    pub fn tools_shared(mut self, tools: impl IntoIterator<Item = Arc<dyn Tool>>) -> Self {
        self.shared_tools.extend(tools);
        self
    }

    /// Registers a tool source whose tools can change while the agent runs (MCP
    /// servers): `tools` as listed now, which profiles are validated against,
    /// and `refresh`, which every top-level run calls for the current list.
    pub fn refreshing_tools(mut self, tools: Vec<Arc<dyn Tool>>, refresh: ToolRefresh) -> Self {
        let names = tools.iter().map(|tool| tool.spec().function.name).collect();
        self.shared_tools.extend(tools);
        self.refresh = Some((refresh, names));
        self
    }

    /// Adds domain profiles alongside `build`, `general`, and tool-dependent `explore`.
    pub fn profiles(mut self, profiles: impl IntoIterator<Item = Profile>) -> Self {
        self.profiles.extend(profiles);
        self
    }

    /// Replaces a registered or builtin profile by name, keeping the child allowlists
    /// that already name it. Use it to widen `explore` with domain tools.
    pub fn replace_profile(mut self, profile: Profile) -> Self {
        self.replaced_profiles.push(profile);
        self
    }

    /// Patches a registered or builtin profile by name, after builtins are resolved
    /// from the prompt snapshot and the registered tools. Omitted patch fields keep
    /// the profile's resolved value; patching an absent name is a build error.
    pub fn patch_profile(mut self, name: impl Into<String>, patch: ProfilePatch) -> Self {
        self.profile_patches.push((name.into(), patch));
        self
    }

    /// Chooses the profile of a stored top-level session from its opaque metadata,
    /// replacing the session's `agent` when it returns a name. Installing a second
    /// selector — a manifest's `select_profile` rules plus this — is a build error.
    pub fn profile_selector(
        mut self,
        select: impl Fn(&Session) -> Option<String> + Send + Sync + 'static,
    ) -> Self {
        self.selector_conflict = self.selector.is_some();
        self.selector = Some(ProfileSelector::new(select));
        self
    }

    pub fn shell_timeouts(mut self, default_ms: u64, maximum_ms: u64) -> Self {
        self.shell_timeouts = Some((default_ms, maximum_ms));
        self
    }

    /// Policy callbacks are inherited by subagents. The CLI uses Sandbox mode;
    /// embedders explicitly own isolation when using the default Trusted mode.
    pub fn execution_policy(
        mut self,
        mode: crate::tool::policy::ExecutionMode,
        network: bool,
        policy: Arc<dyn crate::tool::policy::ToolPolicy>,
    ) -> Self {
        self.execution_mode = mode;
        self.allow_network = network;
        self.policy = Some(policy);
        self
    }

    /// Where prompts come from (default: [`PromptSource::Embedded`]).
    pub fn prompts(mut self, prompts: PromptSource) -> Self {
        self.prompts = Some(prompts);
        self
    }

    /// Run limits (default: [`Limits::default`]).
    pub fn limits(mut self, limits: Limits) -> Self {
        self.limits = Some(limits);
        self
    }

    /// Custom spill storage instead of the default temporary directory.
    pub fn spill(mut self, spill: Arc<OutputSpill>) -> Self {
        self.spill = Some(spill);
        self
    }

    /// A directory of skills (`<name>/SKILL.md`), re-read per run.
    pub fn skills_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.skills_dir = Some(dir.into());
        self
    }

    /// A directory of persistent memories shared by every session, indexed per run
    /// and maintained through the `memory` tool; created when missing.
    pub fn memory_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.memory_dir = Some(dir.into());
        self
    }

    /// Grades every finished run, e.g. an [`LlmEvaluator`](crate::evaluate::LlmEvaluator)
    /// on a cheaper model; verdicts are recorded and steer reflection.
    pub fn evaluator(mut self, evaluator: Arc<dyn Evaluator>) -> Self {
        self.evaluator = Some(evaluator);
        self
    }

    pub fn build(self) -> Result<Agent> {
        if self.selector_conflict {
            return Err(Error::Config(
                "two profile selectors were supplied; a manifest's `select_profile` rules and \
                 `profile_selector` are alternatives, not layers"
                    .to_owned(),
            ));
        }
        if let Some((default, maximum)) = self.shell_timeouts
            && (default == 0 || default > maximum || maximum > 86_400_000)
        {
            return Err(Error::Config(
                "shell timeouts require 0 < default <= maximum <= 24 hours".into(),
            ));
        }
        let model: Arc<dyn ChatModel> = match (self.vendors, self.model) {
            (Some(_), Some(_)) => {
                return Err(Error::Config(
                    "supply either vendors or a custom model, not both".to_owned(),
                ));
            }
            (Some(vendors), None) => Arc::new(
                VendorModel::new(vendors).with_timeouts(self.model_timeouts.unwrap_or_default())?,
            ),
            (None, Some(model)) => model,
            (None, None) => {
                return Err(Error::Config("a model or vendors are required".to_owned()));
            }
        };
        crate::context::usable_tokens(model.context_tokens())?;

        let prompts = self.prompts.unwrap_or(PromptSource::Embedded);
        // Build-time validation of the prompt file; runs re-read it.
        let snapshot = prompts.load()?;

        let mut tools = Toolset::new();
        tools.extend(self.tools)?;
        for tool in self.shared_tools {
            tools.push_shared(tool)?;
        }
        // The runtime adds these per run; a base tool with the same name would fail
        // every run's assembly instead of the build.
        let memory_tool = self.memory_dir.is_some().then_some("memory");
        for reserved in ["skill", "task"].into_iter().chain(memory_tool) {
            if tools.get(reserved).is_some() {
                return Err(Error::Config(format!(
                    "the tool name `{reserved}` is reserved for the runtime"
                )));
            }
        }

        let mut profiles = ProfileSet::builtin(&snapshot, &tools);
        let mut registered: Vec<String> = Vec::new();
        for profile in self.profiles {
            registered.push(profile.name.clone());
            profiles.register(profile)?;
        }
        for profile in self.replaced_profiles {
            profiles.replace(profile)?;
        }
        for (name, patch) in self.profile_patches {
            if registered.contains(&name) {
                return Err(Error::Config(format!(
                    "profile `{name}` is both registered and patched; supply one complete profile"
                )));
            }
            profiles.patch(&name, &patch)?;
        }
        profiles.validate(&tools)?;

        let cwd = match self.cwd {
            Some(cwd) => cwd,
            None => std::env::current_dir().map_err(|err| {
                Error::Config(format!("cannot determine the working directory: {err}"))
            })?,
        };
        // Skill paths are shown to the model, whose tools resolve against `cwd`,
        // so a relative skills directory is pinned to an absolute one here.
        let skills_dir = self
            .skills_dir
            .map(|dir| {
                if !dir.is_dir() {
                    return Err(Error::Config(format!(
                        "the skills directory {} does not exist",
                        dir.display()
                    )));
                }
                std::path::absolute(&dir).map_err(|err| {
                    Error::Config(format!(
                        "cannot resolve the skills directory {}: {err}",
                        dir.display()
                    ))
                })
            })
            .transpose()?;
        Ok(Agent {
            inner: Arc::new(AgentInner {
                model,
                tools: Arc::new(tools),
                instructions: self.instructions,
                prompts,
                cwd,
                shell_timeouts: self.shell_timeouts,
                cap_profile_steps: self.limits.is_some(),
                limits: self.limits.unwrap_or_default(),
                spill: self
                    .spill
                    .unwrap_or_else(|| Arc::new(OutputSpill::new(OutputSpill::default_dir()))),
                profiles: Arc::new(profiles),
                selector: self.selector,
                skills_dir,
                memory: self
                    .memory_dir
                    .map(|dir| Memory::open(dir).map(Arc::new))
                    .transpose()?,
                evaluator: self.evaluator,
                execution_mode: self.execution_mode,
                allow_network: self.allow_network,
                policy: self.policy,
                scopes: Arc::default(),
                refresh: self.refresh,
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use futures_util::{StreamExt, stream};

    use super::*;
    use crate::llm::{ChatDelta, ChatRequest, ChatStream};

    #[tokio::test(start_paused = true)]
    async fn session_scopes_close_after_idle_unless_a_run_reuses_them() {
        let scopes = Arc::new(SessionScopes::default());
        let first = scopes.acquire(1);
        let shared = first.scope.processes.clone();
        drop(first);
        tokio::time::sleep(SESSION_SCOPE_IDLE / 2).await;
        // A new run before the timer fires keeps the same scope alive.
        let second = scopes.acquire(1);
        assert!(Arc::ptr_eq(&second.scope.processes, &shared));
        tokio::time::sleep(SESSION_SCOPE_IDLE).await;
        assert!(scopes.0.lock().unwrap().contains_key(&1), "in use");
        drop(second);
        tokio::time::sleep(SESSION_SCOPE_IDLE + Duration::from_secs(1)).await;
        assert!(scopes.0.lock().unwrap().is_empty(), "closed after idling");
    }

    /// A model that always answers the same text.
    struct AnswerModel(String);

    #[async_trait::async_trait]
    impl ChatModel for AnswerModel {
        fn context_tokens(&self) -> usize {
            128_000
        }

        async fn stream(
            &self,
            _request: ChatRequest,
            _token: &CancellationToken,
        ) -> Result<ChatStream> {
            let text = self.0.clone();
            Ok(stream::iter([
                Ok(ChatDelta::Text(text)),
                Ok(ChatDelta::Done {
                    finish_reason: Some("stop".into()),
                }),
            ])
            .boxed())
        }
    }

    #[tokio::test]
    async fn run_once_seeds_the_system_prompt_with_instructions() {
        let dir = tempfile::tempdir().unwrap();
        let agent = Agent::builder()
            .model(Arc::new(AnswerModel("the answer".into())))
            .cwd(dir.path())
            .instructions("You are an agent working on the Foo service.")
            .spill(Arc::new(OutputSpill::new(dir.path().join("spill"))))
            .build()
            .unwrap();
        assert_eq!(
            agent.run_once("explain this repo").await.unwrap(),
            "the answer"
        );

        let prompts = agent.prompts().load().unwrap();
        let system = agent.system_prompt(&prompts);
        assert!(
            system.contains(&format!("Working directory: {}", dir.path().display())),
            "{system}"
        );
        assert!(
            system.contains("You are an agent working on the Foo service."),
            "{system}"
        );
        assert!(system.contains("untrusted evidence"), "{system}");
    }

    #[tokio::test]
    async fn unknown_and_subagent_profiles_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let agent = Agent::builder()
            .model(Arc::new(AnswerModel("x".into())))
            .cwd(dir.path())
            .spill(Arc::new(OutputSpill::new(dir.path().join("spill"))))
            .build()
            .unwrap();
        let err = agent.run_prompt_as("triage", "hi", None).await.unwrap_err();
        assert!(
            err.to_string().contains("Unknown agent \"triage\""),
            "{err}"
        );
        assert!(
            err.to_string().contains("build"),
            "known profiles are listed: {err}"
        );

        let err = agent
            .run_prompt_as("general", "hi", None)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("subagent profile"), "{err}");
    }

    #[test]
    fn a_relative_skills_dir_is_made_absolute() {
        let agent = Agent::builder()
            .model(Arc::new(AnswerModel("x".into())))
            .cwd("/some/other/cwd")
            .skills_dir("src/tool")
            .build()
            .unwrap();
        let dir = agent.skills_dir().unwrap();
        assert!(dir.is_absolute(), "{}", dir.display());
        assert!(dir.ends_with("src/tool"), "{}", dir.display());
        assert!(!dir.starts_with("/some/other/cwd"), "{}", dir.display());
        let err = Agent::builder()
            .model(Arc::new(AnswerModel("x".into())))
            .skills_dir("skils")
            .build()
            .unwrap_err();
        assert!(err.to_string().contains("skils does not exist"), "{err}");
    }

    #[test]
    fn the_builder_rejects_model_configuration_mistakes() {
        let vendors = Vendors::from_lookup(|key| match key {
            "AI_VENDORS" => Some("x".into()),
            "AI_X_URL" => Some("https://x.test".into()),
            "AI_X_MODEL" => Some("m".into()),
            "AI_X_API_KEY" => Some("k".into()),
            _ => None,
        })
        .unwrap();
        let err = Agent::builder()
            .vendors(vendors.clone())
            .model(Arc::new(AnswerModel("x".into())))
            .build()
            .unwrap_err();
        assert!(err.to_string().contains("not both"), "{err}");

        let err = Agent::builder().build().unwrap_err();
        assert!(err.to_string().contains("required"), "{err}");
    }

    #[test]
    fn the_builder_rejects_context_capacities_below_the_minimum() {
        struct TinyModel;
        #[async_trait::async_trait]
        impl ChatModel for TinyModel {
            fn context_tokens(&self) -> usize {
                512
            }
            async fn stream(
                &self,
                _request: ChatRequest,
                _token: &CancellationToken,
            ) -> Result<ChatStream> {
                Err(Error::Run("unreachable".into()))
            }
        }
        let err = Agent::builder()
            .model(Arc::new(TinyModel))
            .build()
            .unwrap_err();
        assert!(err.to_string().contains("context capacity"), "{err}");
    }

    #[test]
    fn the_builder_validates_profiles_and_tools() {
        use crate::tool::{Typed, builtin_tools};

        struct Named(&'static str);
        #[async_trait::async_trait]
        impl Tool for Named {
            fn spec(&self) -> crate::llm::ToolSpec {
                crate::llm::ToolSpec::function(self.0, "x", serde_json::json!({"type": "object"}))
            }
            async fn call(
                &self,
                _args: serde_json::Value,
                _ctx: &ToolCtx,
            ) -> std::result::Result<String, crate::tool::ToolError> {
                Ok(String::new())
            }
        }
        let dir = tempfile::tempdir().unwrap();
        for reserved in ["skill", "task", "memory"] {
            let err = Agent::builder()
                .model(Arc::new(AnswerModel("x".into())))
                .memory_dir(dir.path())
                .tool(Box::new(Named(reserved)))
                .build()
                .unwrap_err();
            assert!(err.to_string().contains("reserved"), "{err}");
        }
        // Without memory, an application may bring its own `memory` tool.
        Agent::builder()
            .model(Arc::new(AnswerModel("x".into())))
            .tool(Box::new(Named("memory")))
            .build()
            .unwrap();

        let mut tools = builtin_tools();
        tools.push(Box::new(Typed(crate::tool::builtin::ReadTool)));
        let err = Agent::builder()
            .model(Arc::new(AnswerModel("x".into())))
            .tools(tools)
            .build()
            .unwrap_err();
        assert!(
            err.to_string().contains("duplicate tool name `read`"),
            "{err}"
        );

        let err = Agent::builder()
            .model(Arc::new(AnswerModel("x".into())))
            .profiles([Profile {
                tools: Some(vec!["sql".into()]),
                ..Profile::new("triage-like", "references an unregistered tool")
            }])
            .build()
            .unwrap_err();
        assert!(err.to_string().contains("unregistered tool `sql`"), "{err}");

        let err = Agent::builder()
            .model(Arc::new(AnswerModel("x".into())))
            .prompts(PromptSource::File("no/such/prompts.yaml".into()))
            .build()
            .unwrap_err();
        assert!(err.to_string().contains("no/such/prompts.yaml"), "{err}");
    }

    /// A tool named and described as given.
    struct Listed(&'static str, &'static str);

    #[async_trait::async_trait]
    impl Tool for Listed {
        fn spec(&self) -> crate::llm::ToolSpec {
            crate::llm::ToolSpec::function(self.0, self.1, serde_json::json!({"type": "object"}))
        }
        async fn call(
            &self,
            _args: serde_json::Value,
            _ctx: &ToolCtx,
        ) -> std::result::Result<String, crate::tool::ToolError> {
            Ok(String::new())
        }
    }

    /// A model that records the tools of each request it is sent.
    struct ToolsSeen(StdMutex<Vec<Vec<String>>>);

    #[async_trait::async_trait]
    impl ChatModel for ToolsSeen {
        fn context_tokens(&self) -> usize {
            128_000
        }

        async fn stream(
            &self,
            request: ChatRequest,
            _token: &CancellationToken,
        ) -> Result<ChatStream> {
            self.0.lock().unwrap().push(
                request
                    .tools
                    .iter()
                    .map(|tool| format!("{}: {}", tool.function.name, tool.function.description))
                    .collect(),
            );
            Ok(stream::iter([
                Ok(ChatDelta::Text("done".into())),
                Ok(ChatDelta::Done {
                    finish_reason: Some("stop".into()),
                }),
            ])
            .boxed())
        }
    }

    #[tokio::test]
    async fn each_run_lists_a_refreshing_source_s_tools_again() {
        let dir = tempfile::tempdir().unwrap();
        let model = Arc::new(ToolsSeen(StdMutex::new(Vec::new())));
        let listings = Arc::new(StdMutex::new(vec![
            // a failed listing keeps the tools from build
            Err("server down".to_owned()),
            Ok(vec!["srv_query: v2", "srv_scrape: new"]),
        ]));
        let refresh: ToolRefresh = Arc::new(move || {
            let next = listings.lock().unwrap().remove(0);
            Box::pin(async move {
                next.map(|names| {
                    names
                        .into_iter()
                        .map(|entry| {
                            let (name, description) = entry.split_once(": ").unwrap();
                            Arc::new(Listed(name, description)) as Arc<dyn Tool>
                        })
                        .collect()
                })
            })
        });
        let agent = Agent::builder()
            .model(model.clone())
            .cwd(dir.path())
            .spill(Arc::new(OutputSpill::new(dir.path().join("spill"))))
            .refreshing_tools(vec![Arc::new(Listed("srv_query", "v1"))], refresh)
            .build()
            .unwrap();
        agent.run_once("first").await.unwrap();
        agent.run_once("second").await.unwrap();

        let seen = model.0.lock().unwrap();
        let srv = |tools: &Vec<String>| {
            tools
                .iter()
                .filter(|t| t.starts_with("srv_"))
                .cloned()
                .collect::<Vec<_>>()
        };
        assert_eq!(srv(&seen[0]), ["srv_query: v1"]);
        assert_eq!(srv(&seen[1]), ["srv_query: v2", "srv_scrape: new"]);
    }

    #[tokio::test]
    async fn an_interrupt_does_not_wait_for_a_hung_tool_listing() {
        let dir = tempfile::tempdir().unwrap();
        let refresh: ToolRefresh = Arc::new(|| Box::pin(std::future::pending()));
        let agent = Agent::builder()
            .model(Arc::new(AnswerModel("x".into())))
            .cwd(dir.path())
            .spill(Arc::new(OutputSpill::new(dir.path().join("spill"))))
            .refreshing_tools(Vec::new(), refresh)
            .build()
            .unwrap();
        let token = CancellationToken::new();
        token.cancel();
        let run = agent.run_prompt_as_with_token("build", "go", None, token);
        let result = tokio::time::timeout(Duration::from_secs(5), run).await;
        assert!(result.expect("the interrupt ends the listing").is_err());
    }
}
