//! CLI commands for one-shot runs and the session HTTP server.

use std::io::Write;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use crate::tool::policy::ExecutionMode;
use clap::{Args, Parser, Subcommand};
use tokio::sync::mpsc;

use crate::Error;
use crate::agent::Agent;
use crate::config::{AgentSpec, Manifest};
use crate::engine::RunEvent;
use crate::error::Result;
use crate::llm::ChatDelta;
use crate::runner::Runner;
use crate::store::SqliteStore;
use crate::tool::builtin_tools;

/// How long the HTTP server may drain connections after the shutdown token
/// fires; a viewer with a stuck TCP window cannot hold the exit past this.
const DRAIN_GRACE: Duration = Duration::from_secs(5);

/// Mason — a domain-independent agent core.
#[derive(Parser)]
#[command(name = "mason", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// Options shared by every command, each with an environment fallback.
///
/// Every setting a `--config` manifest can also supply is optional here: only a
/// value given explicitly (flag or environment) overrides the manifest, and
/// [`AgentSpec::default`] supplies what neither does. That keeps clap's defaults
/// from outranking a manifest.
#[derive(Args)]
struct CommonArgs {
    /// Declarative app config: instructions, profiles, MCP servers, budgets, grants.
    #[arg(long, env = "MASON_CONFIG")]
    config: Option<PathBuf>,
    /// Sandbox confines subprocesses and file tools; trusted grants full host
    /// access (default: trusted).
    #[arg(long, env = "MASON_EXECUTION_MODE", value_enum)]
    execution_mode: Option<ExecutionMode>,
    /// Explicitly grant network access to sandboxed subprocesses (default: false).
    #[arg(
        long,
        env = "MASON_ALLOW_NETWORK",
        num_args = 0..=1,
        default_missing_value = "true",
        require_equals = true
    )]
    allow_network: Option<bool>,
    /// Explicitly grant configured MCP servers access to agent tool calls
    /// (default: false).
    #[arg(
        long,
        env = "MASON_ALLOW_MCP",
        num_args = 0..=1,
        default_missing_value = "true",
        require_equals = true
    )]
    allow_mcp: Option<bool>,
    /// Run budget, as a duration like `90s`, `15m` or `1h30m` (default: 15m).
    #[arg(long, env = "MASON_RUN_TIMEOUT", value_parser = duration_arg)]
    run_timeout: Option<Duration>,
    /// Model request, turn and stream-idle timeouts (defaults: 2m, 3m, 1m).
    #[arg(long, env = "MASON_MODEL_REQUEST_TIMEOUT", value_parser = duration_arg)]
    model_request_timeout: Option<Duration>,
    #[arg(long, env = "MASON_MODEL_TURN_TIMEOUT", value_parser = duration_arg)]
    model_turn_timeout: Option<Duration>,
    #[arg(long, env = "MASON_MODEL_IDLE_TIMEOUT", value_parser = duration_arg)]
    model_idle_timeout: Option<Duration>,
    /// Default and maximum shell timeouts (defaults: 2m, 10m).
    #[arg(long, env = "MASON_SHELL_TIMEOUT", value_parser = duration_arg)]
    shell_timeout: Option<Duration>,
    #[arg(long, env = "MASON_SHELL_MAX_TIMEOUT", value_parser = duration_arg)]
    shell_max_timeout: Option<Duration>,
    /// Maximum tool-using steps per run; also caps delegated profiles
    /// (default: 150).
    #[arg(long, env = "MASON_MAX_STEPS")]
    max_steps: Option<usize>,
    /// Tool calls running at once, shared with subagents (default: 4).
    #[arg(long, env = "MASON_MAX_CONCURRENT_TOOLS")]
    max_concurrent_tools: Option<usize>,
    /// Tool calls and `task` calls per model reply (defaults: 16, 4).
    #[arg(long, env = "MASON_MAX_TOOL_CALLS_PER_STEP")]
    max_tool_calls_per_step: Option<usize>,
    #[arg(long, env = "MASON_MAX_TASK_CALLS_PER_STEP")]
    max_task_calls_per_step: Option<usize>,
    #[arg(long, env = "MASON_MAX_TOTAL_TOOL_CALLS")]
    max_total_tool_calls: Option<usize>,
    #[arg(long, env = "MASON_MAX_TOTAL_TOKENS")]
    max_total_tokens: Option<usize>,
    #[arg(long, env = "MASON_MAX_RESPONSE_BYTES")]
    max_response_bytes: Option<usize>,
    /// Total spill storage in MiB; includes in-progress captures (default: 256).
    #[arg(long, env = "MASON_SPILL_QUOTA_MIB")]
    spill_quota_mib: Option<u64>,

    /// Working directory (default: the current directory).
    #[arg(long, env = "MASON_CWD")]
    cwd: Option<PathBuf>,
    /// Prompt file to load instead of the embedded defaults.
    #[arg(long, env = "MASON_PROMPTS")]
    prompts: Option<PathBuf>,
    /// Directory of skills (`<name>/SKILL.md`), re-read per run.
    #[arg(long, env = "MASON_SKILLS_DIR")]
    skills_dir: Option<PathBuf>,
    /// Directory of persistent agent memories, shared by every session.
    #[arg(long, env = "MASON_MEMORY_DIR")]
    memory_dir: Option<PathBuf>,
    /// JSON file mapping server names to MCP server configs (opencode format);
    /// replaces a manifest's MCP servers wholesale.
    #[arg(long, env = "MASON_MCP")]
    mcp: Option<PathBuf>,
}

#[derive(Subcommand)]
enum Command {
    /// Inspect configured MCP connections and discovered tools without an LLM call.
    Mcp {
        #[command(flatten)]
        common: CommonArgs,
    },
    #[command(hide = true)]
    ToolWorker { name: String },
    /// Run one prompt on an in-memory conversation; the answer streams to
    /// stdout and tool activity to stderr.
    Run {
        /// The prompt; all positional words are joined.
        #[arg(required = true)]
        prompt: Vec<String>,
        /// The profile to run (default: build).
        #[arg(long, env = "MASON_PROFILE", default_value = "build")]
        profile: String,
        #[command(flatten)]
        common: CommonArgs,
    },
    /// Serve the HTTP API and run session inboxes. Every request needs the
    /// bearer token; sessions can run shell commands (README security notes).
    Serve {
        /// Address to listen on.
        #[arg(long, env = "MASON_LISTEN", default_value = "127.0.0.1:3120")]
        listen: SocketAddr,
        /// Bearer token required by all API clients (at least 16 characters).
        #[arg(long, env = "MASON_API_TOKEN", hide_env_values = true)]
        api_token: Option<String>,
        /// A url each finished, failed or interrupted run is POSTed to.
        #[arg(long, env = "MASON_RUN_WEBHOOK")]
        run_webhook: Option<String>,
        /// Bearer token sent with each run webhook POST.
        #[arg(long, env = "MASON_RUN_WEBHOOK_SECRET", hide_env_values = true)]
        run_webhook_secret: Option<String>,
        /// Maximum concurrently active root sessions.
        #[arg(long, env = "MASON_MAX_ACTIVE_RUNS", default_value_t = 4)]
        max_active_runs: usize,
        /// Maximum admitted running/queued root sessions.
        #[arg(long, env = "MASON_MAX_PENDING_SESSIONS", default_value_t = 128)]
        max_pending_sessions: usize,
        /// SQLite session database (default: <cwd>/.mason/mason.db).
        #[arg(long, env = "MASON_DB")]
        db: Option<PathBuf>,
        #[command(flatten)]
        common: CommonArgs,
    },
}

/// Parses argv and runs the selected command.
pub async fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Mcp { common } => inspect_mcp(&agent_spec(&common)?).await,
        Command::ToolWorker { name } => {
            use tokio::io::AsyncReadExt;
            let mut input = Vec::new();
            tokio::io::stdin()
                .take(16 * 1024 * 1024 + 1)
                .read_to_end(&mut input)
                .await?;
            if input.len() > 16 * 1024 * 1024 {
                return Err(Error::Config("tool arguments exceed worker limit".into()));
            }
            let args = serde_json::from_slice(&input)
                .map_err(|_| Error::Config("invalid tool worker input".into()))?;
            let tool = builtin_tools()
                .into_iter()
                .find(|tool| {
                    tool.spec().function.name == name
                        && matches!(name.as_str(), "read" | "write" | "edit" | "grep" | "glob")
                })
                .ok_or_else(|| Error::Config("unknown file tool".into()))?;
            let ctx = crate::tool::ToolCtx::new(std::env::current_dir()?);
            let result = tool.call(args, &ctx).await.map_err(|err| err.to_string());
            emit(&serde_json::to_string(&result).expect("tool result serialization"));
            Ok(())
        }
        Command::Run {
            prompt,
            profile,
            common,
        } => run(prompt, profile, common).await,
        Command::Serve {
            listen,
            db,
            common,
            api_token,
            run_webhook,
            run_webhook_secret,
            max_active_runs,
            max_pending_sessions,
        } => {
            let token = api_token.ok_or_else(|| {
                Error::Config("MASON_API_TOKEN or --api-token is required for serve".into())
            })?;
            let limits = crate::runner::RunnerLimits {
                max_active_runs,
                max_pending_sessions,
            };
            limits.validate()?;
            // Fail before opening or recovering the database if the address is occupied.
            let listener = tokio::net::TcpListener::bind(listen)
                .await
                .map_err(|err| Error::Config(format!("cannot listen on {listen}: {err}")))?;
            let webhook = run_webhook.map(|url| crate::runner::RunWebhook {
                url,
                secret: run_webhook_secret,
            });
            let security = crate::api::ApiSecurity::new(token)?;
            serve(listener, security, db, common, limits, webhook).await
        }
    }
}

/// The manifest's duration syntax (`90s`, `15m`, `1h30m`) for flags and env.
fn duration_arg(raw: &str) -> std::result::Result<Duration, String> {
    crate::config::parse_duration(raw).map_err(|err| err.to_string())
}

/// Resolves the effective settings: explicit flags and environment over the
/// manifest over mason's defaults.
fn agent_spec(common: &CommonArgs) -> Result<AgentSpec> {
    let mut spec = match &common.config {
        Some(path) => Manifest::load(path)?.into_spec()?,
        None => AgentSpec::default(),
    };
    if let Some(mode) = common.execution_mode {
        spec.execution_mode = mode;
    }
    if let Some(allow) = common.allow_network {
        spec.allow_network = allow;
    }
    if let Some(allow) = common.allow_mcp {
        spec.allow_mcp = allow;
    }
    if let Some(timeout) = common.run_timeout {
        spec.limits.timeout = timeout;
    }
    if let Some(steps) = common.max_steps {
        spec.limits.max_steps = steps;
    }
    if let Some(concurrent) = common.max_concurrent_tools {
        spec.limits.max_concurrent_tools = concurrent;
    }
    if let Some(calls) = common.max_tool_calls_per_step {
        spec.limits.max_tool_calls_per_step = calls;
    }
    if let Some(calls) = common.max_task_calls_per_step {
        spec.limits.max_task_calls_per_step = calls;
    }
    if let Some(calls) = common.max_total_tool_calls {
        spec.limits.max_total_tool_calls = calls;
    }
    if let Some(tokens) = common.max_total_tokens {
        spec.limits.max_total_tokens = tokens;
    }
    if let Some(bytes) = common.max_response_bytes {
        spec.limits.max_response_bytes = bytes;
    }
    if let Some(mib) = common.spill_quota_mib {
        spec.spill_quota_mib = mib;
    }
    if let Some(timeout) = common.shell_timeout {
        spec.shell_timeouts.0 = timeout.as_millis() as u64;
    }
    if let Some(timeout) = common.shell_max_timeout {
        spec.shell_timeouts.1 = timeout.as_millis() as u64;
    }
    if let Some(timeout) = common.model_request_timeout {
        spec.model_timeouts.request = timeout;
    }
    if let Some(timeout) = common.model_turn_timeout {
        spec.model_timeouts.turn = timeout;
    }
    if let Some(timeout) = common.model_idle_timeout {
        spec.model_timeouts.idle = timeout;
    }
    if common.cwd.is_some() {
        spec.cwd = common.cwd.clone();
    }
    if common.prompts.is_some() {
        spec.prompts = common.prompts.clone();
    }
    if common.skills_dir.is_some() {
        spec.skills_dir = common.skills_dir.clone();
    }
    if common.memory_dir.is_some() {
        spec.memory_dir = common.memory_dir.clone();
    }
    if let Some(path) = &common.mcp {
        // Two partial maps have no obvious union: the flag replaces the manifest,
        // and the manifest's own file is never read.
        spec.mcp.clear();
        spec.mcp_file = Some(path.clone());
    }
    Ok(spec)
}

/// Lists the configured servers' tools and every connection failure.
async fn inspect_mcp(spec: &AgentSpec) -> Result<()> {
    #[cfg(not(feature = "mcp"))]
    {
        if spec_has_mcp(spec) {
            return Err(Error::Config(
                "mason was built without the mcp feature".to_owned(),
            ));
        }
        Ok(())
    }
    #[cfg(feature = "mcp")]
    {
        if !spec.has_mcp() {
            return Err(Error::Config(
                "no MCP servers are configured; pass --mcp <file> or set mcp in the --config manifest"
                    .into(),
            ));
        }
        if !spec.allow_mcp {
            return Err(Error::Config(
                "configured MCP servers require an explicit grant \
                 (`allow_mcp: true`, `--allow-mcp` or MASON_ALLOW_MCP=true)"
                    .into(),
            ));
        }
        let servers = crate::mcp::McpServers::connect(
            &spec.mcp_configs()?,
            spec.execution_mode,
            spec.allow_network,
        )
        .await;
        for tool in servers.tools() {
            emit(&format!("{}\n", tool.spec().function.name));
        }
        for failure in servers.failures() {
            eprintln!("{failure}");
        }
        if servers.failures().is_empty() {
            Ok(())
        } else {
            Err(Error::Config("one or more MCP connections failed".into()))
        }
    }
}

#[cfg(not(feature = "mcp"))]
fn spec_has_mcp(spec: &AgentSpec) -> bool {
    !spec.mcp.is_empty()
}

async fn build_agent(common: &CommonArgs) -> Result<Agent> {
    agent_spec(common)?.into_builder().await?.build()
}

async fn run(prompt: Vec<String>, profile: String, common: CommonArgs) -> Result<()> {
    let agent = build_agent(&common).await?;
    run_to_stdout(&agent, &profile, &prompt.join(" ")).await
}

/// Runs one prompt on an in-memory conversation, streaming the answer to stdout
/// and tool activity to stderr. Reusable by embedders with their own wiring.
pub async fn run_to_stdout(agent: &Agent, profile: &str, prompt: &str) -> Result<()> {
    let (tx, mut rx) = mpsc::channel(1024);
    let printer = tokio::spawn(async move {
        // Stop printing at the first sequence gap; reconcile with the complete answer
        // afterward.
        let mut current_turn: Option<u32> = None;
        let mut expected_seq = 0u32;
        let mut contiguous = true;
        let mut printed = String::new();
        while let Some(event) = rx.recv().await {
            match event {
                RunEvent::Delta {
                    turn, seq, delta, ..
                } => {
                    if current_turn != Some(turn) {
                        current_turn = Some(turn);
                        expected_seq = 0;
                        contiguous = true;
                        printed.clear();
                    }
                    if contiguous && seq == expected_seq {
                        expected_seq = seq + 1;
                        if let ChatDelta::Text(text) = &delta {
                            emit(text);
                            printed.push_str(text);
                        }
                    } else {
                        contiguous = false;
                    }
                }
                RunEvent::ToolStarted {
                    name, arguments, ..
                } => {
                    eprintln!("> {name} {}", clip(&arguments, 300));
                }
                _ => {}
            }
        }
        printed
    });
    // Cancellation lets tools clean up before the event queue closes.
    let token = tokio_util::sync::CancellationToken::new();
    let run_future = agent.run_prompt_as_with_token(profile, prompt, Some(tx), token.clone());
    tokio::pin!(run_future);
    let outcome = tokio::select! {
        result = &mut run_future => result?,
        () = shutdown_signal() => {
            token.cancel();
            run_future.await?
        }
    };
    let interrupted = token.is_cancelled();
    let printed = printer.await.expect("the printer task cannot panic");
    let answer = outcome.answer.as_str();
    // Append any answer suffix lost by the observer queue.
    if let Some(rest) = answer.strip_prefix(&printed) {
        emit(rest);
    } else if printed != answer && !answer.is_empty() {
        // Unexpected divergence: completeness beats a clean transcript.
        if !printed.is_empty() {
            emit("\n");
        }
        emit(answer);
    }
    if !answer.is_empty() && !answer.ends_with('\n') {
        emit("\n");
    }
    if !interrupted {
        review_run(agent, &outcome.messages).await;
    }
    Ok(())
}

/// Grades a one-shot run, reflects on it and consolidates memory when due: a
/// one-shot process reviews before exiting, or it would never learn. SIGINT or
/// SIGTERM cancels the review.
async fn review_run(agent: &crate::Agent, messages: &[crate::Message]) {
    use crate::reflect::{RunReview, signals};
    let token = tokio_util::sync::CancellationToken::new();
    let review = async {
        let mut run = RunReview::new(messages);
        let verdict = match agent.evaluate(&run, &token).await {
            Ok(verdict) => verdict,
            Err(err) => {
                if !token.is_cancelled() {
                    tracing::warn!("evaluation failed: {err}");
                }
                None
            }
        };
        if let Some(verdict) = &verdict {
            eprintln!("> evaluated: {} ({})", verdict.outcome, verdict.reason);
        }
        run.verdict = verdict.as_ref();
        let Some(memory) = agent.memory() else {
            return;
        };
        if !signals(&run).is_empty() {
            eprintln!("> reflecting on this run");
            if let Err(err) = agent.reflect(run, token.clone()).await
                && !token.is_cancelled()
            {
                tracing::warn!("reflection failed: {err}");
            }
        }
        if !token.is_cancelled() && memory.consolidation_due().await {
            eprintln!("> consolidating memory");
            if let Err(err) = agent.consolidate(token.clone()).await
                && !token.is_cancelled()
            {
                tracing::warn!("memory consolidation failed: {err}");
            }
        }
    };
    tokio::pin!(review);
    tokio::select! {
        () = &mut review => {}
        () = shutdown_signal() => {
            token.cancel();
            review.await;
        }
    }
}

/// Writes to stdout without panicking when the reader has gone (`| head`).
fn emit(text: &str) {
    let mut stdout = std::io::stdout().lock();
    let _ = stdout
        .write_all(text.as_bytes())
        .and_then(|()| stdout.flush());
}

/// Serves the HTTP API until SIGINT/SIGTERM, then cancels active runs and drains
/// both the runs and the HTTP connections within [`DRAIN_GRACE`].
async fn serve(
    listener: tokio::net::TcpListener,
    security: crate::api::ApiSecurity,
    db: Option<PathBuf>,
    common: CommonArgs,
    limits: crate::runner::RunnerLimits,
    webhook: Option<crate::runner::RunWebhook>,
) -> Result<()> {
    let agent = build_agent(&common).await?;
    let db = db.unwrap_or_else(|| agent.cwd().join(".mason/mason.db"));
    let store = Arc::new(SqliteStore::open(&db).await?);
    let runner = Runner::with_limits(agent, store, limits)?;
    if let Some(webhook) = webhook {
        runner.notify_runs(webhook);
    }
    let recovery = runner.recover_and_drain().await?;
    if !recovery.interrupted.is_empty() {
        tracing::warn!(
            count = recovery.interrupted.len(),
            "marked abandoned runs interrupted"
        );
    }
    emit(&format!(
        "mason serving on http://{}\n",
        listener.local_addr().map_err(|err| {
            Error::Config(format!("cannot determine the listen address: {err}"))
        })?
    ));
    let _ = std::io::stdout().flush();

    let shutdown = tokio_util::sync::CancellationToken::new();
    let cleanup_spill = runner.agent().spill();
    let cleanup_token = shutdown.clone();
    let cleanup = tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(3600));
        loop {
            tokio::select! {
                _ = cleanup_token.cancelled() => break,
                _ = interval.tick() => cleanup_spill.cleanup(std::time::SystemTime::now()).await,
            }
        }
    });
    let app = crate::api::router_with_security(runner.clone(), shutdown.clone(), security);
    let drain_http = async {
        let server = std::future::IntoFuture::into_future(
            axum::serve(listener, app).with_graceful_shutdown({
                let shutdown = shutdown.clone();
                async move {
                    shutdown_signal().await;
                    shutdown.cancel();
                }
            }),
        );
        tokio::pin!(server);
        // Bound HTTP draining even when a viewer cannot flush its TCP connection.
        let drained = {
            let shutdown = shutdown.clone();
            async move {
                shutdown.cancelled().await;
                tokio::time::sleep(DRAIN_GRACE).await;
            }
        };
        let result = tokio::select! {
            result = &mut server => Some(result),
            () = drained => None,
        };
        shutdown.cancel();
        if result.is_none() {
            tracing::warn!(
                "the HTTP drain exceeded {DRAIN_GRACE:?}; closing with connections open"
            );
        }
        result
    };
    let cleanup_runs = async {
        shutdown.cancelled().await;
        runner.shutdown(DRAIN_GRACE).await;
    };
    let (result, ()) = tokio::join!(drain_http, cleanup_runs);
    cleanup.abort();
    match result {
        Some(Err(err)) => Err(Error::Config(format!("the HTTP server failed: {err}"))),
        _ => Ok(()),
    }
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("the ctrl-c handler installs");
    };
    #[cfg(unix)]
    let terminate = async {
        let mut signal = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("the SIGTERM handler installs");
        signal.recv().await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
    eprintln!("mason: shutting down");
}

fn clip(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_owned();
    }
    let clipped: String = text.chars().take(max_chars).collect();
    format!("{clipped}...")
}

#[cfg(test)]
mod tests {
    use clap::{CommandFactory, Parser};

    use super::*;
    use crate::config::{
        DEFAULT_MAX_RESPONSE_BYTES, DEFAULT_MAX_STEPS, DEFAULT_MAX_TOTAL_TOKENS,
        DEFAULT_MAX_TOTAL_TOOL_CALLS, DEFAULT_MODEL_IDLE_TIMEOUT_MS,
        DEFAULT_MODEL_REQUEST_TIMEOUT_MS, DEFAULT_MODEL_TURN_TIMEOUT_MS,
        DEFAULT_RUN_TIMEOUT_SECONDS, DEFAULT_SHELL_MAX_TIMEOUT_MS, DEFAULT_SHELL_TIMEOUT_MS,
        DEFAULT_SPILL_QUOTA_MIB,
    };

    #[test]
    fn the_cli_definition_is_valid() {
        Cli::command().debug_assert();
    }

    /// The shared options as `run` would parse them.
    fn common(args: &[&str]) -> CommonArgs {
        let mut argv: Vec<&str> = vec!["mason", "run"];
        argv.extend_from_slice(args);
        argv.push("a prompt");
        match Cli::try_parse_from(argv).unwrap().command {
            Command::Run { common, .. } => common,
            _ => unreachable!("run was requested"),
        }
    }

    fn manifest(dir: &std::path::Path, text: &str) -> String {
        let path = dir.join("agent.yaml");
        std::fs::write(&path, text).unwrap();
        path.to_str().unwrap().to_owned()
    }

    const MANIFEST: &str = "version: 1\nexecution_mode: trusted\nallow_mcp: true\ninstructions: Domain prose.\ncwd: work\nlimits:\n  max_steps: 40\n  timeout: 20m\nmcp:\n  a:\n    type: local\n    command: [node, a.js]\n";

    #[test]
    fn a_manifest_survives_the_clap_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let config = manifest(dir.path(), MANIFEST);
        let spec = agent_spec(&common(&["--config", &config])).unwrap();
        assert_eq!(spec.execution_mode, ExecutionMode::Trusted);
        assert!(spec.allow_mcp, "the manifest's grant survives");
        assert!(!spec.allow_network);
        assert_eq!(spec.instructions.as_deref(), Some("Domain prose."));
        assert_eq!(spec.limits.max_steps, 40);
        assert_eq!(spec.limits.timeout, Duration::from_secs(20 * 60));
        assert_eq!(spec.cwd, Some(dir.path().join("work")));
        assert_eq!(spec.mcp.len(), 1, "the manifest's servers survive");
        assert_eq!(
            spec.limits.max_total_tool_calls, DEFAULT_MAX_TOTAL_TOOL_CALLS,
            "an omitted budget keeps mason's default"
        );
    }

    #[test]
    fn explicit_flags_override_the_manifest_and_nothing_else() {
        let dir = tempfile::tempdir().unwrap();
        let config = manifest(dir.path(), MANIFEST);
        let spec = agent_spec(&common(&[
            "--config",
            &config,
            "--max-steps",
            "7",
            "--execution-mode",
            "sandbox",
            "--cwd",
            "/tmp/elsewhere",
        ]))
        .unwrap();
        assert_eq!(spec.limits.max_steps, 7);
        assert_eq!(spec.execution_mode, ExecutionMode::Sandbox);
        assert_eq!(
            spec.cwd.as_deref(),
            Some(std::path::Path::new("/tmp/elsewhere")),
            "a flag path is not resolved against the manifest"
        );
        assert!(spec.allow_mcp, "an unset flag leaves the manifest alone");
        assert_eq!(spec.instructions.as_deref(), Some("Domain prose."));
    }

    #[test]
    fn an_explicit_false_grant_revokes_a_manifest_true() {
        let dir = tempfile::tempdir().unwrap();
        let config = manifest(dir.path(), MANIFEST);
        let spec = agent_spec(&common(&["--config", &config, "--allow-mcp=false"])).unwrap();
        assert!(!spec.allow_mcp);
        let spec = agent_spec(&common(&["--config", &config, "--allow-network=false"])).unwrap();
        assert!(!spec.allow_network);
        // Bare flags still grant.
        let spec = agent_spec(&common(&["--allow-mcp", "--allow-network"])).unwrap();
        assert!(spec.allow_mcp && spec.allow_network);
    }

    #[test]
    fn the_mcp_flag_replaces_the_manifest_servers_wholesale() {
        let dir = tempfile::tempdir().unwrap();
        let config = manifest(dir.path(), MANIFEST);
        let other = dir.path().join("other.json");
        std::fs::write(
            &other,
            r#"{"b": {"type": "local", "command": ["node", "b.js"]}}"#,
        )
        .unwrap();
        let spec = agent_spec(&common(&[
            "--config",
            &config,
            "--mcp",
            other.to_str().unwrap(),
        ]))
        .unwrap();
        assert!(spec.mcp.is_empty(), "the manifest's servers are dropped");
        assert_eq!(spec.mcp_file.as_deref(), Some(other.as_path()));
        #[cfg(feature = "mcp")]
        {
            let configs = spec.mcp_configs().unwrap();
            let names: Vec<&String> = configs.keys().collect();
            assert_eq!(names, ["b"], "two partial maps have no obvious union");
        }
    }

    #[test]
    fn an_mcp_override_never_reads_the_manifests_default_file() {
        let dir = tempfile::tempdir().unwrap();
        // The manifest names a file that does not exist; the operator replaces it.
        let config = manifest(
            dir.path(),
            "version: 1\nexecution_mode: trusted\nallow_mcp: true\nmcp_file: absent.json\n",
        );
        let other = dir.path().join("other.json");
        std::fs::write(
            &other,
            r#"{"b": {"type": "local", "command": ["node", "b.js"]}}"#,
        )
        .unwrap();
        let spec = agent_spec(&common(&[
            "--config",
            &config,
            "--mcp",
            other.to_str().unwrap(),
        ]))
        .expect("the absent default must not be read");
        #[cfg(not(feature = "mcp"))]
        let _ = &spec;
        #[cfg(feature = "mcp")]
        assert_eq!(spec.mcp_configs().unwrap().len(), 1);
        // Without the override, the absent file is reported when it is used.
        let spec = agent_spec(&common(&["--config", &config])).unwrap();
        #[cfg(not(feature = "mcp"))]
        let _ = &spec;
        #[cfg(feature = "mcp")]
        assert!(
            spec.mcp_configs()
                .unwrap_err()
                .to_string()
                .contains("cannot read")
        );
    }

    #[test]
    fn without_a_manifest_the_documented_defaults_apply() {
        let spec = agent_spec(&common(&[])).unwrap();
        assert_eq!(spec.execution_mode, ExecutionMode::Trusted);
        assert!(!spec.allow_network && !spec.allow_mcp);
        assert_eq!(spec.instructions, None);
        assert!(spec.profiles.is_empty() && spec.select_profile.is_empty());
        assert_eq!(spec.limits.max_steps, DEFAULT_MAX_STEPS);
        assert_eq!(
            spec.limits.timeout,
            Duration::from_secs(DEFAULT_RUN_TIMEOUT_SECONDS)
        );
        assert_eq!(
            spec.limits.max_total_tool_calls,
            DEFAULT_MAX_TOTAL_TOOL_CALLS
        );
        assert_eq!(spec.limits.max_total_tokens, DEFAULT_MAX_TOTAL_TOKENS);
        assert_eq!(spec.limits.max_response_bytes, DEFAULT_MAX_RESPONSE_BYTES);
        assert_eq!(spec.spill_quota_mib, DEFAULT_SPILL_QUOTA_MIB);
        assert_eq!(
            spec.shell_timeouts,
            (DEFAULT_SHELL_TIMEOUT_MS, DEFAULT_SHELL_MAX_TIMEOUT_MS)
        );
        assert_eq!(
            spec.model_timeouts.request,
            Duration::from_millis(DEFAULT_MODEL_REQUEST_TIMEOUT_MS)
        );
        assert_eq!(
            spec.model_timeouts.turn,
            Duration::from_millis(DEFAULT_MODEL_TURN_TIMEOUT_MS)
        );
        assert_eq!(
            spec.model_timeouts.idle,
            Duration::from_millis(DEFAULT_MODEL_IDLE_TIMEOUT_MS)
        );
    }

    #[test]
    fn flags_override_every_manifest_budget() {
        let dir = tempfile::tempdir().unwrap();
        // The manifest sets every budget too, so each flag must win.
        let config = manifest(
            dir.path(),
            "version: 1\nlimits:\n  timeout: 20m\n  max_concurrent_tools: 2\n  max_tool_calls_per_step: 3\n  max_task_calls_per_step: 1\n  max_total_tool_calls: 5\n  max_total_tokens: 6\n  max_response_bytes: 7\n  spill_quota_mib: 8\n  shell_timeout: 9s\n  shell_max_timeout: 10s\n  model_request_timeout: 11s\n  model_turn_timeout: 12s\n  model_idle_timeout: 13s\n",
        );
        let manifest_only = agent_spec(&common(&["--config", &config])).unwrap();
        assert_eq!(manifest_only.limits.max_concurrent_tools, 2);
        assert_eq!(manifest_only.spill_quota_mib, 8);
        assert_eq!(manifest_only.shell_timeouts, (9_000, 10_000));
        assert_eq!(
            manifest_only.model_timeouts.request,
            Duration::from_secs(11)
        );
        assert_eq!(manifest_only.model_timeouts.turn, Duration::from_secs(12));
        assert_eq!(manifest_only.model_timeouts.idle, Duration::from_secs(13));
        let spec = agent_spec(&common(&[
            "--config",
            &config,
            "--run-timeout",
            "1m",
            "--max-concurrent-tools",
            "12",
            "--max-tool-calls-per-step",
            "13",
            "--max-task-calls-per-step",
            "14",
            "--max-total-tool-calls",
            "11",
            "--max-total-tokens",
            "22",
            "--max-response-bytes",
            "33",
            "--spill-quota-mib",
            "44",
            "--shell-timeout",
            "55ms",
            "--shell-max-timeout",
            "66ms",
            "--model-request-timeout",
            "77ms",
            "--model-turn-timeout",
            "88ms",
            "--model-idle-timeout",
            "99ms",
        ]))
        .unwrap();
        assert_eq!(spec.limits.timeout, Duration::from_secs(60));
        assert_eq!(spec.limits.max_concurrent_tools, 12);
        assert_eq!(spec.limits.max_tool_calls_per_step, 13);
        assert_eq!(spec.limits.max_task_calls_per_step, 14);
        assert_eq!(spec.limits.max_total_tool_calls, 11);
        assert_eq!(spec.limits.max_total_tokens, 22);
        assert_eq!(spec.limits.max_response_bytes, 33);
        assert_eq!(spec.spill_quota_mib, 44);
        assert_eq!(spec.shell_timeouts, (55, 66));
        assert_eq!(spec.model_timeouts.request, Duration::from_millis(77));
        assert_eq!(spec.model_timeouts.turn, Duration::from_millis(88));
        assert_eq!(spec.model_timeouts.idle, Duration::from_millis(99));
        assert!(
            Cli::try_parse_from(["mason", "run", "--run-timeout", "900", "p"])
                .is_err_and(|err| err.to_string().contains("900")),
            "a bare number is rejected like in the manifest"
        );
    }
}
