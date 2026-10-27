//! Object-safe tools, typed argument validation, and cached schema registration.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::{OnceLock, RwLock, Weak};

use schemars::{JsonSchema, schema_for};
use serde::de::DeserializeOwned;
use serde_json::Value;
use tokio::sync::{Mutex, OwnedMutexGuard};
use tokio_util::sync::CancellationToken;

use crate::Error;
use crate::error::Result;
use crate::llm::ToolSpec;

pub mod builtin;
pub mod output;
pub mod policy;
pub mod process;
pub mod replace;

pub use builtin::builtin_tools;
pub use output::{CapturedOutput, Keep, OutputCapture, OutputSpill};

#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    /// Model-visible failure; the run continues.
    #[error("{0}")]
    Failure(String),
    /// Fatal failure; cancels the run and allows bounded sibling cleanup.
    #[error(transparent)]
    Fatal(#[from] Error),
}

impl ToolError {
    /// A model-visible failure.
    pub fn failure(message: impl Into<String>) -> Self {
        ToolError::Failure(message.into())
    }

    /// Whether this error must stop the run.
    pub fn is_fatal(&self) -> bool {
        matches!(self, ToolError::Fatal(_))
    }
}

/// Tool execution context: working directory, cancellation, and output spilling.
#[derive(Clone)]
pub struct ToolCtx {
    /// The directory relative paths resolve against.
    pub cwd: PathBuf,
    /// Cancelled when the run is interrupted or hits its deadline.
    pub token: CancellationToken,
    /// Spill storage and output bounds.
    pub spill: Arc<OutputSpill>,
    pub processes: Arc<process::BackgroundProcesses>,
    pub budget: Option<Arc<crate::engine::RunBudget>>,
    pub shell_timeout_ms: u64,
    pub shell_max_timeout_ms: u64,
    pub execution_mode: policy::ExecutionMode,
    pub allow_network: bool,
    pub policy: Arc<dyn policy::ToolPolicy>,
    /// The running profile's limits on `bash`, `write` and `edit`, and its hooks.
    pub rules: Option<Arc<crate::profile::Rules>>,
    /// Per-session resource scope; different sessions and children never share MCP state.
    pub resources: Arc<std::sync::Mutex<HashMap<String, Arc<dyn std::any::Any + Send + Sync>>>>,
}

/// Background processes and MCP connections that outlive one run: a stored
/// session keeps them between its runs.
#[derive(Clone, Default)]
pub struct ToolScope {
    pub processes: Arc<process::BackgroundProcesses>,
    pub resources: Arc<std::sync::Mutex<HashMap<String, Arc<dyn std::any::Any + Send + Sync>>>>,
}

impl ToolScope {
    /// Stops background processes and closes MCP connections.
    pub async fn close(&self) {
        self.processes.stop_all();
        #[cfg(feature = "mcp")]
        crate::mcp::close_scope(&self.resources).await;
        self.resources.lock().expect("tool resource mutex").clear();
    }
}

impl std::fmt::Debug for ToolCtx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolCtx")
            .field("cwd", &self.cwd)
            .field("cancelled", &self.token.is_cancelled())
            .field("spill", &self.spill)
            .finish()
    }
}

impl ToolCtx {
    /// A context with a fresh token and the default spill directory.
    pub fn new(cwd: impl Into<PathBuf>) -> Self {
        Self {
            cwd: cwd.into(),
            token: CancellationToken::new(),
            spill: Arc::new(OutputSpill::new(OutputSpill::default_dir())),
            processes: Arc::new(process::BackgroundProcesses::default()),
            budget: None,
            shell_timeout_ms: 120_000,
            shell_max_timeout_ms: 600_000,
            execution_mode: policy::ExecutionMode::Trusted,
            allow_network: false,
            policy: Arc::new(policy::AllowTools),
            rules: None,
            resources: Arc::new(std::sync::Mutex::new(HashMap::new())),
        }
    }

    /// The processes and MCP connections this context's tools share.
    pub fn scope(&self) -> ToolScope {
        ToolScope {
            processes: self.processes.clone(),
            resources: self.resources.clone(),
        }
    }

    pub fn with_scope(mut self, scope: &ToolScope) -> Self {
        self.processes = scope.processes.clone();
        self.resources = scope.resources.clone();
        self
    }

    /// Sets the running profile's rules.
    pub fn with_rules(mut self, rules: Arc<crate::profile::Rules>) -> Self {
        self.rules = Some(rules);
        self
    }

    /// Sets the cancellation token (a child of the run's token).
    pub fn with_token(mut self, token: CancellationToken) -> Self {
        self.token = token;
        self
    }

    /// Sets custom spill storage.
    pub fn with_spill(mut self, spill: Arc<OutputSpill>) -> Self {
        self.spill = spill;
        self
    }

    pub fn with_execution(
        mut self,
        mode: policy::ExecutionMode,
        network: bool,
        policy: Arc<dyn policy::ToolPolicy>,
    ) -> Self {
        self.execution_mode = mode;
        self.allow_network = network;
        self.policy = policy;
        self
    }

    pub fn checked_path(&self, path: &str) -> Result<PathBuf, ToolError> {
        if self.execution_mode == policy::ExecutionMode::Sandbox {
            policy::workspace_path(&self.cwd, path)
        } else {
            Ok(self.resolve(path))
        }
    }

    /// Resolves paths against `cwd`; absolute paths are allowed, with no sandbox
    /// confinement.
    pub fn resolve(&self, path: &str) -> PathBuf {
        self.cwd.join(path)
    }
}

/// The object-safe tool interface the engine calls.
#[async_trait::async_trait]
pub trait Tool: Send + Sync {
    /// The wire spec: name, description and arguments JSON Schema. The
    /// `Toolset` computes this once at registration.
    fn spec(&self) -> ToolSpec;

    /// Runs one call with raw JSON arguments. Implementations validate;
    /// [`Typed`] does it through serde.
    async fn call(&self, args: Value, ctx: &ToolCtx) -> std::result::Result<String, ToolError>;

    /// Whether the tool only reads: a `read_only` profile's globs admit no other.
    fn read_only(&self) -> bool {
        false
    }

    /// The MCP server the tool comes from, if any.
    fn mcp_server(&self) -> Option<&str> {
        None
    }
}

/// A tool with typed, schema-derived arguments. Wrap in [`Typed`] to get the
/// [`Tool`] implementation.
#[async_trait::async_trait]
pub trait TypedTool: Send + Sync {
    /// The arguments; builtin argument structs use `deny_unknown_fields`.
    type Args: DeserializeOwned + JsonSchema;

    /// The unique registry name the model calls.
    fn name(&self) -> &str;

    /// What the tool does, for the model.
    fn description(&self) -> &str;

    /// Runs one call with validated arguments.
    async fn run(&self, args: Self::Args, ctx: &ToolCtx) -> std::result::Result<String, ToolError>;

    /// Whether the tool only reads.
    fn read_only(&self) -> bool {
        false
    }
}

/// Derives argument schemas and validation, exposing invalid arguments as model-visible
/// errors.
pub struct Typed<T>(pub T);

#[async_trait::async_trait]
impl<T: TypedTool> Tool for Typed<T> {
    fn spec(&self) -> ToolSpec {
        let schema = schema_for!(T::Args);
        let parameters =
            serde_json::to_value(schema).unwrap_or_else(|_| serde_json::json!({"type": "object"}));
        ToolSpec::function(self.0.name(), self.0.description(), parameters)
    }

    async fn call(&self, args: Value, ctx: &ToolCtx) -> std::result::Result<String, ToolError> {
        let args = serde_json::from_value::<T::Args>(args)
            .map_err(|err| ToolError::Failure(format!("invalid arguments: {err}")))?;
        self.0.run(args, ctx).await
    }

    fn read_only(&self) -> bool {
        self.0.read_only()
    }
}

struct Entry {
    tool: Arc<dyn Tool>,
    spec: ToolSpec,
}

/// Tool registry with unique names and specs cached at registration.
#[derive(Default)]
pub struct Toolset {
    entries: Vec<Entry>,
}

impl Toolset {
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a tool, rejecting duplicate names.
    pub fn push(&mut self, tool: Box<dyn Tool>) -> Result<()> {
        self.push_shared(Arc::from(tool))
    }

    /// Registers a shared tool; per-run toolsets reuse the base registry's
    /// instances through this.
    pub fn push_shared(&mut self, tool: Arc<dyn Tool>) -> Result<()> {
        let spec = tool.spec();
        if self
            .entries
            .iter()
            .any(|entry| entry.spec.function.name == spec.function.name)
        {
            return Err(Error::Config(format!(
                "duplicate tool name `{}`; tool names are unique registry keys",
                spec.function.name
            )));
        }
        self.entries.push(Entry { tool, spec });
        Ok(())
    }

    /// Registers many tools; fails on the first duplicate.
    pub fn extend(&mut self, tools: impl IntoIterator<Item = Box<dyn Tool>>) -> Result<()> {
        for tool in tools {
            self.push(tool)?;
        }
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.entries
            .iter()
            .find(|entry| entry.spec.function.name == name)
            .map(|entry| entry.tool.clone())
    }

    pub fn specs(&self) -> Vec<ToolSpec> {
        self.entries
            .iter()
            .map(|entry| entry.spec.clone())
            .collect()
    }

    /// The registered names, in registration order.
    pub fn names(&self) -> Vec<&str> {
        self.entries
            .iter()
            .map(|entry| entry.spec.function.name.as_str())
            .collect()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Per-path locks serialize edit/write mutations. Weak entries are pruned; shell
/// mutations are not covered.
static PATH_LOCKS: OnceLock<RwLock<HashMap<PathBuf, Weak<Mutex<()>>>>> = OnceLock::new();

/// Locks a full read–modify–write cycle; callers must not re-enter the same path lock.
pub(crate) async fn lock_path(path: &Path) -> OwnedMutexGuard<()> {
    let registry = PATH_LOCKS.get_or_init(Default::default);
    let key = normalize_path(path).await;
    let mutex = {
        let mut map = registry
            .write()
            .expect("the path lock registry is poisoned");
        if map.len() > 64 {
            map.retain(|_, lock| lock.strong_count() > 0);
        }
        match map.get(&key).and_then(Weak::upgrade) {
            Some(mutex) => mutex,
            None => {
                let mutex = Arc::new(Mutex::new(()));
                map.insert(key, Arc::downgrade(&mutex));
                mutex
            }
        }
    };
    mutex.lock_owned().await
}

/// Canonicalizes existing targets or their parent directory so aliases share a lock;
/// falls back to the raw path.
async fn normalize_path(path: &Path) -> PathBuf {
    if let Ok(canonical) = tokio::fs::canonicalize(path).await {
        return canonical;
    }
    match (path.parent(), path.file_name()) {
        (Some(parent), Some(name)) => tokio::fs::canonicalize(parent)
            .await
            .map(|parent| parent.join(name))
            .unwrap_or_else(|_| path.to_path_buf()),
        _ => path.to_path_buf(),
    }
}

#[cfg(test)]
mod tests {
    use serde::Deserialize;

    use super::*;

    #[derive(Deserialize, JsonSchema)]
    #[serde(deny_unknown_fields)]
    struct EchoArgs {
        /// The text to echo.
        message: String,
        /// How many times.
        times: Option<u8>,
    }

    struct Echo;

    #[async_trait::async_trait]
    impl TypedTool for Echo {
        type Args = EchoArgs;
        fn name(&self) -> &str {
            "echo"
        }
        fn description(&self) -> &str {
            "Echo the message."
        }
        async fn run(
            &self,
            args: EchoArgs,
            _ctx: &ToolCtx,
        ) -> std::result::Result<String, ToolError> {
            let times = args.times.unwrap_or(1);
            Ok(args.message.repeat(times as usize))
        }
    }

    fn ctx() -> ToolCtx {
        ToolCtx::new(std::env::temp_dir())
    }

    #[tokio::test]
    async fn typed_tools_validate_arguments_for_the_model() {
        let tool = Typed(Echo);
        let spec = tool.spec();
        assert_eq!(spec.function.name, "echo");
        assert_eq!(spec.kind, "function");

        let out = tool
            .call(serde_json::json!({"message": "ab", "times": 2}), &ctx())
            .await
            .unwrap();
        assert_eq!(out, "abab");

        let err = tool.call(serde_json::json!({}), &ctx()).await.unwrap_err();
        assert!(err.to_string().contains("missing field `message`"), "{err}");

        let err = tool
            .call(serde_json::json!({"message": "x", "surprise": 1}), &ctx())
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("unknown field `surprise`"),
            "{err}"
        );

        let err = tool
            .call(serde_json::json!({"message": "x", "times": "many"}), &ctx())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("invalid type"), "{err}");
        assert!(
            matches!(err, ToolError::Failure(_)),
            "validation is model-visible"
        );
    }

    #[tokio::test]
    async fn toolset_rejects_duplicate_names() {
        let mut set = Toolset::new();
        set.push(Box::new(Typed(Echo))).unwrap();
        let err = set.push(Box::new(Typed(Echo))).unwrap_err();
        assert!(
            err.to_string().contains("duplicate tool name `echo`"),
            "{err}"
        );
        assert_eq!(set.len(), 1);
        assert!(set.get("echo").is_some());
        assert!(set.get("nope").is_none());
        assert_eq!(set.names(), ["echo"]);
    }

    #[test]
    fn resolve_follows_path_join_semantics() {
        let ctx = ToolCtx::new("/work/dir");
        assert_eq!(ctx.resolve("a/b.txt"), PathBuf::from("/work/dir/a/b.txt"));
        assert_eq!(ctx.resolve("/etc/hosts"), PathBuf::from("/etc/hosts"));
    }

    #[tokio::test]
    async fn lock_path_serializes_the_same_target() {
        use std::sync::Mutex;

        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        let plain = dir.path().join("f.txt");
        let spelled = dir.path().join("sub").join("..").join("f.txt");
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut tasks = Vec::new();
        for (index, path) in [plain, spelled].into_iter().enumerate() {
            let log = log.clone();
            tasks.push(tokio::spawn(async move {
                let _guard = lock_path(&path).await;
                log.lock().unwrap().push(format!("enter {index}"));
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                log.lock().unwrap().push(format!("exit {index}"));
            }));
        }
        for task in tasks {
            task.await.unwrap();
        }
        let log = log.lock().unwrap();
        assert_eq!(log.len(), 4, "{log:?}");
        assert!(log[0].starts_with("enter"), "{log:?}");
        let first = log[0].split(' ').nth(1).unwrap();
        assert_eq!(log[1], format!("exit {first}"), "{log:?}");
        assert!(log[2].starts_with("enter"), "{log:?}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn lock_path_follows_symlinks_to_one_target() {
        use std::sync::Mutex;

        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("target.txt");
        std::fs::write(&target, "x\n").unwrap();
        let alias = dir.path().join("alias.txt");
        std::os::unix::fs::symlink(&target, &alias).unwrap();

        let log = Arc::new(Mutex::new(Vec::new()));
        let mut tasks = Vec::new();
        for (index, path) in [target, alias].into_iter().enumerate() {
            let log = log.clone();
            tasks.push(tokio::spawn(async move {
                let _guard = lock_path(&path).await;
                log.lock().unwrap().push(format!("enter {index}"));
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                log.lock().unwrap().push(format!("exit {index}"));
            }));
        }
        for task in tasks {
            task.await.unwrap();
        }
        let log = log.lock().unwrap();
        assert_eq!(log.len(), 4, "{log:?}");
        assert!(log[0].starts_with("enter"), "{log:?}");
        let first = log[0].split(' ').nth(1).unwrap();
        assert_eq!(log[1], format!("exit {first}"), "{log:?}");
        assert!(log[2].starts_with("enter"), "{log:?}");
    }
}
