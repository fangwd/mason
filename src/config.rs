//! Declarative application configuration: the data-shaped subset of [`AgentBuilder`].
//!
//! A manifest is a YAML file that expresses everything an embedding application
//! needs that is data — instructions, profiles, MCP servers, budgets, grants and
//! paths — so `mason serve --config app.yaml` replaces an embedder crate. What a
//! manifest cannot express stays code: native [`Tool`](crate::Tool) implementations,
//! a custom [`ChatModel`](crate::ChatModel) or [`SessionStore`](crate::SessionStore),
//! and a [`ToolPolicy`](crate::tool::policy::ToolPolicy) approval hook.
//!
//! [`Manifest::into_builder`] returns an [`AgentBuilder`], so an application can
//! start declarative and add code on top of the same manifest.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use serde::Deserialize;

use crate::Error;
use crate::agent::AgentBuilder;
use crate::engine::Limits;
use crate::error::Result;
use crate::llm::{ModelTimeouts, Vendors};
use crate::profile::{Profile, ProfilePatch};
use crate::prompt::PromptSource;
use crate::tool::policy::ExecutionMode;

/// The manifest schema this mason understands. A schema version, not a software
/// version: binary compatibility is a release pin plus the healthz version floor.
pub const SCHEMA_VERSION: u32 = 1;

/// Default run budget, matching `Limits::default`.
pub const DEFAULT_RUN_TIMEOUT_SECONDS: u64 = 900;
/// Default model-using steps per run.
pub const DEFAULT_MAX_STEPS: usize = 150;
pub const DEFAULT_MAX_TOTAL_TOOL_CALLS: usize = 1000;
pub const DEFAULT_MAX_TOTAL_TOKENS: usize = 20_000_000;
pub const DEFAULT_MAX_RESPONSE_BYTES: usize = 8 * 1024 * 1024;
/// Default total spill storage, in MiB.
pub const DEFAULT_SPILL_QUOTA_MIB: u64 = 256;
pub const DEFAULT_SHELL_TIMEOUT_MS: u64 = 120_000;
pub const DEFAULT_SHELL_MAX_TIMEOUT_MS: u64 = 600_000;
pub const DEFAULT_MODEL_REQUEST_TIMEOUT_MS: u64 = 120_000;
pub const DEFAULT_MODEL_TURN_TIMEOUT_MS: u64 = 180_000;
pub const DEFAULT_MODEL_IDLE_TIMEOUT_MS: u64 = 60_000;

/// A declarative application configuration.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    /// The schema version; must be [`SCHEMA_VERSION`].
    pub version: u32,
    /// Domain system-prompt prose, appended verbatim after the generic base.
    #[serde(default, deserialize_with = "crate::profile::deny_null")]
    pub instructions: Option<String>,
    /// Domain profiles, and partial patches of builtin ones, by name.
    #[serde(default, deserialize_with = "crate::profile::deny_null")]
    pub profiles: Option<BTreeMap<String, ProfilePatch>>,
    /// Profile selection from session metadata (see the `select_profile` rules).
    /// Ordered; the first matching rule wins.
    #[serde(default, deserialize_with = "crate::profile::deny_null")]
    pub select_profile: Option<Vec<SelectRule>>,
    /// MCP servers, in the opencode format; mutually exclusive with `mcp_file`.
    #[serde(default, deserialize_with = "crate::profile::deny_null")]
    pub mcp: Option<BTreeMap<String, McpEntry>>,
    /// A JSON file of MCP servers, resolved against the manifest's directory.
    #[serde(default, deserialize_with = "crate::profile::deny_null")]
    pub mcp_file: Option<PathBuf>,
    /// Sandbox confines subprocesses and file tools; trusted (the default)
    /// grants host access.
    #[serde(default)]
    pub execution_mode: ExecutionMode,
    /// Network access for sandboxed subprocesses; defaults to false.
    #[serde(default)]
    pub allow_network: bool,
    /// Whether the configured MCP servers may be called; defaults to false.
    #[serde(default)]
    pub allow_mcp: bool,
    /// A complete prompt file, resolved against the manifest's directory.
    #[serde(default, deserialize_with = "crate::profile::deny_null")]
    pub prompts: Option<PathBuf>,
    /// Skills directory, resolved against the manifest's directory.
    #[serde(default, deserialize_with = "crate::profile::deny_null")]
    pub skills_dir: Option<PathBuf>,
    /// Memory directory, resolved against the manifest's directory.
    #[serde(default, deserialize_with = "crate::profile::deny_null")]
    pub memory_dir: Option<PathBuf>,
    /// Partial budget overrides; omitted fields keep their defaults.
    #[serde(default, deserialize_with = "crate::profile::deny_null")]
    pub limits: Option<LimitsPatch>,
    /// Spill directory; only meaningful in trusted mode.
    #[serde(default, deserialize_with = "crate::profile::deny_null")]
    pub spill: Option<PathBuf>,
    /// Working directory, resolved against the manifest's directory.
    #[serde(default, deserialize_with = "crate::profile::deny_null")]
    pub cwd: Option<PathBuf>,
    /// Set by [`Manifest::load`]; relative paths resolve against it.
    #[serde(skip)]
    pub base_dir: PathBuf,
}

/// Partial budget overrides. Durations are human strings (`90s`, `15m`, `1h30m`);
/// a bare number is rejected naming the field.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LimitsPatch {
    #[serde(default, deserialize_with = "crate::profile::deny_null")]
    pub max_steps: Option<usize>,
    #[serde(default, deserialize_with = "crate::profile::deny_null")]
    pub max_concurrent_tools: Option<usize>,
    #[serde(default, deserialize_with = "crate::profile::deny_null")]
    pub max_tool_calls_per_step: Option<usize>,
    #[serde(default, deserialize_with = "crate::profile::deny_null")]
    pub max_task_calls_per_step: Option<usize>,
    /// A human duration (`90s`, `15m`, `1h30m`); a bare number is rejected.
    #[serde(default, deserialize_with = "human_duration")]
    pub timeout: Option<Duration>,
    #[serde(default, deserialize_with = "crate::profile::deny_null")]
    pub max_total_tool_calls: Option<usize>,
    #[serde(default, deserialize_with = "crate::profile::deny_null")]
    pub max_total_tokens: Option<usize>,
    #[serde(default, deserialize_with = "crate::profile::deny_null")]
    pub max_response_bytes: Option<usize>,
    /// Default and maximum `bash` timeouts.
    #[serde(default, deserialize_with = "human_duration")]
    pub shell_timeout: Option<Duration>,
    #[serde(default, deserialize_with = "human_duration")]
    pub shell_max_timeout: Option<Duration>,
    /// Model request (until the first byte), turn and stream-idle timeouts.
    #[serde(default, deserialize_with = "human_duration")]
    pub model_request_timeout: Option<Duration>,
    #[serde(default, deserialize_with = "human_duration")]
    pub model_turn_timeout: Option<Duration>,
    #[serde(default, deserialize_with = "human_duration")]
    pub model_idle_timeout: Option<Duration>,
    /// Total spill storage in MiB.
    #[serde(default, deserialize_with = "crate::profile::deny_null")]
    pub spill_quota_mib: Option<u64>,
}

/// One MCP server entry. `McpServerConfig` is internally tagged and so cannot deny
/// unknown keys; the manifest mirrors it as a strict flat struct and validates the
/// per-type key combinations itself.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct McpEntry {
    /// `local` (a spawned child) or `remote` (streamable HTTP).
    #[serde(rename = "type")]
    pub kind: String,
    /// Local only: the command and its arguments.
    #[serde(default, deserialize_with = "crate::profile::deny_null")]
    pub command: Option<Vec<String>>,
    /// Remote only: the server URL.
    #[serde(default, deserialize_with = "crate::profile::deny_null")]
    pub url: Option<String>,
    /// Local only: the child's working directory.
    #[serde(default, deserialize_with = "crate::profile::deny_null")]
    pub cwd: Option<String>,
    /// Local only: extra environment for the child, over the inherited one.
    #[serde(default, deserialize_with = "crate::profile::deny_null")]
    pub environment: Option<BTreeMap<String, String>>,
    /// Remote only: headers sent with every request.
    #[serde(default, deserialize_with = "crate::profile::deny_null")]
    pub headers: Option<BTreeMap<String, String>>,
    /// `false` skips the server entirely.
    #[serde(default, deserialize_with = "crate::profile::deny_null")]
    pub enabled: Option<bool>,
    /// Connect, discovery and per-call timeout in milliseconds.
    #[serde(default, deserialize_with = "crate::profile::deny_null")]
    pub timeout: Option<u64>,
}

/// One `select_profile` rule: a profile to run when every condition holds.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectRule {
    /// The profile to run; must be registered.
    pub profile: String,
    /// Metadata conditions, ANDed. Paths may be dotted for nested metadata.
    pub when_metadata: BTreeMap<String, MetadataMatch>,
}

/// How one metadata path must look for a rule to match.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
pub enum MetadataMatch {
    /// The path exists and is not JSON null. `0`, `false` and `""` match.
    Present(bool),
    /// The path holds exactly this JSON value; no string coercion.
    Equals(serde_json::Value),
}

/// The model backend a resolved spec builds with.
enum Backend {
    Vendors(Vendors),
    Model(Arc<dyn crate::llm::ChatModel>),
}

/// A profile declaration: a new profile, or a patch of a resolved builtin.
#[derive(Debug, Clone, PartialEq)]
pub enum ProfileSpec {
    New(Profile),
    Patch(String, ProfilePatch),
}

/// Resolved agent settings: manifest values over mason's defaults, with explicit
/// CLI and environment overrides applied by the caller.
#[derive(Debug, Clone)]
pub struct AgentSpec {
    pub execution_mode: ExecutionMode,
    pub allow_network: bool,
    pub allow_mcp: bool,
    pub limits: Limits,
    pub shell_timeouts: (u64, u64),
    pub model_timeouts: ModelTimeouts,
    pub spill_quota_mib: u64,
    /// `None` is the process's working directory.
    pub cwd: Option<PathBuf>,
    pub prompts: Option<PathBuf>,
    pub skills_dir: Option<PathBuf>,
    pub memory_dir: Option<PathBuf>,
    /// An explicit spill directory instead of the mode's default location.
    pub spill_dir: Option<PathBuf>,
    pub mcp: BTreeMap<String, McpServerConfigAlias>,
    /// An external server map, read when the tools are discovered so a `--mcp`
    /// override replaces it without the default ever being read.
    pub mcp_file: Option<PathBuf>,
    pub instructions: Option<String>,
    pub profiles: Vec<ProfileSpec>,
    pub select_profile: Vec<SelectRule>,
}

/// An MCP server configuration, or the manifest's strict entry before conversion
/// when mason is built without the `mcp` feature.
#[cfg(feature = "mcp")]
pub type McpServerConfigAlias = crate::mcp::McpServerConfig;
#[cfg(not(feature = "mcp"))]
pub type McpServerConfigAlias = McpEntry;

impl Default for AgentSpec {
    /// Mason's own defaults: trusted execution, no network grant, no MCP.
    fn default() -> Self {
        Self {
            execution_mode: ExecutionMode::default(),
            allow_network: false,
            allow_mcp: false,
            limits: Limits {
                max_steps: DEFAULT_MAX_STEPS,
                timeout: Duration::from_secs(DEFAULT_RUN_TIMEOUT_SECONDS),
                max_total_tool_calls: DEFAULT_MAX_TOTAL_TOOL_CALLS,
                max_total_tokens: DEFAULT_MAX_TOTAL_TOKENS,
                max_response_bytes: DEFAULT_MAX_RESPONSE_BYTES,
                ..Limits::default()
            },
            shell_timeouts: (DEFAULT_SHELL_TIMEOUT_MS, DEFAULT_SHELL_MAX_TIMEOUT_MS),
            model_timeouts: ModelTimeouts {
                request: Duration::from_millis(DEFAULT_MODEL_REQUEST_TIMEOUT_MS),
                turn: Duration::from_millis(DEFAULT_MODEL_TURN_TIMEOUT_MS),
                idle: Duration::from_millis(DEFAULT_MODEL_IDLE_TIMEOUT_MS),
            },
            spill_quota_mib: DEFAULT_SPILL_QUOTA_MIB,
            cwd: None,
            prompts: None,
            skills_dir: None,
            memory_dir: None,
            spill_dir: None,
            mcp: BTreeMap::new(),
            mcp_file: None,
            instructions: None,
            profiles: Vec::new(),
            select_profile: Vec::new(),
        }
    }
}

impl Manifest {
    /// Reads, parses and statically validates a manifest.
    ///
    /// Tool references and the availability of builtin profiles are validated by
    /// [`AgentBuilder::build`], after MCP discovery and any native tools an
    /// application adds on top.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|err| Error::Config(format!("cannot read {}: {err}", path.display())))?;
        let base_dir = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        Self::parse(&text, base_dir)
            .map_err(|err| Error::Config(format!("{}: {err}", path.display())))
    }

    /// Parses and validates manifest text; relative paths resolve against `base_dir`.
    pub fn parse(text: &str, base_dir: PathBuf) -> Result<Self> {
        let mut manifest: Manifest = serde_saphyr::from_str(text)
            .map_err(|err| Error::Config(format!("the manifest is not valid: {err}")))?;
        manifest.base_dir = base_dir;
        manifest.validate()?;
        Ok(manifest)
    }

    fn validate(&self) -> Result<()> {
        if self.version != SCHEMA_VERSION {
            return Err(Error::Config(format!(
                "unsupported manifest version {}; this mason understands version {SCHEMA_VERSION}",
                self.version
            )));
        }
        if let Some(instructions) = &self.instructions
            && instructions.trim().is_empty()
        {
            return Err(Error::Config("`instructions` must not be blank".into()));
        }
        if self.mcp.is_some() && self.mcp_file.is_some() {
            return Err(Error::Config(
                "`mcp` and `mcp_file` are mutually exclusive".into(),
            ));
        }
        let mut names: Vec<&str> = Vec::new();
        for (name, patch) in self
            .profiles
            .as_ref()
            .map(|p| p.iter())
            .into_iter()
            .flatten()
        {
            if name.trim().is_empty() {
                return Err(Error::Config("a profile name must not be blank".into()));
            }
            if names.contains(&name.as_str()) {
                return Err(Error::Config(format!(
                    "duplicate profile name `{name}`; profile names are unique registry keys"
                )));
            }
            names.push(name);
            // A builtin patch inherits its description; a new profile must supply one.
            if !is_builtin_profile(name)
                && patch
                    .description
                    .as_ref()
                    .is_none_or(|d| d.trim().is_empty())
            {
                return Err(Error::Config(format!(
                    "profile `{name}` must have a nonblank description"
                )));
            }
            if let Some(steps) = patch.max_steps
                && steps == 0
            {
                return Err(Error::Config(format!(
                    "profile `{name}` must allow at least one step"
                )));
            }
        }
        // Validates inline entries only; an `mcp_file` is read when it is used, so
        // a `--mcp` override never trips over the default file it replaces.
        for (name, entry) in self.mcp.as_ref().map(|m| m.iter()).into_iter().flatten() {
            entry.validate(name)?;
        }
        for rule in self.select_profile.as_deref().unwrap_or_default() {
            if rule.profile.trim().is_empty() {
                return Err(Error::Config(
                    "a select_profile rule must name a profile".into(),
                ));
            }
            if rule.when_metadata.is_empty() {
                return Err(Error::Config(format!(
                    "select_profile rule `{}` has no conditions",
                    rule.profile
                )));
            }
            for (path, matcher) in &rule.when_metadata {
                if path.trim().is_empty() {
                    return Err(Error::Config(format!(
                        "select_profile rule `{}` has an empty metadata path",
                        rule.profile
                    )));
                }
                if let MetadataMatch::Present(false) = matcher {
                    return Err(Error::Config(format!(
                        "select_profile rule `{}`: `present: false` is not supported; \
                         omit the condition instead",
                        rule.profile
                    )));
                }
            }
        }
        // A rule may only target a profile this manifest declares or a builtin;
        // an application that registers profiles in code installs its own selector.
        let declared: Vec<&str> = self
            .profiles
            .as_ref()
            .map(|profiles| profiles.keys().map(String::as_str).collect())
            .unwrap_or_default();
        for rule in self.select_profile.as_deref().unwrap_or_default() {
            if !is_builtin_profile(&rule.profile) && !declared.contains(&rule.profile.as_str()) {
                return Err(Error::Config(format!(
                    "select_profile targets the undeclared profile `{}`",
                    rule.profile
                )));
            }
            if targets_subagent(&rule.profile, self.profiles.as_ref()) {
                return Err(Error::Config(format!(
                    "select_profile targets `{}`, a subagent profile; a rule picks the profile \
                     of a top-level session, so it must name one that can run as root",
                    rule.profile
                )));
            }
        }
        Ok(())
    }

    /// Resolves the manifest into settings, with mason's defaults underneath.
    pub fn into_spec(&self) -> Result<AgentSpec> {
        let defaults = AgentSpec::default();
        let mut limits = defaults.limits;
        let mut shell_timeouts = defaults.shell_timeouts;
        let mut model_timeouts = defaults.model_timeouts;
        let mut spill_quota_mib = defaults.spill_quota_mib;
        if let Some(patch) = &self.limits {
            apply_limits_patch(&mut limits, patch)?;
            if let Some(timeout) = patch.shell_timeout {
                shell_timeouts.0 = timeout.as_millis() as u64;
            }
            if let Some(timeout) = patch.shell_max_timeout {
                shell_timeouts.1 = timeout.as_millis() as u64;
            }
            if let Some(timeout) = patch.model_request_timeout {
                model_timeouts.request = timeout;
            }
            if let Some(timeout) = patch.model_turn_timeout {
                model_timeouts.turn = timeout;
            }
            if let Some(timeout) = patch.model_idle_timeout {
                model_timeouts.idle = timeout;
            }
            if let Some(mib) = patch.spill_quota_mib {
                spill_quota_mib = mib;
            }
        }
        let selection = self.mcp_selection()?;
        let mut profiles = Vec::new();
        for (name, patch) in self
            .profiles
            .as_ref()
            .map(|profiles| profiles.iter())
            .into_iter()
            .flatten()
        {
            profiles.push(if is_builtin_profile(name) {
                ProfileSpec::Patch(name.clone(), patch.clone())
            } else {
                ProfileSpec::New(patch.into_profile(name)?)
            });
        }
        Ok(AgentSpec {
            execution_mode: self.execution_mode,
            allow_network: self.allow_network,
            allow_mcp: self.allow_mcp,
            limits,
            shell_timeouts,
            model_timeouts,
            spill_quota_mib,
            cwd: self.cwd.as_ref().map(|path| self.resolve(path)),
            prompts: self.prompts.as_ref().map(|path| self.resolve(path)),
            skills_dir: self.skills_dir.as_ref().map(|path| self.resolve(path)),
            memory_dir: self.memory_dir.as_ref().map(|path| self.resolve(path)),
            spill_dir: self.spill.as_ref().map(|path| self.resolve(path)),
            mcp: selection.servers,
            mcp_file: selection.file,
            instructions: self.instructions.clone(),
            profiles,
            select_profile: self.select_profile.clone().unwrap_or_default(),
        })
    }

    /// Resolves a manifest path against the manifest's own directory.
    fn resolve(&self, path: &Path) -> PathBuf {
        resolve_against(&self.base_dir, path)
    }

    /// The manifest's MCP selection. An `mcp_file` is *not* read here: a `--mcp`
    /// override replaces it, and reading the file it replaces would fail startup
    /// on a default the operator has just overridden.
    fn mcp_selection(&self) -> Result<McpSelection> {
        Ok(match (&self.mcp, &self.mcp_file) {
            (Some(entries), _) => McpSelection {
                servers: entries
                    .iter()
                    .map(|(name, entry)| {
                        Ok((name.clone(), entry.to_config(name, Some(&self.base_dir))?))
                    })
                    .collect::<Result<BTreeMap<_, _>>>()?,
                file: None,
            },
            (None, Some(path)) => McpSelection {
                servers: BTreeMap::new(),
                file: Some(self.resolve(path)),
            },
            (None, None) => McpSelection::default(),
        })
    }

    /// Resolves the manifest into an [`AgentBuilder`]: MCP tools discovered, paths
    /// resolved, profiles and instructions applied, models from the environment.
    pub async fn into_builder(self) -> Result<AgentBuilder> {
        self.into_spec()?.into_builder().await
    }

    /// The same, with a custom model backend instead of the environment's vendors.
    pub async fn into_builder_with_model(
        self,
        model: Arc<dyn crate::llm::ChatModel>,
    ) -> Result<AgentBuilder> {
        self.into_spec()?.into_builder_with_model(model).await
    }
}

/// Reads an opencode-format MCP server map from a JSON file. A relative server
/// `cwd` resolves against `base_dir` (the file's own directory when it is known),
/// and is left alone otherwise.
pub fn load_mcp_file(
    path: &Path,
    base_dir: Option<&Path>,
) -> Result<BTreeMap<String, McpServerConfigAlias>> {
    let text = std::fs::read_to_string(path)
        .map_err(|err| Error::Config(format!("cannot read {}: {err}", path.display())))?;
    let entries: BTreeMap<String, McpEntry> = serde_json::from_str(&text)
        .map_err(|err| Error::Config(format!("cannot parse {}: {err}", path.display())))?;
    entries
        .iter()
        .map(|(name, entry)| {
            entry
                .validate(name)
                .map_err(|err| Error::Config(format!("{}: {err}", path.display())))?;
            Ok((name.clone(), entry.to_config(name, base_dir)?))
        })
        .collect()
}

/// Resolves a path against a base directory, leaving an absolute path alone.
fn resolve_against(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

/// What a manifest selects: inline servers, or a file read when they are needed.
#[derive(Debug, Clone, Default)]
struct McpSelection {
    servers: BTreeMap<String, McpServerConfigAlias>,
    file: Option<PathBuf>,
}

impl McpEntry {
    /// Checks the type and the keys that belong to it, naming the offending field.
    pub fn validate(&self, name: &str) -> Result<()> {
        let field = |field: &str| format!("mcp server `{name}`: `{field}`");
        match self.kind.as_str() {
            "local" => {
                for invalid in ["url", "headers"] {
                    if self.has(invalid) {
                        return Err(Error::Config(format!(
                            "{} is only valid for a remote server",
                            field(invalid)
                        )));
                    }
                }
                let command = self
                    .command
                    .as_ref()
                    .ok_or_else(|| Error::Config(format!("{} is required", field("command"))))?;
                if command.is_empty() {
                    return Err(Error::Config(format!(
                        "{} must not be empty",
                        field("command")
                    )));
                }
                if command.iter().any(|part| part.trim().is_empty()) {
                    return Err(Error::Config(format!(
                        "{} has an empty argument",
                        field("command")
                    )));
                }
            }
            "remote" => {
                for invalid in ["command", "cwd", "environment"] {
                    if self.has(invalid) {
                        return Err(Error::Config(format!(
                            "{} is only valid for a local server",
                            field(invalid)
                        )));
                    }
                }
                if self.url.as_ref().is_none_or(|url| url.trim().is_empty()) {
                    return Err(Error::Config(format!(
                        "{} is required and must not be blank",
                        field("url")
                    )));
                }
            }
            other => {
                return Err(Error::Config(format!(
                    "mcp server `{name}`: unknown type `{other}`; use `local` or `remote`"
                )));
            }
        }
        if let Some(timeout) = self.timeout
            && timeout == 0
        {
            return Err(Error::Config(format!(
                "{} must be positive",
                field("timeout")
            )));
        }
        Ok(())
    }

    fn has(&self, field: &str) -> bool {
        match field {
            "command" => self.command.is_some(),
            "url" => self.url.is_some(),
            "cwd" => self.cwd.is_some(),
            "environment" => self.environment.is_some(),
            "headers" => self.headers.is_some(),
            _ => false,
        }
    }

    #[cfg(feature = "mcp")]
    fn to_config(
        &self,
        name: &str,
        base_dir: Option<&Path>,
    ) -> Result<crate::mcp::McpServerConfig> {
        self.validate(name)?;
        let expand = |text: &String| expand_env(text, name);
        let expand_map = |map: &BTreeMap<String, String>| {
            map.iter()
                .map(|(key, value)| Ok((key.clone(), expand(value)?)))
                .collect::<Result<_>>()
        };
        Ok(match self.kind.as_str() {
            "local" => crate::mcp::McpServerConfig::Local {
                command: self
                    .command
                    .iter()
                    .flatten()
                    .map(expand)
                    .collect::<Result<_>>()?,
                // An explicit relative cwd means the declaring file's directory,
                // not whatever directory mason happened to be launched from.
                cwd: self.cwd.as_ref().map(|cwd| {
                    base_dir
                        .map(|base| resolve_against(base, Path::new(cwd)))
                        .unwrap_or_else(|| PathBuf::from(cwd))
                        .to_string_lossy()
                        .into_owned()
                }),
                environment: self.environment.as_ref().map(expand_map).transpose()?,
                enabled: self.enabled,
                timeout: self.timeout,
            },
            _ => crate::mcp::McpServerConfig::Remote {
                url: expand(self.url.as_ref().expect("validated"))?,
                headers: self.headers.as_ref().map(expand_map).transpose()?,
                enabled: self.enabled,
                timeout: self.timeout,
            },
        })
    }

    #[cfg(not(feature = "mcp"))]
    fn to_config(&self, name: &str, _base_dir: Option<&Path>) -> Result<McpEntry> {
        self.validate(name)?;
        Ok(self.clone())
    }
}

/// Expands opencode-style `{env:NAME}` references from the process environment,
/// so MCP credentials can stay out of config files.
#[cfg(feature = "mcp")]
fn expand_env(text: &str, server: &str) -> Result<String> {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("{env:") {
        out.push_str(&rest[..start]);
        let after = &rest[start + "{env:".len()..];
        let end = after.find('}').ok_or_else(|| {
            Error::Config(format!(
                "mcp server `{server}` has an unclosed {{env:...}} reference"
            ))
        })?;
        let name = &after[..end];
        let value = std::env::var(name).map_err(|_| {
            Error::Config(format!(
                "mcp server `{server}` references {{env:{name}}}, which is not set"
            ))
        })?;
        out.push_str(&value);
        rest = &after[end + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

impl SelectRule {
    /// Whether every condition holds for this session metadata.
    pub fn matches(&self, metadata: &serde_json::Value) -> bool {
        self.when_metadata
            .iter()
            .all(|(path, matcher)| matcher.matches(lookup(metadata, path)))
    }
}

/// Follows a dotted path through metadata; `None` when any segment is missing.
fn lookup<'a>(metadata: &'a serde_json::Value, path: &str) -> Option<&'a serde_json::Value> {
    path.split('.')
        .try_fold(metadata, |value, segment| value.get(segment))
}

impl MetadataMatch {
    fn matches(&self, value: Option<&serde_json::Value>) -> bool {
        match self {
            // A missing path and an explicit null both fail to match.
            MetadataMatch::Present(true) => value.is_some_and(|value| !value.is_null()),
            MetadataMatch::Present(false) => value.is_none_or(|value| value.is_null()),
            MetadataMatch::Equals(expected) => value == Some(expected),
        }
    }
}

/// Deserializes a duration string. An omitted field is handled by `default`; a null,
/// a bare number or an unknown unit is rejected with a duration-specific message.
fn human_duration<'de, D>(deserializer: D) -> std::result::Result<Option<Duration>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct DurationText;

    impl serde::de::Visitor<'_> for DurationText {
        type Value = Option<Duration>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a duration string like `15m` or `90s`")
        }

        fn visit_str<E: serde::de::Error>(
            self,
            value: &str,
        ) -> std::result::Result<Option<Duration>, E> {
            parse_duration(value).map(Some).map_err(|err| {
                // The visitor only serves `limits.timeout`, so name it here.
                serde::de::Error::custom(format!("`limits.timeout`: {err}"))
            })
        }
    }

    deserializer.deserialize_str(DurationText)
}

/// Applies the supplied budget fields, leaving the rest at their current value.
fn apply_limits_patch(limits: &mut Limits, patch: &LimitsPatch) -> Result<()> {
    if let Some(max_steps) = patch.max_steps {
        limits.max_steps = max_steps;
    }
    if let Some(concurrent) = patch.max_concurrent_tools {
        limits.max_concurrent_tools = concurrent;
    }
    if let Some(per_step) = patch.max_tool_calls_per_step {
        limits.max_tool_calls_per_step = per_step;
    }
    if let Some(task_calls) = patch.max_task_calls_per_step {
        limits.max_task_calls_per_step = task_calls;
    }
    if let Some(total_calls) = patch.max_total_tool_calls {
        limits.max_total_tool_calls = total_calls;
    }
    if let Some(total_tokens) = patch.max_total_tokens {
        limits.max_total_tokens = total_tokens;
    }
    if let Some(response_bytes) = patch.max_response_bytes {
        limits.max_response_bytes = response_bytes;
    }
    if let Some(timeout) = patch.timeout {
        limits.timeout = timeout;
    }
    Ok(())
}

/// Parses `<number><unit>` pairs: `ms`, `s`, `m`, `h` (`90s`, `15m`, `1h30m`).
pub fn parse_duration(raw: &str) -> Result<Duration> {
    let text = raw.trim();
    if text.is_empty() {
        return Err(Error::Config("the duration is empty".into()));
    }
    let mut total = Duration::ZERO;
    let mut rest = text;
    let mut seen = false;
    while !rest.is_empty() {
        let digits = rest
            .find(|character: char| !character.is_ascii_digit())
            .unwrap_or(rest.len());
        if digits == 0 {
            return Err(Error::Config(format!(
                "`{raw}` does not start with a number"
            )));
        }
        let value: u64 = rest[..digits]
            .parse()
            .map_err(|_| Error::Config(format!("`{raw}` has a number that is too large")))?;
        rest = &rest[digits..];
        // A unit may be followed by whitespace before the next pair (`1h 30m`).
        let unit = match rest.find(|character: char| character.is_ascii_digit()) {
            Some(index) => {
                let (unit, remainder) = rest.split_at(index);
                rest = remainder;
                unit
            }
            None => {
                let unit = rest;
                rest = "";
                unit
            }
        }
        .trim();
        // Checked throughout: a huge value must be a configuration error, not a
        // debug-build panic or a release-build wrap.
        let millis = match unit {
            "ms" => Some(value),
            "s" => value.checked_mul(1_000),
            "m" => value.checked_mul(60_000),
            "h" => value.checked_mul(3_600_000),
            "" => {
                return Err(Error::Config(format!(
                    "`{raw}` has no unit; a duration needs one, like `15m` or `90s`"
                )));
            }
            other => {
                return Err(Error::Config(format!(
                    "`{raw}` has the unknown unit `{other}`; use ms, s, m or h"
                )));
            }
        };
        let Some(millis) = millis else {
            return Err(Error::Config(format!("`{raw}` is too large")));
        };
        total = total
            .checked_add(Duration::from_millis(millis))
            .ok_or_else(|| Error::Config(format!("`{raw}` is too large")))?;
        seen = true;
        rest = rest.trim_start();
    }
    if !seen || total.is_zero() {
        return Err(Error::Config(format!("`{raw}` is not a positive duration")));
    }
    Ok(total)
}

/// Whether a rule target can only run as a subagent, and so could never own a
/// session: the builtin subagents, or a declared profile that says it is one.
fn targets_subagent(name: &str, declared: Option<&BTreeMap<String, ProfilePatch>>) -> bool {
    match name {
        "general" | "explore" => true,
        "build" => false,
        _ => declared
            .and_then(|profiles| profiles.get(name))
            .and_then(|patch| patch.subagent)
            .unwrap_or(false),
    }
}

/// Whether a name is one of mason's builtin profiles.
pub fn is_builtin_profile(name: &str) -> bool {
    matches!(name, "build" | "general" | "explore")
}

impl AgentSpec {
    /// Builds the agent these settings describe, with vendors and the optional
    /// evaluator's vendors (`AI_EVALUATOR_VENDORS`) from the environment.
    pub async fn into_builder(self) -> Result<AgentBuilder> {
        let timeouts = self.model_timeouts;
        let mut builder = self.into_builder_with_vendors(Vendors::from_env()?).await?;
        if let Some(vendors) = Vendors::evaluator_from_env()? {
            let model = crate::llm::VendorModel::new(vendors).with_timeouts(timeouts)?;
            builder = builder.evaluator(Arc::new(crate::evaluate::LlmEvaluator::new(Arc::new(
                model,
            ))));
        }
        Ok(builder)
    }

    /// Builds the agent these settings describe with these vendors.
    pub async fn into_builder_with_vendors(self, vendors: Vendors) -> Result<AgentBuilder> {
        self.assemble(Backend::Vendors(vendors)).await
    }

    /// Builds the agent these settings describe with a custom model backend, so an
    /// application can compose a manifest with its own model the same way it can
    /// with its own tools.
    pub async fn into_builder_with_model(
        self,
        model: Arc<dyn crate::llm::ChatModel>,
    ) -> Result<AgentBuilder> {
        self.assemble(Backend::Model(model)).await
    }

    /// Builtin tools wrapped for the execution mode, MCP tools discovered, and the
    /// manifest's instructions and profiles applied.
    async fn assemble(self, backend: Backend) -> Result<AgentBuilder> {
        validate_limits(&self.limits, self.spill_quota_mib)?;
        let cwd = self.effective_cwd()?;
        if self.execution_mode == ExecutionMode::Sandbox {
            crate::tool::policy::preflight(&cwd)
                .await
                .map_err(|err| Error::Config(err.to_string()))?;
        }
        let mcp_tools = self.discover_mcp().await?;
        let spill = Arc::new(
            crate::OutputSpill::new(self.spill_path(&cwd)?)
                .with_storage_limit(self.spill_quota_mib.saturating_mul(1024 * 1024)),
        );
        spill
            .prepare()
            .await
            .map_err(|err| Error::Config(err.to_string()))?;
        let sandboxed = self.execution_mode == ExecutionMode::Sandbox;
        let builtin = crate::tool::builtin_tools().into_iter().map(|tool| {
            let tool: Arc<dyn crate::tool::Tool> = Arc::from(tool);
            if sandboxed
                && matches!(
                    tool.spec().function.name.as_str(),
                    "read" | "write" | "edit" | "grep" | "glob"
                )
            {
                Arc::new(crate::tool::policy::IsolatedFileTool(tool)) as Arc<dyn crate::tool::Tool>
            } else {
                tool
            }
        });
        let builder = crate::Agent::builder()
            .tools_shared(builtin)
            .limits(self.limits.clone())
            .spill(spill)
            .shell_timeouts(self.shell_timeouts.0, self.shell_timeouts.1)
            .model_timeouts(self.model_timeouts)
            .execution_policy(
                self.execution_mode,
                self.allow_network,
                Arc::new(crate::tool::policy::CliPolicy {
                    allow_mcp: self.allow_mcp,
                }),
            )
            .cwd(cwd)
            .prompts(match &self.prompts {
                Some(path) => PromptSource::File(path.clone()),
                None => PromptSource::Embedded,
            });
        let mut builder = match backend {
            Backend::Vendors(vendors) => builder.vendors(vendors),
            Backend::Model(model) => builder.model(model),
        };
        builder = self.with_mcp_tools(builder, mcp_tools)?;
        if let Some(instructions) = &self.instructions {
            builder = builder.instructions(instructions.clone());
        }
        if let Some(dir) = &self.skills_dir {
            builder = builder.skills_dir(dir.clone());
        }
        if let Some(dir) = &self.memory_dir {
            builder = builder.memory_dir(dir.clone());
        }
        for spec in self.profiles {
            builder = match spec {
                ProfileSpec::New(profile) => builder.profiles([profile]),
                ProfileSpec::Patch(name, patch) => builder.patch_profile(name, patch),
            };
        }
        if !self.select_profile.is_empty() {
            builder = builder.profile_selector(selector_from_rules(self.select_profile.clone()));
        }
        Ok(builder)
    }

    /// The spill directory: an explicit one in trusted mode, otherwise the
    /// mode's default location.
    fn spill_path(&self, cwd: &Path) -> Result<PathBuf> {
        match (&self.spill_dir, self.execution_mode) {
            (Some(dir), ExecutionMode::Sandbox) => Err(Error::Config(format!(
                "`spill: {}` is not available with execution_mode sandbox, which confines \
                 spilled output to the workspace; remove it or use trusted mode",
                dir.display()
            ))),
            (Some(dir), ExecutionMode::Trusted) => Ok(dir.clone()),
            (None, ExecutionMode::Sandbox) => {
                crate::tool::policy::workspace_path(cwd, ".mason/output")
                    .map_err(|err| Error::Config(err.to_string()))
            }
            (None, ExecutionMode::Trusted) => Ok(cwd.join(".mason/output")),
        }
    }

    /// The working directory these settings resolve to.
    pub fn effective_cwd(&self) -> Result<PathBuf> {
        match &self.cwd {
            Some(cwd) => Ok(cwd.clone()),
            None => std::env::current_dir().map_err(|err| {
                Error::Config(format!("cannot determine the working directory: {err}"))
            }),
        }
    }

    /// Whether any MCP server is configured, inline or by file.
    pub fn has_mcp(&self) -> bool {
        self.mcp_file.is_some() || !self.mcp.is_empty()
    }

    /// The configured servers, with a local server's missing `cwd` filled in from
    /// the effective working directory.
    #[cfg(feature = "mcp")]
    pub fn mcp_configs(&self) -> Result<BTreeMap<String, crate::mcp::McpServerConfig>> {
        let cwd = self.effective_cwd()?;
        let mut configs = match &self.mcp_file {
            // A file's own relative paths resolve against that file.
            Some(path) => load_mcp_file(path, path.parent())?,
            None => self.mcp.clone(),
        };
        for config in configs.values_mut() {
            if let crate::mcp::McpServerConfig::Local { cwd: directory, .. } = config
                && directory.is_none()
            {
                *directory = Some(cwd.to_string_lossy().into_owned());
            }
        }
        Ok(configs)
    }

    /// The discovered MCP tools, listed again by every top-level run so a server's
    /// changed tools need no restart.
    fn with_mcp_tools(
        &self,
        builder: AgentBuilder,
        tools: Vec<Arc<dyn crate::tool::Tool>>,
    ) -> Result<AgentBuilder> {
        #[cfg(feature = "mcp")]
        if self.has_mcp() && self.allow_mcp {
            let catalog = Arc::new(crate::mcp::McpRefresh::new(
                self.mcp_configs()?,
                self.execution_mode,
                self.allow_network,
                &tools,
            ));
            let refresh: crate::agent::ToolRefresh = Arc::new(move |profile, scope, timeout| {
                let catalog = catalog.clone();
                Box::pin(async move { Ok(catalog.refresh(&profile, &scope, timeout).await) })
            });
            return Ok(builder.refreshing_tools(tools, refresh));
        }
        Ok(builder.tools_shared(tools))
    }

    async fn discover_mcp(&self) -> Result<Vec<Arc<dyn crate::tool::Tool>>> {
        #[cfg(not(feature = "mcp"))]
        {
            if self.has_mcp() {
                return Err(Error::Config(
                    "mason was built without the mcp feature; remove the `mcp` section".into(),
                ));
            }
            Ok(Vec::new())
        }
        #[cfg(feature = "mcp")]
        {
            if !self.has_mcp() {
                return Ok(Vec::new());
            }
            if !self.allow_mcp {
                return Err(Error::Config(
                    "configured MCP servers require an explicit grant \
                     (`allow_mcp: true`, `--allow-mcp` or MASON_ALLOW_MCP=true)"
                        .into(),
                ));
            }
            let configs = self.mcp_configs()?;
            // Discovery connections are closed by `connect`; the tool adapters open
            // their own connection per run, so nothing is held open here.
            let servers =
                crate::mcp::McpServers::connect(&configs, self.execution_mode, self.allow_network)
                    .await;
            if !servers.failures().is_empty() {
                return Err(Error::Config(format!(
                    "configured MCP servers failed ({}); run `mason mcp --allow-mcp` for diagnostics",
                    servers.failures().join("; ")
                )));
            }
            Ok(servers.tools())
        }
    }
}

/// The `ProfileSelector` a manifest's rules describe.
fn selector_from_rules(
    rules: Vec<SelectRule>,
) -> impl Fn(&crate::store::Session) -> Option<String> + Send + Sync + 'static {
    move |session: &crate::store::Session| {
        rules
            .iter()
            .find(|rule| rule.matches(&session.metadata))
            .map(|rule| rule.profile.clone())
    }
}

fn validate_limits(limits: &Limits, spill_quota_mib: u64) -> Result<()> {
    const MAX_TIMEOUT: Duration = Duration::from_secs(7 * 86400);
    if limits.timeout.is_zero() || limits.timeout > MAX_TIMEOUT {
        return Err(Error::Config(
            "timeout must be positive and at most 7 days".into(),
        ));
    }
    for (name, value) in [
        ("max_steps", limits.max_steps),
        ("max_total_tool_calls", limits.max_total_tool_calls),
        ("max_total_tokens", limits.max_total_tokens),
        ("max_response_bytes", limits.max_response_bytes),
        ("max_concurrent_tools", limits.max_concurrent_tools),
        ("max_tool_calls_per_step", limits.max_tool_calls_per_step),
        ("max_task_calls_per_step", limits.max_task_calls_per_step),
    ] {
        if value == 0 {
            return Err(Error::Config(format!("{name} must be positive")));
        }
    }
    if spill_quota_mib < 10 {
        return Err(Error::Config(format!(
            "spill_quota_mib must be at least 10, not {spill_quota_mib}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limit_errors_name_the_offending_field() {
        let ok = AgentSpec::default().limits;
        let check =
            |limits: Limits, mib: u64| validate_limits(&limits, mib).unwrap_err().to_string();
        assert!(
            check(
                Limits {
                    max_steps: 0,
                    ..ok.clone()
                },
                256
            )
            .contains("max_steps")
        );
        assert!(
            check(
                Limits {
                    max_concurrent_tools: 0,
                    ..ok.clone()
                },
                256
            )
            .contains("max_concurrent_tools")
        );
        assert!(
            check(
                Limits {
                    timeout: Duration::from_secs(8 * 86400),
                    ..ok.clone()
                },
                256
            )
            .contains("at most 7 days")
        );
        assert!(check(ok.clone(), 5).contains("spill_quota_mib must be at least 10, not 5"));
        assert!(validate_limits(&ok, 256).is_ok());
    }

    /// A manifest exercising every section, with relative paths.
    const FULL: &str = r#"
version: 1
instructions: |
  You are an agent working on the Foo service.
profiles:
  unattended:
    description: Unattended investigation.
    prompt: Investigate read-only.
    tools: [read, grep, acme_sql]
    max_steps: 30
  explore:
    tools: [read, grep, glob, acme_fetch]
  build:
    allowed_subagents: [general, explore]
mcp:
  acme:
    type: local
    command: [node, tools/mcp-server.js]
    environment:
      ACME_CONFIG: etc/config.json
execution_mode: trusted
allow_network: true
allow_mcp: true
skills_dir: skills
memory_dir: memory
limits:
  max_steps: 40
  timeout: 20m
cwd: work
"#;

    fn parse(text: &str) -> Result<Manifest> {
        Manifest::parse(text, PathBuf::from("/app/conf"))
    }

    fn spec(text: &str) -> Result<AgentSpec> {
        parse(text)?.into_spec()
    }

    /// The declaration for one profile name, whatever its kind.
    fn find<'a>(spec: &'a AgentSpec, name: &str) -> Option<&'a ProfileSpec> {
        spec.profiles.iter().find(|spec| match spec {
            ProfileSpec::New(profile) => profile.name == name,
            ProfileSpec::Patch(patch_name, _) => patch_name == name,
        })
    }

    #[test]
    fn a_full_manifest_resolves_against_its_own_directory() {
        let spec = spec(FULL).unwrap();
        assert_eq!(spec.execution_mode, ExecutionMode::Trusted);
        assert!(spec.allow_network && spec.allow_mcp);
        assert_eq!(
            spec.instructions.as_deref(),
            Some("You are an agent working on the Foo service.\n")
        );
        assert_eq!(spec.cwd.as_deref(), Some(Path::new("/app/conf/work")));
        assert_eq!(
            spec.skills_dir.as_deref(),
            Some(Path::new("/app/conf/skills"))
        );
        assert_eq!(
            spec.memory_dir.as_deref(),
            Some(Path::new("/app/conf/memory"))
        );
        assert_eq!(spec.limits.max_steps, 40, "a supplied budget wins");
        assert_eq!(spec.limits.timeout, Duration::from_secs(20 * 60));
        assert_eq!(
            spec.limits.max_total_tool_calls, DEFAULT_MAX_TOTAL_TOOL_CALLS,
            "an omitted budget keeps its default"
        );
        assert_eq!(spec.profiles.len(), 3);
        assert!(matches!(
            find(&spec, "unattended"),
            Some(ProfileSpec::New(_))
        ));
        assert!(matches!(
            find(&spec, "explore"),
            Some(ProfileSpec::Patch(_, _))
        ));
        assert!(matches!(
            find(&spec, "build"),
            Some(ProfileSpec::Patch(_, _))
        ));
        #[cfg(not(feature = "mcp"))]
        {
            let entry = spec.mcp.get("acme").expect("the server is configured");
            assert_eq!(
                entry.command.as_deref(),
                Some(["node".to_owned(), "tools/mcp-server.js".to_owned()].as_slice())
            );
            assert_eq!(entry.cwd, None, "an omitted cwd follows the agent's");
            assert_eq!(
                entry
                    .environment
                    .as_ref()
                    .and_then(|env| env.get("ACME_CONFIG"))
                    .map(String::as_str),
                Some("etc/config.json")
            );
        }
        #[cfg(feature = "mcp")]
        match spec.mcp.get("acme").expect("the server is configured") {
            crate::mcp::McpServerConfig::Local {
                command,
                environment,
                cwd,
                ..
            } => {
                assert_eq!(command.as_slice(), ["node", "tools/mcp-server.js"]);
                assert_eq!(cwd.as_deref(), None, "an omitted cwd follows the agent's");
                assert_eq!(
                    environment
                        .as_ref()
                        .and_then(|env| env.get("ACME_CONFIG"))
                        .map(String::as_str),
                    Some("etc/config.json")
                );
            }
            _ => panic!("a local server"),
        }
    }

    #[cfg(feature = "mcp")]
    #[test]
    fn mcp_env_references_expand_from_the_process_environment() {
        let home = std::env::var("HOME").unwrap();
        let parsed = spec(
            "version: 1\nmcp:\n  gh:\n    type: remote\n    url: https://x/{env:HOME}\n    headers:\n      Authorization: Bearer {env:HOME}!\n  local:\n    type: local\n    command: [run, '--home={env:HOME}']\n    environment:\n      TOKEN: '{env:HOME}'\n",
        )
        .unwrap();
        match parsed.mcp.get("gh").unwrap() {
            crate::mcp::McpServerConfig::Remote { url, headers, .. } => {
                assert_eq!(url, &format!("https://x/{home}"));
                assert_eq!(
                    headers.as_ref().unwrap()["Authorization"],
                    format!("Bearer {home}!")
                );
            }
            _ => panic!("a remote server"),
        }
        match parsed.mcp.get("local").unwrap() {
            crate::mcp::McpServerConfig::Local {
                command,
                environment,
                ..
            } => {
                assert_eq!(command[1], format!("--home={home}"));
                assert_eq!(environment.as_ref().unwrap()["TOKEN"], home);
            }
            _ => panic!("a local server"),
        }
        let err = spec(
            "version: 1\nmcp:\n  gh:\n    type: remote\n    url: https://x\n    headers:\n      A: '{env:MASON_SURELY_UNSET_VARIABLE}'\n",
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("mcp server `gh` references {env:MASON_SURELY_UNSET_VARIABLE}"),
            "{err}"
        );
    }

    #[test]
    fn builtin_names_patch_and_new_names_register() {
        let spec = spec(FULL).unwrap();
        let Some(ProfileSpec::New(profile)) = find(&spec, "unattended") else {
            panic!("unattended registers as a new profile")
        };
        assert_eq!(profile.name, "unattended");
        assert_eq!(profile.description, "Unattended investigation.");
        assert_eq!(profile.prompt.as_deref(), Some("Investigate read-only."));
        assert_eq!(
            profile.tools,
            Some(vec![
                "read".to_owned(),
                "grep".to_owned(),
                "acme_sql".to_owned()
            ])
        );
        assert_eq!(profile.max_steps, 30);
        assert!(!profile.subagent);

        let Some(ProfileSpec::Patch(name, patch)) = find(&spec, "explore") else {
            panic!("explore patches a builtin")
        };
        assert_eq!(name, "explore");
        assert_eq!(patch.description, None, "only the tools were supplied");
        assert_eq!(patch.tools.as_ref().map(Vec::len), Some(4));
    }

    #[test]
    fn a_duplicate_profile_name_is_rejected() {
        let text = "version: 1\nexecution_mode: sandbox\nprofiles:\n  a:\n    description: one\n    max_steps: 1\n  a:\n    description: two\n";
        let error = parse(text).unwrap_err().to_string();
        assert!(error.contains("a"), "{error}");
        assert!(
            error.contains("duplicate") || error.contains("Duplicate"),
            "the parser or the validator rejects it: {error}"
        );
    }

    #[test]
    fn malformed_manifests_are_rejected_naming_the_field() {
        let cases = [
            (
                "version: 2\nexecution_mode: sandbox\n",
                "unsupported manifest version 2",
            ),
            ("execution_mode: sandbox\n", "version"),
            ("version: 1\nexecution_mode: sideways\n", "execution_mode"),
            ("version: 1\nexecution_mode: sandbox\nnope: 1\n", "nope"),
            (
                "version: 1\nexecution_mode: sandbox\ninstructions: '   '\n",
                "instructions",
            ),
            (
                "version: 1\nexecution_mode: sandbox\nprofiles:\n  x:\n    max_steps: 3\n",
                "profile `x` must have a nonblank description",
            ),
            (
                "version: 1\nexecution_mode: sandbox\nprofiles:\n  x:\n    description: d\n    max_steps: 0\n",
                "at least one step",
            ),
            (
                "version: 1\nexecution_mode: sandbox\nlimits:\n  timeout: 900\n",
                "limits.timeout",
            ),
            (
                "version: 1\nexecution_mode: sandbox\nlimits:\n  timeout: 15 parsecs\n",
                "unknown unit",
            ),
            (
                "version: 1\nexecution_mode: sandbox\nmcp:\n  a:\n    type: local\n    command: [x]\nmcp_file: m.json\n",
                "mutually exclusive",
            ),
            (
                "version: 1\nexecution_mode: sandbox\nmcp:\n  a:\n    type: local\n    url: http://x\n    command: [x]\n",
                "only valid for a remote server",
            ),
            (
                "version: 1\nexecution_mode: sandbox\nmcp:\n  a:\n    type: remote\n    url: http://x\n    command: [x]\n",
                "only valid for a local server",
            ),
            (
                "version: 1\nexecution_mode: sandbox\nmcp:\n  a:\n    type: local\n",
                "command` is required",
            ),
            (
                "version: 1\nexecution_mode: sandbox\nmcp:\n  a:\n    type: local\n    command: []\n",
                "must not be empty",
            ),
            (
                "version: 1\nexecution_mode: sandbox\nmcp:\n  a:\n    type: remote\n",
                "url` is required",
            ),
            (
                "version: 1\nexecution_mode: sandbox\nmcp:\n  a:\n    type: carrier-pigeon\n    url: http://x\n",
                "unknown type",
            ),
            (
                "version: 1\nexecution_mode: sandbox\nmcp:\n  a:\n    type: local\n    command: [x]\n    nope: 1\n",
                "nope",
            ),
            (
                "version: 1\nexecution_mode: sandbox\nselect_profile:\n  - profile: ghost\n    when_metadata:\n      tenant_id:\n        present: true\n",
                "undeclared profile `ghost`",
            ),
            (
                "version: 1\nexecution_mode: sandbox\nselect_profile:\n  - profile: build\n    when_metadata: {}\n",
                "has no conditions",
            ),
            (
                "version: 1\nexecution_mode: sandbox\nselect_profile:\n  - profile: build\n    when_metadata:\n      tenant_id:\n        present: false\n",
                "present: false",
            ),
        ];
        for (text, expected) in cases {
            let error = parse(text).unwrap_err().to_string();
            assert!(
                error.contains(expected),
                "{text}\n-> {error}\nwant {expected}"
            );
        }
    }

    #[test]
    fn an_explicit_null_is_not_an_omitted_field() {
        let error = parse("version: 1\nexecution_mode: sandbox\ninstructions: null\n")
            .unwrap_err()
            .to_string();
        assert!(error.contains("instructions"), "{error}");
        assert!(error.contains("null"), "{error}");
    }

    #[test]
    fn durations_parse_unit_pairs() {
        assert_eq!(parse_duration("500ms").unwrap(), Duration::from_millis(500));
        assert_eq!(parse_duration("90s").unwrap(), Duration::from_secs(90));
        assert_eq!(parse_duration("15m").unwrap(), Duration::from_secs(900));
        assert_eq!(parse_duration("2h").unwrap(), Duration::from_secs(7200));
        assert_eq!(
            parse_duration(" 1h30m ").unwrap(),
            Duration::from_secs(5400)
        );
        assert_eq!(parse_duration("1h 30m").unwrap(), Duration::from_secs(5400));
        for bad in ["", "15", "m", "15mm", "0s", "-5s", "1h 30"] {
            assert!(parse_duration(bad).is_err(), "`{bad}` must be rejected");
        }
    }

    #[cfg(feature = "mcp")]
    #[test]
    fn a_local_server_without_a_cwd_follows_the_effective_cwd() {
        let resolved = spec(
            "version: 1\nexecution_mode: trusted\nallow_mcp: true\ncwd: work\nmcp:\n  a:\n    type: local\n    command: [node, server.js]\n",
        )
        .unwrap();
        #[cfg(feature = "mcp")]
        {
            let configs = resolved.mcp_configs().unwrap();
            match configs.get("a").unwrap() {
                crate::mcp::McpServerConfig::Local { cwd, .. } => {
                    assert_eq!(cwd.as_deref(), Some("/app/conf/work"));
                }
                _ => panic!("a local server"),
            }
        }
        // An explicit cwd in the entry wins.
        let resolved = spec(
            "version: 1\nexecution_mode: trusted\nallow_mcp: true\ncwd: work\nmcp:\n  a:\n    type: local\n    command: [node]\n    cwd: /elsewhere\n",
        )
        .unwrap();
        #[cfg(feature = "mcp")]
        match resolved.mcp_configs().unwrap().get("a").unwrap() {
            crate::mcp::McpServerConfig::Local { cwd, .. } => {
                assert_eq!(cwd.as_deref(), Some("/elsewhere"));
            }
            _ => panic!("a local server"),
        }
    }

    /// The error a discovery attempt produces, panicking when it succeeds.
    async fn discover_error(spec: &AgentSpec) -> String {
        match spec.discover_mcp().await.map(|_| ()) {
            Err(error) => error.to_string(),
            Ok(()) => panic!("configured MCP servers must not be silently ignored"),
        }
    }

    #[tokio::test]
    async fn configured_mcp_servers_need_an_explicit_grant() {
        let ungranted = spec(
            "version: 1\nexecution_mode: trusted\nmcp:\n  a:\n    type: local\n    command: [node, server.js]\n",
        )
        .unwrap();
        let error = discover_error(&ungranted).await;
        #[cfg(feature = "mcp")]
        assert!(error.contains("explicit grant"), "{error}");
        #[cfg(not(feature = "mcp"))]
        assert!(error.contains("without the mcp feature"), "{error}");

        // A granted server whose command cannot run is a startup error, not a
        // silently partial registry.
        let broken = spec(
            "version: 1\nexecution_mode: trusted\nallow_mcp: true\nmcp:\n  a:\n    type: local\n    command: [/nonexistent/mason-test-server]\n    timeout: 500\n",
        )
        .unwrap();
        let error = discover_error(&broken).await;
        #[cfg(feature = "mcp")]
        {
            assert!(error.contains("MCP servers failed"), "{error}");
            assert!(error.contains("unavailable"), "{error}");
        }
        #[cfg(not(feature = "mcp"))]
        assert!(error.contains("without the mcp feature"), "{error}");
    }

    #[test]
    fn the_spill_directory_follows_the_execution_mode() {
        let trusted = spec("version: 1\nexecution_mode: trusted\n").unwrap();
        assert_eq!(
            trusted.spill_path(Path::new("/app/conf")).unwrap(),
            PathBuf::from("/app/conf/.mason/output")
        );
        let explicit = spec("version: 1\nexecution_mode: trusted\nspill: var/spill\n").unwrap();
        assert_eq!(
            explicit.spill_path(Path::new("/app/conf")).unwrap(),
            PathBuf::from("/app/conf/var/spill")
        );
        let sandboxed = spec("version: 1\nexecution_mode: sandbox\nspill: var/spill\n").unwrap();
        let error = sandboxed
            .spill_path(Path::new("/app/conf"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("sandbox"), "{error}");
    }

    /// The rules a manifest declares, for matching tests.
    fn rules(text: &str) -> Vec<SelectRule> {
        spec(text).unwrap().select_profile
    }

    const RULES: &str = "version: 1\nexecution_mode: sandbox\nprofiles:\n  unattended:\n    description: Unattended.\n  reviewer:\n    description: Reviews.\nselect_profile:\n  - profile: unattended\n    when_metadata:\n      tenant_id:\n        present: true\n  - profile: reviewer\n    when_metadata:\n      tier:\n        equals: review\n      region.nested.country:\n        equals: AU\n";

    fn matched(metadata: serde_json::Value) -> Option<String> {
        rules(RULES)
            .iter()
            .find(|rule| rule.matches(&metadata))
            .map(|rule| rule.profile.clone())
    }

    #[test]
    fn present_matches_a_value_that_is_not_null() {
        use serde_json::json;
        assert_eq!(
            matched(json!({"tenant_id": 7})).as_deref(),
            Some("unattended")
        );
        for value in [json!(0), json!(false), json!("")] {
            assert_eq!(
                matched(json!({"tenant_id": value})).as_deref(),
                Some("unattended"),
                "a falsy value is still present: {value}"
            );
        }
        assert_eq!(matched(json!({"tenant_id": serde_json::Value::Null})), None);
        assert_eq!(matched(json!({})), None);
        assert_eq!(matched(serde_json::Value::Null), None);
    }

    #[test]
    fn equals_is_typed_and_conditions_are_anded() {
        use serde_json::json;
        assert_eq!(
            matched(json!({"tier": "review", "region": {"nested": {"country": "AU"}}})).as_deref(),
            Some("reviewer"),
            "a dotted path reaches nested metadata"
        );
        assert_eq!(
            matched(json!({"tier": "review"})),
            None,
            "every condition must hold"
        );
        assert_eq!(
            matched(json!({"tier": "review", "region": {"nested": {"country": "NZ"}}})),
            None,
            "no string coercion or fuzzy matching"
        );
        assert_eq!(matched(json!({"tier": json!(2)})), None);
    }

    #[test]
    fn the_first_matching_rule_wins() {
        use serde_json::json;
        assert_eq!(
            matched(
                json!({"tenant_id": 1, "tier": "review", "region": {"nested": {"country": "AU"}}})
            )
            .as_deref(),
            Some("unattended"),
            "rules are ordered, not merged"
        );
    }

    #[test]
    fn ambiguous_match_forms_are_rejected() {
        let text = "version: 1\nexecution_mode: sandbox\nselect_profile:\n  - profile: build\n    when_metadata:\n      tenant_id:\n        present: true\n        equals: 1\n";
        let error = parse(text).unwrap_err().to_string();
        assert!(
            error.contains("equals") || error.contains("unknown field"),
            "{error}"
        );
    }

    #[cfg(feature = "mcp")]
    #[test]
    fn an_explicit_mcp_cwd_resolves_against_the_manifest() {
        let resolved = spec(
            "version: 1\nexecution_mode: trusted\nallow_mcp: true\nmcp:\n  a:\n    type: local\n    command: [node]\n    cwd: tools\n  b:\n    type: local\n    command: [node]\n    cwd: /opt/absolute\n",
        )
        .unwrap();
        let configs = resolved.mcp_configs().unwrap();
        let cwd_of = |name: &str| match configs.get(name).unwrap() {
            crate::mcp::McpServerConfig::Local { cwd, .. } => cwd.clone(),
            _ => panic!("a local server"),
        };
        assert_eq!(cwd_of("a").as_deref(), Some("/app/conf/tools"));
        assert_eq!(cwd_of("b").as_deref(), Some("/opt/absolute"));
    }

    #[test]
    fn an_mcp_file_is_read_when_it_is_used_not_when_the_manifest_loads() {
        let text = "version: 1\nexecution_mode: trusted\nallow_mcp: true\nmcp_file: missing.json\n";
        // Loading and resolving must not touch the file: a `--mcp` override
        // replaces it, and the default may legitimately be absent.
        // Only the `mcp` build mutates it, to install an override.
        #[allow(unused_mut)]
        let mut resolved = spec(text).unwrap();
        assert_eq!(
            resolved.mcp_file.as_deref(),
            Some(Path::new("/app/conf/missing.json"))
        );
        #[cfg(feature = "mcp")]
        {
            let error = resolved.mcp_configs().unwrap_err().to_string();
            assert!(error.contains("cannot read"), "{error}");

            // The override a CLI flag installs is the one that gets read.
            let dir = tempfile::tempdir().unwrap();
            std::fs::write(
                dir.path().join("other.json"),
                r#"{"b": {"type": "local", "command": ["node", "b.js"], "cwd": "sub"}}"#,
            )
            .unwrap();
            resolved.mcp.clear();
            resolved.mcp_file = Some(dir.path().join("other.json"));
            let configs = resolved.mcp_configs().unwrap();
            match configs.get("b").unwrap() {
                crate::mcp::McpServerConfig::Local { cwd, .. } => {
                    assert_eq!(
                        cwd.as_deref(),
                        Some(dir.path().join("sub").to_str().unwrap()),
                        "a file's relative cwd resolves against that file"
                    );
                }
                _ => panic!("a local server"),
            }
        }
    }

    #[test]
    fn a_rule_cannot_target_a_subagent_profile() {
        let head = "version: 1\nexecution_mode: sandbox\nprofiles:\n  helper:\n    description: Helps.\n    subagent: true\n  worker:\n    description: Works.\nselect_profile:\n  - profile: ";
        for target in ["explore", "general", "helper"] {
            let text = format!(
                "{head}{target}\n    when_metadata:\n      tenant_id:\n        present: true\n"
            );
            let error = parse(&text).unwrap_err().to_string();
            assert!(error.contains("subagent profile"), "{target}: {error}");
        }
        // A top-level target is fine.
        for target in ["build", "worker"] {
            let text = format!(
                "{head}{target}\n    when_metadata:\n      tenant_id:\n        present: true\n"
            );
            assert!(parse(&text).is_ok(), "{target} must be accepted");
        }
    }

    #[test]
    fn durations_reject_overflow_instead_of_wrapping() {
        for raw in [
            "307445734561825861m",   // x 60_000 overflows
            "18446744073709551615h", // x 3_600_000 overflows
            "18446744073709551615s", // x 1_000 overflows
            "99999999999999999999s", // does not fit u64 at all
        ] {
            let error = parse_duration(raw).unwrap_err().to_string();
            assert!(
                error.contains("too large"),
                "`{raw}` must be rejected, got: {error}"
            );
        }
        // Just under the edge still works.
        assert_eq!(
            parse_duration("18446744073709551615ms").unwrap(),
            Duration::from_millis(u64::MAX)
        );
    }

    #[cfg(feature = "mcp")]
    #[test]
    fn an_mcp_file_is_loaded_and_validated() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("servers.json"),
            r#"{"a": {"type": "local", "command": ["x"]}, "b": {"type": "local", "command": []}}"#,
        )
        .unwrap();
        let text = format!(
            "version: 1\nexecution_mode: sandbox\nmcp_file: {}\n",
            dir.path().join("servers.json").display()
        );
        // The file is read when the servers are resolved, not when the manifest
        // loads, so a `--mcp` override never trips over the file it replaces.
        let error = Manifest::parse(&text, PathBuf::from("/unused"))
            .unwrap()
            .into_spec()
            .unwrap()
            .mcp_configs()
            .unwrap_err()
            .to_string();
        assert!(error.contains("must not be empty"), "{error}");

        std::fs::write(
            dir.path().join("servers.json"),
            r#"{"a": {"type": "local", "command": ["x"]}}"#,
        )
        .unwrap();
        let spec = Manifest::parse(&text, PathBuf::from("/unused"))
            .unwrap()
            .into_spec()
            .unwrap();
        assert!(spec.mcp.is_empty(), "the file is not read at load time");
        assert_eq!(spec.mcp_configs().unwrap().len(), 1);
    }
}

#[cfg(test)]
mod build_tests {
    //! Builder-level tests: they need a model backend, so they construct `Vendors`
    //! through `from_lookup` instead of touching the process environment.

    use super::*;
    use crate::Agent;
    use crate::llm::{ToolSpec, Vendors};
    use crate::tool::{Tool, ToolCtx, ToolError};

    fn vendors() -> Vendors {
        Vendors::from_lookup(|key| match key {
            "AI_VENDORS" => Some("mock".to_owned()),
            "AI_MOCK_URL" => Some("http://127.0.0.1:1/v1".to_owned()),
            "AI_MOCK_MODEL" => Some("mock-model".to_owned()),
            "AI_MOCK_API_KEY" => Some("mock-key".to_owned()),
            _ => None,
        })
        .unwrap()
    }

    /// A stand-in for an application's own in-process tool.
    struct NativeTool;

    #[async_trait::async_trait]
    impl Tool for NativeTool {
        fn spec(&self) -> ToolSpec {
            ToolSpec::function(
                "acme_sql",
                "Runs a read-only query.",
                serde_json::json!({"type": "object", "properties": {}}),
            )
        }
        async fn call(&self, _: serde_json::Value, _: &ToolCtx) -> Result<String, ToolError> {
            Ok("native".to_owned())
        }
    }

    fn write_manifest(dir: &Path, text: &str) -> std::path::PathBuf {
        let path = dir.join("agent.yaml");
        std::fs::write(&path, text).unwrap();
        path
    }

    fn manifest(dir: &Path, text: &str) -> Manifest {
        Manifest::load(write_manifest(dir, text)).unwrap()
    }

    const FIXTURE: &str = "version: 1\nexecution_mode: trusted\ninstructions: |\n  Domain prose here.\nprofiles:\n  reviewer:\n    description: Reviews changes.\n    prompt: Review carefully.\n    tools: [read, grep, acme_sql]\n    max_steps: 12\n  explore:\n    tools: [read, grep, glob, skill, acme_sql]\nlimits:\n  max_steps: 40\n  timeout: 20m\n  max_concurrent_tools: 2\n";

    #[tokio::test]
    async fn a_manifest_builds_the_same_agent_as_the_equivalent_builder_calls() {
        let dir = tempfile::tempdir().unwrap();
        // An absolute `cwd` passes through unchanged, so both sides agree.
        let text = format!("{FIXTURE}cwd: {}\n", dir.path().display());
        let spec = manifest(dir.path(), &text).into_spec().unwrap();
        let from_manifest = spec
            .into_builder_with_vendors(vendors())
            .await
            .unwrap()
            // An application adds its own in-process tools on top of the manifest.
            .tool(Box::new(NativeTool))
            .build()
            .unwrap();

        let explicit = Agent::builder()
            .vendors(vendors())
            .cwd(dir.path())
            .instructions("Domain prose here.\n")
            .tools_shared(crate::tool::builtin_tools().into_iter().map(Arc::from))
            .limits(Limits {
                max_steps: 40,
                timeout: Duration::from_secs(20 * 60),
                max_concurrent_tools: 2,
                ..Limits::default()
            })
            .spill(Arc::new(crate::OutputSpill::new(
                dir.path().join(".mason/output"),
            )))
            .shell_timeouts(DEFAULT_SHELL_TIMEOUT_MS, DEFAULT_SHELL_MAX_TIMEOUT_MS)
            .model_timeouts(ModelTimeouts {
                request: Duration::from_millis(DEFAULT_MODEL_REQUEST_TIMEOUT_MS),
                turn: Duration::from_millis(DEFAULT_MODEL_TURN_TIMEOUT_MS),
                idle: Duration::from_millis(DEFAULT_MODEL_IDLE_TIMEOUT_MS),
            })
            .execution_policy(
                ExecutionMode::Trusted,
                false,
                Arc::new(crate::tool::policy::CliPolicy { allow_mcp: false }),
            )
            .prompts(PromptSource::Embedded)
            .profiles([Profile {
                name: "reviewer".into(),
                description: "Reviews changes.".into(),
                prompt: Some("Review carefully.".into()),
                tools: Some(vec![
                    "read".to_owned(),
                    "grep".to_owned(),
                    "acme_sql".to_owned(),
                ]),
                max_steps: 12,
                ..Profile::new(String::new(), String::new())
            }])
            .tool(Box::new(NativeTool))
            .patch_profile(
                "explore",
                ProfilePatch {
                    tools: Some(vec![
                        "read".to_owned(),
                        "grep".to_owned(),
                        "glob".to_owned(),
                        "skill".to_owned(),
                        "acme_sql".to_owned(),
                    ]),
                    ..Default::default()
                },
            )
            .build()
            .unwrap();

        assert_eq!(from_manifest.tools().names(), explicit.tools().names());
        assert_eq!(
            from_manifest.profiles().iter().cloned().collect::<Vec<_>>(),
            explicit.profiles().iter().cloned().collect::<Vec<_>>()
        );
        assert_eq!(from_manifest.instructions(), explicit.instructions());
        assert_eq!(from_manifest.limits(), explicit.limits());
        assert_eq!(from_manifest.cwd(), explicit.cwd());
    }

    #[tokio::test]
    async fn a_profile_may_reference_a_native_tool_added_after_the_manifest() {
        let dir = tempfile::tempdir().unwrap();
        let spec = manifest(dir.path(), FIXTURE).into_spec().unwrap();
        // Without the application's own tool the reference is rejected at build.
        let error = spec
            .clone()
            .into_builder_with_vendors(vendors())
            .await
            .unwrap()
            .build()
            .unwrap_err()
            .to_string();
        assert!(error.contains("acme_sql"), "{error}");
        assert!(error.contains("unregistered tool"), "{error}");

        // Adding it in code satisfies the same reference.
        let agent = spec
            .into_builder_with_vendors(vendors())
            .await
            .unwrap()
            .tool(Box::new(NativeTool))
            .build()
            .unwrap();
        assert!(agent.tools().get("acme_sql").is_some());
        assert_eq!(
            agent.profiles().get("explore").unwrap().tools,
            Some(vec![
                "read".to_owned(),
                "grep".to_owned(),
                "glob".to_owned(),
                "skill".to_owned(),
                "acme_sql".to_owned()
            ]),
            "a tools-only patch kept the builtin's other settings"
        );
        assert!(agent.profiles().get("explore").unwrap().subagent);
    }

    /// A stand-in for an application's own model backend.
    struct StubModel;

    #[async_trait::async_trait]
    impl crate::llm::ChatModel for StubModel {
        fn context_tokens(&self) -> usize {
            128_000
        }
        async fn stream(
            &self,
            _request: crate::llm::ChatRequest,
            _token: &crate::CancellationToken,
        ) -> crate::Result<crate::llm::ChatStream> {
            Err(crate::Error::Run("unused".to_owned()))
        }
    }

    #[tokio::test]
    async fn a_manifest_composes_with_a_custom_model() {
        let dir = tempfile::tempdir().unwrap();
        let text = format!("{FIXTURE}cwd: {}\n", dir.path().display());
        let agent = Manifest::load(write_manifest(dir.path(), &text))
            .unwrap()
            .into_builder_with_model(Arc::new(StubModel))
            .await
            .unwrap()
            .tool(Box::new(NativeTool))
            .build()
            .unwrap();
        assert_eq!(agent.instructions().unwrap(), "Domain prose here.\n");
        assert!(agent.profiles().get("reviewer").is_some());
        assert!(agent.tools().get("acme_sql").is_some());
        assert_eq!(agent.model().context_tokens(), 128_000);
    }

    #[tokio::test]
    async fn a_manifest_selector_and_a_code_selector_cannot_both_be_installed() {
        let dir = tempfile::tempdir().unwrap();
        let text = format!(
            "{FIXTURE}select_profile:\n  - profile: reviewer\n    when_metadata:\n      tenant_id:\n        present: true\n"
        );
        let spec = manifest(dir.path(), &text).into_spec().unwrap();
        let error = spec
            .into_builder_with_vendors(vendors())
            .await
            .unwrap()
            .tool(Box::new(NativeTool))
            .profile_selector(|_| None)
            .build()
            .unwrap_err()
            .to_string();
        assert!(error.contains("two profile selectors"), "{error}");
    }
}
