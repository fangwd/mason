//! Profile permissions and explicit subagent delegation through the `task` tool.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use serde::Deserialize;

use crate::Error;
use crate::error::Result;
use crate::llm::ToolSpec;
use crate::prompt::PromptSet;
use crate::store::Session;
use crate::tool::{Tool as ToolTrait, ToolCtx, ToolError, Toolset};

/// What an agent may do.
#[derive(Debug, Clone, PartialEq)]
pub struct Profile {
    /// The unique profile name (`agent` on the wire).
    pub name: String,
    /// Shown to users and to parent agents choosing children.
    pub description: String,
    /// Extra system-prompt prose for this profile.
    pub prompt: Option<String>,
    /// `None` allows every base tool. Runtime-owned `task` and `skill` names may be
    /// listed explicitly.
    pub tools: Option<Vec<String>>,
    /// Explicit child allowlist; subagent profiles must leave it empty.
    pub allowed_subagents: Vec<String>,
    /// Subagent-only profile, launchable through `task` but not as a top-level session.
    pub subagent: bool,
    /// Model calls before the agent must wrap up.
    pub max_steps: usize,
}

impl Profile {
    /// A top-level profile with every base tool and no children.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            prompt: None,
            tools: None,
            allowed_subagents: Vec::new(),
            subagent: false,
            max_steps: 50,
        }
    }

    /// A subagent profile: launchable through `task`, never delegating.
    pub fn subagent_profile(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            subagent: true,
            ..Self::new(name, description)
        }
    }

    /// Whether this profile's tool policy allows `name`.
    pub fn allows(&self, name: &str) -> bool {
        self.tools
            .as_ref()
            .is_none_or(|tools| tools.iter().any(|tool| tool == name))
    }
}

/// Rejects an explicit null so it cannot be confused with an omitted field.
pub(crate) fn deny_null<'de, D, T>(deserializer: D) -> std::result::Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::Deserialize<'de>,
{
    // An omitted field never reaches here (`#[serde(default)]`); a present null does.
    match Option::<T>::deserialize(deserializer)? {
        Some(value) => Ok(Some(value)),
        None => Err(serde::de::Error::custom(
            "an explicit null is not allowed; omit the field instead",
        )),
    }
}

/// A partial profile override: every `None` keeps the resolved value.
///
/// Patching a builtin yields the complete profile the registry stores, so omitted
/// fields keep the builtin's resolved values. A supplied list replaces the
/// builtin's list; an empty list clears it.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfilePatch {
    #[serde(default, deserialize_with = "deny_null")]
    pub description: Option<String>,
    #[serde(default, deserialize_with = "deny_null")]
    pub prompt: Option<String>,
    #[serde(default, deserialize_with = "deny_null")]
    pub tools: Option<Vec<String>>,
    #[serde(default, deserialize_with = "deny_null")]
    pub allowed_subagents: Option<Vec<String>>,
    #[serde(default, deserialize_with = "deny_null")]
    pub max_steps: Option<usize>,
    /// New profiles only; a builtin's subagent status cannot change.
    #[serde(default, deserialize_with = "deny_null")]
    pub subagent: Option<bool>,
}

impl ProfilePatch {
    /// The profile a new (non-builtin) name describes.
    pub fn into_profile(&self, name: &str) -> Result<Profile> {
        let mut profile = if self.subagent.unwrap_or(false) {
            Profile::subagent_profile(name, self.description.clone().unwrap_or_default())
        } else {
            Profile::new(name, self.description.clone().unwrap_or_default())
        };
        self.apply(&mut profile)?;
        Ok(profile)
    }

    /// Copies the supplied fields onto a resolved profile.
    pub fn apply(&self, profile: &mut Profile) -> Result<()> {
        if texts_from_snapshot(&profile.name)
            && (self.description.is_some() || self.prompt.is_some())
        {
            return Err(Error::Config(format!(
                "profile `{}` takes its description and prompt from the prompt snapshot; \
                 replace them through a prompts file instead of patching them",
                profile.name
            )));
        }
        if BUILTIN_PROFILES.contains(&profile.name.as_str()) && self.subagent.is_some() {
            return Err(Error::Config(format!(
                "a builtin profile's subagent status cannot change: `{}`",
                profile.name
            )));
        }
        if let Some(description) = &self.description {
            profile.description = description.clone();
        }
        if let Some(prompt) = &self.prompt {
            profile.prompt = Some(prompt.clone());
        }
        if let Some(tools) = &self.tools {
            profile.tools = Some(tools.clone());
        }
        if let Some(children) = &self.allowed_subagents {
            profile.allowed_subagents = children.clone();
        }
        if let Some(max_steps) = self.max_steps {
            profile.max_steps = max_steps;
        }
        if let Some(subagent) = self.subagent {
            profile.subagent = subagent;
        }
        Ok(())
    }
}

/// Profile registry with `build`, `general`, and tool-dependent `explore` defaults.
#[derive(Debug, Clone, Default)]
pub struct ProfileSet {
    profiles: Vec<Profile>,
}

impl ProfileSet {
    /// Builtin profiles and child allowlists. Prompt texts are resolved from each run's
    /// snapshot.
    pub fn builtin(prompts: &PromptSet, tools: &Toolset) -> Self {
        let mut set = Self::default();
        set.profiles.push(Profile {
            name: "general".into(),
            description: prompts.profiles.general.description.clone(),
            prompt: prompts.profiles.general.prompt.clone(),
            ..Profile::subagent_profile(String::new(), String::new())
        });
        let mut build_children = vec!["general".to_owned()];
        let explore_tools = ["read", "grep", "glob"];
        if explore_tools.iter().all(|name| tools.get(name).is_some()) {
            set.profiles.push(Profile {
                name: "explore".into(),
                description: prompts.profiles.explore.description.clone(),
                prompt: prompts.profiles.explore.prompt.clone(),
                tools: Some(
                    explore_tools
                        .into_iter()
                        .chain(std::iter::once("skill"))
                        .map(str::to_owned)
                        .collect(),
                ),
                ..Profile::subagent_profile(String::new(), String::new())
            });
            build_children.push("explore".into());
        }
        set.profiles.push(Profile {
            name: "build".into(),
            description: "The default agent.".into(),
            allowed_subagents: build_children,
            max_steps: 150,
            ..Profile::new(String::new(), String::new())
        });
        set
    }

    /// Adds a (domain) profile, rejecting duplicate and builtin names.
    pub fn register(&mut self, profile: Profile) -> Result<()> {
        if BUILTIN_PROFILES.contains(&profile.name.as_str())
            || self
                .profiles
                .iter()
                .any(|existing| existing.name == profile.name)
        {
            return Err(Error::Config(format!(
                "duplicate profile name `{}`; profile names are unique registry keys",
                profile.name
            )));
        }
        self.profiles.push(profile);
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<&Profile> {
        self.profiles.iter().find(|profile| profile.name == name)
    }

    /// Replaces a registered profile by name, keeping its position (and so the
    /// child allowlists that already name it). Builtin texts still come from the
    /// prompt snapshot, so replacing `explore` widens its tools, not its wording.
    pub fn replace(&mut self, profile: Profile) -> Result<()> {
        let Some(existing) = self
            .profiles
            .iter_mut()
            .find(|existing| existing.name == profile.name)
        else {
            return Err(Error::Config(format!(
                "no profile named `{}` to replace",
                profile.name
            )));
        };
        *existing = profile;
        Ok(())
    }

    /// Applies a partial override to a registered profile, keeping its position (and
    /// so the child allowlists that already name it).
    pub fn patch(&mut self, name: &str, patch: &ProfilePatch) -> Result<()> {
        let Some(existing) = self
            .profiles
            .iter_mut()
            .find(|profile| profile.name == name)
        else {
            return Err(Error::Config(format!("no profile named `{name}` to patch")));
        };
        patch.apply(existing)
    }

    /// The registered names, in registration order.
    pub fn names(&self) -> Vec<&str> {
        self.profiles
            .iter()
            .map(|profile| profile.name.as_str())
            .collect()
    }

    /// Validates tool and child references; subagent profiles cannot delegate.
    pub fn validate(&self, tools: &Toolset) -> Result<()> {
        for profile in &self.profiles {
            if profile.max_steps == 0 {
                return Err(Error::Config(format!(
                    "profile `{}` must allow at least one step",
                    profile.name
                )));
            }
            if let Some(names) = &profile.tools {
                for name in names {
                    if name == "skill" || name == "task" {
                        continue; // runtime-owned, assembled per run
                    }
                    if tools.get(name).is_none() {
                        return Err(Error::Config(format!(
                            "profile `{}` references the unregistered tool `{name}`",
                            profile.name
                        )));
                    }
                }
            }
            if profile.subagent && !profile.allowed_subagents.is_empty() {
                return Err(Error::Config(format!(
                    "subagent profile `{}` cannot delegate to further subagents",
                    profile.name
                )));
            }
            for child in &profile.allowed_subagents {
                let Some(child_profile) = self.get(child) else {
                    return Err(Error::Config(format!(
                        "profile `{}` references the unknown subagent profile `{child}`",
                        profile.name
                    )));
                };
                if !child_profile.subagent {
                    return Err(Error::Config(format!(
                        "profile `{}` lists `{child}` as a subagent, but it is not one",
                        profile.name
                    )));
                }
            }
        }
        Ok(())
    }

    /// The profiles in registration order.
    pub fn iter(&self) -> impl Iterator<Item = &Profile> {
        self.profiles.iter()
    }
}

/// Names whose texts come from the prompt snapshot; `explore` is reserved even
/// when its tools are absent, so a domain profile cannot be re-texted by name.
const BUILTIN_PROFILES: [&str; 3] = ["build", "general", "explore"];

/// The callback behind a [`ProfileSelector`].
pub type SelectProfile = dyn Fn(&Session) -> Option<String> + Send + Sync;

/// Chooses the profile a stored top-level session runs, from its opaque metadata.
///
/// A returned name replaces the session's stored `agent`; it is resolved against the
/// registry like any other profile name, so an unknown choice fails that run with a
/// diagnostic instead of falling back silently. Child sessions run the profile their
/// `task` call asked for and never pass through a selector.
#[derive(Clone)]
pub struct ProfileSelector(Arc<SelectProfile>);

impl ProfileSelector {
    pub fn new(select: impl Fn(&Session) -> Option<String> + Send + Sync + 'static) -> Self {
        Self(Arc::new(select))
    }

    pub(crate) fn select(&self, session: &Session) -> Option<String> {
        (self.0)(session)
    }
}

impl std::fmt::Debug for ProfileSelector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ProfileSelector(..)")
    }
}

/// Whether a profile's user-facing texts come from the prompt snapshot rather than
/// from the profile itself.
fn texts_from_snapshot(name: &str) -> bool {
    matches!(name, "general" | "explore")
}

/// Builtin texts use the run's prompt snapshot; domain profiles retain their registered
/// text.
pub fn resolve_profile_text(prompts: &PromptSet, profile: &Profile) -> (String, Option<String>) {
    match profile.name.as_str() {
        "general" => (
            prompts.profiles.general.description.clone(),
            prompts.profiles.general.prompt.clone(),
        ),
        "explore" => (
            prompts.profiles.explore.description.clone(),
            prompts.profiles.explore.prompt.clone(),
        ),
        _ => (profile.description.clone(), profile.prompt.clone()),
    }
}

/// What a `task` call asks for.
#[derive(Debug, Clone)]
pub struct SubagentRequest {
    /// The resolved child profile.
    pub profile: Profile,
    pub description: String,
    /// The prompt for the child's fresh conversation.
    pub prompt: String,
    /// The parent's cancellation token: children live and die with it.
    pub token: CancellationToken,
    pub budget: Option<Arc<crate::engine::RunBudget>>,
}

/// What a finished child run returns. Stored child sessions carry an id; in-memory
/// children do not.
#[derive(Debug, Clone)]
pub struct SubagentResult {
    /// The child session id, when the child is stored.
    pub session_id: Option<u64>,
    pub answer: String,
}

/// The injected subagent runner.
pub type LaunchSubagent = Arc<
    dyn Fn(SubagentRequest) -> Pin<Box<dyn Future<Output = Result<SubagentResult>> + Send>>
        + Send
        + Sync,
>;

/// Per-run `task` tool with prompt-derived descriptions and execution-time allowlist
/// checks.
pub struct TaskTool {
    children: Vec<(Profile, String)>,
    launch: LaunchSubagent,
    spec: ToolSpec,
}

impl TaskTool {
    /// Builds the tool from the permitted children and their resolved
    /// descriptions.
    pub fn new(
        prompts: &PromptSet,
        max_task_calls_per_step: usize,
        children: Vec<(Profile, String)>,
        launch: LaunchSubagent,
    ) -> Self {
        let list = children
            .iter()
            .map(|(profile, description)| format!("- {}: {description}", profile.name))
            .collect::<Vec<_>>()
            .join("\n");
        let description = format!(
            "{}\n\nLaunch at most {max_task_calls_per_step} task calls per reply; keep each investigation focused.\n\nAvailable agent types:\n{list}",
            prompts.task.description
        );
        let names: Vec<String> = children
            .iter()
            .map(|(profile, _)| profile.name.clone())
            .collect();
        let spec = ToolSpec::function(
            "task",
            description,
            serde_json::json!({
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "A short (3-5 word) description of the task"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The task for the agent to perform"
                    },
                    "agent": {
                        "type": "string",
                        "description": format!("The type of agent: {}", names.join(", "))
                    }
                },
                "required": ["description", "prompt", "agent"]
            }),
        );
        Self {
            children,
            launch,
            spec,
        }
    }
}

#[async_trait::async_trait]
impl ToolTrait for TaskTool {
    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }

    async fn call(
        &self,
        args: serde_json::Value,
        ctx: &ToolCtx,
    ) -> std::result::Result<String, ToolError> {
        let text = |key: &str| {
            args.get(key)
                .and_then(|value| value.as_str())
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
        };
        let (Some(description), Some(prompt), Some(agent)) =
            (text("description"), text("prompt"), text("agent"))
        else {
            return Err(ToolError::failure(
                "invalid arguments: the task tool requires description, prompt and agent",
            ));
        };
        let Some((profile, _)) = self.children.iter().find(|(child, _)| child.name == agent) else {
            let permitted: Vec<&str> = self
                .children
                .iter()
                .map(|(child, _)| child.name.as_str())
                .collect();
            return Err(ToolError::failure(format!(
                "\"{agent}\" is not a permitted subagent. Permitted: {}",
                permitted.join(", ")
            )));
        };
        let result = (self.launch)(SubagentRequest {
            profile: profile.clone(),
            description,
            prompt,
            token: ctx.token.clone(),
            budget: ctx.budget.clone(),
        })
        .await
        .map_err(|err| match err {
            Error::BudgetExceeded(_) => ToolError::Fatal(err),
            err => ToolError::failure(err.to_string()),
        })?;
        Ok(match result.session_id {
            Some(session) => format!(
                "<task session=\"{session}\" agent=\"{}\">\n<task_result>\n{}\n</task_result>\n</task>",
                profile.name, result.answer
            ),
            None => format!(
                "<task agent=\"{}\">\n<task_result>\n{}\n</task_result>\n</task>",
                profile.name, result.answer
            ),
        })
    }
}

/// Child profiles and descriptions resolved from the current prompt snapshot.
pub fn resolve_children(
    prompts: &PromptSet,
    set: &ProfileSet,
    names: &[String],
) -> Vec<(Profile, String)> {
    names
        .iter()
        .filter_map(|name| set.get(name))
        .map(|profile| {
            let (description, _) = resolve_profile_text(prompts, profile);
            (profile.clone(), description)
        })
        .collect()
}

/// Allows delegation only for non-subagents whose tool policy and child allowlist
/// permit it.
pub fn task_available(profile: &Profile) -> bool {
    !profile.subagent && profile.allows("task") && !profile.allowed_subagents.is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prompt::{EMBEDDED_PROMPTS, PromptSet};
    use crate::tool::builtin_tools;

    fn prompts() -> PromptSet {
        PromptSet::parse(EMBEDDED_PROMPTS).unwrap()
    }

    fn full_toolset() -> Toolset {
        let mut set = Toolset::new();
        set.extend(builtin_tools()).unwrap();
        set
    }

    /// A registry with the builtins resolved from the embedded prompts.
    fn builtins() -> ProfileSet {
        ProfileSet::builtin(&prompts(), &full_toolset())
    }

    fn patch(text: &str) -> ProfilePatch {
        serde_saphyr::from_str(text).unwrap()
    }

    #[test]
    fn a_tools_only_patch_keeps_everything_else_the_builtin_resolved() {
        let mut set = builtins();
        let before = set.get("explore").unwrap().clone();
        set.patch(
            "explore",
            &patch("tools: [read, grep, glob, skill, acme_fetch]\n"),
        )
        .unwrap();
        let after = set.get("explore").unwrap();
        assert_eq!(
            after.tools,
            Some(vec![
                "read".to_owned(),
                "grep".to_owned(),
                "glob".to_owned(),
                "skill".to_owned(),
                "acme_fetch".to_owned()
            ])
        );
        assert_eq!(after.description, before.description);
        assert_eq!(after.prompt, before.prompt);
        assert_eq!(after.max_steps, before.max_steps);
        assert_eq!(after.subagent, before.subagent);
        assert_eq!(after.allowed_subagents, before.allowed_subagents);
        // Its snapshot text is untouched, so the widened profile still reads as explore.
        let (description, prompt) = resolve_profile_text(&prompts(), after);
        assert_eq!(description, before.description);
        assert_eq!(prompt, before.prompt);
    }

    #[test]
    fn a_supplied_list_replaces_and_an_empty_one_clears() {
        let mut set = builtins();
        assert_eq!(
            set.get("build").unwrap().allowed_subagents,
            ["general", "explore"]
        );
        set.patch("build", &patch("allowed_subagents: [general]\n"))
            .unwrap();
        assert_eq!(set.get("build").unwrap().allowed_subagents, ["general"]);
        set.patch("build", &patch("allowed_subagents: []\n"))
            .unwrap();
        assert!(
            set.get("build").unwrap().allowed_subagents.is_empty(),
            "an empty list clears the allowlist rather than keeping it"
        );
        assert!(!task_available(set.get("build").unwrap()));

        set.patch("explore", &patch("tools: []\n")).unwrap();
        assert_eq!(set.get("explore").unwrap().tools, Some(Vec::new()));
    }

    #[test]
    fn a_builtin_patch_cannot_retext_or_resubagent() {
        let mut set = builtins();
        let error = set
            .patch("explore", &patch("description: Mine\n"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("prompt snapshot"), "{error}");
        let error = set
            .patch("general", &patch("prompt: Mine\n"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("prompt snapshot"), "{error}");
        let error = set
            .patch("explore", &patch("subagent: false\n"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("subagent status cannot change"), "{error}");
        // `build` keeps its own texts, so patching them is allowed.
        set.patch("build", &patch("description: The domain default.\n"))
            .unwrap();
        assert_eq!(set.get("build").unwrap().description, "The domain default.");
    }

    #[test]
    fn patching_an_absent_profile_is_an_error() {
        let mut set = ProfileSet::builtin(&prompts(), &Toolset::new());
        assert!(set.get("explore").is_none(), "explore needs its tools");
        let error = set
            .patch("explore", &patch("tools: [read]\n"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("no profile named `explore`"), "{error}");
    }

    #[test]
    fn a_patch_builds_a_complete_new_profile() {
        let profile = patch("description: Reviews pull requests.\nmax_steps: 12\nsubagent: true\n")
            .into_profile("reviewer")
            .unwrap();
        assert_eq!(profile.name, "reviewer");
        assert_eq!(profile.description, "Reviews pull requests.");
        assert_eq!(profile.max_steps, 12);
        assert!(profile.subagent);
        assert_eq!(profile.tools, None, "omitted tools allow every base tool");
        assert!(profile.allowed_subagents.is_empty());
    }

    #[test]
    fn an_explicit_null_patch_field_is_rejected() {
        let error = serde_saphyr::from_str::<ProfilePatch>("tools:\n")
            .unwrap_err()
            .to_string();
        assert!(error.contains("null"), "{error}");
        let error = serde_saphyr::from_str::<ProfilePatch>("nope: 1\n")
            .unwrap_err()
            .to_string();
        assert!(error.contains("nope"), "{error}");
    }

    #[test]
    fn the_builtin_set_follows_the_registered_tools() {
        let set = ProfileSet::builtin(&prompts(), &full_toolset());
        assert!(set.get("build").is_some());
        assert!(set.get("general").is_some());
        let explore = set
            .get("explore")
            .expect("explore joins when its tools exist");
        let explore_tools: Vec<&str> = explore
            .tools
            .as_deref()
            .unwrap()
            .iter()
            .map(String::as_str)
            .collect();
        assert_eq!(explore_tools, ["read", "grep", "glob", "skill"]);
        assert!(explore.subagent);
        let build = set.get("build").unwrap();
        assert_eq!(build.allowed_subagents, ["general", "explore"]);
        assert_eq!(build.max_steps, 150);
        assert!(!build.subagent);
        assert!(build.description.contains("default agent"));
        // general carries the vendored description from the prompt file.
        assert!(
            set.get("general")
                .unwrap()
                .description
                .contains("General-purpose")
        );

        // Without the search tools, explore is not registered and build
        // allows only general.
        let mut empty = Toolset::new();
        empty
            .push(Box::new(crate::tool::Typed(crate::tool::builtin::ReadTool)))
            .unwrap();
        let set = ProfileSet::builtin(&prompts(), &empty);
        assert!(set.get("explore").is_none());
        assert_eq!(set.get("build").unwrap().allowed_subagents, ["general"]);
    }

    #[test]
    fn builtin_profile_names_are_reserved_even_without_their_tools() {
        let mut set = ProfileSet::builtin(&prompts(), &Toolset::new());
        assert!(
            set.get("explore").is_none(),
            "no read/grep/glob: no explore"
        );
        let err = set
            .register(Profile::new("explore", "DOMAIN DESC"))
            .unwrap_err();
        assert!(err.to_string().contains("`explore`"), "{err}");
    }

    #[test]
    fn validation_catches_bad_references() {
        let tools = full_toolset();
        let mut set = ProfileSet::builtin(&prompts(), &tools);

        let unknown_tool = Profile {
            tools: Some(vec!["read".into(), "sql".into()]),
            ..Profile::new("triage-like", "domain")
        };
        set.register(unknown_tool).unwrap();
        let err = set.validate(&tools).unwrap_err();
        assert!(err.to_string().contains("unregistered tool `sql`"), "{err}");

        let mut set = ProfileSet::builtin(&prompts(), &tools);
        set.register(Profile::new("x", "refs a missing child"))
            .unwrap();
        set.profiles.last_mut().unwrap().allowed_subagents = vec!["nope".into()];
        let err = set.validate(&tools).unwrap_err();
        assert!(
            err.to_string().contains("unknown subagent profile `nope`"),
            "{err}"
        );

        let mut set = ProfileSet::builtin(&prompts(), &tools);
        set.register(Profile::new("top", "not a subagent")).unwrap();
        set.register(Profile {
            allowed_subagents: vec!["top".into()],
            ..Profile::new("parent", "wants a non-subagent child")
        })
        .unwrap();
        let err = set.validate(&tools).unwrap_err();
        assert!(err.to_string().contains("but it is not one"), "{err}");

        let mut set = ProfileSet::builtin(&prompts(), &tools);
        set.register(Profile {
            subagent: true,
            allowed_subagents: vec!["general".into()],
            ..Profile::new("sneaky", "delegating subagent")
        })
        .unwrap();
        let err = set.validate(&tools).unwrap_err();
        assert!(err.to_string().contains("cannot delegate"), "{err}");

        let mut set = ProfileSet::builtin(&prompts(), &tools);
        let err = set.register(Profile::new("build", "again")).unwrap_err();
        assert!(err.to_string().contains("duplicate profile name"), "{err}");

        // `skill` and `task` validate without registration, and zero steps fail.
        let mut set = ProfileSet::builtin(&prompts(), &tools);
        set.register(Profile {
            tools: Some(vec!["skill".into(), "task".into()]),
            allowed_subagents: vec!["general".into()],
            ..Profile::new("domain", "ok")
        })
        .unwrap();
        set.validate(&tools).unwrap();
        set.profiles.last_mut().unwrap().max_steps = 0;
        let err = set.validate(&tools).unwrap_err();
        assert!(err.to_string().contains("at least one step"), "{err}");
    }

    #[test]
    fn task_availability_needs_policy_and_children() {
        let tools = full_toolset();
        let set = ProfileSet::builtin(&prompts(), &tools);
        assert!(task_available(set.get("build").unwrap()));
        assert!(
            !task_available(set.get("general").unwrap()),
            "subagents never delegate"
        );

        let restricted = Profile {
            tools: Some(vec!["read".into()]),
            ..Profile::new("no-task", "task not in its policy")
        };
        assert!(
            !task_available(&Profile {
                allowed_subagents: vec!["general".into()],
                ..restricted.clone()
            }),
            "the tool policy must allow task"
        );
    }

    #[tokio::test]
    async fn the_task_tool_lists_and_enforces_its_allowlist() {
        let prompts = prompts();
        let set = ProfileSet::builtin(&prompts, &full_toolset());
        let children =
            resolve_children(&prompts, &set, &set.get("build").unwrap().allowed_subagents);
        let launched: Arc<std::sync::Mutex<Vec<String>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        let launch: LaunchSubagent = Arc::new({
            let launched = launched.clone();
            move |request: SubagentRequest| {
                let launched = launched.clone();
                Box::pin(async move {
                    launched.lock().unwrap().push(request.profile.name.clone());
                    Ok(SubagentResult {
                        session_id: Some(7),
                        answer: format!("did: {}", request.prompt),
                    })
                })
            }
        });
        let tool = TaskTool::new(&prompts, 4, children, launch);
        let spec = tool.spec();
        assert!(
            spec.function
                .description
                .contains("Launch at most 4 task calls per reply")
        );
        assert!(
            spec.function
                .description
                .contains("- general: General-purpose")
        );
        assert!(spec.function.description.contains("- explore: Fast agent"));
        assert_eq!(
            spec.function.parameters["properties"]["agent"]["description"],
            serde_json::json!("The type of agent: general, explore")
        );

        let ctx = ToolCtx::new(std::env::temp_dir());
        let out = tool
            .call(
                serde_json::json!({"description": "d", "prompt": "p", "agent": "general"}),
                &ctx,
            )
            .await
            .unwrap();
        assert_eq!(
            out,
            "<task session=\"7\" agent=\"general\">\n<task_result>\ndid: p\n</task_result>\n</task>"
        );
        assert_eq!(*launched.lock().unwrap(), ["general"]);

        let err = tool
            .call(
                serde_json::json!({"description": "d", "prompt": "p", "agent": "build"}),
                &ctx,
            )
            .await
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("\"build\" is not a permitted subagent"),
            "{err}"
        );
        assert!(
            err.to_string().contains("Permitted: general, explore"),
            "{err}"
        );

        let err = tool
            .call(serde_json::json!({"prompt": "p", "agent": "general"}), &ctx)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("description"), "{err}");
    }

    #[test]
    fn child_texts_refresh_from_the_snapshot() {
        // A reloaded prompt file changes builtin child descriptions for the
        // runs that use it.
        let updated = EMBEDDED_PROMPTS.replacen(
            "General-purpose agent for researching complex questions",
            "Reloaded general description",
            1,
        );
        assert_ne!(
            updated, EMBEDDED_PROMPTS,
            "the fixture replacement must apply"
        );
        let fresh = PromptSet::parse(&updated).unwrap();
        let set = ProfileSet::builtin(
            &PromptSet::parse(EMBEDDED_PROMPTS).unwrap(),
            &full_toolset(),
        );
        let (description, _) = resolve_profile_text(&fresh, set.get("general").unwrap());
        assert!(
            description.starts_with("Reloaded general description"),
            "{description}"
        );
    }
}
