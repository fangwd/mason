//! Validated YAML prompts, embedded or reloaded per run. Runs and their children share
//! an immutable snapshot; interpolation is single-pass `{{name}}` substitution.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::Deserialize;

use crate::Error;
use crate::error::Result;

/// Embedded prompt data shares the source of the shipped YAML defaults.
pub const EMBEDDED_PROMPTS: &str = include_str!("../prompts/prompts.yaml");

/// Why a prompt set could not be loaded.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PromptError {
    /// The prompt file could not be read.
    #[error("cannot read the prompt file {}: {source}", path.display())]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    /// The YAML source does not have the expected shape. Where serde can
    /// identify one, the message names the offending field.
    #[error("cannot parse the prompt set: {0}")]
    Parse(String),
    /// A prompt field failed validation.
    #[error("invalid prompt field `{field}`: {reason}")]
    Field {
        /// The YAML path of the offending field, e.g. `compaction.template`.
        field: String,
        reason: String,
    },
}

/// Where an agent's prompts come from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromptSource {
    /// The prompt set embedded at compile time ([`EMBEDDED_PROMPTS`]).
    Embedded,
    /// Overrides over the embedded prompts: the file sets only the keys it
    /// changes. Validated at build time and reloaded at the start of each
    /// top-level run.
    File(PathBuf),
}

impl PromptSource {
    /// Loads an immutable snapshot shared by a run and its children.
    pub fn load(&self) -> Result<Arc<PromptSet>> {
        let set = match self {
            PromptSource::Embedded => PromptSet::parse(EMBEDDED_PROMPTS)?,
            PromptSource::File(path) => PromptSet::read(path)?,
        };
        Ok(Arc::new(set))
    }
}

/// Validated, immutable prompts shared by a run, its compactions, and its children.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PromptSet {
    /// Generic system-prompt building blocks.
    pub system: SystemPrompts,
    /// Sent with tools disabled when a run hits its step limit.
    pub max_steps: String,
    /// Sent once after a reply was cut off at the provider's output limit.
    pub output_limit: String,
    pub compaction: CompactionPrompts,
    pub task: TaskPrompts,
    pub skill: SkillPrompts,
    pub memory: MemoryPrompts,
    pub reflection: ReflectionPrompts,
    pub consolidation: ConsolidationPrompts,
    pub evaluation: EvaluationPrompts,
    pub profiles: ProfilePrompts,
}

/// System-prompt sections; applications append domain instructions through the builder.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SystemPrompts {
    /// What the agent is and how it should work. Permits `{{cwd}}`.
    pub base: String,
    /// The untrusted-input warning. Permits no placeholders.
    pub untrusted: String,
    /// Environment facts. Permits `{{cwd}}`, `{{platform}}` and `{{date}}`.
    pub environment: String,
}

/// Context-compaction prompts (opencode-derived).
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompactionPrompts {
    /// System prompt for the summarization turn.
    pub system: String,
    /// The required summary structure.
    pub template: String,
    /// How to merge a prior summary with newer conversation.
    pub update_instructions: String,
    /// Instruction when there is no prior summary.
    pub new_summary: String,
    /// Wraps a stored summary when rebuilding history. Permits `{{summary}}`.
    pub checkpoint: String,
}

/// Task-tool prompts.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskPrompts {
    /// Task description prefix; run assembly appends call limits and allowed child
    /// profiles.
    pub description: String,
}

/// Skill-tool prompts.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SkillPrompts {
    pub description: String,
    /// System-prompt section listing installed skills. Permits `{{skills}}`.
    pub section: String,
}

/// Memory-tool prompts.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryPrompts {
    pub description: String,
    /// System-prompt section with saving guidance and the memory index. Permits
    /// `{{index}}`.
    pub section: String,
}

/// Prompts of the post-run reflection that maintains memory.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReflectionPrompts {
    /// Reflection's system-prompt preamble; the memory section follows it.
    pub system: String,
    /// The review request. Permits `{{signals}}` and `{{transcript}}`.
    pub prompt: String,
}

/// Prompts of the consolidation that merges, prunes and promotes memories.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConsolidationPrompts {
    pub system: String,
    /// The consolidation request. Permits `{{memories}}` and `{{skills}}`.
    pub prompt: String,
    /// The `learned_skill` tool description.
    pub skill_tool: String,
}

/// Prompts of the LLM evaluator that grades finished runs.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationPrompts {
    pub system: String,
    /// The grading request. Permits `{{transcript}}`.
    pub prompt: String,
    /// The `verdict` tool description.
    pub tool: String,
}

/// Texts of the built-in subagent profiles.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfilePrompts {
    pub general: ProfileText,
    pub explore: ProfileText,
}

/// A built-in profile's user-facing texts.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileText {
    /// Shown to a parent agent choosing children for the task tool.
    pub description: String,
    /// Optional profile-specific system prompt.
    pub prompt: Option<String>,
}

/// Facts interpolated into the system prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Environment {
    pub cwd: String,
    /// The platform, e.g. `macos` or `linux`.
    pub platform: String,
    /// Today's date in UTC, `YYYY-MM-DD`.
    pub date: String,
}

impl Environment {
    /// Captures the current platform and today's UTC date for `cwd`.
    pub fn capture(cwd: impl Into<String>) -> Self {
        Self {
            cwd: cwd.into(),
            platform: std::env::consts::OS.to_owned(),
            date: chrono::Utc::now().format("%Y-%m-%d").to_string(),
        }
    }
}

impl PromptSet {
    /// Parses and validates a YAML prompt set.
    pub fn parse(yaml: &str) -> Result<Self> {
        let set: Self = serde_saphyr::from_str(yaml)
            .map_err(|err| Error::Prompt(PromptError::Parse(err.to_string())))?;
        set.validate()?;
        Ok(set)
    }

    /// Reads a prompt file and validates it as overrides of the embedded prompts.
    pub fn read(path: &Path) -> Result<Self> {
        let yaml = std::fs::read_to_string(path).map_err(|source| {
            Error::Prompt(PromptError::Io {
                path: path.to_path_buf(),
                source,
            })
        })?;
        Self::overlay(&yaml)
    }

    /// Deep-merges `yaml` over the embedded prompts (maps merge, other values
    /// replace), then validates the result like a complete set.
    pub fn overlay(yaml: &str) -> Result<Self> {
        let parse_error = |err: String| Error::Prompt(PromptError::Parse(err));
        let parse = |text: &str| {
            serde_saphyr::from_str::<serde_json::Value>(text)
                .map_err(|err| parse_error(err.to_string()))
        };
        let mut merged = parse(EMBEDDED_PROMPTS)?;
        match parse(yaml)? {
            serde_json::Value::Null => {}
            overrides @ serde_json::Value::Object(_) => merge(&mut merged, overrides),
            _ => return Err(parse_error("the prompt file must be a mapping".to_owned())),
        }
        let set: Self =
            serde_json::from_value(merged).map_err(|err| parse_error(err.to_string()))?;
        set.validate()?;
        Ok(set)
    }

    /// Renders the base, untrusted-input warning, and environment; run assembly appends
    /// domain, profile, and skill text.
    pub fn system_prompt(&self, env: &Environment) -> String {
        let base = interpolate(&self.system.base, &[("cwd", env.cwd.as_str())]);
        let environment = interpolate(
            &self.system.environment,
            &[
                ("cwd", env.cwd.as_str()),
                ("platform", env.platform.as_str()),
                ("date", env.date.as_str()),
            ],
        );
        [base, self.system.untrusted.clone(), environment].join("\n\n")
    }

    /// The system-prompt section listing the installed skills.
    pub fn skill_section(&self, skills: &str) -> String {
        interpolate(&self.skill.section, &[("skills", skills)])
    }

    /// The system-prompt section with the memory index.
    pub fn memory_section(&self, index: &str) -> String {
        interpolate(&self.memory.section, &[("index", index)])
    }

    /// The reflection request for one run.
    pub fn reflection_prompt(&self, signals: &str, transcript: &str) -> String {
        interpolate(
            &self.reflection.prompt,
            &[("signals", signals), ("transcript", transcript)],
        )
    }

    /// The consolidation request over the current memories and learned skills.
    pub fn consolidation_prompt(&self, memories: &str, skills: &str) -> String {
        interpolate(
            &self.consolidation.prompt,
            &[("memories", memories), ("skills", skills)],
        )
    }

    /// The grading request for one run.
    pub fn evaluation_prompt(&self, transcript: &str) -> String {
        interpolate(&self.evaluation.prompt, &[("transcript", transcript)])
    }

    /// The history checkpoint wrapping a compaction summary.
    pub fn checkpoint(&self, summary: &str) -> String {
        interpolate(&self.compaction.checkpoint, &[("summary", summary)])
    }

    /// Validates nonempty prompts and permitted placeholders.
    fn validate(&self) -> Result<()> {
        for (field, template, allowed) in self.templates() {
            validate_template(field, template, allowed)?;
        }
        Ok(())
    }

    /// Enumerates prompt paths, text, and permitted placeholders.
    fn templates(&self) -> Vec<(&str, &str, &'static [&'static str])> {
        let mut templates = vec![
            ("system.base", self.system.base.as_str(), &["cwd"][..]),
            ("system.untrusted", self.system.untrusted.as_str(), &[][..]),
            (
                "system.environment",
                self.system.environment.as_str(),
                &["cwd", "platform", "date"][..],
            ),
            ("max_steps", self.max_steps.as_str(), &[][..]),
            ("output_limit", self.output_limit.as_str(), &[][..]),
            (
                "compaction.system",
                self.compaction.system.as_str(),
                &[][..],
            ),
            (
                "compaction.template",
                self.compaction.template.as_str(),
                &[][..],
            ),
            (
                "compaction.update_instructions",
                self.compaction.update_instructions.as_str(),
                &[][..],
            ),
            (
                "compaction.new_summary",
                self.compaction.new_summary.as_str(),
                &[][..],
            ),
            (
                "compaction.checkpoint",
                self.compaction.checkpoint.as_str(),
                &["summary"][..],
            ),
            ("task.description", self.task.description.as_str(), &[][..]),
            (
                "skill.description",
                self.skill.description.as_str(),
                &[][..],
            ),
            (
                "skill.section",
                self.skill.section.as_str(),
                &["skills"][..],
            ),
            (
                "memory.description",
                self.memory.description.as_str(),
                &[][..],
            ),
            (
                "memory.section",
                self.memory.section.as_str(),
                &["index"][..],
            ),
            (
                "reflection.system",
                self.reflection.system.as_str(),
                &[][..],
            ),
            (
                "reflection.prompt",
                self.reflection.prompt.as_str(),
                &["signals", "transcript"][..],
            ),
            (
                "consolidation.system",
                self.consolidation.system.as_str(),
                &[][..],
            ),
            (
                "consolidation.prompt",
                self.consolidation.prompt.as_str(),
                &["memories", "skills"][..],
            ),
            (
                "consolidation.skill_tool",
                self.consolidation.skill_tool.as_str(),
                &[][..],
            ),
            (
                "evaluation.system",
                self.evaluation.system.as_str(),
                &[][..],
            ),
            (
                "evaluation.prompt",
                self.evaluation.prompt.as_str(),
                &["transcript"][..],
            ),
            ("evaluation.tool", self.evaluation.tool.as_str(), &[][..]),
            (
                "profiles.general.description",
                self.profiles.general.description.as_str(),
                &[][..],
            ),
            (
                "profiles.explore.description",
                self.profiles.explore.description.as_str(),
                &[][..],
            ),
        ];
        for (field, prompt) in [
            ("profiles.general.prompt", &self.profiles.general.prompt),
            ("profiles.explore.prompt", &self.profiles.explore.prompt),
        ] {
            if let Some(prompt) = prompt {
                templates.push((field, prompt.as_str(), &[][..]));
            }
        }
        templates
    }
}

/// Rejects blank templates, malformed placeholders, and unsupported placeholder names.
fn validate_template(field: &str, template: &str, allowed: &[&str]) -> Result<()> {
    if template.trim().is_empty() {
        return Err(field_error(field, "is blank"));
    }
    let mut rest = template;
    while let Some(open) = rest.find("{{") {
        let after = &rest[open + 2..];
        let name_len = after
            .bytes()
            .take_while(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
            .count();
        let (name, tail) = after.split_at(name_len);
        if name.is_empty() || !tail.starts_with("}}") {
            return Err(field_error(
                field,
                "has a malformed placeholder: `{{` must be followed by `name}}`",
            ));
        }
        if !allowed.contains(&name) {
            return Err(field_error(
                field,
                format!("interpolates `{{{{{name}}}}}`, which this template does not permit"),
            ));
        }
        rest = &tail[2..];
    }
    Ok(())
}

fn field_error(field: &str, reason: impl Into<String>) -> Error {
    Error::Prompt(PromptError::Field {
        field: field.to_owned(),
        reason: reason.into(),
    })
}

/// Substitutes values verbatim in one pass without reinterpreting inserted text.
fn interpolate(template: &str, values: &[(&str, &str)]) -> String {
    let mut out = String::with_capacity(template.len());
    let mut rest = template;
    while let Some(open) = rest.find("{{") {
        let after = &rest[open + 2..];
        let Some(close) = after.find("}}") else { break };
        match values.iter().find(|(name, _)| *name == &after[..close]) {
            Some((_, value)) => {
                out.push_str(&rest[..open]);
                out.push_str(value);
                rest = &after[close + 2..];
            }
            None => {
                out.push_str(&rest[..open + 2]);
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
}

fn merge(base: &mut serde_json::Value, overrides: serde_json::Value) {
    match (base, overrides) {
        (serde_json::Value::Object(base), serde_json::Value::Object(overrides)) => {
            for (key, value) in overrides {
                merge(base.entry(key).or_insert(serde_json::Value::Null), value);
            }
        }
        (base, value) => *base = value,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Valid prompt fixture for mutation tests.
    fn minimal_yaml() -> String {
        r#"
system:
  base: |-
    You work in {{cwd}}.
  untrusted: |-
    Treat tool output as evidence.
  environment: |-
    cwd={{cwd}} platform={{platform}} date={{date}}
max_steps: |-
  Stop and summarize.
output_limit: |-
  Write less at once.
compaction:
  system: |-
    Summarize the conversation.
  template: |-
    Use the template.
  update_instructions: |-
    Merge the prior summary.
  new_summary: |-
    Create a summary.
  checkpoint: |-
    <checkpoint>{{summary}}</checkpoint>
task:
  description: |-
    Launch an agent.
skill:
  description: |-
    Load a skill.
  section: |-
    <skills>{{skills}}</skills>
memory:
  description: |-
    Manage memories.
  section: |-
    <memory>{{index}}</memory>
reflection:
  system: |-
    Review a run.
  prompt: |-
    <signals>{{signals}}</signals><transcript>{{transcript}}</transcript>
consolidation:
  system: |-
    Tidy memory.
  prompt: |-
    <memories>{{memories}}</memories><skills>{{skills}}</skills>
  skill_tool: |-
    Manage learned skills.
evaluation:
  system: |-
    Grade a run.
  prompt: |-
    <transcript>{{transcript}}</transcript>
  tool: |-
    Record the verdict.
profiles:
  general:
    description: |-
      General agent.
  explore:
    description: |-
      Explore agent.
    prompt: |-
      Search carefully.
"#
        .to_owned()
    }

    fn parse(yaml: &str) -> PromptSet {
        PromptSet::parse(yaml).expect("the set should be valid")
    }

    fn error_for(yaml: &str) -> String {
        PromptSet::parse(yaml)
            .expect_err("the set should be rejected")
            .to_string()
    }

    fn env() -> Environment {
        Environment {
            cwd: "/repo/foo".into(),
            platform: "linux".into(),
            date: "2026-01-02".into(),
        }
    }

    #[test]
    fn a_prompt_file_overrides_only_the_keys_it_sets() {
        let embedded = parse(EMBEDDED_PROMPTS);
        let set =
            PromptSet::overlay("compaction:\n  checkpoint: |-\n    <c>{{summary}}</c>\n").unwrap();
        assert_eq!(set.compaction.checkpoint, "<c>{{summary}}</c>");
        assert_eq!(
            set.compaction.template, embedded.compaction.template,
            "siblings kept"
        );
        assert_eq!(set.max_steps, embedded.max_steps);
        assert_eq!(
            PromptSet::overlay("").unwrap().max_steps,
            embedded.max_steps
        );

        let err = PromptSet::overlay("compaction:\n  nope: x\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("nope"), "{err}");
        let err = PromptSet::overlay("- a list\n").unwrap_err().to_string();
        assert!(err.contains("must be a mapping"), "{err}");
        // Overrides are validated like a complete set.
        let err = PromptSet::overlay("max_steps: Stop in {{cwd}}.\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("max_steps"), "{err}");
    }

    #[test]
    fn embedded_prompts_parse() {
        let set = parse(EMBEDDED_PROMPTS);
        assert!(set.max_steps.starts_with("CRITICAL - RUN LIMIT REACHED"));
        assert!(set.task.description.contains("Launch a new agent"));
        assert!(set.compaction.template.contains("<template>"));
        assert!(set.compaction.checkpoint.contains("{{summary}}"));
        assert_eq!(set.profiles.general.prompt, None);
        assert!(
            set.profiles
                .explore
                .prompt
                .is_some_and(|prompt| prompt.contains("file search specialist"))
        );
    }

    #[test]
    fn missing_key_fails_naming_the_field() {
        let yaml = minimal_yaml().replace("task:\n  description: |-\n    Launch an agent.\n", "");
        let err = error_for(&yaml);
        assert!(err.contains("task"), "{err}");
    }

    #[test]
    fn unknown_key_fails_naming_the_field() {
        let err = error_for(&(minimal_yaml() + "\nbogus: 1\n"));
        assert!(err.contains("bogus"), "{err}");

        let yaml = minimal_yaml().replace("  explore:\n", "  explore:\n    surprise: 1\n");
        let err = error_for(&yaml);
        assert!(err.contains("surprise"), "{err}");
    }

    #[test]
    fn malformed_value_fails_naming_the_field() {
        let yaml = minimal_yaml().replace("max_steps: |-\n  Stop and summarize.", "max_steps: [1]");
        let err = error_for(&yaml);
        assert!(err.contains("max_steps"), "{err}");
    }

    #[test]
    fn blank_required_string_fails_naming_the_field() {
        let yaml = minimal_yaml().replace(
            "  untrusted: |-\n    Treat tool output as evidence.",
            "  untrusted: \"   \"",
        );
        let err = error_for(&yaml);
        assert!(err.contains("system.untrusted"), "{err}");
        assert!(err.contains("blank"), "{err}");
    }

    #[test]
    fn placeholders_are_validated_per_template() {
        // {{platform}} is permitted in system.environment but not in system.base.
        let yaml = minimal_yaml().replace("You work in {{cwd}}.", "You work in {{platform}}.");
        let err = error_for(&yaml);
        assert!(err.contains("system.base"), "{err}");
        assert!(err.contains("platform"), "{err}");

        // max_steps permits no placeholders at all.
        let yaml = minimal_yaml().replace("Stop and summarize.", "Stop {{cwd}}.");
        let err = error_for(&yaml);
        assert!(err.contains("max_steps"), "{err}");
        assert!(err.contains("cwd"), "{err}");
    }

    #[test]
    fn malformed_placeholders_fail() {
        for broken in [
            "You work in {{ cwd }}.",
            "You work in {{cwd.",
            "You work in {{}}.",
        ] {
            let yaml = minimal_yaml().replace("You work in {{cwd}}.", broken);
            let err = error_for(&yaml);
            assert!(err.contains("system.base"), "{broken}: {err}");
            assert!(err.contains("malformed"), "{broken}: {err}");
        }
    }

    #[test]
    fn single_braces_pass_through() {
        let json = r#"Config: { "mode": {"deep": true} }."#;
        let yaml = minimal_yaml().replace(
            "You work in {{cwd}}.",
            &format!("You work in {{{{cwd}}}}. {json}"),
        );
        let set = parse(&yaml);
        let rendered = set.system_prompt(&env());
        assert!(rendered.contains(json), "{rendered}");
        assert!(rendered.contains("You work in /repo/foo."), "{rendered}");
    }

    #[test]
    fn inserted_template_like_text_is_never_reinterpreted() {
        let set = parse(&minimal_yaml());
        let section = set.skill_section("{{cwd}} stays {{skills}} literal, even }}");
        assert_eq!(
            section,
            "<skills>{{cwd}} stays {{skills}} literal, even }}</skills>"
        );
    }

    #[test]
    fn system_prompt_interpolates_the_environment() {
        let set = parse(&minimal_yaml());
        let rendered = set.system_prompt(&env());
        assert!(rendered.contains("You work in /repo/foo."), "{rendered}");
        assert!(
            rendered.contains("cwd=/repo/foo platform=linux date=2026-01-02"),
            "{rendered}"
        );
        assert!(!rendered.contains("{{"), "{rendered}");
    }

    #[test]
    fn system_prompt_snapshot() {
        let set = parse(EMBEDDED_PROMPTS);
        insta::assert_snapshot!("system_prompt", set.system_prompt(&env()));
    }

    #[test]
    fn checkpoint_wraps_the_summary() {
        let set = parse(&minimal_yaml());
        assert_eq!(set.checkpoint("S"), "<checkpoint>S</checkpoint>");
    }

    #[test]
    fn file_source_rereads_between_runs_and_snapshots_stay_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("prompts.yaml");
        std::fs::write(&path, minimal_yaml()).unwrap();

        let source = PromptSource::File(path.clone());
        let first = source.load().unwrap();
        assert_eq!(first.max_steps, "Stop and summarize.");

        // The next run re-reads the file; the earlier snapshot must not change.
        std::fs::write(
            &path,
            minimal_yaml().replace("Stop and summarize.", "Stop now."),
        )
        .unwrap();
        let second = source.load().unwrap();

        assert_eq!(second.max_steps, "Stop now.");
        assert_eq!(first.max_steps, "Stop and summarize.");
    }

    #[test]
    fn missing_file_fails_naming_the_path() {
        let err = PromptSource::File(PathBuf::from("no/such/prompts.yaml"))
            .load()
            .unwrap_err();
        assert!(err.to_string().contains("no/such/prompts.yaml"), "{err}");
    }

    #[test]
    fn capture_fills_platform_and_utc_date() {
        let captured = Environment::capture("/tmp/x");
        assert_eq!(captured.cwd, "/tmp/x");
        assert!(!captured.platform.is_empty());
        let date = captured.date.as_bytes();
        assert_eq!(date.len(), 10, "{}", captured.date);
        assert_eq!((date[4], date[7]), (b'-', b'-'), "{}", captured.date);
    }
}
