//! Skill discovery and loading from `SKILL.md` folders, snapshotted per top-level run.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::Deserialize;

use crate::llm::ToolSpec;
use crate::tool::{Tool, ToolCtx, ToolError};

#[derive(Debug, Clone, PartialEq)]
pub struct Skill {
    /// The frontmatter name; the model loads skills by it.
    pub name: String,
    /// The frontmatter description, listed in the prompt.
    pub description: String,
    pub dir: PathBuf,
    pub body: String,
}

/// Skill frontmatter; additional metadata fields are ignored.
#[derive(Debug, Deserialize)]
struct SkillFrontmatter {
    name: Option<String>,
    #[serde(default)]
    description: Option<String>,
}

pub const MAX_SKILL_FILES: usize = 10;

/// Loads skills in folder order, warning about and skipping unreadable files and
/// invalid or missing metadata. Folders without a `SKILL.md` are ignored.
pub async fn load_skills(root: &Path) -> Vec<Skill> {
    let entries = match tokio::fs::read_dir(root).await {
        Ok(entries) => entries,
        Err(err) => {
            tracing::warn!(dir = %root.display(), "cannot read the skills directory: {err}");
            return Vec::new();
        }
    };
    let mut folders: Vec<PathBuf> = Vec::new();
    let mut stream = entries;
    while let Ok(Some(entry)) = stream.next_entry().await {
        // Metadata follows symlinks, so linked skill folders load too.
        let is_dir = tokio::fs::metadata(entry.path())
            .await
            .map(|meta| meta.is_dir())
            .unwrap_or(false);
        if is_dir {
            folders.push(entry.path());
        }
    }
    folders.sort();
    let mut skills: Vec<Skill> = Vec::new();
    for dir in folders {
        let path = dir.join("SKILL.md");
        let text = match tokio::fs::read_to_string(&path).await {
            Ok(text) => text,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => continue,
            Err(err) => {
                tracing::warn!(skill = %path.display(), "skipping an unreadable skill: {err}");
                continue;
            }
        };
        match parse_skill(&text, dir) {
            Ok(skill) => {
                if skills.iter().any(|existing| existing.name == skill.name) {
                    tracing::warn!(
                        skill = %path.display(),
                        "skipping a skill whose name `{}` is already taken",
                        skill.name
                    );
                } else {
                    skills.push(skill);
                }
            }
            Err(err) => {
                tracing::warn!(skill = %path.display(), "skipping a skill: {err}");
            }
        }
    }
    skills
}

/// Splits `---` fenced YAML frontmatter from the body.
pub(crate) fn split_frontmatter(text: &str) -> Option<(&str, &str)> {
    use std::sync::OnceLock;
    static FRONTMATTER: OnceLock<regex::Regex> = OnceLock::new();
    let frontmatter = FRONTMATTER.get_or_init(|| {
        regex::Regex::new(r"^---\r?\n([\s\S]*?)\r?\n---\r?\n?")
            .expect("the frontmatter pattern compiles")
    });
    let caps = frontmatter.captures(text)?;
    Some((
        caps.get(1).expect("the pattern has a group").as_str(),
        &text[caps.get(0).expect("the match exists").len()..],
    ))
}

/// Parses one `SKILL.md`: YAML frontmatter with a `name` between `---` fences,
/// then the body.
pub fn parse_skill(text: &str, dir: PathBuf) -> std::result::Result<Skill, String> {
    let Some((yaml, body)) = split_frontmatter(text) else {
        return Err("it does not start with `---` frontmatter".to_owned());
    };
    let meta: SkillFrontmatter =
        serde_saphyr::from_str(yaml).map_err(|err| format!("invalid frontmatter YAML: {err}"))?;
    let Some(name) = meta.name.filter(|name| !name.trim().is_empty()) else {
        return Err("its frontmatter has no `name`".to_owned());
    };
    Ok(Skill {
        name: name.trim().to_owned(),
        description: meta.description.unwrap_or_default(),
        dir,
        body: body.trim().to_owned(),
    })
}

/// The `<available_skills>` entries for the prompt section (the value of the
/// `{{skills}}` placeholder).
pub fn skills_entries(skills: &[Skill]) -> String {
    skills
        .iter()
        .map(|skill| {
            format!(
                "  <skill>\n    <name>{}</name>\n    <description>{}</description>\n  </skill>",
                skill.name, skill.description
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The runtime-owned `skill` tool: loads one skill's instructions and lists
/// up to [`MAX_SKILL_FILES`] supporting files.
pub struct SkillTool {
    skills: Arc<Vec<Skill>>,
    spec: ToolSpec,
}

impl SkillTool {
    /// Builds the tool with its description from the run's prompt snapshot.
    pub fn new(description: String, skills: Arc<Vec<Skill>>) -> Self {
        let spec = ToolSpec::function(
            "skill",
            description,
            serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The name of the skill to load"}
                },
                "required": ["name"]
            }),
        );
        Self { skills, spec }
    }
}

#[async_trait::async_trait]
impl Tool for SkillTool {
    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }

    async fn call(
        &self,
        args: serde_json::Value,
        _ctx: &ToolCtx,
    ) -> std::result::Result<String, ToolError> {
        let Some(name) = args.get("name").and_then(|value| value.as_str()) else {
            return Err(ToolError::failure(
                "invalid arguments: missing field `name`",
            ));
        };
        let Some(skill) = self.skills.iter().find(|skill| skill.name == name) else {
            let available: Vec<&str> = self
                .skills
                .iter()
                .map(|skill| skill.name.as_str())
                .collect();
            return Err(ToolError::failure(format!(
                "Skill \"{name}\" not found. Available skills: {}",
                if available.is_empty() {
                    "none".to_owned()
                } else {
                    available.join(", ")
                }
            )));
        };
        let dir = skill.dir.clone();
        let mut files = tokio::task::spawn_blocking(move || supporting_files(&dir))
            .await
            .map_err(|err| ToolError::failure(format!("cannot list skill files: {err}")))?;
        files.sort();
        files.truncate(MAX_SKILL_FILES);
        let mut out = format!(
            "<skill_content name=\"{}\">\n# Skill: {}\n\n{}\n\nBase directory for this skill: {}\nRelative paths in this skill (e.g., scripts/, reference/) are relative to this base directory.",
            skill.name,
            skill.name,
            skill.body,
            skill.dir.display()
        );
        if !files.is_empty() {
            out.push_str("\n\n<skill_files>");
            for file in &files {
                out.push_str(&format!("\n<file>{}</file>", file.display()));
            }
            out.push_str("\n</skill_files>");
        }
        out.push_str("\n</skill_content>");
        Ok(out)
    }
}

/// Lists supporting files within a bounded walk, excluding `SKILL.md` and symlinks.
fn supporting_files(dir: &Path) -> Vec<PathBuf> {
    fn walk(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
        if depth > 8 {
            return;
        }
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if file_type.is_dir() {
                walk(&path, depth + 1, out);
            } else if file_type.is_file() && path.file_name().is_none_or(|n| n != "SKILL.md") {
                out.push(path);
            }
        }
    }
    let mut out = Vec::new();
    walk(dir, 0, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn loads_skills_and_skips_broken_ones() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("skills");
        std::fs::create_dir(&root).unwrap();
        let root = root.as_path();

        let quoted = root.join("quoted");
        std::fs::create_dir(&quoted).unwrap();
        std::fs::write(
            quoted.join("SKILL.md"),
            "---\nname: quoted-skill\ndescription: \"Handles: colons, and \\\"quotes\\\"\"\n---\n\nDo the thing.\n",
        )
        .unwrap();
        std::fs::create_dir(quoted.join("scripts")).unwrap();
        std::fs::write(quoted.join("scripts/run.sh"), "echo hi").unwrap();
        std::fs::write(quoted.join("reference.md"), "ref").unwrap();

        let multiline = root.join("multiline");
        std::fs::create_dir(&multiline).unwrap();
        std::fs::write(
            multiline.join("SKILL.md"),
            "---\nname: multiline-skill\ndescription: >-\n  First line of the description\n  continues here\n---\nBody text.\n",
        )
        .unwrap();

        let bare = root.join("bare");
        std::fs::create_dir(&bare).unwrap();
        std::fs::write(bare.join("SKILL.md"), "just markdown, no frontmatter").unwrap();

        let nameless = root.join("nameless");
        std::fs::create_dir(&nameless).unwrap();
        std::fs::write(
            nameless.join("SKILL.md"),
            "---\ndescription: nobody\n---\nBody.",
        )
        .unwrap();

        let broken = root.join("broken");
        std::fs::create_dir(&broken).unwrap();
        std::fs::write(
            broken.join("SKILL.md"),
            "---\nname: [unbalanced\n---\nBody.",
        )
        .unwrap();

        std::fs::create_dir(root.join("empty-folder")).unwrap();
        assert!(load_skills(&root.join("nowhere")).await.is_empty());

        // A later folder reusing a name is skipped, not silently shadowing.
        let duplicate = root.join("zz-duplicate");
        std::fs::create_dir(&duplicate).unwrap();
        std::fs::write(
            duplicate.join("SKILL.md"),
            "---\nname: quoted-skill\ndescription: impostor\n---\nOther body.",
        )
        .unwrap();

        let mut expected = vec!["multiline-skill", "quoted-skill"];
        #[cfg(unix)]
        {
            // A symlinked skill folder loads like a real one.
            let outside = dir.path().join("outside-skill");
            std::fs::create_dir(&outside).unwrap();
            std::fs::write(
                outside.join("SKILL.md"),
                "---\nname: linked-skill\n---\nLinked body.",
            )
            .unwrap();
            std::os::unix::fs::symlink(&outside, root.join("linked")).unwrap();
            expected.insert(0, "linked-skill");
        }

        let skills = load_skills(root).await;
        let names: Vec<&str> = skills.iter().map(|skill| skill.name.as_str()).collect();
        assert_eq!(names, expected, "broken and duplicate skills are skipped");

        let quoted = skills
            .iter()
            .find(|skill| skill.name == "quoted-skill")
            .unwrap();
        assert_eq!(quoted.description, "Handles: colons, and \"quotes\"");
        assert_eq!(quoted.body, "Do the thing.");

        let multiline = skills
            .iter()
            .find(|skill| skill.name == "multiline-skill")
            .unwrap();
        assert_eq!(
            multiline.description, "First line of the description continues here",
            "block scalars fold"
        );
    }

    #[test]
    fn parse_skill_shapes() {
        let skill = parse_skill(
            "---\nname: n\ndescription: d\nlicense: MIT\n---\n\nbody\n",
            PathBuf::from("/s"),
        )
        .unwrap();
        assert_eq!(skill.name, "n");
        assert_eq!(skill.description, "d");
        assert_eq!(skill.body, "body");

        let skill = parse_skill("---\r\nname: n\r\n---\r\nbody", PathBuf::from("/s")).unwrap();
        assert_eq!(skill.name, "n");
        assert_eq!(skill.description, "", "description is optional");

        assert!(
            parse_skill("plain text", PathBuf::from("/s"))
                .unwrap_err()
                .contains("frontmatter")
        );
        assert!(
            parse_skill("---\ndescription: x\n---\nb", PathBuf::from("/s"))
                .unwrap_err()
                .contains("no `name`")
        );
        // Numeric names are coerced to strings; malformed YAML remains an error.
        assert_eq!(
            parse_skill("---\nname: 42\n---\nb", PathBuf::from("/s"))
                .unwrap()
                .name,
            "42"
        );
        assert!(parse_skill("---\nname: [unbalanced\n---\nb", PathBuf::from("/s")).is_err());
    }

    #[tokio::test]
    async fn the_skill_tool_renders_content_and_lists_supporting_files() {
        let dir = tempfile::tempdir().unwrap();
        let skill_dir = dir.path().join("deploy");
        std::fs::create_dir(&skill_dir).unwrap();
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: deploy\ndescription: Ships it\n---\nSteps here.",
        )
        .unwrap();
        for i in 0..12 {
            std::fs::write(skill_dir.join(format!("f{i:02}.txt")), "x").unwrap();
        }
        let skills = Arc::new(load_skills(dir.path()).await);
        assert_eq!(skills.len(), 1);
        let tool = SkillTool::new("Load a specialized skill".into(), skills);
        let ctx = ToolCtx::new(dir.path());

        let out = tool
            .call(serde_json::json!({"name": "deploy"}), &ctx)
            .await
            .unwrap();
        assert!(
            out.starts_with("<skill_content name=\"deploy\">\n# Skill: deploy\n\nSteps here.\n"),
            "{out}"
        );
        assert!(
            out.contains(&format!(
                "Base directory for this skill: {}",
                skill_dir.display()
            )),
            "{out}"
        );
        let listed = out.matches("<file>").count();
        assert_eq!(listed, MAX_SKILL_FILES, "at most 10 supporting files");
        assert!(
            out.contains("f00.txt") && !out.contains("f10.txt"),
            "the listing is deterministic: {out}"
        );

        let err = tool
            .call(serde_json::json!({"name": "nope"}), &ctx)
            .await
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("Skill \"nope\" not found. Available skills: deploy"),
            "{err}"
        );
        let err = tool.call(serde_json::json!({}), &ctx).await.unwrap_err();
        assert!(err.to_string().contains("name"), "{err}");
    }

    #[test]
    fn the_prompt_section_lists_skills() {
        let skills = vec![
            Skill {
                name: "a".into(),
                description: "first".into(),
                dir: PathBuf::from("/a"),
                body: String::new(),
            },
            Skill {
                name: "b".into(),
                description: "second".into(),
                dir: PathBuf::from("/b"),
                body: String::new(),
            },
        ];
        let entries = skills_entries(&skills);
        let prompts = crate::prompt::PromptSet::parse(crate::prompt::EMBEDDED_PROMPTS).unwrap();
        let section = prompts.skill_section(&entries);
        assert!(section.contains("<available_skills>"), "{section}");
        assert!(
            section.contains("<name>a</name>\n    <description>first</description>"),
            "{section}"
        );
        assert!(section.contains("<name>b</name>"), "{section}");
    }
}
