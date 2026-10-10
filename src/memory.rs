//! Persistent agent memory: one Markdown file per lesson in a directory shared by
//! every session of an agent, indexed into the system prompt per top-level run,
//! plus the skills consolidation learns from them.

use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::SystemTime;

use serde::Deserialize;
use serde_json::{Value, json};

use crate::Error;
use crate::error::Result;
use crate::llm::ToolSpec;
use crate::skill::split_frontmatter;
use crate::tool::{Tool, ToolCtx, ToolError};

pub const MAX_MEMORIES: usize = 200;
pub const MAX_BODY_BYTES: usize = 4096;
pub const MAX_DESCRIPTION_CHARS: usize = 200;
pub const MAX_LEARNED_SKILLS: usize = 50;
pub const MAX_SKILL_BYTES: usize = 16 * 1024;
/// Consolidation is due once this many memories exist and
/// [`CONSOLIDATE_CHANGED`] of them changed since the last one...
pub const CONSOLIDATE_AT: usize = 30;
pub const CONSOLIDATE_CHANGED: usize = 10;
/// ...or once memory nears its cap or its index outgrows [`INDEX_BYTES`].
pub const CONSOLIDATE_NEAR_CAP: usize = 180;
/// The shortest gap between two consolidation attempts, failed ones included.
pub const CONSOLIDATION_RETRY: std::time::Duration = std::time::Duration::from_secs(60 * 60);
/// Memory-index bytes every system prompt carries, at most.
pub const INDEX_BYTES: usize = 8 * 1024;

const SKILLS: &str = "skills";
const ARCHIVE: &str = ".archive";
const CONSOLIDATED: &str = ".consolidated";
const ATTEMPTED: &str = ".consolidation-attempt";
const LOCK: &str = ".lock";

/// What a memory directory holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// `<name>.md`, indexed into every run's system prompt.
    Memory,
    /// `skills/<name>/SKILL.md`, listed and loaded like the embedder's skills.
    Skill,
}

impl Kind {
    fn noun(self) -> &'static str {
        match self {
            Kind::Memory => "memory",
            Kind::Skill => "learned skill",
        }
    }

    fn max_entries(self) -> usize {
        match self {
            Kind::Memory => MAX_MEMORIES,
            Kind::Skill => MAX_LEARNED_SKILLS,
        }
    }

    fn max_body(self) -> usize {
        match self {
            Kind::Memory => MAX_BODY_BYTES,
            Kind::Skill => MAX_SKILL_BYTES,
        }
    }
}

/// One index line: an entry's name and its one-line description.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryEntry {
    pub name: String,
    pub description: String,
}

#[derive(Debug, Deserialize)]
struct Frontmatter {
    description: Option<String>,
}

/// A memory directory owned by one process; writes are serialized within it.
#[derive(Debug)]
pub struct Memory {
    dir: PathBuf,
    write: tokio::sync::Mutex<()>,
    // Holds the OS lock that keeps other processes out for the store's lifetime.
    _owner: std::fs::File,
}

fn private_dir(dir: &Path) -> std::io::Result<()> {
    let mut builder = std::fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(dir)
}

impl Memory {
    /// Creates the directory and its learned-skills folder (0700 on Unix) when
    /// missing and pins it to an absolute path.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        private_dir(&dir.join(SKILLS)).map_err(|err| {
            Error::Config(format!(
                "cannot create the memory directory {}: {err}",
                dir.display()
            ))
        })?;
        let dir = std::path::absolute(dir).map_err(|err| {
            Error::Config(format!(
                "cannot resolve the memory directory {}: {err}",
                dir.display()
            ))
        })?;
        let mut options = std::fs::OpenOptions::new();
        options.read(true).write(true).create(true).truncate(false);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let owner = options.open(dir.join(LOCK))?;
        owner.try_lock().map_err(|_| {
            Error::Config(format!(
                "the memory directory {} is already owned by another agent or process",
                dir.display()
            ))
        })?;
        Ok(Self {
            dir,
            write: tokio::sync::Mutex::new(()),
            _owner: owner,
        })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Where learned skills live, in the `<name>/SKILL.md` layout of a skills dir.
    pub fn skills_dir(&self) -> PathBuf {
        self.dir.join(SKILLS)
    }

    /// Valid memories, most recently changed first, at most [`MAX_MEMORIES`];
    /// invalid files are skipped with a warning.
    pub async fn index(&self) -> Vec<MemoryEntry> {
        self.list(Kind::Memory).await
    }

    /// Valid entries of one kind, most recently changed first, at most the kind's
    /// cap.
    pub async fn list(&self, kind: Kind) -> Vec<MemoryEntry> {
        let root = match kind {
            Kind::Memory => self.dir.clone(),
            Kind::Skill => self.skills_dir(),
        };
        let mut entries = match tokio::fs::read_dir(&root).await {
            Ok(entries) => entries,
            Err(err) => {
                tracing::warn!(dir = %root.display(), "cannot read the memory directory: {err}");
                return Vec::new();
            }
        };
        let mut index = Vec::new();
        while let Ok(Some(entry)) = entries.next_entry().await {
            let file_name = entry.file_name();
            let Some(name) = file_name
                .to_str()
                .and_then(|file| match kind {
                    Kind::Memory => file.strip_suffix(".md"),
                    Kind::Skill => Some(file),
                })
                .filter(|name| valid_name(name))
            else {
                continue;
            };
            match self.read(kind, name).await {
                Ok(Some(text)) => match parse(kind, &text) {
                    Ok((description, _)) => {
                        let changed = modified(&self.path(kind, name))
                            .await
                            .unwrap_or(SystemTime::UNIX_EPOCH);
                        index.push((
                            changed,
                            MemoryEntry {
                                name: name.to_owned(),
                                description,
                            },
                        ));
                    }
                    Err(err) => tracing::warn!(entry = name, "skipping a {}: {err}", kind.noun()),
                },
                Ok(None) => {}
                Err(err) => tracing::warn!(entry = name, "skipping a {}: {err}", kind.noun()),
            }
        }
        index.sort_by(|(a_changed, a), (b_changed, b)| {
            b_changed.cmp(a_changed).then_with(|| a.name.cmp(&b.name))
        });
        index.truncate(kind.max_entries());
        index.into_iter().map(|(_, entry)| entry).collect()
    }

    /// Every entry of one kind in full, for consolidation.
    pub async fn contents(&self, kind: Kind) -> Vec<(String, String)> {
        let mut contents = Vec::new();
        for entry in self.list(kind).await {
            if let Ok(Some(text)) = self.read(kind, &entry.name).await {
                contents.push((entry.name, text));
            }
        }
        contents
    }

    /// Whether enough memories changed since the last consolidation, memory nears
    /// its cap, or its index outgrows [`INDEX_BYTES`]; never within
    /// [`CONSOLIDATION_RETRY`] of the last attempt.
    pub async fn consolidation_due(&self) -> bool {
        if modified(&self.dir.join(ATTEMPTED))
            .await
            .and_then(|time| time.elapsed().ok())
            .is_some_and(|elapsed| elapsed < CONSOLIDATION_RETRY)
        {
            return false;
        }
        let index = self.index().await;
        if index.len() >= CONSOLIDATE_NEAR_CAP || index_entries(&index).len() > INDEX_BYTES {
            return true;
        }
        if index.len() < CONSOLIDATE_AT {
            return false;
        }
        let since = modified(&self.dir.join(CONSOLIDATED))
            .await
            .unwrap_or(SystemTime::UNIX_EPOCH);
        let mut changed = 0;
        for entry in &index {
            if modified(&self.path(Kind::Memory, &entry.name))
                .await
                .is_some_and(|time| time > since)
            {
                changed += 1;
            }
        }
        changed >= CONSOLIDATE_CHANGED
    }

    /// Records a consolidation attempt, which defers the next by
    /// [`CONSOLIDATION_RETRY`] whether or not it succeeds.
    pub async fn mark_attempt(&self) -> Result<()> {
        tokio::fs::write(self.dir.join(ATTEMPTED), chrono::Utc::now().to_rfc3339()).await?;
        Ok(())
    }

    /// Records a finished consolidation; later changes count toward the next.
    pub async fn mark_consolidated(&self) -> Result<()> {
        tokio::fs::write(self.dir.join(CONSOLIDATED), chrono::Utc::now().to_rfc3339()).await?;
        Ok(())
    }

    fn path(&self, kind: Kind, name: &str) -> PathBuf {
        match kind {
            Kind::Memory => self.dir.join(format!("{name}.md")),
            Kind::Skill => self.skills_dir().join(name).join("SKILL.md"),
        }
    }

    /// The entry's text, or `None` when absent; anything but a regular file (or,
    /// for a skill, a real folder) is an error.
    async fn read(&self, kind: Kind, name: &str) -> std::result::Result<Option<String>, String> {
        let noun = kind.noun();
        if kind == Kind::Skill {
            match tokio::fs::symlink_metadata(self.skills_dir().join(name)).await {
                Ok(meta) if meta.is_dir() => {}
                Ok(_) => return Err(format!("{noun} `{name}` is not a regular folder")),
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
                Err(err) => return Err(format!("cannot read {noun} `{name}`: {err}")),
            }
        }
        let path = self.path(kind, name);
        match tokio::fs::symlink_metadata(&path).await {
            Ok(meta) if meta.is_file() => tokio::fs::read_to_string(&path)
                .await
                .map(Some)
                .map_err(|err| format!("cannot read {noun} `{name}`: {err}")),
            Ok(_) => Err(format!("{noun} `{name}` is not a regular file")),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(err) => Err(format!("cannot read {noun} `{name}`: {err}")),
        }
    }

    /// Writes through a temporary file so readers never see a partial entry.
    async fn write(&self, kind: Kind, name: &str, text: &str) -> std::result::Result<(), String> {
        use tokio::io::AsyncWriteExt;
        let failed = |err: std::io::Error| format!("cannot write {} `{name}`: {err}", kind.noun());
        let path = self.path(kind, name);
        let folder = path.parent().expect("entries live in a folder");
        if kind == Kind::Skill {
            private_dir(folder).map_err(failed)?;
        }
        let temp = folder.join(format!(".{name}.tmp"));
        let mut options = tokio::fs::OpenOptions::new();
        options.write(true).create(true).truncate(true);
        #[cfg(unix)]
        options
            .mode(0o600)
            .custom_flags(nix::fcntl::OFlag::O_NOFOLLOW.bits());
        let mut file = options.open(&temp).await.map_err(failed)?;
        file.write_all(text.as_bytes()).await.map_err(failed)?;
        file.sync_all().await.map_err(failed)?;
        tokio::fs::rename(&temp, path).await.map_err(failed)
    }

    /// Moves an entry to `.archive/` instead of deleting it.
    async fn archive(&self, kind: Kind, name: &str) -> std::result::Result<(), String> {
        let failed = |err: std::io::Error| format!("cannot delete {} `{name}`: {err}", kind.noun());
        let archive = self.dir.join(ARCHIVE);
        private_dir(&archive).map_err(failed)?;
        let stamp = chrono::Utc::now().format("%Y%m%dT%H%M%S%.3f");
        let prefix = match kind {
            Kind::Memory => "",
            Kind::Skill => "skill-",
        };
        tokio::fs::rename(
            self.path(kind, name),
            archive.join(format!("{prefix}{name}.{stamp}.md")),
        )
        .await
        .map_err(failed)?;
        if kind == Kind::Skill {
            // Only an emptied folder goes; anything else the folder held stays.
            let _ = tokio::fs::remove_dir(self.skills_dir().join(name)).await;
        }
        Ok(())
    }
}

async fn modified(path: &Path) -> Option<SystemTime> {
    tokio::fs::metadata(path).await.ok()?.modified().ok()
}

/// The value of the memory section's `{{index}}` placeholder: the most recent
/// entries within [`INDEX_BYTES`], then a count of the rest.
pub fn prompt_index(index: &[MemoryEntry]) -> String {
    let mut lines = Vec::new();
    let mut used = 0;
    for entry in index {
        let line = format!("- {}: {}", entry.name, entry.description);
        if used + line.len() + 1 > INDEX_BYTES {
            break;
        }
        used += line.len() + 1;
        lines.push(line);
    }
    if lines.is_empty() && index.is_empty() {
        return "(no memories yet)".to_owned();
    }
    let rest = index.len() - lines.len();
    if rest > 0 {
        lines.push(format!(
            "- ... {rest} older memories; `view` without a name lists all"
        ));
    }
    lines.join("\n")
}

/// Every entry, one line each, most recent first.
pub fn index_entries(index: &[MemoryEntry]) -> String {
    if index.is_empty() {
        return "(no memories yet)".to_owned();
    }
    index
        .iter()
        .map(|entry| format!("- {}: {}", entry.name, entry.description))
        .collect::<Vec<_>>()
        .join("\n")
}

fn valid_name(name: &str) -> bool {
    static NAME: OnceLock<regex::Regex> = OnceLock::new();
    NAME.get_or_init(|| {
        regex::Regex::new(r"^[a-z0-9][a-z0-9-]{0,63}$").expect("the name pattern compiles")
    })
    .is_match(name)
}

fn check_name(kind: Kind, name: &str) -> std::result::Result<(), String> {
    if valid_name(name) {
        Ok(())
    } else {
        Err(format!(
            "invalid {} name `{name}`: use lowercase letters, digits and hyphens, at most 64 characters",
            kind.noun()
        ))
    }
}

fn check(kind: Kind, description: &str, body: &str) -> std::result::Result<(), String> {
    if description.trim().is_empty() || description.contains('\n') {
        return Err("the description must be one nonempty line".to_owned());
    }
    if description.chars().count() > MAX_DESCRIPTION_CHARS {
        return Err(format!(
            "the description exceeds {MAX_DESCRIPTION_CHARS} characters"
        ));
    }
    if body.trim().is_empty() {
        return Err("the body is empty".to_owned());
    }
    if body.len() > kind.max_body() {
        return Err(match kind {
            Kind::Memory => {
                format!("the body exceeds {MAX_BODY_BYTES} bytes; keep one lesson per memory")
            }
            Kind::Skill => format!("the body exceeds {MAX_SKILL_BYTES} bytes"),
        });
    }
    Ok(())
}

/// Splits an entry's file into its description and body.
fn parse(kind: Kind, text: &str) -> std::result::Result<(String, String), String> {
    let Some((yaml, body)) = split_frontmatter(text) else {
        return Err("it does not start with `---` frontmatter".to_owned());
    };
    let meta: Frontmatter =
        serde_saphyr::from_str(yaml).map_err(|err| format!("invalid frontmatter YAML: {err}"))?;
    let description = meta
        .description
        .map(|description| description.trim().to_owned())
        .unwrap_or_default();
    let body = body.trim().to_owned();
    check(kind, &description, &body)?;
    Ok((description, body))
}

/// The canonical file text; strings are JSON, which is valid YAML. Skills carry
/// the `name` skill loading requires.
fn render(kind: Kind, name: &str, description: &str, body: &str, source: Option<&str>) -> String {
    let name = match kind {
        Kind::Memory => String::new(),
        Kind::Skill => format!("name: {}\n", Value::String(name.to_owned())),
    };
    let source = source
        .map(|source| format!("source: {}\n", Value::String(source.to_owned())))
        .unwrap_or_default();
    format!(
        "---\n{name}description: {}\nupdated: {}\n{source}---\n{}\n",
        Value::String(description.trim().to_owned()),
        chrono::Utc::now().format("%Y-%m-%d"),
        body.trim()
    )
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Command {
    View,
    Create,
    StrReplace,
    Delete,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Args {
    command: Command,
    name: Option<String>,
    description: Option<String>,
    body: Option<String>,
    old: Option<String>,
    new: Option<String>,
}

fn required(value: Option<String>, field: &str) -> std::result::Result<String, ToolError> {
    value.ok_or_else(|| ToolError::failure(format!("invalid arguments: missing field `{field}`")))
}

/// The runtime-owned `memory` tool, or consolidation's `learned_skill` tool;
/// subagents get a read-only `memory`.
pub struct MemoryTool {
    memory: Arc<Memory>,
    kind: Kind,
    read_only: bool,
    source: Option<String>,
    /// Names a learned skill may not take: the embedder's skills.
    taken: Vec<String>,
    spec: ToolSpec,
}

impl MemoryTool {
    /// The `memory` tool, with its description from the run's prompt snapshot.
    pub fn new(description: String, memory: Arc<Memory>, read_only: bool) -> Self {
        Self::build("memory", Kind::Memory, description, memory, read_only)
    }

    /// The `learned_skill` tool; `taken` holds the embedder's skill names.
    pub fn learned_skills(description: String, memory: Arc<Memory>, taken: Vec<String>) -> Self {
        let mut tool = Self::build("learned_skill", Kind::Skill, description, memory, false);
        tool.taken = taken;
        tool
    }

    fn build(
        tool_name: &str,
        kind: Kind,
        description: String,
        memory: Arc<Memory>,
        read_only: bool,
    ) -> Self {
        let commands: &[&str] = if read_only {
            &["view"]
        } else {
            &["view", "create", "str_replace", "delete"]
        };
        let mut properties = json!({
            "command": {"type": "string", "enum": commands},
            "name": {"type": "string", "description": "The name, e.g. `orders-schema`"},
        });
        if !read_only {
            let body = match kind {
                Kind::Memory => "create: the lesson, then **Why:** and **How to apply:** lines",
                Kind::Skill => "create: the skill's instructions in Markdown",
            };
            let extra = json!({
                "description": {"type": "string", "description": "create: one line shown in the index"},
                "body": {"type": "string", "description": body},
                "old": {"type": "string", "description": "str_replace: the exact text to replace"},
                "new": {"type": "string", "description": "str_replace: the replacement text"},
            });
            properties
                .as_object_mut()
                .expect("properties is an object")
                .extend(extra.as_object().expect("extra is an object").clone());
        }
        let spec = ToolSpec::function(
            tool_name,
            description,
            json!({
                "type": "object",
                "properties": properties,
                "required": ["command"],
                "additionalProperties": false
            }),
        );
        Self {
            memory,
            kind,
            read_only,
            source: None,
            taken: Vec::new(),
            spec,
        }
    }

    /// Records where entries written through this tool came from, e.g. `session 12`.
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    async fn view(&self, name: Option<String>) -> std::result::Result<String, String> {
        let Some(name) = name else {
            let entries = self.memory.list(self.kind).await;
            return Ok(match self.kind {
                Kind::Memory => index_entries(&entries),
                Kind::Skill if entries.is_empty() => "(no learned skills yet)".to_owned(),
                Kind::Skill => index_entries(&entries),
            });
        };
        check_name(self.kind, &name)?;
        self.memory
            .read(self.kind, &name)
            .await?
            .ok_or_else(|| format!("{} `{name}` does not exist", self.kind.noun()))
    }

    async fn create(
        &self,
        name: String,
        description: String,
        body: String,
    ) -> std::result::Result<String, String> {
        let (kind, noun) = (self.kind, self.kind.noun());
        check_name(kind, &name)?;
        check(kind, &description, &body)?;
        if self.taken.contains(&name) {
            return Err(format!("`{name}` is the name of an installed skill"));
        }
        let _write = self.memory.write.lock().await;
        if self.memory.read(kind, &name).await?.is_some() {
            return Err(format!(
                "{noun} `{name}` already exists; update it with str_replace or delete it first"
            ));
        }
        if self.memory.list(kind).await.len() >= kind.max_entries() {
            return Err(format!(
                "there are {} {noun} entries; merge or delete some first",
                kind.max_entries()
            ));
        }
        let text = render(kind, &name, &description, &body, self.source.as_deref());
        self.memory.write(kind, &name, &text).await?;
        Ok(format!("Created {noun} `{name}`."))
    }

    async fn str_replace(
        &self,
        name: String,
        old: String,
        new: String,
    ) -> std::result::Result<String, String> {
        let (kind, noun) = (self.kind, self.kind.noun());
        check_name(kind, &name)?;
        if old.is_empty() {
            return Err("`old` is empty".to_owned());
        }
        let _write = self.memory.write.lock().await;
        let text = self
            .memory
            .read(kind, &name)
            .await?
            .ok_or_else(|| format!("{noun} `{name}` does not exist"))?;
        match text.matches(old.as_str()).count() {
            0 => return Err(format!("`old` does not occur in {noun} `{name}`")),
            1 => {}
            count => {
                return Err(format!(
                    "`old` occurs {count} times in {noun} `{name}`; include more context"
                ));
            }
        }
        let (description, body) = parse(kind, &text.replacen(old.as_str(), &new, 1))
            .map_err(|err| format!("the edit would leave {noun} `{name}` invalid: {err}"))?;
        let text = render(kind, &name, &description, &body, self.source.as_deref());
        self.memory.write(kind, &name, &text).await?;
        Ok(format!("Updated {noun} `{name}`."))
    }

    async fn delete(&self, name: String) -> std::result::Result<String, String> {
        let (kind, noun) = (self.kind, self.kind.noun());
        check_name(kind, &name)?;
        let _write = self.memory.write.lock().await;
        if self.memory.read(kind, &name).await?.is_none() {
            return Err(format!("{noun} `{name}` does not exist"));
        }
        self.memory.archive(kind, &name).await?;
        Ok(format!("Deleted {noun} `{name}`."))
    }
}

#[async_trait::async_trait]
impl Tool for MemoryTool {
    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }

    async fn call(&self, args: Value, _ctx: &ToolCtx) -> std::result::Result<String, ToolError> {
        let args: Args = serde_json::from_value(args)
            .map_err(|err| ToolError::failure(format!("invalid arguments: {err}")))?;
        if self.read_only && !matches!(args.command, Command::View) {
            return Err(ToolError::failure(
                "memory is read-only here; only `view` is available",
            ));
        }
        let result = match args.command {
            Command::View => self.view(args.name).await,
            Command::Create => {
                self.create(
                    required(args.name, "name")?,
                    required(args.description, "description")?,
                    required(args.body, "body")?,
                )
                .await
            }
            Command::StrReplace => {
                self.str_replace(
                    required(args.name, "name")?,
                    required(args.old, "old")?,
                    required(args.new, "new")?,
                )
                .await
            }
            Command::Delete => self.delete(required(args.name, "name")?).await,
        };
        result.map_err(ToolError::Failure)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool(dir: &Path, read_only: bool) -> MemoryTool {
        MemoryTool::new(
            "Manage memories.".into(),
            Arc::new(Memory::open(dir).unwrap()),
            read_only,
        )
    }

    async fn call(tool: &MemoryTool, args: Value) -> std::result::Result<String, String> {
        tool.call(args, &ToolCtx::new("."))
            .await
            .map_err(|err| err.to_string())
    }

    fn create(name: &str) -> Value {
        json!({
            "command": "create",
            "name": name,
            "description": "orders table uses placed_on",
            "body": "Use `placed_on`.\n**Why:** no created_at.\n**How to apply:** time filters."
        })
    }

    #[tokio::test]
    async fn create_view_replace_and_delete() {
        let dir = tempfile::tempdir().unwrap();
        let tool = tool(dir.path(), false);
        assert_eq!(
            call(&tool, json!({"command": "view"})).await.unwrap(),
            "(no memories yet)"
        );

        call(&tool, create("orders-schema")).await.unwrap();
        assert_eq!(
            call(&tool, json!({"command": "view"})).await.unwrap(),
            "- orders-schema: orders table uses placed_on"
        );
        let text = call(&tool, json!({"command": "view", "name": "orders-schema"}))
            .await
            .unwrap();
        assert!(text.starts_with("---\ndescription: \"orders table uses placed_on\"\nupdated: "));
        assert!(text.contains("**How to apply:** time filters."), "{text}");

        call(
            &tool,
            json!({"command": "str_replace", "name": "orders-schema",
                   "old": "uses placed_on\"", "new": "uses ordered_at\""}),
        )
        .await
        .unwrap();
        assert_eq!(
            tool.memory.index().await,
            vec![MemoryEntry {
                name: "orders-schema".into(),
                description: "orders table uses ordered_at".into()
            }]
        );

        call(&tool, json!({"command": "delete", "name": "orders-schema"}))
            .await
            .unwrap();
        assert!(tool.memory.index().await.is_empty());
        let archived: Vec<String> = std::fs::read_dir(dir.path().join(".archive"))
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect();
        assert_eq!(archived.len(), 1);
        assert!(archived[0].starts_with("orders-schema."), "{archived:?}");
        assert!(
            call(&tool, json!({"command": "delete", "name": "orders-schema"}))
                .await
                .unwrap_err()
                .contains("does not exist")
        );
    }

    #[tokio::test]
    async fn writes_record_their_source() {
        let dir = tempfile::tempdir().unwrap();
        let tool = tool(dir.path(), false).with_source("session 12");
        call(&tool, create("a")).await.unwrap();
        let text = std::fs::read_to_string(dir.path().join("a.md")).unwrap();
        assert!(text.contains("\nsource: \"session 12\"\n---\n"), "{text}");
        assert_eq!(tool.memory.index().await.len(), 1);
    }

    #[tokio::test]
    async fn create_refuses_to_overwrite() {
        let dir = tempfile::tempdir().unwrap();
        let tool = tool(dir.path(), false);
        call(&tool, create("a")).await.unwrap();
        let err = call(&tool, create("a")).await.unwrap_err();
        assert!(err.contains("already exists"), "{err}");
    }

    #[tokio::test]
    async fn str_replace_needs_one_match_and_a_valid_result() {
        let dir = tempfile::tempdir().unwrap();
        let tool = tool(dir.path(), false);
        call(&tool, create("a")).await.unwrap();
        let replace = |old: &str, new: &str| json!({"command": "str_replace", "name": "a", "old": old, "new": new});
        let err = call(&tool, replace("missing", "x")).await.unwrap_err();
        assert!(err.contains("does not occur"), "{err}");
        let err = call(&tool, replace("**", "x")).await.unwrap_err();
        assert!(err.contains("occurs 4 times"), "{err}");
        let err = call(&tool, replace("---\ndescription", "description"))
            .await
            .unwrap_err();
        assert!(err.contains("invalid"), "{err}");
    }

    #[tokio::test]
    async fn invalid_names_and_contents_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let tool = tool(dir.path(), false);
        for name in ["../escape", "Upper", "-lead", "a/b", "a.md", ""] {
            let err = call(&tool, create(name)).await.unwrap_err();
            assert!(err.contains("invalid memory name"), "{name}: {err}");
        }
        let mut multiline = create("a");
        multiline["description"] = json!("two\nlines");
        assert!(
            call(&tool, multiline)
                .await
                .unwrap_err()
                .contains("one nonempty line")
        );
        let mut large = create("a");
        large["body"] = json!("x".repeat(MAX_BODY_BYTES + 1));
        assert!(call(&tool, large).await.unwrap_err().contains("exceeds"));
        let err = call(&tool, json!({"command": "create", "name": "a"}))
            .await
            .unwrap_err();
        assert!(err.contains("missing field `description`"), "{err}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn symlinks_are_neither_read_nor_replaced() {
        let dir = tempfile::tempdir().unwrap();
        let outside = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(outside.path(), "---\ndescription: secret\n---\nsecret").unwrap();
        std::os::unix::fs::symlink(outside.path(), dir.path().join("link.md")).unwrap();
        let tool = tool(dir.path(), false);
        assert!(tool.memory.index().await.is_empty());
        for args in [
            json!({"command": "view", "name": "link"}),
            create("link"),
            json!({"command": "delete", "name": "link"}),
        ] {
            let err = call(&tool, args).await.unwrap_err();
            assert!(err.contains("not a regular file"), "{err}");
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn memories_are_private() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let tool = tool(&dir.path().join("memory"), false);
        call(&tool, create("a")).await.unwrap();
        let mode = |path: &Path| std::fs::metadata(path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode(&dir.path().join("memory")), 0o700);
        assert_eq!(mode(&dir.path().join("memory/a.md")), 0o600);
    }

    #[tokio::test]
    async fn the_entry_cap_holds() {
        let dir = tempfile::tempdir().unwrap();
        for i in 0..MAX_MEMORIES {
            std::fs::write(
                dir.path().join(format!("m{i}.md")),
                render(Kind::Memory, "m", "a lesson", "the body", None),
            )
            .unwrap();
        }
        let tool = tool(dir.path(), false);
        let err = call(&tool, create("one-more")).await.unwrap_err();
        assert!(err.contains("merge or delete"), "{err}");
    }

    #[tokio::test]
    async fn the_index_skips_invalid_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("good.md"),
            render(Kind::Memory, "m", "kept", "body", None),
        )
        .unwrap();
        std::fs::write(dir.path().join("bare.md"), "no frontmatter").unwrap();
        std::fs::write(
            dir.path().join("Bad Name.md"),
            render(Kind::Memory, "m", "x", "y", None),
        )
        .unwrap();
        std::fs::write(dir.path().join("notes.txt"), "ignored").unwrap();
        let memory = Memory::open(dir.path()).unwrap();
        assert_eq!(
            memory.index().await,
            vec![MemoryEntry {
                name: "good".into(),
                description: "kept".into()
            }]
        );
    }

    #[tokio::test]
    async fn the_read_only_tool_only_views() {
        let dir = tempfile::tempdir().unwrap();
        let tool = tool(dir.path(), true);
        let spec = serde_json::to_value(tool.spec()).unwrap();
        assert_eq!(
            spec.pointer("/function/parameters/properties/command/enum"),
            Some(&json!(["view"]))
        );
        assert!(
            spec.pointer("/function/parameters/properties/body")
                .is_none()
        );
        let err = call(&tool, create("a")).await.unwrap_err();
        assert!(err.contains("read-only"), "{err}");
        assert!(call(&tool, json!({"command": "view"})).await.is_ok());
    }

    fn skill_tool(dir: &Path) -> MemoryTool {
        MemoryTool::learned_skills(
            "Manage learned skills.".into(),
            Arc::new(Memory::open(dir).unwrap()),
            vec!["deploy".into()],
        )
    }

    #[tokio::test]
    async fn learned_skills_load_as_skills_and_archive_on_delete() {
        let dir = tempfile::tempdir().unwrap();
        let tool = skill_tool(dir.path());
        assert_eq!(
            call(&tool, json!({"command": "view"})).await.unwrap(),
            "(no learned skills yet)"
        );
        call(
            &tool,
            json!({"command": "create", "name": "release-notes",
                   "description": "Drafts release notes from merged PRs",
                   "body": "1. List merged PRs.\n2. Group them."}),
        )
        .await
        .unwrap();
        let skills = crate::skill::load_skills(&tool.memory.skills_dir()).await;
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "release-notes");
        assert_eq!(
            skills[0].description,
            "Drafts release notes from merged PRs"
        );
        assert!(
            tool.memory.index().await.is_empty(),
            "skills are not memories"
        );

        let err = call(
            &tool,
            json!({"command": "create", "name": "deploy", "description": "x", "body": "y"}),
        )
        .await
        .unwrap_err();
        assert!(err.contains("installed skill"), "{err}");

        call(&tool, json!({"command": "delete", "name": "release-notes"}))
            .await
            .unwrap();
        assert!(!tool.memory.skills_dir().join("release-notes").exists());
        assert!(
            crate::skill::load_skills(&tool.memory.skills_dir())
                .await
                .is_empty()
        );
        let archived = std::fs::read_dir(dir.path().join(".archive"))
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .file_name();
        assert!(
            archived
                .to_str()
                .unwrap()
                .starts_with("skill-release-notes."),
            "{archived:?}"
        );
    }

    #[tokio::test]
    async fn consolidation_is_due_on_recent_changes_or_near_the_cap() {
        let dir = tempfile::tempdir().unwrap();
        let memory = Memory::open(dir.path()).unwrap();
        let write = |i: usize| {
            std::fs::write(
                dir.path().join(format!("m{i}.md")),
                render(Kind::Memory, "m", "a lesson", "the body", None),
            )
            .unwrap()
        };
        for i in 0..CONSOLIDATE_AT - 1 {
            write(i);
        }
        assert!(!memory.consolidation_due().await, "too few memories");
        write(CONSOLIDATE_AT);
        assert!(memory.consolidation_due().await);

        memory.mark_consolidated().await.unwrap();
        assert!(!memory.consolidation_due().await, "nothing changed since");
        // Later changes count toward the next consolidation.
        let later = SystemTime::now() + std::time::Duration::from_secs(5);
        for i in 0..CONSOLIDATE_CHANGED {
            write(i);
            let file = std::fs::File::options()
                .write(true)
                .open(dir.path().join(format!("m{i}.md")))
                .unwrap();
            file.set_modified(later).unwrap();
        }
        assert!(memory.consolidation_due().await);

        memory.mark_consolidated().await.unwrap();
        for i in CONSOLIDATE_AT..=CONSOLIDATE_NEAR_CAP {
            write(i);
        }
        let marker = std::fs::File::options()
            .write(true)
            .open(dir.path().join(CONSOLIDATED))
            .unwrap();
        marker
            .set_modified(SystemTime::now() + std::time::Duration::from_secs(60))
            .unwrap();
        assert!(memory.consolidation_due().await, "near the cap");
    }

    #[tokio::test]
    async fn the_index_skips_the_archive_and_learned_skills() {
        let dir = tempfile::tempdir().unwrap();
        let tool = tool(dir.path(), false);
        call(&tool, create("kept")).await.unwrap();
        call(&tool, create("gone")).await.unwrap();
        call(&tool, json!({"command": "delete", "name": "gone"}))
            .await
            .unwrap();
        let names: Vec<String> = tool
            .memory
            .index()
            .await
            .into_iter()
            .map(|entry| entry.name)
            .collect();
        assert_eq!(names, ["kept"]);
    }

    #[test]
    fn the_prompt_index_keeps_the_newest_within_its_budget() {
        let entry = |i: usize| MemoryEntry {
            name: format!("m{i}"),
            description: "x".repeat(150),
        };
        assert_eq!(prompt_index(&[]), "(no memories yet)");
        let small: Vec<MemoryEntry> = (0..3).map(entry).collect();
        assert_eq!(prompt_index(&small), index_entries(&small));

        let large: Vec<MemoryEntry> = (0..100).map(entry).collect();
        let rendered = prompt_index(&large);
        assert!(rendered.len() <= INDEX_BYTES + 80, "{}", rendered.len());
        assert!(rendered.starts_with("- m0: "), "newest first");
        let shown = rendered.lines().count() - 1;
        assert!(
            rendered.ends_with(&format!(
                "- ... {} older memories; `view` without a name lists all",
                100 - shown
            )),
            "{rendered}"
        );
    }

    #[tokio::test]
    async fn the_index_lists_the_most_recently_changed_first() {
        let dir = tempfile::tempdir().unwrap();
        let tool = tool(dir.path(), false);
        call(&tool, create("older")).await.unwrap();
        call(&tool, create("newer")).await.unwrap();
        let past = SystemTime::now() - std::time::Duration::from_secs(60);
        std::fs::File::options()
            .write(true)
            .open(dir.path().join("older.md"))
            .unwrap()
            .set_modified(past)
            .unwrap();
        let names: Vec<String> = tool
            .memory
            .index()
            .await
            .into_iter()
            .map(|entry| entry.name)
            .collect();
        assert_eq!(names, ["newer", "older"]);
    }

    #[tokio::test]
    async fn an_oversized_index_is_due_but_attempts_back_off() {
        let dir = tempfile::tempdir().unwrap();
        let memory = Memory::open(dir.path()).unwrap();
        for i in 0..50 {
            std::fs::write(
                dir.path().join(format!("m{i}.md")),
                render(Kind::Memory, "m", &"d".repeat(190), "the body", None),
            )
            .unwrap();
        }
        memory.mark_consolidated().await.unwrap();
        let later = SystemTime::now() + std::time::Duration::from_secs(60);
        std::fs::File::options()
            .write(true)
            .open(dir.path().join(CONSOLIDATED))
            .unwrap()
            .set_modified(later)
            .unwrap();
        assert!(
            memory.consolidation_due().await,
            "the index outgrew its budget"
        );
        memory.mark_attempt().await.unwrap();
        assert!(
            !memory.consolidation_due().await,
            "an attempt defers the next"
        );
    }

    #[test]
    fn one_owner_per_memory_directory() {
        let dir = tempfile::tempdir().unwrap();
        let first = Memory::open(dir.path()).unwrap();
        let err = Memory::open(dir.path()).unwrap_err();
        assert!(err.to_string().contains("already owned"), "{err}");
        drop(first);
        Memory::open(dir.path()).unwrap();
    }
}
