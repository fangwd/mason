//! Builtin read, write, edit, bash, grep, and glob tools with shared search helpers.

use std::path::{Path, PathBuf};

use crate::tool::ToolCtx;
use crate::tool::{Tool, ToolError, Typed};
use tokio_util::sync::CancellationToken;

mod bash;
mod edit;
mod glob;
mod grep;
mod read;
mod write;

pub use bash::BashTool;
pub use edit::EditTool;
pub use glob::GlobTool;
pub use grep::GrepTool;
pub use read::ReadTool;
pub use write::WriteTool;

/// Result-list bound for the search tools.
pub(super) const MAX_RESULTS: usize = 100;

pub fn builtin_tools() -> Vec<Box<dyn Tool>> {
    vec![
        Box::new(Typed(ReadTool)),
        Box::new(Typed(WriteTool)),
        Box::new(Typed(EditTool)),
        Box::new(Typed(BashTool)),
        Box::new(Typed(GrepTool)),
        Box::new(Typed(GlobTool)),
        Box::new(Typed(crate::tool::process::ProcessTool)),
    ]
}

/// One search hit, display-ready.
pub(super) struct Hit {
    pub path: String,
    pub line: usize,
    pub text: String,
}

/// Renders bounded search results with narrowing guidance when truncated.
pub(super) fn limited(lines: Vec<String>, what: &str) -> String {
    if lines.is_empty() {
        return format!("No {what} found");
    }
    let total = lines.len();
    let mut out = lines
        .into_iter()
        .take(MAX_RESULTS)
        .collect::<Vec<_>>()
        .join("\n");
    if total > MAX_RESULTS {
        out.push_str(&format!(
            "\n\n(Showing {MAX_RESULTS} of at least {total} {what}; narrow the search.)"
        ));
    }
    out
}

/// Searches hidden files and excludes `.git`. Unless `include_ignored`, it
/// respects ignore files whether or not the root is inside a Git repository.
pub(super) fn walk_files(
    root: &Path,
    include_ignored: bool,
    token: &CancellationToken,
) -> impl Iterator<Item = PathBuf> {
    let token = token.clone();
    let walk_root = root.to_path_buf();
    ignore::WalkBuilder::new(root)
        .standard_filters(!include_ignored)
        .hidden(false)
        .follow_links(false)
        .require_git(false)
        .filter_entry(move |entry| {
            !token.is_cancelled() && !has_git_component(&walk_root, entry.path())
        })
        .build()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_type().is_some_and(|type_| type_.is_file()))
        .map(|entry| entry.into_path())
}

/// Opens a file for reading only if the opened handle is a regular file. The
/// open itself cannot block (`O_NONBLOCK` on Unix), so a FIFO or device, even
/// one swapped in after a path check, is refused instead of hanging a thread.
pub(super) async fn open_regular(path: &Path) -> Result<tokio::fs::File, ToolError> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(nix::fcntl::OFlag::O_NONBLOCK.bits());
    }
    let file = options
        .open(path)
        .map_err(|err| ToolError::failure(format!("cannot read {}: {err}", path.display())))?;
    let meta = file
        .metadata()
        .map_err(|err| ToolError::failure(format!("cannot stat {}: {err}", path.display())))?;
    if !meta.is_file() {
        return Err(ToolError::failure(format!(
            "{} is not a regular file",
            path.display()
        )));
    }
    // O_NONBLOCK has no effect on regular-file reads, so it can stay set.
    Ok(tokio::fs::File::from_std(file))
}

pub(super) fn check_cancelled(ctx: &ToolCtx) -> Result<(), ToolError> {
    if ctx.token.is_cancelled() {
        Err(ToolError::Fatal(crate::Error::Cancelled))
    } else {
        Ok(())
    }
}

pub(super) async fn run_search(
    ctx: &ToolCtx,
    search: impl FnOnce(ToolCtx) -> Result<String, ToolError> + Send + 'static,
) -> Result<String, ToolError> {
    let token = ctx.token.child_token();
    let _guard = token.clone().drop_guard();
    let ctx = ctx.clone().with_token(token);
    tokio::task::spawn_blocking(move || search(ctx))
        .await
        .map_err(|error| {
            ToolError::Fatal(crate::Error::Run(format!("search worker failed: {error}")))
        })?
}

/// Checks for `.git` below the search root, ignoring ancestor components.
pub(super) fn has_git_component(root: &Path, path: &Path) -> bool {
    path.strip_prefix(root)
        .map(|relative| {
            relative
                .components()
                .any(|component| component.as_os_str() == ".git")
        })
        .unwrap_or(false)
}

/// A compiled include/glob pattern. Slash-free patterns match the basename
/// at any depth (gitignore/rg semantics); patterns with slashes match the
/// root-relative path.
pub(super) struct IncludeGlob {
    matcher: globset::GlobMatcher,
    path_relative: bool,
}

pub(super) fn compile_include_glob(pattern: &str) -> Result<IncludeGlob, ToolError> {
    let pattern = pattern.strip_prefix("./").unwrap_or(pattern);
    // `*` stops at `/` as in rg and git; only `**` spans directories.
    let glob = globset::GlobBuilder::new(pattern)
        .literal_separator(true)
        .build()
        .map_err(|err| ToolError::failure(format!("invalid glob pattern {pattern}: {err}")))?;
    Ok(IncludeGlob {
        matcher: glob.compile_matcher(),
        path_relative: pattern.contains('/'),
    })
}

pub(super) fn glob_accepts(glob: &IncludeGlob, root: &Path, file: &Path) -> bool {
    let relative = match file.strip_prefix(root) {
        Ok(relative) if !relative.as_os_str().is_empty() => relative.to_path_buf(),
        _ => PathBuf::from(file.file_name().unwrap_or(file.as_os_str())),
    };
    if glob.path_relative {
        glob.matcher.is_match(&relative)
    } else {
        relative
            .file_name()
            .is_some_and(|name| glob.matcher.is_match(name))
    }
}

/// The display form of a walked path: relative to the working directory
/// when possible, like `rg` run from it.
pub(super) fn display_relative(cwd: &Path, path: &Path) -> String {
    path.strip_prefix(cwd)
        .unwrap_or(path)
        .to_string_lossy()
        .into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_tool_schemas_match_the_old_argument_contracts() {
        let tools = builtin_tools();
        let specs: Vec<_> = tools.iter().map(|tool| tool.spec()).collect();
        let names: Vec<&str> = specs
            .iter()
            .map(|spec| spec.function.name.as_str())
            .collect();
        assert_eq!(
            names,
            ["read", "write", "edit", "bash", "grep", "glob", "process"]
        );

        insta::assert_snapshot!(
            "builtin_tool_schemas",
            serde_json::to_string_pretty(&specs).unwrap()
        );

        for spec in &specs {
            assert_eq!(spec.kind, "function");
            assert!(!spec.function.description.is_empty());
            assert_eq!(spec.function.parameters["type"], "object");
            assert!(spec.function.parameters["properties"].is_object());
        }
        let required = |i: usize| specs[i].function.parameters["required"].clone();
        assert_eq!(required(0), serde_json::json!(["filePath"]));
        assert_eq!(required(1), serde_json::json!(["filePath", "content"]));
        assert_eq!(
            required(2),
            serde_json::json!(["filePath", "oldString", "newString"])
        );
        assert_eq!(required(3), serde_json::json!(["command"]));
        assert_eq!(required(4), serde_json::json!(["pattern"]));
        assert_eq!(required(5), serde_json::json!(["pattern"]));
        let read = &specs[0].function.parameters["properties"];
        assert_eq!(read["filePath"]["type"], "string");
        assert!(read["offset"].get("default").is_none() || read["offset"]["default"] == 1);
        let edit = &specs[2].function.parameters["properties"];
        assert!(edit["replaceAll"].to_string().contains("bool"), "{edit}");
    }

    #[test]
    fn limited_lists_report_the_bound() {
        assert_eq!(limited(vec![], "matches"), "No matches found");
        assert_eq!(limited(vec!["a".into()], "files"), "a");
        let many = (0..101).map(|i| format!("l{i}")).collect();
        let out = limited(many, "matches");
        assert!(
            out.contains("(Showing 100 of at least 101 matches; narrow the search.)"),
            "{out}"
        );
        assert_eq!(out.lines().count(), 102); // 100 hits + blank + note
    }
}
