//! File editing with the opencode-derived fuzzy matcher chain.

use schemars::JsonSchema;
use serde::Deserialize;
use tokio::io::AsyncReadExt;

use crate::tool::replace;
use crate::tool::{ToolCtx, ToolError, TypedTool};

/// The matcher chain loads and scans the whole file; larger files are refused.
const MAX_EDIT_BYTES: u64 = 16 * 1024 * 1024;

pub struct EditTool;

/// Arguments of the `edit` tool.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct EditArgs {
    /// Path to the file.
    pub file_path: String,
    /// The text to replace.
    pub old_string: String,
    /// The replacement text.
    pub new_string: String,
    /// Replace every occurrence (default false).
    #[serde(default)]
    pub replace_all: bool,
}

#[async_trait::async_trait]
impl TypedTool for EditTool {
    type Args = EditArgs;

    fn name(&self) -> &str {
        "edit"
    }

    fn description(&self) -> &str {
        "Replace oldString with newString in a file. oldString must match one place in the file \
         (include surrounding lines to make it unique) unless replaceAll is set. \
         An empty oldString creates a new file with newString as its content."
    }

    async fn run(&self, args: EditArgs, ctx: &ToolCtx) -> Result<String, ToolError> {
        let path = ctx.checked_path(&args.file_path)?;
        if args.old_string == args.new_string {
            return Err(ToolError::failure("oldString and newString are identical"));
        }
        // Hold the path lock through creation checks and the full read–modify–write
        // cycle.
        let _guard = crate::tool::lock_path(&path).await;
        if args.old_string.is_empty() {
            if tokio::fs::try_exists(&path).await.unwrap_or(false) {
                return Err(ToolError::failure(format!(
                    "{} exists; use write to replace it",
                    path.display()
                )));
            }
            return super::write::write_file(&path, &args.new_string).await;
        }
        // Size and type are checked on the opened handle, not the path.
        let mut file = super::open_regular(&path).await?;
        let size = file
            .metadata()
            .await
            .map_err(|err| ToolError::failure(format!("cannot stat {}: {err}", path.display())))?
            .len();
        if size > MAX_EDIT_BYTES {
            return Err(ToolError::failure(format!(
                "{} is {size} bytes, above the {MAX_EDIT_BYTES}-byte edit limit; use bash for files this large",
                path.display()
            )));
        }
        let mut content = String::new();
        file.read_to_string(&mut content)
            .await
            .map_err(|err| ToolError::failure(format!("cannot read {}: {err}", path.display())))?;
        let EditArgs {
            old_string,
            new_string,
            replace_all,
            ..
        } = args;
        // The fuzzy matchers are CPU-bound: keep them off the async workers.
        let updated = tokio::task::spawn_blocking(move || {
            replace_keeping_line_endings(&content, &old_string, &new_string, replace_all)
        })
        .await
        .map_err(|err| ToolError::failure(format!("the edit worker failed: {err}")))?
        .map_err(|err| ToolError::failure(err.to_string()))?;
        super::write::write_file(&path, &updated).await?;
        Ok(format!("Edited {}", path.display()))
    }
}

/// Matches a CRLF file as LF, so the model's LF strings fit, and writes CRLF
/// back. Files with mixed or LF endings are edited as they are.
fn replace_keeping_line_endings(
    content: &str,
    old: &str,
    new: &str,
    replace_all: bool,
) -> Result<String, replace::ReplaceError> {
    let crlf = content.contains("\r\n")
        && content.matches('\n').count() == content.matches("\r\n").count();
    if !crlf {
        return replace::replace(content, old, new, replace_all);
    }
    let lf = |text: &str| text.replace("\r\n", "\n");
    replace::replace(&lf(content), &lf(old), &lf(new), replace_all)
        .map(|updated| updated.replace('\n', "\r\n"))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::tool::{OutputSpill, Tool, Typed};
    use std::sync::Arc;

    async fn edit(dir: &std::path::Path, args: serde_json::Value) -> Result<String, ToolError> {
        let ctx = ToolCtx::new(dir).with_spill(Arc::new(OutputSpill::new(dir.join("spill"))));
        Typed(EditTool).call(args, &ctx).await
    }

    #[test]
    fn crlf_files_keep_their_line_endings() {
        let updated = replace_keeping_line_endings("a\r\nb\r\nc\r\n", "a\nb\n", "x\ny\n", false);
        assert_eq!(updated.unwrap(), "x\r\ny\r\nc\r\n");
        let mixed = replace_keeping_line_endings("a\r\nb\nc\n", "b", "y", false);
        assert_eq!(mixed.unwrap(), "a\r\ny\nc\n");
    }

    #[tokio::test]
    async fn uses_the_replacers_and_rejects_ambiguous_matches() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("e.ts");
        // Indentation differs from the file; the fallback replacers still find it.
        std::fs::write(&file, "if (x) {\n    return 1;\n}\nreturn 1;\n").unwrap();
        edit(
            dir.path(),
            json!({"filePath": "e.ts", "oldString": "if (x) {\nreturn 1;\n}", "newString": "if (x) {\n    return 2;\n}"}),
        )
        .await
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(&file).unwrap(),
            "if (x) {\n    return 2;\n}\nreturn 1;\n"
        );

        std::fs::write(&file, "a\na\n").unwrap();
        let err = edit(
            dir.path(),
            json!({"filePath": "e.ts", "oldString": "a", "newString": "b"}),
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("multiple matches"), "{err}");
    }

    #[tokio::test]
    async fn replace_all_swaps_every_occurrence() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("a.txt");
        std::fs::write(&file, "x-1 x-2 x-3").unwrap();
        edit(
            dir.path(),
            json!({"filePath": "a.txt", "oldString": "x-", "newString": "y-", "replaceAll": true}),
        )
        .await
        .unwrap();
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "y-1 y-2 y-3");
    }

    #[tokio::test]
    async fn empty_old_string_creates_new_files_only() {
        let dir = tempfile::tempdir().unwrap();
        let out = edit(
            dir.path(),
            json!({"filePath": "new.txt", "oldString": "", "newString": "fresh"}),
        )
        .await
        .unwrap();
        assert_eq!(
            out,
            format!("Wrote {}", dir.path().join("new.txt").display())
        );
        assert_eq!(
            std::fs::read_to_string(dir.path().join("new.txt")).unwrap(),
            "fresh"
        );

        let err = edit(
            dir.path(),
            json!({"filePath": "new.txt", "oldString": "", "newString": "x"}),
        )
        .await
        .unwrap_err();
        assert!(
            err.to_string().contains("exists; use write to replace it"),
            "{err}"
        );
    }

    #[tokio::test]
    async fn oversized_files_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("huge.txt"),
            vec![b'x'; MAX_EDIT_BYTES as usize + 1],
        )
        .unwrap();
        let err = edit(
            dir.path(),
            json!({"filePath": "huge.txt", "oldString": "x", "newString": "y"}),
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("edit limit"), "{err}");
    }

    #[tokio::test]
    async fn identical_strings_and_missing_files_fail_cleanly() {
        let dir = tempfile::tempdir().unwrap();
        let err = edit(
            dir.path(),
            json!({"filePath": "x.txt", "oldString": "a", "newString": "a"}),
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("identical"), "{err}");

        let err = edit(
            dir.path(),
            json!({"filePath": "missing.txt", "oldString": "a", "newString": "b"}),
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("cannot read"), "{err}");
        assert!(
            !dir.path().join("missing.txt").exists(),
            "a failed edit writes nothing"
        );
    }
}
