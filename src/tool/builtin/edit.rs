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
        let (result, content, old_string, new_string) = tokio::task::spawn_blocking(move || {
            let result =
                replace_keeping_line_endings(&content, &old_string, &new_string, replace_all);
            (result, content, old_string, new_string)
        })
        .await
        .map_err(|err| ToolError::failure(format!("the edit worker failed: {err}")))?;
        let updated = match result {
            Ok(updated) => updated,
            Err(err) => {
                let mut message = err.to_string();
                // Miss hints quote file lines, so they need the run's `read` access.
                if matches!(err, replace::ReplaceError::NotFound)
                    && ctx
                        .may_read(&serde_json::json!({"filePath": args.file_path}))
                        .await?
                    && let Some(hint) = tokio::task::spawn_blocking(move || {
                        miss_hint(&content, &old_string, &new_string)
                    })
                    .await
                    .map_err(|err| ToolError::failure(format!("the edit worker failed: {err}")))?
                {
                    message.push_str("\n\n");
                    message.push_str(&hint);
                }
                return Err(ToolError::failure(message));
            }
        };
        super::write::write_file(&path, &updated).await?;
        Ok(format!("Edited {}", path.display()))
    }
}

/// Diagnostic only; adapted from opencode @84f66054. Never widens a match.
fn miss_hint(content: &str, old: &str, new: &str) -> Option<String> {
    use std::collections::HashMap;

    let normalize = |s: &str| s.split_whitespace().collect::<Vec<_>>().join(" ");
    if new.trim().chars().count() >= 20 && normalize(content).contains(&normalize(new)) {
        return Some(
            "newString is already in the file, so this edit may already be applied.".into(),
        );
    }
    let lines: Vec<_> = content.lines().collect();
    let search: Vec<_> = old.lines().collect();
    let mut positions: HashMap<String, Vec<usize>> = HashMap::new();
    for (index, line) in lines.iter().enumerate() {
        let key = normalize(line);
        if key.chars().any(char::is_alphanumeric) {
            positions.entry(key).or_default().push(index);
        }
    }
    let mut votes: HashMap<isize, usize> = HashMap::new();
    let mut wanted = 0;
    let mut comparisons = 0usize;
    for (offset, line) in search.iter().enumerate() {
        let key = normalize(line);
        if !key.chars().any(char::is_alphanumeric) {
            continue;
        }
        wanted += 1;
        if let Some(indices) = positions.get(&key) {
            comparisons = comparisons.saturating_add(indices.len());
            if comparisons > 100_000 {
                return None;
            }
            for &index in indices {
                *votes.entry(index as isize - offset as isize).or_default() += 1;
            }
        }
    }
    let (&start, &count) = votes
        .iter()
        .max_by_key(|&(start, count)| (*count, -start))?;
    if count * 2 < wanted {
        return None;
    }
    let matched: Vec<usize> = search
        .iter()
        .enumerate()
        .filter_map(|(offset, line)| {
            let index = start + offset as isize;
            (index >= 0
                && (index as usize) < lines.len()
                && !normalize(line).is_empty()
                && normalize(line) == normalize(lines[index as usize]))
            .then_some(index as usize)
        })
        .collect();
    let from = *matched.first()?;
    let to = (matched.last()? + 1).min(from + 12);
    let mut result = String::from("The closest lines in the file:");
    for (index, line) in lines[from..to].iter().enumerate() {
        let preview: String = line.chars().take(200).collect();
        let suffix = if line.chars().count() > 200 {
            "…"
        } else {
            ""
        };
        result.push_str(&format!("\n{}: {preview}{suffix}", from + index + 1));
    }
    Some(result)
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

    #[test]
    fn miss_hints_are_bounded_and_only_show_relevant_lines() {
        assert!(
            miss_hint(
                "the replacement is already here",
                "missing",
                "the replacement is already here"
            )
            .unwrap()
            .contains("already in the file")
        );
        let hint = miss_hint(
            "secret before\nalpha one\nchanged\nomega two\nsecret after",
            "alpha one\nold value\nomega two",
            "new",
        )
        .unwrap();
        assert!(hint.contains("2: alpha one") && hint.contains("4: omega two"));
        assert!(!hint.contains("secret"));
        assert!(miss_hint("unrelated", "nothing matching", "new").is_none());
    }

    #[tokio::test]
    async fn edit_miss_hints_only_when_the_run_may_read() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("e"), "alpha one\nchanged\nomega two").unwrap();
        let args = json!({"filePath":"e", "oldString":"alpha one\nold value\nomega two", "newString":"new"});
        let err = edit(dir.path(), args.clone()).await.unwrap_err();
        assert!(err.to_string().contains("closest lines"), "{err}");

        let mut ctx = ToolCtx::new(dir.path());
        ctx.access = Some(Arc::new(crate::tool::RunAccess {
            tools: Arc::new(crate::tool::Toolset::new()),
            rules: None,
            edited: std::sync::Mutex::new(Default::default()),
            parent: None,
        }));
        let err = Typed(EditTool).call(args.clone(), &ctx).await.unwrap_err();
        assert!(!err.to_string().contains("closest lines"), "{err}");

        struct ReadPolicy(fn() -> ToolError);
        #[async_trait::async_trait]
        impl crate::tool::policy::ToolPolicy for ReadPolicy {
            async fn authorize(
                &self,
                name: &str,
                _: &serde_json::Value,
                _: &ToolCtx,
            ) -> Result<(), ToolError> {
                if name == "read" {
                    Err((self.0)())
                } else {
                    Ok(())
                }
            }
        }
        let mut ctx = ToolCtx::new(dir.path());
        ctx.policy = Arc::new(ReadPolicy(|| ToolError::failure("denied")));
        let err = Typed(EditTool).call(args.clone(), &ctx).await.unwrap_err();
        assert!(
            !err.is_fatal() && !err.to_string().contains("closest lines"),
            "{err}"
        );

        ctx.policy = Arc::new(ReadPolicy(|| ToolError::Fatal(crate::Error::Cancelled)));
        let err = Typed(EditTool).call(args, &ctx).await.unwrap_err();
        assert!(err.is_fatal(), "{err}");
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
