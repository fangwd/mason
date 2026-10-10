//! Glob-based file listing using the same traversal as `grep`.

use schemars::JsonSchema;
use serde::Deserialize;

use crate::tool::builtin::{
    MAX_RESULTS, check_cancelled, compile_include_glob, display_relative, glob_accepts, limited,
    run_search, walk_files,
};
use crate::tool::{ToolCtx, ToolError, TypedTool};

pub struct GlobTool;

/// Arguments of the `glob` tool.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GlobArgs {
    /// The glob pattern, e.g. "src/**/*.ts".
    pub pattern: String,
    /// Directory to search (default: the working directory).
    pub path: Option<String>,
    /// Also list ignored files such as node_modules or build output (default: false).
    pub include_ignored: Option<bool>,
}

#[async_trait::async_trait]
impl TypedTool for GlobTool {
    type Args = GlobArgs;

    fn name(&self) -> &str {
        "glob"
    }

    fn description(&self) -> &str {
        "List files matching a glob pattern such as \"src/**/*.ts\" (up to 100). Respects .gitignore unless \
         includeIgnored is set."
    }

    async fn run(&self, args: GlobArgs, ctx: &ToolCtx) -> Result<String, ToolError> {
        run_search(ctx, move |ctx| search(args, &ctx)).await
    }
}

fn search(args: GlobArgs, ctx: &ToolCtx) -> Result<String, ToolError> {
    check_cancelled(ctx)?;
    let root = args
        .path
        .as_ref()
        .map_or_else(|| ctx.checked_path("."), |path| ctx.checked_path(path))?;
    if !root.exists() {
        return Err(ToolError::failure(format!(
            "no such file or directory: {}",
            root.display()
        )));
    }
    let glob = compile_include_glob(&args.pattern)?;
    let mut files: Vec<String> = Vec::new();
    for file in walk_files(&root, args.include_ignored.unwrap_or(false), &ctx.token) {
        check_cancelled(ctx)?;
        if !glob_accepts(&glob, &root, &file) {
            continue;
        }
        files.push(display_relative(&ctx.cwd, &file));
        if files.len() > MAX_RESULTS {
            break;
        }
    }
    check_cancelled(ctx)?;
    files.sort();
    Ok(limited(files, "files"))
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::Arc;

    use serde_json::json;

    use super::*;
    use crate::tool::{OutputSpill, Tool, Typed};

    async fn glob(dir: &Path, args: serde_json::Value) -> Result<String, ToolError> {
        let ctx = ToolCtx::new(dir).with_spill(Arc::new(OutputSpill::new(dir.join("spill"))));
        Typed(GlobTool).call(args, &ctx).await
    }

    #[tokio::test]
    async fn a_single_star_does_not_cross_directories() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("src/nested")).unwrap();
        std::fs::write(dir.path().join("src/a.ts"), "").unwrap();
        std::fs::write(dir.path().join("src/nested/b.ts"), "").unwrap();
        for pattern in ["src/*.ts", "./src/*.ts"] {
            let out = glob(dir.path(), json!({"pattern": pattern})).await.unwrap();
            assert_eq!(out, "src/a.ts", "{pattern}: {out}");
        }
        let out = glob(dir.path(), json!({"pattern": "src/**/*.ts"}))
            .await
            .unwrap();
        assert_eq!(out, "src/a.ts\nsrc/nested/b.ts");
    }

    #[tokio::test]
    async fn matches_brace_patterns() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("g1.ts"), "x").unwrap();
        std::fs::write(dir.path().join("g2.md"), "x").unwrap();
        std::fs::write(dir.path().join("g3.txt"), "x").unwrap();
        let out = glob(dir.path(), json!({"pattern": "g*.{ts,md}"}))
            .await
            .unwrap();
        let mut files: Vec<&str> = out.lines().collect();
        files.sort_unstable();
        assert_eq!(files, ["g1.ts", "g2.md"]);
    }

    #[tokio::test]
    async fn slashless_patterns_match_at_any_depth_and_respects_gitignore() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        std::fs::create_dir_all(dir.path().join("src/nested")).unwrap();
        std::fs::write(dir.path().join("src/a.ts"), "x").unwrap();
        std::fs::write(dir.path().join("src/nested/b.ts"), "x").unwrap();
        std::fs::write(dir.path().join(".git/c.ts"), "x").unwrap();
        std::fs::write(dir.path().join(".gitignore"), "src/a.ts\n").unwrap();

        let out = glob(dir.path(), json!({"pattern": "*.ts"})).await.unwrap();
        assert_eq!(
            out, "src/nested/b.ts",
            "ignored and .git files stay out: {out}"
        );

        let out = glob(dir.path(), json!({"pattern": "src/**/*.ts"}))
            .await
            .unwrap();
        assert_eq!(
            out, "src/nested/b.ts",
            "slash patterns match full relative paths: {out}"
        );

        let out = glob(dir.path(), json!({"pattern": "*.rs"})).await.unwrap();
        assert_eq!(out, "No files found");

        let err = glob(dir.path(), json!({"pattern": "["})).await.unwrap_err();
        assert!(err.to_string().contains("invalid glob pattern"), "{err}");
    }

    #[tokio::test]
    async fn stops_at_the_result_limit() {
        let dir = tempfile::tempdir().unwrap();
        for i in 0..150 {
            std::fs::write(dir.path().join(format!("f{i:03}.txt")), "x").unwrap();
        }
        let out = glob(dir.path(), json!({"pattern": "*.txt"})).await.unwrap();
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.len(), MAX_RESULTS + 2);
        assert!(
            out.contains("(Showing 100 of at least 101 files; narrow the search.)"),
            "{out}"
        );
        assert!(lines[0] == "f000.txt", "{out}");
    }
}
