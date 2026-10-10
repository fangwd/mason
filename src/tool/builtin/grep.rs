//! Content search using ripgrep crates, respecting ignore files and excluding `.git`.

use std::io;

use grep_searcher::{BinaryDetection, Searcher, SearcherBuilder, Sink, SinkMatch};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::tool::builtin::{
    Hit, MAX_RESULTS, check_cancelled, compile_include_glob, display_relative, glob_accepts,
    limited, run_search, walk_files,
};
use crate::tool::output::clip_line;
use crate::tool::{ToolCtx, ToolError, TypedTool};

/// Line-buffer bound: a file with a longer line is skipped instead of loaded whole.
const HEAP_LIMIT: usize = 64 * 1024 * 1024;

pub struct GrepTool;

/// Arguments of the `grep` tool.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GrepArgs {
    /// The regular expression to search for (ripgrep syntax).
    pub pattern: String,
    /// File or directory to search (default: the working directory).
    pub path: Option<String>,
    /// Only search files matching this glob, e.g. "*.ts".
    pub include: Option<String>,
    /// Match regardless of letter case (default: false).
    pub case_insensitive: Option<bool>,
    /// Also search ignored files such as node_modules or build output (default: false).
    pub include_ignored: Option<bool>,
}

#[async_trait::async_trait]
impl TypedTool for GrepTool {
    type Args = GrepArgs;

    fn name(&self) -> &str {
        "grep"
    }

    fn description(&self) -> &str {
        "Search file contents with a regular expression (ripgrep syntax). Returns \"path:line:text\" for up to \
         100 matching lines. Respects .gitignore unless includeIgnored is set; binary files are skipped."
    }

    async fn run(&self, args: GrepArgs, ctx: &ToolCtx) -> Result<String, ToolError> {
        run_search(ctx, move |ctx| search(args, &ctx)).await
    }
}

fn search(args: GrepArgs, ctx: &ToolCtx) -> Result<String, ToolError> {
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
    if args.pattern.contains('\n') {
        return Err(ToolError::failure(
            "multi-line patterns are not supported; the search matches one line at a time",
        ));
    }
    let matcher = grep_regex::RegexMatcherBuilder::new()
        .case_insensitive(args.case_insensitive.unwrap_or(false))
        .build(&args.pattern)
        .map_err(|err| ToolError::failure(err.to_string()))?;
    let include = args
        .include
        .as_deref()
        .map(compile_include_glob)
        .transpose()?;

    // Stop after one extra hit to detect truncation without collecting unbounded
    // results.
    let mut hits: Vec<Hit> = Vec::new();
    let mut skipped = 0usize;
    // Like ripgrep, a NUL byte marks a binary file and ends its search.
    let mut searcher = SearcherBuilder::new()
        .heap_limit(Some(HEAP_LIMIT))
        .binary_detection(BinaryDetection::quit(b'\0'))
        .build();
    'walk: for file in walk_files(&root, args.include_ignored.unwrap_or(false), &ctx.token) {
        check_cancelled(ctx)?;
        if let Some(include) = &include
            && !glob_accepts(include, &root, &file)
        {
            continue;
        }
        let mut sink = HitSink {
            display: display_relative(&ctx.cwd, &file),
            hits: &mut hits,
            token: ctx.token.clone(),
        };
        // One unreadable or oversized file must not fail the whole search.
        let Ok(source) = std::fs::File::open(&file) else {
            skipped += 1;
            continue;
        };
        let reader = SearchReader {
            source,
            token: ctx.token.clone(),
        };
        if searcher.search_reader(&matcher, reader, &mut sink).is_err() {
            if ctx.token.is_cancelled() {
                return Err(ToolError::Fatal(crate::Error::Cancelled));
            }
            skipped += 1;
            continue;
        }
        if hits.len() > MAX_RESULTS {
            break 'walk;
        }
    }
    check_cancelled(ctx)?;
    hits.sort_by(|a, b| (&a.path, a.line).cmp(&(&b.path, b.line)));
    let mut out = limited(
        hits.iter()
            .map(|hit| format!("{}:{}:{}", hit.path, hit.line, hit.text))
            .collect(),
        "matches",
    );
    if skipped > 0 {
        out.push_str(&format!(
            "\n\n({skipped} files could not be searched and were skipped.)"
        ));
    }
    Ok(out)
}

struct SearchReader {
    source: std::fs::File,
    token: tokio_util::sync::CancellationToken,
}

impl io::Read for SearchReader {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if self.token.is_cancelled() {
            return Err(io::Error::other("search cancelled"));
        }
        io::Read::read(&mut self.source, buffer)
    }
}

struct HitSink<'a> {
    display: String,
    hits: &'a mut Vec<Hit>,
    token: tokio_util::sync::CancellationToken,
}

impl Sink for HitSink<'_> {
    type Error = io::Error;

    fn matched(&mut self, _searcher: &Searcher, mat: &SinkMatch<'_>) -> Result<bool, io::Error> {
        if self.token.is_cancelled() {
            return Err(io::Error::other("search cancelled"));
        }
        let text = String::from_utf8_lossy(mat.bytes());
        self.hits.push(Hit {
            path: self.display.clone(),
            line: mat.line_number().unwrap_or(0) as usize,
            text: clip_line(text.trim_end_matches(['\n', '\r'])),
        });
        // `false` stops the search: the walk is abandoned past the limit.
        Ok(self.hits.len() <= MAX_RESULTS)
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::Arc;

    use serde_json::json;

    use super::*;
    use crate::tool::builtin::has_git_component;
    use crate::tool::{OutputSpill, Tool, Typed};

    /// Search fixture with hidden, ignored, nested, and long-line files.
    fn fixture() -> (tempfile::TempDir, ToolCtx) {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(root.join(".gitignore"), "ignored.txt\n").unwrap();
        std::fs::create_dir(root.join(".git")).unwrap();
        std::fs::write(root.join(".git/inside.txt"), "needle in git\n").unwrap();
        std::fs::create_dir(root.join("sub")).unwrap();
        std::fs::write(root.join("g1.ts"), "const needle = 1;\nother\n").unwrap();
        std::fs::write(root.join("g2.md"), "needle in docs\n").unwrap();
        std::fs::write(root.join("sub/deep.ts"), "needle deep\n").unwrap();
        std::fs::write(root.join(".hidden.ts"), "needle hidden\n").unwrap();
        std::fs::write(root.join("ignored.txt"), "needle ignored\n").unwrap();
        std::fs::write(
            root.join("long.txt"),
            format!("needle {}\n", "x".repeat(3_000)),
        )
        .unwrap();
        let ctx = ToolCtx::new(root).with_spill(Arc::new(OutputSpill::new(root.join("spill"))));
        (dir, ctx)
    }

    async fn grep(ctx: &ToolCtx, args: serde_json::Value) -> Result<String, ToolError> {
        Typed(GrepTool).call(args, ctx).await
    }

    #[tokio::test]
    async fn searches_with_include_and_reports_absence_and_bad_patterns() {
        let (_dir, ctx) = fixture();
        let out = grep(
            &ctx,
            json!({"pattern": "need+le", "include": "*.ts", "path": "g1.ts"}),
        )
        .await
        .unwrap();
        assert_eq!(out, "g1.ts:1:const needle = 1;");

        let out = grep(&ctx, json!({"pattern": "absent-text"})).await.unwrap();
        assert_eq!(out, "No matches found");

        let err = grep(&ctx, json!({"pattern": "("})).await.unwrap_err();
        assert!(err.to_string().contains("regex parse error"), "{err}");
    }

    #[tokio::test]
    async fn skips_binary_files_and_supports_case_and_ignored_options() {
        let (dir, ctx) = fixture();
        std::fs::write(dir.path().join("blob.bin"), b"needle\0\xff\xfeneedle").unwrap();
        let out = grep(&ctx, json!({"pattern": "needle"})).await.unwrap();
        assert!(!out.contains("blob.bin:1:needle\0"), "{out}");
        assert!(!out.contains('\u{fffd}'), "{out}");

        let out = grep(
            &ctx,
            json!({"pattern": "NEEDLE DEEP", "caseInsensitive": true}),
        )
        .await
        .unwrap();
        assert_eq!(out, "sub/deep.ts:1:needle deep");

        let out = grep(&ctx, json!({"pattern": "needle ignored"}))
            .await
            .unwrap();
        assert_eq!(out, "No matches found");
        let out = grep(
            &ctx,
            json!({"pattern": "needle ignored", "includeIgnored": true}),
        )
        .await
        .unwrap();
        assert_eq!(out, "ignored.txt:1:needle ignored");
    }

    #[tokio::test]
    async fn honours_hidden_gitignore_and_git_exclusion() {
        let (_dir, ctx) = fixture();
        let out = grep(&ctx, json!({"pattern": "needle", "include": "*.ts"}))
            .await
            .unwrap();
        assert!(out.contains("g1.ts:1:"), "{out}");
        assert!(
            out.contains("sub/deep.ts:1:"),
            "nested dirs are walked: {out}"
        );
        assert!(
            out.contains(".hidden.ts:1:"),
            "--hidden: dotfiles are searched: {out}"
        );
        assert!(
            !out.contains("ignored.txt"),
            ".gitignore is respected: {out}"
        );

        let out = grep(&ctx, json!({"pattern": "needle"})).await.unwrap();
        assert!(
            !out.contains(".git/"),
            "the .git directory is excluded: {out}"
        );
        assert!(!out.contains("ignored.txt"), "{out}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn unreadable_files_are_skipped_not_fatal() {
        use std::os::unix::fs::PermissionsExt;
        if nix::unistd::Uid::effective().is_root() {
            return; // root reads everything
        }
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("ok.txt"), "needle\n").unwrap();
        let locked = dir.path().join("locked.txt");
        std::fs::write(&locked, "needle\n").unwrap();
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).unwrap();
        let ctx = ToolCtx::new(dir.path())
            .with_spill(Arc::new(OutputSpill::new(dir.path().join("spill"))));
        let out = grep(&ctx, json!({"pattern": "needle"})).await.unwrap();
        assert!(out.starts_with("ok.txt:1:needle"), "{out}");
        assert!(
            out.ends_with("(1 files could not be searched and were skipped.)"),
            "{out}"
        );
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o644)).unwrap();
    }

    #[tokio::test]
    async fn multi_line_patterns_are_rejected_explicitly() {
        let (_dir, ctx) = fixture();
        let err = grep(&ctx, json!({"pattern": "needle\nother"}))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("multi-line"), "{err}");
    }

    #[tokio::test]
    async fn gitignore_applies_outside_git_repositories() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(".gitignore"), "ignored.txt\n").unwrap();
        std::fs::write(dir.path().join("ignored.txt"), "needle\n").unwrap();
        std::fs::write(dir.path().join("kept.txt"), "needle\n").unwrap();
        let ctx = ToolCtx::new(dir.path())
            .with_spill(Arc::new(OutputSpill::new(dir.path().join("spill"))));
        let out = grep(&ctx, json!({"pattern": "needle"})).await.unwrap();
        assert_eq!(out, "kept.txt:1:needle");
    }

    #[tokio::test]
    async fn clips_long_lines_in_previews() {
        let (_dir, ctx) = fixture();
        let out = grep(&ctx, json!({"pattern": "needle", "path": "long.txt"}))
            .await
            .unwrap();
        let text = out.split_once(':').unwrap().1;
        let text = text.split_once(':').unwrap().1;
        assert_eq!(text.chars().count(), 2_003, "2000 chars plus the ellipsis");
        assert!(text.ends_with("..."));
    }

    #[tokio::test]
    async fn stops_at_the_result_limit_and_tells_the_model_to_narrow() {
        let dir = tempfile::tempdir().unwrap();
        let content = (0..250)
            .map(|i| format!("match {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(dir.path().join("many.txt"), &content).unwrap();
        let ctx = ToolCtx::new(dir.path())
            .with_spill(Arc::new(OutputSpill::new(dir.path().join("spill"))));
        let out = grep(&ctx, json!({"pattern": "match"})).await.unwrap();
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.len(), MAX_RESULTS + 2); // 100 hits + blank + note
        assert!(
            out.contains("(Showing 100 of at least 101 matches; narrow the search.)"),
            "{out}"
        );
        assert!(lines[0].starts_with("many.txt:1:match 0"), "{out}");
    }

    #[tokio::test]
    async fn a_missing_path_is_a_model_visible_failure() {
        let (_dir, ctx) = fixture();
        let err = grep(&ctx, json!({"pattern": "x", "path": "nowhere"}))
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("no such file or directory"),
            "{err}"
        );
    }

    #[test]
    fn git_component_detection_is_relative_to_the_root() {
        assert!(has_git_component(
            Path::new("/w"),
            Path::new("/w/.git/config")
        ));
        assert!(has_git_component(
            Path::new("/w"),
            Path::new("/w/sub/.git/x")
        ));
        // An ancestor named.git above the root must not exclude everything.
        assert!(!has_git_component(
            Path::new("/home/.git/work"),
            Path::new("/home/.git/work/a.txt")
        ));
    }
}
