//! Paged text-file reads and directory listings.

use schemars::JsonSchema;
use serde::Deserialize;
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncReadExt, AsyncSeekExt, SeekFrom};

use crate::tool::output::{MAX_BYTES, MAX_LINES, clip_line};
use crate::tool::{ToolCtx, ToolError, TypedTool};

/// Page byte budget, leaving room for the footer within the model-visible bound.
const PAGE_BYTES: usize = MAX_BYTES - 2 * 1024;
/// Page line budget: the blank line and footer must also fit the bound.
const PAGE_LINES: usize = MAX_LINES - 2;

pub struct ReadTool;

/// Arguments of the `read` tool.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ReadArgs {
    /// Path to the file, absolute or relative to the working directory.
    pub file_path: String,
    /// Line number to start from (1-based).
    #[serde(default = "default_offset")]
    pub offset: usize,
    /// Number of lines to read (default and maximum 1998, one page).
    #[serde(default = "default_limit")]
    pub limit: usize,
}

fn default_offset() -> usize {
    1
}

fn default_limit() -> usize {
    PAGE_LINES
}

#[async_trait::async_trait]
impl TypedTool for ReadTool {
    type Args = ReadArgs;

    fn name(&self) -> &str {
        "read"
    }

    fn description(&self) -> &str {
        "Read a file (or list a directory). Lines are prefixed with their 1-based number (\"12: text\"). \
         Reads about 2000 lines (or 48 KiB) per page; use offset and limit for larger files."
    }

    async fn run(&self, args: ReadArgs, ctx: &ToolCtx) -> Result<String, ToolError> {
        let path = ctx.checked_path(&args.file_path)?;
        let meta = tokio::fs::metadata(&path)
            .await
            .map_err(|err| ToolError::failure(format!("cannot stat {}: {err}", path.display())))?;
        if meta.is_dir() {
            return list_dir(&path).await;
        }
        let mut file = super::open_regular(&path).await?;
        let mut head = vec![0u8; 4096];
        let read = file
            .read(&mut head)
            .await
            .map_err(|err| ToolError::failure(format!("cannot read {}: {err}", path.display())))?;
        if head[..read].contains(&0) {
            return Err(ToolError::failure(format!(
                "Cannot read binary file: {}",
                path.display()
            )));
        }
        file.seek(SeekFrom::Start(0))
            .await
            .map_err(|err| ToolError::failure(format!("cannot read {}: {err}", path.display())))?;

        let offset = args.offset.max(1);
        // Clamp page size to bound memory and model-visible output.
        let limit = args.limit.clamp(1, PAGE_LINES);
        let mut reader = tokio::io::BufReader::new(file);
        let mut body: Vec<String> = Vec::new();
        let mut bytes = 0usize;
        let mut page_full = false;
        let mut total = 0usize;
        let mut line = Vec::new();
        let mut ends_with_newline = false;
        loop {
            if total.is_multiple_of(1024) && ctx.token.is_cancelled() {
                return Err(ToolError::Fatal(crate::Error::Cancelled));
            }
            let (newline, any) =
                read_line_bounded(&mut reader, &mut line)
                    .await
                    .map_err(|err| {
                        ToolError::failure(format!("cannot read {}: {err}", path.display()))
                    })?;
            if !any {
                break;
            }
            ends_with_newline = newline;
            total += 1;
            if total >= offset && body.len() < limit && !page_full {
                let text = String::from_utf8_lossy(&line);
                let rendered = format!("{total}: {}", clip_line(&text));
                // Long lines fill the byte budget first; the page always shows at
                // least one line.
                if !body.is_empty() && bytes + rendered.len() + 1 > PAGE_BYTES {
                    page_full = true;
                } else {
                    bytes += rendered.len() + 1;
                    body.push(rendered);
                }
            }
        }
        // Count the final fragment as a line, including empty files and trailing
        // newlines.
        let virtual_last = if total == 0 {
            Some(1)
        } else if ends_with_newline {
            Some(total + 1)
        } else {
            None
        };
        if let Some(n) = virtual_last {
            total = n;
            if n >= offset && body.len() < limit && !page_full {
                body.push(format!("{n}: "));
            }
        }
        if offset > total {
            return Err(ToolError::failure(format!(
                "Offset {offset} is beyond the end of the file ({total} lines)"
            )));
        }
        let shown = offset + body.len().saturating_sub(1);
        let footer = if body.is_empty() || shown < total {
            format!(
                "(Showing lines {offset}-{shown} of {total}. Use offset={} to continue.)",
                shown + 1
            )
        } else {
            format!("(End of file - {total} lines)")
        };
        Ok(format!("{}\n\n{footer}", body.join("\n")))
    }
}

/// Buffer cap per line; discard overflow while scanning to the newline.
const LINE_CAP: usize = 16 * 1024;

/// Reads a line while retaining at most [`LINE_CAP`] bytes. Returns
/// `(ended_with_newline, any_bytes_read)`.
async fn read_line_bounded<R>(reader: &mut R, out: &mut Vec<u8>) -> std::io::Result<(bool, bool)>
where
    R: AsyncBufRead + Unpin,
{
    out.clear();
    let mut any = false;
    loop {
        let (consumed, newline) = {
            let buf = reader.fill_buf().await?;
            if buf.is_empty() {
                return Ok((false, any));
            }
            any = true;
            let room = LINE_CAP.saturating_sub(out.len());
            match buf.iter().position(|byte| *byte == b'\n') {
                Some(index) => {
                    out.extend_from_slice(&buf[..index.min(room)]);
                    (index + 1, true)
                }
                None => {
                    out.extend_from_slice(&buf[..buf.len().min(room)]);
                    (buf.len(), false)
                }
            }
        };
        reader.consume(consumed);
        if newline {
            return Ok((true, any));
        }
    }
}

async fn list_dir(path: &std::path::Path) -> Result<String, ToolError> {
    let mut entries = tokio::fs::read_dir(path)
        .await
        .map_err(|err| ToolError::failure(format!("cannot list {}: {err}", path.display())))?;
    let mut names: Vec<String> = Vec::new();
    while let Some(entry) = entries
        .next_entry()
        .await
        .map_err(|err| ToolError::failure(format!("cannot list {}: {err}", path.display())))?
    {
        let is_dir = entry.file_type().await.map(|t| t.is_dir()).unwrap_or(false);
        let mut name = entry.file_name().to_string_lossy().into_owned();
        if is_dir {
            name.push('/');
        }
        names.push(name);
    }
    names.sort();
    Ok(names.join("\n"))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::tool::{OutputSpill, Tool, Typed};
    use std::sync::Arc;

    async fn read(dir: &std::path::Path, args: serde_json::Value) -> Result<String, ToolError> {
        let ctx = ToolCtx::new(dir).with_spill(Arc::new(OutputSpill::new(dir.join("spill"))));
        Typed(ReadTool).call(args, &ctx).await
    }

    #[tokio::test]
    async fn pages_long_files() {
        let dir = tempfile::tempdir().unwrap();
        let content = (1..=10)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(dir.path().join("long.txt"), &content).unwrap();
        let out = read(
            dir.path(),
            json!({"filePath": "long.txt", "offset": 3, "limit": 2}),
        )
        .await
        .unwrap();
        assert_eq!(
            out,
            "3: line 3\n4: line 4\n\n(Showing lines 3-4 of 10. Use offset=5 to continue.)"
        );

        let out = read(dir.path(), json!({"filePath": "long.txt"}))
            .await
            .unwrap();
        assert!(out.ends_with("(End of file - 10 lines)"), "{out}");
        assert!(out.starts_with("1: line 1\n"), "{out}");
    }

    #[tokio::test]
    async fn counts_the_virtual_last_line_like_split() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "a\n").unwrap();
        let out = read(dir.path(), json!({"filePath": "a.txt", "offset": 2}))
            .await
            .unwrap();
        assert_eq!(out, "2: \n\n(End of file - 2 lines)");

        std::fs::write(dir.path().join("empty.txt"), "").unwrap();
        let out = read(dir.path(), json!({"filePath": "empty.txt"}))
            .await
            .unwrap();
        assert_eq!(out, "1: \n\n(End of file - 1 lines)");
    }

    #[tokio::test]
    async fn lists_directories_sorted_with_slashes() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("b.txt"), "x").unwrap();
        std::fs::write(dir.path().join("a.txt"), "x").unwrap();
        let out = read(dir.path(), json!({"filePath": "."})).await.unwrap();
        let mut lines: Vec<&str> = out.lines().collect();
        lines.retain(|l| *l != "spill/"); // the test spill dir may exist
        assert_eq!(lines, ["a.txt", "b.txt", "sub/"]);
    }

    #[tokio::test]
    async fn rejects_binary_files_and_bad_offsets() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("b.bin"), b"abc\0def").unwrap();
        let err = read(dir.path(), json!({"filePath": "b.bin"}))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("Cannot read binary file"), "{err}");

        std::fs::write(dir.path().join("t.txt"), "one\ntwo\n").unwrap();
        let err = read(dir.path(), json!({"filePath": "t.txt", "offset": 99}))
            .await
            .unwrap_err();
        assert_eq!(
            err.to_string(),
            "Offset 99 is beyond the end of the file (3 lines)"
        );

        let err = read(dir.path(), json!({"filePath": "missing.txt"}))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("cannot stat"), "{err}");
    }

    #[tokio::test]
    async fn clips_long_lines_in_the_preview() {
        let dir = tempfile::tempdir().unwrap();
        let long = "x".repeat(3_000);
        std::fs::write(dir.path().join("l.txt"), format!("short\n{long}\n")).unwrap();
        let out = read(dir.path(), json!({"filePath": "l.txt"}))
            .await
            .unwrap();
        let second = out.lines().nth(1).unwrap();
        assert!(second.starts_with("2: xxx"), "{second}");
        assert!(second.ends_with("..."), "{second}");
        assert_eq!(second.chars().count(), "2: ".chars().count() + 2_000 + 3);
    }

    #[tokio::test]
    async fn clamps_huge_limits_to_the_page_bound() {
        let dir = tempfile::tempdir().unwrap();
        let content = (1..=5_000)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(dir.path().join("big.txt"), &content).unwrap();
        let out = read(
            dir.path(),
            json!({"filePath": "big.txt", "limit": 1_000_000_000}),
        )
        .await
        .unwrap();
        assert!(
            out.contains("(Showing lines 1-1998 of 5000. Use offset=1999 to continue.)"),
            "{out}"
        );
        assert!(!out.contains("1999: "), "the page stops at the bound");
        assert!(
            out.lines().count() <= MAX_LINES,
            "a full page plus footer fits the model-visible bound: {}",
            out.lines().count()
        );
    }

    #[tokio::test]
    async fn pages_by_bytes_as_well_as_lines() {
        let dir = tempfile::tempdir().unwrap();
        let line = "z".repeat(1_000);
        let content = vec![line; 100].join("\n");
        std::fs::write(dir.path().join("wide.txt"), &content).unwrap();
        let out = read(dir.path(), json!({"filePath": "wide.txt"}))
            .await
            .unwrap();
        assert!(out.len() <= MAX_BYTES, "{} bytes", out.len());
        let shown = out.lines().filter(|l| l.contains(": zzz")).count();
        assert!((40..100).contains(&shown), "{shown} lines shown");
        assert!(
            out.ends_with(&format!(
                "(Showing lines 1-{shown} of 100. Use offset={} to continue.)",
                shown + 1
            )),
            "{out}"
        );

        let rest = read(
            dir.path(),
            json!({"filePath": "wide.txt", "offset": shown + 1}),
        )
        .await
        .unwrap();
        assert!(rest.starts_with(&format!("{}: zzz", shown + 1)), "{rest}");
    }

    #[tokio::test]
    async fn bounds_newline_free_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("flat.txt"), "y".repeat(4 * 1024 * 1024)).unwrap();
        let out = read(dir.path(), json!({"filePath": "flat.txt"}))
            .await
            .unwrap();
        assert!(out.starts_with("1: yyy"), "{out}");
        assert!(out.contains("..."), "the huge line is clipped");
        assert!(
            out.len() < 10_000,
            "the output stays bounded: {}",
            out.len()
        );
        assert!(out.contains("(End of file - 1 lines)"), "{out}");
    }
}
