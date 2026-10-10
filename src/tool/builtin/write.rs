//! Atomic file writing with parent-directory creation.

use std::path::Path;

use schemars::JsonSchema;
use serde::Deserialize;

use crate::tool::{ToolCtx, ToolError, TypedTool};

pub struct WriteTool;

/// Arguments of the `write` tool.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct WriteArgs {
    /// Path to the file.
    pub file_path: String,
    /// The full file content.
    pub content: String,
}

/// Writes content atomically (temp file plus rename in the target's directory,
/// through a symlink to its target) and creates parent directories. A failure
/// midway leaves the original untouched. The caller must hold the path lock.
pub(crate) async fn write_file(path: &Path, content: &str) -> Result<String, ToolError> {
    let failure = |what: &str, at: &Path, err: std::io::Error| {
        ToolError::failure(format!("cannot {what} {}: {err}", at.display()))
    };
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|err| failure("create", parent, err))?;
    }
    let existing = match tokio::fs::metadata(path).await {
        Ok(meta) => Some(meta),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => None,
        Err(err) => return Err(failure("stat", path, err)),
    };
    if existing.as_ref().is_some_and(|meta| !meta.is_file()) {
        return Err(ToolError::failure(format!(
            "{} is not a regular file",
            path.display()
        )));
    }
    let target = match existing {
        Some(_) => tokio::fs::canonicalize(path)
            .await
            .map_err(|err| failure("resolve", path, err))?,
        None => path.to_path_buf(),
    };
    let name = target
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let temp = target.with_file_name(format!(".{name}.{}.tmp", uuid::Uuid::new_v4()));
    let written = async {
        tokio::fs::write(&temp, content).await?;
        if let Some(meta) = &existing {
            tokio::fs::set_permissions(&temp, meta.permissions()).await?;
        }
        tokio::fs::rename(&temp, &target).await
    }
    .await;
    if let Err(err) = written {
        let _ = tokio::fs::remove_file(&temp).await;
        return Err(failure("write", path, err));
    }
    Ok(format!("Wrote {}", path.display()))
}

#[async_trait::async_trait]
impl TypedTool for WriteTool {
    type Args = WriteArgs;

    fn name(&self) -> &str {
        "write"
    }

    fn description(&self) -> &str {
        "Write a file, replacing it if it exists. Parent directories are created. Prefer edit for changing existing files."
    }

    async fn run(&self, args: WriteArgs, ctx: &ToolCtx) -> Result<String, ToolError> {
        let path = ctx.checked_path(&args.file_path)?;
        // Serialize against concurrent edit/write calls on the same path.
        let _guard = crate::tool::lock_path(&path).await;
        write_file(&path, &args.content).await
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::tool::{OutputSpill, Tool, Typed};
    use std::sync::Arc;
    use std::time::Duration;

    #[cfg(unix)]
    #[tokio::test]
    async fn writes_go_through_symlinks_keep_permissions_and_leave_no_temp_files() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolCtx::new(dir.path())
            .with_spill(Arc::new(OutputSpill::new(dir.path().join("spill"))));
        let target = dir.path().join("real.txt");
        std::fs::write(&target, "old").unwrap();
        std::fs::set_permissions(&target, std::fs::Permissions::from_mode(0o640)).unwrap();
        std::os::unix::fs::symlink(&target, dir.path().join("link.txt")).unwrap();

        Typed(WriteTool)
            .call(json!({"filePath": "link.txt", "content": "new"}), &ctx)
            .await
            .unwrap();
        assert!(
            std::fs::symlink_metadata(dir.path().join("link.txt"))
                .unwrap()
                .file_type()
                .is_symlink(),
            "the link is kept, the target is replaced"
        );
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "new");
        assert_eq!(
            std::fs::metadata(&target).unwrap().permissions().mode() & 0o777,
            0o640
        );
        let leftovers = std::fs::read_dir(dir.path())
            .unwrap()
            .filter(|entry| {
                entry
                    .as_ref()
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .ends_with(".tmp")
            })
            .count();
        assert_eq!(leftovers, 0);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn special_files_are_refused_instead_of_blocking() {
        use crate::tool::builtin::{EditTool, ReadTool};
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolCtx::new(dir.path())
            .with_spill(Arc::new(OutputSpill::new(dir.path().join("spill"))));
        let status = std::process::Command::new("mkfifo")
            .arg(dir.path().join("pipe"))
            .status()
            .unwrap();
        assert!(status.success());
        let calls: Vec<(&str, Box<dyn Tool>, serde_json::Value)> = vec![
            (
                "read",
                Box::new(Typed(ReadTool)),
                json!({"filePath": "pipe"}),
            ),
            (
                "edit",
                Box::new(Typed(EditTool)),
                json!({"filePath": "pipe", "oldString": "a", "newString": "b"}),
            ),
            (
                "write",
                Box::new(Typed(WriteTool)),
                json!({"filePath": "pipe", "content": "x"}),
            ),
        ];
        // The guard is on the opened handle, so a FIFO swapped in after a path
        // check is refused too; `open_regular` is probed directly for that.
        let err = tokio::time::timeout(
            Duration::from_secs(2),
            crate::tool::builtin::open_regular(&dir.path().join("pipe")),
        )
        .await
        .expect("the open must not block")
        .unwrap_err();
        assert!(err.to_string().contains("not a regular file"), "{err}");
        for (name, tool, args) in calls {
            // A FIFO with no peer blocks open/read/write forever on a blocking
            // thread that cancellation cannot release.
            let result = tokio::time::timeout(Duration::from_secs(2), tool.call(args, &ctx))
                .await
                .unwrap_or_else(|_| panic!("{name} blocked on a FIFO"));
            let err = result.unwrap_err();
            assert!(
                err.to_string().contains("not a regular file"),
                "{name}: {err}"
            );
        }
    }

    #[tokio::test]
    async fn writes_files_and_creates_parents() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolCtx::new(dir.path())
            .with_spill(Arc::new(OutputSpill::new(dir.path().join("spill"))));
        let out = Typed(WriteTool)
            .call(json!({"filePath": "a/b/c.txt", "content": "hello"}), &ctx)
            .await
            .unwrap();
        assert_eq!(
            out,
            format!("Wrote {}", dir.path().join("a/b/c.txt").display())
        );
        assert_eq!(
            std::fs::read_to_string(dir.path().join("a/b/c.txt")).unwrap(),
            "hello"
        );

        Typed(WriteTool)
            .call(json!({"filePath": "a/b/c.txt", "content": "again"}), &ctx)
            .await
            .unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.path().join("a/b/c.txt")).unwrap(),
            "again"
        );
    }
}
