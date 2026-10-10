//! Shell commands with bounded output capture. Unix cancellation kills the process
//! group; other platforms only kill the direct child.

use std::process::Stdio;
use std::time::Duration;

use bytes::Bytes;
use schemars::JsonSchema;
use serde::Deserialize;
use tokio::io::AsyncReadExt;
use tokio::process::Child;
use tokio::sync::mpsc;
use tokio::time::{Instant, sleep_until};

use crate::error::Error;
use crate::tool::process::GroupGuard;
use crate::tool::{ToolCtx, ToolError, TypedTool};

/// Bound on draining output after killing the process.
const DRAIN_GRACE: Duration = Duration::from_secs(5);
/// How long output is still collected after the command itself exited while
/// background processes keep the pipes open.
const EXIT_DRAIN_GRACE: Duration = Duration::from_secs(1);
const LINGER_NOTE: &str = "Background processes are owned by this session and keep running between messages; use process to list or stop them. Later output is discarded.";

pub struct BashTool;

/// Arguments of the `bash` tool.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct BashArgs {
    /// The command to run.
    pub command: String,
    /// Directory to run in (default: the working directory).
    pub workdir: Option<String>,
    /// Timeout in ms; host defaults are usually 120000 with a 600000 maximum.
    pub timeout: Option<u64>,
}

#[async_trait::async_trait]
impl TypedTool for BashTool {
    type Args = BashArgs;

    fn name(&self) -> &str {
        "bash"
    }

    fn description(&self) -> &str {
        "Run a shell command and return its combined stdout/stderr and exit code. \
         Use workdir instead of cd. Long output is truncated to its tail with the full output saved to a file. \
         Returns when the command exits. Background processes belong to this session and keep running between messages; use process to list or stop them. Later output is discarded."
    }

    async fn run(&self, args: BashArgs, ctx: &ToolCtx) -> Result<String, ToolError> {
        if ctx.token.is_cancelled() {
            return Err(ToolError::Fatal(Error::Cancelled));
        }
        let timeout_ms = args
            .timeout
            .unwrap_or(ctx.shell_timeout_ms)
            .min(ctx.shell_max_timeout_ms);
        let workdir = ctx.checked_path(args.workdir.as_deref().unwrap_or("."))?;
        // Always bash, whatever the user's login shell: the model writes bash.
        let mut isolated = crate::tool::policy::subprocess(
            "/bin/bash",
            &["-c".into(), args.command],
            &ctx.cwd,
            ctx.execution_mode,
            ctx.allow_network,
        )?;
        let command = &mut isolated.command;
        command
            .current_dir(&workdir)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let child = command
            .spawn()
            .map_err(|err| ToolError::failure(format!("cannot run /bin/bash: {err}")))?;
        run_child(child, ctx, timeout_ms, isolated._scratch.take()).await
    }
}

async fn run_child(
    mut child: Child,
    ctx: &ToolCtx,
    timeout_ms: u64,
    scratch: Option<tempfile::TempDir>,
) -> Result<String, ToolError> {
    let mut guard = GroupGuard::new(child.id());
    let (tx, mut rx) = mpsc::channel::<Bytes>(64);
    if let Some(stdout) = child.stdout.take() {
        spawn_drain(stdout, tx.clone());
    }
    if let Some(stderr) = child.stderr.take() {
        spawn_drain(stderr, tx.clone());
    }
    drop(tx); // only the drain tasks can end the stream now

    let mut capture = ctx.spill.capture();
    let deadline = Instant::now() + Duration::from_millis(timeout_ms);
    let mut note = String::new();
    let mut killed = false;
    let mut exited: Option<std::io::Result<std::process::ExitStatus>> = None;
    let mut exit_drain_deadline: Option<Instant> = None;
    let mut kill_drain_deadline: Option<Instant> = None;
    let mut lingering = false;

    loop {
        if killed {
            // Post-kill drain is bounded by an absolute deadline: a process that
            // escaped the group and keeps writing cannot hold the call open.
            let until = *kill_drain_deadline.get_or_insert_with(|| Instant::now() + DRAIN_GRACE);
            match tokio::time::timeout_at(until, rx.recv()).await {
                Ok(Some(bytes)) => match capture.append(&bytes).await {
                    Ok(false) => continue,
                    _ => break,
                },
                _ => break,
            }
        }
        tokio::select! {
            _ = ctx.token.cancelled() => {
                kill_group(&mut guard, &mut child);
                note = "Command aborted.".to_owned();
                killed = true;
            }
            // A command that already exited is never reported as timed out; its
            // background processes are left alone.
            _ = sleep_until(deadline), if exited.is_none() => {
                kill_group(&mut guard, &mut child);
                note = timeout_note(timeout_ms);
                killed = true;
            }
            status = child.wait(), if exited.is_none() => {
                // Background processes may still hold the pipes: collect late output
                // briefly, then return instead of waiting out the whole timeout.
                exited = Some(status);
                exit_drain_deadline = Some(Instant::now() + EXIT_DRAIN_GRACE);
            }
            _ = sleep_until(exit_drain_deadline.unwrap_or_else(Instant::now)),
                if exit_drain_deadline.is_some() =>
            {
                lingering = true;
                break;
            }
            maybe = rx.recv() => match maybe {
                Some(bytes) => match capture.append(&bytes).await {
                    Ok(false) => {}
                    // Kill the producer at the capture cap; finish() reports
                    // truncation.
                    Ok(true) => {
                        kill_group(&mut guard, &mut child);
                        killed = true;
                    }
                    Err(err) => {
                        // On spill failure, kill the group and remove the partial file.
                        kill_group(&mut guard, &mut child);
                        if let Some(path) = capture.path() {
                            let _ = std::fs::remove_file(path);
                        }
                        return Err(err);
                    }
                },
                None => break, // every pipe closed
            },
        }
    }
    // Keep reading the pipes so background writers are not killed by SIGPIPE on
    // their next write; the output is discarded.
    tokio::spawn(async move { while rx.recv().await.is_some() {} });

    // Apply cancellation and the deadline even if the child has closed its pipes.
    let status = match exited {
        Some(status) => status,
        None => tokio::select! {
            status = child.wait() => status,
            _ = sleep_until(deadline), if !killed => {
                kill_group(&mut guard, &mut child);
                note = timeout_note(timeout_ms);
                child.wait().await
            }
            _ = ctx.token.cancelled(), if !killed => {
                kill_group(&mut guard, &mut child);
                note = "Command aborted.".to_owned();
                child.wait().await
            }
        },
    };
    // The command already ran: a full registry stops its background processes
    // but still reports the output and exit code.
    if !killed
        && let Some(pid) = guard.pid()
        && let Err(err) = ctx.processes.register(pid, scratch)
    {
        note = format!(
            "{err}; this command's background processes were stopped. Use process to stop unused ones."
        );
    }
    guard.disarm();
    if lingering && note.is_empty() {
        note = LINGER_NOTE.to_owned();
    }
    let status =
        status.map_err(|err| ToolError::failure(format!("cannot reap the command: {err}")))?;

    let code = status
        .code()
        .map_or_else(|| "killed".to_owned(), |code| code.to_string());
    // The trailer rides inside the capture bounds so truncation never drops it.
    let mut trailer = format!("\n\n[exit code {code}]");
    if !note.is_empty() {
        trailer.push('\n');
        trailer.push_str(&note);
    }
    let captured = capture.finish(&trailer).await?;
    Ok(captured.text)
}

fn timeout_note(timeout_ms: u64) -> String {
    format!(
        "Command killed after exceeding the {timeout_ms} ms timeout; retry with a larger timeout if it needs longer."
    )
}

/// Drains one pipe into the bounded capture queue.
fn spawn_drain<R>(mut source: R, tx: mpsc::Sender<Bytes>)
where
    R: tokio::io::AsyncRead + Unpin + Send + 'static,
{
    tokio::spawn(async move {
        let mut buf = vec![0u8; 64 * 1024];
        loop {
            match source.read(&mut buf).await {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    if tx.send(Bytes::copy_from_slice(&buf[..n])).await.is_err() {
                        break;
                    }
                }
            }
        }
    });
}

fn kill_group(guard: &mut GroupGuard, child: &mut Child) {
    guard.kill();
    // Fallback for the direct child (and the only lever on non-Unix).
    let _ = child.start_kill();
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_json::json;
    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::tool::{OutputSpill, Tool, Typed};

    fn ctx(dir: &std::path::Path) -> ToolCtx {
        ToolCtx::new(dir).with_spill(Arc::new(OutputSpill::new(dir.join("spill"))))
    }

    async fn bash(ctx: &ToolCtx, args: serde_json::Value) -> Result<String, ToolError> {
        Typed(BashTool).call(args, ctx).await
    }

    #[tokio::test]
    async fn runs_bash_whatever_the_login_shell() {
        let dir = tempfile::tempdir().unwrap();
        let out = bash(
            &ctx(dir.path()),
            json!({"command": "rm -f *.none; arr=(a b); echo \"$BASH_VERSION ${arr[0]}\""}),
        )
        .await
        .unwrap();
        assert!(out.contains(" a\n"), "{out}");
        assert!(out.ends_with("[exit code 0]"), "{out}");
    }

    #[tokio::test]
    async fn reports_exit_codes_and_kills_on_timeout() {
        let dir = tempfile::tempdir().unwrap();
        let out = bash(
            &ctx(dir.path()),
            json!({"command": "echo out; echo err >&2; exit 3"}),
        )
        .await
        .unwrap();
        assert!(out.ends_with("[exit code 3]"), "{out}");
        assert!(out.contains("out"), "{out}");
        assert!(out.contains("err"), "{out}");

        let started = std::time::Instant::now();
        let out = bash(
            &ctx(dir.path()),
            json!({"command": "sleep 5 & sleep 5", "timeout": 200}),
        )
        .await
        .unwrap();
        assert!(
            out.contains("[exit code killed]\nCommand killed after exceeding the 200 ms timeout"),
            "{out}"
        );
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "the kill must not wait for sleep 5"
        );
    }

    #[tokio::test]
    async fn timeout_kills_descendants_not_just_the_direct_child() {
        let dir = tempfile::tempdir().unwrap();
        let marker = dir.path().join("marker");
        let command = format!("(sleep 2; touch {}) & sleep 10", marker.display());
        let out = bash(
            &ctx(dir.path()),
            json!({"command": command, "timeout": 300}),
        )
        .await
        .unwrap();
        assert!(out.contains("[exit code killed]"), "{out}");
        // The backgrounded subshell shares the process group: after its
        // scheduled wake-up, the marker must never appear.
        tokio::time::sleep(Duration::from_millis(2_600)).await;
        assert!(!marker.exists(), "a group member outlived the kill");
    }

    #[tokio::test]
    async fn interruption_kills_the_group_and_reports_the_abort() {
        let dir = tempfile::tempdir().unwrap();
        let token = CancellationToken::new();
        let ctx = ctx(dir.path()).with_token(token.clone());
        let canceller = token.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(200)).await;
            canceller.cancel();
        });
        let started = std::time::Instant::now();
        let out = bash(&ctx, json!({"command": "sleep 30"})).await.unwrap();
        assert!(out.contains("[exit code killed]"), "{out}");
        assert!(out.contains("Command aborted."), "{out}");
        assert!(started.elapsed() < Duration::from_secs(10));
    }

    #[tokio::test]
    async fn a_failed_spill_write_stops_the_capture_cleanly() {
        let dir = tempfile::tempdir().unwrap();
        // The spill "directory" is a regular file: creating the spill fails.
        let blocker = dir.path().join("blocker");
        std::fs::write(&blocker, "not a directory").unwrap();
        let ctx = ToolCtx::new(dir.path()).with_spill(Arc::new(OutputSpill::new(&blocker)));
        let started = std::time::Instant::now();
        let err = bash(
            &ctx,
            json!({"command": "head -c 60000 /dev/zero | tr '\\0' 'x'", "timeout": 30000}),
        )
        .await
        .unwrap_err();
        let message = err.to_string();
        assert!(
            matches!(err, ToolError::Failure(_)),
            "model-visible failure: {message}"
        );
        assert!(message.contains("spill directory"), "{message}");
        assert!(
            started.elapsed() < Duration::from_secs(20),
            "the child was killed, not waited out"
        );
    }

    #[tokio::test]
    async fn the_capture_cap_stops_the_command_and_is_reported() {
        let dir = tempfile::tempdir().unwrap();
        let spill = Arc::new(OutputSpill::new(dir.path().join("spill")).with_capture_cap(60_000));
        let ctx = ToolCtx::new(dir.path()).with_spill(spill);
        let out = bash(&ctx, json!({"command": "yes x", "timeout": 30000}))
            .await
            .unwrap();
        assert!(out.contains("capture cap"), "{out}");
        assert!(out.contains("[exit code killed]"), "{out}");
        let spilled: Vec<_> = std::fs::read_dir(dir.path().join("spill"))
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| {
                path.file_name()
                    .unwrap()
                    .to_string_lossy()
                    .starts_with("output-")
            })
            .collect();
        assert_eq!(spilled.len(), 1);
        let size = std::fs::metadata(&spilled[0]).unwrap().len();
        assert!(size == 60_000, "the file holds the captured prefix: {size}");
    }

    #[tokio::test]
    async fn simultaneous_stdout_and_stderr_cannot_deadlock() {
        let dir = tempfile::tempdir().unwrap();
        // 200 KiB on each pipe at the same time: a single-pipe drain would
        // deadlock on the full pipe buffer.
        let command = "(head -c 200000 /dev/zero | tr '\\0' 'o' &) ; head -c 200000 /dev/zero | tr '\\0' 'e' >&2 ; sleep 1";
        let out = bash(
            &ctx(dir.path()),
            json!({"command": command, "timeout": 30000}),
        )
        .await
        .unwrap();
        assert!(out.contains("[exit code 0]"), "{out}");
        assert!(
            out.contains("lines truncated. Full output saved to:"),
            "{out}"
        );
        let path = out
            .split_once("Full output saved to: ")
            .unwrap()
            .1
            .lines()
            .next()
            .unwrap();
        let spilled = std::fs::read_to_string(path).unwrap();
        assert!(
            spilled.contains('o') && spilled.contains('e'),
            "both pipes were captured"
        );
        assert!(spilled.len() >= 390_000, "captured {}", spilled.len());
    }

    #[tokio::test]
    async fn large_output_keeps_the_exit_code_within_the_bound() {
        use crate::tool::output::{MAX_BYTES, MAX_LINES};
        let dir = tempfile::tempdir().unwrap();
        let out = bash(
            &ctx(dir.path()),
            json!({"command": "seq 1 3000; exit 3", "timeout": 30000}),
        )
        .await
        .unwrap();
        assert!(
            out.ends_with("[exit code 3]"),
            "{}",
            &out[out.len() - 200..]
        );
        assert!(out.lines().count() <= MAX_LINES, "{}", out.lines().count());
        assert!(out.len() <= MAX_BYTES, "{}", out.len());
        assert_eq!(out.matches("Full output saved to").count(), 1, "{out}");

        // A killed command with a large log still reports how it ended.
        let out = bash(
            &ctx(dir.path()),
            json!({"command": "seq 1 3000; sleep 5", "timeout": 300}),
        )
        .await
        .unwrap();
        assert!(
            out.ends_with("[exit code killed]\nCommand killed after exceeding the 300 ms timeout; retry with a larger timeout if it needs longer."),
            "{}",
            &out[out.len() - 200..]
        );
        assert!(out.lines().count() <= MAX_LINES);
        assert!(out.len() <= MAX_BYTES);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_full_process_registry_still_reports_the_command_result() {
        use std::os::unix::process::CommandExt;
        let dir = tempfile::tempdir().unwrap();
        let owner = ctx(dir.path());
        let mut sleepers = Vec::new();
        for _ in 0..32 {
            let child = std::process::Command::new("sleep")
                .arg("30")
                .process_group(0)
                .spawn()
                .unwrap();
            owner.processes.register(child.id(), None).unwrap();
            sleepers.push(child);
        }
        let out = bash(
            &owner,
            json!({"command": "sleep 5 & echo built", "timeout": 30000}),
        )
        .await
        .unwrap();
        assert!(out.starts_with("built"), "{out}");
        assert!(out.contains("[exit code 0]"), "{out}");
        assert!(out.contains("background process limit reached"), "{out}");
        owner.processes.stop_all();
        for mut child in sleepers {
            let _ = child.wait();
        }
    }

    #[tokio::test]
    async fn backgrounded_processes_do_not_block_the_call() {
        let dir = tempfile::tempdir().unwrap();
        let started = std::time::Instant::now();
        let out = bash(
            &ctx(dir.path()),
            json!({"command": "sleep 5 & echo started", "timeout": 30000}),
        )
        .await
        .unwrap();
        assert!(out.starts_with("started"), "{out}");
        assert!(out.contains("[exit code 0]"), "{out}");
        assert!(out.ends_with(LINGER_NOTE), "{out}");
        assert!(
            started.elapsed() < Duration::from_secs(4),
            "returned at exit plus the drain grace, not at the pipe EOF: {:?}",
            started.elapsed()
        );
    }

    #[tokio::test]
    async fn background_writers_survive_the_return() {
        let dir = tempfile::tempdir().unwrap();
        let marker = dir.path().join("marker");
        let owner = ctx(dir.path());
        let out = bash(
            &owner,
            json!({"command": "(for i in 1 2 3 4 5 6; do echo bg; sleep 0.3; done; touch marker) & echo started", "timeout": 30000}),
        )
        .await
        .unwrap();
        assert!(out.ends_with(LINGER_NOTE), "{out}");
        let started = std::time::Instant::now();
        while !marker.exists() {
            assert!(
                started.elapsed() < Duration::from_secs(10),
                "the background loop died after bash returned (SIGPIPE on its next write)"
            );
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    #[tokio::test]
    async fn the_timeout_does_not_fire_during_the_exit_drain() {
        let dir = tempfile::tempdir().unwrap();
        let marker = dir.path().join("marker");
        let owner = ctx(dir.path());
        let out = bash(
            &owner,
            json!({"command": "(sleep 1.5; touch marker) & echo hi", "timeout": 800}),
        )
        .await
        .unwrap();
        assert!(out.contains("[exit code 0]"), "{out}");
        assert!(
            !out.contains("timeout"),
            "an exited command was reported as timed out: {out}"
        );
        assert!(out.ends_with(LINGER_NOTE), "{out}");
        let started = std::time::Instant::now();
        while !marker.exists() {
            assert!(
                started.elapsed() < Duration::from_secs(10),
                "the background process was killed by the timeout after the command exited"
            );
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    #[tokio::test]
    async fn the_post_kill_drain_is_bounded_despite_an_escaped_writer() {
        let dir = tempfile::tempdir().unwrap();
        let started = std::time::Instant::now();
        // The perl loop leaves the process group, so the kill misses it, and it
        // writes more often than the old idle timeout.
        let out = bash(
            &ctx(dir.path()),
            json!({
                "command": "perl -e 'setpgrp(0,0); $|=1; for (1..8) { print \"tick\\n\"; sleep 2 }' & sleep 60",
                "timeout": 500
            }),
        )
        .await
        .unwrap();
        assert!(
            out.contains("killed after exceeding the 500 ms timeout"),
            "{out}"
        );
        assert!(
            started.elapsed() < DRAIN_GRACE + Duration::from_secs(3),
            "the drain ran past its bound: {:?}",
            started.elapsed()
        );
    }

    #[tokio::test]
    async fn quiet_commands_report_no_output() {
        let dir = tempfile::tempdir().unwrap();
        let out = bash(&ctx(dir.path()), json!({"command": "true"}))
            .await
            .unwrap();
        assert_eq!(out, "(no output)\n\n[exit code 0]");
    }

    #[tokio::test]
    async fn workdir_runs_in_the_given_directory() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        let out = bash(
            &ctx(dir.path()),
            json!({"command": "pwd", "workdir": "sub"}),
        )
        .await
        .unwrap();
        // macOS tempdirs live behind a /var -> /private/var symlink; pwd
        // reports the physical path.
        let expected = dir.path().canonicalize().unwrap().join("sub");
        assert!(
            out.starts_with(&expected.to_string_lossy().into_owned()),
            "{out}"
        );
    }

    #[tokio::test]
    async fn an_unspawnable_command_is_a_model_visible_failure() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolCtx::new(dir.path())
            .with_spill(Arc::new(OutputSpill::new(dir.path().join("spill"))));
        let err = bash(&ctx, json!({"command": "true", "workdir": "no/such/dir"}))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Failure(_)), "{err}");
    }
}
