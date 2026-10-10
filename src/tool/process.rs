//! Background process groups owned by a session (or a one-shot run). Dropping the
//! registry kills surviving groups.
use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

use schemars::JsonSchema;
use serde::Deserialize;

use super::{ToolCtx, ToolError, TypedTool};

/// A group whose last member exited is forgotten within this interval, before
/// the kernel can hand its id to an unrelated process.
const EXIT_POLL: Duration = Duration::from_secs(1);

#[derive(Default)]
pub struct BackgroundProcesses {
    entries: Mutex<HashMap<u32, Option<tempfile::TempDir>>>,
}

impl BackgroundProcesses {
    /// Tracks a surviving group until it exits or the registry stops it.
    pub fn register(
        self: &Arc<Self>,
        pid: u32,
        scratch: Option<tempfile::TempDir>,
    ) -> Result<(), ToolError> {
        {
            let mut entries = self.entries.lock().expect("process registry mutex");
            entries.retain(|pid, _| alive(*pid));
            if !alive(pid) {
                return Ok(());
            }
            if entries.len() >= 32 {
                kill(pid);
                return Err(ToolError::failure("background process limit reached"));
            }
            entries.insert(pid, scratch);
        }
        tokio::spawn(watch(Arc::downgrade(self), pid));
        Ok(())
    }

    pub fn list(&self) -> Vec<u32> {
        let mut entries = self.entries.lock().expect("process registry mutex");
        entries.retain(|pid, _| alive(*pid));
        let mut pids: Vec<_> = entries.keys().copied().collect();
        pids.sort_unstable();
        pids
    }

    pub fn stop(&self, pid: u32) -> bool {
        let mut entries = self.entries.lock().expect("process registry mutex");
        if entries.remove(&pid).is_some() {
            kill(pid);
            true
        } else {
            false
        }
    }

    pub fn stop_all(&self) {
        let mut entries = self.entries.lock().expect("process registry mutex");
        for (pid, _) in entries.drain() {
            kill(pid);
        }
    }

    /// Drops the entry once every member of the group has exited.
    fn forget_exited(&self, pid: u32) -> bool {
        let mut entries = self.entries.lock().expect("process registry mutex");
        if alive(pid) {
            return false;
        }
        entries.remove(&pid);
        true
    }
}

async fn watch(registry: Weak<BackgroundProcesses>, pid: u32) {
    loop {
        tokio::time::sleep(EXIT_POLL).await;
        let Some(registry) = registry.upgrade() else {
            return;
        };
        if registry.forget_exited(pid) {
            return;
        }
    }
}

impl Drop for BackgroundProcesses {
    fn drop(&mut self) {
        self.stop_all();
    }
}

#[cfg(unix)]
fn alive(pid: u32) -> bool {
    nix::sys::signal::killpg(nix::unistd::Pid::from_raw(pid as i32), None).is_ok()
}
#[cfg(unix)]
fn kill(pid: u32) {
    let _ = nix::sys::signal::killpg(
        nix::unistd::Pid::from_raw(pid as i32),
        nix::sys::signal::Signal::SIGKILL,
    );
}
#[cfg(not(unix))]
fn alive(_: u32) -> bool {
    false
}
#[cfg(not(unix))]
fn kill(_: u32) {}

/// Kills a child's Unix process group when dropped. Disarm right after reaping
/// the child: once its pid is free the group id may belong to someone else.
#[cfg(unix)]
pub(crate) struct GroupGuard(Option<u32>);

#[cfg(unix)]
impl GroupGuard {
    pub(crate) fn new(pid: Option<u32>) -> Self {
        Self(pid)
    }

    pub(crate) fn pid(&self) -> Option<u32> {
        self.0
    }

    pub(crate) fn kill(&mut self) {
        if let Some(pid) = self.0 {
            kill(pid);
        }
    }

    pub(crate) fn disarm(&mut self) {
        self.0 = None;
    }
}

#[cfg(unix)]
impl Drop for GroupGuard {
    fn drop(&mut self) {
        self.kill();
    }
}

/// Non-Unix fallback: callers only kill the direct child.
#[cfg(not(unix))]
pub(crate) struct GroupGuard;

#[cfg(not(unix))]
impl GroupGuard {
    pub(crate) fn new(_pid: Option<u32>) -> Self {
        Self
    }

    pub(crate) fn pid(&self) -> Option<u32> {
        None
    }

    pub(crate) fn kill(&mut self) {}

    pub(crate) fn disarm(&mut self) {}
}

pub struct ProcessTool;
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ProcessArgs {
    pub action: ProcessAction,
    pub pid: Option<u32>,
}
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum ProcessAction {
    List,
    Stop,
}

#[async_trait::async_trait]
impl TypedTool for ProcessTool {
    type Args = ProcessArgs;
    fn name(&self) -> &str {
        "process"
    }
    fn description(&self) -> &str {
        "List or stop background shell process groups owned by this session. They keep running between messages until stopped."
    }
    async fn run(&self, args: ProcessArgs, ctx: &ToolCtx) -> Result<String, ToolError> {
        match args.action {
            ProcessAction::List => {
                Ok(serde_json::to_string(&ctx.processes.list()).expect("pid list"))
            }
            ProcessAction::Stop => {
                let pid = args
                    .pid
                    .ok_or_else(|| ToolError::failure("pid is required to stop a process"))?;
                if !ctx.processes.stop(pid) {
                    return Err(ToolError::failure("process is not owned by this session"));
                }
                Ok(format!("Stopped process group {pid}"))
            }
        }
    }
}
