//! Execution policy hooks and fail-closed subprocess isolation for the CLI.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::Value;
use tokio::process::Command;

use super::{Tool, ToolCtx, ToolError};

#[derive(Debug, Clone, Copy, Default, clap::ValueEnum, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExecutionMode {
    /// Only workspace files; subprocesses require an available OS sandbox.
    Sandbox,
    /// Explicit host access, intended for an embedder's trusted tools.
    #[default]
    Trusted,
}

/// Called before every tool, including MCP tools and delegated work. An embedder
/// may present an approval UI here; returning an error prevents execution.
#[async_trait::async_trait]
pub trait ToolPolicy: Send + Sync {
    async fn authorize(&self, name: &str, args: &Value, ctx: &ToolCtx) -> Result<(), ToolError>;
}

pub struct AllowTools;
#[async_trait::async_trait]
impl ToolPolicy for AllowTools {
    async fn authorize(&self, _: &str, _: &Value, _: &ToolCtx) -> Result<(), ToolError> {
        Ok(())
    }
}

/// CLI grants are supplied by the caller, never by model arguments.
pub struct CliPolicy {
    pub allow_mcp: bool,
}

#[async_trait::async_trait]
impl ToolPolicy for CliPolicy {
    async fn authorize(&self, name: &str, args: &Value, ctx: &ToolCtx) -> Result<(), ToolError> {
        match name {
            "read" | "write" | "edit" => {
                if let Some(path) = args.get("filePath").and_then(Value::as_str) {
                    ctx.checked_path(path)?;
                }
            }
            "grep" | "glob" | "bash" => {
                let key = if name == "bash" { "workdir" } else { "path" };
                ctx.checked_path(args.get(key).and_then(Value::as_str).unwrap_or("."))?;
            }
            "skill" | "task" | "process" => {}
            _ if self.allow_mcp => {}
            _ => {
                return Err(ToolError::failure(
                    "external tools require an explicit grant (--allow-mcp)",
                ));
            }
        }
        Ok(())
    }
}

/// Resolve existing symlinks and the closest existing ancestor of a new path.
pub fn workspace_path(cwd: &Path, requested: &str) -> Result<PathBuf, ToolError> {
    let root = cwd
        .canonicalize()
        .map_err(|_| ToolError::failure("cannot resolve workspace"))?;
    let joined = root.join(requested);
    let mut normalized = PathBuf::new();
    for component in joined.components() {
        match component {
            std::path::Component::ParentDir => {
                normalized.pop();
            }
            std::path::Component::CurDir => {}
            component => normalized.push(component.as_os_str()),
        }
    }
    let mut ancestor = normalized.as_path();
    let mut tail = Vec::new();
    let resolved = loop {
        match ancestor.canonicalize() {
            Ok(path) => break path,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                // Broken symlinks must not be treated as new paths.
                if std::fs::symlink_metadata(ancestor).is_ok() {
                    return Err(ToolError::failure("cannot resolve symbolic link"));
                }
                tail.push(
                    ancestor
                        .file_name()
                        .ok_or_else(|| ToolError::failure("invalid path"))?
                        .to_owned(),
                );
                ancestor = ancestor
                    .parent()
                    .ok_or_else(|| ToolError::failure("invalid path"))?;
            }
            Err(_) => return Err(ToolError::failure("cannot resolve workspace path")),
        }
    };
    if !resolved.starts_with(&root) {
        return Err(ToolError::failure(
            "path is outside the workspace; host access requires trusted execution",
        ));
    }
    let mut path = resolved;
    for part in tail.into_iter().rev() {
        path.push(part);
    }
    Ok(path)
}

/// Fail before model/provider work when the selected sandbox cannot start.
pub async fn preflight(cwd: &Path) -> Result<(), ToolError> {
    let mut isolated = subprocess("/usr/bin/true", &[], cwd, ExecutionMode::Sandbox, false)?;
    isolated
        .command
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped());
    let output = isolated.command.output().await.map_err(|_| {
        ToolError::failure(
            "OS sandbox unavailable; Linux requires /usr/bin/bwrap and enabled user namespaces",
        )
    })?;
    if !output.status.success() {
        return Err(ToolError::failure(format!(
            "OS sandbox cannot start: {}",
            String::from_utf8_lossy(&output.stderr)
                .chars()
                .take(500)
                .collect::<String>()
        )));
    }
    Ok(())
}

/// Mach services ordinary command-line tools need: user and group lookup,
/// logging, preferences, TLS trust evaluation and name resolution.
/// LaunchServices, Apple Events and the pasteboard are deliberately absent:
/// they would let a sandboxed command open unsandboxed applications or URLs,
/// script other applications, or read the clipboard.
#[cfg(target_os = "macos")]
const MACH_SERVICES: &[&str] = &[
    "com.apple.system.opendirectoryd.libinfo",
    "com.apple.system.opendirectoryd.membership",
    "com.apple.system.DirectoryService.libinfo_v1",
    "com.apple.system.DirectoryService.membership_v1",
    "com.apple.system.libinfo.muser",
    "com.apple.system.logger",
    "com.apple.logd",
    "com.apple.diagnosticd",
    "com.apple.system.notification_center",
    "com.apple.SecurityServer",
    "com.apple.securityd.xpc",
    "com.apple.trustd",
    "com.apple.trustd.agent",
    "com.apple.ocspd",
    "com.apple.networkd",
    "com.apple.nehelper",
    "com.apple.nesessionmanager",
    "com.apple.mDNSResponder",
    "com.apple.dnssd.service",
    "com.apple.SystemConfiguration.configd",
    "com.apple.SystemConfiguration.DNSConfiguration",
    "com.apple.SystemConfiguration.NetworkInformation",
    "com.apple.cfprefsd.daemon",
    "com.apple.cfprefsd.agent",
    "com.apple.bsd.dirhelper",
    "com.apple.FSEvents",
    "com.apple.xpc.activity.unmanaged",
];

/// Environment kept for sandboxed subprocesses (matched case-insensitively).
const PASSED_ENV: &[&str] = &[
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "NODE_EXTRA_CA_CERTS",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
];

/// The command runs in its own process group whose id is the child pid, so one
/// `killpg` reaches every descendant. The temporary directory stays alive for
/// the subprocess lifetime.
pub struct IsolatedCommand {
    pub command: Command,
    pub(crate) _scratch: Option<tempfile::TempDir>,
}

pub fn subprocess(
    program: &str,
    args: &[String],
    cwd: &Path,
    mode: ExecutionMode,
    network: bool,
) -> Result<IsolatedCommand, ToolError> {
    if mode == ExecutionMode::Trusted {
        let mut command = Command::new(program);
        command.args(args).current_dir(cwd);
        #[cfg(unix)]
        command.process_group(0);
        return Ok(IsolatedCommand {
            command,
            _scratch: None,
        });
    }
    let cwd = cwd
        .canonicalize()
        .map_err(|_| ToolError::failure("cannot resolve workspace"))?;
    let scratch = tempfile::tempdir()
        .map_err(|_| ToolError::failure("cannot create sandbox scratch directory"))?;
    let scratch_path = scratch
        .path()
        .canonicalize()
        .map_err(|_| ToolError::failure("cannot resolve sandbox scratch directory"))?;
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    let mut command;
    #[cfg(target_os = "macos")]
    {
        // Deny by default; grants cover the workspace, scratch space, and system
        // runtimes. Apple's dyld bootstrap rules are imported so newly built
        // executables can start on current macOS versions. The CLI's file tools also validate workspace paths and execute
        // inside this sandbox, so a concurrent symlink swap cannot reach the host.
        let quote =
            |path: &Path| serde_json::to_string(&path.to_string_lossy()).expect("path string");
        let executable = if Path::new(program).is_absolute() {
            Some(PathBuf::from(program))
        } else {
            None
        };
        let services = MACH_SERVICES
            .iter()
            .map(|name| format!("(global-name \"{name}\")"))
            .collect::<String>();
        let mut profile = format!(
            "(version 1)(deny default)(import \"dyld-support.sb\")(allow syscall*)(allow process-fork)(allow process-exec)(allow process-info*)(allow sysctl-read)(allow mach-lookup {services})(allow file-read-metadata)(allow file-read* file-map-executable (subpath \"/usr\") (subpath \"/bin\") (subpath \"/sbin\") (subpath \"/opt\") (subpath \"/System\") (subpath \"/Library\") (regex #\"^/Applications/Xcode[^/]*\\.app/\") (literal \"/private/var/db/xcode_select_link\") (literal \"/private/etc/ssl/openssl.cnf\") (literal \"/private/etc/ssl/cert.pem\") (literal \"/private/etc/resolv.conf\") (literal \"/private/etc/hosts\") (subpath \"/private/etc/ssl/certs\") (literal \"/private/var/run/resolv.conf\") (subpath {}))(allow file* (subpath {}))(allow file* (subpath {}))(allow file-read* file-write* (literal \"/dev/null\") (literal \"/dev/urandom\") (literal \"/dev/random\") (literal \"/dev/zero\") (literal \"/dev/stdin\") (literal \"/dev/stdout\") (literal \"/dev/stderr\") (subpath \"/dev/fd\"))",
            quote(&cwd),
            quote(&cwd),
            quote(&scratch_path)
        );
        if let Some(executable) = executable {
            profile.push_str(&format!(
                "(allow file-read* file-map-executable (literal {}))",
                quote(&executable)
            ));
        }
        // Run state (the session database) stays private; only spilled tool
        // output is readable, so the model can page it.
        profile.push_str(&format!(
            "(deny file-read* file-write* (subpath {}))(allow file-read* (subpath {}))",
            quote(&cwd.join(".mason")),
            quote(&cwd.join(".mason/output"))
        ));
        if network {
            profile.push_str("(allow network*)");
        }
        command = Command::new("/usr/bin/sandbox-exec");
        command.args(["-p", &profile, program]).args(args);
    }
    #[cfg(target_os = "linux")]
    {
        // No --die-with-parent: bwrap exits with the command while its pid
        // namespace lives on, so background processes survive the call and are
        // stopped through the process group like on macOS. No --new-session
        // either, as that would leave the group; the controlling terminal is
        // masked instead (macOS denies /dev/tty through the profile).
        command = Command::new("/usr/bin/bwrap");
        command.args([
            "--unshare-all",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--dev-bind",
            "/dev/null",
            "/dev/tty",
            "--dev-bind",
            "/dev/null",
            "/dev/console",
        ]);
        for system in ["/usr", "/bin", "/sbin", "/lib", "/lib64"] {
            if Path::new(system).exists() {
                command.args(["--ro-bind", system, system]);
            }
        }
        for configuration in [
            "/etc/ld.so.cache",
            "/etc/localtime",
            "/etc/resolv.conf",
            "/etc/hosts",
            "/etc/nsswitch.conf",
            "/etc/passwd",
            "/etc/group",
            "/etc/ssl/certs",
            "/etc/ssl/openssl.cnf",
            "/etc/alternatives",
        ] {
            if Path::new(configuration).exists() {
                command.args(["--ro-bind", configuration, configuration]);
            }
        }
        command.arg("--bind").arg(&cwd).arg(&cwd);
        // Run state (the session database) stays private; only spilled tool
        // output is readable, so the model can page it.
        if cwd.join(".mason").is_dir() {
            command.arg("--tmpfs").arg(cwd.join(".mason"));
            if cwd.join(".mason/output").is_dir() {
                command
                    .arg("--ro-bind")
                    .arg(cwd.join(".mason/output"))
                    .arg(cwd.join(".mason/output"));
            }
        }
        command.arg("--bind").arg(&scratch_path).arg(&scratch_path);
        if Path::new(program).is_absolute() && !Path::new(program).starts_with(&cwd) {
            command.arg("--ro-bind").arg(program).arg(program);
        }
        if network {
            command.arg("--share-net");
        }
        // The ephemeral root would otherwise accept writes anywhere outside the
        // binds; like macOS, only the workspace, scratch space and devices are
        // writable. The working directory is inherited (callers may set a
        // workspace subdirectory), so no --chdir.
        command
            .args(["--remount-ro", "/", "--"])
            .arg(program)
            .args(args);
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    return Err(ToolError::failure(
        "sandbox execution is supported on macOS and Linux; trusted execution requires explicit opt-in",
    ));
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    {
        command
            .current_dir(&cwd)
            .env_clear()
            .env("PATH", std::env::var_os("PATH").unwrap_or_default())
            .env("HOME", &scratch_path)
            .env("TMPDIR", &scratch_path)
            .env("LANG", "en_US.UTF-8");
        // Proxy and CA settings so network-granted tools work behind a firewall.
        for (key, value) in std::env::vars_os() {
            if key
                .to_str()
                .is_some_and(|key| PASSED_ENV.contains(&key.to_ascii_uppercase().as_str()))
            {
                command.env(key, value);
            }
        }
        // The group that bwrap's pid namespace inherits; one killpg reaches all.
        command.process_group(0);
        Ok(IsolatedCommand {
            command,
            _scratch: Some(scratch),
        })
    }
}

/// Isolate Rust filesystem tools too, rather than relying on path checks alone.
pub struct IsolatedFileTool(pub Arc<dyn Tool>);
#[async_trait::async_trait]
impl Tool for IsolatedFileTool {
    fn spec(&self) -> crate::llm::ToolSpec {
        self.0.spec()
    }
    async fn call(&self, args: Value, ctx: &ToolCtx) -> Result<String, ToolError> {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let executable = std::env::current_exe()
            .map_err(|_| ToolError::failure("cannot locate Mason executable"))?;
        let name = self.spec().function.name;
        // Workers are separate processes, so the per-path lock that serializes
        // read-modify-write cycles must be held here.
        let _guard = match args.get("filePath").and_then(Value::as_str) {
            Some(path) if matches!(name.as_str(), "write" | "edit") => {
                Some(super::lock_path(&ctx.checked_path(path)?).await)
            }
            _ => None,
        };
        let mut isolated = subprocess(
            executable.to_string_lossy().as_ref(),
            &["tool-worker".into(), name],
            &ctx.cwd,
            ExecutionMode::Sandbox,
            false,
        )?;
        isolated
            .command
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);
        let mut child = isolated.command.spawn().map_err(|_| {
            ToolError::failure("cannot start filesystem sandbox; install bubblewrap on Linux")
        })?;
        let mut group = super::process::GroupGuard::new(child.id());
        let mut stdin = child.stdin.take().expect("piped stdin");
        let stdout = child.stdout.take().expect("piped stdout");
        let stderr = child.stderr.take().expect("piped stderr");
        let input = serde_json::to_vec(&args)
            .map_err(|_| ToolError::failure("cannot encode tool arguments"))?;
        let writer = tokio::spawn(async move { stdin.write_all(&input).await });
        let collect = async {
            let mut out = Vec::new();
            let mut err = Vec::new();
            let read_out = async { stdout.take(1024 * 1024 + 1).read_to_end(&mut out).await };
            let read_err = async { stderr.take(4097).read_to_end(&mut err).await };
            let (out_result, err_result) = tokio::join!(read_out, read_err);
            out_result.map_err(|_| ToolError::failure("cannot read worker output"))?;
            err_result.map_err(|_| ToolError::failure("cannot read worker diagnostics"))?;
            if out.len() > 1024 * 1024 || err.len() > 4096 {
                return Err(ToolError::failure(
                    "filesystem worker output exceeded its byte limit",
                ));
            }
            let status = child
                .wait()
                .await
                .map_err(|_| ToolError::failure("filesystem worker failed"))?;
            Ok::<_, ToolError>((status, out, err))
        };
        let collected = tokio::select! {
            _ = tokio::time::sleep(std::time::Duration::from_secs(120)) => Err(ToolError::failure("filesystem worker exceeded its deadline")),
            _ = ctx.token.cancelled() => Err(ToolError::Fatal(crate::Error::Cancelled)),
            output = collect => output,
        };
        writer.abort();
        let (status, stdout, stderr) = collected?;
        group.disarm();
        if !status.success() {
            return Err(ToolError::failure(format!(
                "filesystem sandbox denied the operation or could not start ({status}): {}",
                String::from_utf8_lossy(&stderr)
                    .chars()
                    .take(500)
                    .collect::<String>()
            )));
        }
        let result: Result<String, String> = serde_json::from_slice(&stdout)
            .map_err(|_| ToolError::failure("invalid filesystem sandbox response"))?;
        result.map_err(ToolError::failure)
    }
}
