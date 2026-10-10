//! Stdio and streamable HTTP MCP servers, exposing tools as `<server>_<tool>`.
//! Each run connects its own servers; unavailable ones are reported as failures.
//! Deadlines and cancellation do not roll back remote effects.
//!
//! Configuration uses opencode's server-map format:
//! ```json
//! {
//!   "files": { "type": "local", "command": ["some-mcp-server"] },
//!   "search": { "type": "remote", "url": "https://example.com/mcp" }
//! }
//! ```

use std::collections::{BTreeMap, HashMap};
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use http::{HeaderName, HeaderValue};
use rmcp::model::{
    CallToolRequest, CallToolRequestParams, CancelledNotificationParam, ClientRequest,
    ContentBlock, ResourceContents, ServerResult,
};
use rmcp::service::{PeerRequestOptions, RunningService};
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use rmcp::transport::{IntoTransport, StreamableHttpClientTransport, TokioChildProcess};
use rmcp::{Peer, RoleClient, ServiceExt};
use serde::Deserialize;
use serde_json::Value;

use crate::error::Error;
use crate::llm::ToolSpec;
use crate::tool::process::GroupGuard;
use crate::tool::{Tool, ToolCtx, ToolError};

/// Default per-server connect and discovery timeout.
pub const DEFAULT_TIMEOUT_MS: u64 = 30_000;
/// Longest tool name providers accept.
const MAX_TOOL_NAME: usize = 64;

type Running = RunningService<RoleClient, ()>;

/// MCP server configuration in opencode format.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum McpServerConfig {
    /// A local server spawned as a child process speaking MCP over stdio.
    Local {
        command: Vec<String>,
        /// Working directory for the child.
        #[serde(default)]
        cwd: Option<String>,
        /// Extra environment variables (merged over the inherited env).
        #[serde(default)]
        environment: Option<HashMap<String, String>>,
        /// `false` skips the server entirely.
        #[serde(default)]
        enabled: Option<bool>,
        /// Timeout in ms for connecting and listing tools; calls are bounded by the run.
        #[serde(default)]
        timeout: Option<u64>,
    },
    /// A remote server speaking streamable HTTP.
    Remote {
        url: String,
        /// Headers sent with every request (e.g. `Authorization`).
        #[serde(default)]
        headers: Option<HashMap<String, String>>,
        /// `false` skips the server entirely.
        #[serde(default)]
        enabled: Option<bool>,
        /// Timeout in ms for connecting and listing tools; calls are bounded by the run.
        #[serde(default)]
        timeout: Option<u64>,
    },
}

impl McpServerConfig {
    /// Whether the server participates (only an explicit `false` disables).
    pub fn enabled(&self) -> bool {
        !matches!(
            self,
            McpServerConfig::Local {
                enabled: Some(false),
                ..
            } | McpServerConfig::Remote {
                enabled: Some(false),
                ..
            }
        )
    }

    /// The configured connect and discovery timeout, as in opencode, or the
    /// 30-second default (also for `0`, which would otherwise disable the server
    /// silently). Calls are bounded by the run, not by this timeout.
    pub fn timeout(&self) -> Duration {
        let millis = match self {
            McpServerConfig::Local { timeout, .. } | McpServerConfig::Remote { timeout, .. } => {
                timeout
                    .filter(|millis| *millis > 0)
                    .unwrap_or(DEFAULT_TIMEOUT_MS)
            }
        };
        Duration::from_millis(millis)
    }
}

/// One connected MCP server and the tools it exposed.
pub struct McpConnection {
    name: String,
    /// Declared before `client` so the group dies before the child is reaped.
    group: Option<GroupGuard>,
    client: Running,
    _scratch: Option<tempfile::TempDir>,
    tools: Vec<Arc<dyn Tool>>,
}

impl McpConnection {
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The server's tools, named `<server>_<tool>`.
    pub fn tools(&self) -> &[Arc<dyn Tool>] {
        &self.tools
    }

    /// Whether the server went away (a crashed or exited process, a dropped stream).
    fn is_closed(&self) -> bool {
        self.client.peer().is_transport_closed()
    }

    /// Closes the connection and, for local servers, stops the child process
    /// group.
    pub async fn close(mut self) {
        if let Some(group) = &mut self.group {
            group.kill();
        }
        let _ = self.client.cancel().await;
    }
}

/// The tools of the configured MCP servers. Each tool connects its server
/// once per run scope; configured servers that cannot connect are failures.
pub struct McpServers {
    tools: Vec<Arc<dyn Tool>>,
    failures: Vec<String>,
}

impl McpServers {
    /// Discovers the tools of every enabled server in key order.
    pub async fn connect(
        configs: &BTreeMap<String, McpServerConfig>,
        mode: crate::tool::policy::ExecutionMode,
        network: bool,
    ) -> Self {
        let mut tools: Vec<Arc<dyn Tool>> = Vec::new();
        let mut failures = Vec::new();
        for (name, config) in configs.iter().filter(|(_, config)| config.enabled()) {
            match connect_config(name, config, mode, network).await {
                Ok(connection) => {
                    for tool in connection.tools() {
                        let spec = tool.spec();
                        if tools
                            .iter()
                            .any(|existing| existing.spec().function.name == spec.function.name)
                        {
                            failures
                                .push(format!("MCP tool name collision: {}", spec.function.name));
                            continue;
                        }
                        tools.push(Arc::new(ScopedMcpTool {
                            spec,
                            server: name.clone(),
                            config: config.clone(),
                            mode,
                            network,
                        }));
                    }
                    connection.close().await;
                }
                Err(error) => {
                    let error = format!("MCP server {name:?} unavailable: {error}");
                    tracing::warn!("{error}");
                    failures.push(error);
                }
            }
        }
        Self { tools, failures }
    }

    pub fn failures(&self) -> &[String] {
        &self.failures
    }

    /// The combined tools, for `Toolset` registration.
    pub fn tools(&self) -> Vec<Arc<dyn Tool>> {
        self.tools.clone()
    }
}

/// Connects and discovers tools under deadlines, closing the transport on failure.
pub async fn connect_config(
    name: &str,
    config: &McpServerConfig,
    mode: crate::tool::policy::ExecutionMode,
    network: bool,
) -> std::result::Result<McpConnection, String> {
    let timeout = config.timeout();
    let (client, scratch, group) =
        tokio::time::timeout(timeout, serve_config(config, mode, network))
            .await
            .map_err(|_| format!("Timed out after {} ms", timeout.as_millis()))??;
    let mut connection = finish_connection(name, client, timeout).await?;
    connection._scratch = scratch;
    connection.group = group;
    Ok(connection)
}

/// Connects over an established custom or test transport.
pub async fn connect_transport<T, E, A>(
    name: &str,
    transport: T,
    timeout: Duration,
) -> std::result::Result<McpConnection, String>
where
    T: IntoTransport<RoleClient, E, A>,
    E: std::error::Error + Send + Sync + 'static,
{
    let client = tokio::time::timeout(timeout, ().serve(transport))
        .await
        .map_err(|_| format!("Timed out after {} ms", timeout.as_millis()))?
        .map_err(|err| err.to_string())?;
    finish_connection(name, client, timeout).await
}

async fn finish_connection(
    name: &str,
    client: Running,
    timeout: Duration,
) -> std::result::Result<McpConnection, String> {
    let discovered = match discover(&client, timeout).await {
        Ok(tools) => tools,
        Err(err) => {
            // A failed discovery closes its client (and child process).
            let _ = client.cancel().await;
            return Err(err);
        }
    };
    let tools = discovered
        .into_iter()
        .map(|tool| Arc::new(McpTool::new(name, tool, client.peer().clone())) as Arc<dyn Tool>)
        .collect();
    Ok(McpConnection {
        name: name.to_owned(),
        group: None,
        client,
        _scratch: None,
        tools,
    })
}

async fn serve_config(
    config: &McpServerConfig,
    mode: crate::tool::policy::ExecutionMode,
    network: bool,
) -> std::result::Result<(Running, Option<tempfile::TempDir>, Option<GroupGuard>), String> {
    match config {
        McpServerConfig::Local {
            command,
            cwd,
            environment,
            ..
        } => {
            let Some((program, args)) = command.split_first() else {
                return Err("the command is empty".to_owned());
            };
            let directory = cwd
                .as_deref()
                .map(std::path::PathBuf::from)
                .unwrap_or(std::env::current_dir().map_err(|err| err.to_string())?);
            let mut isolated =
                crate::tool::policy::subprocess(program, args, &directory, mode, network)
                    .map_err(|err| err.to_string())?;
            if let Some(environment) = environment {
                isolated.command.envs(environment);
            }
            let diagnostics = Arc::new(std::sync::Mutex::new(Vec::<u8>::new()));
            let (transport, stderr) = TokioChildProcess::builder(isolated.command)
                .stderr(Stdio::piped())
                .spawn()
                .map_err(|err| err.to_string())?;
            let group = GroupGuard::new(transport.id());
            if let Some(mut stderr) = stderr {
                let diagnostics = diagnostics.clone();
                tokio::spawn(async move {
                    use tokio::io::AsyncReadExt;
                    let mut buffer = [0u8; 1024];
                    while let Ok(size) = stderr.read(&mut buffer).await {
                        if size == 0 {
                            break;
                        }
                        let mut tail = diagnostics.lock().expect("MCP diagnostics mutex");
                        tail.extend_from_slice(&buffer[..size]);
                        let excess = tail.len().saturating_sub(4096);
                        tail.drain(..excess);
                    }
                });
            }
            match ().serve(transport).await {
                Ok(client) => Ok((client, isolated._scratch, Some(group))),
                Err(error) => {
                    let mut tail = String::from_utf8_lossy(
                        &diagnostics.lock().expect("MCP diagnostics mutex"),
                    )
                    .into_owned();
                    // Subprocess diagnostics are bounded and redact explicitly supplied secrets.
                    if let Some(environment) = environment {
                        for value in environment.values().filter(|value| value.len() >= 4) {
                            tail = tail.replace(value, "[redacted]");
                        }
                    }
                    for (key, value) in std::env::vars().filter(|(key, value)| {
                        (key.contains("KEY")
                            || key.contains("TOKEN")
                            || key.contains("SECRET")
                            || key.contains("PASSWORD"))
                            && value.len() >= 4
                    }) {
                        let _ = key;
                        tail = tail.replace(&value, "[redacted]");
                    }
                    tail.retain(|ch| !ch.is_control() || ch == '\n' || ch == '\t');
                    Err(format!("{error}; server stderr: {tail}"))
                }
            }
        }
        McpServerConfig::Remote { url, headers, .. } => {
            let mut config = StreamableHttpClientTransportConfig::with_uri(url.clone());
            if let Some(headers) = headers {
                for (key, value) in headers {
                    let header = HeaderName::from_bytes(key.as_bytes())
                        .map_err(|err| format!("invalid header name {key}: {err}"))?;
                    let value = HeaderValue::from_str(value)
                        .map_err(|err| format!("invalid header value for {key}: {err}"))?;
                    config.custom_headers.insert(header, value);
                }
            }
            ().serve(StreamableHttpClientTransport::from_config(config))
                .await
                .map(|client| (client, None, None))
                .map_err(|err| err.to_string())
        }
    }
}

/// Discovers the server's tools when it advertises the capability.
async fn discover(
    client: &Running,
    timeout: Duration,
) -> std::result::Result<Vec<rmcp::model::Tool>, String> {
    let has_tools = client
        .peer_info()
        .is_some_and(|info| info.capabilities.tools.is_some());
    if !has_tools {
        return Ok(Vec::new());
    }
    tokio::time::timeout(timeout, client.peer().list_all_tools())
        .await
        .map_err(|_| format!("Timed out after {} ms listing tools", timeout.as_millis()))?
        .map_err(|err| err.to_string())
}

/// Sanitizes identifiers by replacing characters outside `[a-zA-Z0-9_-]`.
pub fn sanitize(value: &str) -> String {
    value
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Close run-local connections before dropping their scratch directories.
pub(crate) async fn close_scope(
    resources: &Arc<std::sync::Mutex<HashMap<String, Arc<dyn std::any::Any + Send + Sync>>>>,
) {
    let connections: Vec<_> = resources
        .lock()
        .expect("tool resource mutex")
        .values()
        .filter_map(|value| {
            value
                .clone()
                .downcast::<tokio::sync::Mutex<Option<McpConnection>>>()
                .ok()
        })
        .collect();
    for connection in connections {
        if let Some(connection) = connection.lock().await.take() {
            let _ = tokio::time::timeout(crate::engine::CLEANUP_GRACE, connection.close()).await;
        }
    }
}

/// A new connection for each run/child, shared only by tools in that scope.
struct ScopedMcpTool {
    spec: ToolSpec,
    server: String,
    config: McpServerConfig,
    mode: crate::tool::policy::ExecutionMode,
    network: bool,
}

#[async_trait::async_trait]
impl Tool for ScopedMcpTool {
    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }
    async fn call(&self, args: Value, ctx: &ToolCtx) -> std::result::Result<String, ToolError> {
        type Connection = tokio::sync::Mutex<Option<McpConnection>>;
        let connection = {
            let mut resources = ctx.resources.lock().expect("tool resource mutex");
            resources
                .entry(format!("mcp:{}", self.server))
                .or_insert_with(|| Arc::new(Connection::new(None)))
                .clone()
                .downcast::<Connection>()
                .expect("MCP scope type")
        };
        // The lock only covers connecting; calls to one server run concurrently.
        let tool = {
            let mut connection = connection.lock().await;
            // A server that went away since the last call is started again.
            if connection.as_ref().is_some_and(McpConnection::is_closed)
                && let Some(closed) = connection.take()
            {
                let _ = tokio::time::timeout(crate::engine::CLEANUP_GRACE, closed.close()).await;
            }
            if connection.is_none() {
                let mut config = self.config.clone();
                if let McpServerConfig::Local { cwd, .. } = &mut config
                    && cwd.is_none()
                {
                    *cwd = Some(ctx.cwd.to_string_lossy().into_owned());
                }
                *connection = Some(
                    connect_config(&self.server, &config, self.mode, self.network)
                        .await
                        .map_err(ToolError::failure)?,
                );
            }
            connection
                .as_ref()
                .expect("connected")
                .tools()
                .iter()
                .find(|tool| tool.spec().function.name == self.spec.function.name)
                .cloned()
                .ok_or_else(|| ToolError::failure("MCP tool disappeared after reconnect"))?
        };
        tool.call(args, ctx).await
    }
}

/// One MCP server tool, exposed to the model as `<server>_<tool>`.
struct McpTool {
    spec: ToolSpec,
    peer: Peer<RoleClient>,
    remote_name: String,
}

impl McpTool {
    fn new(server: &str, tool: rmcp::model::Tool, peer: Peer<RoleClient>) -> Self {
        // Providers bound function names (OpenAI: 64 characters); sanitized names
        // are ASCII, so truncation is safe. Collisions are caught at registration.
        let mut name = format!("{}_{}", sanitize(server), sanitize(&tool.name));
        name.truncate(MAX_TOOL_NAME);
        let description = tool.description.as_deref().unwrap_or_default().to_owned();
        // Preserve the server's schema and its additional-property policy.
        let parameters = Value::Object((*tool.input_schema).clone());
        Self {
            spec: ToolSpec::function(name, description, parameters),
            peer,
            remote_name: tool.name.to_string(),
        }
    }
}

#[async_trait::async_trait]
impl Tool for McpTool {
    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }

    async fn call(&self, args: Value, ctx: &ToolCtx) -> std::result::Result<String, ToolError> {
        // Validate arguments against the advertised schema before sending them.
        if let Some(problem) = check_args(&self.spec.function.parameters, &args) {
            return Err(ToolError::failure(format!(
                "the {} tool was called with invalid arguments: {problem}",
                self.spec.function.name
            )));
        }
        let Value::Object(arguments) = args else {
            return Err(ToolError::failure(
                "invalid arguments: expected a JSON object",
            ));
        };
        let params = CallToolRequestParams::new(self.remote_name.clone()).with_arguments(arguments);
        // No per-call timeout: long calls (a browser navigation, a build) are
        // bounded by run cancellation and the run deadline below.
        let options = PeerRequestOptions::no_options();
        let request = ClientRequest::CallToolRequest(CallToolRequest::new(params));
        let handle = self
            .peer
            .send_request_with_option(request, options)
            .await
            .map_err(|err| ToolError::failure(format!("the MCP request failed: {err}")))?;
        let request_id = handle.id.clone();
        let mut responder = tokio::spawn(handle.await_response());
        let result = tokio::select! {
            joined = &mut responder => match joined {
                Ok(Ok(result)) => result,
                Ok(Err(err)) => {
                    return Err(ToolError::failure(format!("the MCP call failed: {err}")));
                }
                Err(join_err) => {
                    return Err(ToolError::failure(format!("the MCP call task failed: {join_err}")));
                }
            },
            _ = ctx.token.cancelled() => {
                responder.abort();
                // Request remote cancellation where supported; effects the
                // server already applied are not rolled back.
                let _ = self
                    .peer
                    .notify_cancelled(CancelledNotificationParam::new(
                        Some(request_id),
                        Some("the run was cancelled".to_owned()),
                    ))
                    .await;
                return Err(ToolError::Fatal(Error::Cancelled));
            }
        };
        let ServerResult::CallToolResult(result) = result else {
            return Err(ToolError::failure(
                "the MCP server sent an unexpected response",
            ));
        };
        let text = flatten(&result.content);
        if result.is_error.unwrap_or(false) {
            return Err(ToolError::failure(if text.is_empty() {
                "The MCP tool returned an error".to_owned()
            } else {
                text
            }));
        }
        if !text.is_empty() {
            return Ok(text);
        }
        Ok(match &result.structured_content {
            Some(structured) => structured.to_string(),
            None => "(no output)".to_owned(),
        })
    }
}

/// Flattens content blocks to model-visible text.
fn flatten(content: &[ContentBlock]) -> String {
    content
        .iter()
        .map(|block| match block {
            ContentBlock::Text(text) => text.text.to_string(),
            ContentBlock::Resource(resource) => match &resource.resource {
                ResourceContents::TextResourceContents { text, .. } => text.clone(),
                ResourceContents::BlobResourceContents {
                    mime_type: Some(mime),
                    ..
                } => {
                    format!("[resource {mime} content omitted]")
                }
                ResourceContents::BlobResourceContents { .. } => {
                    "[resource content omitted]".to_owned()
                }
                _ => "[resource content omitted]".to_owned(),
            },
            ContentBlock::Image(image) => format!("[image {} content omitted]", image.mime_type),
            ContentBlock::Audio(audio) => format!("[audio {} content omitted]", audio.mime_type),
            ContentBlock::ResourceLink(_) => "[resource_link content omitted]".to_owned(),
            _ => "[content omitted]".to_owned(),
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Checks required fields, closed-schema unknown fields, and simple argument types.
fn check_args(schema: &Value, args: &Value) -> Option<String> {
    let Value::Object(args) = args else {
        return Some("arguments must be a JSON object".to_owned());
    };
    if let Some(required) = schema.get("required").and_then(Value::as_array) {
        for name in required.iter().filter_map(Value::as_str) {
            if !args.contains_key(name) {
                return Some(format!("missing required argument \"{name}\""));
            }
        }
    }
    let closed = schema.get("additionalProperties").and_then(Value::as_bool) == Some(false)
        && schema
            .get("patternProperties")
            .and_then(Value::as_object)
            .is_none_or(|patterns| patterns.is_empty());
    let properties = schema.get("properties").and_then(Value::as_object);
    for (name, value) in args {
        let Some(property) = properties.and_then(|properties| properties.get(name)) else {
            if closed {
                return Some(format!("unknown argument \"{name}\""));
            }
            continue;
        };
        if let Some(expected) = property.get("type").and_then(Value::as_str) {
            let actual = json_type(value);
            if expected != actual && !(expected == "number" && actual == "integer") {
                return Some(format!("argument \"{name}\" must be a {expected}"));
            }
        }
    }
    None
}

fn json_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(number) => {
            if number.is_i64() || number.is_u64() {
                "integer"
            } else {
                "number"
            }
        }
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn sanitize_replaces_unsafe_characters() {
        assert_eq!(sanitize("my files"), "my_files");
        assert_eq!(sanitize("a.b/c:d"), "a_b_c_d");
        assert_eq!(sanitize("keep-_1"), "keep-_1");
        assert_eq!(sanitize("héllo"), "h_llo");
    }

    #[test]
    fn check_args_reports_missing_unknown_and_mistyped() {
        let schema = json!({
            "type": "object",
            "properties": {"text": {"type": "string"}, "times": {"type": "integer"}},
            "required": ["text"],
            "additionalProperties": false
        });
        assert_eq!(
            check_args(&schema, &json!({"text": "hi", "times": 2})),
            None
        );
        assert_eq!(
            check_args(&schema, &json!({"times": 2})).as_deref(),
            Some("missing required argument \"text\"")
        );
        assert_eq!(
            check_args(&schema, &json!({"text": "hi", "extra": 1})).as_deref(),
            Some("unknown argument \"extra\"")
        );
        assert_eq!(
            check_args(&schema, &json!({"text": 3})).as_deref(),
            Some("argument \"text\" must be a string")
        );
        // integers satisfy "number", and untyped schemas are not checked
        let numbers = json!({"type": "object", "properties": {"n": {"type": "number"}}});
        assert_eq!(check_args(&numbers, &json!({"n": 3})), None);
        assert_eq!(
            check_args(&schema, &json!([])).as_deref(),
            Some("arguments must be a JSON object")
        );
    }

    #[test]
    fn config_parsing_matches_the_opencode_format() {
        let configs: BTreeMap<String, McpServerConfig> = serde_json::from_value(json!({
            "files": {"type": "local", "command": ["npx", "-y", "some-mcp-server"], "environment": {"K": "V"}},
            "search": {"type": "remote", "url": "https://example.com/mcp", "headers": {"Authorization": "Bearer x"}, "timeout": 5000},
            "off": {"type": "local", "command": ["x"], "enabled": false},
            "zero": {"type": "local", "command": ["x"], "timeout": 0}
        }))
        .unwrap();
        assert!(configs["files"].enabled());
        assert_eq!(
            configs["files"].timeout(),
            Duration::from_millis(DEFAULT_TIMEOUT_MS)
        );
        assert_eq!(
            configs["zero"].timeout(),
            Duration::from_millis(DEFAULT_TIMEOUT_MS),
            "a zero timeout falls back to the default instead of disabling the server"
        );
        assert!(!configs["off"].enabled());
        assert_eq!(configs["search"].timeout(), Duration::from_secs(5));
    }
}
