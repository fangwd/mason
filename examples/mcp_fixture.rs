//! Stdio MCP fixture exposing echo, failure, and structured-result tools.

use std::sync::Arc;

use rmcp::ServerHandler;
use rmcp::model::{
    CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock, ErrorData,
    ListToolsResult, PaginatedRequestParams, ServerCapabilities, ServerConfig, Tool,
};
use rmcp::service::RequestContext;
use rmcp::{RoleServer, ServiceExt};
use serde_json::{Value, json};

type JsonObject = serde_json::Map<String, Value>;

#[derive(Clone)]
struct Fixture;

fn object_schema(properties: Value, required: &[&str]) -> Arc<JsonObject> {
    Arc::new(
        json!({
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": false
        })
        .as_object()
        .expect("an object schema")
        .clone(),
    )
}

impl ServerHandler for Fixture {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        Ok(ListToolsResult {
            tools: vec![
                Tool::new(
                    "echo",
                    "Echoes the text",
                    object_schema(
                        json!({"text": {"type": "string"}, "times": {"type": "integer"}}),
                        &["text"],
                    ),
                ),
                Tool::new("fail", "Always fails", object_schema(json!({}), &[])),
                Tool::new(
                    "structured",
                    "Returns structured content",
                    object_schema(json!({}), &[]),
                ),
            ],
            ..Default::default()
        })
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, ErrorData> {
        let arguments = request.arguments.clone().unwrap_or_default();
        match request.name.as_ref() {
            "echo" => {
                let text = arguments
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let times = arguments
                    .get("times")
                    .and_then(Value::as_u64)
                    .unwrap_or(1)
                    .max(1);
                Ok(CallToolResult::success(vec![ContentBlock::text(
                    if std::env::var_os("MASON_FIXTURE_PID").is_some() {
                        std::process::id().to_string()
                    } else {
                        text.repeat(times as usize)
                    },
                )])
                .into())
            }
            "fail" => Ok(CallToolResult::error(vec![ContentBlock::text("it broke")]).into()),
            "structured" => {
                let mut result = CallToolResult::success(Vec::new());
                result.structured_content = Some(json!({"ok": true}));
                Ok(result.into())
            }
            other => Err(ErrorData::internal_error(
                format!("unknown tool {other}"),
                None,
            )),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let service = Fixture
        .serve((tokio::io::stdin(), tokio::io::stdout()))
        .await?;
    service.waiting().await?;
    Ok(())
}
