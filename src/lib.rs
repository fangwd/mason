//! Mason — a domain-independent agent core.
//!
//! Streaming model/tool execution, context management, profiles, skills, sessions,
//! and HTTP/MCP interfaces. Embed through [`Agent`] with custom tools and prompts.
//!
//! See the README for embedding, limits, and security guidance.

#![forbid(unsafe_code)]

pub mod agent;
pub mod api;
pub mod cli;
pub mod config;
pub mod context;
pub mod engine;
pub mod error;
pub mod evaluate;
pub mod llm;
#[cfg(feature = "mcp")]
pub mod mcp;
pub mod memory;
pub mod profile;
pub mod prompt;
pub mod reflect;
pub mod runner;
pub mod skill;
pub mod store;
pub mod tool;

pub use agent::{Agent, AgentBuilder, SessionRun};
pub use config::{AgentSpec, Manifest};
pub use engine::{Limits, RunConfig, RunEvent, RunOutcome};
pub use error::{Error, Result};
pub use llm::{
    CancellationToken, ChatDelta, ChatModel, ChatRequest, ChatStream, ChatTurn, LlmError, Message,
    Role, ToolCall, ToolSpec, TurnAccumulator, Usage, Vendor, VendorModel, Vendors,
    is_context_overflow,
};
pub use profile::{Profile, ProfileSelector, ProfileSet};
pub use prompt::{Environment, PromptSet, PromptSource};
pub use runner::Runner;
pub use skill::{Skill, load_skills};
pub use store::{
    Conversation, Delivery, Event as StoreEvent, EventId, Input, NewInput, NewSession, Recovery,
    RunClaim, Session, SessionId, SessionQuery, SessionStore, SessionSummary, SqliteStore,
    StoredInput,
};
pub use tool::{
    Keep, OutputSpill, Tool, ToolCtx, ToolError, Toolset, Typed, TypedTool, builtin_tools,
};
