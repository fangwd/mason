//! Crate-wide error and result types.

/// Result alias defaulting to the crate's [`Error`].
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Crate-wide error type. Variants are added as modules land. The enum is
/// non-exhaustive so growth is not a breaking change for embedders that match on it.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("invalid configuration: {0}")]
    Config(String),

    /// A prompt set could not be read, parsed or validated.
    #[error(transparent)]
    Prompt(#[from] crate::prompt::PromptError),

    /// An LLM turn failed: provider status, transport, protocol, deadline or
    /// cancellation.
    #[error(transparent)]
    Llm(#[from] crate::llm::error::LlmError),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error("run {0} budget exhausted")]
    BudgetExceeded(&'static str),

    #[error("server admission queue is full")]
    Busy,

    #[error("the run was cancelled")]
    Cancelled,

    /// The run exceeded its wall-clock deadline.
    #[error("the run exceeded its {0:?} deadline")]
    RunDeadline(std::time::Duration),

    /// A run-level failure: doom-loop guard, panicking tool call, or a
    /// persistence callback error.
    #[error("{0}")]
    Run(String),

    #[error(transparent)]
    Store(#[from] sqlx::Error),
}
