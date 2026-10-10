//! LLM error classification, retry backoff, and context-overflow detection.

use std::sync::OnceLock;
use std::time::Duration;

use rand::RngExt;
use regex::Regex;

/// An LLM turn failure.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum LlmError {
    /// Provider HTTP failure with a bounded response-body excerpt.
    #[error("HTTP {status}: {body}")]
    Status {
        status: u16,
        /// The response body, truncated to 500 characters.
        body: String,
        /// A parsed `Retry-After` header in milliseconds, when present.
        retry_after_ms: Option<u64>,
    },
    /// The request could not be completed over the network: connect failures,
    /// dropped connections and attempt deadlines.
    #[error("{0}")]
    Transport(String),
    /// The provider violated the wire protocol: SSE framing, chunk shape,
    /// provider error events or an incomplete turn.
    #[error("{0}")]
    Protocol(String),
    /// The turn exceeded the budget spanning all attempts, all vendors and
    /// the whole stream. `last` is the failed attempt that led here, if any.
    #[error(
        "the turn exceeded its deadline{}",
        last.as_ref().map(|err| format!(" after: {err}")).unwrap_or_default()
    )]
    TurnDeadline { last: Option<String> },
    #[error("the turn was cancelled")]
    Cancelled,
}

impl LlmError {
    /// Retries rate limits, overloads, server errors, and transport failures.
    pub fn is_retryable(&self) -> bool {
        match self {
            LlmError::Status { status, .. } => *status == 408 || *status == 429 || *status >= 500,
            LlmError::Transport(_) => true,
            LlmError::Protocol(_) | LlmError::TurnDeadline { .. } | LlmError::Cancelled => false,
        }
    }

    /// Whether this failure says the request exceeded the model's context. Retryable
    /// statuses (throttling, overloads) never count, whatever their body says, and
    /// the patterns see the raw body so anchored exclusions work.
    pub fn is_context_overflow(&self) -> bool {
        match self {
            LlmError::Status { status, body, .. } => {
                !self.is_retryable()
                    && (is_context_overflow(body) || (*status == 413 && body.trim().is_empty()))
            }
            LlmError::Protocol(message) => is_context_overflow(message),
            LlmError::Transport(_) | LlmError::TurnDeadline { .. } | LlmError::Cancelled => false,
        }
    }
}

/// Parses positive seconds from `Retry-After`; HTTP-date values are unsupported.
pub fn retry_after_ms(value: &str) -> Option<u64> {
    let seconds: f64 = value.trim().parse().ok()?;
    if seconds.is_finite() && seconds > 0.0 {
        Some((seconds * 1000.0) as u64)
    } else {
        None
    }
}

/// Uses positive `Retry-After` up to 60 s; otherwise exponential backoff from 2 s with
/// up to 25% jitter, capped at 30 s. `attempt` is one-based.
pub fn retry_delay(attempt: usize, err: &LlmError) -> Duration {
    const RETRY_AFTER_CAP_MS: u64 = 60_000;
    const BACKOFF_BASE_MS: f64 = 2_000.0;
    const BACKOFF_CAP_MS: f64 = 30_000.0;
    if let LlmError::Status {
        retry_after_ms: Some(ms),
        ..
    } = err
    {
        return Duration::from_millis((*ms).min(RETRY_AFTER_CAP_MS));
    }
    let base = BACKOFF_BASE_MS * 2f64.powi(attempt.saturating_sub(1).min(10) as i32);
    let jitter = rand::rng().random_range(0.0..0.25);
    Duration::from_millis((base + base * jitter).ceil().min(BACKOFF_CAP_MS) as u64)
}

/// Context-overflow patterns derived from opencode's `llm/src/provider-error.ts`.
const OVERFLOW_PATTERNS: &[&str] = &[
    "(?i)prompt is too long",
    "(?i)request_too_large",
    "(?i)input is too long for requested model",
    "(?i)exceeds the context window",
    "(?i)exceeds (?:the )?(?:model'?s )?maximum context length(?: of [\\d,]+ tokens?|\\s*\\([\\d,]+\\))",
    "(?i)input token count.*exceeds the maximum",
    "(?i)tokens in request more than max tokens allowed",
    "(?i)maximum prompt length is \\d+",
    "(?i)reduce the length of the messages",
    "(?i)maximum context length is \\d+ tokens",
    "(?i)exceeds (?:the )?maximum allowed input length of [\\d,]+ tokens",
    "(?i)input \\(\\d+ tokens\\) is longer than the model'?s context length \\(\\d+ tokens\\)",
    "(?i)exceeds the limit of \\d+",
    "(?i)exceeds the available context size",
    "(?i)greater than the context length",
    "(?i)context window exceeds limit",
    "(?i)exceeded model token limit",
    "(?i)context[_ ]length[_ ]exceeded",
    "(?i)request entity too large",
    "(?i)context length is only \\d+ tokens",
    "(?i)input length.*exceeds.*context length",
    "(?i)prompt too long; exceeded (?:max )?context length",
    "(?i)too large for model with \\d+ maximum context length",
    "(?i)prompt has [\\d,]+ tokens?, but the configured context size is [\\d,]+ tokens",
    "(?i)model_context_window_exceeded",
    "(?i)too many tokens",
    "(?i)token limit exceeded",
    // DashScope (Qwen): "Range of input length should be [1, 98304]".
    "(?i)range of input length should be",
];

/// Throttling patterns take precedence over context-overflow matches.
const OVERFLOW_EXCLUSIONS: &[&str] = &[
    "(?i)^(throttling error|service unavailable):",
    "(?i)rate limit",
    "(?i)too many requests",
];

/// Whether a rendered error describes the request exceeding the model's context. The
/// engine compacts once and retries when it does.
pub fn is_context_overflow(message: &str) -> bool {
    static COMPILED: OnceLock<(Vec<Regex>, Vec<Regex>)> = OnceLock::new();
    let (patterns, exclusions) = COMPILED.get_or_init(|| {
        let compile = |list: &[&str]| {
            list.iter()
                .map(|pattern| Regex::new(pattern).expect("a vendored overflow pattern"))
                .collect()
        };
        (compile(OVERFLOW_PATTERNS), compile(OVERFLOW_EXCLUSIONS))
    });
    !exclusions.iter().any(|re| re.is_match(message))
        && patterns.iter().any(|re| re.is_match(message))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn status(code: u16) -> LlmError {
        LlmError::Status {
            status: code,
            body: String::new(),
            retry_after_ms: None,
        }
    }

    #[test]
    fn retryable_statuses_match_the_source_rules() {
        for code in [408, 429, 500, 502, 503, 599] {
            assert!(status(code).is_retryable(), "{code} should be retryable");
        }
        for code in [400, 401, 403, 404, 413, 422, 499] {
            assert!(
                !status(code).is_retryable(),
                "{code} should not be retryable"
            );
        }
        assert!(LlmError::Transport("connection reset".into()).is_retryable());
        assert!(!LlmError::Protocol("bad chunk".into()).is_retryable());
        assert!(!LlmError::TurnDeadline { last: None }.is_retryable());
        assert!(!LlmError::Cancelled.is_retryable());
    }

    #[test]
    fn retry_after_header_parsing() {
        assert_eq!(retry_after_ms("1"), Some(1_000));
        assert_eq!(retry_after_ms("0.5"), Some(500));
        assert_eq!(retry_after_ms(" 2 "), Some(2_000));
        assert_eq!(retry_after_ms("0"), None);
        assert_eq!(retry_after_ms("-3"), None);
        assert_eq!(retry_after_ms("soon"), None);
    }

    #[test]
    fn retry_delay_honours_retry_after_and_caps_it() {
        let err = LlmError::Status {
            status: 429,
            body: String::new(),
            retry_after_ms: Some(5_000),
        };
        assert_eq!(retry_delay(1, &err), Duration::from_millis(5_000));

        let capped = LlmError::Status {
            status: 429,
            body: String::new(),
            retry_after_ms: Some(120_000),
        };
        assert_eq!(retry_delay(1, &capped), Duration::from_millis(60_000));
    }

    #[test]
    fn retry_delay_backs_off_exponentially_with_jitter_in_range() {
        // Jitter is asserted as a range, never a fixed sleep.
        for _ in 0..50 {
            let first = retry_delay(1, &LlmError::Transport("x".into()));
            assert!(
                (2_000..=2_500u64).contains(&(first.as_millis() as u64)),
                "{first:?}"
            );
            let second = retry_delay(2, &LlmError::Transport("x".into()));
            assert!(
                (4_000..=5_000u64).contains(&(second.as_millis() as u64)),
                "{second:?}"
            );
        }
    }

    #[test]
    fn overflow_classification_ignores_retryable_statuses() {
        let status = |status: u16, body: &str| LlmError::Status {
            status,
            body: body.to_owned(),
            retry_after_ms: None,
        };
        assert!(
            status(400, "This model's maximum context length is 8192 tokens").is_context_overflow()
        );
        assert!(status(413, "request entity too large").is_context_overflow());
        assert!(
            !status(429, "maximum context length is 8192 tokens").is_context_overflow(),
            "throttling is never an overflow"
        );
        assert!(!status(503, "prompt is too long").is_context_overflow());
        assert!(
            !status(
                400,
                "Throttling error: maximum context length is 8192 tokens"
            )
            .is_context_overflow(),
            "anchored exclusions match the raw body"
        );
        assert!(
            LlmError::Protocol("the provider reported an error: context_length_exceeded".into())
                .is_context_overflow()
        );
        assert!(!LlmError::Transport("too many tokens".into()).is_context_overflow());
    }

    #[test]
    fn only_an_empty_body_413_is_an_overflow() {
        let status = |status: u16, body: &str| LlmError::Status {
            status,
            body: body.to_owned(),
            retry_after_ms: None,
        };
        assert!(status(413, "  ").is_context_overflow());
        // A bare 400 is what proxies return for anything; it must keep failing over.
        assert!(!status(400, "").is_context_overflow());
        assert!(!status(401, "").is_context_overflow());
        assert!(!status(429, "").is_context_overflow());
        assert!(!status(500, "").is_context_overflow());
    }

    #[test]
    fn context_overflow_detection() {
        let positives = [
            "HTTP 400: {\"error\":{\"message\":\"This model's maximum context length is 8192 tokens.\"}}",
            "HTTP 400: {\"code\":\"context_length_exceeded\"}",
            "Prompt is too long: 120000 tokens",
            "input (9000 tokens) is longer than the model's context length (8000 tokens)",
            "Please reduce the length of the messages",
            "request entity too large",
            "too many tokens",
        ];
        for message in positives {
            assert!(
                is_context_overflow(message),
                "should be an overflow: {message}"
            );
        }
        let negatives = [
            "boom",
            "HTTP 500: internal error",
            // Throttling that mentions tokens is excluded.
            "rate limit exceeded: too many tokens",
            "Too Many Requests",
            "Throttling error: maximum context length is 8192 tokens",
        ];
        for message in negatives {
            assert!(
                !is_context_overflow(message),
                "should not be an overflow: {message}"
            );
        }
    }
}
