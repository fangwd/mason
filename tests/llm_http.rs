//! LLM client integration tests over HTTP.

use mason::{
    CancellationToken, ChatModel, ChatRequest, ChatTurn, Message, Usage, VendorModel, Vendors,
};
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn model(server: &MockServer, stream: bool) -> VendorModel {
    let mut vars = std::collections::HashMap::from([
        ("AI_VENDORS".to_owned(), "mock".to_owned()),
        ("AI_MOCK_URL".to_owned(), server.uri()),
        ("AI_MOCK_MODEL".to_owned(), "mock-model".to_owned()),
        ("AI_MOCK_API_KEY".to_owned(), "mock-key".to_owned()),
    ]);
    if !stream {
        vars.insert("AI_MOCK_STREAM".to_owned(), "false".to_owned());
    }
    VendorModel::new(Vendors::from_lookup(move |key| vars.get(key).cloned()).unwrap())
}

fn request() -> ChatRequest {
    ChatRequest {
        messages: vec![Message::user("hi")],
        tools: vec![],
    }
}

const HELLO_SSE: &str = concat!(
    "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"Hel\"}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2}}\n\n",
    "data: [DONE]\n\n",
);

#[tokio::test]
async fn streams_over_real_http() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer mock-key"))
        .and(header("content-type", "application/json"))
        .and(body_partial_json(serde_json::json!({
            "model": "mock-model",
            "stream": true,
            "stream_options": {"include_usage": true},
            "messages": [{"role": "user", "content": "hi"}],
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(HELLO_SSE),
        )
        .expect(1)
        .mount(&server)
        .await;

    let stream = model(&server, true)
        .stream(request(), &CancellationToken::new())
        .await
        .unwrap();
    let turn = ChatTurn::fold(stream).await.unwrap();
    assert_eq!(turn.message.content.as_deref(), Some("Hello"));
    assert_eq!(
        turn.usage,
        Some(Usage {
            prompt_tokens: 3,
            completion_tokens: 2
        })
    );
}

#[tokio::test]
async fn single_shot_vendors_post_without_stream_flags() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "choices": [{
                "message": {"role": "assistant", "content": "Hello"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let stream = model(&server, false)
        .stream(request(), &CancellationToken::new())
        .await
        .unwrap();
    let turn = ChatTurn::fold(stream).await.unwrap();
    assert_eq!(turn.message.content.as_deref(), Some("Hello"));
    assert_eq!(
        turn.usage,
        Some(Usage {
            prompt_tokens: 3,
            completion_tokens: 1
        })
    );

    let received = server.received_requests().await.unwrap();
    assert_eq!(received.len(), 1);
    let body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
    assert!(
        body.get("stream").is_none(),
        "a single-shot vendor must not ask for a stream"
    );
    assert!(body.get("stream_options").is_none());
    assert_eq!(body["model"], "mock-model");
}

#[tokio::test]
async fn a_non_retryable_status_is_terminal() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(401).set_body_string("nope"))
        .expect(1)
        .mount(&server)
        .await;

    let stream = model(&server, true)
        .stream(request(), &CancellationToken::new())
        .await
        .unwrap();
    let err = ChatTurn::fold(stream).await.unwrap_err();
    assert!(err.to_string().starts_with("HTTP 401: nope"), "{err}");
}

#[tokio::test]
async fn a_retry_after_backoff_retries_over_real_http() {
    // One 429 with Retry-After: 1, then success: the retry must honour the
    // header and reach the same mock server again (about one second, real time).
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(429)
                .insert_header("retry-after", "1")
                .set_body_string("slow down"),
        )
        .up_to_n_times(1)
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(HELLO_SSE),
        )
        .expect(1)
        .mount(&server)
        .await;

    let stream = model(&server, true)
        .stream(request(), &CancellationToken::new())
        .await
        .unwrap();
    let turn = ChatTurn::fold(stream).await.unwrap();
    assert_eq!(turn.message.content.as_deref(), Some("Hello"));
}

#[tokio::test]
async fn keyless_vendors_send_their_headers_and_extra_body_without_authorization() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("api-key", "azure-key"))
        .and(body_partial_json(serde_json::json!({"max_tokens": 32000})))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(HELLO_SSE),
        )
        .expect(1)
        .mount(&server)
        .await;
    let vars = std::collections::HashMap::from([
        ("AI_VENDORS".to_owned(), "local".to_owned()),
        ("AI_LOCAL_URL".to_owned(), server.uri()),
        ("AI_LOCAL_MODEL".to_owned(), "m".to_owned()),
        (
            "AI_LOCAL_HEADERS".to_owned(),
            r#"{"api-key": "azure-key"}"#.to_owned(),
        ),
        (
            "AI_LOCAL_EXTRA_BODY".to_owned(),
            r#"{"max_tokens": 32000}"#.to_owned(),
        ),
    ]);
    let model = VendorModel::new(Vendors::from_lookup(move |key| vars.get(key).cloned()).unwrap());
    let stream = model
        .stream(request(), &CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(
        ChatTurn::fold(stream)
            .await
            .unwrap()
            .message
            .content
            .as_deref(),
        Some("Hello")
    );
    let requests = server.received_requests().await.unwrap();
    assert!(!requests[0].headers.contains_key("authorization"));
}
