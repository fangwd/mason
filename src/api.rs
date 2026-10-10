//! Session HTTP API with cursor-based event replay and transient SSE deltas.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::rejection::JsonRejection;
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use chrono::{DateTime, Utc};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::{broadcast, mpsc};
use tokio_util::sync::CancellationToken;

use crate::engine::RunEvent;
use crate::profile::ProfileSet;
use crate::runner::Runner;
use crate::store::{
    Delivery, Event as StoreEvent, NewInput, NewSession, Session, SessionId, SessionQuery,
    SessionStore, SessionUpdate,
};

/// Bearer authentication for a directly exposed API. Embedders can instead
/// install their own authentication around `router`.
#[derive(Clone)]
pub struct ApiSecurity {
    token: String,
}

impl ApiSecurity {
    pub fn new(token: String) -> crate::Result<Self> {
        if token.len() < 16 || !token.bytes().all(|byte| byte.is_ascii_graphic()) {
            return Err(crate::Error::Config(
                "API token must contain at least 16 non-whitespace characters".into(),
            ));
        }
        Ok(Self { token })
    }

    fn permits(&self, headers: &HeaderMap) -> bool {
        let Some(provided) = headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
        else {
            return false;
        };
        provided.len() == self.token.len()
            && provided
                .bytes()
                .zip(self.token.bytes())
                .fold(0u8, |diff, (a, b)| diff | (a ^ b))
                == 0
    }
}

async fn protect_api(
    State(security): State<Arc<ApiSecurity>>,
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    if !security.permits(request.headers()) {
        return (
            StatusCode::UNAUTHORIZED,
            Json(json!({"error": "missing or invalid bearer token"})),
        )
            .into_response();
    }
    next.run(request).await
}

/// Authenticated router, including SSE endpoints. Tokens are headers, never URLs.
/// [`health_router`] stays outside the layer: a supervisor must be able to probe
/// liveness and version without holding the secret.
pub fn router_with_security(
    runner: Arc<Runner>,
    shutdown: CancellationToken,
    security: ApiSecurity,
) -> Router {
    Router::new()
        .merge(health_router())
        .merge(
            sessions_router(runner, shutdown).layer(axum::middleware::from_fn_with_state(
                Arc::new(security),
                protect_api,
            )),
        )
}

/// Liveness and the crate version, exempt from the API token. Its purpose is a
/// version floor: an application can refuse to run against an agent that is too
/// old for its manifest instead of misbehaving at run time.
pub fn health_router() -> Router {
    Router::new().route("/healthz", get(healthz))
}

async fn healthz() -> Response {
    Json(json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
    }))
    .into_response()
}

pub struct ApiState {
    pub runner: Arc<Runner>,
    pub store: Arc<dyn SessionStore>,
    /// Profile registry for session validation.
    pub profiles: Arc<ProfileSet>,
    /// Closes SSE streams at server shutdown.
    pub shutdown: CancellationToken,
}

/// Builds the API router for a runner. Open SSE streams stay open until
/// their client disconnects; use [`router_with_shutdown`] to bound them.
pub fn router(runner: Arc<Runner>) -> Router {
    router_with_shutdown(runner, CancellationToken::new())
}

/// Builds a router whose SSE streams close when `shutdown` is cancelled.
pub fn router_with_shutdown(runner: Arc<Runner>, shutdown: CancellationToken) -> Router {
    Router::new()
        .merge(health_router())
        .merge(sessions_router(runner, shutdown))
}

/// The session routes, without the health endpoint.
fn sessions_router(runner: Arc<Runner>, shutdown: CancellationToken) -> Router {
    let profiles = Arc::new(runner.agent().profiles().clone());
    let store = runner.store().clone();
    let state = Arc::new(ApiState {
        runner,
        store,
        profiles,
        shutdown,
    });
    Router::new()
        .route("/sessions", post(create_session).get(list_sessions))
        .route(
            "/sessions/{id}",
            get(get_session)
                .patch(update_session)
                .delete(delete_session),
        )
        .route("/sessions/{id}/inputs", get(list_inputs))
        .route("/sessions/{id}/inputs/{input}", delete(withdraw_input))
        .route("/sessions/{id}/messages", post(post_message))
        .route("/sessions/{id}/interrupt", post(interrupt))
        .route("/sessions/{id}/resume", post(resume))
        .route("/sessions/{id}/events", get(events))
        .with_state(state)
}

fn bad_request(message: impl Into<String>) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(json!({"error": message.into()})),
    )
        .into_response()
}

fn not_found(message: impl Into<String>) -> Response {
    (
        StatusCode::NOT_FOUND,
        Json(json!({"error": message.into()})),
    )
        .into_response()
}

fn server_error(err: crate::Error) -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(json!({"error": err.to_string()})),
    )
        .into_response()
}

fn parse_id(raw: &str) -> Option<SessionId> {
    raw.parse::<SessionId>().ok()
}

#[derive(Default, Deserialize)]
struct CreateSessionRequest {
    title: Option<String>,
    agent: Option<String>,
    /// Opaque metadata for the embedder's profile selector; the API stores it
    /// unread.
    #[serde(default)]
    metadata: Value,
}

#[derive(Default, Deserialize)]
struct MessageRequest {
    content: Option<String>,
    #[serde(default)]
    delivery: Delivery,
    source: Option<String>,
    key: Option<String>,
}

fn request_body<Request: Default>(
    body: Result<Option<Json<Request>>, JsonRejection>,
) -> Result<Request, JsonRejection> {
    Ok(body?.map(|Json(body)| body).unwrap_or_default())
}

fn request_error(error: JsonRejection) -> Response {
    let status = match error.status() {
        StatusCode::UNPROCESSABLE_ENTITY => StatusCode::BAD_REQUEST,
        status => status,
    };
    (status, Json(json!({"error": error.body_text()}))).into_response()
}

async fn create_session(
    State(state): State<Arc<ApiState>>,
    body: Result<Option<Json<CreateSessionRequest>>, JsonRejection>,
) -> Response {
    let body = match request_body(body) {
        Ok(body) => body,
        Err(error) => return request_error(error),
    };
    if let Some(agent) = body.agent.as_deref() {
        let Some(profile) = state.profiles.get(agent) else {
            return bad_request(format!("Unknown agent \"{agent}\""));
        };
        if profile.subagent {
            return bad_request(format!("\"{agent}\" is a subagent profile"));
        }
    }
    match state
        .store
        .create_session(NewSession {
            title: body.title,
            agent: body.agent,
            metadata: body.metadata,
            ..Default::default()
        })
        .await
    {
        Ok(id) => (StatusCode::CREATED, Json(json!({"id": id}))).into_response(),
        Err(err) => server_error(err),
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ListQuery {
    agent: Option<String>,
    parent_id: Option<SessionId>,
    limit: Option<u32>,
    before: Option<String>,
}

async fn list_sessions(
    State(state): State<Arc<ApiState>>,
    Query(query): Query<ListQuery>,
) -> Response {
    let before = match &query.before {
        Some(raw) => match DateTime::parse_from_rfc3339(raw) {
            Ok(parsed) => Some(parsed.with_timezone(&Utc)),
            Err(_) => return bad_request("before must be an ISO date"),
        },
        None => None,
    };
    let list_query = SessionQuery {
        agent: query.agent,
        parent_id: query.parent_id,
        limit: query.limit,
        before,
    };
    match state.store.list_sessions(&list_query).await {
        Ok(summaries) => {
            let sessions: Vec<Value> = summaries
                .iter()
                .map(|summary| {
                    let mut value = serde_json::to_value(summary).unwrap_or_default();
                    // `busy` and `stalled` are supplied by the runner, not the store.
                    value["busy"] = json!(state.runner.is_busy(summary.id));
                    value["stalled"] = json!(state.runner.is_stalled(summary.id));
                    value
                })
                .collect();
            Json(json!({"sessions": sessions})).into_response()
        }
        Err(err) => server_error(err),
    }
}

async fn get_session(State(state): State<Arc<ApiState>>, Path(raw): Path<String>) -> Response {
    let Some(id) = parse_id(&raw) else {
        return not_found(format!("No session {raw}"));
    };
    match state.store.get_session(id).await {
        Ok(Some(session)) => session_response(&state, session),
        Ok(None) => not_found(format!("No session {raw}")),
        Err(err) => server_error(err),
    }
}

fn session_response(state: &ApiState, session: Session) -> Response {
    Json(json!({
        "id": session.id,
        "parentId": session.parent_id,
        "agent": session.agent,
        "title": session.title,
        "createdAt": session.created_at,
        "metadata": session.metadata,
        "busy": state.runner.is_busy(session.id),
        "stalled": state.runner.is_stalled(session.id),
    }))
    .into_response()
}

/// Drains a stalled session's queued inputs again after a failed run.
async fn resume(State(state): State<Arc<ApiState>>, Path(raw): Path<String>) -> Response {
    let Some(id) = parse_id(&raw) else {
        return not_found(format!("No session {raw}"));
    };
    match state.store.get_session(id).await {
        Ok(Some(_)) => {}
        Ok(None) => return not_found(format!("No session {raw}")),
        Err(err) => return server_error(err),
    }
    match state.runner.try_wake(id) {
        Ok(()) => StatusCode::ACCEPTED.into_response(),
        Err(crate::Error::Busy) => (
            StatusCode::TOO_MANY_REQUESTS,
            Json(json!({"error": "server admission queue is full"})),
        )
            .into_response(),
        Err(err) => server_error(err),
    }
}

/// Sets `title` and/or `metadata`; an explicit `"metadata": null` clears it.
async fn update_session(
    State(state): State<Arc<ApiState>>,
    Path(raw): Path<String>,
    body: Result<Json<serde_json::Map<String, Value>>, JsonRejection>,
) -> Response {
    let Some(id) = parse_id(&raw) else {
        return not_found(format!("No session {raw}"));
    };
    let mut body = match body {
        Ok(Json(body)) => body,
        Err(error) => return request_error(error),
    };
    let title = match body.remove("title") {
        None => None,
        Some(Value::String(title)) => Some(title),
        Some(_) => return bad_request("title must be a string"),
    };
    let metadata = body.remove("metadata");
    if let Some(field) = body.keys().next() {
        return bad_request(format!("unknown field `{field}`; use title or metadata"));
    }
    match state
        .store
        .update_session(id, SessionUpdate { title, metadata })
        .await
    {
        Ok(false) => return not_found(format!("No session {raw}")),
        Ok(true) => {}
        Err(err) => return server_error(err),
    }
    match state.store.get_session(id).await {
        Ok(Some(session)) => session_response(&state, session),
        Ok(None) => not_found(format!("No session {raw}")),
        Err(err) => server_error(err),
    }
}

/// Deletes an idle session with its history and child sessions.
async fn delete_session(State(state): State<Arc<ApiState>>, Path(raw): Path<String>) -> Response {
    let Some(id) = parse_id(&raw) else {
        return not_found(format!("No session {raw}"));
    };
    if state.runner.is_busy(id) {
        return (
            StatusCode::CONFLICT,
            Json(json!({"error": "the session is running; interrupt it first"})),
        )
            .into_response();
    }
    match state.store.delete_session(id).await {
        Ok(true) => {
            state.runner.agent().close_session_scope(id).await;
            StatusCode::NO_CONTENT.into_response()
        }
        Ok(false) => not_found(format!("No session {raw}")),
        Err(err) => server_error(err),
    }
}

/// Withdraws an input that has not entered the conversation yet.
async fn withdraw_input(
    State(state): State<Arc<ApiState>>,
    Path((raw, raw_input)): Path<(String, String)>,
) -> Response {
    let (Some(id), Ok(input)) = (parse_id(&raw), raw_input.parse::<u64>()) else {
        return not_found(format!("No pending input {raw_input} in session {raw}"));
    };
    match state.store.withdraw_input(id, input).await {
        Ok(true) => StatusCode::NO_CONTENT.into_response(),
        Ok(false) => not_found(format!("No pending input {raw_input} in session {raw}")),
        Err(err) => server_error(err),
    }
}

/// The session's inbox: pending and consumed inputs, in insertion order.
#[derive(Default, Deserialize)]
struct InputsQuery {
    after: Option<u64>,
    limit: Option<u32>,
}

async fn list_inputs(
    State(state): State<Arc<ApiState>>,
    Path(raw): Path<String>,
    Query(query): Query<InputsQuery>,
) -> Response {
    let Some(id) = parse_id(&raw) else {
        return not_found(format!("No session {raw}"));
    };
    match state.store.get_session(id).await {
        Ok(Some(_)) => {}
        Ok(None) => return not_found(format!("No session {raw}")),
        Err(err) => return server_error(err),
    }
    match state
        .store
        .inputs_page(
            id,
            query.after.unwrap_or(0),
            query.limit.unwrap_or(500).clamp(1, 1000),
        )
        .await
    {
        Ok(inputs) => Json(json!({ "inputs": inputs })).into_response(),
        Err(err) => server_error(err),
    }
}

async fn post_message(
    State(state): State<Arc<ApiState>>,
    Path(raw): Path<String>,
    body: Result<Option<Json<MessageRequest>>, JsonRejection>,
) -> Response {
    let Some(id) = parse_id(&raw) else {
        return not_found(format!("No session {raw}"));
    };
    match state.store.get_session(id).await {
        Ok(Some(session)) if session.parent_id.is_some() => {
            return (
                StatusCode::CONFLICT,
                Json(json!({"error": "subagent messages must be sent through the parent session"})),
            )
                .into_response();
        }
        Ok(Some(_)) => {}
        Ok(None) => return not_found(format!("No session {raw}")),
        Err(err) => return server_error(err),
    }
    let body = match request_body(body) {
        Ok(body) => body,
        Err(error) => return request_error(error),
    };
    let Some(content) = body.content.filter(|content| !content.is_empty()) else {
        return bad_request("content is required");
    };
    if content.len() > 128 * 1024 {
        return bad_request("content exceeds the 128 KiB input limit");
    }
    let input = NewInput {
        content,
        delivery: body.delivery,
        source: body
            .source
            .filter(|source| !source.is_empty())
            .unwrap_or_else(|| "human".into()),
        key: body.key,
    };
    match state.runner.submit(id, input).await {
        Ok(input_id) => (StatusCode::ACCEPTED, Json(json!({"inputId": input_id}))).into_response(),
        Err(crate::Error::Busy) => (
            StatusCode::TOO_MANY_REQUESTS,
            Json(json!({"error": "session admission queue is full"})),
        )
            .into_response(),
        Err(err) => server_error(err),
    }
}

async fn interrupt(State(state): State<Arc<ApiState>>, Path(raw): Path<String>) -> Response {
    let Some(id) = parse_id(&raw) else {
        return not_found(format!("No session {raw}"));
    };
    match state.store.get_session(id).await {
        Ok(Some(_)) => {}
        Ok(None) => return not_found(format!("No session {raw}")),
        Err(err) => return server_error(err),
    }
    state.runner.interrupt(id);
    StatusCode::NO_CONTENT.into_response()
}

/// Page size of the JSON events list; clients page with `after`.
const DEFAULT_EVENT_PAGE: u32 = 500;
const MAX_EVENT_PAGE: u32 = 1000;

#[derive(Debug, Deserialize)]
struct EventsQuery {
    after: Option<String>,
    /// JSON lists only: default 500, clamped to 1..=1000.
    limit: Option<u32>,
}

async fn events(
    State(state): State<Arc<ApiState>>,
    Path(raw): Path<String>,
    Query(query): Query<EventsQuery>,
    headers: HeaderMap,
) -> Response {
    let Some(id) = parse_id(&raw) else {
        return not_found(format!("No session {raw}"));
    };
    match state.store.get_session(id).await {
        Ok(Some(_)) => {}
        Ok(None) => return not_found(format!("No session {raw}")),
        Err(err) => return server_error(err),
    }
    // An explicit `after` overrides `Last-Event-ID`.
    let cursor = match &query.after {
        Some(raw_after) => match raw_after.parse::<u64>() {
            Ok(after) => after,
            Err(_) => return bad_request("after must be a number"),
        },
        None => headers
            .get("last-event-id")
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(0),
    };

    let wants_stream = headers
        .get(axum::http::header::ACCEPT)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.contains("text/event-stream"));
    if !wants_stream {
        let limit = query
            .limit
            .unwrap_or(DEFAULT_EVENT_PAGE)
            .clamp(1, MAX_EVENT_PAGE);
        return match state.store.events_page(id, cursor, limit).await {
            Ok(events) => Json(events).into_response(),
            Err(err) => server_error(err),
        };
    }

    // Subscribe before replay to avoid gaps; the cursor deduplicates overlaps.
    let store_rx = state.store.subscribe(id);
    let delta_rx = state.runner.subscribe_deltas(id);
    let (tx, rx) = mpsc::channel::<std::result::Result<SseEvent, Infallible>>(256);
    tokio::spawn(sse_task(state, id, cursor, store_rx, delta_rx, tx));
    Sse::new(tokio_stream::wrappers::ReceiverStream::new(rx))
        .keep_alive(
            KeepAlive::new()
                .interval(Duration::from_secs(15))
                .text("heartbeat"),
        )
        .into_response()
}

fn committed_sse(event: &StoreEvent) -> SseEvent {
    let base = SseEvent::default()
        .id(event.id.to_string())
        .event(event.event_type.clone());
    match base.clone().json_data(event) {
        Ok(sse) => sse,
        Err(_) => base.data(event.data.to_string()),
    }
}

fn delta_sse(event: &RunEvent) -> SseEvent {
    // Transient frames omit IDs; inherited Last-Event-ID values cannot deduplicate
    // them.
    let name = match event {
        RunEvent::Delta { .. } => "delta",
        RunEvent::Committed { .. } => "committed",
        RunEvent::ToolStarted { .. } => "tool",
    };
    let base = SseEvent::default().event(name);
    match base.clone().json_data(event) {
        Ok(sse) => sse,
        Err(_) => base.data("{}"),
    }
}

/// Sends a frame without allowing viewer backpressure to block shutdown.
async fn send_frame(
    tx: &mpsc::Sender<std::result::Result<SseEvent, Infallible>>,
    shutdown: &CancellationToken,
    frame: SseEvent,
) -> bool {
    tokio::select! {
        sent = tx.send(Ok(frame)) => sent.is_ok(),
        _ = shutdown.cancelled() => false,
    }
}

/// Committed events are replayed in pages of this size.
const REPLAY_PAGE: u32 = 1_000;

/// Streams committed events after the cursor, page by page, advancing it. Returns
/// `false` when the viewer is gone, the store failed or the server is shutting down.
async fn replay(
    state: &ApiState,
    session_id: SessionId,
    cursor: &mut u64,
    tx: &mpsc::Sender<std::result::Result<SseEvent, Infallible>>,
) -> bool {
    loop {
        let Ok(events) = state
            .store
            .events_page(session_id, *cursor, REPLAY_PAGE)
            .await
        else {
            return false;
        };
        for event in &events {
            *cursor = (*cursor).max(event.id);
            if !send_frame(tx, &state.shutdown, committed_sse(event)).await {
                return false;
            }
        }
        if events.len() < REPLAY_PAGE as usize {
            return true;
        }
    }
}

async fn sse_task(
    state: Arc<ApiState>,
    session_id: SessionId,
    after: u64,
    mut store_rx: broadcast::Receiver<Arc<StoreEvent>>,
    mut delta_rx: broadcast::Receiver<Arc<RunEvent>>,
    tx: mpsc::Sender<std::result::Result<SseEvent, Infallible>>,
) {
    let mut cursor = after;
    if !replay(&state, session_id, &mut cursor, &tx).await {
        return;
    }
    loop {
        tokio::select! {
            _ = state.shutdown.cancelled() => return,
            // A viewer that disconnected on an idle session would otherwise park
            // here, pinning both subscriptions, until the session's next event.
            _ = tx.closed() => return,
            item = store_rx.recv() => match item {
                Ok(event) => {
                    if event.id <= cursor {
                        continue; // already covered by the replay
                    }
                    cursor = event.id;
                    if !send_frame(&tx, &state.shutdown, committed_sse(&event)).await {
                        return;
                    }
                }
                Err(broadcast::error::RecvError::Lagged(_)) => {
                    // Replay from the cursor and close after broadcast lag; the client
                    // resubscribes.
                    replay(&state, session_id, &mut cursor, &tx).await;
                    return;
                }
                Err(broadcast::error::RecvError::Closed) => return,
            },
            item = delta_rx.recv() => match item {
                Ok(event) => {
                    if !send_frame(&tx, &state.shutdown, delta_sse(&event)).await {
                        return;
                    }
                }
                // Deltas cannot be replayed; committed messages remain authoritative.
                Err(broadcast::error::RecvError::Lagged(_)) => continue,
                Err(broadcast::error::RecvError::Closed) => return,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{ChatModel, ChatRequest, ChatStream};
    use crate::store::SqliteStore;
    use crate::{Agent, OutputSpill};

    struct NoModel;

    #[async_trait::async_trait]
    impl ChatModel for NoModel {
        fn context_tokens(&self) -> usize {
            128_000
        }
        async fn stream(
            &self,
            _request: ChatRequest,
            _token: &CancellationToken,
        ) -> crate::Result<ChatStream> {
            Err(crate::Error::Run("unused".into()))
        }
    }

    #[tokio::test]
    async fn a_disconnected_viewer_of_an_idle_session_ends_its_task() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(
            SqliteStore::open(dir.path().join("mason.db"))
                .await
                .unwrap(),
        );
        let agent = Agent::builder()
            .model(Arc::new(NoModel))
            .cwd(dir.path())
            .spill(Arc::new(OutputSpill::new(dir.path().join("spill"))))
            .build()
            .unwrap();
        let runner = Runner::new(agent, store.clone());
        let state = Arc::new(ApiState {
            profiles: Arc::new(runner.agent().profiles().clone()),
            store: store.clone(),
            runner: runner.clone(),
            shutdown: CancellationToken::new(),
        });
        let session = store.create_session(NewSession::default()).await.unwrap();

        let store_rx = store.subscribe(session);
        let delta_rx = runner.subscribe_deltas(session);
        let (tx, rx) = mpsc::channel(256);
        let task = tokio::spawn(sse_task(state, session, 0, store_rx, delta_rx, tx));
        drop(rx); // the client went away; no event will ever arrive

        tokio::time::timeout(Duration::from_secs(2), task)
            .await
            .expect("the SSE task must end when its viewer disconnects")
            .unwrap();
    }
}
