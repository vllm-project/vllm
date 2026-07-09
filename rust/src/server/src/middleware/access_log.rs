//! Per-request HTTP access log, enabled by default: one INFO line per response
//! on the `vllm_server::access` target. Mirrors Python's uvicorn access log
//! (`--disable-uvicorn-access-log` turns it off; `--disable-access-log-for-endpoints`
//! suppresses paths). Only the line format diverges: a structured `tracing` event,
//! not uvicorn's string, and no client address (the hand-rolled hyper serve loop
//! never exposes the peer address to middleware). Which requests log, and the
//! on/off/exclusion semantics, match.

use std::sync::Arc;

use axum::extract::{Request, State};
use axum::middleware::Next;
use axum::response::Response;
use tracing::info;

use crate::state::AppState;

/// Emit the access-log line for a response unless the request path is excluded.
///
/// Attached outside the auth layer so rejected (e.g. 401) responses are logged too.
pub async fn access_log(State(state): State<Arc<AppState>>, req: Request, next: Next) -> Response {
    // Match exclusions on the query-stripped path, and skip before allocating:
    // these are the high-frequency probes the filter exists to quiet.
    if state.access_log.is_excluded(req.uri().path()) {
        return next.run(req).await;
    }

    let method = req.method().clone();
    // The logged line keeps the query string; the exclusion match above drops it.
    let target = req
        .uri()
        .path_and_query()
        .map(|pq| pq.as_str().to_owned())
        .unwrap_or_else(|| req.uri().path().to_owned());
    let version = req.version();

    let response = next.run(req).await;

    info!(
        target: "vllm_server::access",
        %method,
        path = %target,
        version = ?version,
        status = response.status().as_u16(),
        "http request",
    );

    response
}
