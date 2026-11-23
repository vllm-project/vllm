// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Per-response HTTP access log.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

use axum::extract::{ConnectInfo, Request, State};
use axum::middleware::Next;
use axum::response::Response;
use tracing::info;

use crate::state::AppState;

/// Log one INFO event per response, unless the request path is excluded: the
/// client address, method, request target with its query string, HTTP
/// version, status, and the latency until the response head is ready.
///
/// A streaming response is logged when its stream starts, so its latency
/// does not cover the body.
pub async fn log_access(
    State(state): State<Arc<AppState>>,
    request: Request,
    next: Next,
) -> Response {
    let path = request.uri().path();
    if (state.access_log.excluded_paths.iter()).any(|excluded| excluded == path) {
        return next.run(request).await;
    }

    // Absent for Unix-domain sockets.
    let client = request
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|ConnectInfo(addr)| *addr);
    let method = request.method().clone();
    let uri = request
        .uri()
        .path_and_query()
        .map_or_else(|| path.to_owned(), |target| target.as_str().to_owned());
    let version = request.version();

    let start = Instant::now();
    let response = next.run(request).await;
    info!(
        client = client.map(display),
        %method,
        uri,
        ?version,
        status = response.status().as_u16(),
        latency = ?start.elapsed(),
        "HTTP response"
    );
    response
}
