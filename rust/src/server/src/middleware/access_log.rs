// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Per-response HTTP access log, like the uvicorn access log of Python's
//! server.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::extract::{ConnectInfo, Request, State};
use axum::middleware::Next;
use axum::response::Response;
use tracing::info;

use crate::state::AppState;

/// Log one line per response in uvicorn's access-log shape,
/// `<client> - "<METHOD> <path?query> HTTP/<version>" <status> <reason>`,
/// unless the request path is excluded.
///
/// Like uvicorn, the line is logged once the response head is ready, so a
/// streaming response is logged when its stream starts.
pub async fn log_access(
    State(state): State<Arc<AppState>>,
    request: Request,
    next: Next,
) -> Response {
    let path = request.uri().path();
    if (state.access_log.excluded_paths.iter()).any(|excluded| excluded == path) {
        return next.run(request).await;
    }

    // uvicorn prints no client for Unix-domain sockets.
    let client = request
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|ConnectInfo(addr)| addr.to_string())
        .unwrap_or_default();
    let method = request.method().clone();
    let target = request
        .uri()
        .path_and_query()
        .map_or_else(|| path.to_owned(), |target| target.as_str().to_owned());
    let version = request.version();

    let response = next.run(request).await;
    info!(
        "{client} - \"{method} {target} {version:?}\" {}",
        response.status()
    );
    response
}
