// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::{Arc, Weak};

use axum::extract::{MatchedPath, Request, State};
use axum::middleware::Next;
use axum::response::Response;

use super::body_guard::hold_until_body_dropped;
use crate::state::AppState;

/// Endpoints that will be tracked for server load.
///
/// Derived from the Python frontend's actual `@load_aware_call` coverage. This
/// includes alias paths that delegate into decorated handlers, such as
/// `/v1/rerank` and `/v2/rerank`.
const TRACKED_HANDLERS: &[&str] = &[
    "/v1/responses",
    "/v1/responses/{response_id}",
    "/v1/responses/{response_id}/cancel",
    "/v1/messages",
    "/v1/messages/count_tokens",
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/audio/transcriptions",
    "/v1/audio/translations",
    "/v1/embeddings",
    "/pooling",
    "/classify",
    "/score",
    "/v1/score",
    "/rerank",
    "/v1/rerank",
    "/v2/rerank",
    "/inference/v1/generate",
    "/generative_scoring",
];

/// Track frontend-local in-flight inference requests for the `/load` endpoint.
pub async fn track_server_load(
    State(state): State<Arc<AppState>>,
    req: Request,
    next: Next,
) -> Response {
    let handler = req
        .extensions()
        .get::<MatchedPath>()
        .map_or_else(|| "none", |path| path.as_str());

    if !TRACKED_HANDLERS.contains(&handler) {
        return next.run(req).await;
    }

    state.increment_server_load();
    let guard = ServerLoadGuard {
        state: Arc::downgrade(&state),
    };
    hold_until_body_dropped(next.run(req).await, guard)
}

/// A guard that decrements the server load when dropped.
///
/// It is held by the response body, so the load is decremented when the body
/// is fully consumed and dropped.
struct ServerLoadGuard {
    state: Weak<AppState>,
}

impl Drop for ServerLoadGuard {
    fn drop(&mut self) {
        if let Some(state) = self.state.upgrade() {
            state.decrement_server_load();
        }
    }
}
