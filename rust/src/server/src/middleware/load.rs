use std::pin::Pin;
use std::sync::{Arc, Weak};
use std::task::{Context, Poll};

use axum::body::{Body, Bytes, HttpBody};
use axum::extract::{MatchedPath, Request, State};
use axum::middleware::Next;
use axum::response::Response;
use http_body::{Frame, SizeHint};

use crate::state::AppState;

/// Endpoints that will be tracked for server load.
///
/// Derived from the Python frontend's actual `@load_aware_call` coverage. This includes alias
/// paths that delegate into decorated handlers, such as `/v1/rerank` and `/v2/rerank`.
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
    let response = next.run(req).await;

    let (parts, body) = response.into_parts();
    Response::from_parts(
        parts,
        Body::new(LoadTrackedBody {
            inner: body,
            _guard: guard,
        }),
    )
}

/// A guard that decrements the server load when dropped.
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

/// A wrapper around response bodies that tracks server load by holding a `ServerLoadGuard`, which
/// will decrement the load when the body is fully consumed and dropped.
struct LoadTrackedBody {
    inner: Body,
    _guard: ServerLoadGuard,
}

// Simply delegate all `HttpBody` methods to the inner body.
impl HttpBody for LoadTrackedBody {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        Pin::new(&mut self.inner).poll_frame(cx)
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}
