use std::sync::Arc;

use axum::extract::{Request, State};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use tokio_util::task::AbortOnDropHandle;
use tracing::error;

use crate::error::{ApiError, server_error};
use crate::state::AppState;

/// Request paths that are run on the request runtime.
///
/// These routes can perform CPU-heavy request preparation, including JSON
/// extraction, validation, chat-template rendering, tokenization, request
/// lowering, and engine submission. Lightweight operational routes stay on the
/// HTTP runtime.
const OFFLOADED_PATHS: &[&str] = &[
    "/v1/chat/completions",
    "/v1/completions",
    "/tokenize",
    "/inference/v1/generate",
];

/// Run heavyweight inference and tokenization routes on the request runtime.
///
/// Axum extractors and route handlers execute inside `next.run(req)`, so
/// offloading here moves request parsing and preprocessing off the HTTP runtime
/// without wrapping each handler manually.
pub async fn offload_inference_routes(
    State(state): State<Arc<AppState>>,
    req: Request,
    next: Next,
) -> Response {
    if !should_offload(req.uri().path()) {
        return next.run(req).await;
    }

    // Note: For non-streaming requests, the output processing is also offloaded
    // to the request runtime by doing this.
    // For streaming requests, the SSE stream responses will still be polled on
    // the HTTP runtime. Benchmarking shows it's typically not worthwhile to poll
    // the stream on the request runtime and bridge it through a channel.
    let task = AbortOnDropHandle::new(state.request_runtime().spawn(next.run(req)));
    match task.await {
        Ok(response) => response,
        Err(error) => {
            error!(%error, "request runtime task failed");
            server_error!("request runtime task failed").into_response()
        }
    }
}

fn should_offload(path: &str) -> bool {
    OFFLOADED_PATHS.contains(&path)
}

#[cfg(test)]
mod tests {
    use super::should_offload;

    #[test]
    fn offloads_generation_and_tokenization_paths() {
        assert!(should_offload("/v1/chat/completions"));
        assert!(should_offload("/v1/completions"));
        assert!(should_offload("/tokenize"));
        assert!(should_offload("/inference/v1/generate"));
    }

    #[test]
    fn passes_through_lightweight_paths() {
        assert!(!should_offload("/health"));
        assert!(!should_offload("/metrics"));
        assert!(!should_offload("/v1/models"));
        assert!(!should_offload("/detokenize"));
    }
}
