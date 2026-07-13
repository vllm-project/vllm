use std::sync::Arc;
use std::task::{Context, Poll};

use axum::http::Request;
use axum::response::{IntoResponse, Response};
use futures::future::BoxFuture;
use tokio_util::task::AbortOnDropHandle;
use tonic::Status;
use tower::Service;
use tower::layer::layer_fn;
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
    // HTTP routes:
    "/v1/chat/completions",
    "/v1/completions",
    "/tokenize",
    "/detokenize",
    "/inference/v1/generate",
    // gRPC routes:
    "/vllm.Generate/Generate",
    "/vllm.Generate/GenerateStream",
];

/// Return a Tower layer that runs selected data-plane requests on the request runtime,
/// so that we can offset heavy request parsing and preprocessing from the HTTP runtime.
pub(crate) fn request_runtime_layer<S>(
    state: Arc<AppState>,
) -> impl tower::Layer<S, Service = RequestRuntimeService<S>> + Clone {
    layer_fn(move |inner| RequestRuntimeService {
        inner,
        state: state.clone(),
    })
}

/// Service produced by [`request_runtime_layer`].
#[derive(Clone)]
pub(crate) struct RequestRuntimeService<S> {
    inner: S,
    state: Arc<AppState>,
}

impl<S, B> Service<Request<B>> for RequestRuntimeService<S>
where
    S: Service<Request<B>> + Clone + Send + 'static,
    S::Future: Send + 'static,
    S::Response: RequestRuntimeErrorResponse + Send + 'static,
    S::Error: Send + 'static,
    B: Send + 'static,
{
    type Error = S::Error;
    type Future = BoxFuture<'static, Result<Self::Response, Self::Error>>;
    type Response = S::Response;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request<B>) -> Self::Future {
        if !should_offload(req.uri().path()) {
            return Box::pin(self.inner.call(req));
        }

        // Axum extractors and route handlers execute inside the inner service,
        // so offloading here moves request parsing and preprocessing off the
        // HTTP runtime without wrapping each handler manually. For streaming
        // HTTP responses, the response body is still polled on the HTTP runtime.
        let clone = self.inner.clone();
        let mut inner = std::mem::replace(&mut self.inner, clone);
        let task = AbortOnDropHandle::new(self.state.request_runtime().spawn(inner.call(req)));

        Box::pin(async move {
            match task.await {
                Ok(result) => result,
                Err(error) => {
                    error!(%error, "request runtime task failed");
                    Ok(S::Response::request_runtime_error_response())
                }
            }
        })
    }
}

trait RequestRuntimeErrorResponse {
    fn request_runtime_error_response() -> Self;
}

impl RequestRuntimeErrorResponse for Response {
    fn request_runtime_error_response() -> Self {
        server_error!("request runtime task failed").into_response()
    }
}

impl RequestRuntimeErrorResponse for axum::http::Response<tonic::body::Body> {
    fn request_runtime_error_response() -> Self {
        Status::internal("request runtime task failed").into_http()
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
        assert!(should_offload("/detokenize"));
        assert!(should_offload("/inference/v1/generate"));
        assert!(should_offload("/vllm.Generate/Generate"));
        assert!(should_offload("/vllm.Generate/GenerateStream"));
    }

    #[test]
    fn passes_through_lightweight_paths() {
        assert!(!should_offload("/health"));
        assert!(!should_offload("/metrics"));
        assert!(!should_offload("/v1/models"));
    }
}
