// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Instant;

use axum::body::{Body, Bytes, HttpBody};
use axum::extract::{MatchedPath, Request};
use axum::http::Method;
use axum::middleware::Next;
use axum::response::Response;
use http_body::{Frame, SizeHint};
use vllm_metrics::{HttpHandlerLabels, HttpRequestLabels, METRICS};

/// Endpoints that will be excluded from HTTP metrics tracking.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/entrypoints/serve/instrumentator/metrics.py#L28-L38>
const EXCLUDED_HANDLERS: &[&str] = &[
    "/metrics",
    "/health",
    "/load",
    "/ping",
    "/version",
    "/server_info",
    // Rust frontend extra:
    "/reset_prefix_cache",
    "/reset_mm_cache",
    "/reset_encoder_cache",
    "/collective_rpc",
];

/// Map the request method to a bounded Prometheus label.
///
/// HTTP parsers accept arbitrary method tokens. Recording the raw token as a
/// label would let unique values create unbounded time series.
pub(crate) fn http_metrics_method(method: &Method) -> &str {
    match *method {
        Method::GET
        | Method::POST
        | Method::PUT
        | Method::PATCH
        | Method::DELETE
        | Method::HEAD
        | Method::OPTIONS
        | Method::CONNECT
        | Method::TRACE => method.as_str(),
        _ => "other",
    }
}

/// Record API-server HTTP metrics with Python-compatible
/// (`PrometheusFastApiInstrumentator` style) family names and labels.
pub async fn track_http_metrics(req: Request, next: Next) -> Response {
    let method = http_metrics_method(req.method()).to_string();
    let handler = req
        .extensions()
        .get::<MatchedPath>()
        .map_or_else(|| "none".to_string(), |path| path.as_str().to_string());

    if EXCLUDED_HANDLERS.contains(&handler.as_str()) {
        return next.run(req).await;
    }

    let mut guard = HttpMetricsGuard {
        method,
        handler,
        // Match the Python middleware's fallback before response headers exist.
        status: "5xx",
        started_at: Instant::now(),
    };
    let response = next.run(req).await;
    guard.status = status_group(response.status().as_u16());

    let (parts, body) = response.into_parts();
    Response::from_parts(
        parts,
        Body::new(HttpMetricsBody {
            inner: body,
            _guard: guard,
        }),
    )
}

struct HttpMetricsGuard {
    method: String,
    handler: String,
    status: &'static str,
    started_at: Instant,
}

impl Drop for HttpMetricsGuard {
    fn drop(&mut self) {
        let elapsed = self.started_at.elapsed().as_secs_f64();
        let metrics = &METRICS.api_server;

        metrics
            .http_requests
            .get_or_create(&HttpRequestLabels {
                method: self.method.clone(),
                status: self.status,
                handler: self.handler.clone(),
            })
            .inc();

        metrics
            .http_request_duration_seconds
            .get_or_create(&HttpHandlerLabels {
                method: self.method.clone(),
                handler: self.handler.clone(),
            })
            .observe(elapsed);

        metrics.http_request_duration_highr_seconds.observe(elapsed);
    }
}

struct HttpMetricsBody {
    inner: Body,
    _guard: HttpMetricsGuard,
}

impl HttpBody for HttpMetricsBody {
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

fn status_group(status: u16) -> &'static str {
    match status / 100 {
        1 => "1xx",
        2 => "2xx",
        3 => "3xx",
        4 => "4xx",
        5 => "5xx",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_methods_keep_their_names() {
        for (method, expected) in [
            (Method::GET, "GET"),
            (Method::POST, "POST"),
            (Method::PUT, "PUT"),
            (Method::PATCH, "PATCH"),
            (Method::DELETE, "DELETE"),
            (Method::HEAD, "HEAD"),
            (Method::OPTIONS, "OPTIONS"),
            (Method::CONNECT, "CONNECT"),
            (Method::TRACE, "TRACE"),
        ] {
            assert_eq!(http_metrics_method(&method), expected);
        }
    }

    #[test]
    fn unknown_methods_collapse_to_other() {
        let method = Method::from_bytes(b"XVULNCARD000001").expect("valid token");
        assert_eq!(http_metrics_method(&method), "other");
        let other = Method::from_bytes(b"XVULNCARD000002").expect("valid token");
        assert_eq!(http_metrics_method(&other), "other");
    }
}
