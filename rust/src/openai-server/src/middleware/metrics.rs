use std::time::Instant;

use axum::extract::{MatchedPath, Request};
use axum::middleware::Next;
use axum::response::Response;
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
    "/is_sleeping",
];

/// Record API-server HTTP metrics with Python-compatible (`PrometheusFastApiInstrumentator` style)
/// family names and labels.
pub async fn track_http_metrics(req: Request, next: Next) -> Response {
    let method = req.method().as_str().to_string();
    let handler = req
        .extensions()
        .get::<MatchedPath>()
        .map_or_else(|| "none".to_string(), |path| path.as_str().to_string());
    let excluded = EXCLUDED_HANDLERS.contains(&handler.as_str());
    let started_at = Instant::now();

    let response = next.run(req).await;

    if excluded {
        return response;
    }

    let elapsed = started_at.elapsed().as_secs_f64();
    let status = status_group(response.status().as_u16());

    let metrics = &METRICS.api_server;

    metrics
        .http_requests
        .get_or_create(&HttpRequestLabels {
            method: method.clone(),
            status,
            handler: handler.clone(),
        })
        .inc();

    metrics
        .http_request_duration_seconds
        .get_or_create(&HttpHandlerLabels { method, handler })
        .observe(elapsed);

    metrics.http_request_duration_highr_seconds.observe(elapsed);

    response
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
