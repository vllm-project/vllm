use std::collections::{BTreeMap, HashMap};
use std::sync::Once;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::http::HeaderMap;
use serde_json::Value;
use thiserror_ext::AsReport;
use tracing::warn;
use uuid::Uuid;

use crate::error::ApiError;

const TRACE_HEADERS: [&str; 2] = ["traceparent", "tracestate"];

static TRACING_DISABLED_WARNING: Once = Once::new();

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ResolvedRequestContext {
    pub request_id: String,
    pub data_parallel_rank: Option<u32>,
    pub trace_headers: Option<BTreeMap<String, String>>,
}

/// Return the current Unix timestamp in seconds for OpenAI response objects.
pub fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

/// Construct an API error for a failed utility call to the engine core.
pub fn utility_call_error(method: &str, error: impl AsReport) -> ApiError {
    ApiError::server_error(format!("failed to call {method}: {}", error.as_report()))
}

/// Merge `kv_transfer_params` into the `vllm_xargs` map, mirroring the Python
/// vLLM behavior where `kv_transfer_params` is injected into `extra_args` for
/// engine-core consumption.
pub fn merge_kv_transfer_params(
    mut xargs: Option<HashMap<String, Value>>,
    kv_transfer_params: Option<&HashMap<String, Value>>,
) -> Option<HashMap<String, Value>> {
    if let Some(kv_params) = kv_transfer_params {
        let map = xargs.get_or_insert_with(HashMap::new);
        map.insert(
            "kv_transfer_params".to_string(),
            // This is safe because we know that `kv_params` is already valid JSON.
            serde_json::to_value(kv_params).unwrap(),
        );
    }
    xargs
}

/// Convert OpenAI-style `logit_bias` with string token-ID keys into the
/// internal `HashMap<u32, f32>` representation, validating that every key
/// parses as a `u32`.
pub fn convert_logit_bias(
    logit_bias: Option<HashMap<String, f32>>,
) -> Result<Option<HashMap<u32, f32>>, ApiError> {
    logit_bias
        .map(|bias| {
            bias.into_iter()
                .map(|(key, value)| {
                    key.parse().map(|k| (k, value)).map_err(|_| {
                        ApiError::invalid_request(
                            format!(
                                "Invalid key in 'logit_bias': '{key}' is not a valid token ID. \
                                 Token IDs must be non-negative integers."
                            ),
                            Some("logit_bias"),
                        )
                    })
                })
                .collect()
        })
        .transpose()
}

fn contains_trace_headers(headers: &HeaderMap) -> bool {
    TRACE_HEADERS.iter().any(|name| headers.contains_key(*name))
}

/// Extract the W3C trace-context headers (`traceparent`/`tracestate`) into a
/// lowercase-keyed map. Mirrors Python `vllm.tracing.utils.extract_trace_headers`.
fn extract_trace_headers(headers: &HeaderMap) -> BTreeMap<String, String> {
    TRACE_HEADERS
        .iter()
        .filter_map(|name| {
            headers
                .get(*name)
                .and_then(|value| value.to_str().ok())
                .map(|value| ((*name).to_string(), value.to_string()))
        })
        .collect()
}

/// Resolve the trace headers to forward to the engine, gated on whether the
/// engine has tracing enabled. Mirrors Python `_get_trace_headers`.
fn resolve_trace_headers(
    headers: &HeaderMap,
    tracing_enabled: bool,
) -> Option<BTreeMap<String, String>> {
    if tracing_enabled {
        return Some(extract_trace_headers(headers));
    }
    if contains_trace_headers(headers) {
        TRACING_DISABLED_WARNING.call_once(|| {
            warn!("Received a request with trace context but tracing is disabled");
        });
    }
    None
}

/// Extract common request metadata from HTTP headers: the external request ID,
/// the optional data-parallel rank used for engine routing, and the trace
/// headers to forward to the engine when tracing is enabled.
pub fn resolve_request_context(
    headers: &HeaderMap,
    request_id: Option<&str>,
    tracing_enabled: bool,
) -> ResolvedRequestContext {
    // `None` when the header is absent or cannot be parsed as a `u32`.
    let data_parallel_rank = headers
        .get("X-data-parallel-rank")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse().ok());

    // Extract request id from header.
    let request_id_header = headers.get("X-Request-Id").and_then(|value| value.to_str().ok());
    let request_id = resolve_base_request_id(request_id_header, request_id);

    let trace_headers = resolve_trace_headers(headers, tracing_enabled);

    ResolvedRequestContext {
        request_id,
        data_parallel_rank,
        trace_headers,
    }
}

/// Resolve the base external request ID before API-specific prefixes such as
/// `chatcmpl-`.
pub fn resolve_base_request_id(
    request_id_header: Option<&str>,
    request_id: Option<&str>,
) -> String {
    request_id_header.or(request_id).map(ToOwned::to_owned).unwrap_or_else(|| {
        let mut id = Uuid::new_v4().simple().to_string();
        id.truncate(8);
        id
    })
}

#[cfg(test)]
mod tests {
    use axum::http::{HeaderMap, HeaderName, HeaderValue};

    use super::*;

    fn headers_from(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut headers = HeaderMap::new();
        for (name, value) in pairs {
            headers.insert(
                HeaderName::from_bytes(name.as_bytes()).expect("valid header name"),
                HeaderValue::from_str(value).expect("valid header value"),
            );
        }
        headers
    }

    #[test]
    fn forwards_w3c_trace_headers_when_tracing_enabled() {
        let headers = headers_from(&[("traceparent", "abc"), ("tracestate", "k=v")]);
        let ctx = resolve_request_context(&headers, Some("req"), true);
        let trace = ctx.trace_headers.expect("trace headers forwarded");
        assert_eq!(trace.get("traceparent").map(String::as_str), Some("abc"));
        assert_eq!(trace.get("tracestate").map(String::as_str), Some("k=v"));
        assert_eq!(trace.len(), 2);
    }

    #[test]
    fn forwards_empty_map_when_enabled_without_trace_headers() {
        let ctx = resolve_request_context(&HeaderMap::new(), Some("req"), true);
        let trace = ctx.trace_headers.expect("Some empty map mirrors Python");
        assert!(trace.is_empty());
    }

    #[test]
    fn forwards_only_w3c_trace_headers() {
        let headers = headers_from(&[
            ("traceparent", "abc"),
            ("x-request-id", "not-forwarded"),
            ("baggage", "not-forwarded"),
        ]);
        let ctx = resolve_request_context(&headers, Some("req"), true);
        let trace = ctx.trace_headers.expect("trace headers forwarded");
        assert_eq!(trace.len(), 1);
        assert_eq!(trace.get("traceparent").map(String::as_str), Some("abc"));
    }

    #[test]
    fn normalizes_trace_header_names_to_lowercase() {
        let headers = headers_from(&[("TraceParent", "abc")]);
        let ctx = resolve_request_context(&headers, Some("req"), true);
        let trace = ctx.trace_headers.expect("trace headers forwarded");
        assert_eq!(trace.get("traceparent").map(String::as_str), Some("abc"));
    }

    #[test]
    fn suppresses_trace_headers_when_tracing_disabled() {
        let headers = headers_from(&[("traceparent", "abc")]);
        let ctx = resolve_request_context(&headers, Some("req"), false);
        assert!(ctx.trace_headers.is_none());
    }
}
