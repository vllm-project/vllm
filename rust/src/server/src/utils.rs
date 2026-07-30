// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::{BTreeMap, HashMap};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::http::HeaderMap;
use serde_json::Value;
use thiserror_ext::AsReport;
use uuid::Uuid;

use crate::error::ApiError;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ResolvedRequestContext {
    pub request_id: String,
    pub data_parallel_rank: Option<u32>,
    pub trace_headers: Option<BTreeMap<String, String>>,
}

/// W3C trace-context headers propagated to engine-core, mirroring Python
/// vLLM's `TRACE_HEADERS` in `vllm/tracing/utils.py`.
const TRACE_HEADERS: [&str; 2] = ["traceparent", "tracestate"];

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

/// Merge `ec_transfer_params` into the `vllm_xargs` map, mirroring the Python
/// vLLM behavior where `ec_transfer_params` is injected into `extra_args` for
/// engine-core consumption.
pub fn merge_ec_transfer_params(
    mut xargs: Option<HashMap<String, Value>>,
    ec_transfer_params: Option<&HashMap<String, Value>>,
) -> Option<HashMap<String, Value>> {
    if let Some(ec_params) = ec_transfer_params {
        let map = xargs.get_or_insert_with(HashMap::new);
        map.insert(
            "ec_transfer_params".to_string(),
            // This is safe because we know that `ec_params` is already valid JSON.
            serde_json::to_value(ec_params).unwrap(),
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

/// Extract common request metadata from HTTP headers: the external request ID,
/// the optional data-parallel rank used for engine routing, and the W3C
/// trace-context headers forwarded to engine-core.
pub fn resolve_request_context(
    headers: &HeaderMap,
    request_id: Option<&str>,
) -> ResolvedRequestContext {
    // `None` when the header is absent or cannot be parsed as a `u32`.
    let data_parallel_rank = headers
        .get("X-data-parallel-rank")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse().ok());

    // Extract request id from header.
    let request_id_header = headers.get("X-Request-Id").and_then(|value| value.to_str().ok());
    let request_id = resolve_base_request_id(request_id_header, request_id);

    // Unlike Python vLLM, extraction is not gated on the engine having tracing
    // enabled: the Rust server emits no spans itself, so `trace_headers` is
    // pure propagation and is populated whenever the headers are present.
    // Non-UTF-8 header values are skipped silently.
    let trace_headers: BTreeMap<String, String> = TRACE_HEADERS
        .into_iter()
        .filter_map(|name| {
            headers
                .get(name)
                .and_then(|value| value.to_str().ok())
                .map(|value| (name.to_string(), value.to_string()))
        })
        .collect();
    let trace_headers = (!trace_headers.is_empty()).then_some(trace_headers);

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
    use std::collections::BTreeMap;

    use axum::http::HeaderMap;

    use super::resolve_request_context;

    #[test]
    fn resolve_request_context_extracts_trace_headers() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "traceparent",
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01".parse().unwrap(),
        );
        headers.insert("tracestate", "congo=t61rcWkgMzE".parse().unwrap());

        let ctx = resolve_request_context(&headers, None);
        assert_eq!(
            ctx.trace_headers,
            Some(BTreeMap::from([
                (
                    "traceparent".to_string(),
                    "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01".to_string(),
                ),
                ("tracestate".to_string(), "congo=t61rcWkgMzE".to_string()),
            ]))
        );
    }

    #[test]
    fn resolve_request_context_extracts_traceparent_alone() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "traceparent",
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01".parse().unwrap(),
        );

        let ctx = resolve_request_context(&headers, None);
        assert_eq!(
            ctx.trace_headers,
            Some(BTreeMap::from([(
                "traceparent".to_string(),
                "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01".to_string(),
            )]))
        );
    }

    #[test]
    fn resolve_request_context_leaves_trace_headers_none_when_absent() {
        let ctx = resolve_request_context(&HeaderMap::new(), None);
        assert_eq!(ctx.trace_headers, None);
    }

    #[test]
    fn resolve_request_context_ignores_non_trace_headers() {
        let mut headers = HeaderMap::new();
        headers.insert("X-Request-Id", "req-1".parse().unwrap());
        headers.insert("baggage", "key=value".parse().unwrap());

        let ctx = resolve_request_context(&headers, None);
        assert_eq!(ctx.trace_headers, None);
        assert_eq!(ctx.request_id, "req-1");
    }
}
