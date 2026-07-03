use std::collections::HashMap;
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
    pub session_id: Option<String>,
}

const SESSION_ID_HEADERS: [&str; 2] = ["X-Session-ID", "X-Correlation-ID"];

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

/// Inject the session id into `vllm_xargs["session_id"]`, mirroring the Python
/// frontend. An explicit body value is preserved.
pub fn merge_session_id(
    mut xargs: Option<HashMap<String, Value>>,
    session_id: Option<&str>,
) -> Option<HashMap<String, Value>> {
    if let Some(session_id) = session_id {
        let map = xargs.get_or_insert_with(HashMap::new);
        map.entry("session_id".to_string())
            .or_insert_with(|| Value::String(session_id.to_owned()));
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

/// Extract common request metadata from HTTP headers: the external request ID
/// and the optional data-parallel rank used for engine routing.
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

    // First present session header wins.
    let session_id = SESSION_ID_HEADERS.iter().find_map(|name| {
        headers
            .get(*name)
            .and_then(|value| value.to_str().ok())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
    });

    ResolvedRequestContext {
        request_id,
        data_parallel_rank,
        session_id,
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
