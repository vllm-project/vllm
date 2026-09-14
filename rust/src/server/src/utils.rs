// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::BoxError;
use axum::http::HeaderMap;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures::Stream;
use serde::de::Error as _;
use serde::{Deserialize, Deserializer};
use serde_json::Value;
use thiserror_ext::AsReport;
use uuid::Uuid;

use crate::error::{ApiError, bail_invalid_request};

mod npy;
pub(crate) use npy::npy_base64;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ResolvedRequestContext {
    pub request_id: String,
    pub data_parallel_rank: Option<u32>,
    pub priority: Option<i32>,
    pub session_id: Option<String>,
}

/// Return the current Unix timestamp in seconds for OpenAI response objects.
pub fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

/// Build a streaming SSE response, optionally sending keep-alive comments
/// while the stream is idle.
///
/// With `keep_alive_interval` set, a `: keep-alive` comment is sent whenever no
/// event has been produced for that long (queued, prefill, between tokens, or
/// while a parser buffers a long tool call). SSE parsers ignore comments, but
/// they still count as bytes for read timeouts in reverse proxies and clients.
pub fn sse_response<S, E>(stream: S, keep_alive_interval: Option<Duration>) -> Response
where
    S: Stream<Item = Result<Event, E>> + Send + 'static,
    E: Into<BoxError>,
{
    let sse = Sse::new(stream);
    match keep_alive_interval {
        Some(interval) => sse
            .keep_alive(KeepAlive::new().interval(interval).text("keep-alive"))
            .into_response(),
        None => sse.into_response(),
    }
}

/// Construct an API error for a failed utility call to the engine core.
pub fn utility_call_error(method: &str, error: impl AsReport) -> ApiError {
    ApiError::server_error(format!("failed to call {method}: {}", error.as_report()))
}

/// Deserialize a boolean query parameter with the spellings FastAPI accepts.
///
/// Query values are text, and serde's `bool` only takes lowercase `true` and
/// `false`, while clients of the Python server also send others: `requests`
/// encodes Python's `True` as `True`.
pub(crate) fn deserialize_query_bool<'de, D>(deserializer: D) -> Result<bool, D::Error>
where
    D: Deserializer<'de>,
{
    let value = String::deserialize(deserializer)?;
    match value.to_ascii_lowercase().as_str() {
        "true" | "t" | "yes" | "y" | "on" | "1" => Ok(true),
        "false" | "f" | "no" | "n" | "off" | "0" => Ok(false),
        _ => Err(D::Error::custom(format!("invalid boolean value `{value}`"))),
    }
}

/// Merge `kv_transfer_params` into the `vllm_xargs` map, mirroring the Python
/// vLLM behavior where `kv_transfer_params` is injected into `extra_args` for
/// engine-core consumption.
///
/// Rejects a caller-supplied `kv_transfer_params` key inside `vllm_xargs`;
/// that field must be sent as a top-level request field.
pub fn merge_kv_transfer_params(
    mut xargs: Option<HashMap<String, Value>>,
    kv_transfer_params: Option<&HashMap<String, Value>>,
) -> Result<Option<HashMap<String, Value>>, ApiError> {
    reject_reserved_xarg(xargs.as_ref(), "kv_transfer_params")?;
    if let Some(kv_params) = kv_transfer_params {
        let map = xargs.get_or_insert_with(HashMap::new);
        map.insert(
            "kv_transfer_params".to_string(),
            // This is safe because we know that `kv_params` is already valid JSON.
            serde_json::to_value(kv_params).unwrap(),
        );
    }
    Ok(xargs)
}

/// Merge `ec_transfer_params` into the `vllm_xargs` map, mirroring the Python
/// vLLM behavior where `ec_transfer_params` is injected into `extra_args` for
/// engine-core consumption.
///
/// Rejects a caller-supplied `ec_transfer_params` key inside `vllm_xargs`;
/// that field must be sent as a top-level request field.
pub fn merge_ec_transfer_params(
    mut xargs: Option<HashMap<String, Value>>,
    ec_transfer_params: Option<&HashMap<String, Value>>,
) -> Result<Option<HashMap<String, Value>>, ApiError> {
    reject_reserved_xarg(xargs.as_ref(), "ec_transfer_params")?;
    if let Some(ec_params) = ec_transfer_params {
        let map = xargs.get_or_insert_with(HashMap::new);
        map.insert(
            "ec_transfer_params".to_string(),
            // This is safe because we know that `ec_params` is already valid JSON.
            serde_json::to_value(ec_params).unwrap(),
        );
    }
    Ok(xargs)
}

fn reject_reserved_xarg(
    xargs: Option<&HashMap<String, Value>>,
    reserved: &'static str,
) -> Result<(), ApiError> {
    if xargs.is_some_and(|map| map.contains_key(reserved)) {
        return Err(ApiError::invalid_request(
            format!("'{reserved}' must be sent as a top-level field, not via vllm_xargs"),
            Some("vllm_xargs"),
        ));
    }
    Ok(())
}

pub fn resolve_session_id(
    ctx: &ResolvedRequestContext,
    request_session_id: Option<&str>,
    xargs: Option<&HashMap<String, Value>>,
) -> Option<String> {
    request_session_id
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .or_else(|| ctx.session_id.clone())
        .or_else(|| {
            xargs
                .and_then(|map| map.get("session_id"))
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
        })
}

/// Convert OpenAI-style `logit_bias` with string token-ID keys into the
/// internal `HashMap<u32, f32>` representation, validating that every key
/// parses as a `u32`. All invalid keys are reported in one error, like Python.
pub fn convert_logit_bias(
    logit_bias: Option<HashMap<String, f32>>,
) -> Result<Option<HashMap<u32, f32>>, ApiError> {
    let Some(logit_bias) = logit_bias else {
        return Ok(None);
    };

    let mut converted = HashMap::with_capacity(logit_bias.len());
    let mut invalid_keys = Vec::new();
    for (key, value) in logit_bias {
        match key.parse() {
            Ok(token_id) => {
                converted.insert(token_id, value);
            }
            Err(_) => invalid_keys.push(key),
        }
    }
    if !invalid_keys.is_empty() {
        // The JSON object's order is lost in the map, so sort for a stable
        // message.
        invalid_keys.sort_unstable();
        bail_invalid_request!(
            param = "logit_bias",
            "logit_bias contains key(s) that are not valid token IDs: {invalid_keys:?}. \
             Token IDs must be non-negative integers."
        );
    }
    Ok(Some(converted))
}

/// Extract common request metadata from HTTP headers: the external request ID,
/// session ID, priority, and the optional data-parallel rank used for engine
/// routing.
pub fn resolve_request_context(
    headers: &HeaderMap,
    request_id: Option<&str>,
) -> ResolvedRequestContext {
    // `None` when the header is absent or cannot be parsed as a `u32`.
    let data_parallel_rank = headers
        .get("X-data-parallel-rank")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse().ok());
    let priority = headers
        .get("X-Vllm-Priority")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.trim().parse().ok());

    // Extract request id from header.
    let request_id_header = headers.get("X-Request-Id").and_then(|value| value.to_str().ok());
    let request_id = resolve_base_request_id(request_id_header, request_id);
    let session_id = headers
        .get("X-Session-ID")
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.is_empty())
        .map(str::to_owned);

    ResolvedRequestContext {
        request_id,
        data_parallel_rank,
        priority,
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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::convert::Infallible;

    use axum::extract::Query;
    use axum::http::Uri;
    use futures::{StreamExt as _, stream};
    use serde_json::json;
    use tokio::time::{Instant, sleep};

    use super::*;

    /// The spellings FastAPI (pydantic) accepts for a `bool` query parameter,
    /// probed against a live FastAPI app. Everything else is rejected there too.
    #[test]
    fn query_bool_matches_fastapi_spellings() {
        #[derive(Debug, Deserialize)]
        struct Params {
            #[serde(deserialize_with = "deserialize_query_bool")]
            flag: bool,
        }
        let parse = |value: &str| {
            let uri: Uri = format!("/?flag={value}").parse().expect("valid uri");
            Query::<Params>::try_from_uri(&uri).map(|Query(params)| params.flag).ok()
        };

        for value in [
            "true", "True", "TRUE", "tRuE", "t", "T", "yes", "Yes", "YES", "y", "Y", "on", "On",
            "ON", "1",
        ] {
            assert_eq!(parse(value), Some(true), "{value}");
        }
        for value in [
            "false", "False", "FALSE", "f", "F", "no", "No", "NO", "n", "N", "off", "Off", "OFF",
            "0",
        ] {
            assert_eq!(parse(value), Some(false), "{value}");
        }
        for value in [
            "", "%201", "1%20", "%20true", "2", "-1", "1.0", "0.0", "maybe", "truth", "none",
            "null",
        ] {
            assert_eq!(parse(value), None, "{value}");
        }
    }

    /// Every key that is not a token ID is named in one error, in a stable
    /// order, instead of only the first one the map yields.
    #[test]
    fn convert_logit_bias_reports_all_invalid_keys() {
        let logit_bias = HashMap::from([
            ("bad2".to_string(), 20.0),
            ("7".to_string(), 1.0),
            ("bad1".to_string(), 50.0),
            ("-1".to_string(), 5.0),
        ]);
        let error = convert_logit_bias(Some(logit_bias)).expect_err("invalid keys");
        expect_test::expect![[r#"
            InvalidRequest {
                message: "logit_bias contains key(s) that are not valid token IDs: [\"-1\", \"bad1\", \"bad2\"]. Token IDs must be non-negative integers.",
                param: Some(
                    "logit_bias",
                ),
            }
        "#]].assert_debug_eq(&error);
    }

    #[test]
    fn convert_logit_bias_converts_valid_keys() {
        let logit_bias = HashMap::from([("7".to_string(), 1.0), ("42".to_string(), -2.5)]);
        let converted = convert_logit_bias(Some(logit_bias)).expect("valid keys");
        assert_eq!(converted, Some(HashMap::from([(7, 1.0), (42, -2.5)])));
    }

    /// Collect `(elapsed seconds, frame)` pairs from a response whose event
    /// stream idles for 20 seconds between its two events.
    async fn idle_stream_frames(keep_alive_interval: Option<Duration>) -> Vec<(u64, String)> {
        let events = stream::iter([Ok::<_, Infallible>(Event::default().data("first"))]).chain(
            stream::once(async {
                sleep(Duration::from_secs(20)).await;
                Ok(Event::default().data("second"))
            }),
        );
        let start = Instant::now();
        sse_response(events, keep_alive_interval)
            .into_body()
            .into_data_stream()
            .map(|frame| {
                let frame = frame.expect("read frame");
                (
                    start.elapsed().as_secs(),
                    String::from_utf8(frame.to_vec()).expect("utf-8 frame"),
                )
            })
            .collect()
            .await
    }

    /// An idle gap is filled with `: keep-alive` comments at the configured
    /// interval, and real events still pass through in order.
    #[tokio::test(start_paused = true)]
    async fn sse_response_sends_keep_alive_comments_while_idle() {
        let frames = idle_stream_frames(Some(Duration::from_secs(8))).await;
        expect_test::expect![[r#"
            [
                (
                    0,
                    "data: first\n\n",
                ),
                (
                    8,
                    ": keep-alive\n\n",
                ),
                (
                    16,
                    ": keep-alive\n\n",
                ),
                (
                    20,
                    "data: second\n\n",
                ),
            ]
        "#]]
        .assert_debug_eq(&frames);
    }

    /// Without an interval, an idle gap produces no extra bytes.
    #[tokio::test(start_paused = true)]
    async fn sse_response_without_keep_alive_interval_sends_no_comments() {
        let frames = idle_stream_frames(None).await;
        expect_test::expect![[r#"
            [
                (
                    0,
                    "data: first\n\n",
                ),
                (
                    20,
                    "data: second\n\n",
                ),
            ]
        "#]]
        .assert_debug_eq(&frames);
    }

    #[test]
    fn merge_kv_rejects_reserved_xargs_key() {
        let mut xargs = HashMap::new();
        xargs.insert("kv_transfer_params".to_string(), json!("x"));
        let err = merge_kv_transfer_params(Some(xargs), None).unwrap_err();
        assert!(matches!(
            err,
            ApiError::InvalidRequest {
                param: Some("vllm_xargs"),
                ..
            }
        ));
    }

    #[test]
    fn merge_ec_rejects_reserved_xargs_key() {
        let mut xargs = HashMap::new();
        xargs.insert("ec_transfer_params".to_string(), json!({"a": 1}));
        let err = merge_ec_transfer_params(Some(xargs), None).unwrap_err();
        assert!(matches!(
            err,
            ApiError::InvalidRequest {
                param: Some("vllm_xargs"),
                ..
            }
        ));
    }

    #[test]
    fn merge_kv_overlays_typed_field() {
        let mut xargs = HashMap::new();
        xargs.insert("custom".to_string(), json!(1));
        let mut kv = HashMap::new();
        kv.insert("do_remote_prefill".to_string(), json!(true));
        let merged = merge_kv_transfer_params(Some(xargs), Some(&kv)).unwrap().unwrap();
        assert_eq!(merged["custom"], json!(1));
        assert_eq!(
            merged["kv_transfer_params"]["do_remote_prefill"],
            json!(true)
        );
    }
}
