// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! `POST /v1/messages/count_tokens` (Anthropic Messages API).
//!
//! Mirrors the Python endpoint: the request is lowered through the same
//! conversion path as `/v1/messages`, rendered and tokenized without engine
//! submission (the `/tokenize` precedent), and the resulting prompt length is
//! returned in Anthropic's `count_tokens` response shape. This exercises the
//! entire request-conversion layer before any generation code exists (#47753).

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use thiserror_ext::AsReport as _;

use super::convert::prepare_count_tokens_request;
use super::error::{AnthropicApiError, AnthropicJson};
use super::types::{
    AnthropicContextManagement, AnthropicCountTokensRequest, AnthropicCountTokensResponse,
};
use crate::error::{ApiError, chat_submit_error, server_error};
use crate::state::AppState;
use crate::utils::resolve_base_request_id;

// ============================================================================
// Oracle-mirrored route
// ============================================================================

/// Count prompt tokens for one Anthropic Messages request.
///
/// Mirrors the Python vLLM `count_tokens` (the api_router route plus the
/// serving method of the same name); the route/impl split is Rust-only
/// plumbing so every error takes the [`AnthropicApiError`] envelope.
pub(crate) async fn count_tokens(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    AnthropicJson(body): AnthropicJson<AnthropicCountTokensRequest>,
) -> Response {
    match count_tokens_impl(&state, &headers, body).await {
        Ok(response) => Json(response).into_response(),
        Err(error) => AnthropicApiError(error).into_response(),
    }
}

/// Mirrors the flow of the Python vLLM serving `count_tokens`: convert the
/// request, render without engine submission, and sum the prompt tokens
/// (Python sums `prompt_token_ids` across engine inputs; PR 1's text-only
/// path has exactly one). The render/tokenize steps follow the Rust
/// `/tokenize` chat precedent.
async fn count_tokens_impl(
    state: &AppState,
    headers: &HeaderMap,
    body: AnthropicCountTokensRequest,
) -> Result<AnthropicCountTokensResponse, ApiError> {
    if !state.served_model_names().iter().any(|name| name == &body.model) {
        return Err(ApiError::model_not_found(body.model));
    }

    let request_id = count_tokens_request_id(headers);
    let chat_request = prepare_count_tokens_request(body, request_id, MERGE_INLINE_SYSTEM)?;
    let text_request = state
        .chat
        .request_processor()
        .prepare_for_tokenization(chat_request)
        .await
        // Unlike the `/tokenize` precedent, validation-shaped render failures
        // (e.g. bad `chat_template_kwargs`) map to 400 here: Python's
        // `count_tokens` surfaces them through the serving layer's
        // `create_error_response`, which defaults to 400.
        .map_err(|error| chat_submit_error("count_tokens failed", error))?;
    let tokens = state
        .chat
        .text()
        .request_processor()
        .tokenize(text_request)
        .map_err(|e| server_error!("count_tokens failed: {}", e.to_report_string()))?;

    let input_tokens = tokens.len() as u64;
    Ok(AnthropicCountTokensResponse {
        input_tokens,
        context_management: AnthropicContextManagement {
            original_input_tokens: input_tokens,
        },
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
// Pieces with no named Python counterpart on this route: the provisional
// stand-in for the oracle's startup probe, and Rust-side request-ID plumbing.

/// Whether inline `role: system` messages are hoisted into the leading system
/// block.
///
/// Stands in for the Python vLLM `_detect_merge_inline_system` startup probe
/// (rendering a probe conversation against the chat template per model); we
/// ship its conservative fallback (always merge — what Python does when no
/// template is available) and defer the probe to PR 2, since it needs
/// renderer access at route-state construction time.
const MERGE_INLINE_SYSTEM: bool = true;

/// Match the `/tokenize` request-ID convention with an Anthropic prefix.
///
/// No Python counterpart on this route — request IDs come from FastAPI-side
/// middleware there; the Rust stack threads them explicitly.
fn count_tokens_request_id(headers: &HeaderMap) -> String {
    let base = resolve_base_request_id(
        headers.get("X-Request-Id").and_then(|value| value.to_str().ok()),
        None,
    );
    format!("anthropic-count-{base}")
}

#[cfg(test)]
mod tests {
    use axum::http::{HeaderMap, HeaderValue};

    use super::count_tokens_request_id;

    #[test]
    fn count_tokens_request_id_prefers_x_request_id_header() {
        let mut headers = HeaderMap::new();
        headers.insert("X-Request-Id", HeaderValue::from_static("client-req-1"));
        assert_eq!(
            count_tokens_request_id(&headers),
            "anthropic-count-client-req-1"
        );
    }

    #[test]
    fn count_tokens_request_id_generates_uuid_when_header_missing() {
        let headers = HeaderMap::new();
        let id = count_tokens_request_id(&headers);
        assert!(id.starts_with("anthropic-count-"));
        assert_ne!(id, "anthropic-count-");
    }
}
