// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! `POST /tokenize` and `POST /detokenize` (root paths, matching Python).
//!
//! Encode/decode runs entirely in-process via [`DynTokenizer`]; the inference
//! engine is not involved.

mod types;

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::HeaderMap;

use crate::error::{ApiError, chat_submit_error, text_submit_error};
use crate::routes::openai::utils::validated_json::ValidatedJson;
use crate::routes::tokenize::types::{
    DetokenizeRequest, DetokenizeResponse, TokenizeChatRequest, TokenizeCompletionRequest,
    TokenizeRequest, TokenizeResponse,
};
use crate::state::AppState;
use crate::utils::{header_value, resolve_base_request_id};

/// Match Python `tokenize-{base}` where base is `X-Request-Id` or a new UUID.
fn tokenize_request_id(headers: &HeaderMap) -> Result<String, ApiError> {
    let header: Option<String> = header_value(headers, "X-Request-Id")?;
    let base = resolve_base_request_id(header.as_deref(), None);
    Ok(format!("tokenize-{base}"))
}

/// Build the `token_strs` vector when requested, via the tokenizer vocab.
fn token_strs(tokenizer: &vllm_text::tokenizer::DynTokenizer, ids: &[u32]) -> Vec<String> {
    // Unknown IDs yield "" — intentional; matches Python's convert_ids_to_tokens behaviour.
    ids.iter().map(|&id| tokenizer.id_to_token(id).unwrap_or_default()).collect()
}

fn resolve_max_tokenize_tokens(
    configured_limit: Option<usize>,
    max_model_len: u32,
) -> Option<usize> {
    match configured_limit {
        Some(0) => None,
        Some(limit) => Some(limit),
        None => Some((max_model_len as usize * 2).max(131_072)),
    }
}

fn check_max_tokenize_tokens(count: usize, limit: usize, qualifier: &str) -> Result<(), ApiError> {
    if count > limit {
        return Err(ApiError::InvalidRequest {
            message: format!(
                "This model's maximum context length is {limit} tokens. However, your prompt contains {qualifier}{count} input tokens, which exceeds this limit even without any output tokens. Please reduce the length of the input prompt."
            ),
            param: Some("input_tokens"),
        });
    }
    Ok(())
}

pub async fn tokenize(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<TokenizeRequest>,
) -> Result<Json<TokenizeResponse>, ApiError> {
    let request_id = tokenize_request_id(&headers)?;
    let tokenizer = state.chat.text().tokenizer();
    let max_model_len = state.llm().info().max_model_len;
    let max_tokenize_tokens =
        resolve_max_tokenize_tokens(state.api_server_options.max_tokenize_tokens, max_model_len);

    let (tokens, want_strs) = match body {
        TokenizeRequest::Completion(req) => {
            tokenize_completion(&state, &request_id, req, max_tokenize_tokens)?
        }
        TokenizeRequest::Chat(req) => {
            tokenize_chat(&state, &request_id, req, max_tokenize_tokens).await?
        }
    };

    let token_strs = want_strs.then(|| token_strs(&tokenizer, &tokens));
    Ok(Json(TokenizeResponse {
        count: tokens.len(),
        max_model_len,
        tokens,
        token_strs,
    }))
}

fn tokenize_completion(
    state: &AppState,
    request_id: &str,
    req: TokenizeCompletionRequest,
    max_tokenize_tokens: Option<usize>,
) -> Result<(Vec<u32>, bool), ApiError> {
    state.base_model_names().check(req.model.as_deref())?;
    let tokenizer = state.chat.text().tokenizer();
    if let Some(limit) = max_tokenize_tokens {
        let estimated = tokenizer.estimate_max_token(&req.prompt);
        check_max_tokenize_tokens(estimated, limit, "at least ")?;
    }
    let return_token_strs = req.return_token_strs;
    let tokens = state
        .chat
        .text()
        .request_processor()
        .tokenize(req.into_text_request(request_id.to_string()))
        .map_err(|error| text_submit_error("tokenize failed", error))?;
    if let Some(limit) = max_tokenize_tokens {
        check_max_tokenize_tokens(tokens.len(), limit, "")?;
    }
    Ok((tokens, return_token_strs))
}

/// HTTP adapter for the chat-shaped `/tokenize` body.
///
/// Checks the model name, prepares without engine submission, and maps errors
/// to [`ApiError`].
async fn tokenize_chat(
    state: &AppState,
    request_id: &str,
    req: TokenizeChatRequest,
    max_tokenize_tokens: Option<usize>,
) -> Result<(Vec<u32>, bool), ApiError> {
    state.base_model_names().check(req.model.as_deref())?;
    let return_token_strs = req.return_token_strs;
    // `continue_final_message` / `add_generation_prompt` mutual exclusion is
    // enforced in `normalize_generation_prompt_mode` inside `into_chat_request`.
    let text_request = state
        .chat
        .request_processor()
        .prepare_for_tokenization(req.into_chat_request(request_id.to_string())?)
        .await
        .map_err(|error| chat_submit_error("tokenize failed", error))?;
    let tokens = state
        .chat
        .text()
        .request_processor()
        .tokenize(text_request)
        .map_err(|error| text_submit_error("tokenize failed", error))?;
    if let Some(limit) = max_tokenize_tokens {
        check_max_tokenize_tokens(tokens.len(), limit, "")?;
    }
    Ok((tokens, return_token_strs))
}

pub async fn detokenize(
    State(state): State<Arc<AppState>>,
    ValidatedJson(body): ValidatedJson<DetokenizeRequest>,
) -> Result<Json<DetokenizeResponse>, ApiError> {
    state.base_model_names().check(body.model.as_deref())?;
    let max_model_len = state.llm().info().max_model_len;
    let max_tokenize_tokens =
        resolve_max_tokenize_tokens(state.api_server_options.max_tokenize_tokens, max_model_len);
    if let Some(limit) = max_tokenize_tokens {
        check_max_tokenize_tokens(body.tokens.len(), limit, "")?;
    }
    let tokenizer = state.chat.text().tokenizer();
    let prompt = tokenizer
        .decode(&body.tokens, /* skip_special_tokens = */ false)
        .map_err(|error| text_submit_error("detokenize failed", error.into()))?;
    Ok(Json(DetokenizeResponse { prompt }))
}

#[cfg(test)]
mod tests {
    use axum::http::{HeaderMap, HeaderValue};

    use super::{check_max_tokenize_tokens, resolve_max_tokenize_tokens, tokenize_request_id};
    use crate::error::ApiError;

    #[test]
    fn resolve_max_tokenize_tokens_explicit() {
        assert_eq!(resolve_max_tokenize_tokens(Some(100), 2048), Some(100));
        assert_eq!(resolve_max_tokenize_tokens(Some(0), 2048), None);
    }

    #[test]
    fn resolve_max_tokenize_tokens_default() {
        // max(max_model_len * 2, 131_072)
        assert_eq!(resolve_max_tokenize_tokens(None, 2048), Some(131_072));
        assert_eq!(resolve_max_tokenize_tokens(None, 100_000), Some(200_000));
    }

    #[test]
    fn check_max_tokenize_tokens_bounds() {
        assert!(check_max_tokenize_tokens(10, 10, "").is_ok());
        let err = check_max_tokenize_tokens(11, 10, "at least ").unwrap_err();
        match err {
            ApiError::InvalidRequest { message, param } => {
                assert_eq!(param, Some("input_tokens"));
                assert!(message.contains("maximum context length is 10"));
                assert!(message.contains("at least 11 input tokens"));
            }
            _ => panic!("unexpected error type"),
        }
    }

    #[test]
    fn tokenize_request_id_prefers_x_request_id_header() {
        let mut headers = HeaderMap::new();
        headers.insert("X-Request-Id", HeaderValue::from_static("client-req-1"));
        assert_eq!(
            tokenize_request_id(&headers).unwrap(),
            "tokenize-client-req-1"
        );
    }

    #[test]
    fn tokenize_request_id_generates_uuid_when_header_missing() {
        let headers = HeaderMap::new();
        let id = tokenize_request_id(&headers).unwrap();
        assert!(id.starts_with("tokenize-"));
        assert_ne!(id, "tokenize-");
    }
}
