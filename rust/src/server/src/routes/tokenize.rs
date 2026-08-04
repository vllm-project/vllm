//! `POST /tokenize` and `POST /detokenize` (root paths, matching Python).
//!
//! Encode/decode runs entirely in-process via [`DynTokenizer`]; the inference
//! engine is not involved.

mod types;

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use thiserror_ext::AsReport as _;

use crate::error::{ApiError, server_error};
use crate::routes::openai::utils::validated_json::ValidatedJson;
use crate::routes::tokenize::types::{
    DetokenizeRequest, DetokenizeResponse, TokenizeChatRequest, TokenizeCompletionRequest,
    TokenizeRequest, TokenizeResponse,
};
use crate::state::AppState;
use crate::utils::resolve_base_request_id;

/// Match Python `tokenize-{base}` where base is `X-Request-Id` or a new UUID.
fn tokenize_request_id(headers: &HeaderMap) -> String {
    let base = resolve_base_request_id(
        headers.get("X-Request-Id").and_then(|value| value.to_str().ok()),
        None,
    );
    format!("tokenize-{base}")
}

/// Reject an unknown model name, matching the other handlers.
fn check_model(state: &AppState, model: Option<&str>) -> Result<(), ApiError> {
    if let Some(model) = model
        && !state.served_model_names().iter().any(|n| n == model)
    {
        return Err(ApiError::model_not_found(model.to_string()));
    }
    Ok(())
}

/// Build the `token_strs` vector when requested, via the tokenizer vocab.
fn token_strs(tokenizer: &vllm_text::tokenizer::DynTokenizer, ids: &[u32]) -> Vec<String> {
    // Unknown IDs yield "" — intentional; matches Python's convert_ids_to_tokens behaviour.
    ids.iter().map(|&id| tokenizer.id_to_token(id).unwrap_or_default()).collect()
}

pub async fn tokenize(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<TokenizeRequest>,
) -> Response {
    let request_id = tokenize_request_id(&headers);
    let tokenizer = state.chat.text().tokenizer();
    let max_model_len = state.chat.engine_core_client().max_model_len();

    let result = match body {
        // Completion form: encode the raw `prompt` string (no chat template).
        TokenizeRequest::Completion(req) => tokenize_completion(&state, &tokenizer, req),
        // Chat form: render `messages` through the template, then encode (see `tokenize_chat`).
        TokenizeRequest::Chat(req) => tokenize_chat(&state, &request_id, req).await,
    };

    match result {
        Ok((tokens, want_strs)) => {
            let token_strs = want_strs.then(|| token_strs(&tokenizer, &tokens));
            Json(TokenizeResponse {
                count: tokens.len(),
                max_model_len,
                tokens,
                token_strs,
            })
            .into_response()
        }
        Err(error) => error.into_response(),
    }
}

fn tokenize_completion(
    state: &AppState,
    tokenizer: &vllm_text::tokenizer::DynTokenizer,
    req: TokenizeCompletionRequest,
) -> Result<(Vec<u32>, bool), ApiError> {
    check_model(state, req.model.as_deref())?;
    let tokens = tokenizer
        .encode(&req.prompt, req.add_special_tokens)
        .map_err(|e| server_error!("tokenize failed: {}", e.to_report_string()))?;
    Ok((tokens, req.return_token_strs))
}

/// HTTP adapter for the chat-shaped `/tokenize` body.
///
/// Not [`vllm_chat::ChatLlm::tokenize_chat`]: this checks the model name and maps
/// errors to [`ApiError`]; the chat-crate method does render → finalize → encode.
async fn tokenize_chat(
    state: &AppState,
    request_id: &str,
    req: TokenizeChatRequest,
) -> Result<(Vec<u32>, bool), ApiError> {
    check_model(state, req.model.as_deref())?;
    let return_token_strs = req.return_token_strs;
    // `continue_final_message` / `add_generation_prompt` mutual exclusion is
    // enforced in `normalize_generation_prompt_mode` inside `into_chat_request`.
    let tokens = state
        .chat
        .tokenize_chat(req.into_chat_request(request_id.to_string())?)
        .await
        .map_err(|e| server_error!("tokenize failed: {}", e.to_report_string()))?;
    Ok((tokens, return_token_strs))
}

pub async fn detokenize(
    State(state): State<Arc<AppState>>,
    ValidatedJson(body): ValidatedJson<DetokenizeRequest>,
) -> Response {
    if let Err(error) = check_model(&state, body.model.as_deref()) {
        return error.into_response();
    }
    let tokenizer = state.chat.text().tokenizer();
    match tokenizer.decode(&body.tokens, /* skip_special_tokens = */ false) {
        Ok(prompt) => Json(DetokenizeResponse { prompt }).into_response(),
        Err(e) => server_error!("detokenize failed: {}", e.to_report_string()).into_response(),
    }
}

#[cfg(test)]
mod tests {
    use axum::http::{HeaderMap, HeaderValue};

    use super::tokenize_request_id;

    #[test]
    fn tokenize_request_id_prefers_x_request_id_header() {
        let mut headers = HeaderMap::new();
        headers.insert("X-Request-Id", HeaderValue::from_static("client-req-1"));
        assert_eq!(tokenize_request_id(&headers), "tokenize-client-req-1");
    }

    #[test]
    fn tokenize_request_id_generates_uuid_when_header_missing() {
        let headers = HeaderMap::new();
        let id = tokenize_request_id(&headers);
        assert!(id.starts_with("tokenize-"));
        assert_ne!(id, "tokenize-");
    }
}
