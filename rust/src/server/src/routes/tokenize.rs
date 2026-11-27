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
use axum::extract::rejection::JsonRejection;
use axum::http::HeaderMap;

use crate::error::{ApiError, chat_submit_error, text_submit_error};
use crate::routes::tokenize::types::{
    DetokenizeRequest, DetokenizeResponse, TokenizeChatRequest, TokenizeCompletionRequest,
    TokenizeRequest, TokenizeResponse,
};
use crate::state::AppState;
use crate::utils::resolve_request_id;

/// Build the `token_strs` vector when requested, via the tokenizer vocab.
fn token_strs(tokenizer: &vllm_text::tokenizer::DynTokenizer, ids: &[u32]) -> Vec<String> {
    // Unknown IDs yield "" — intentional; matches Python's convert_ids_to_tokens behaviour.
    ids.iter().map(|&id| tokenizer.id_to_token(id).unwrap_or_default()).collect()
}

pub async fn tokenize(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Json<TokenizeRequest>, JsonRejection>,
) -> Result<Json<TokenizeResponse>, ApiError> {
    let Json(body) = body?;
    // Python names it `tokenize-<X-Request-Id or a random ID>`.
    let request_id = resolve_request_id(&headers, "tokenize-", None)?;
    let tokenizer = state.chat.text().tokenizer();
    let max_model_len = state.llm().info().max_model_len;

    let (tokens, want_strs) = match body {
        TokenizeRequest::Completion(req) => tokenize_completion(&state, &request_id, req)?,
        TokenizeRequest::Chat(req) => tokenize_chat(&state, &request_id, req).await?,
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
) -> Result<(Vec<u32>, bool), ApiError> {
    state.base_model_names().check(req.model.as_deref())?;
    let return_token_strs = req.return_token_strs;
    let tokens = state
        .chat
        .text()
        .request_processor()
        .tokenize(req.into_text_request(request_id.to_string()))
        .map_err(|error| text_submit_error("tokenize failed", error))?;
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
    Ok((tokens, return_token_strs))
}

pub async fn detokenize(
    State(state): State<Arc<AppState>>,
    body: Result<Json<DetokenizeRequest>, JsonRejection>,
) -> Result<Json<DetokenizeResponse>, ApiError> {
    let Json(body) = body?;
    state.base_model_names().check(body.model.as_deref())?;
    let tokenizer = state.chat.text().tokenizer();
    let prompt = tokenizer
        .decode(&body.tokens, /* skip_special_tokens = */ false)
        .map_err(|error| text_submit_error("detokenize failed", error.into()))?;
    Ok(Json(DetokenizeResponse { prompt }))
}
