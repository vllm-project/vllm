use std::sync::Arc;

use axum::{Json, extract::State};
use serde::{Deserialize, Serialize};
use validator::Validate;

use crate::{
    error::ApiError,
    routes::{openai::utils::validated_json::ValidatedJson, tokenize::TokenizeRequest::Completion},
    state::AppState,
};

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, Validate)]
pub(crate) struct TokenizeCompletionRequest {
    model: Option<String>,
    prompt: String,
    #[serde(default = "default_add_special_tokens")]
    add_special_tokens: bool,
    return_token_strs: Option<bool>,
}

const fn default_add_special_tokens() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub(crate) enum TokenizeRequest {
    Completion(TokenizeCompletionRequest),
}

#[derive(Serialize)]
pub(crate) struct TokenizeResponse {
    count: u64,
    max_model_len: u64,
    tokens: Vec<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    token_strs: Option<Vec<String>>,
}

/// Tokenize request
pub async fn tokenize(
    State(state): State<Arc<AppState>>,
    ValidatedJson(body): ValidatedJson<TokenizeRequest>,
) -> Result<Json<TokenizeResponse>, ApiError> {
    match body {
        Completion(req) => completion_tokenize(state, req).await,
    }
}

async fn completion_tokenize(
    state: Arc<AppState>,
    req: TokenizeCompletionRequest,
) -> Result<Json<TokenizeResponse>, ApiError> {
    if let Some(model) = &req.model
        && !state.served_model_names().iter().any(|n| n == model)
    {
        return Err(ApiError::ModelNotFound {
            model: model.to_string(),
        });
    }


    todo!()
}
