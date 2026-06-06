use std::mem::take;
use std::sync::Arc;

use axum::{Json, extract::State};
use serde::{Deserialize, Serialize};
use thiserror_ext::AsReport;
use vllm_text::Prompt;

use crate::error::ApiError;
use crate::routes::tokenize::TokenizeRequest::Completion;
use crate::state::AppState;

/// Request for tokenizing text or token IDs.
///
/// Supports the `completion` variant which tokenizes a prompt string
/// or a list of token IDs and optionally returns the string representation
/// of each token.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct TokenizeCompletionRequest {
    /// Name of the model to use for tokenization. If omitted, the default model
    /// is selected from `state.served_model_names()`.
    model: Option<String>,

    /// The prompt to tokenize: either a text string or pre-existing token IDs.
    prompt: Prompt,

    /// Whether to add special tokens (e.g., BOS/EOS) during encoding.
    /// Defaults to `true`.
    #[serde(default = "default_add_special_tokens")]
    add_special_tokens: bool,

    /// If `Some(true)`, the response will include `token_strs` with the
    /// string representation of each token ID.
    return_token_strs: Option<bool>,
}

const fn default_add_special_tokens() -> bool {
    true
}

/// Tokenization request, deserialized using `#[serde(untagged)]`.
///
/// Currently supports only the `completion` variant.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub(crate) enum TokenizeRequest {
    Completion(TokenizeCompletionRequest),
}

/// Response body returned by the tokenize endpoint.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct TokenizeResponse {
    /// Total number of tokens returned.
    count: usize,

    /// Maximum model context length.
    /// `state.config().max_model_len`.
    max_model_len: u32,

    /// The tokenized token IDs.
    tokens: Vec<u32>,

    /// Optional string representation of each token
    token_strs: Option<Vec<String>>,
}

/// Handle a tokenize request by routing to the appropriate variant handler.
pub async fn tokenize(
    State(state): State<Arc<AppState>>,
    Json(body): Json<TokenizeRequest>,
) -> Result<Json<TokenizeResponse>, ApiError> {
    match body {
        Completion(req) => completion_tokenize(state, req).await,
    }
}

async fn completion_tokenize(
    state: Arc<AppState>,
    mut req: TokenizeCompletionRequest,
) -> Result<Json<TokenizeResponse>, ApiError> {
    if let Some(model) = &req.model
        && !state.served_model_names().iter().any(|n| n == model)
    {
        return Err(ApiError::ModelNotFound {
            model: model.to_string(),
        });
    }

    let tokenizer = state.chat.text().tokenizer();
    let prompt_token_ids = match take(&mut req.prompt) {
        Prompt::Text(text) => match tokenizer.encode(&text, req.add_special_tokens) {
            Ok(token_ids) => token_ids,
            Err(error) => {
                return Err(ApiError::ServerError {
                    message: format!(
                        "failed to tokenize completion request: {}",
                        error.to_report_string()
                    ),
                });
            }
        },
        Prompt::TokenIds(token_ids) => token_ids,
    };

    let token_strs = if let Some(true) = req.return_token_strs {
        let strs = prompt_token_ids.iter().filter_map(|&id| tokenizer.id_to_token(id)).collect();
        Some(strs)
    } else {
        None
    };

    Ok(Json(TokenizeResponse {
        count: prompt_token_ids.len(),
        max_model_len: state.chat.text().max_model_len(),
        tokens: prompt_token_ids,
        token_strs,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenization_request_roundtrip() {
        let original = TokenizeCompletionRequest {
            model: Some("test-model".into()),
            prompt: Prompt::Text("hello world".into()),
            add_special_tokens: false,
            return_token_strs: Some(true),
        };

        let json = serde_json::to_string(&original).expect("serialize");
        let deserialized: TokenizeCompletionRequest =
            serde_json::from_str(&json).expect("deserialize");

        assert_eq!(deserialized.model, original.model);
        assert_eq!(deserialized.add_special_tokens, original.add_special_tokens);
        assert_eq!(deserialized.return_token_strs, original.return_token_strs);
        assert!(matches!(&deserialized.prompt, Prompt::Text(t) if t == "hello world"));
    }

    #[test]
    fn tokenization_request_default_add_special_tokens() {
        let json = r#"{"prompt":"hello"}"#;
        let req: TokenizeCompletionRequest = serde_json::from_str(json).expect("deserialize");
        assert!(req.add_special_tokens);
        assert!(req.model.is_none());
        assert!(req.return_token_strs.is_none());
    }
}
