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
use crate::routes::openai::utils::types::{ChatMessage, ContentPart, MessageContent};
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

fn resolve_max_tokenize_tokens(configured_limit: Option<i64>, max_model_len: u32) -> Option<usize> {
    match configured_limit {
        Some(limit) if limit <= 0 => None,
        Some(limit) => Some(limit as usize),
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

fn estimate_chat_tokens(
    tokenizer: &vllm_text::tokenizer::DynTokenizer,
    messages: &[ChatMessage],
) -> usize {
    let mut total = 0;
    for message in messages {
        match message {
            ChatMessage::System { content, .. }
            | ChatMessage::User { content, .. }
            | ChatMessage::Tool { content, .. }
            | ChatMessage::Developer { content, .. }
            | ChatMessage::Custom { content, .. } => {
                total += estimate_content_tokens(tokenizer, content);
            }
            ChatMessage::Assistant {
                content,
                tool_calls,
                reasoning,
                name: _,
            } => {
                if let Some(content) = content {
                    total += estimate_content_tokens(tokenizer, content);
                }
                if let Some(reasoning) = reasoning {
                    total += tokenizer.estimate_min_token(reasoning);
                }
                if let Some(tool_calls) = tool_calls {
                    for tool_call in tool_calls {
                        total += tokenizer.estimate_min_token(&tool_call.function.name);
                        if let Some(arguments) = &tool_call.function.arguments {
                            total += tokenizer.estimate_min_token(arguments);
                        }
                    }
                }
            }
            ChatMessage::Function { content, name } => {
                total += tokenizer.estimate_min_token(name);
                total += tokenizer.estimate_min_token(content);
            }
        }
    }
    total
}

fn estimate_content_tokens(
    tokenizer: &vllm_text::tokenizer::DynTokenizer,
    content: &MessageContent,
) -> usize {
    match content {
        MessageContent::Text(text) => tokenizer.estimate_min_token(text),
        MessageContent::Parts(parts) => {
            let mut sum = 0;
            for part in parts {
                match part {
                    ContentPart::Text { text } | ContentPart::Refusal { refusal: text } => {
                        sum += tokenizer.estimate_min_token(text);
                    }
                    ContentPart::ToolReference { name } => {
                        sum += tokenizer.estimate_min_token(name);
                    }
                    ContentPart::ImageUrl { .. }
                    | ContentPart::VideoUrl { .. }
                    | ContentPart::AudioUrl { .. }
                    | ContentPart::InputAudio { .. } => {}
                }
            }
            sum
        }
    }
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
        let estimated = tokenizer.estimate_min_token(&req.prompt);
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
    let tokenizer = state.chat.text().tokenizer();
    if let Some(limit) = max_tokenize_tokens {
        let estimated = estimate_chat_tokens(&tokenizer, &req.messages);
        check_max_tokenize_tokens(estimated, limit, "at least ")?;
    }
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
    use std::sync::Arc;

    use axum::http::{HeaderMap, HeaderValue};
    use vllm_tokenizer::Tokenizer;

    use super::{
        check_max_tokenize_tokens, estimate_chat_tokens, resolve_max_tokenize_tokens,
        tokenize_request_id,
    };
    use crate::error::ApiError;
    use crate::routes::openai::utils::types::{ChatMessage, ContentPart, MessageContent, ToolCall};

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

    #[test]
    fn estimate_chat_tokens_sums_message_text_and_tools() {
        struct MockTokenizer(usize);
        impl Tokenizer for MockTokenizer {
            fn encode(
                &self,
                _text: &str,
                _add_special_tokens: bool,
            ) -> vllm_tokenizer::Result<Vec<u32>> {
                Ok(Vec::new())
            }
            fn encode_ordinary(&self, _text: &str) -> vllm_tokenizer::Result<Vec<u32>> {
                Ok(Vec::new())
            }
            fn decode(
                &self,
                _tokens: &[u32],
                _skip_special_tokens: bool,
            ) -> vllm_tokenizer::Result<String> {
                Ok(String::new())
            }
            fn vocab_size(&self) -> usize {
                100
            }
            fn token_to_id(&self, _token: &str) -> Option<u32> {
                None
            }
            fn id_to_token(&self, _id: u32) -> Option<String> {
                None
            }
            fn estimate_min_token(&self, text: &str) -> usize {
                text.len().checked_div(self.0).unwrap_or(0)
            }
        }

        // Each token is at most 10 characters.
        let tokenizer: vllm_text::tokenizer::DynTokenizer = Arc::new(MockTokenizer(10));

        let messages = vec![
            ChatMessage::System {
                content: MessageContent::Text("0123456789".to_string()), // 10 chars -> 1 token
                name: None,
            },
            ChatMessage::User {
                content: MessageContent::Parts(vec![
                    ContentPart::Text {
                        text: "01234567890123456789".to_string(), // 20 chars -> 2 tokens
                    },
                    ContentPart::Refusal {
                        refusal: "0123456789".to_string(), // 10 chars -> 1 token
                    },
                    ContentPart::ToolReference {
                        name: "tool_name_".to_string(), // 10 chars -> 1 token
                    },
                ]),
                name: None,
            },
            ChatMessage::Assistant {
                content: Some(MessageContent::Text("0123456789".to_string())), // 10 chars -> 1 token
                name: None,
                reasoning: Some("0123456789".to_string()), // 10 chars -> 1 token
                tool_calls: Some(vec![ToolCall {
                    id: "call_1".to_string(),
                    tool_type: "function".to_string(),
                    function: crate::routes::openai::utils::types::FunctionCallResponse {
                        name: "func_name_".to_string(), // 10 chars -> 1 token
                        arguments: Some("01234567890123456789".to_string()), // 20 chars -> 2 tokens
                    },
                }]),
            },
            ChatMessage::Function {
                name: "func_name_".to_string(),    // 10 chars -> 1 token
                content: "0123456789".to_string(), // 10 chars -> 1 token
            },
        ];

        let estimated = estimate_chat_tokens(&tokenizer, &messages);
        // Total expected: 1 + 2 + 1 + 1 + 1 + 1 + 1 + 2 + 1 + 1 = 12 tokens
        assert_eq!(estimated, 12);
    }
}
