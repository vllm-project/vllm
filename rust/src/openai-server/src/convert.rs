use std::collections::HashMap;

use openai_protocol::chat::{ChatCompletionRequest, ChatMessage, MessageContent};
use openai_protocol::common::ContentPart;
use uuid::Uuid;
use vllm_chat::{
    ChatContent, ChatContentPart, ChatMessage as VllmChatMessage, ChatOptions, ChatRequest,
    ChatRole, UserSamplingParams,
};

use crate::error::ApiError;

mod validate;

/// Lowered chat request plus the public response metadata carried by every SSE chunk.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedRequest {
    /// Stable OpenAI-style response ID, also reused as the internal chat request ID.
    pub response_id: String,
    /// Public model ID echoed back to the client.
    pub response_model: String,
    /// Lowered text-only chat request for `vllm-chat`.
    pub chat_request: ChatRequest,
}

/// Validate and lower one OpenAI chat completion request into the internal chat format.
pub fn prepare_chat_request(
    request: &ChatCompletionRequest,
    configured_model: &str,
) -> Result<PreparedRequest, ApiError> {
    validate::validate_request_compat(request, configured_model)?;

    let response_id = format!("chatcmpl-{}", Uuid::new_v4());
    let messages = request
        .messages
        .iter()
        .map(convert_message)
        .collect::<Result<Vec<_>, _>>()?;

    let mut template_kwargs = HashMap::new();
    if let Some(kwargs) = &request.chat_template_kwargs {
        template_kwargs.extend(kwargs.clone());
    }

    let chat_request = ChatRequest {
        request_id: response_id.clone(),
        messages,
        sampling_params: UserSamplingParams {
            // TODO: single source of truth for default sampling parameters
            temperature: request.temperature.unwrap_or(1.0),
            top_p: request.top_p.unwrap_or(1.0),
            top_k: request.top_k.unwrap_or(0),
            max_tokens: request.max_completion_tokens.unwrap_or(65536),
            min_tokens: request.min_tokens.unwrap_or(0),
            include_stop_str_in_output: false,
            stop_token_ids: request.stop_token_ids.clone().unwrap_or_default(),
            ignore_eos: request.ignore_eos,
            skip_special_tokens: request.skip_special_tokens,
        },
        chat_options: ChatOptions {
            add_generation_prompt: !request.continue_final_message,
            continue_final_message: request.continue_final_message,
            template_kwargs,
        },
    };

    Ok(PreparedRequest {
        response_id,
        response_model: configured_model.to_string(),
        chat_request,
    })
}

/// Lower one OpenAI chat message into the text-only `vllm-chat` message shape.
// TODO: use `TryFrom`?
fn convert_message(message: &ChatMessage) -> Result<VllmChatMessage, ApiError> {
    match message {
        ChatMessage::System { content, .. } => Ok(VllmChatMessage::new(
            ChatRole::System,
            convert_content(content)?,
        )),
        ChatMessage::User { content, .. } => Ok(VllmChatMessage::new(
            ChatRole::User,
            convert_content(content)?,
        )),
        ChatMessage::Assistant {
            content,
            tool_calls,
            reasoning_content,
            ..
        } => {
            if tool_calls.as_ref().is_some_and(|calls| !calls.is_empty()) {
                return Err(ApiError::invalid_request(
                    "Assistant tool_calls are not supported.",
                ));
            }
            if reasoning_content.is_some() {
                return Err(ApiError::invalid_request(
                    "Assistant reasoning content is not supported.",
                ));
            }

            let Some(content) = content else {
                return Err(ApiError::invalid_request(
                    "Assistant messages must contain text content.",
                ));
            };

            Ok(VllmChatMessage::new(
                ChatRole::Assistant,
                convert_content(content)?,
            ))
        }
        ChatMessage::Tool { .. } => Err(ApiError::invalid_request(
            "Tool messages are not supported.",
        )),
        ChatMessage::Function { .. } => Err(ApiError::invalid_request(
            "Function messages are not supported.",
        )),
        ChatMessage::Developer { .. } => Err(ApiError::invalid_request(
            "Developer messages are not supported.",
        )),
    }
}

/// Convert one OpenAI message content value into the supported text-only internal format.
fn convert_content(content: &MessageContent) -> Result<ChatContent, ApiError> {
    match content {
        MessageContent::Text(text) => Ok(ChatContent::Text(text.clone())),
        MessageContent::Parts(parts) => parts
            .iter()
            .map(|part| match part {
                ContentPart::Text { text } => Ok(ChatContentPart::text(text.clone())),
                _ => Err(ApiError::invalid_request(
                    "Only text content parts are supported.",
                )),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(ChatContent::Parts),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use openai_protocol::chat::{ChatCompletionRequest, ChatMessage, MessageContent};
    use openai_protocol::common::ContentPart;
    use serde_json::json;

    use super::prepare_chat_request;

    fn base_request() -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "Qwen/Qwen1.5-0.5B-Chat".to_string(),
            messages: vec![ChatMessage::User {
                content: MessageContent::Text("hello".to_string()),
                name: None,
            }],
            stream: true,
            ..Default::default()
        }
    }

    #[test]
    fn prepare_chat_request_maps_text_parts() {
        let mut request = base_request();
        request.messages = vec![ChatMessage::User {
            content: MessageContent::Parts(vec![ContentPart::Text {
                text: "hello".to_string(),
            }]),
            name: None,
        }];
        request.continue_final_message = true;
        request.skip_special_tokens = false;
        request.chat_template_kwargs = Some(HashMap::from([("foo".to_string(), json!("bar"))]));

        let prepared =
            prepare_chat_request(&request, "Qwen/Qwen1.5-0.5B-Chat").expect("request is valid");

        assert!(prepared.response_id.starts_with("chatcmpl-"));
        assert!(!prepared.chat_request.chat_options.add_generation_prompt);
        assert!(prepared.chat_request.chat_options.continue_final_message);
        assert!(!prepared.chat_request.sampling_params.skip_special_tokens);
        assert_eq!(
            prepared.chat_request.chat_options.template_kwargs["foo"],
            json!("bar")
        );
    }

    #[test]
    fn prepare_chat_request_rejects_unsupported_roles() {
        let request = ChatCompletionRequest {
            messages: vec![ChatMessage::Developer {
                content: MessageContent::Text("hello".to_string()),
                tools: None,
                name: None,
            }],
            ..base_request()
        };

        assert!(prepare_chat_request(&request, "Qwen/Qwen1.5-0.5B-Chat").is_err());
    }

    #[test]
    fn prepare_chat_request_rejects_non_text_content_parts() {
        let request = ChatCompletionRequest {
            messages: vec![ChatMessage::User {
                content: MessageContent::Parts(vec![ContentPart::ImageUrl {
                    image_url: openai_protocol::common::ImageUrl {
                        url: "https://example.com/image.png".to_string(),
                        detail: None,
                    },
                }]),
                name: None,
            }],
            ..base_request()
        };

        assert!(prepare_chat_request(&request, "Qwen/Qwen1.5-0.5B-Chat").is_err());
    }
}
