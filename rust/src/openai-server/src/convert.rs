use std::collections::HashMap;

use openai_protocol::chat::{
    ChatCompletionRequest, ChatCompletionStreamResponse, ChatMessage, ChatMessageDelta,
    ChatStreamChoice, MessageContent,
};
use openai_protocol::common::{ContentPart, StringOrArray, ToolChoice, ToolChoiceValue};
use serde_json::Value;
use uuid::Uuid;
use vllm_chat::{
    ChatContent, ChatContentPart, ChatMessage as VllmChatMessage, ChatOptions, ChatRequest,
    ChatRole, UserSamplingParams,
};
use vllm_engine_core_client::protocol::{FinishReason, StopReason};

use crate::error::ApiError;

#[derive(Debug, Clone, PartialEq)]
pub struct PreparedRequest {
    pub response_id: String,
    pub response_model: String,
    pub chat_request: ChatRequest,
}

pub fn prepare_chat_request(
    request: &ChatCompletionRequest,
    configured_model: &str,
) -> Result<PreparedRequest, ApiError> {
    validate_request_compat(request, configured_model)?;

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
            temperature: request.temperature.unwrap_or(1.0),
            top_p: request.top_p.unwrap_or(1.0),
            top_k: request.top_k.unwrap_or(0),
            max_tokens: request.max_completion_tokens.unwrap_or(16),
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

pub fn start_chunk(
    response_id: &str,
    response_model: &str,
    created: u64,
) -> ChatCompletionStreamResponse {
    ChatCompletionStreamResponse {
        id: response_id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: response_model.to_string(),
        system_fingerprint: None,
        choices: vec![ChatStreamChoice {
            index: 0,
            delta: ChatMessageDelta {
                role: Some("assistant".to_string()),
                content: None,
                tool_calls: None,
                reasoning_content: None,
            },
            logprobs: None,
            finish_reason: None,
            matched_stop: None,
        }],
        usage: None,
    }
}

pub fn text_chunk(
    response_id: &str,
    response_model: &str,
    created: u64,
    delta: String,
) -> ChatCompletionStreamResponse {
    ChatCompletionStreamResponse {
        id: response_id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: response_model.to_string(),
        system_fingerprint: None,
        choices: vec![ChatStreamChoice {
            index: 0,
            delta: ChatMessageDelta {
                role: None,
                content: Some(delta),
                tool_calls: None,
                reasoning_content: None,
            },
            logprobs: None,
            finish_reason: None,
            matched_stop: None,
        }],
        usage: None,
    }
}

pub fn final_chunk(
    response_id: &str,
    response_model: &str,
    created: u64,
    finish_reason: FinishReason,
    stop_reason: Option<StopReason>,
) -> Result<ChatCompletionStreamResponse, ApiError> {
    let finish_reason = match finish_reason {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
        FinishReason::Repetition => "stop",
        FinishReason::Abort | FinishReason::Error => {
            return Err(ApiError::server_error(
                "stream terminated without a valid OpenAI finish reason",
            ));
        }
    };

    let matched_stop = stop_reason.map(stop_reason_to_json);

    Ok(
        ChatCompletionStreamResponse::builder(response_id.to_string(), response_model.to_string())
            .created(created)
            .add_choice_finish_reason(0, finish_reason, matched_stop)
            .build(),
    )
}

fn validate_request_compat(
    request: &ChatCompletionRequest,
    configured_model: &str,
) -> Result<(), ApiError> {
    if request.model != configured_model {
        return Err(ApiError::model_not_found(request.model.clone()));
    }

    if !request.stream {
        return Err(ApiError::invalid_request_param(
            "Only stream=true is supported.",
            "stream",
        ));
    }

    if request.n.unwrap_or(1) > 1 {
        return Err(ApiError::invalid_request_param(
            "Only n=1 is supported.",
            "n",
        ));
    }

    if has_non_empty_stop(request.stop.as_ref()) {
        return Err(ApiError::invalid_request_param(
            "Stop strings are not supported by the minimal Rust frontend.",
            "stop",
        ));
    }

    reject_non_zero(
        request.frequency_penalty,
        "frequency_penalty",
        "Only frequency_penalty=0 is supported.",
    )?;
    reject_non_zero(
        request.presence_penalty,
        "presence_penalty",
        "Only presence_penalty=0 is supported.",
    )?;
    reject_non_zero(
        request.verbosity.map(|value| value as f32),
        "verbosity",
        "Verbosity control is not supported.",
    )?;

    if request.logprobs {
        return Err(ApiError::invalid_request_param(
            "logprobs are not supported.",
            "logprobs",
        ));
    }

    if request.top_logprobs.unwrap_or(0) > 0 {
        return Err(ApiError::invalid_request_param(
            "top_logprobs are not supported.",
            "top_logprobs",
        ));
    }

    if request.stream_options.is_some() {
        return Err(ApiError::invalid_request_param(
            "stream_options are not supported.",
            "stream_options",
        ));
    }

    if request.response_format.is_some() {
        return Err(ApiError::invalid_request_param(
            "response_format is not supported.",
            "response_format",
        ));
    }

    if request
        .tools
        .as_ref()
        .is_some_and(|tools| !tools.is_empty())
    {
        return Err(ApiError::invalid_request_param(
            "Tool calling is not supported.",
            "tools",
        ));
    }

    if let Some(tool_choice) = &request.tool_choice
        && !matches!(tool_choice, ToolChoice::Value(ToolChoiceValue::None))
    {
        return Err(ApiError::invalid_request_param(
            "tool_choice is not supported.",
            "tool_choice",
        ));
    }

    reject_non_default(
        request.logit_bias.as_ref(),
        "logit_bias",
        "logit_bias is not supported.",
    )?;
    reject_non_default(
        request.metadata.as_ref(),
        "metadata",
        "metadata is not supported.",
    )?;
    reject_non_default(
        request.modalities.as_ref(),
        "modalities",
        "modalities are not supported.",
    )?;
    reject_non_default(
        request.prompt_cache_key.as_ref(),
        "prompt_cache_key",
        "prompt_cache_key is not supported.",
    )?;
    reject_non_default(
        request.reasoning_effort.as_ref(),
        "reasoning_effort",
        "reasoning controls are not supported.",
    )?;
    reject_non_default(
        request.safety_identifier.as_ref(),
        "safety_identifier",
        "safety_identifier is not supported.",
    )?;
    reject_non_default(
        request.service_tier.as_ref(),
        "service_tier",
        "service_tier is not supported.",
    )?;
    #[expect(deprecated, reason = "OpenAI protocol still exposes legacy seed")]
    reject_non_default(request.seed.as_ref(), "seed", "seed is not supported.")?;
    reject_non_default(request.min_p.as_ref(), "min_p", "min_p is not supported.")?;
    reject_non_default(
        request.repetition_penalty.as_ref(),
        "repetition_penalty",
        "repetition_penalty is not supported.",
    )?;
    reject_non_default(
        request.regex.as_ref(),
        "regex",
        "regex constraints are not supported.",
    )?;
    reject_non_default(
        request.ebnf.as_ref(),
        "ebnf",
        "ebnf constraints are not supported.",
    )?;
    reject_non_default(
        request.lora_path.as_ref(),
        "lora_path",
        "lora_path is not supported.",
    )?;
    reject_non_default(
        request.session_params.as_ref(),
        "session_params",
        "session_params are not supported.",
    )?;
    reject_non_default(
        request
            .chat_template_kwargs
            .as_ref()
            .and_then(|kwargs| kwargs.get("reasoning_effort")),
        "chat_template_kwargs",
        "reasoning controls are not supported.",
    )?;

    if request.parallel_tool_calls.is_some_and(|value| value)
        && request
            .tools
            .as_ref()
            .is_some_and(|tools| !tools.is_empty())
    {
        return Err(ApiError::invalid_request_param(
            "parallel_tool_calls is not supported.",
            "parallel_tool_calls",
        ));
    }

    if request.no_stop_trim {
        return Err(ApiError::invalid_request_param(
            "no_stop_trim is not supported.",
            "no_stop_trim",
        ));
    }

    if request.return_hidden_states {
        return Err(ApiError::invalid_request_param(
            "return_hidden_states is not supported.",
            "return_hidden_states",
        ));
    }

    if request.sampling_seed.is_some() {
        return Err(ApiError::invalid_request_param(
            "sampling_seed is not supported.",
            "sampling_seed",
        ));
    }

    Ok(())
}

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

fn has_non_empty_stop(stop: Option<&StringOrArray>) -> bool {
    match stop {
        None => false,
        Some(StringOrArray::String(value)) => !value.is_empty(),
        Some(StringOrArray::Array(values)) => values.iter().any(|value| !value.is_empty()),
    }
}

fn reject_non_zero(value: Option<f32>, param: &str, message: &str) -> Result<(), ApiError> {
    if value.unwrap_or(0.0) != 0.0 {
        return Err(ApiError::invalid_request_param(message, param));
    }
    Ok(())
}

fn reject_non_default<T>(value: Option<&T>, param: &str, message: &str) -> Result<(), ApiError> {
    if value.is_some() {
        return Err(ApiError::invalid_request_param(message, param));
    }
    Ok(())
}

fn stop_reason_to_json(stop_reason: StopReason) -> Value {
    match stop_reason {
        StopReason::Text(text) => Value::String(text),
        StopReason::TokenId(token_id) => Value::Number(serde_json::Number::from(token_id)),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use openai_protocol::chat::{ChatCompletionRequest, ChatMessage, MessageContent};
    use openai_protocol::common::{ContentPart, ResponseFormat, StreamOptions, Tool, ToolChoice};
    use serde_json::json;
    use vllm_engine_core_client::protocol::{FinishReason, StopReason};

    use super::{final_chunk, prepare_chat_request, text_chunk};

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

        assert_eq!(prepared.response_id.starts_with("chatcmpl-"), true);
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

    #[test]
    fn prepare_chat_request_rejects_non_empty_stop() {
        let request = ChatCompletionRequest {
            stop: Some(openai_protocol::common::StringOrArray::String(
                "stop".to_string(),
            )),
            ..base_request()
        };

        assert!(prepare_chat_request(&request, "Qwen/Qwen1.5-0.5B-Chat").is_err());
    }

    #[test]
    fn prepare_chat_request_rejects_non_zero_penalties_and_tools() {
        let request = ChatCompletionRequest {
            frequency_penalty: Some(0.5),
            ..base_request()
        };
        assert!(prepare_chat_request(&request, "Qwen/Qwen1.5-0.5B-Chat").is_err());

        let request = ChatCompletionRequest {
            tools: Some(vec![Tool {
                tool_type: "function".to_string(),
                function: openai_protocol::common::Function {
                    name: "tool".to_string(),
                    description: None,
                    parameters: json!({}),
                    strict: None,
                },
            }]),
            ..base_request()
        };
        assert!(prepare_chat_request(&request, "Qwen/Qwen1.5-0.5B-Chat").is_err());
    }

    #[test]
    fn prepare_chat_request_rejects_response_format_and_stream_options() {
        let request = ChatCompletionRequest {
            response_format: Some(ResponseFormat::Text),
            ..base_request()
        };
        assert!(prepare_chat_request(&request, "Qwen/Qwen1.5-0.5B-Chat").is_err());

        let request = ChatCompletionRequest {
            stream_options: Some(StreamOptions {
                include_usage: Some(true),
            }),
            ..base_request()
        };
        assert!(prepare_chat_request(&request, "Qwen/Qwen1.5-0.5B-Chat").is_err());
    }

    #[test]
    fn prepare_chat_request_accepts_noop_tool_choice_none() {
        let request = ChatCompletionRequest {
            tool_choice: Some(ToolChoice::Value(
                openai_protocol::common::ToolChoiceValue::None,
            )),
            ..base_request()
        };

        prepare_chat_request(&request, "Qwen/Qwen1.5-0.5B-Chat").expect("tool_choice=none is ok");
    }

    #[test]
    fn text_chunk_uses_content_only_delta() {
        let chunk = text_chunk("chatcmpl-1", "model", 1, "hello".to_string());
        assert_eq!(chunk.choices[0].delta.role, None);
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("hello"));
    }

    #[test]
    fn final_chunk_maps_stop_finish_reason_and_matched_stop() {
        let chunk = final_chunk(
            "chatcmpl-1",
            "model",
            1,
            FinishReason::Stop,
            Some(StopReason::Text("stop".to_string())),
        )
        .expect("finish reason is valid");

        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("stop"));
        assert_eq!(chunk.choices[0].matched_stop, Some(json!("stop")));
    }

    #[test]
    fn final_chunk_maps_length_finish_reason() {
        let chunk = final_chunk(
            "chatcmpl-1",
            "model",
            1,
            FinishReason::Length,
            Some(StopReason::TokenId(42)),
        )
        .expect("finish reason is valid");

        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("length"));
        assert_eq!(chunk.choices[0].matched_stop, Some(json!(42)));
    }

    #[test]
    fn final_chunk_rejects_abort_and_error_finish_reasons() {
        assert!(final_chunk("chatcmpl-1", "model", 1, FinishReason::Abort, None).is_err());
        assert!(final_chunk("chatcmpl-1", "model", 1, FinishReason::Error, None).is_err());
    }
}
