use std::collections::HashMap;

use openai_protocol::chat::{ChatMessage, MessageContent};
use openai_protocol::common::{ContentPart, StringOrArray, ToolChoice, ToolChoiceValue};
use uuid::Uuid;
use vllm_chat::{
    AssistantContentBlock, AssistantToolCall, ChatContent, ChatContentPart,
    ChatMessage as VllmChatMessage, ChatOptions, ChatRequest, ChatTool, ChatToolChoice,
    SamplingParams,
};

use super::types::ChatCompletionRequest;
use super::validate;
use crate::error::{ApiError, bail_invalid_request};
use crate::utils::convert_logit_bias;

/// Lowered chat request plus the public response metadata carried by every SSE chunk.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedRequest {
    /// Stable OpenAI-style response ID, also reused as the internal chat request ID.
    pub response_id: String,
    /// Public model ID echoed back to the client.
    pub response_model: String,
    /// Whether the caller asked for the final streamed usage chunk.
    pub include_usage: bool,
    /// Whether the caller requested output logprobs on chat choices.
    pub requested_logprobs: bool,
    /// Whether the caller requested top-level prompt logprobs.
    pub include_prompt_logprobs: bool,
    /// Lowered text-only chat request for `vllm-chat`.
    pub chat_request: ChatRequest,
    /// Last assistant-role message content to echo back when `echo=true`.
    pub echo: Option<String>,
}

/// Validate and lower one OpenAI chat completion request into the internal chat format.
pub fn prepare_chat_request(
    request: &ChatCompletionRequest,
    configured_model: &str,
) -> Result<PreparedRequest, ApiError> {
    validate::validate_request_compat(request, configured_model)?;

    let response_id = format!("chatcmpl-{}", Uuid::new_v4());
    let messages = request.messages.iter().map(convert_message).try_collect()?;

    let mut template_kwargs = HashMap::new();
    if let Some(kwargs) = &request.chat_template_kwargs {
        template_kwargs.extend(kwargs.clone());
    }

    let include_usage = (request.stream_options.as_ref())
        .and_then(|options| options.include_usage)
        .unwrap_or(false);
    let requested_logprobs = request.logprobs;
    let echo = request
        .echo
        .then(|| extract_last_assistant_content(&request.messages))
        .flatten();

    // Auto-enable prompt logprobs for non-streaming echo, matching Python vLLM's behavior.
    let top_logprobs = request.top_logprobs.unwrap_or(0);
    let prompt_logprobs = request
        .prompt_logprobs
        .or((request.echo && !request.stream).then_some(top_logprobs));
    let include_prompt_logprobs = prompt_logprobs.is_some();

    let chat_request = ChatRequest {
        request_id: response_id.clone(),
        messages,
        sampling_params: SamplingParams {
            temperature: request.temperature,
            top_p: request.top_p,
            top_k: request.top_k,
            seed: request.seed,
            max_tokens: request.max_completion_tokens,
            min_tokens: request.min_tokens,
            logprobs: request.logprobs.then_some(top_logprobs),
            prompt_logprobs,
            min_p: request.min_p,
            frequency_penalty: request.frequency_penalty,
            presence_penalty: request.presence_penalty,
            repetition_penalty: request.repetition_penalty,
            stop_token_ids: request.stop_token_ids.clone(),
            ignore_eos: request.ignore_eos,
            logit_bias: convert_logit_bias(request.logit_bias.clone())?,
            allowed_token_ids: request.allowed_token_ids.clone(),
            bad_words: request.bad_words.clone(),
            vllm_xargs: request.vllm_xargs.clone(),
        },
        chat_options: ChatOptions {
            add_generation_prompt: request.add_generation_prompt && !request.continue_final_message,
            continue_final_message: request.continue_final_message,
            template_kwargs,
        },
        tools: convert_tools(request.tools.as_deref())?,
        tool_choice: convert_tool_choice(request.tool_choice.as_ref())?,
        decode_options: vllm_text::output::TextDecodeOptions {
            skip_special_tokens: request.skip_special_tokens,
            include_stop_str_in_output: request.include_stop_str_in_output,
            stop_strings: request.stop.as_ref().map(|stop| match stop {
                StringOrArray::String(string) => vec![string.clone()],
                StringOrArray::Array(arr) => arr.clone(),
            }),
            min_tokens: request.min_tokens.unwrap_or(0),
        },
        intermediate: request.stream,
        priority: request.priority.unwrap_or(0),
    };

    Ok(PreparedRequest {
        response_id,
        response_model: configured_model.to_string(),
        include_usage,
        requested_logprobs,
        include_prompt_logprobs,
        chat_request,
        echo,
    })
}

/// Extract the text content of the last message if it has the assistant role.
fn extract_last_assistant_content(messages: &[ChatMessage]) -> Option<String> {
    let ChatMessage::Assistant { content, .. } = messages.last()? else {
        return None;
    };
    let text = match content.as_ref()? {
        MessageContent::Text(text) => text.clone(),
        MessageContent::Parts(parts) => parts
            .iter()
            .filter_map(|p| match p {
                ContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
    };
    (!text.is_empty()).then_some(text)
}

/// Lower one OpenAI chat message into the `vllm-chat` message shape.
fn convert_message(message: &ChatMessage) -> Result<VllmChatMessage, ApiError> {
    match message {
        ChatMessage::System { content, .. } => {
            Ok(VllmChatMessage::system(convert_content(content)?))
        }
        ChatMessage::User { content, .. } => Ok(VllmChatMessage::user(convert_content(content)?)),
        ChatMessage::Assistant {
            content,
            tool_calls,
            reasoning_content,
            name: _,
        } => {
            let mut blocks = Vec::new();
            if let Some(reasoning_content) = reasoning_content
                && !reasoning_content.is_empty()
            {
                blocks.push(AssistantContentBlock::Reasoning {
                    text: reasoning_content.clone(),
                });
            }
            if let Some(content) = content {
                blocks.extend(convert_assistant_text_blocks(content)?);
            }
            if let Some(tool_calls) = tool_calls {
                blocks.extend(convert_assistant_tool_calls(tool_calls)?);
            }
            if blocks.is_empty() {
                bail_invalid_request!(
                    "Assistant messages must contain text, reasoning content, or tool_calls."
                );
            }

            Ok(VllmChatMessage::assistant_blocks(blocks))
        }
        ChatMessage::Tool {
            content,
            tool_call_id,
        } => Ok(VllmChatMessage::tool_response(
            convert_content(content)?,
            tool_call_id.clone(),
        )),
        ChatMessage::Function { .. } => {
            bail_invalid_request!("Function messages are not supported.")
        }
        ChatMessage::Developer { .. } => {
            bail_invalid_request!("Developer messages are not supported.")
        }
    }
}

/// Convert the given OpenAI message content value into the internal format in `vllm-chat`.
fn convert_content(content: &MessageContent) -> Result<ChatContent, ApiError> {
    match content {
        MessageContent::Text(text) => Ok(ChatContent::Text(text.clone())),
        MessageContent::Parts(parts) => parts
            .iter()
            .map(|part| match part {
                ContentPart::Text { text } => Ok(ChatContentPart::text(text.clone())),
                _ => bail_invalid_request!("Only text content parts are supported."),
            })
            .try_collect()
            .map(ChatContent::Parts),
    }
}

/// Convert the given OpenAI assistant message content into the internal format in `vllm-chat`.
fn convert_assistant_text_blocks(
    content: &MessageContent,
) -> Result<Vec<AssistantContentBlock>, ApiError> {
    match content {
        MessageContent::Text(text) => Ok(vec![AssistantContentBlock::Text { text: text.clone() }]),
        MessageContent::Parts(parts) => parts
            .iter()
            .map(|part| match part {
                ContentPart::Text { text } => {
                    Ok(AssistantContentBlock::Text { text: text.clone() })
                }
                _ => bail_invalid_request!("Only text content parts are supported."),
            })
            .try_collect(),
    }
}

fn convert_assistant_tool_calls(
    tool_calls: &[openai_protocol::common::ToolCall],
) -> Result<Vec<AssistantContentBlock>, ApiError> {
    tool_calls
        .iter()
        .map(|tool_call| {
            if tool_call.tool_type != "function" {
                bail_invalid_request!("Only function tool calls are supported.");
            }

            Ok(AssistantContentBlock::ToolCall(AssistantToolCall {
                id: tool_call.id.clone(),
                name: tool_call.function.name.clone(),
                arguments: tool_call
                    .function
                    .arguments
                    .clone()
                    .unwrap_or_else(|| "{}".to_string()),
            }))
        })
        .collect()
}

fn convert_tools(
    tools: Option<&[openai_protocol::common::Tool]>,
) -> Result<Vec<ChatTool>, ApiError> {
    tools
        .unwrap_or(&[])
        .iter()
        .map(|tool| {
            if tool.tool_type != "function" {
                bail_invalid_request!("Only function tools are supported.");
            }
            Ok(ChatTool {
                name: tool.function.name.clone(),
                description: tool.function.description.clone(),
                parameters: tool.function.parameters.clone(),
                strict: tool.function.strict,
            })
        })
        .collect()
}

fn convert_tool_choice(tool_choice: Option<&ToolChoice>) -> Result<ChatToolChoice, ApiError> {
    match tool_choice {
        None | Some(ToolChoice::Value(ToolChoiceValue::Auto)) => Ok(ChatToolChoice::Auto),
        Some(ToolChoice::Value(ToolChoiceValue::None)) => Ok(ChatToolChoice::None),
        _ => bail_invalid_request!("tool_choice={:?} is not supported yet.", tool_choice),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use openai_protocol::chat::{ChatMessage, MessageContent};
    use openai_protocol::common::{ContentPart, Tool, ToolChoice};
    use serde_json::json;

    use super::prepare_chat_request;
    use crate::routes::openai::chat_completions::types::ChatCompletionRequest;

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
        request.add_generation_prompt = false;
        request.continue_final_message = true;
        request.skip_special_tokens = false;
        request.chat_template_kwargs = Some(HashMap::from([("foo".to_string(), json!("bar"))]));

        let mut prepared =
            prepare_chat_request(&request, "Qwen/Qwen1.5-0.5B-Chat").expect("request is valid");

        assert!(prepared.response_id.starts_with("chatcmpl-"));
        prepared.chat_request.request_id = "<placeholder>".to_string();
        expect_test::expect![[r#"
            ChatRequest {
                request_id: "<placeholder>",
                messages: [
                    User {
                        content: Parts(
                            [
                                Text {
                                    text: "hello",
                                },
                            ],
                        ),
                    },
                ],
                sampling_params: SamplingParams {
                    temperature: None,
                    top_p: None,
                    top_k: None,
                    seed: None,
                    max_tokens: None,
                    min_tokens: None,
                    logprobs: None,
                    prompt_logprobs: None,
                    min_p: None,
                    frequency_penalty: None,
                    presence_penalty: None,
                    repetition_penalty: None,
                    stop_token_ids: None,
                    ignore_eos: false,
                    logit_bias: None,
                    allowed_token_ids: None,
                    bad_words: None,
                    vllm_xargs: None,
                },
                chat_options: ChatOptions {
                    add_generation_prompt: false,
                    continue_final_message: true,
                    template_kwargs: {
                        "foo": String("bar"),
                    },
                },
                tools: [],
                tool_choice: Auto,
                decode_options: TextDecodeOptions {
                    skip_special_tokens: false,
                    include_stop_str_in_output: false,
                    stop_strings: None,
                    min_tokens: 0,
                },
                intermediate: true,
                priority: 0,
            }
        "#]]
        .assert_debug_eq(&prepared.chat_request);
    }

    #[test]
    fn prepare_chat_request_keeps_optional_sampling_fields_unset() {
        let mut prepared = prepare_chat_request(&base_request(), "Qwen/Qwen1.5-0.5B-Chat")
            .expect("request is valid");

        assert!(prepared.response_id.starts_with("chatcmpl-"));
        prepared.chat_request.request_id = "<placeholder>".to_string();
        expect_test::expect![[r#"
            ChatRequest {
                request_id: "<placeholder>",
                messages: [
                    User {
                        content: Text(
                            "hello",
                        ),
                    },
                ],
                sampling_params: SamplingParams {
                    temperature: None,
                    top_p: None,
                    top_k: None,
                    seed: None,
                    max_tokens: None,
                    min_tokens: None,
                    logprobs: None,
                    prompt_logprobs: None,
                    min_p: None,
                    frequency_penalty: None,
                    presence_penalty: None,
                    repetition_penalty: None,
                    stop_token_ids: None,
                    ignore_eos: false,
                    logit_bias: None,
                    allowed_token_ids: None,
                    bad_words: None,
                    vllm_xargs: None,
                },
                chat_options: ChatOptions {
                    add_generation_prompt: true,
                    continue_final_message: false,
                    template_kwargs: {},
                },
                tools: [],
                tool_choice: Auto,
                decode_options: TextDecodeOptions {
                    skip_special_tokens: true,
                    include_stop_str_in_output: false,
                    stop_strings: None,
                    min_tokens: 0,
                },
                intermediate: true,
                priority: 0,
            }
        "#]]
        .assert_debug_eq(&prepared.chat_request);
    }

    #[test]
    fn prepare_chat_request_preserves_sampling_passthrough_fields() {
        let request = ChatCompletionRequest {
            seed: Some(42),
            min_p: Some(0.2),
            frequency_penalty: Some(0.3),
            presence_penalty: Some(0.4),
            repetition_penalty: Some(1.1),
            ..base_request()
        };

        let mut prepared =
            prepare_chat_request(&request, "Qwen/Qwen1.5-0.5B-Chat").expect("request is valid");
        prepared.chat_request.request_id = "<placeholder>".to_string();
        expect_test::expect![[r#"
            ChatRequest {
                request_id: "<placeholder>",
                messages: [
                    User {
                        content: Text(
                            "hello",
                        ),
                    },
                ],
                sampling_params: SamplingParams {
                    temperature: None,
                    top_p: None,
                    top_k: None,
                    seed: Some(
                        42,
                    ),
                    max_tokens: None,
                    min_tokens: None,
                    logprobs: None,
                    prompt_logprobs: None,
                    min_p: Some(
                        0.2,
                    ),
                    frequency_penalty: Some(
                        0.3,
                    ),
                    presence_penalty: Some(
                        0.4,
                    ),
                    repetition_penalty: Some(
                        1.1,
                    ),
                    stop_token_ids: None,
                    ignore_eos: false,
                    logit_bias: None,
                    allowed_token_ids: None,
                    bad_words: None,
                    vllm_xargs: None,
                },
                chat_options: ChatOptions {
                    add_generation_prompt: true,
                    continue_final_message: false,
                    template_kwargs: {},
                },
                tools: [],
                tool_choice: Auto,
                decode_options: TextDecodeOptions {
                    skip_special_tokens: true,
                    include_stop_str_in_output: false,
                    stop_strings: None,
                    min_tokens: 0,
                },
                intermediate: true,
                priority: 0,
            }
        "#]]
        .assert_debug_eq(&prepared.chat_request);
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
    fn prepare_chat_request_accepts_assistant_reasoning_history() {
        let request = ChatCompletionRequest {
            messages: vec![ChatMessage::Assistant {
                content: Some(MessageContent::Text("answer".to_string())),
                name: None,
                tool_calls: None,
                reasoning_content: Some("inner".to_string()),
            }],
            ..base_request()
        };

        let mut prepared =
            prepare_chat_request(&request, "Qwen/Qwen1.5-0.5B-Chat").expect("request is valid");
        prepared.chat_request.request_id = "<placeholder>".to_string();
        expect_test::expect![[r#"
            ChatRequest {
                request_id: "<placeholder>",
                messages: [
                    Assistant {
                        content: [
                            Reasoning {
                                text: "inner",
                            },
                            Text {
                                text: "answer",
                            },
                        ],
                    },
                ],
                sampling_params: SamplingParams {
                    temperature: None,
                    top_p: None,
                    top_k: None,
                    seed: None,
                    max_tokens: None,
                    min_tokens: None,
                    logprobs: None,
                    prompt_logprobs: None,
                    min_p: None,
                    frequency_penalty: None,
                    presence_penalty: None,
                    repetition_penalty: None,
                    stop_token_ids: None,
                    ignore_eos: false,
                    logit_bias: None,
                    allowed_token_ids: None,
                    bad_words: None,
                    vllm_xargs: None,
                },
                chat_options: ChatOptions {
                    add_generation_prompt: true,
                    continue_final_message: false,
                    template_kwargs: {},
                },
                tools: [],
                tool_choice: Auto,
                decode_options: TextDecodeOptions {
                    skip_special_tokens: true,
                    include_stop_str_in_output: false,
                    stop_strings: None,
                    min_tokens: 0,
                },
                intermediate: true,
                priority: 0,
            }
        "#]]
        .assert_debug_eq(&prepared.chat_request);
    }

    #[test]
    fn prepare_chat_request_accepts_tools_and_tool_history() {
        let request = ChatCompletionRequest {
            messages: vec![
                ChatMessage::Assistant {
                    content: None,
                    name: None,
                    tool_calls: Some(vec![openai_protocol::common::ToolCall {
                        id: "call_1".to_string(),
                        tool_type: "function".to_string(),
                        function: openai_protocol::common::FunctionCallResponse {
                            name: "get_weather".to_string(),
                            arguments: Some(r#"{"city":"Paris"}"#.to_string()),
                        },
                    }]),
                    reasoning_content: None,
                },
                ChatMessage::Tool {
                    content: MessageContent::Text("Sunny".to_string()),
                    tool_call_id: "call_1".to_string(),
                },
            ],
            tools: Some(vec![Tool {
                tool_type: "function".to_string(),
                function: openai_protocol::common::Function {
                    name: "get_weather".to_string(),
                    description: Some("Get weather".to_string()),
                    parameters: json!({
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    }),
                    strict: None,
                },
            }]),
            tool_choice: Some(ToolChoice::Value(
                openai_protocol::common::ToolChoiceValue::None,
            )),
            ..base_request()
        };

        let mut prepared =
            prepare_chat_request(&request, "Qwen/Qwen1.5-0.5B-Chat").expect("request is valid");
        prepared.chat_request.request_id = "<placeholder>".to_string();
        expect_test::expect![[r#"
            ChatRequest {
                request_id: "<placeholder>",
                messages: [
                    Assistant {
                        content: [
                            ToolCall(
                                AssistantToolCall {
                                    id: "call_1",
                                    name: "get_weather",
                                    arguments: "{\"city\":\"Paris\"}",
                                },
                            ),
                        ],
                    },
                    ToolResponse {
                        content: Text(
                            "Sunny",
                        ),
                        tool_call_id: "call_1",
                    },
                ],
                sampling_params: SamplingParams {
                    temperature: None,
                    top_p: None,
                    top_k: None,
                    seed: None,
                    max_tokens: None,
                    min_tokens: None,
                    logprobs: None,
                    prompt_logprobs: None,
                    min_p: None,
                    frequency_penalty: None,
                    presence_penalty: None,
                    repetition_penalty: None,
                    stop_token_ids: None,
                    ignore_eos: false,
                    logit_bias: None,
                    allowed_token_ids: None,
                    bad_words: None,
                    vllm_xargs: None,
                },
                chat_options: ChatOptions {
                    add_generation_prompt: true,
                    continue_final_message: false,
                    template_kwargs: {},
                },
                tools: [
                    ChatTool {
                        name: "get_weather",
                        description: Some(
                            "Get weather",
                        ),
                        parameters: Object {
                            "type": String("object"),
                            "properties": Object {
                                "city": Object {
                                    "type": String("string"),
                                },
                            },
                        },
                        strict: None,
                    },
                ],
                tool_choice: None,
                decode_options: TextDecodeOptions {
                    skip_special_tokens: true,
                    include_stop_str_in_output: false,
                    stop_strings: None,
                    min_tokens: 0,
                },
                intermediate: true,
                priority: 0,
            }
        "#]]
        .assert_debug_eq(&prepared.chat_request);
    }

    #[test]
    fn prepare_chat_request_lowers_logprobs_fields() {
        let request = ChatCompletionRequest {
            stream: false,
            logprobs: true,
            prompt_logprobs: Some(2),
            ..base_request()
        };

        let prepared =
            prepare_chat_request(&request, "Qwen/Qwen1.5-0.5B-Chat").expect("request is valid");

        assert!(prepared.requested_logprobs);
        assert!(prepared.include_prompt_logprobs);
        assert_eq!(prepared.chat_request.sampling_params.logprobs, Some(0));
        assert_eq!(
            prepared.chat_request.sampling_params.prompt_logprobs,
            Some(2)
        );
    }

    #[test]
    fn prepare_chat_request_keeps_prompt_logprobs_independent_from_echo() {
        let request = ChatCompletionRequest {
            logprobs: true,
            top_logprobs: Some(3),
            echo: true,
            ..base_request()
        };

        let prepared =
            prepare_chat_request(&request, "Qwen/Qwen1.5-0.5B-Chat").expect("request is valid");

        assert_eq!(prepared.chat_request.sampling_params.logprobs, Some(3));
        assert_eq!(prepared.chat_request.sampling_params.prompt_logprobs, None);
        assert!(!prepared.include_prompt_logprobs);
    }
}
