use openai_protocol::common::{ToolChoice, ToolChoiceValue};

use super::types::ChatCompletionRequest;
use crate::error::{ApiError, bail_invalid_request};

/// Enforce the minimal compatibility contract for the Rust OpenAI server.
pub(super) fn validate_request_compat(
    request: &ChatCompletionRequest,
    configured_model: &str,
) -> Result<(), ApiError> {
    if request.model != configured_model {
        return Err(ApiError::model_not_found(request.model.clone()));
    }

    if request.stream_options.is_some() && !request.stream {
        bail_invalid_request!(
            param = "stream_options",
            "stream_options are only supported when stream=true."
        );
    }

    if request.n.unwrap_or(1) > 1 {
        bail_invalid_request!(param = "n", "Only n=1 is supported.");
    }

    if request.top_logprobs.is_some() && !request.logprobs {
        bail_invalid_request!(
            param = "top_logprobs",
            "top_logprobs can only be used when logprobs=true."
        );
    }

    if let Some(prompt_logprobs) = request.prompt_logprobs {
        if prompt_logprobs < 0 && prompt_logprobs != -1 {
            bail_invalid_request!(
                param = "prompt_logprobs",
                "prompt_logprobs must be a non-negative value or -1."
            );
        }

        if request.stream && (prompt_logprobs > 0 || prompt_logprobs == -1) {
            bail_invalid_request!(
                param = "prompt_logprobs",
                "prompt_logprobs are not available when stream=true."
            );
        }
    }

    if let Some(tools) = request.tools.as_ref() {
        for tool in tools {
            if tool.tool_type != "function" {
                bail_invalid_request!(param = "tools", "Only function tools are supported.");
            }
        }
    }

    if let Some(tool_choice) = &request.tool_choice {
        match tool_choice {
            ToolChoice::Value(ToolChoiceValue::Auto | ToolChoiceValue::None) => {}
            ToolChoice::Value(ToolChoiceValue::Required) => {
                bail_invalid_request!(
                    param = "tool_choice",
                    "tool_choice=required is not supported yet."
                );
            }
            ToolChoice::Function { .. } => {
                bail_invalid_request!(
                    param = "tool_choice",
                    "Named function tool_choice is not supported yet."
                );
            }
            ToolChoice::AllowedTools { .. } => {
                bail_invalid_request!(
                    param = "tool_choice",
                    "allowed_tools tool_choice is not supported yet."
                );
            }
        }
    }

    reject_non_default(
        request.reasoning_effort.as_ref(),
        "reasoning_effort",
        "reasoning controls are not supported.",
    )?;
    reject_non_default(
        request
            .chat_template_kwargs
            .as_ref()
            .and_then(|kwargs| kwargs.get("reasoning_effort")),
        "chat_template_kwargs",
        "reasoning controls are not supported.",
    )?;

    if request.use_beam_search {
        bail_invalid_request!(
            param = "use_beam_search",
            "use_beam_search is not supported."
        );
    }

    // ---- Reject parameters that are accepted for deserialization but not yet implemented ----

    if request.parallel_tool_calls.is_some() {
        bail_invalid_request!(
            param = "parallel_tool_calls",
            "parallel_tool_calls is not supported."
        );
    }

    reject_non_default(
        request.length_penalty.as_ref(),
        "length_penalty",
        "length_penalty is not supported.",
    )?;
    if !request.spaces_between_special_tokens {
        bail_invalid_request!(
            param = "spaces_between_special_tokens",
            "spaces_between_special_tokens is not supported."
        );
    }
    reject_non_default(
        request.truncate_prompt_tokens.as_ref(),
        "truncate_prompt_tokens",
        "truncate_prompt_tokens is not supported.",
    )?;
    reject_non_default(
        request.thinking_token_budget.as_ref(),
        "thinking_token_budget",
        "thinking_token_budget is not supported.",
    )?;
    if !request.include_reasoning {
        bail_invalid_request!(
            param = "include_reasoning",
            "include_reasoning is not supported."
        );
    }
    reject_non_default(
        request.media_io_kwargs.as_ref(),
        "media_io_kwargs",
        "media_io_kwargs is not supported.",
    )?;
    reject_non_default(
        request.mm_processor_kwargs.as_ref(),
        "mm_processor_kwargs",
        "mm_processor_kwargs is not supported.",
    )?;
    reject_non_default(
        request.repetition_detection.as_ref(),
        "repetition_detection",
        "repetition_detection is not supported.",
    )?;

    if let Some(options) = &request.stream_options
        && options.continuous_usage_stats.is_some()
    {
        bail_invalid_request!(
            param = "stream_options",
            "continuous_usage_stats is not supported."
        );
    }

    Ok(())
}

/// Reject one option unless it is entirely absent.
fn reject_non_default<T>(
    value: Option<&T>,
    param: &'static str,
    message: &str,
) -> Result<(), ApiError> {
    if value.is_some() {
        bail_invalid_request!(param = param, "{}", message);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use openai_protocol::chat::{ChatMessage, MessageContent};
    use openai_protocol::common::{FunctionChoice, Tool, ToolChoice, ToolReference};
    use serde_json::json;

    use super::validate_request_compat;
    use crate::routes::openai::chat_completions::types::ChatCompletionRequest;
    use crate::routes::openai::utils::structured_outputs::ResponseFormat;

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
    fn validate_request_compat_accepts_stop() {
        let request = ChatCompletionRequest {
            stop: Some(openai_protocol::common::StringOrArray::String(
                "stop".to_string(),
            )),
            ..base_request()
        };

        validate_request_compat(&request, "Qwen/Qwen1.5-0.5B-Chat")
            .expect("stop strings should be accepted");
    }

    #[test]
    fn validate_request_compat_accepts_non_zero_penalties_and_function_tools() {
        let request = ChatCompletionRequest {
            frequency_penalty: Some(0.5),
            presence_penalty: Some(0.25),
            min_p: Some(0.2),
            repetition_penalty: Some(1.1),
            seed: Some(7),
            ..base_request()
        };
        validate_request_compat(&request, "Qwen/Qwen1.5-0.5B-Chat")
            .expect("sampling fields should be accepted");

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
        validate_request_compat(&request, "Qwen/Qwen1.5-0.5B-Chat")
            .expect("function tools should be accepted");
    }

    #[test]
    fn validate_request_compat_accepts_output_logprobs() {
        let request = ChatCompletionRequest {
            logprobs: true,
            ..base_request()
        };
        validate_request_compat(&request, "Qwen/Qwen1.5-0.5B-Chat")
            .expect("logprobs should be accepted");
    }

    #[test]
    fn validate_request_compat_rejects_top_logprobs_without_logprobs() {
        let request = ChatCompletionRequest {
            top_logprobs: Some(0),
            ..base_request()
        };
        assert!(validate_request_compat(&request, "Qwen/Qwen1.5-0.5B-Chat").is_err());
    }

    #[test]
    fn validate_request_compat_rejects_streaming_prompt_logprobs_requests() {
        let request = ChatCompletionRequest {
            prompt_logprobs: Some(1),
            ..base_request()
        };
        assert!(validate_request_compat(&request, "Qwen/Qwen1.5-0.5B-Chat").is_err());

        let request = ChatCompletionRequest {
            prompt_logprobs: Some(-1),
            ..base_request()
        };
        assert!(validate_request_compat(&request, "Qwen/Qwen1.5-0.5B-Chat").is_err());
    }

    #[test]
    fn validate_request_compat_rejects_invalid_prompt_logprobs_value() {
        let request = ChatCompletionRequest {
            stream: false,
            prompt_logprobs: Some(-2),
            ..base_request()
        };
        assert!(validate_request_compat(&request, "Qwen/Qwen1.5-0.5B-Chat").is_err());
    }

    #[test]
    fn validate_request_compat_accepts_response_format() {
        let request = ChatCompletionRequest {
            response_format: Some(ResponseFormat::Text),
            ..base_request()
        };
        validate_request_compat(&request, "Qwen/Qwen1.5-0.5B-Chat")
            .expect("response_format=text should be accepted");

        let request = ChatCompletionRequest {
            response_format: Some(ResponseFormat::JsonObject),
            ..base_request()
        };
        validate_request_compat(&request, "Qwen/Qwen1.5-0.5B-Chat")
            .expect("response_format=json_object should be accepted");
    }

    #[test]
    fn validate_request_compat_accepts_noop_tool_choice_none() {
        let request = ChatCompletionRequest {
            tool_choice: Some(ToolChoice::Value(
                openai_protocol::common::ToolChoiceValue::None,
            )),
            ..base_request()
        };

        validate_request_compat(&request, "Qwen/Qwen1.5-0.5B-Chat")
            .expect("tool_choice=none is ok");
    }

    #[test]
    fn validate_request_compat_rejects_required_and_named_tool_choices() {
        let required = ChatCompletionRequest {
            tool_choice: Some(ToolChoice::Value(
                openai_protocol::common::ToolChoiceValue::Required,
            )),
            ..base_request()
        };
        assert!(validate_request_compat(&required, "Qwen/Qwen1.5-0.5B-Chat").is_err());

        let named = ChatCompletionRequest {
            tool_choice: Some(ToolChoice::Function {
                tool_type: "function".to_string(),
                function: FunctionChoice {
                    name: "tool".to_string(),
                },
            }),
            ..base_request()
        };
        assert!(validate_request_compat(&named, "Qwen/Qwen1.5-0.5B-Chat").is_err());

        let allowed_tools = ChatCompletionRequest {
            tool_choice: Some(ToolChoice::AllowedTools {
                tool_type: "allowed_tools".to_string(),
                mode: "auto".to_string(),
                tools: vec![ToolReference::Function {
                    name: "tool".to_string(),
                }],
            }),
            ..base_request()
        };
        assert!(validate_request_compat(&allowed_tools, "Qwen/Qwen1.5-0.5B-Chat").is_err());
    }
}
