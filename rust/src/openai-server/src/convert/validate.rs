use openai_protocol::chat::ChatCompletionRequest;
use openai_protocol::common::{StringOrArray, ToolChoice, ToolChoiceValue};

use crate::error::ApiError;

/// Enforce the minimal compatibility contract for the Rust OpenAI server.
pub(super) fn validate_request_compat(
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

    // if request.stream_options.is_some() {
    //     return Err(ApiError::invalid_request_param(
    //         "stream_options are not supported.",
    //         "stream_options",
    //     ));
    // }

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

/// Return whether the request asks for any non-empty stop string behavior.
fn has_non_empty_stop(stop: Option<&StringOrArray>) -> bool {
    match stop {
        None => false,
        Some(StringOrArray::String(value)) => !value.is_empty(),
        Some(StringOrArray::Array(values)) => values.iter().any(|value| !value.is_empty()),
    }
}

/// Reject one numeric option unless it is absent or exactly zero.
fn reject_non_zero(value: Option<f32>, param: &str, message: &str) -> Result<(), ApiError> {
    if value.unwrap_or(0.0) != 0.0 {
        return Err(ApiError::invalid_request_param(message, param));
    }
    Ok(())
}

/// Reject one option unless it is entirely absent.
fn reject_non_default<T>(value: Option<&T>, param: &str, message: &str) -> Result<(), ApiError> {
    if value.is_some() {
        return Err(ApiError::invalid_request_param(message, param));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use openai_protocol::chat::{ChatCompletionRequest, ChatMessage, MessageContent};
    use openai_protocol::common::{ResponseFormat, StreamOptions, Tool, ToolChoice};
    use serde_json::json;

    use super::validate_request_compat;

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
    fn validate_request_compat_rejects_non_empty_stop() {
        let request = ChatCompletionRequest {
            stop: Some(openai_protocol::common::StringOrArray::String(
                "stop".to_string(),
            )),
            ..base_request()
        };

        assert!(validate_request_compat(&request, "Qwen/Qwen1.5-0.5B-Chat").is_err());
    }

    #[test]
    fn validate_request_compat_rejects_non_zero_penalties_and_tools() {
        let request = ChatCompletionRequest {
            frequency_penalty: Some(0.5),
            ..base_request()
        };
        assert!(validate_request_compat(&request, "Qwen/Qwen1.5-0.5B-Chat").is_err());

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
        assert!(validate_request_compat(&request, "Qwen/Qwen1.5-0.5B-Chat").is_err());
    }

    #[test]
    fn validate_request_compat_rejects_response_format_and_stream_options() {
        let request = ChatCompletionRequest {
            response_format: Some(ResponseFormat::Text),
            ..base_request()
        };
        assert!(validate_request_compat(&request, "Qwen/Qwen1.5-0.5B-Chat").is_err());

        let request = ChatCompletionRequest {
            stream_options: Some(StreamOptions {
                include_usage: Some(true),
            }),
            ..base_request()
        };
        assert!(validate_request_compat(&request, "Qwen/Qwen1.5-0.5B-Chat").is_err());
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
}
