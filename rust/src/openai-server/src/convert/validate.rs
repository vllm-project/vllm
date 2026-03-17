use openai_protocol::chat::ChatCompletionRequest;
use openai_protocol::common::{StringOrArray, ToolChoice, ToolChoiceValue};
use tracing::warn;

use crate::error::{ApiError, bail_invalid_request};

/// Enforce the minimal compatibility contract for the Rust OpenAI server.
pub(super) fn validate_request_compat(
    request: &ChatCompletionRequest,
    configured_model: &str,
) -> Result<(), ApiError> {
    if request.model != configured_model {
        return Err(ApiError::model_not_found(request.model.clone()));
    }

    if !request.stream {
        bail_invalid_request!(param = "stream", "Only stream=true is supported.");
    }

    if request.n.unwrap_or(1) > 1 {
        bail_invalid_request!(param = "n", "Only n=1 is supported.");
    }

    if has_non_empty_stop(request.stop.as_ref()) {
        bail_invalid_request!(
            param = "stop",
            "Stop strings are not supported by the minimal Rust frontend."
        );
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
        bail_invalid_request!(param = "logprobs", "logprobs are not supported.");
    }

    if request.top_logprobs.unwrap_or(0) > 0 {
        bail_invalid_request!(param = "top_logprobs", "top_logprobs are not supported.");
    }

    if request.stream_options.is_some() {
        warn!("stream_options are currently no-op.");
    }

    if request.response_format.is_some() {
        bail_invalid_request!(
            param = "response_format",
            "response_format is not supported."
        );
    }

    if request
        .tools
        .as_ref()
        .is_some_and(|tools| !tools.is_empty())
    {
        bail_invalid_request!(param = "tools", "Tool calling is not supported.");
    }

    if let Some(tool_choice) = &request.tool_choice
        && !matches!(tool_choice, ToolChoice::Value(ToolChoiceValue::None))
    {
        bail_invalid_request!(param = "tool_choice", "tool_choice is not supported.");
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
        bail_invalid_request!(
            param = "parallel_tool_calls",
            "parallel_tool_calls is not supported."
        );
    }

    if request.no_stop_trim {
        bail_invalid_request!(param = "no_stop_trim", "no_stop_trim is not supported.");
    }

    if request.return_hidden_states {
        bail_invalid_request!(
            param = "return_hidden_states",
            "return_hidden_states is not supported."
        );
    }

    if request.sampling_seed.is_some() {
        bail_invalid_request!(param = "sampling_seed", "sampling_seed is not supported.");
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
fn reject_non_zero(value: Option<f32>, param: &'static str, message: &str) -> Result<(), ApiError> {
    if value.unwrap_or(0.0) != 0.0 {
        bail_invalid_request!(param = param, "{}", message);
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
    use openai_protocol::chat::{ChatCompletionRequest, ChatMessage, MessageContent};
    use openai_protocol::common::{ResponseFormat, Tool, ToolChoice};
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
    fn validate_request_compat_rejects_response_format() {
        let request = ChatCompletionRequest {
            response_format: Some(ResponseFormat::Text),
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
