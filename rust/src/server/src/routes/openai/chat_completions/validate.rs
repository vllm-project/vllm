// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use super::types::ChatCompletionRequest;
use crate::error::{ApiError, bail_invalid_request};

/// Validate OpenAI chat-completions protocol invariants and reject
/// unimplemented request fields.
///
/// Cross-route domain rules (sampling bounds, `max_tokens`, logprobs, token
/// IDs, `cache_salt`, chat messages/tools) belong in `vllm-text` or
/// `vllm-chat`, not here.
pub(super) fn validate_request(request: &ChatCompletionRequest) -> Result<(), ApiError> {
    if request.stream_options.as_ref().is_some_and(|options| !options.is_empty()) && !request.stream
    {
        bail_invalid_request!(
            param = "stream_options",
            "stream_options are only supported when stream=true."
        );
    }

    if request.top_logprobs.is_some_and(|n| n != 0) && !request.logprobs {
        bail_invalid_request!(
            param = "top_logprobs",
            "top_logprobs can only be used when logprobs=true."
        );
    }

    if request.logprob_token_ids.as_ref().is_some_and(|ids| !ids.is_empty()) && !request.logprobs {
        bail_invalid_request!(
            param = "logprob_token_ids",
            "when using `logprob_token_ids`, `logprobs` must be set to true."
        );
    }

    if request.stream && request.prompt_logprobs.is_some_and(|n| n > 0 || n == -1) {
        bail_invalid_request!(
            param = "prompt_logprobs",
            "prompt_logprobs are not available when stream=true."
        );
    }

    reject_unsupported_fields(request)?;

    Ok(())
}

/// Reject request fields that are accepted for deserialization but not yet
/// implemented in the Rust frontend.
fn reject_unsupported_fields(request: &ChatCompletionRequest) -> Result<(), ApiError> {
    if request.use_beam_search {
        bail_invalid_request!(
            param = "use_beam_search",
            "use_beam_search is not supported."
        );
    }

    // Python's default is 1.0, which SDKs may send explicitly.
    if request.length_penalty.is_some_and(|penalty| penalty != 1.0) {
        bail_invalid_request!(param = "length_penalty", "length_penalty is not supported.");
    }
    if request.media_io_kwargs.is_some() {
        bail_invalid_request!(
            param = "media_io_kwargs",
            "media_io_kwargs is not supported."
        );
    }
    if request.mm_processor_kwargs.is_some() {
        bail_invalid_request!(
            param = "mm_processor_kwargs",
            "mm_processor_kwargs is not supported."
        );
    }
    if request
        .thinking_loop_break
        .is_some_and(|loop_break| !loop_break.follows_server_config())
    {
        bail_invalid_request!(
            param = "thinking_loop_break",
            "thinking_loop_break is not supported."
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use serde_json::json;

    use super::validate_request;
    use crate::routes::openai::chat_completions::types::ChatCompletionRequest;
    use crate::routes::openai::utils::structured_outputs::ResponseFormat;
    use crate::routes::openai::utils::types::{
        ChatMessage, Function, MessageContent, ReasoningEffort, StringOrArray, Tool, ToolChoice,
        ToolChoiceValue,
    };

    fn non_streaming_request(stream_options: serde_json::Value) -> ChatCompletionRequest {
        serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": false,
            "stream_options": stream_options,
        }))
        .expect("parse request")
    }

    fn base_request() -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: Some("Qwen/Qwen1.5-0.5B-Chat".to_string()),
            messages: vec![ChatMessage::User {
                content: MessageContent::Text("hello".to_string()),
                name: None,
            }],
            stream: true,
            ..Default::default()
        }
    }

    #[test]
    fn validate_request_accepts_empty_stream_options_without_streaming() {
        let request = non_streaming_request(json!({}));
        validate_request(&request).expect("empty stream_options sets nothing");
    }

    #[test]
    fn validate_request_rejects_stream_options_without_streaming() {
        let cases = [
            json!({"include_usage": false}),
            json!({"continuous_usage_stats": true}),
        ];
        for stream_options in cases {
            let request = non_streaming_request(stream_options.clone());
            assert!(validate_request(&request).is_err(), "{stream_options}");
        }
    }

    #[test]
    fn validate_request_accepts_stop() {
        let request = ChatCompletionRequest {
            stop: Some(StringOrArray::String("stop".to_string())),
            ..base_request()
        };

        validate_request(&request).expect("stop strings should be accepted");
    }

    #[test]
    fn validate_request_accepts_non_zero_penalties_and_function_tools() {
        let request = ChatCompletionRequest {
            frequency_penalty: Some(0.5),
            presence_penalty: Some(0.25),
            min_p: Some(0.2),
            repetition_penalty: Some(1.1),
            seed: Some(7),
            ..base_request()
        };
        validate_request(&request).expect("sampling fields should be accepted");

        let request = ChatCompletionRequest {
            tools: Some(vec![Tool {
                tool_type: "function".to_string(),
                function: Function {
                    name: "tool".to_string(),
                    description: None,
                    parameters: Some(serde_json::Map::new()),
                    strict: None,
                    defer_loading: None,
                },
                defer_loading: None,
            }]),
            ..base_request()
        };
        validate_request(&request).expect("function tools should be accepted");

        let request = ChatCompletionRequest {
            messages: vec![ChatMessage::Developer {
                content: MessageContent::Text("policy".to_string()),
                tools: Some(vec![Tool {
                    tool_type: "function".to_string(),
                    function: Function {
                        name: "tool".to_string(),
                        description: None,
                        parameters: Some(serde_json::Map::new()),
                        strict: None,
                        defer_loading: None,
                    },
                    defer_loading: None,
                }]),
                name: None,
            }],
            ..base_request()
        };
        validate_request(&request).expect("developer function tools should be accepted");
    }

    #[test]
    fn validate_request_rejects_only_non_default_length_penalty() {
        let request = ChatCompletionRequest {
            length_penalty: Some(1.0),
            ..base_request()
        };
        validate_request(&request).expect("the default length_penalty should be accepted");

        let request = ChatCompletionRequest {
            length_penalty: Some(2.0),
            ..base_request()
        };
        assert!(validate_request(&request).is_err());
    }

    #[test]
    fn validate_request_rejects_non_default_thinking_loop_break() {
        for (thinking_loop_break, accepted) in [
            (json!(null), true),
            (json!(true), true),
            (json!(false), false),
            (json!("force"), false),
            (json!("ramp"), false),
        ] {
            let request: ChatCompletionRequest = serde_json::from_value(json!({
                "model": "Qwen/Qwen1.5-0.5B-Chat",
                "messages": [{"role": "user", "content": "hello"}],
                "thinking_loop_break": thinking_loop_break,
            }))
            .expect("parse request");
            assert_eq!(
                validate_request(&request).is_ok(),
                accepted,
                "{thinking_loop_break}"
            );
        }

        let parsed = serde_json::from_value::<ChatCompletionRequest>(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "messages": [{"role": "user", "content": "hello"}],
            "thinking_loop_break": "always",
        }));
        assert!(parsed.is_err());
    }

    #[test]
    fn validate_request_accepts_output_logprobs() {
        let request = ChatCompletionRequest {
            logprobs: true,
            ..base_request()
        };
        validate_request(&request).expect("logprobs should be accepted");
    }

    #[test]
    fn validate_request_accepts_reasoning_effort() {
        let request = ChatCompletionRequest {
            reasoning_effort: Some(ReasoningEffort::Max),
            chat_template_kwargs: Some(HashMap::from([(
                "reasoning_effort".to_string(),
                json!("low"),
            )])),
            ..base_request()
        };

        validate_request(&request).expect("reasoning_effort should be accepted");
    }

    #[test]
    fn validate_request_accepts_spaces_between_special_tokens_false() {
        let request = ChatCompletionRequest {
            spaces_between_special_tokens: false,
            ..base_request()
        };

        validate_request(&request).expect("spaces_between_special_tokens=false should be accepted");
    }

    #[test]
    fn validate_request_accepts_include_reasoning_false() {
        let request = ChatCompletionRequest {
            include_reasoning: false,
            ..base_request()
        };

        validate_request(&request).expect("include_reasoning=false should be accepted");
    }

    #[test]
    fn validate_request_rejects_top_logprobs_without_logprobs() {
        for top_logprobs in [-1, 1] {
            let request = ChatCompletionRequest {
                top_logprobs: Some(top_logprobs),
                ..base_request()
            };
            assert!(validate_request(&request).is_err(), "{top_logprobs}");
        }
    }

    #[test]
    fn validate_request_accepts_zero_top_logprobs_without_logprobs() {
        let request = ChatCompletionRequest {
            top_logprobs: Some(0),
            ..base_request()
        };
        validate_request(&request).expect("top_logprobs=0 does not request alternatives");
    }

    #[test]
    fn validate_request_rejects_streaming_prompt_logprobs_requests() {
        let request = ChatCompletionRequest {
            prompt_logprobs: Some(1),
            ..base_request()
        };
        assert!(validate_request(&request).is_err());

        let request = ChatCompletionRequest {
            prompt_logprobs: Some(-1),
            ..base_request()
        };
        assert!(validate_request(&request).is_err());
    }

    #[test]
    fn validate_request_accepts_response_format() {
        let request = ChatCompletionRequest {
            response_format: Some(ResponseFormat::Text),
            ..base_request()
        };
        validate_request(&request).expect("response_format=text should be accepted");

        let request = ChatCompletionRequest {
            response_format: Some(ResponseFormat::JsonObject),
            ..base_request()
        };
        validate_request(&request).expect("response_format=json_object should be accepted");
    }

    #[test]
    fn validate_request_accepts_noop_tool_choice_none() {
        let request = ChatCompletionRequest {
            tool_choice: Some(ToolChoice::Value(ToolChoiceValue::None)),
            ..base_request()
        };

        validate_request(&request).expect("tool_choice=none is ok");
    }
}
