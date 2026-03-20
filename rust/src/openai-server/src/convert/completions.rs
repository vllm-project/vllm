use openai_protocol::common::StringOrArray;
use openai_protocol::completion::CompletionRequest as OpenAiCompletionRequest;
use openai_protocol::validated::Normalizable;
use serde::de::Error as _;
use serde::{Deserialize, Deserializer};
use serde_json::{Map, Value};
use uuid::Uuid;
use validator::Validate;
use vllm_text::{Prompt, SamplingParams, TextDecodeOptions, TextRequest};

use crate::error::{ApiError, bail_invalid_request};

/// A wrapper around [`OpenAiCompletionRequest`] that supports both text and token-ID-array prompts.
///
/// Uses a custom deserializer because the upstream `OpenAiCompletionRequest` requires `prompt` as
/// `StringOrArray`, which cannot represent token-ID arrays. The custom impl parses our extended
/// `Prompt` first, then patches a string placeholder into the upstream struct so its parsing and
/// defaults apply to all other fields.
#[derive(Debug, Clone, Validate)]
pub struct CompletionRequest {
    /// Local prompt field that extends upstream completions parsing with token-ID-array support.
    pub prompt: Prompt,
    /// Upstream request shape used for all other fields and their defaults.
    pub inner: OpenAiCompletionRequest,
}

impl Normalizable for CompletionRequest {}

impl<'de> Deserialize<'de> for CompletionRequest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let mut object = Map::<String, Value>::deserialize(deserializer)?;
        let prompt_value = object
            .get("prompt")
            .cloned()
            .ok_or_else(|| D::Error::missing_field("prompt"))?;
        let prompt = serde_json::from_value::<Prompt>(prompt_value).map_err(D::Error::custom)?;

        let upstream_prompt = match &prompt {
            Prompt::Text(text) => Value::String(text.clone()),
            // Upstream `OpenAiCompletionRequest` cannot deserialize token-ID arrays in `prompt`,
            // so use a harmless placeholder string solely to reuse its field parsing and defaults.
            Prompt::TokenIds(_) => Value::String(String::new()),
        };
        object.insert("prompt".to_string(), upstream_prompt);

        // Re-parse the patched object through the upstream type so this wrapper does not need to
        // locally mirror every completions field or default.
        let inner = serde_json::from_value(Value::Object(object)).map_err(D::Error::custom)?;
        Ok(Self { prompt, inner })
    }
}

/// Lowered completion request plus the public response metadata carried by every SSE chunk.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedRequest {
    /// Stable OpenAI-style response ID, also reused as the internal text request ID.
    pub response_id: String,
    /// Public model ID echoed back to the client.
    pub response_model: String,
    /// Whether the caller asked for the final streamed usage chunk.
    pub include_usage: bool,
    /// Lowered text request for the shared `vllm-text` facade.
    pub text_request: TextRequest,
}

/// Validate and lower one OpenAI completions request into the internal text-generation format.
pub fn prepare_completion_request(
    request: &CompletionRequest,
    configured_model: &str,
) -> Result<PreparedRequest, ApiError> {
    validate_request_compat(request, configured_model)?;

    let response_id = format!("cmpl-{}", Uuid::new_v4());
    let include_usage = (request.inner.stream_options.as_ref())
        .and_then(|options| options.include_usage)
        .unwrap_or(false);

    let text_request = TextRequest {
        request_id: response_id.clone(),
        prompt: request.prompt.clone(),
        sampling_params: SamplingParams {
            temperature: request.inner.temperature,
            top_p: request.inner.top_p,
            top_k: request.inner.top_k,
            seed: request.inner.sampling_seed,
            max_tokens: request.inner.max_tokens,
            min_tokens: request.inner.min_tokens,
            min_p: request.inner.min_p,
            frequency_penalty: request.inner.frequency_penalty,
            presence_penalty: request.inner.presence_penalty,
            repetition_penalty: request.inner.repetition_penalty,
            stop_token_ids: request.inner.stop_token_ids.clone(),
            ignore_eos: request.inner.ignore_eos,
        },
        decode_options: TextDecodeOptions {
            skip_special_tokens: request.inner.skip_special_tokens,
            // `no_stop_trim=true` is the closest existing toggle for keeping the terminal stop
            // token visible in decoded output.
            include_stop_str_in_output: request.inner.no_stop_trim,
        },
    };

    Ok(PreparedRequest {
        response_id,
        response_model: configured_model.to_string(),
        include_usage,
        text_request,
    })
}

fn validate_request_compat(
    request: &CompletionRequest,
    configured_model: &str,
) -> Result<(), ApiError> {
    // This path is intentionally scoped to the minimum surface needed by `vllm-bench` random
    // workload compatibility, so unsupported legacy completions features fail early here.
    if request.inner.model != configured_model {
        return Err(ApiError::model_not_found(request.inner.model.clone()));
    }

    if !request.inner.stream {
        bail_invalid_request!(param = "stream", "Only stream=true is supported.");
    }

    if request.inner.n.unwrap_or(1) > 1 {
        bail_invalid_request!(param = "n", "Only n=1 is supported.");
    }

    if request.inner.echo {
        bail_invalid_request!(param = "echo", "echo is not supported.");
    }

    if request.inner.suffix.is_some() {
        bail_invalid_request!(param = "suffix", "suffix is not supported.");
    }

    if request.inner.best_of.is_some() {
        bail_invalid_request!(param = "best_of", "best_of is not supported.");
    }

    if has_non_empty_stop(request.inner.stop.as_ref()) {
        bail_invalid_request!(
            param = "stop",
            "Stop strings are not supported by the minimal Rust frontend."
        );
    }

    if request.inner.logprobs.is_some() {
        bail_invalid_request!(param = "logprobs", "logprobs are not supported.");
    }

    if request.inner.seed.is_some() {
        bail_invalid_request!(
            param = "seed",
            "seed is not supported, use sampling_seed instead."
        );
    }

    if request.inner.logit_bias.is_some() {
        bail_invalid_request!(param = "logit_bias", "logit_bias is not supported.");
    }

    if request.inner.regex.is_some() {
        bail_invalid_request!(param = "regex", "regex constraints are not supported.");
    }

    if request.inner.ebnf.is_some() {
        bail_invalid_request!(param = "ebnf", "ebnf constraints are not supported.");
    }

    if request.inner.json_schema.is_some() {
        bail_invalid_request!(param = "json_schema", "json_schema is not supported.");
    }

    if request.inner.lora_path.is_some() {
        bail_invalid_request!(param = "lora_path", "lora_path is not supported.");
    }

    if request.inner.session_params.is_some() {
        bail_invalid_request!(
            param = "session_params",
            "session_params are not supported."
        );
    }

    if request.inner.return_hidden_states {
        bail_invalid_request!(
            param = "return_hidden_states",
            "return_hidden_states is not supported."
        );
    }

    Ok(())
}

fn has_non_empty_stop(stop: Option<&StringOrArray>) -> bool {
    match stop {
        None => false,
        Some(StringOrArray::String(value)) => !value.is_empty(),
        Some(StringOrArray::Array(values)) => values.iter().any(|value| !value.is_empty()),
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::common::StringOrArray;
    use serde_json::json;
    use vllm_text::Prompt;

    use super::{CompletionRequest, prepare_completion_request};

    fn base_request_json() -> serde_json::Value {
        json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "prompt": "hello",
            "stream": true
        })
    }

    #[test]
    fn completion_http_request_deserializes_text_prompt() {
        let request: CompletionRequest =
            serde_json::from_value(base_request_json()).expect("parse request");

        assert_eq!(request.prompt, Prompt::Text("hello".to_string()));
        assert_eq!(
            request.inner.prompt,
            StringOrArray::String("hello".to_string())
        );
    }

    #[test]
    fn completion_http_request_deserializes_token_id_prompt() {
        let request: CompletionRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "prompt": [11, 22, 33],
            "stream": true,
            "ignore_eos": true,
            "max_tokens": 7
        }))
        .expect("parse request");

        assert_eq!(request.prompt, Prompt::TokenIds(vec![11, 22, 33]));
        assert_eq!(request.inner.prompt, StringOrArray::String(String::new()));
        assert_eq!(request.inner.max_tokens, Some(7));
        assert!(request.inner.ignore_eos);
    }

    #[test]
    fn prepare_completion_request_maps_sampling_fields() {
        let request: CompletionRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "prompt": [11, 22, 33],
            "stream": true,
            "stream_options": {"include_usage": true},
            "max_tokens": 7,
            "top_p": 0.9,
            "top_k": 42,
            "min_p": 0.1,
            "frequency_penalty": 0.2,
            "presence_penalty": 0.3,
            "repetition_penalty": 1.1,
            "ignore_eos": true,
            "skip_special_tokens": false
        }))
        .expect("parse request");

        let prepared =
            prepare_completion_request(&request, "Qwen/Qwen1.5-0.5B-Chat").expect("prepare");

        assert!(prepared.include_usage);
        assert_eq!(
            prepared.text_request.prompt,
            Prompt::TokenIds(vec![11, 22, 33])
        );
        assert_eq!(prepared.text_request.sampling_params.max_tokens, Some(7));
        assert_eq!(prepared.text_request.sampling_params.top_p, Some(0.9));
        assert_eq!(prepared.text_request.sampling_params.top_k, Some(42));
        assert_eq!(prepared.text_request.sampling_params.min_p, Some(0.1));
        assert_eq!(
            prepared.text_request.sampling_params.frequency_penalty,
            Some(0.2)
        );
        assert_eq!(
            prepared.text_request.sampling_params.presence_penalty,
            Some(0.3)
        );
        assert_eq!(
            prepared.text_request.sampling_params.repetition_penalty,
            Some(1.1)
        );
        assert!(prepared.text_request.sampling_params.ignore_eos);
        assert!(!prepared.text_request.decode_options.skip_special_tokens);
    }

    #[test]
    fn prepare_completion_request_rejects_unsupported_fields() {
        let request: CompletionRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "prompt": "hello",
            "stream": true,
            "logprobs": 1
        }))
        .expect("parse request");

        assert!(prepare_completion_request(&request, "Qwen/Qwen1.5-0.5B-Chat").is_err());
    }
}
