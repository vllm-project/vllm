use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use uuid::Uuid;
use vllm_engine_core_client::protocol::lora::LoraRequest;
use vllm_engine_core_client::protocol::multimodal::MmFeatures;
use vllm_engine_core_client::protocol::{
    EngineCoreRequest, EngineCoreSamplingParams, ReasoningParserKwargs,
};

use crate::error::{Error, Result};

/// Tokenized decoder-only generate request accepted by [`crate::Llm`].
///
/// This is the first-stage Rust subset of the inputs that eventually flow into
/// Python `AsyncLLM.generate()`. The boundary is intentionally above
/// [`EngineCoreRequest`], but below higher-level text and multimodal
/// preprocessing.
///
/// Original Python API reference:
/// <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/engine/protocol.py#L67-L84>
#[derive(Debug, Clone, PartialEq)]
pub struct GenerateRequest {
    /// Unique ID of the request.
    pub request_id: String,
    /// Token IDs of the prompt.
    pub prompt_token_ids: Vec<u32>,
    /// Sampling parameters forwarded to engine-core.
    pub sampling_params: EngineCoreSamplingParams,
    /// Optional multimodal features already prepared by `vllm-chat`.
    pub mm_features: Option<MmFeatures>,
    /// Unix timestamp, in seconds, when this request arrived at the frontend.
    ///
    /// When omitted, the Rust frontend fills it immediately before sending the
    /// request to engine-core, matching Python's default arrival-time behavior.
    pub arrival_time: Option<f64>,
    /// Optional salt used to partition prefix-cache entries for this request.
    pub cache_salt: Option<String>,
    /// Optional tracing headers to forward to engine-core and downstream
    /// observability hooks.
    pub trace_headers: Option<BTreeMap<String, String>>,
    /// Request scheduling priority. Lower values are scheduled earlier.
    pub priority: i32,
    /// Optional data-parallel rank override for routing this request.
    pub data_parallel_rank: Option<u32>,
    /// Optional reasoning-parser kwargs forwarded to engine-side structured
    /// output logic.
    pub reasoning_parser_kwargs: Option<ReasoningParserKwargs>,
    /// Optional LoRA adapter request applied to this generation.
    pub lora_request: Option<LoraRequest>,
}

#[derive(Debug)]
pub(crate) struct PreparedGenerateRequest {
    pub engine_request: EngineCoreRequest,
}

impl GenerateRequest {
    /// Validate and lower this request into the raw engine-core request format.
    pub(crate) fn prepare(self, randomize_request_id: bool) -> Result<PreparedGenerateRequest> {
        if self.prompt_token_ids.is_empty() {
            return Err(Error::EmptyPromptTokenIds {
                request_id: self.request_id,
            });
        }
        let GenerateRequest {
            request_id,
            prompt_token_ids,
            sampling_params,
            mm_features,
            arrival_time,
            cache_salt,
            trace_headers,
            priority,
            data_parallel_rank,
            reasoning_parser_kwargs,
            lora_request,
        } = self;

        let external_request_id = request_id;
        let engine_request_id = if randomize_request_id {
            let random_suffix = Uuid::new_v4().simple().to_string();
            format!("{external_request_id}-{}", &random_suffix[..8])
        } else {
            external_request_id.clone()
        };
        Ok(PreparedGenerateRequest {
            engine_request: EngineCoreRequest {
                request_id: engine_request_id,
                prompt_token_ids: Some(prompt_token_ids),
                mm_features,
                sampling_params: Some(sampling_params),
                pooling_params: None,
                arrival_time: arrival_time.unwrap_or_else(current_unix_timestamp_secs),
                lora_request,
                cache_salt,
                data_parallel_rank,
                prompt_embeds: None,
                prompt_is_token_ids: None,
                client_index: 0,
                current_wave: 0,
                priority,
                trace_headers,
                resumable: false,
                external_req_id: Some(external_request_id),
                // Rust parser doesn't expose this information, leave it unset and let the
                // reasoning logic in engine-sided structured output manager handle it.
                reasoning_ended: None,
                reasoning_parser_kwargs,
                abort_immediately: false,
            },
        })
    }
}

impl PreparedGenerateRequest {
    /// Return the original prompt token IDs copied into the raw engine request.
    pub fn prompt_token_ids(&self) -> &[u32] {
        self.engine_request
            .prompt_token_ids
            .as_ref()
            .expect("prepared request must have prompt token ids")
    }
}

fn current_unix_timestamp_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is before unix epoch")
        .as_secs_f64()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use vllm_engine_core_client::protocol::{EngineCoreSamplingParams, ReasoningParserKwargs};

    use super::GenerateRequest;
    use crate::error::Error;

    fn sample_request() -> GenerateRequest {
        GenerateRequest {
            request_id: "req-1".to_string(),
            prompt_token_ids: vec![11, 22, 33],
            sampling_params: EngineCoreSamplingParams::for_test(),
            mm_features: None,
            arrival_time: Some(42.5),
            cache_salt: Some("salt".to_string()),
            trace_headers: Some(BTreeMap::from([(
                "x-trace-id".to_string(),
                "abc".to_string(),
            )])),
            priority: 3,
            data_parallel_rank: Some(2),
            reasoning_parser_kwargs: Some(ReasoningParserKwargs {
                chat_template_kwargs: [(
                    "chat_template_kwargs".to_string(),
                    serde_json::json!({
                        "enable_thinking": true,
                    }),
                )]
                .into(),
            }),
            lora_request: None,
        }
    }

    #[test]
    fn prepare_builds_engine_core_request() {
        let prepared = sample_request().prepare(true).unwrap();

        assert_eq!(prepared.prompt_token_ids(), &[11, 22, 33]);

        let request = prepared.engine_request;
        assert_eq!(request.external_req_id.as_deref(), Some("req-1"));
        assert!(request.request_id.starts_with("req-1-"));
        assert_ne!(request.request_id, "req-1");
        assert_eq!(request.prompt_token_ids.as_deref(), Some(&[11, 22, 33][..]));
        assert_eq!(request.arrival_time, 42.5);
        assert_eq!(request.cache_salt.as_deref(), Some("salt"));
        assert_eq!(request.data_parallel_rank, Some(2));
        assert_eq!(
            request.trace_headers,
            Some(BTreeMap::from([(
                "x-trace-id".to_string(),
                "abc".to_string(),
            )]))
        );
        assert_eq!(request.reasoning_ended, None);
        assert_eq!(
            request
                .reasoning_parser_kwargs
                .as_ref()
                .and_then(|kwargs| kwargs.chat_template_kwargs.get("chat_template_kwargs")),
            Some(&serde_json::json!({
                "enable_thinking": true
            }))
        );
    }

    #[test]
    fn prepare_rejects_empty_prompt_tokens() {
        let mut request = sample_request();
        request.prompt_token_ids.clear();

        let error = request.prepare(true).unwrap_err();
        assert!(matches!(
            error,
            Error::EmptyPromptTokenIds { request_id } if request_id == "req-1"
        ));
    }

    #[test]
    fn prepare_can_preserve_external_request_id() {
        let prepared = sample_request().prepare(false).unwrap();

        let request = prepared.engine_request;
        assert_eq!(request.external_req_id.as_deref(), Some("req-1"));
        assert_eq!(request.request_id, "req-1");
    }

    #[test]
    fn prepare_forwards_multimodal_features() {
        let mut request = sample_request();
        request.mm_features = Some(Vec::new());

        let prepared = request.prepare(false).unwrap();

        assert_eq!(prepared.engine_request.mm_features, Some(Vec::new()));
    }
}
