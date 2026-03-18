use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use vllm_engine_core_client::protocol::{
    EngineCoreRequest, EngineCoreSamplingParams, OpaqueValue, RequestOutputKind,
};

use crate::error::{Error, Result};

/// Tokenized decoder-only generate request accepted by [`crate::Llm`].
///
/// This is the first-stage Rust subset of the inputs that eventually flow into Python
/// `AsyncLLM.generate()`. The boundary is intentionally above [`EngineCoreRequest`], but below
/// higher-level text and multimodal preprocessing.
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
    ///
    /// The `output_kind` field controls whether [`crate::GenerateOutput::token_ids`] is
    /// delta-only, cumulative, or final-only.
    pub sampling_params: EngineCoreSamplingParams,

    pub arrival_time: Option<f64>,
    pub cache_salt: Option<String>,
    pub trace_headers: Option<BTreeMap<String, String>>,
    pub priority: i32,
    pub data_parallel_rank: Option<u32>,
    pub reasoning_ended: Option<bool>,
    pub lora_request: Option<OpaqueValue>,
}

#[derive(Debug)]
pub(crate) struct PreparedGenerateRequest {
    pub engine_request: EngineCoreRequest,
}

impl GenerateRequest {
    /// Validate and lower this request into the raw engine-core request format.
    pub(crate) fn prepare(self) -> Result<PreparedGenerateRequest> {
        if self.prompt_token_ids.is_empty() {
            return Err(Error::EmptyPromptTokenIds {
                request_id: self.request_id,
            });
        }
        let GenerateRequest {
            request_id,
            prompt_token_ids,
            sampling_params,
            arrival_time,
            cache_salt,
            trace_headers,
            priority,
            data_parallel_rank,
            reasoning_ended,
            lora_request,
        } = self;

        Ok(PreparedGenerateRequest {
            engine_request: EngineCoreRequest {
                request_id,
                prompt_token_ids: Some(prompt_token_ids),
                mm_features: None,
                sampling_params: Some(sampling_params),
                pooling_params: None,
                arrival_time: arrival_time.unwrap_or_else(current_unix_timestamp_secs),
                lora_request,
                cache_salt,
                data_parallel_rank,
                prompt_embeds: None,
                client_index: 0,
                current_wave: 0,
                priority,
                trace_headers,
                resumable: false,
                external_req_id: None,
                reasoning_ended,
            },
        })
    }
}

impl PreparedGenerateRequest {
    /// Return the requested output aggregation mode.
    pub fn output_kind(&self) -> RequestOutputKind {
        self.engine_request
            .sampling_params
            .as_ref()
            .expect("prepared request must have sampling params")
            .output_kind
    }

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

    use vllm_engine_core_client::protocol::{EngineCoreSamplingParams, RequestOutputKind};

    use super::GenerateRequest;
    use crate::error::Error;

    fn sample_request() -> GenerateRequest {
        GenerateRequest {
            request_id: "req-1".to_string(),
            prompt_token_ids: vec![11, 22, 33],
            sampling_params: EngineCoreSamplingParams::for_test(),
            arrival_time: Some(42.5),
            cache_salt: Some("salt".to_string()),
            trace_headers: Some(BTreeMap::from([(
                "x-trace-id".to_string(),
                "abc".to_string(),
            )])),
            priority: 3,
            data_parallel_rank: Some(2),
            reasoning_ended: Some(true),
            lora_request: None,
        }
    }

    #[test]
    fn prepare_builds_engine_core_request() {
        let prepared = sample_request().prepare().unwrap();

        assert_eq!(prepared.output_kind(), RequestOutputKind::Cumulative);
        assert_eq!(prepared.prompt_token_ids(), &[11, 22, 33]);

        let request = prepared.engine_request;
        expect_test::expect![[r#"
            EngineCoreRequest {
                request_id: "req-1",
                prompt_token_ids: Some(
                    [
                        11,
                        22,
                        33,
                    ],
                ),
                mm_features: None,
                sampling_params: Some(
                    EngineCoreSamplingParams {
                        temperature: 1.0,
                        top_p: 1.0,
                        top_k: 0,
                        seed: None,
                        max_tokens: 65536,
                        min_tokens: 0,
                        min_p: 0.0,
                        frequency_penalty: 0.0,
                        presence_penalty: 0.0,
                        repetition_penalty: 1.0,
                        stop_token_ids: [],
                        eos_token_id: None,
                        all_stop_token_ids: {},
                        output_kind: Cumulative,
                    },
                ),
                pooling_params: None,
                arrival_time: 42.5,
                lora_request: None,
                cache_salt: Some(
                    "salt",
                ),
                data_parallel_rank: Some(
                    2,
                ),
                prompt_embeds: None,
                client_index: 0,
                current_wave: 0,
                priority: 3,
                trace_headers: Some(
                    {
                        "x-trace-id": "abc",
                    },
                ),
                resumable: false,
                external_req_id: None,
                reasoning_ended: Some(
                    true,
                ),
            }
        "#]]
        .assert_debug_eq(&request);
    }

    #[test]
    fn prepare_rejects_empty_prompt_tokens() {
        let mut request = sample_request();
        request.prompt_token_ids.clear();

        let error = request.prepare().unwrap_err();
        assert!(matches!(
            error,
            Error::EmptyPromptTokenIds { request_id } if request_id == "req-1"
        ));
    }
}
