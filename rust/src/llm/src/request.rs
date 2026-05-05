use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use uuid::Uuid;
use vllm_engine_core_client::protocol::{EngineCoreRequest, EngineCoreSamplingParams, OpaqueValue};

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

    // Fields below are currently likely unused by callers.
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
            arrival_time,
            cache_salt,
            trace_headers,
            priority,
            data_parallel_rank,
            reasoning_ended,
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
                external_req_id: Some(external_request_id),
                reasoning_ended,
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

    use vllm_engine_core_client::protocol::EngineCoreSamplingParams;

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
        assert_eq!(request.reasoning_ended, Some(true));
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
}
