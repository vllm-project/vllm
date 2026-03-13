use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use vllm_engine_core_client::protocol::{
    EngineCoreRequest, OpaqueValue, RequestOutputKind, SamplingParams,
};

use crate::error::{Error, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct GenerateRequest {
    pub request_id: String,
    pub prompt_token_ids: Vec<u32>,
    pub sampling_params: SamplingParams,
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
    pub(crate) fn prepare(self) -> Result<PreparedGenerateRequest> {
        if self.prompt_token_ids.is_empty() {
            return Err(Error::EmptyPromptTokenIds {
                request_id: self.request_id,
            });
        }
        if self.sampling_params.n != 1 {
            return Err(Error::UnsupportedSamplingCount {
                request_id: self.request_id,
                n: self.sampling_params.n,
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
    pub fn output_kind(&self) -> RequestOutputKind {
        self.engine_request
            .sampling_params
            .as_ref()
            .expect("prepared request must have sampling params")
            .output_kind
    }

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

    use vllm_engine_core_client::protocol::{RequestOutputKind, SamplingParams};

    use super::GenerateRequest;
    use crate::error::Error;

    fn sample_request() -> GenerateRequest {
        GenerateRequest {
            request_id: "req-1".to_string(),
            prompt_token_ids: vec![11, 22, 33],
            sampling_params: SamplingParams {
                output_kind: RequestOutputKind::Cumulative,
                ..Default::default()
            },
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
        assert_eq!(request.request_id, "req-1");
        assert_eq!(request.prompt_token_ids, Some(vec![11, 22, 33]));
        assert_eq!(request.arrival_time, 42.5);
        assert_eq!(request.cache_salt.as_deref(), Some("salt"));
        assert_eq!(request.priority, 3);
        assert_eq!(request.data_parallel_rank, Some(2));
        assert_eq!(request.reasoning_ended, Some(true));
        assert_eq!(
            request
                .trace_headers
                .as_ref()
                .and_then(|headers| headers.get("x-trace-id")),
            Some(&"abc".to_string())
        );
        assert!(request.sampling_params.is_some());
        assert!(request.pooling_params.is_none());
        assert!(request.external_req_id.is_none());
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

    #[test]
    fn prepare_rejects_sampling_n_above_one() {
        let mut request = sample_request();
        request.sampling_params.n = 2;

        let error = request.prepare().unwrap_err();
        assert!(matches!(
            error,
            Error::UnsupportedSamplingCount { request_id, n }
                if request_id == "req-1" && n == 2
        ));
    }
}
