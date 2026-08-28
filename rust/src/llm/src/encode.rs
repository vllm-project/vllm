// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeMap;

use futures::StreamExt as _;
use vllm_engine_core_client::EngineCoreOutputStream;
use vllm_engine_core_client::protocol::lora::LoraRequest;
use vllm_engine_core_client::protocol::output::EngineCoreFinishReason;
use vllm_engine_core_client::protocol::pooling::{EngineCorePoolingParams, PoolingTask};
use vllm_engine_core_client::protocol::request::EngineCoreRequest;
use vllm_engine_core_client::protocol::tensor::WireTensor;

use crate::error::{Error, Result};
use crate::output::FinishReason;
use crate::request::prepare_request_id;
use crate::request_metrics::{PoolingRequestMetricsTracker, current_unix_timestamp_secs};

/// Normalized parameters for one token-level pooling request.
///
/// Model-aware defaults and validation are performed by the caller before
/// reaching this token-level API.
#[derive(Debug, Clone, PartialEq)]
pub struct PoolingParams {
    /// Whether to apply activation function to the pooler outputs.
    pub use_activation: bool,
    /// Reduce the dimensions of embeddings if model support matryoshka
    /// representation.
    pub dimensions: Option<u32>,
    /// If set, only the score corresponding to the `step_tag_id` in the
    /// generated sentence should be returned. Otherwise, the scores for all
    /// tokens are returned.
    pub step_tag_id: Option<u32>,
    /// A list of indices for the vocabulary dimensions to be extracted,
    /// such as the token IDs of `good_token` and `bad_token` in the
    /// `math-shepherd-mistral-7b-prm` model.
    pub returned_token_ids: Option<Vec<u32>>,
}

impl PoolingParams {
    fn into_engine(self, task: PoolingTask) -> EngineCorePoolingParams {
        EngineCorePoolingParams {
            use_activation: self.use_activation,
            dimensions: self.dimensions,
            step_tag_id: self.step_tag_id,
            returned_token_ids: self.returned_token_ids,
            task,
        }
    }
}

/// Tokenized pooling request accepted by [`crate::Llm::encode`].
#[derive(Debug, Clone, PartialEq)]
pub struct EncodeRequest {
    /// Unique ID of the request.
    pub request_id: String,
    /// Token IDs of the prompt.
    pub prompt_token_ids: Vec<u32>,
    /// The task used for pooling.
    pub task: PoolingTask,
    /// Pooling parameters forwarded to engine-core.
    pub pooling_params: PoolingParams,
    /// Unix timestamp, in seconds, when this request arrived at the frontend.
    pub arrival_time: Option<f64>,
    /// Optional salt used to partition prefix-cache entries for this request.
    pub cache_salt: Option<String>,
    /// Optional tracing headers to forward to engine-core.
    pub trace_headers: Option<BTreeMap<String, String>>,
    /// Request scheduling priority. Lower values are scheduled earlier.
    pub priority: i32,
    /// Optional data-parallel rank override for routing this request.
    pub data_parallel_rank: Option<u32>,
    /// Stable session identity shared by related requests.
    pub session_id: Option<String>,
    /// Optional LoRA adapter request applied to this request.
    pub lora_request: Option<LoraRequest>,
}

/// One owned pooling tensor returned by engine-core.
#[derive(Debug, Clone, PartialEq)]
pub struct PoolingOutput {
    /// Tensor shape reported by the pooler.
    pub shape: Vec<usize>,
    /// Tensor values normalized to float32.
    pub data: Vec<f32>,
}

impl TryFrom<WireTensor> for PoolingOutput {
    type Error = String;

    fn try_from(tensor: WireTensor) -> std::result::Result<Self, Self::Error> {
        let data = tensor.to_f32_vec()?;
        Ok(Self {
            shape: tensor.shape,
            data,
        })
    }
}

/// Final output of one token-level pooling request.
#[derive(Debug, Clone, PartialEq)]
pub struct EncodeOutput {
    /// Internal engine request ID that produced this output.
    pub request_id: String,
    /// Original prompt token IDs.
    pub prompt_token_ids: Vec<u32>,
    /// Final pooling tensor.
    pub output: PoolingOutput,
    /// Number of prompt tokens served from cache.
    pub cached_token_count: usize,
}

#[derive(Debug)]
pub(crate) struct PreparedEncodeRequest {
    pub engine_request: EngineCoreRequest,
}

impl EncodeRequest {
    pub(crate) fn prepare(self, randomize_request_id: bool) -> Result<PreparedEncodeRequest> {
        if self.prompt_token_ids.is_empty() {
            return Err(Error::EmptyPromptTokenIds {
                request_id: self.request_id,
            });
        }
        let Self {
            request_id,
            prompt_token_ids,
            task,
            pooling_params,
            arrival_time,
            cache_salt,
            trace_headers,
            priority,
            data_parallel_rank,
            session_id,
            lora_request,
        } = self;
        let engine_request_id = prepare_request_id(&request_id, randomize_request_id);

        Ok(PreparedEncodeRequest {
            engine_request: EngineCoreRequest {
                request_id: engine_request_id,
                prompt_token_ids: Some(prompt_token_ids),
                mm_features: None,
                sampling_params: None,
                pooling_params: Some(pooling_params.into_engine(task)),
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
                external_req_id: Some(request_id),
                reasoning_ended: None,
                reasoning_parser_kwargs: None,
                abort_immediately: false,
                session_id,
            },
        })
    }
}

impl PreparedEncodeRequest {
    pub(crate) fn prompt_token_ids(&self) -> &[u32] {
        self.engine_request
            .prompt_token_ids
            .as_deref()
            .expect("prepared request must have prompt token ids")
    }
}

pub(crate) async fn collect_encode_output(
    prompt_token_ids: Vec<u32>,
    mut stream: EngineCoreOutputStream,
    mut request_metrics: PoolingRequestMetricsTracker,
) -> Result<EncodeOutput> {
    let mut pooling_output: Option<WireTensor> = None;
    let mut cached_token_count = 0;

    loop {
        let raw = match stream.next().await {
            Some(Ok(raw)) => raw,
            Some(Err(error)) => {
                request_metrics.record_finished(current_unix_timestamp_secs(), FinishReason::Error);
                return Err(error.into());
            }
            None => {
                unreachable!("engine-core stream closes only after a final output or an error")
            }
        };
        request_metrics.observe_output(raw.timestamp, &raw.output);
        let output = raw.output;
        cached_token_count = cached_token_count.max(
            output
                .prefill_stats
                .as_ref()
                .map_or(0, |stats| stats.num_cached_tokens as usize),
        );

        if let Some(tensor) = output.pooling_output
            && pooling_output.replace(tensor).is_some()
        {
            request_metrics.record_finished(current_unix_timestamp_secs(), FinishReason::Error);
            return Err(Error::PoolingRequest {
                request_id: output.request_id,
                message: "received more than one pooling output".to_string(),
            });
        }

        if let Some(finish_reason) = output.finish_reason {
            let received_at = current_unix_timestamp_secs();
            let semantic_finish_reason =
                FinishReason::from_engine(finish_reason, output.stop_reason);
            if finish_reason == EngineCoreFinishReason::Error {
                request_metrics.record_finished(received_at, FinishReason::Error);
                return Err(Error::PoolingRequest {
                    request_id: output.request_id,
                    message: "engine-core reported an error".to_string(),
                });
            }
            let Some(tensor) = pooling_output else {
                request_metrics.record_finished(received_at, semantic_finish_reason);
                return Err(Error::PoolingRequest {
                    request_id: output.request_id,
                    message: "request finished without a pooling output".to_string(),
                });
            };
            let pooling_output = match PoolingOutput::try_from(tensor) {
                Ok(output) => output,
                Err(message) => {
                    request_metrics.record_finished(received_at, FinishReason::Error);
                    return Err(Error::PoolingRequest {
                        request_id: output.request_id,
                        message: format!("failed to decode pooling output: {message}"),
                    });
                }
            };
            request_metrics.record_finished(received_at, semantic_finish_reason);
            return Ok(EncodeOutput {
                request_id: output.request_id,
                prompt_token_ids,
                output: pooling_output,
                cached_token_count,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_request() -> EncodeRequest {
        EncodeRequest {
            request_id: "embed-1".to_string(),
            prompt_token_ids: vec![11, 22],
            task: PoolingTask::Embed,
            pooling_params: PoolingParams {
                use_activation: false,
                dimensions: Some(128),
                step_tag_id: None,
                returned_token_ids: None,
            },
            arrival_time: Some(42.5),
            cache_salt: Some("salt".to_string()),
            trace_headers: Some(BTreeMap::from([(
                "x-trace-id".to_string(),
                "abc".to_string(),
            )])),
            priority: 3,
            data_parallel_rank: Some(2),
            session_id: Some("session-1".to_string()),
            lora_request: None,
        }
    }

    #[test]
    fn prepare_lowers_normalized_pooling_request() {
        let prepared = sample_request().prepare(false).unwrap();
        let request = prepared.engine_request;

        assert_eq!(request.request_id, "embed-1");
        assert_eq!(request.external_req_id.as_deref(), Some("embed-1"));
        assert!(request.sampling_params.is_none());
        assert_eq!(
            request.pooling_params,
            Some(EngineCorePoolingParams {
                use_activation: false,
                dimensions: Some(128),
                step_tag_id: None,
                returned_token_ids: None,
                task: PoolingTask::Embed,
            })
        );
        assert_eq!(request.arrival_time, 42.5);
        assert_eq!(request.cache_salt.as_deref(), Some("salt"));
        assert_eq!(request.data_parallel_rank, Some(2));
        assert_eq!(request.session_id.as_deref(), Some("session-1"));
    }

    #[test]
    fn prepare_rejects_empty_prompt_tokens() {
        let mut request = sample_request();
        request.prompt_token_ids.clear();

        assert!(matches!(
            request.prepare(true).unwrap_err(),
            Error::EmptyPromptTokenIds { request_id } if request_id == "embed-1"
        ));
    }
}
