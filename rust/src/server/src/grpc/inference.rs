// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use asynk_strim_attr::{TryYielder, try_stream};
use futures::{Stream, StreamExt as _, pin_mut};
use thiserror_ext::AsReport as _;
use tonic::{Request, Response, Status};
use tracing::{Span, info, info_span, warn};
use tracing_futures::Instrument as _;
use uuid::Uuid;
use vllm_engine_core_client::protocol::multimodal::MmModality;
use vllm_engine_core_client::protocol::tensor::{WireArrayData, WireTensor};
use vllm_llm::current_unix_timestamp_secs;
use vllm_text::{
    DecodedTextEvent, Prompt, SampledDelta, TextOutputStream, TextOutputStreamExt as _, TextRequest,
};

use super::convert::{self, ResponseOpts};
use super::{InferenceServer, pb};
use crate::state::AppState;

pub(crate) type InferenceGrpcService = InferenceServer<InferenceServiceImpl>;

const DATA_PARALLEL_RANK_METADATA_KEY: &str = "x-data-parallel-rank";

/// gRPC inference service backed by the shared application state.
pub struct InferenceServiceImpl {
    state: Arc<AppState>,
}

struct PreparedGrpcRequest {
    text_request: TextRequest,
    request_span: Span,
    started_at: Instant,
}

/// Keep producer metadata for matching EC items; leave unmatched inputs intact.
fn apply_encoder_cache_placeholders(text_request: &mut TextRequest) {
    let is_decode_kv_consumer = text_request
        .sampling_params
        .vllm_xargs
        .as_ref()
        .and_then(|args| args.get("kv_transfer_params"))
        .and_then(|params| params.get("do_remote_prefill"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let ec_items = text_request
        .sampling_params
        .vllm_xargs
        .as_ref()
        .and_then(|args| args.get("ec_transfer_params"))
        .and_then(|params| params.get("ec_items"))
        .and_then(serde_json::Value::as_array)
        .cloned();

    // Decode uses EC metadata only to prepare the prompt; EngineCore consumes KV.
    if is_decode_kv_consumer && let Some(args) = text_request.sampling_params.vllm_xargs.as_mut() {
        args.remove("ec_transfer_params");
    }

    let Some(ec_items) = ec_items else {
        return;
    };
    let Some(features) = text_request.mm_features.as_mut() else {
        return;
    };
    let ec_items_by_hash: std::collections::HashMap<_, _> = ec_items
        .iter()
        .filter_map(serde_json::Value::as_object)
        .filter_map(|item| {
            item.get("mm_hash")
                .and_then(serde_json::Value::as_str)
                .map(|mm_hash| (mm_hash, item))
        })
        .collect();

    for feature in features.iter_mut() {
        let Some(item) = ec_items_by_hash.get(feature.identifier.as_str()) else {
            continue;
        };
        let Some(data) = feature.data.as_mut() else {
            continue;
        };
        let metadata_keys: Vec<_> = data
            .keys()
            .filter(|key| key.as_str() != "mm_hash" && item.contains_key(key.as_str()))
            .cloned()
            .collect();
        if metadata_keys.is_empty() {
            continue;
        }
        data.retain(|key, _| metadata_keys.contains(key));
    }
}

impl InferenceServiceImpl {
    pub fn new(state: Arc<AppState>) -> Self {
        Self { state }
    }

    async fn prepare_request(
        &self,
        mut proto_request: pb::GenerateRequest,
        data_parallel_rank: Option<u32>,
        stream: bool,
        rpc: &'static str,
    ) -> Result<PreparedGrpcRequest, Status> {
        let started_at = Instant::now();
        let arrival_time = current_unix_timestamp_secs();
        if proto_request.request_id.is_empty() {
            proto_request.request_id = Uuid::new_v4().to_string();
        }
        let request_id = proto_request.request_id.clone();
        let model = if proto_request.model.is_empty() {
            self.state.primary_model_name().to_string()
        } else {
            proto_request.model.clone()
        };
        let lora_name = proto_request.lora_name.clone();
        let media_count = proto_request.media.len();
        let request_span = info_span!(
            "grpc_inference",
            %request_id,
            rpc,
            %model,
            %lora_name,
            media_count,
            ?data_parallel_rank,
        );
        info!(parent: &request_span, "gRPC inference request received");

        let result = async {
            let media = std::mem::take(&mut proto_request.media);
            let media_options = super::media::options_from_proto(
                proto_request.mm_processor_kwargs.take(),
                proto_request.media_io_kwargs.take(),
            );
            if !self.state.api_server_options.trust_request_mm_kwargs
                && (!media_options.processor_kwargs.is_empty()
                    || !media_options.media_io_kwargs.is_empty())
            {
                return Err(Status::invalid_argument(
                    "per-request mm_processor_kwargs and media_io_kwargs are disabled; start \
                     the server with --trust-request-mm-kwargs",
                ));
            }
            let mut text_request =
                convert::to_text_request(proto_request, stream, &self.state.base_model_names())?;
            text_request.arrival_time = Some(arrival_time);
            text_request.data_parallel_rank = data_parallel_rank;

            if !lora_name.is_empty() {
                if !self.state.llm().info().supports_lora {
                    return Err(Status::failed_precondition(
                        "engine was not started with LoRA enabled",
                    ));
                }
                let resolution = self.state.resolve_model_with_loras(Some(&lora_name)).await;
                text_request.lora_request = Some(resolution.lora_request.ok_or_else(|| {
                    Status::not_found(format!("LoRA adapter `{lora_name}` is not loaded"))
                })?);
            }

            let ec_transfer_present = text_request
                .sampling_params
                .vllm_xargs
                .as_ref()
                .and_then(|args| args.get("ec_transfer_params"))
                .is_some();
            let media = super::media::from_proto(media, ec_transfer_present)?;
            if !media.is_empty() {
                let Prompt::TokenIds(mut token_ids) = text_request.prompt else {
                    return Err(Status::invalid_argument(
                        "multimodal gRPC requests must provide token_ids input",
                    ));
                };
                let mm_features = self
                    .state
                    .chat
                    .prepare_media_with_options(media, &media_options, &mut token_ids)
                    .await
                    .map_err(|error| {
                        if error.is_request_validation_error() {
                            Status::invalid_argument(error.to_report_string())
                        } else {
                            Status::internal(error.to_report_string())
                        }
                    })?;
                text_request.prompt = Prompt::TokenIds(token_ids);
                text_request.mm_features = mm_features;
            }
            apply_encoder_cache_placeholders(&mut text_request);

            Ok(text_request)
        }
        .instrument(request_span.clone())
        .await;

        let text_request = match result {
            Ok(text_request) => text_request,
            Err(status) => {
                warn!(
                    parent: &request_span,
                    grpc_code = ?status.code(),
                    elapsed_ms = started_at.elapsed().as_millis() as u64,
                    error = %status.message(),
                    "gRPC inference request preparation failed"
                );
                return Err(status);
            }
        };

        Ok(PreparedGrpcRequest {
            text_request,
            request_span,
            started_at,
        })
    }
}

fn prompt_placeholders(text_request: &TextRequest) -> Vec<pb::PromptPlaceholder> {
    text_request
        .mm_features
        .iter()
        .flatten()
        .map(|feature| pb::PromptPlaceholder {
            modality: match feature.modality {
                MmModality::Image => pb::Modality::Image,
                MmModality::Audio => pb::Modality::Audio,
                MmModality::Video => pb::Modality::Video,
            } as i32,
            mm_hash: feature
                .mm_hash
                .clone()
                .filter(|hash| !hash.is_empty())
                .unwrap_or_else(|| feature.identifier.clone()),
            offset: u64::try_from(feature.mm_position.offset).unwrap_or(u64::MAX),
            length: u64::try_from(feature.mm_position.length).unwrap_or(u64::MAX),
            is_embed: is_embed_mask(feature.mm_position.is_embed.as_ref()),
            identifier: feature.identifier.clone(),
        })
        .collect()
}

/// An all-true mask is sent as empty, which the consumer reads as "every
/// position takes an embedding".
fn is_embed_mask(mask: Option<&WireTensor>) -> Vec<bool> {
    let Some(WireArrayData::RawView(bytes)) = mask.map(|mask| &mask.data) else {
        return Vec::new();
    };
    if bytes.iter().all(|&byte| byte != 0) {
        return Vec::new();
    }
    bytes.iter().map(|&byte| byte != 0).collect()
}

#[cfg(test)]
mod placeholder_tests {
    use super::*;

    #[test]
    fn all_true_masks_are_sent_empty_and_partial_masks_verbatim() {
        let all = WireTensor::from_bool(vec![2], vec![true, true]).unwrap();
        assert!(is_embed_mask(Some(&all)).is_empty());
        let partial = WireTensor::from_bool(vec![3], vec![false, true, false]).unwrap();
        assert_eq!(is_embed_mask(Some(&partial)), [false, true, false]);
        assert!(is_embed_mask(None).is_empty());
    }
}

fn data_parallel_rank_from_metadata(
    request: &Request<pb::GenerateRequest>,
) -> Result<Option<u32>, Status> {
    let Some(value) = request.metadata().get(DATA_PARALLEL_RANK_METADATA_KEY) else {
        return Ok(None);
    };
    let value = value.to_str().map_err(|_| {
        Status::invalid_argument("x-data-parallel-rank metadata must be an unsigned 32-bit integer")
    })?;
    value.trim().parse::<u32>().map(Some).map_err(|_| {
        Status::invalid_argument("x-data-parallel-rank metadata must be an unsigned 32-bit integer")
    })
}

#[tonic::async_trait]
impl pb::inference_server::Inference for InferenceServiceImpl {
    type GenerateStreamStream =
        Pin<Box<dyn Stream<Item = Result<pb::GenerateResponse, Status>> + Send>>;

    /// Unary generate: collect all output and return a single response.
    async fn generate(
        &self,
        request: Request<pb::GenerateRequest>,
    ) -> Result<Response<pb::GenerateResponse>, Status> {
        let data_parallel_rank = data_parallel_rank_from_metadata(&request)?;
        let proto_req = request.into_inner();
        let response_opts = ResponseOpts::from_proto(proto_req.response.as_ref());
        let PreparedGrpcRequest {
            text_request,
            request_span,
            started_at,
        } = self.prepare_request(proto_req, data_parallel_rank, false, "Generate").await?;

        let mm_placeholders = prompt_placeholders(&text_request);
        let stream = self
            .state
            .chat
            .text()
            .generate(text_request)
            .instrument(request_span.clone())
            .await
            .map_err(|error| log_text_error(&request_span, started_at, "submission", error))?;

        let collected = stream
            .collect_output()
            .instrument(request_span.clone())
            .await
            .map_err(|error| log_text_error(&request_span, started_at, "collection", error))?;
        // Build the single aggregated response.
        let mut prompt_info = convert::to_prompt_info(
            &collected.prompt_token_ids,
            collected.prompt_logprobs.as_ref(),
            &response_opts,
        );
        prompt_info.mm_placeholders = mm_placeholders;

        let finish_info = vllm_text::Finished {
            usage: collected.usage,
            finish_reason: collected.finish_reason,
            kv_transfer_params: collected.kv_transfer_params.map(Box::new),
            ec_transfer_params: collected.ec_transfer_params.map(Box::new),
            sampling_mask: collected.sampling_mask,
            spec_decode_metrics: collected.spec_decode_metrics,
            weight_versions: collected.weight_versions,
            timestamps: collected.timestamps,
        };

        let outputs = convert::to_sequence_output(
            &collected.text,
            &collected.token_ids,
            collected.logprobs.as_ref(),
            Some(&finish_info),
            &response_opts,
        )?;
        info!(
            parent: &request_span,
            elapsed_ms = started_at.elapsed().as_millis() as u64,
            "gRPC inference request completed"
        );

        Ok(Response::new(pb::GenerateResponse {
            prompt_info: Some(prompt_info),
            outputs: Some(outputs),
        }))
    }

    /// Streaming generate: yield incremental responses as tokens are produced.
    async fn generate_stream(
        &self,
        request: Request<pb::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStreamStream>, Status> {
        let data_parallel_rank = data_parallel_rank_from_metadata(&request)?;
        let proto_req = request.into_inner();
        let response_opts = ResponseOpts::from_proto(proto_req.response.as_ref());
        let PreparedGrpcRequest {
            text_request,
            request_span,
            started_at,
        } = self
            .prepare_request(proto_req, data_parallel_rank, true, "GenerateStream")
            .await?;

        let mm_placeholders = prompt_placeholders(&text_request);
        let stream = self
            .state
            .chat
            .text()
            .generate(text_request)
            .instrument(request_span.clone())
            .await
            .map_err(|error| log_text_error(&request_span, started_at, "submission", error))?;

        let response_stream = generate_stream_responses(
            stream,
            response_opts,
            request_span.clone(),
            started_at,
            mm_placeholders,
        )
        .instrument(request_span);
        Ok(Response::new(Box::pin(response_stream)))
    }
}

/// Convert one decoded text stream into gRPC streaming responses.
///
/// The response body owns the generation stream, so a client disconnect drops
/// it and aborts the request in the engine, even before any output arrives.
#[try_stream]
async fn generate_stream_responses(
    stream: impl TextOutputStream,
    response_opts: ResponseOpts,
    request_span: Span,
    started_at: Instant,
    mm_placeholders: Vec<pb::PromptPlaceholder>,
    mut y: TryYielder<pb::GenerateResponse, Status>,
) -> Result<(), Status> {
    pin_mut!(stream);

    while let Some(event) = stream.next().await {
        let response = match event {
            Err(error) => Err(log_text_error(&request_span, started_at, "stream", error)),
            Ok(DecodedTextEvent::Start {
                prompt_token_ids,
                prompt_logprobs,
            }) => {
                let mut prompt_info = convert::to_prompt_info(
                    &prompt_token_ids,
                    prompt_logprobs.as_ref(),
                    &response_opts,
                );
                prompt_info.mm_placeholders = mm_placeholders.clone();
                Ok(pb::GenerateResponse {
                    prompt_info: Some(prompt_info),
                    outputs: None,
                })
            }
            Ok(DecodedTextEvent::TextDelta {
                decoded,
                sampled:
                    SampledDelta {
                        token_ids,
                        logprobs,
                    },
                finished,
            }) => convert::to_sequence_output(
                &decoded.text,
                &token_ids,
                logprobs.as_ref(),
                finished.as_deref(),
                &response_opts,
            )
            .map(|outputs| pb::GenerateResponse {
                prompt_info: None,
                outputs: Some(outputs),
            }),
        };
        y.yield_ok(response?).await;
    }
    info!(
        parent: &request_span,
        elapsed_ms = started_at.elapsed().as_millis() as u64,
        "gRPC inference stream closed"
    );
    Ok(())
}

fn text_error_to_status(error: vllm_text::Error) -> Status {
    let message = error.to_report_string();
    if error.is_request_validation_error() {
        Status::invalid_argument(message)
    } else {
        Status::internal(message)
    }
}

fn log_text_error(
    request_span: &Span,
    started_at: Instant,
    phase: &'static str,
    error: vllm_text::Error,
) -> Status {
    let status = text_error_to_status(error);
    warn!(
        parent: request_span,
        phase,
        grpc_code = ?status.code(),
        elapsed_ms = started_at.elapsed().as_millis() as u64,
        error = %status.message(),
        "gRPC inference request failed"
    );
    status
}
