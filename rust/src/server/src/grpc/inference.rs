// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use futures::{Stream, StreamExt as _};
use thiserror_ext::AsReport as _;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};
use tracing::{Span, info, info_span, warn};
use tracing_futures::Instrument as _;
use uuid::Uuid;
use vllm_llm::current_unix_timestamp_secs;
use vllm_text::{DecodedTextEvent, Prompt, SampledDelta, TextOutputStreamExt as _, TextRequest};

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
            let mut text_request =
                convert::to_text_request(proto_request, stream, self.state.served_model_names())?;
            text_request.arrival_time = Some(arrival_time);
            text_request.data_parallel_rank = data_parallel_rank;

            if !lora_name.is_empty() {
                if !self.state.engine_core_client().ready_response().supports_lora {
                    return Err(Status::failed_precondition(
                        "engine was not started with LoRA enabled",
                    ));
                }
                let resolution = self.state.resolve_model_with_loras(Some(&lora_name)).await;
                text_request.lora_request = Some(resolution.lora_request.ok_or_else(|| {
                    Status::not_found(format!("LoRA adapter `{lora_name}` is not loaded"))
                })?);
            }

            let media = convert::media_parts_from_request(media)?;
            if !media.is_empty() {
                let Prompt::TokenIds(mut token_ids) = text_request.prompt else {
                    return Err(Status::invalid_argument(
                        "multimodal gRPC requests must provide token_ids input",
                    ));
                };
                let mm_features = self
                    .state
                    .chat
                    .prepare_media(media, &mut token_ids)
                    .await
                    .map_err(|error| Status::internal(error.to_report_string()))?;
                text_request.prompt = Prompt::TokenIds(token_ids);
                text_request.mm_features = mm_features;
            }

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
        info!(
            parent: &request_span,
            elapsed_ms = started_at.elapsed().as_millis() as u64,
            "gRPC inference request completed"
        );

        // Build the single aggregated response.
        let prompt_info = convert::to_prompt_info(
            &collected.prompt_token_ids,
            collected.prompt_logprobs.as_ref(),
            &response_opts,
        );

        let finish_info = vllm_text::Finished {
            usage: collected.usage,
            finish_reason: collected.finish_reason,
            kv_transfer_params: collected.kv_transfer_params,
            ec_transfer_params: collected.ec_transfer_params,
        };

        let outputs = convert::to_sequence_output(
            &collected.text,
            &collected.token_ids,
            collected.logprobs.as_ref(),
            Some(&finish_info),
            &response_opts,
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

        let stream = self
            .state
            .chat
            .text()
            .generate(text_request)
            .instrument(request_span.clone())
            .await
            .map_err(|error| log_text_error(&request_span, started_at, "submission", error))?;

        let (tx, rx) = mpsc::channel(32);

        let task_span = request_span.clone();
        tokio::spawn(
            async move {
                futures::pin_mut!(stream);
                while let Some(event) = stream.next().await {
                    let response = match event {
                        Err(error) => Err(log_text_error(&task_span, started_at, "stream", error)),
                        Ok(DecodedTextEvent::Start {
                            prompt_token_ids,
                            prompt_logprobs,
                        }) => {
                            let prompt_info = convert::to_prompt_info(
                                &prompt_token_ids,
                                prompt_logprobs.as_ref(),
                                &response_opts,
                            );
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
                        }) => Ok(pb::GenerateResponse {
                            prompt_info: None,
                            outputs: Some(convert::to_sequence_output(
                                &decoded.text,
                                &token_ids,
                                logprobs.as_ref(),
                                finished.as_deref(),
                                &response_opts,
                            )),
                        }),
                    };

                    if tx.send(response).await.is_err() {
                        break;
                    }
                }
                info!(
                    parent: &task_span,
                    elapsed_ms = started_at.elapsed().as_millis() as u64,
                    "gRPC inference stream closed"
                );
            }
            .instrument(request_span),
        );

        let response_stream = ReceiverStream::new(rx);
        Ok(Response::new(Box::pin(response_stream)))
    }
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
