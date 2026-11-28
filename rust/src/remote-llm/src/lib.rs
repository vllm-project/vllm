// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! [`Llm`] over a remote `vllm.Inference` gRPC endpoint.

mod convert;

use std::num::NonZeroU32;

use anyhow::Context as _;
use asynk_strim_attr::{TryYielder, try_stream};
use futures::{Stream, StreamExt, TryStreamExt};
use thiserror_ext::AsReport as _;
use tokio::sync::watch;
use tokio_util::task::AbortOnDropHandle;
use tonic::{
    Code, Request, Status,
    transport::{Channel, Endpoint},
};
use uuid::Uuid;
use vllm_engine_core_client::protocol::output::StopReason;
use vllm_llm::{
    BoxGenerateOutputStream, Error, FinishReason, GenerateOutput, GeneratePromptInfo,
    GenerateRequest, Llm, LlmInfo, PromptCacheUsage, Result, WeightVersions,
};
use vllm_text::backend::TextBackend;

use vllm_proto as pb;

use self::pb::inference_client::InferenceClient;

/// Token-only client for a vllm.Inference endpoint.
pub struct GrpcLlm {
    readiness: watch::Receiver<bool>,
    _readiness_task: AbortOnDropHandle<()>,
    channel: Channel,
    model: String,
    info: LlmInfo,
    health: watch::Sender<bool>,
}

impl GrpcLlm {
    pub fn new(channel: Channel, model: String, max_model_len: u32) -> Self {
        let info = LlmInfo {
            max_model_len,
            model_dtype: None,
            supports_lora: false,
            allow_missing_mm_embeddings: false,
            vllm_version: "unknown".to_owned(),
            fingerprint: None,
        };
        let (ready, readiness) = watch::channel(false);
        let mut health = tonic_health::pb::health_client::HealthClient::new(channel.clone());
        let readiness_task = tokio::spawn(async move {
            loop {
                let mut request = Request::new(tonic_health::pb::HealthCheckRequest {
                    service: "vllm.Inference".to_owned(),
                });
                request.set_timeout(std::time::Duration::from_secs(1));
                let result =
                    tokio::time::timeout(std::time::Duration::from_secs(1), health.check(request))
                        .await;
                let serving = matches!(result, Ok(Ok(reply)) if reply.get_ref().status == tonic_health::ServingStatus::Serving as i32);
                ready.send_if_modified(|current| std::mem::replace(current, serving) != serving);
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            }
        });
        Self {
            channel,
            readiness,
            _readiness_task: AbortOnDropHandle::new(readiness_task),
            model,
            info,
            health: watch::channel(true).0,
        }
    }
}

#[tonic::async_trait]
impl Llm for GrpcLlm {
    fn info(&self) -> &LlmInfo {
        &self.info
    }
    fn subscribe_health(&self) -> watch::Receiver<bool> {
        self.health.subscribe()
    }
    fn subscribe_readiness(&self) -> watch::Receiver<bool> {
        self.readiness.clone()
    }
    async fn generate_n(
        &self,
        mut request: GenerateRequest,
        n: NonZeroU32,
    ) -> Result<Vec<BoxGenerateOutputStream>> {
        if n.get() != 1 {
            return Err(Error::Unsupported { field: "n > 1" });
        }
        reject_unsupported_fields(&request)?;
        let external_request_id = std::mem::take(&mut request.request_id);
        let rpc_request_id = Uuid::new_v4();
        let grpc_request = convert::to_grpc_request(&self.model, rpc_request_id, request)?;
        let mut client = InferenceClient::new(self.channel.clone());
        let response = client.generate_stream(grpc_request).await.map_err(status_to_llm_error)?;
        let frames = response.into_inner().map_err(status_to_llm_error);
        Ok(vec![
            generate_outputs_from_proto(external_request_id, frames).boxed(),
        ])
    }
    async fn abort(&self, _external_ids: &[String]) -> Result<()> {
        Err(Error::Unsupported {
            field: "abort by request ID",
        })
    }
    async fn shutdown(self: Box<Self>) -> Result<()> {
        Ok(())
    }
}

/// Settings for serving generation through a remote vLLM gRPC endpoint.
pub struct RemoteGrpcConfig {
    endpoint: Endpoint,
    max_model_len: Option<NonZeroU32>,
}

impl RemoteGrpcConfig {
    /// Parse the worker URL and record an optional context limit override.
    pub fn new(url: String, max_model_len: Option<NonZeroU32>) -> anyhow::Result<Self> {
        Ok(Self {
            endpoint: Endpoint::from_shared(url)?,
            max_model_len,
        })
    }

    pub fn into_llm(
        self,
        model: String,
        text_backend: &dyn TextBackend,
    ) -> anyhow::Result<GrpcLlm> {
        let max_model_len = if let Some(limit) = self.max_model_len {
            limit.get()
        } else {
            let limit = text_backend
                .max_model_len()?
                .context("config.json has no context length; pass --max-model-len")?;
            anyhow::ensure!(
                limit > 0,
                "invalid context length in config.json: expected a positive token count"
            );
            limit
        };
        Ok(GrpcLlm::new(
            self.endpoint.connect_lazy(),
            model,
            max_model_len,
        ))
    }
}

fn reject_unsupported_fields(request: &GenerateRequest) -> Result<()> {
    if request.mm_features.is_some() {
        return Err(Error::Unsupported {
            field: "multimodal inputs",
        });
    }
    if request.lora_request.is_some() {
        return Err(Error::Unsupported {
            field: "lora_request",
        });
    }
    if request.kv_hints.is_some() {
        return Err(Error::Unsupported { field: "kv_hints" });
    }
    if request.data_parallel_rank.is_some() {
        return Err(Error::Unsupported {
            field: "data_parallel_rank",
        });
    }

    let sampling_params = &request.sampling_params;
    if sampling_params.extra_args.is_some() {
        return Err(Error::Unsupported {
            field: "extra_args",
        });
    }
    if sampling_params.structured_outputs.is_some() {
        return Err(Error::Unsupported {
            field: "structured_outputs",
        });
    }
    if sampling_params.logprobs.is_some() || sampling_params.logprob_token_ids.is_some() {
        return Err(Error::Unsupported { field: "logprobs" });
    }
    if sampling_params.prompt_logprobs.is_some()
        || sampling_params.prompt_logprob_token_ids.is_some()
    {
        return Err(Error::Unsupported {
            field: "prompt_logprobs",
        });
    }
    if sampling_params.repetition_detection.is_some() {
        return Err(Error::Unsupported {
            field: "repetition_detection",
        });
    }
    if sampling_params.stream_interval.is_some() {
        return Err(Error::Unsupported {
            field: "stream_interval",
        });
    }
    if sampling_params.routed_experts_prompt_start != 0 {
        return Err(Error::Unsupported {
            field: "routed_experts_prompt_start",
        });
    }

    Ok(())
}

#[try_stream]
async fn generate_outputs_from_proto(
    request_id: String,
    frames: impl Stream<Item = Result<pb::GenerateResponse>> + Send,
    mut y: TryYielder<GenerateOutput, Error>,
) -> Result<()> {
    futures::pin_mut!(frames);
    let first = frames.try_next().await?.ok_or_else(|| remote_error("missing prompt frame"))?;
    let prompt = first.prompt_info.ok_or_else(|| remote_error("missing prompt info"))?;
    if first.outputs.is_some() || prompt.token_ids.len() != prompt.num_prompt_tokens as usize {
        return Err(remote_error("invalid prompt frame"));
    }
    let mut prompt = Some(Box::new(GeneratePromptInfo {
        prompt_token_ids: prompt.token_ids.into(),
        prompt_logprobs: None,
        prompt_token_id_logprobs: None,
    }));
    let mut output_tokens = 0u64;
    while let Some(frame) = frames.try_next().await? {
        let out = frame.outputs.ok_or_else(|| remote_error("missing sequence output"))?;
        if frame.prompt_info.is_some()
            || out.index != 0
            || out.token_ids.len() != out.num_tokens as usize
        {
            return Err(remote_error("invalid token delta"));
        }
        output_tokens += u64::from(out.num_tokens);
        let terminal = out.finish_info.as_ref();
        let finish_reason = terminal.map(finish_reason_from_proto).transpose()?;
        // A worker can send a terminal message followed by a non-OK RPC status.
        if let Some(info) = terminal {
            if frames.try_next().await?.is_some() {
                return Err(remote_error("output after finish"));
            }
            if u64::from(info.num_output_tokens) != output_tokens {
                return Err(remote_error(format!(
                    "worker reported {} output tokens, streamed token count was {output_tokens}",
                    info.num_output_tokens
                )));
            }
        }
        y.yield_ok(GenerateOutput {
            request_id: request_id.clone(),
            prompt_info: prompt.take(),
            weight_versions: WeightVersions::new(None, out.token_ids.len()),
            token_ids: out.token_ids,
            logprobs: None,
            finish_reason,
            prompt_cache_usage: PromptCacheUsage {
                cached_token_count: terminal.and_then(|f| f.num_cached_tokens).unwrap_or_default()
                    as usize,
                ..Default::default()
            },
            kv_transfer_params: None,
            ec_transfer_params: None,
            sampling_mask: None,
            spec_decode_metrics: None,
            routed_experts: Default::default(),
            timestamps: Default::default(),
        })
        .await;
        if terminal.is_some() {
            return Ok(());
        }
    }
    Err(remote_error("generation ended without finish info"))
}

fn finish_reason_from_proto(info: &pb::FinishInfo) -> Result<FinishReason> {
    use pb::finish_info::{FinishReason as PbFinishReason, StopReason as PbStopReason};
    match PbFinishReason::try_from(info.finish_reason) {
        Ok(PbFinishReason::Length) => Ok(FinishReason::Length),
        Ok(PbFinishReason::Aborted) => Ok(FinishReason::Abort),
        Ok(PbFinishReason::Stop) => Ok(FinishReason::Stop(match &info.stop_reason {
            Some(PbStopReason::StopTokenId(id)) => Some(StopReason::TokenId(*id)),
            Some(PbStopReason::StopString(s)) => Some(StopReason::Text(s.clone())),
            _ => None,
        })),
        _ => Err(remote_error("invalid worker finish reason")),
    }
}

fn remote_error(message: impl Into<String>) -> Error {
    Error::Remote {
        message: message.into(),
    }
}

fn status_to_llm_error(status: Status) -> Error {
    let message = status.to_report_string();
    match status.code() {
        Code::InvalidArgument | Code::OutOfRange | Code::FailedPrecondition => {
            Error::RemoteInvalidRequest { message }
        }
        Code::Unavailable | Code::ResourceExhausted => Error::Unavailable { message },
        _ => remote_error(message),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn terminal_frames(num_output_tokens: u32) -> [Result<pb::GenerateResponse>; 2] {
        [
            Ok(pb::GenerateResponse {
                prompt_info: Some(pb::PromptInfo::default()),
                outputs: None,
            }),
            Ok(pb::GenerateResponse {
                prompt_info: None,
                outputs: Some(pb::SequenceOutput {
                    token_ids: vec![17],
                    num_tokens: 1,
                    finish_info: Some(pb::FinishInfo {
                        finish_reason: pb::finish_info::FinishReason::Length as i32,
                        num_output_tokens,
                        ..Default::default()
                    }),
                    ..Default::default()
                }),
            }),
        ]
    }

    #[tokio::test]
    async fn terminal_count_must_match_streamed_tokens() {
        let frames = futures::stream::iter(terminal_frames(2));
        let mut decoded = generate_outputs_from_proto("test".into(), frames).boxed();
        assert!(
            matches!(decoded.next().await, Some(Err(Error::Remote { message }))
            if message.contains("token count"))
        );
    }

    #[tokio::test]
    async fn terminal_frame_requires_successful_rpc_status() {
        let frames = futures::stream::iter(terminal_frames(1).into_iter().chain([Err(
            status_to_llm_error(Status::unavailable("worker stopped")),
        )]));
        let mut decoded = generate_outputs_from_proto("test".into(), frames).boxed();
        let error = decoded
            .next()
            .await
            .expect("status")
            .expect_err("terminal frame must wait for status");
        assert!(
            matches!(error, Error::Unavailable { message } if message.contains("worker stopped"))
        );
    }
}
