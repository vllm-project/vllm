// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use asynk_strim_attr::{TryYielder, try_stream};
use futures::{Stream, StreamExt, TryStreamExt};
use std::{
    collections::HashMap,
    num::NonZeroU32,
    sync::{Arc, Mutex},
};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use tonic::{
    Code, Request, Status, Streaming,
    transport::{Channel, Endpoint},
};
use vllm_engine_core_client::protocol::output::StopReason;
use vllm_llm::{
    BoxGenerateOutputStream, Error, FinishReason, GenerateOutput, GeneratePromptInfo,
    GenerateRequest, Llm, LlmInfo, Result,
};
use vllm_proto::{self as pb, inference_client::InferenceClient};

use tokio_util::task::AbortOnDropHandle;
use uuid::Uuid;

struct InflightRequest {
    external_request_id: String,
    cancel: CancellationToken,
    response: Mutex<Option<Streaming<pb::GenerateResponse>>>,
}

impl InflightRequest {
    fn abort(&self) {
        self.cancel.cancel();
        self.response.lock().unwrap().take();
    }

    async fn next_response(&self) -> Result<Option<pb::GenerateResponse>> {
        let message = futures::future::poll_fn(|cx| {
            let mut response = self.response.lock().unwrap();
            match response.as_mut() {
                Some(stream) => stream.poll_next_unpin(cx),
                None => std::task::Poll::Ready(None),
            }
        });
        tokio::select! {
            biased;
            _ = self.cancel.cancelled() => Err(remote_error("request aborted")),
            result = message => result.transpose().map_err(status_to_llm_error),
        }
    }
}

type InflightRequests = Arc<Mutex<HashMap<Uuid, Arc<InflightRequest>>>>;

/// Token-only client for a vllm.Inference endpoint.
pub struct GrpcLlm {
    readiness: watch::Receiver<bool>,
    _readiness_task: AbortOnDropHandle<()>,
    channel: Channel,
    model: String,
    info: LlmInfo,
    health: watch::Sender<bool>,
    inflight: InflightRequests,
}

impl GrpcLlm {
    /// Connect lazily to a worker using model information supplied by the caller.
    pub fn new(url: &str, model: String, info: LlmInfo) -> anyhow::Result<Self> {
        let channel = Endpoint::from_shared(url.to_owned())?.connect_lazy();
        Ok(Self::from_channel(channel, model, info))
    }

    pub(super) fn from_channel(channel: Channel, model: String, info: LlmInfo) -> Self {
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
            inflight: Default::default(),
        }
    }
}

fn remote_error(error: impl ToString) -> Error {
    Error::Remote {
        message: error.to_string(),
    }
}

fn status_to_llm_error(status: Status) -> Error {
    match status.code() {
        Code::InvalidArgument | Code::OutOfRange | Code::FailedPrecondition => {
            Error::RemoteInvalidRequest {
                message: status.to_string(),
            }
        }
        Code::Unavailable | Code::ResourceExhausted => Error::Unavailable {
            message: status.to_string(),
        },
        _ => remote_error(status),
    }
}

fn numbers_fit_protobuf_struct(value: &serde_json::Value) -> bool {
    use serde_json::Value;
    match value {
        // Struct stores numbers as f64. Compare integer values in i128 so even
        // a u64::MAX rounded up to 2^64 is detected.
        Value::Number(n) => n.as_i128().is_none_or(|i| i as f64 as i128 == i),
        Value::Array(items) => items.iter().all(numbers_fit_protobuf_struct),
        Value::Object(fields) => fields.values().all(numbers_fit_protobuf_struct),
        _ => true,
    }
}

fn to_grpc_request(
    model: &str,
    rpc_request_id: Uuid,
    request: GenerateRequest,
) -> Result<Request<pb::GenerateRequest>> {
    let sampling_params = request.sampling_params;
    for (field, present) in [
        ("multimodal inputs", request.mm_features.is_some()),
        ("lora_request", request.lora_request.is_some()),
        ("kv_hints", request.kv_hints.is_some()),
        ("data_parallel_rank", request.data_parallel_rank.is_some()),
        ("extra_args", sampling_params.extra_args.is_some()),
        (
            "structured_outputs",
            sampling_params.structured_outputs.is_some(),
        ),
        (
            "logprobs",
            sampling_params.logprobs.is_some() || sampling_params.logprob_token_ids.is_some(),
        ),
        (
            "prompt_logprobs",
            sampling_params.prompt_logprobs.is_some()
                || sampling_params.prompt_logprob_start.is_some()
                || sampling_params.prompt_logprob_token_ids.is_some(),
        ),
        (
            "repetition_detection",
            sampling_params.repetition_detection.is_some(),
        ),
        ("stream_interval", sampling_params.stream_interval.is_some()),
        (
            "routed_experts_prompt_start",
            sampling_params.routed_experts_prompt_start != 0,
        ),
    ] {
        if present {
            return Err(Error::Unsupported { field });
        }
    }
    let kwargs = request
        .reasoning_parser_kwargs
        .filter(|k| !k.chat_template_kwargs.is_empty())
        .map(|k| {
            let value = serde_json::Value::Object(k.chat_template_kwargs.into_iter().collect());
            if !numbers_fit_protobuf_struct(&value) {
                return Err(Error::Unsupported {
                    field: "chat_template_kwargs number outside protobuf Struct precision",
                });
            }
            Ok(super::convert::json_to_proto_struct(&value).expect("JSON object"))
        })
        .transpose()?;
    let mut grpc_request = Request::new(pb::GenerateRequest {
        model: model.into(),
        request_id: rpc_request_id.to_string(),
        prompt: Some(pb::generate_request::Prompt::TokenIds(pb::TokenIds {
            ids: request.prompt_token_ids,
        })),
        temperature: Some(sampling_params.temperature),
        sampling: Some(pb::RandomSampling {
            num_sequences: 1,
            top_p: Some(sampling_params.top_p),
            top_k: Some(sampling_params.top_k),
            min_p: Some(sampling_params.min_p),
            seed: sampling_params.seed,
        }),
        decoding: Some(pb::DecodingParameters {
            presence_penalty: sampling_params.presence_penalty,
            frequency_penalty: sampling_params.frequency_penalty,
            repetition_penalty: sampling_params.repetition_penalty,
            allowed_token_ids: sampling_params.allowed_token_ids.unwrap_or_default(),
            logit_bias: sampling_params.logit_bias.unwrap_or_default(),
            bad_words_token_ids: sampling_params
                .bad_words_token_ids
                .unwrap_or_default()
                .into_iter()
                .map(|ids| pb::TokenIds { ids })
                .collect(),
            ..Default::default()
        }),
        stopping: Some(pb::StoppingCriteria {
            max_new_tokens: sampling_params.max_tokens,
            min_new_tokens: sampling_params.min_tokens,
            stop_token_ids: sampling_params.stop_token_ids,
            ignore_eos: sampling_params.eos_token_id.is_none(),
            thinking_token_budget: sampling_params
                .thinking_token_budget
                .map(i64::try_from)
                .transpose()
                .map_err(remote_error)?,
            ..Default::default()
        }),
        response: Some(pb::ResponseOptions {
            output_text: Some(false),
            output_token_ids: true,
            prompt_token_ids: true,
            ..Default::default()
        }),
        priority: request.priority,
        session_id: request.session_id,
        watermarking: Some(sampling_params.watermarking),
        kv: Some(pb::KvCacheParameters {
            cache_salt: request.cache_salt.unwrap_or_default(),
            bypass_prefix_cache: sampling_params.skip_reading_prefix_cache.unwrap_or(false),
            ..Default::default()
        }),
        engine_reasoning_gate: (request.reasoning_ended.is_some() || kwargs.is_some()).then_some(
            pb::EngineReasoningGate {
                reasoning_ended: request.reasoning_ended,
                chat_template_kwargs: kwargs,
            },
        ),
        ..Default::default()
    });
    for (key, value) in request.trace_headers.into_iter().flatten() {
        let key =
            tonic::metadata::MetadataKey::<tonic::metadata::Ascii>::from_bytes(key.as_bytes())
                .map_err(remote_error)?;
        grpc_request.metadata_mut().insert(key, value.parse().map_err(remote_error)?);
    }
    Ok(grpc_request)
}

struct RequestGuard {
    rpc_request_id: Uuid,
    call: Arc<InflightRequest>,
    inflight: InflightRequests,
}
impl Drop for RequestGuard {
    fn drop(&mut self) {
        self.inflight.lock().unwrap().remove(&self.rpc_request_id);
    }
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
            weight_versions: vllm_llm::WeightVersions::new(None, out.token_ids.len()),
            token_ids: out.token_ids,
            logprobs: None,
            finish_reason,
            prompt_cache_usage: vllm_llm::PromptCacheUsage {
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
        let external_request_id = std::mem::take(&mut request.request_id);
        let rpc_request_id = Uuid::new_v4();
        let grpc_request = to_grpc_request(&self.model, rpc_request_id, request)?;
        let call = Arc::new(InflightRequest {
            external_request_id,
            cancel: CancellationToken::new(),
            response: Mutex::new(None),
        });
        self.inflight.lock().unwrap().insert(rpc_request_id, call.clone());
        let guard = RequestGuard {
            rpc_request_id,
            call,
            inflight: self.inflight.clone(),
        };
        let mut client = InferenceClient::new(self.channel.clone());
        let response = tokio::select! {
            _ = guard.call.cancel.cancelled() => return Err(remote_error("request aborted")),
            response = client.generate_stream(grpc_request) => response.map_err(status_to_llm_error)?,
        };
        {
            let mut stream = guard.call.response.lock().unwrap();
            if guard.call.cancel.is_cancelled() {
                return Err(remote_error("request aborted"));
            }
            *stream = Some(response.into_inner());
        }
        let external_request_id = guard.call.external_request_id.clone();
        let frames = futures::stream::try_unfold(guard, |guard| async move {
            let frame = guard.call.next_response().await?;
            Ok(frame.map(|frame| (frame, guard)))
        });
        Ok(vec![
            generate_outputs_from_proto(external_request_id, frames).boxed(),
        ])
    }
    async fn abort(&self, external_ids: &[String]) -> Result<()> {
        for call in self.inflight.lock().unwrap().values() {
            if external_ids.is_empty() || external_ids.contains(&call.external_request_id) {
                call.abort();
            }
        }
        Ok(())
    }
    async fn shutdown(self: Box<Self>) -> Result<()> {
        self.abort(&[]).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn template_kwargs_require_exact_numbers_in_protobuf_struct() {
        use serde_json::json;
        assert!(numbers_fit_protobuf_struct(
            &json!({"nested": [1.0, 0.25, 1u64 << 63]})
        ));
        for integer in [
            json!(9_007_199_254_740_993u64),
            json!(-9_007_199_254_740_993i64),
            json!(u64::MAX),
        ] {
            assert!(!numbers_fit_protobuf_struct(&json!({"nested": [integer]})));
        }
    }

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
