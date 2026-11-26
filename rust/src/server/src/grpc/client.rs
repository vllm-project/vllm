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

struct Call {
    request_id: String,
    cancel: CancellationToken,
    response: Mutex<Option<Streaming<pb::GenerateResponse>>>,
}

impl Call {
    fn abort(&self) {
        self.cancel.cancel();
        self.response.lock().unwrap().take();
    }

    async fn next(&self) -> Result<Option<pb::GenerateResponse>> {
        let message = futures::future::poll_fn(|cx| {
            let mut response = self.response.lock().unwrap();
            match response.as_mut() {
                Some(stream) => stream.poll_next_unpin(cx),
                None => std::task::Poll::Ready(None),
            }
        });
        tokio::select! {
            biased;
            _ = self.cancel.cancelled() => Err(remote("request aborted")),
            result = message => result.transpose().map_err(status),
        }
    }
}

type Active = Arc<Mutex<HashMap<Uuid, Arc<Call>>>>;

/// Token-only client for a vllm.Inference endpoint.
pub struct GrpcLlm {
    readiness: watch::Receiver<bool>,
    _probe: AbortOnDropHandle<()>,
    channel: Channel,
    model: String,
    info: LlmInfo,
    health: watch::Sender<bool>,
    active: Active,
}

impl GrpcLlm {
    /// Connect lazily to a worker with the given served model and context limit.
    pub fn new(url: &str, model: String, max_model_len: NonZeroU32) -> anyhow::Result<Self> {
        let channel = Endpoint::from_shared(url.to_owned())?.connect_lazy();
        Ok(Self::from_channel(channel, model, max_model_len))
    }

    pub(super) fn from_channel(channel: Channel, model: String, max_model_len: NonZeroU32) -> Self {
        let (ready, readiness) = watch::channel(false);
        let mut health = tonic_health::pb::health_client::HealthClient::new(channel.clone());
        let probe = tokio::spawn(async move {
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
            _probe: AbortOnDropHandle::new(probe),
            model,
            info: LlmInfo {
                max_model_len: max_model_len.get(),
                model_dtype: None,
                supports_lora: false,
                allow_missing_mm_embeddings: false,
                vllm_version: "unknown".to_owned(),
                fingerprint: vllm_llm::LlmFingerprint {
                    vllm_version: "unknown".to_owned(),
                    config_hash: None,
                    tensor_parallel_size: 1,
                    pipeline_parallel_size: 1,
                    data_parallel_size: 1,
                    enable_expert_parallel: false,
                },
            },
            health: watch::channel(true).0,
            active: Default::default(),
        }
    }
}

fn remote(e: impl ToString) -> Error {
    Error::Remote {
        message: e.to_string(),
    }
}
fn status(s: Status) -> Error {
    match s.code() {
        Code::InvalidArgument | Code::OutOfRange | Code::FailedPrecondition => {
            Error::RemoteInvalidRequest {
                message: s.to_string(),
            }
        }
        Code::Unavailable | Code::ResourceExhausted => Error::Unavailable {
            message: s.to_string(),
        },
        _ => remote(s),
    }
}

fn encode(model: &str, wire_id: Uuid, r: GenerateRequest) -> Result<Request<pb::GenerateRequest>> {
    let s = r.sampling_params;
    for (field, present) in [
        ("multimodal inputs", r.mm_features.is_some()),
        ("lora_request", r.lora_request.is_some()),
        ("kv_hints", r.kv_hints.is_some()),
        ("data_parallel_rank", r.data_parallel_rank.is_some()),
        ("extra_args", s.extra_args.is_some()),
        ("structured_outputs", s.structured_outputs.is_some()),
        (
            "logprobs",
            s.logprobs.is_some() || s.logprob_token_ids.is_some(),
        ),
        (
            "prompt_logprobs",
            s.prompt_logprobs.is_some()
                || s.prompt_logprob_start.is_some()
                || s.prompt_logprob_token_ids.is_some(),
        ),
        ("repetition_detection", s.repetition_detection.is_some()),
        ("stream_interval", s.stream_interval.is_some()),
        (
            "routed_experts_prompt_start",
            s.routed_experts_prompt_start != 0,
        ),
    ] {
        if present {
            return Err(Error::Unsupported { field });
        }
    }
    let kwargs = r
        .reasoning_parser_kwargs
        .filter(|k| !k.chat_template_kwargs.is_empty())
        .map(|k| {
            let value = serde_json::Value::Object(k.chat_template_kwargs.into_iter().collect());
            let encoded = super::convert::json_to_proto_struct(&value).expect("JSON object");
            if super::convert::proto_struct_to_json(&encoded) != value {
                return Err(Error::Unsupported {
                    field: "chat_template_kwargs number outside protobuf Struct precision",
                });
            }
            Ok(encoded)
        })
        .transpose()?;
    let mut req = Request::new(pb::GenerateRequest {
        model: model.into(),
        request_id: wire_id.to_string(),
        prompt: Some(pb::generate_request::Prompt::TokenIds(pb::TokenIds {
            ids: r.prompt_token_ids,
        })),
        temperature: Some(s.temperature),
        sampling: Some(pb::RandomSampling {
            num_sequences: 1,
            top_p: Some(s.top_p),
            top_k: Some(s.top_k),
            min_p: Some(s.min_p),
            seed: s.seed,
        }),
        decoding: Some(pb::DecodingParameters {
            presence_penalty: s.presence_penalty,
            frequency_penalty: s.frequency_penalty,
            repetition_penalty: s.repetition_penalty,
            allowed_token_ids: s.allowed_token_ids.unwrap_or_default(),
            logit_bias: s.logit_bias.unwrap_or_default(),
            bad_words_token_ids: s
                .bad_words_token_ids
                .unwrap_or_default()
                .into_iter()
                .map(|ids| pb::TokenIds { ids })
                .collect(),
            ..Default::default()
        }),
        stopping: Some(pb::StoppingCriteria {
            max_new_tokens: s.max_tokens,
            min_new_tokens: s.min_tokens,
            stop_token_ids: s.stop_token_ids,
            ignore_eos: s.eos_token_id.is_none(),
            thinking_token_budget: s
                .thinking_token_budget
                .map(i64::try_from)
                .transpose()
                .map_err(remote)?,
            ..Default::default()
        }),
        response: Some(pb::ResponseOptions {
            output_text: Some(false),
            output_token_ids: true,
            prompt_token_ids: true,
            ..Default::default()
        }),
        priority: r.priority,
        session_id: r.session_id,
        watermarking: Some(s.watermarking),
        kv: Some(pb::KvCacheParameters {
            cache_salt: r.cache_salt.unwrap_or_default(),
            bypass_prefix_cache: s.skip_reading_prefix_cache.unwrap_or(false),
            ..Default::default()
        }),
        engine_reasoning_gate: (r.reasoning_ended.is_some() || kwargs.is_some()).then_some(
            pb::EngineReasoningGate {
                reasoning_ended: r.reasoning_ended,
                chat_template_kwargs: kwargs,
            },
        ),
        ..Default::default()
    });
    for (key, value) in r.trace_headers.into_iter().flatten() {
        let key =
            tonic::metadata::MetadataKey::<tonic::metadata::Ascii>::from_bytes(key.as_bytes())
                .map_err(remote)?;
        req.metadata_mut().insert(key, value.parse().map_err(remote)?);
    }
    Ok(req)
}

struct Guard {
    key: Uuid,
    call: Arc<Call>,
    active: Active,
}
impl Drop for Guard {
    fn drop(&mut self) {
        self.active.lock().unwrap().remove(&self.key);
    }
}

fn finish(f: &pb::FinishInfo) -> Result<FinishReason> {
    use pb::finish_info::{FinishReason as F, StopReason as S};
    match F::try_from(f.finish_reason) {
        Ok(F::Length) => Ok(FinishReason::Length),
        Ok(F::Aborted) => Ok(FinishReason::Abort),
        Ok(F::Stop) => Ok(FinishReason::Stop(match &f.stop_reason {
            Some(S::StopTokenId(id)) => Some(StopReason::TokenId(*id)),
            Some(S::StopString(s)) => Some(StopReason::Text(s.clone())),
            _ => None,
        })),
        _ => Err(remote("invalid worker finish reason")),
    }
}

#[try_stream]
async fn outputs(
    request_id: String,
    frames: impl Stream<Item = Result<pb::GenerateResponse>> + Send,
    mut y: TryYielder<GenerateOutput, Error>,
) -> Result<()> {
    futures::pin_mut!(frames);
    let first = frames.try_next().await?.ok_or_else(|| remote("missing prompt frame"))?;
    let prompt = first.prompt_info.ok_or_else(|| remote("missing prompt info"))?;
    if first.outputs.is_some() || prompt.token_ids.len() != prompt.num_prompt_tokens as usize {
        return Err(remote("invalid prompt frame"));
    }
    let mut prompt = Some(Box::new(GeneratePromptInfo {
        prompt_token_ids: prompt.token_ids.into(),
        prompt_logprobs: None,
        prompt_token_id_logprobs: None,
    }));
    while let Some(frame) = frames.try_next().await? {
        let out = frame.outputs.ok_or_else(|| remote("missing sequence output"))?;
        if frame.prompt_info.is_some()
            || out.index != 0
            || out.token_ids.len() != out.num_tokens as usize
        {
            return Err(remote("invalid token delta"));
        }
        let terminal = out.finish_info.as_ref();
        let finish_reason = terminal.map(finish).transpose()?;
        // A worker can send a terminal message followed by a non-OK RPC status.
        if terminal.is_some() && frames.try_next().await?.is_some() {
            return Err(remote("output after finish"));
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
    Err(remote("generation ended without finish info"))
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
        mut r: GenerateRequest,
        n: NonZeroU32,
    ) -> Result<Vec<BoxGenerateOutputStream>> {
        if n.get() != 1 {
            return Err(Error::Unsupported { field: "n > 1" });
        }
        let id = std::mem::take(&mut r.request_id);
        let key = Uuid::new_v4();
        let req = encode(&self.model, key, r)?;
        let call = Arc::new(Call {
            request_id: id,
            cancel: CancellationToken::new(),
            response: Mutex::new(None),
        });
        self.active.lock().unwrap().insert(key, call.clone());
        let guard = Guard {
            key,
            call,
            active: self.active.clone(),
        };
        let mut client = InferenceClient::new(self.channel.clone());
        let res = tokio::select! {
            _ = guard.call.cancel.cancelled() => return Err(remote("request aborted")),
            response = client.generate_stream(req) => response.map_err(status)?,
        };
        {
            let mut stream = guard.call.response.lock().unwrap();
            if guard.call.cancel.is_cancelled() {
                return Err(remote("request aborted"));
            }
            *stream = Some(res.into_inner());
        }
        let request_id = guard.call.request_id.clone();
        let frames = futures::stream::try_unfold(guard, |guard| async move {
            let frame = guard.call.next().await?;
            Ok(frame.map(|frame| (frame, guard)))
        });
        Ok(vec![outputs(request_id, frames).boxed()])
    }
    async fn abort(&self, ids: &[String]) -> Result<()> {
        for call in self.active.lock().unwrap().values() {
            if ids.is_empty() || ids.contains(&call.request_id) {
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

    #[tokio::test]
    async fn terminal_frame_requires_successful_rpc_status() {
        let frames = futures::stream::iter([
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
                        ..Default::default()
                    }),
                    ..Default::default()
                }),
            }),
            Err(status(Status::unavailable("worker stopped"))),
        ]);
        let mut decoded = outputs("test".into(), frames).boxed();
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
