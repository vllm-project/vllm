// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use asynk_strim_attr::{TryYielder, try_stream};
use eventsource_stream::Eventsource as _;
use futures::StreamExt as _;
use serde::Deserialize;
use vllm_engine_core_client::protocol::logprobs::{
    Logprobs, MaybeWireLogprobs, PositionLogprobs, TokenLogprob,
};
use vllm_engine_core_client::protocol::output::{EngineCoreFinishReason, EngineCoreOutput};
use vllm_engine_core_client::protocol::sampling_mask::{MaybeWireSamplingMask, SamplingMask};
use vllm_engine_core_client::protocol::stats::PrefillStats;
use vllm_engine_core_client::{
    EngineCoreStreamDelivery, EngineCoreStreamOutput, EngineOutputStreamAdapter, WeightVersionSpan,
    monotonic_timestamp_secs,
};

use super::*;
use crate::routes::inference::generate::types::{GenerateLogProbs, GenerateResponseStreamChoice};

#[derive(Deserialize)]
#[serde(untagged)]
enum GenerateEvent {
    Error { error: Value },
    Tokens(TokenResponse),
}

#[derive(Deserialize)]
struct TokenResponse {
    choices: Vec<GenerateResponseStreamChoice>,
    usage: Option<Value>,
    kv_transfer_params: Option<Value>,
    ec_transfer_params: Option<Value>,
}

struct RequestGuard {
    id: String,
    active: Arc<Mutex<HashMap<String, CancellationToken>>>,
}
impl Drop for RequestGuard {
    fn drop(&mut self) {
        self.active.lock().expect("active requests mutex").remove(&self.id);
    }
}

pub(super) async fn submit(
    client: &EngineHTTPClient,
    request: EngineCoreRequest,
) -> Result<Box<dyn EngineOutputStream>> {
    let body = request_body(&client.model, &request)?;
    let mut http = client
        .post("inference/v1/generate")
        .header("X-Request-Id", &request.request_id)
        .json(&body);
    if let Some(rank) = request.data_parallel_rank {
        http = http.header("X-data-parallel-rank", rank);
    }
    if let Some(session) = &request.session_id {
        http = http.header("X-Session-ID", session);
    }
    for (key, value) in request.trace_headers.as_ref().into_iter().flatten() {
        http = http.header(key, value);
    }
    let cancel = CancellationToken::new();
    {
        let mut active = client.active.lock().expect("active requests mutex");
        if active.contains_key(&request.request_id) {
            return Err(Error::DuplicateRequestId {
                request_id: request.request_id,
            });
        }
        active.insert(request.request_id.clone(), cancel.clone());
    }
    let guard = RequestGuard {
        id: request.request_id.clone(),
        active: client.active.clone(),
    };
    let response = tokio::select! {
        _ = cancel.cancelled() => return Err(remote_error(499, "generation cancelled")),
        response = send(http) => response?,
    };
    // No abort-on-drop hook: dropping the adapter drops `receive`, which
    // closes the HTTP response and unregisters the request via `guard`.
    // Explicit aborts go through `cancel` instead.
    let outputs = receive(response, request.request_id.clone(), cancel, guard)
        .map(|output| output.map(EngineCoreStreamDelivery::One));
    Ok(Box::new(EngineOutputStreamAdapter::new(
        request.request_id,
        0,
        outputs,
        None,
    )))
}

pub(super) fn request_body(model: &str, request: &EngineCoreRequest) -> Result<Value> {
    let sampling = request
        .sampling_params
        .as_ref()
        .ok_or_else(|| remote_error(400, "HTTP generation requires sampling parameters"))?;
    if request.mm_features.is_some()
        || request.kv_hints.is_some()
        || request.prompt_embeds.is_some()
        || request.pooling_params.is_some()
        || request.resumable
        || request.abort_immediately
        || request.prompt_is_token_ids.is_some()
    {
        return Err(remote_error(
            400,
            "HTTP generation does not support multimodal tensors, KV hints, pooling, resumable inputs, or pre-aborted requests",
        ));
    }
    if sampling.prompt_logprobs.is_some()
        || sampling.prompt_logprob_token_ids.is_some()
        || sampling.logprob_token_ids.as_ref().is_some_and(|ids| !ids.is_empty())
    {
        return Err(remote_error(
            400,
            "HTTP token streaming does not support prompt_logprobs or selective logprobs",
        ));
    }
    if sampling.bad_words_token_ids.as_ref().is_some_and(|ids| !ids.is_empty()) {
        return Err(remote_error(
            400,
            "HTTP generation cannot represent pretokenized bad_words constraints",
        ));
    }
    let tokens = request
        .prompt_token_ids
        .as_ref()
        .filter(|tokens| !tokens.is_empty())
        .ok_or_else(|| remote_error(400, "HTTP generation requires prompt token IDs"))?;
    let mut params = serde_json::to_value(sampling)
        .map_err(|error| remote_error(400, error.to_report_string()))?;
    params["ignore_eos"] = json!(sampling.eos_token_id.is_none());
    params["detokenize"] = json!(false);
    params["n"] = json!(1);
    // Reuse the already-resolved sampling values. The downstream input processor
    // derives EOS metadata again from the matching model configuration.
    params.as_object_mut().unwrap().remove("_eos_token_id");
    params.as_object_mut().unwrap().remove("_all_stop_token_ids");
    params.as_object_mut().unwrap().remove("_bad_words_token_ids");
    if let Some(structured) = params.get_mut("structured_outputs").and_then(Value::as_object_mut) {
        structured.remove("_backend");
    }
    let mut extra = sampling.extra_args.clone().unwrap_or_default();
    let kv = extra.remove("kv_transfer_params");
    let ec = extra.remove("ec_transfer_params");
    params["extra_args"] = json!(extra);
    Ok(json!({
        "request_id": request.request_id,
        "model": request.lora_request.as_ref().map_or(model, |lora| lora.lora_name.as_str()),
        "token_ids": tokens,
        "sampling_params": params,
        "stream": true,
        "stream_options": {"include_usage": true, "continuous_usage_stats": true},
        "output_mode": "tokens",
        "cache_salt": request.cache_salt,
        "priority": request.priority,
        "kv_transfer_params": kv,
        "ec_transfer_params": ec,
        "reasoning_ended": request.reasoning_ended,
        "reasoning_parser_kwargs": request.reasoning_parser_kwargs,
    }))
}

#[try_stream]
async fn receive(
    response: Response,
    request_id: String,
    cancel: CancellationToken,
    guard: RequestGuard,
    mut y: TryYielder<EngineCoreStreamOutput, Error>,
) -> Result<()> {
    let _guard = guard;
    let mut cached_tokens = 0;
    let mut created_tokens = 0;
    let mut token_count = 0;
    let mut weight_versions: Vec<WeightVersionSpan> = Vec::new();
    let mut events = response.bytes_stream().eventsource();
    let mut terminal: Option<EngineCoreOutput> = None;
    loop {
        let event = tokio::select! {
            _ = cancel.cancelled() => {
                let weight_versions = reported_weight_versions(weight_versions, token_count);
                y.yield_ok(envelope(EngineCoreOutput { request_id, finish_reason: Some(EngineCoreFinishReason::Abort), ..Default::default() }, weight_versions)).await;
                return Ok(());
            }
            event = events.next() => event,
        };
        let event = event
            .ok_or_else(|| remote_error(502, "generate stream closed before [DONE]"))?
            .map_err(|error| remote_error(502, error.to_report_string()))?;
        if event.data == "[DONE]" {
            let mut output = terminal.ok_or_else(|| {
                remote_error(502, "generate stream ended without a finish reason")
            })?;
            output.prefill_stats = Some(Box::new(PrefillStats {
                num_cached_tokens: cached_tokens,
                num_cache_creation_tokens: created_tokens,
                ..Default::default()
            }));
            if weight_versions.last().is_some_and(|span| span.end != token_count) {
                return Err(remote_error(
                    502,
                    "weight version spans do not cover output tokens",
                ));
            }
            let weight_versions = reported_weight_versions(weight_versions, token_count);
            y.yield_ok(envelope(output, weight_versions)).await;
            return Ok(());
        }
        let event = serde_json::from_str::<GenerateEvent>(&event.data)
            .map_err(|error| remote_error(502, error.to_report_string()))?;
        let body = match event {
            GenerateEvent::Error { error } => return Err(remote_error(502, error.to_string())),
            GenerateEvent::Tokens(body) => body,
        };
        if let Some(details) =
            body.usage.as_ref().and_then(|usage| usage.get("prompt_tokens_details"))
        {
            if let Some(value) = details.get("cached_tokens") {
                cached_tokens = serde_json::from_value::<u32>(value.clone())
                    .map_err(|error| remote_error(502, error.to_report_string()))?;
            }
            if let Some(value) = details.get("created_cache_tokens") {
                created_tokens = serde_json::from_value::<u32>(value.clone())
                    .map_err(|error| remote_error(502, error.to_report_string()))?;
            }
        }
        if body.choices.is_empty() {
            continue;
        }
        if terminal.is_some() {
            return Err(remote_error(502, "tokens received after terminal output"));
        }
        if body.choices.len() != 1 || body.choices[0].index != 0 {
            return Err(remote_error(
                502,
                "expected one generate choice with index 0",
            ));
        }
        let choice = body.choices.into_iter().next().unwrap();
        token_count += choice.token_ids.len();
        if let Some(spans) = choice.weight_versions {
            let mut end = 0;
            for span in &spans {
                if span.start != end || span.end <= span.start || span.end > token_count {
                    return Err(remote_error(502, "invalid weight version spans"));
                }
                end = span.end;
            }
            if end != token_count {
                return Err(remote_error(
                    502,
                    "weight version spans do not cover output tokens",
                ));
            }
            weight_versions = spans;
        }
        let finish_reason = match choice.finish_reason.as_deref() {
            None => None,
            Some("stop") => Some(EngineCoreFinishReason::Stop),
            Some("length") => Some(EngineCoreFinishReason::Length),
            Some("abort") => Some(EngineCoreFinishReason::Abort),
            Some("repetition") => Some(EngineCoreFinishReason::Repetition),
            Some(reason) => {
                return Err(remote_error(
                    502,
                    format!("generate finished with {reason}"),
                ));
            }
        };
        let logprobs = choice.logprobs.map(output_logprobs);
        if logprobs
            .as_ref()
            .is_some_and(|probs| probs.positions.len() != choice.token_ids.len())
            || choice
                .sampling_mask
                .as_ref()
                .is_some_and(|rows| rows.len() != choice.token_ids.len())
        {
            return Err(remote_error(
                502,
                "generate logprobs or sampling masks do not align with token IDs",
            ));
        }
        if choice.routed_experts.is_some() {
            return Err(remote_error(
                502,
                "HTTP routed-expert decoding is not implemented",
            ));
        }
        let output = EngineCoreOutput {
            request_id: request_id.clone(),
            new_token_ids: choice.token_ids,
            new_logprobs: logprobs.map(MaybeWireLogprobs::Direct),
            finish_reason,
            stop_reason: choice.stop_reason,
            new_sampling_mask: choice
                .sampling_mask
                .map(|rows| MaybeWireSamplingMask::Direct(SamplingMask { rows })),
            kv_transfer_params: body.kv_transfer_params.map(Box::new),
            ec_transfer_params: body.ec_transfer_params.map(Box::new),
            ..Default::default()
        };
        if output.finished() {
            terminal = Some(output);
        } else {
            y.yield_ok(envelope(output, None)).await;
        }
    }
}

/// Spans to attach to a terminal output: `None` when the wire reported no
/// label for a non-empty output, or its spans no longer cover every token.
fn reported_weight_versions(
    spans: Vec<WeightVersionSpan>,
    token_count: usize,
) -> Option<Vec<WeightVersionSpan>> {
    if (spans.is_empty() && token_count > 0)
        || spans.last().is_some_and(|span| span.end != token_count)
    {
        None
    } else {
        Some(spans)
    }
}

fn envelope(
    output: EngineCoreOutput,
    weight_versions: Option<Vec<WeightVersionSpan>>,
) -> EngineCoreStreamOutput {
    EngineCoreStreamOutput {
        timestamp: monotonic_timestamp_secs(),
        weight_version: None,
        weight_versions,
        output,
    }
}
pub(super) fn output_logprobs(wire: GenerateLogProbs) -> Logprobs {
    Logprobs {
        positions: wire
            .content
            .unwrap_or_default()
            .into_iter()
            .map(|row| {
                let mut entries = vec![TokenLogprob {
                    token_id: row.token_id,
                    logprob: row.logprob,
                    rank: row.rank.unwrap_or(0),
                }];
                entries.extend(
                    row.top_logprobs
                        .into_iter()
                        .filter(|entry| entry.token_id != row.token_id)
                        .map(|entry| TokenLogprob {
                            token_id: entry.token_id,
                            logprob: entry.logprob,
                            rank: entry.rank.unwrap_or(0),
                        }),
                );
                PositionLogprobs { entries }
            })
            .collect(),
    }
}
