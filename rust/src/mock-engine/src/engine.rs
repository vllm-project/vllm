// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::hash::{Hash as _, Hasher as _};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Result, anyhow};
use rand::rngs::StdRng;
use rand::{Rng as _, SeedableRng as _};
use rmpv::Value;
use serde::Serialize;
use tokio::sync::mpsc;
use tokio::task::yield_now;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};
use vllm_engine_core_client::protocol::output::{
    EngineCoreFinishReason, EngineCoreOutput, EngineCoreOutputs, RequestBatchOutputs,
    UtilityCallOutput,
};
use vllm_engine_core_client::protocol::request::EngineCoreRequest;
use vllm_engine_core_client::protocol::utility::{
    EngineCoreUtilityRequest, UtilityOutput, UtilityResultEnvelope,
};

use super::Opt;

/// Derive a stable per-request seed from the CLI seed, engine, and request id.
fn request_seed(base_seed: u64, engine_index: u32, request_id: &str) -> u64 {
    let mut hasher = std::hash::DefaultHasher::new();
    base_seed.hash(&mut hasher);
    engine_index.hash(&mut hasher);
    request_id.hash(&mut hasher);
    hasher.finish()
}

/// Current UNIX timestamp in seconds for engine-core output envelopes.
fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or_default()
}

/// Build one request output with only token IDs and terminal status populated.
fn request_output(
    request_id: String,
    new_token_ids: Vec<u32>,
    finish_reason: Option<EngineCoreFinishReason>,
) -> EngineCoreOutput {
    EngineCoreOutput {
        request_id,
        new_token_ids,
        finish_reason,
        ..Default::default()
    }
}

/// Produce an empty output with a terminal finish reason for an invalid request.
fn empty_finish_outputs(
    engine_index: u32,
    request_id: String,
    finish_reason: EngineCoreFinishReason,
) -> EngineCoreOutputs {
    let output = request_output(request_id, Vec::new(), Some(finish_reason));
    let finished_requests = BTreeSet::from([output.request_id.clone()]);

    RequestBatchOutputs {
        engine_index,
        outputs: vec![output],
        timestamp: now_secs(),
        finished_requests: Some(finished_requests),
        ..Default::default()
    }
    .into()
}

/// Encode a utility result into the protocol's msgpack value envelope.
fn utility_envelope<T>(value: T) -> Result<UtilityResultEnvelope>
where
    T: Serialize,
{
    Ok(UtilityResultEnvelope::without_type_info(
        rmpv::ext::to_value(value)?,
    ))
}

/// Produce the minimal utility responses needed by the Rust frontend.
fn utility_response(
    engine_index: u32,
    request: EngineCoreUtilityRequest,
) -> Result<EngineCoreOutputs> {
    let result = match request.method_name.as_str() {
        "get_supported_tasks" => utility_envelope(vec!["generate"]),
        "is_sleeping" => utility_envelope(false),
        "reset_prefix_cache" => utility_envelope(true),
        "reset_mm_cache"
        | "reset_encoder_cache"
        | "profile"
        | "sleep"
        | "wake_up"
        | "execute_dummy_batch" => utility_envelope(()),
        _ => utility_envelope(Value::Nil),
    }?;

    Ok(UtilityCallOutput {
        engine_index,
        timestamp: now_secs(),
        output: UtilityOutput {
            call_id: request.call_id,
            failure_message: None,
            result: Some(result),
        },
    }
    .into())
}

/// Message sent from the frontend to the mock engine task to drive the engine loop.
pub(crate) enum EngineInput {
    Request(Box<EngineCoreRequest>),
    Abort(Vec<String>),
    Utility(EngineCoreUtilityRequest),
    StartDpWave,
}

/// Message sent from the mock engine task to the frontend for one engine output batch.
pub(crate) struct EngineOutput {
    pub client_index: u32,
    pub outputs: EngineCoreOutputs,
}

/// Per-request decode state owned by one mock engine.
#[derive(Debug)]
struct ActiveRequest {
    request_id: String,
    client_index: u32,
    prompt_len: usize,
    max_tokens: usize,
    generated: usize,
    rng: StdRng,
}

impl ActiveRequest {
    /// Create a new active request from an incoming EngineCoreRequest, or return an immediate
    /// finish reason if the request is invalid.
    fn new(
        engine_index: u32,
        request: Box<EngineCoreRequest>,
        opt: &Opt,
    ) -> Result<Self, EngineCoreFinishReason> {
        let request_id = request.request_id;
        let client_index = request.client_index;
        let prompt_len = request.prompt_token_ids.as_ref().map(Vec::len).unwrap_or_default();

        let Some(sampling_params) = request.sampling_params else {
            warn!(
                request_id,
                "request has no sampling params; returning engine error"
            );
            return Err(EngineCoreFinishReason::Error);
        };
        let max_tokens = sampling_params.max_tokens as usize;

        if opt.log_requests {
            info!(
                request_id,
                prompt_len,
                max_tokens,
                chunk_size = opt.output_token_chunk_size,
                "mock request started"
            );
        }

        if max_tokens == 0 {
            return Err(EngineCoreFinishReason::Length);
        }

        Ok(ActiveRequest {
            rng: StdRng::seed_from_u64(request_seed(opt.seed, engine_index, &request_id)),
            request_id,
            client_index,
            prompt_len,
            max_tokens,
            generated: 0,
        })
    }

    /// Advance this request by one mock engine step.
    fn step(&mut self, opt: &Opt) -> EngineCoreOutput {
        let remaining = self.max_tokens - self.generated;
        let chunk_len = remaining.min(opt.output_token_chunk_size);
        let mut new_token_ids = Vec::with_capacity(chunk_len);
        for _ in 0..chunk_len {
            new_token_ids.push(self.rng.random_range(0..opt.vocab_size));
        }
        self.generated += chunk_len;

        let finished = self.generated >= self.max_tokens;
        request_output(
            self.request_id.clone(),
            new_token_ids,
            finished.then_some(EngineCoreFinishReason::Length),
        )
    }
}

/// Internal state for one mock engine instance, owned by the engine loop task.
struct Engine {
    engine_index: u32,
    opt: Opt,
    active_requests: HashMap<String, ActiveRequest>,
}

impl Engine {
    /// Drain one frontend request message received on the input DEALER socket.
    fn handle_input(&mut self, input: EngineInput) -> Result<Vec<EngineOutput>> {
        let mut outputs = Vec::new();

        match input {
            EngineInput::Request(request) => {
                let request_id = request.request_id.clone();
                let client_index = request.client_index;

                if self.active_requests.contains_key(&request_id) {
                    warn!(
                        engine_index = self.engine_index,
                        request_id, "duplicate mock request id"
                    );
                    return Ok(vec![EngineOutput {
                        client_index,
                        outputs: empty_finish_outputs(
                            self.engine_index,
                            request_id,
                            EngineCoreFinishReason::Error,
                        ),
                    }]);
                }

                match ActiveRequest::new(self.engine_index, request, &self.opt) {
                    Ok(request) => {
                        self.active_requests.insert(request_id, request);
                    }
                    Err(finish_reason) => {
                        return Ok(vec![EngineOutput {
                            client_index,
                            outputs: empty_finish_outputs(
                                self.engine_index,
                                request_id,
                                finish_reason,
                            ),
                        }]);
                    }
                }
            }

            EngineInput::Abort(request_ids) => {
                let mut outputs_by_client =
                    BTreeMap::<u32, (Vec<EngineCoreOutput>, BTreeSet<String>)>::new();
                for request_id in request_ids {
                    if let Some(request) = self.active_requests.remove(&request_id) {
                        let output = request_output(
                            request_id.clone(),
                            Vec::new(),
                            Some(EngineCoreFinishReason::Abort),
                        );
                        let (outputs, finished_requests) = outputs_by_client
                            .entry(request.client_index)
                            .or_insert_with(|| (Vec::new(), BTreeSet::new()));
                        outputs.push(output);
                        finished_requests.insert(request_id.clone());
                        if self.opt.log_requests {
                            info!(request_id, finish_reason = "abort", "mock request aborted");
                        }
                    }
                }
                for (client_index, (client_outputs, finished_requests)) in outputs_by_client {
                    outputs.push({
                        let outputs = RequestBatchOutputs {
                            engine_index: self.engine_index,
                            outputs: client_outputs,
                            timestamp: now_secs(),
                            finished_requests: Some(finished_requests),
                            ..Default::default()
                        }
                        .into();
                        EngineOutput {
                            client_index,
                            outputs,
                        }
                    });
                }
            }

            EngineInput::Utility(request) => {
                debug!(
                    engine_index = self.engine_index,
                    call_id = %request.call_id,
                    method = request.method_name,
                    "mock utility request"
                );
                let client_index = request.client_index;
                outputs.push({
                    let outputs = utility_response(self.engine_index, request)?;
                    EngineOutput {
                        client_index,
                        outputs,
                    }
                });
            }

            EngineInput::StartDpWave => {
                debug!(
                    engine_index = self.engine_index,
                    "ignoring START_DP_WAVE in mock engine"
                );
            }
        }

        Ok(outputs)
    }

    /// Advance active requests once and return one batched engine output.
    fn step(&mut self) -> Vec<EngineOutput> {
        if self.active_requests.is_empty() {
            return Vec::new();
        }

        let mut outputs_by_client =
            BTreeMap::<u32, (Vec<EngineCoreOutput>, BTreeSet<String>)>::new();
        let mut all_finished_requests = BTreeSet::new();

        for request in self.active_requests.values_mut() {
            let client_index = request.client_index;
            let output = request.step(&self.opt);
            let request_id = request.request_id.clone();
            let finished = output.finished();
            if output.finished() {
                all_finished_requests.insert(request_id.clone());
                if self.opt.log_requests {
                    info!(
                        request_id,
                        prompt_len = request.prompt_len,
                        output_tokens = request.generated,
                        finish_reason = "length",
                        "mock request finished"
                    );
                }
            }
            let (outputs, finished_requests) = outputs_by_client
                .entry(client_index)
                .or_insert_with(|| (Vec::new(), BTreeSet::new()));
            if finished {
                finished_requests.insert(request_id.clone());
            }
            outputs.push(output);
        }

        for request_id in &all_finished_requests {
            self.active_requests.remove(request_id);
        }

        outputs_by_client
            .into_iter()
            .filter_map(|(client_index, (outputs, finished_requests))| {
                (!outputs.is_empty()).then(|| EngineOutput {
                    client_index,
                    outputs: RequestBatchOutputs {
                        engine_index: self.engine_index,
                        outputs,
                        timestamp: now_secs(),
                        finished_requests: (!finished_requests.is_empty())
                            .then_some(finished_requests),
                        ..Default::default()
                    }
                    .into(),
                })
            })
            .collect()
    }
}

/// Run the main loop for the mock engine, receiving `EngineInput` from `input_rx`
/// and sending `EngineOutput` to `output_tx` until `shutdown` is cancelled.
pub(crate) async fn run_engine_loop(
    engine_index: u32,
    opt: Opt,
    mut input_rx: mpsc::UnboundedReceiver<EngineInput>,
    output_tx: mpsc::Sender<EngineOutput>,
    shutdown: CancellationToken,
) -> Result<()> {
    let mut engine = Engine {
        engine_index,
        opt,
        active_requests: HashMap::new(),
    };

    loop {
        let outputs = tokio::select! {
            biased;
            _ = shutdown.cancelled() => break,

            input = input_rx.recv() => {
                let input = input
                    .ok_or_else(|| anyhow!("mock engine input channel closed"))?;
                engine.handle_input(input)?
            }

            // If there are active requests, step them once after yielding to the scheduler to
            // avoid blocking the engine loop while still making steady progress on request outputs.
            _ = yield_now(), if !engine.active_requests.is_empty() => {
                engine.step()
            }
        };

        for output in outputs {
            output_tx
                .send(output)
                .await
                .map_err(|_| anyhow!("mock engine IO task shut down"))?;
        }
    }

    Ok(())
}
