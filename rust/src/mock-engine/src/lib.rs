use std::collections::{BTreeMap, BTreeSet};
use std::hash::{Hash, Hasher};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use clap::Parser;
use futures::future::select_all;
use rand::rngs::StdRng;
use rand::{Rng as _, SeedableRng as _};
use rmpv::Value;
use serde::Serialize;
use tokio::task::JoinSet;
use tokio::task::yield_now;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};
use vllm_engine_core_client::EngineId;
use vllm_engine_core_client::mock_engine::{
    MockEngineConfig, MockEngineDataSockets, MockEngineSockets, connect_to_frontend,
};
use vllm_engine_core_client::protocol::{
    EngineCoreFinishReason, EngineCoreOutput, EngineCoreOutputs, EngineCoreRequest,
    EngineCoreRequestType, decode_msgpack, encode_msgpack,
    utility::{EngineCoreUtilityRequest, UtilityOutput, UtilityResultEnvelope},
};
use zeromq::ZmqMessage;
use zeromq::prelude::{SocketRecv, SocketSend};

/// Standalone engine-core protocol emulator for frontend stress testing.
#[derive(Debug, Clone, Parser)]
#[command(
    name = "vllm-mock-engine",
    about = "Run a mock vLLM headless engine for Rust frontend stress testing."
)]
pub struct Opt {
    /// Frontend-owned ZMQ handshake address.
    #[arg(long, default_value = "tcp://127.0.0.1:29550")]
    pub handshake_address: String,

    /// Number of mock engine identities to register with the frontend.
    #[arg(long, default_value_t = 1)]
    pub engine_count: usize,

    /// Number of accepted output tokens included in each EngineCoreOutput.
    #[arg(long, default_value_t = 1)]
    pub output_token_chunk_size: usize,

    /// Random token IDs are sampled uniformly from 0..vocab_size.
    #[arg(long, default_value_t = 32_000)]
    pub vocab_size: u32,

    /// Base seed for deterministic random token generation.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,

    /// Log a summary line for each request.
    #[arg(long)]
    pub log_requests: bool,
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

/// Wrap one request output in the engine-core batch envelope.
fn outputs_for_request(
    engine_index: u32,
    output: EngineCoreOutput,
    finished: bool,
) -> EngineCoreOutputs {
    let finished_requests = finished.then(|| BTreeSet::from([output.request_id.clone()]));
    EngineCoreOutputs {
        engine_index,
        outputs: vec![output],
        timestamp: now_secs(),
        finished_requests,
        ..Default::default()
    }
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

    Ok(EngineCoreOutputs {
        engine_index,
        utility_output: Some(UtilityOutput {
            call_id: request.call_id,
            failure_message: None,
            result: Some(result),
        }),
        timestamp: now_secs(),
        ..Default::default()
    })
}

/// Convert one ADD request into active decode state.
fn active_request_from_add(
    engine_index: u32,
    request: EngineCoreRequest,
    opt: &Opt,
) -> std::result::Result<ActiveRequest, EngineCoreOutputs> {
    let request_id = request.request_id;
    let client_index = request.client_index;
    let prompt_len = request.prompt_token_ids.as_ref().map(Vec::len).unwrap_or_default();

    let Some(sampling_params) = request.sampling_params else {
        warn!(
            request_id,
            "request has no sampling params; returning engine error"
        );
        let output = request_output(
            request_id.clone(),
            Vec::new(),
            Some(EngineCoreFinishReason::Error),
        );
        return Err(outputs_for_request(engine_index, output, true));
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
        let output = request_output(
            request_id.clone(),
            Vec::new(),
            Some(EngineCoreFinishReason::Length),
        );
        return Err(outputs_for_request(engine_index, output, true));
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

impl ActiveRequest {
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

/// Advance active requests once and return one batched engine output.
fn step_active_requests(
    engine_index: u32,
    opt: &Opt,
    active_requests: &mut BTreeMap<String, ActiveRequest>,
) -> Vec<(u32, EngineCoreOutputs)> {
    if active_requests.is_empty() {
        return Vec::new();
    }

    let request_ids = active_requests.keys().cloned().collect::<Vec<_>>();
    let mut outputs_by_client = BTreeMap::<u32, (Vec<EngineCoreOutput>, BTreeSet<String>)>::new();
    let mut all_finished_requests = BTreeSet::new();

    for request_id in request_ids {
        let Some(request) = active_requests.get_mut(&request_id) else {
            continue;
        };
        let client_index = request.client_index;
        let output = request.step(opt);
        let finished = output.finished();
        if output.finished() {
            all_finished_requests.insert(request_id.clone());
            if opt.log_requests {
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
        active_requests.remove(request_id);
    }

    outputs_by_client
        .into_iter()
        .filter_map(|(client_index, (outputs, finished_requests))| {
            (!outputs.is_empty()).then(|| {
                (
                    client_index,
                    EngineCoreOutputs {
                        engine_index,
                        outputs,
                        timestamp: now_secs(),
                        finished_requests: (!finished_requests.is_empty())
                            .then_some(finished_requests),
                        ..Default::default()
                    },
                )
            })
        })
        .collect()
}

async fn send_engine_outputs(
    push: &mut zeromq::PushSocket,
    outputs: EngineCoreOutputs,
) -> Result<()> {
    push.send(ZmqMessage::from(encode_msgpack(&outputs)?)).await?;
    Ok(())
}

async fn send_engine_outputs_to_client(
    data_sockets: &mut [MockEngineDataSockets],
    client_index: u32,
    outputs: EngineCoreOutputs,
) -> Result<()> {
    let socket_index = client_index as usize;
    let socket_index = if socket_index < data_sockets.len() {
        socket_index
    } else {
        warn!(
            client_index,
            socket_count = data_sockets.len(),
            "client index exceeds connected frontend sockets; using socket 0"
        );
        0
    };
    send_engine_outputs(&mut data_sockets[socket_index].push, outputs).await
}

async fn recv_from_any_client(
    data_sockets: &mut [MockEngineDataSockets],
) -> Result<(usize, Vec<Vec<u8>>)> {
    let recv_futures = data_sockets
        .iter_mut()
        .enumerate()
        .map(|(index, sockets)| Box::pin(async move { (index, sockets.dealer.recv().await) }))
        .collect::<Vec<_>>();
    let ((index, message), _, _) = select_all(recv_futures).await;
    let frames = message?.into_vec().into_iter().map(|frame| frame.as_ref().to_vec()).collect();
    Ok((index, frames))
}

/// Drain one frontend request message received on the input DEALER socket.
async fn handle_engine_message<F>(
    engine_index: u32,
    frames: Vec<F>,
    opt: &Opt,
    active_requests: &mut BTreeMap<String, ActiveRequest>,
    data_sockets: &mut [MockEngineDataSockets],
) -> Result<()>
where
    F: AsRef<[u8]>,
{
    if frames.is_empty() {
        warn!(engine_index, "received empty engine request message");
        return Ok(());
    }

    let request_type_frame = frames[0].as_ref();
    let Some(request_type) = EngineCoreRequestType::from_frame(request_type_frame) else {
        warn!(
            engine_index,
            ?request_type_frame,
            "unknown engine request type"
        );
        return Ok(());
    };

    if frames.len() != 2 {
        warn!(
            engine_index,
            frame_count = frames.len(),
            request_type = ?request_type,
            "invalid frame count for engine request"
        );
        return Ok(());
    }

    match request_type {
        EngineCoreRequestType::Add => {
            let request: EngineCoreRequest = decode_msgpack(frames[1].as_ref())?;
            let request_id = request.request_id.clone();
            if active_requests.contains_key(&request_id) {
                warn!(engine_index, request_id, "duplicate mock request id");
                let output =
                    request_output(request_id, Vec::new(), Some(EngineCoreFinishReason::Error));
                send_engine_outputs_to_client(
                    data_sockets,
                    request.client_index,
                    outputs_for_request(engine_index, output, true),
                )
                .await?;
                return Ok(());
            }

            let client_index = request.client_index;
            match active_request_from_add(engine_index, request, opt) {
                Ok(request) => {
                    active_requests.insert(request_id, request);
                }
                Err(outputs) => {
                    send_engine_outputs_to_client(data_sockets, client_index, outputs).await?
                }
            }
        }
        EngineCoreRequestType::Abort => {
            let request_ids: Vec<String> = decode_msgpack(frames[1].as_ref())?;
            let mut outputs_by_client =
                BTreeMap::<u32, (Vec<EngineCoreOutput>, BTreeSet<String>)>::new();
            for request_id in request_ids {
                if let Some(request) = active_requests.remove(&request_id) {
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
                    if opt.log_requests {
                        info!(request_id, finish_reason = "abort", "mock request aborted");
                    }
                }
            }
            for (client_index, (outputs, finished_requests)) in outputs_by_client {
                send_engine_outputs_to_client(
                    data_sockets,
                    client_index,
                    EngineCoreOutputs {
                        engine_index,
                        outputs,
                        timestamp: now_secs(),
                        finished_requests: Some(finished_requests),
                        ..Default::default()
                    },
                )
                .await?;
            }
        }
        EngineCoreRequestType::Utility => {
            let request: EngineCoreUtilityRequest = decode_msgpack(frames[1].as_ref())?;
            debug!(
                engine_index,
                call_id = %request.call_id,
                method = request.method_name,
                "mock utility request"
            );
            let client_index = request.client_index;
            send_engine_outputs_to_client(
                data_sockets,
                client_index,
                utility_response(engine_index, request)?,
            )
            .await?;
        }
        EngineCoreRequestType::StartDpWave => {
            debug!(engine_index, "ignoring START_DP_WAVE in mock engine");
        }
    }

    Ok(())
}

/// Run one mock engine until shutdown or transport failure.
async fn run_engine(engine_id: EngineId, opt: Opt, shutdown: CancellationToken) -> Result<()> {
    let engine_index = engine_id
        .engine_index()
        .context("mock engine id must encode a two-byte engine index")?;
    let MockEngineSockets {
        dealer,
        push,
        additional_data_sockets,
        coordinator: _coordinator,
        ..
    } = connect_to_frontend(
        &opt.handshake_address,
        engine_id,
        MockEngineConfig::default(),
    )
    .await
    .with_context(|| format!("failed to connect mock engine {engine_index}"))?;

    info!(engine_index, "mock engine connected");

    let mut data_sockets = Vec::with_capacity(additional_data_sockets.len() + 1);
    data_sockets.push(MockEngineDataSockets { dealer, push });
    data_sockets.extend(additional_data_sockets);

    let mut active_requests = BTreeMap::<String, ActiveRequest>::new();

    loop {
        if active_requests.is_empty() {
            tokio::select! {
                _ = shutdown.cancelled() => break,
                received = recv_from_any_client(&mut data_sockets) => {
                    let (_, frames) = received?;
                    handle_engine_message(
                        engine_index,
                        frames,
                        &opt,
                        &mut active_requests,
                        &mut data_sockets,
                    ).await?;
                }
            }
            continue;
        }

        // Prefer control/admission messages that are already available before
        // advancing decode. When none are ready, yield once and emit a single
        // engine-step batch covering every active request.
        tokio::select! {
            biased;
            _ = shutdown.cancelled() => break,
            received = recv_from_any_client(&mut data_sockets) => {
                let (_, frames) = received?;
                handle_engine_message(
                    engine_index,
                    frames,
                    &opt,
                    &mut active_requests,
                    &mut data_sockets,
                ).await?;
                continue;
            }
            _ = yield_now() => {}
        }

        for (client_index, outputs) in
            step_active_requests(engine_index, &opt, &mut active_requests)
        {
            send_engine_outputs_to_client(&mut data_sockets, client_index, outputs).await?;
        }
    }

    active_requests.clear();
    info!(engine_index, "mock engine shut down");
    Ok(())
}

/// Run all requested mock engines until cancellation or one engine task fails.
pub async fn run(opt: Opt, shutdown: CancellationToken) -> Result<()> {
    info!(?opt, "starting mock engine");

    let mut engines = JoinSet::new();
    for engine_index in 0..opt.engine_count {
        let engine_id = EngineId::from_engine_index(engine_index as u32);
        engines.spawn(run_engine(engine_id, opt.clone(), shutdown.clone()));
    }

    loop {
        tokio::select! {
            _ = shutdown.cancelled() => {
                engines.abort_all();
                while engines.join_next().await.is_some() {}
                return Ok(());
            }

            joined = engines.join_next() => {
                match joined {
                    Some(Ok(Ok(()))) => bail!("mock engine exited unexpectedly"),
                    Some(Ok(Err(error))) => return Err(error),
                    Some(Err(error)) => return Err(error).context("mock engine task join failed"),
                    None => return Ok(()),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
