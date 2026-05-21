use std::collections::{BTreeSet, HashMap};
use std::hash::{Hash, Hasher};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use asynk_strim_attr::{Yielder, stream};
use clap::Parser;
use futures::{StreamExt as _, pin_mut};
use rand::rngs::StdRng;
use rand::{Rng as _, SeedableRng as _};
use rmpv::Value;
use serde::Serialize;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};
use vllm_engine_core_client::EngineId;
use vllm_engine_core_client::mock_engine::{
    MockEngineConfig, MockEngineSockets, connect_to_frontend,
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

/// Per-request decode settings captured before spawning a decode task.
#[derive(Debug)]
struct DecodeRequest {
    /// Engine-core ADD request received from the frontend.
    request: EngineCoreRequest,
    /// Python-compatible engine index included in output envelopes.
    engine_index: u32,
    /// Shared CLI options for token generation and logging.
    opt: Opt,
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

/// Simulate decode by emitting random token IDs until max_tokens is reached.
#[stream]
async fn decode_stream(
    request: DecodeRequest,
    mut abort_rx: oneshot::Receiver<()>,
    mut y: Yielder<EngineCoreOutputs>,
) {
    let request_id = request.request.request_id;
    let prompt_len = request.request.prompt_token_ids.as_ref().map(Vec::len).unwrap_or_default();

    let Some(sampling_params) = request.request.sampling_params else {
        warn!(
            request_id,
            "request has no sampling params; returning engine error"
        );
        let output = request_output(
            request_id.clone(),
            Vec::new(),
            Some(EngineCoreFinishReason::Error),
        );
        y.yield_item(outputs_for_request(request.engine_index, output, true)).await;
        return;
    };

    let max_tokens = sampling_params.max_tokens as usize;
    if request.opt.log_requests {
        info!(
            request_id,
            prompt_len,
            max_tokens,
            chunk_size = request.opt.output_token_chunk_size,
            "mock request started"
        );
    }

    if max_tokens == 0 {
        let output = request_output(
            request_id.clone(),
            Vec::new(),
            Some(EngineCoreFinishReason::Length),
        );
        y.yield_item(outputs_for_request(request.engine_index, output, true)).await;
        return;
    }

    let mut rng = StdRng::seed_from_u64(request_seed(
        request.opt.seed,
        request.engine_index,
        &request_id,
    ));
    let mut generated = 0usize;
    while generated < max_tokens {
        if abort_rx.try_recv().is_ok() {
            return;
        }

        let remaining = max_tokens - generated;
        let chunk_len = remaining.min(request.opt.output_token_chunk_size);
        let mut new_token_ids = Vec::with_capacity(chunk_len);
        for _ in 0..chunk_len {
            new_token_ids.push(rng.random_range(0..request.opt.vocab_size));
        }
        generated += chunk_len;

        let finished = generated >= max_tokens;
        let finish_reason = finished.then_some(EngineCoreFinishReason::Length);
        let output = request_output(request_id.clone(), new_token_ids, finish_reason);
        y.yield_item(outputs_for_request(request.engine_index, output, finished)).await;

        if finished {
            if request.opt.log_requests {
                info!(
                    request_id,
                    prompt_len,
                    output_tokens = generated,
                    finish_reason = "length",
                    "mock request finished"
                );
            }
            return;
        }

        tokio::select! {
            _ = &mut abort_rx => return,
            _ = tokio::task::yield_now() => {}
        }
    }
}

/// Forward one decode stream into the shared output channel.
async fn run_decode_request(
    request: DecodeRequest,
    abort_rx: oneshot::Receiver<()>,
    output_tx: mpsc::UnboundedSender<EngineCoreOutputs>,
) -> Result<String> {
    let request_id = request.request.request_id.clone();
    let stream = decode_stream(request, abort_rx);
    pin_mut!(stream);

    while let Some(outputs) = stream.next().await {
        if output_tx.send(outputs).is_err() {
            break;
        }
    }

    Ok(request_id)
}

/// Handle one frontend request message received on the input DEALER socket.
fn handle_engine_message<F>(
    engine_index: u32,
    frames: Vec<F>,
    opt: &Opt,
    active_requests: &mut HashMap<String, oneshot::Sender<()>>,
    decode_tasks: &mut JoinSet<Result<String>>,
    output_tx: &mpsc::UnboundedSender<EngineCoreOutputs>,
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
                output_tx.send(outputs_for_request(engine_index, output, true)).ok();
                return Ok(());
            }

            let (abort_tx, abort_rx) = oneshot::channel();
            active_requests.insert(request_id, abort_tx);
            let decode = DecodeRequest {
                request,
                engine_index,
                opt: opt.clone(),
            };
            let output_tx = output_tx.clone();
            decode_tasks.spawn(run_decode_request(decode, abort_rx, output_tx));
        }
        EngineCoreRequestType::Abort => {
            let request_ids: Vec<String> = decode_msgpack(frames[1].as_ref())?;
            for request_id in request_ids {
                if let Some(abort_tx) = active_requests.remove(&request_id) {
                    let _ = abort_tx.send(());
                    let output = request_output(
                        request_id.clone(),
                        Vec::new(),
                        Some(EngineCoreFinishReason::Abort),
                    );
                    output_tx.send(outputs_for_request(engine_index, output, true)).ok();
                    if opt.log_requests {
                        info!(request_id, finish_reason = "abort", "mock request aborted");
                    }
                }
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
            output_tx.send(utility_response(engine_index, request)?).ok();
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
        mut dealer,
        mut push,
        ..
    } = connect_to_frontend(
        &opt.handshake_address,
        engine_id,
        MockEngineConfig::default(),
    )
    .await
    .with_context(|| format!("failed to connect mock engine {engine_index}"))?;

    info!(engine_index, "mock engine connected");

    let (output_tx, mut output_rx) = mpsc::unbounded_channel::<EngineCoreOutputs>();
    let writer = tokio::spawn(async move {
        while let Some(outputs) = output_rx.recv().await {
            push.send(ZmqMessage::from(encode_msgpack(&outputs)?)).await?;
        }
        Ok::<(), anyhow::Error>(())
    });

    let mut active_requests = HashMap::<String, oneshot::Sender<()>>::new();
    let mut decode_tasks = JoinSet::new();

    loop {
        tokio::select! {
            _ = shutdown.cancelled() => {
                break;
            }

            Some(joined) = decode_tasks.join_next() => {
                let request_id = joined.context("decode task join failed")??;
                active_requests.remove(&request_id);
            }

            message = dealer.recv() => {
                let frames = message?.into_vec();
                handle_engine_message(
                    engine_index,
                    frames,
                    &opt,
                    &mut active_requests,
                    &mut decode_tasks,
                    &output_tx,
                )?;
            }
        }
    }

    for (_, abort_tx) in active_requests {
        let _ = abort_tx.send(());
    }
    decode_tasks.abort_all();
    while decode_tasks.join_next().await.is_some() {}
    drop(output_tx);
    writer.await.context("output writer join failed")??;
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
