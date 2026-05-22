use std::collections::{BTreeMap, BTreeSet};
use std::hash::{Hash, Hasher};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use futures::{Stream, StreamExt as _, stream};
use rand::rngs::StdRng;
use rand::{Rng as _, SeedableRng as _};
use rmpv::Value;
use serde::Serialize;
use tokio::sync::mpsc;
use tokio::task::{JoinSet, yield_now};
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};
use vllm_engine_core_client::EngineId;
use vllm_engine_core_client::mock_engine::{
    MockEngineConfig, MockEngineDataSockets, MockEngineSockets, connect_to_frontend,
};
use vllm_engine_core_client::protocol::utility::{
    EngineCoreUtilityRequest, UtilityOutput, UtilityResultEnvelope,
};
use vllm_engine_core_client::protocol::{
    EngineCoreFinishReason, EngineCoreOutput, EngineCoreOutputs, EngineCoreRequest,
    EngineCoreRequestType, decode_msgpack, encode_msgpack,
};
use zeromq::prelude::{SocketRecv, SocketSend};
use zeromq::{DealerSocket, PushSocket, ZmqMessage};

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

fn empty_finish_outputs(
    engine_index: u32,
    request_id: String,
    finish_reason: EngineCoreFinishReason,
) -> EngineCoreOutputs {
    let output = request_output(request_id, Vec::new(), Some(finish_reason));
    let finished_requests = BTreeSet::from([output.request_id.clone()]);

    EngineCoreOutputs {
        engine_index,
        outputs: vec![output],
        timestamp: now_secs(),
        finished_requests: Some(finished_requests),
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

enum EngineInput {
    Request(EngineCoreRequest),
    Abort(Vec<String>),
    Utility(EngineCoreUtilityRequest),
    StartDpWave,
}

struct EngineOutput {
    client_index: u32,
    outputs: EngineCoreOutputs,
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
    /// Create a new active request from an incoming EngineCoreRequest, or return an immediate finish reason if the request is invalid.
    fn new(
        engine_index: u32,
        request: EngineCoreRequest,
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

struct Engine {
    engine_index: u32,
    opt: Opt,
    active_requests: BTreeMap<String, ActiveRequest>,
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
                        let outputs = EngineCoreOutputs {
                            engine_index: self.engine_index,
                            outputs: client_outputs,
                            timestamp: now_secs(),
                            finished_requests: Some(finished_requests),
                            ..Default::default()
                        };
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

        let request_ids = self.active_requests.keys().cloned().collect::<Vec<_>>();
        let mut outputs_by_client =
            BTreeMap::<u32, (Vec<EngineCoreOutput>, BTreeSet<String>)>::new();
        let mut all_finished_requests = BTreeSet::new();

        for request_id in request_ids {
            let Some(request) = self.active_requests.get_mut(&request_id) else {
                continue;
            };
            let client_index = request.client_index;
            let output = request.step(&self.opt);
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
                    outputs: EngineCoreOutputs {
                        engine_index: self.engine_index,
                        outputs,
                        timestamp: now_secs(),
                        finished_requests: (!finished_requests.is_empty())
                            .then_some(finished_requests),
                        ..Default::default()
                    },
                })
            })
            .collect()
    }
}

async fn send_engine_outputs_to_client(
    push_sockets: &mut [PushSocket],
    EngineOutput {
        client_index,
        outputs,
    }: EngineOutput,
) -> Result<()> {
    let message = ZmqMessage::from(encode_msgpack(&outputs)?);
    push_sockets[client_index as usize].send(message).await?;
    Ok(())
}

fn dealer_input_stream(dealer: DealerSocket) -> impl Stream<Item = Result<EngineInput>> {
    stream::unfold(dealer, |mut dealer| async {
        let input = loop {
            let message =
                match dealer.recv().await.context("failed to receive message from dealer socket") {
                    Ok(message) => message,
                    Err(err) => break Err(err),
                };

            match decode_request(message) {
                Ok(input) => break Ok(input),
                Err(err) => {
                    warn!(%err, "failed to decode engine request message; ignoring");
                }
            }
        };

        Some((input, dealer))
    })
}

fn decode_request(message: ZmqMessage) -> Result<EngineInput> {
    let frames = message.into_vec();

    if frames.is_empty() {
        bail!("empty engine request message");
    }

    let request_type_frame = frames[0].as_ref();
    let Some(request_type) = EngineCoreRequestType::from_frame(request_type_frame) else {
        bail!("unknown engine request type: {:?}", request_type_frame);
    };

    if frames.len() != 2 {
        bail!("invalid frame count for engine request: {}", frames.len());
    }

    let input = match request_type {
        EngineCoreRequestType::Add => {
            let request: EngineCoreRequest = decode_msgpack(frames[1].as_ref())?;
            EngineInput::Request(request)
        }
        EngineCoreRequestType::Abort => {
            let request_ids: Vec<String> = decode_msgpack(frames[1].as_ref())?;
            EngineInput::Abort(request_ids)
        }
        EngineCoreRequestType::Utility => {
            let request: EngineCoreUtilityRequest = decode_msgpack(frames[1].as_ref())?;
            EngineInput::Utility(request)
        }
        EngineCoreRequestType::StartDpWave => EngineInput::StartDpWave,
    };

    Ok(input)
}

async fn run_io_loop(
    data_sockets: Vec<MockEngineDataSockets>,
    input_tx: mpsc::UnboundedSender<EngineInput>,
    mut output_rx: mpsc::Receiver<EngineOutput>,
    shutdown: CancellationToken,
) -> Result<()> {
    let (dealers, mut push_sockets): (Vec<_>, Vec<_>) =
        data_sockets.into_iter().map(|sockets| (sockets.dealer, sockets.push)).unzip();
    let mut input_streams =
        stream::select_all(dealers.into_iter().map(dealer_input_stream).map(Box::pin));

    loop {
        tokio::select! {
            _ = shutdown.cancelled() => return Ok(()),
            input = input_streams.next() => {
                let input = input
                    .ok_or_else(|| anyhow!("mock engine input streams closed"))??;
                input_tx
                    .send(input)
                    .map_err(|_| anyhow!("mock engine state task shut down"))?;
            }
            output = output_rx.recv() => {
                let output = output
                    .ok_or_else(|| anyhow!("mock engine output channel closed"))?;
                send_engine_outputs_to_client(&mut push_sockets, output).await?;
            }
        }
    }
}

async fn run_engine_loop(
    engine_index: u32,
    opt: Opt,
    mut input_rx: mpsc::UnboundedReceiver<EngineInput>,
    output_tx: mpsc::Sender<EngineOutput>,
    shutdown: CancellationToken,
) -> Result<()> {
    let mut engine = Engine {
        engine_index,
        opt,
        active_requests: BTreeMap::new(),
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

/// Run one mock engine until shutdown or transport failure.
async fn run_engine(engine_index: u32, opt: Opt, shutdown: CancellationToken) -> Result<()> {
    let MockEngineSockets { data_sockets, .. } = connect_to_frontend(
        &opt.handshake_address,
        EngineId::from_engine_index(engine_index),
        MockEngineConfig::default(),
    )
    .await
    .with_context(|| format!("mock engine {engine_index} failed to connect to frontend"))?;

    info!(engine_index, "mock engine connected to frontend");

    let (input_tx, input_rx) = mpsc::unbounded_channel();
    let (output_tx, output_rx) = mpsc::channel(64);

    let mut io_loop = tokio::spawn(run_io_loop(
        data_sockets,
        input_tx,
        output_rx,
        shutdown.clone(),
    ));
    let mut engine_loop = tokio::spawn(run_engine_loop(
        engine_index,
        opt,
        input_rx,
        output_tx,
        shutdown.clone(),
    ));

    tokio::select! {
        biased;

        _ = shutdown.cancelled() => {
            io_loop.abort();
            engine_loop.abort();
            io_loop.await.ok();
            engine_loop.await.ok();
        }

        result = &mut io_loop => {
            error!(engine_index, "mock engine IO loop exited unexpectedly");
            engine_loop.abort();
            engine_loop.await.ok();
            result??;
        }
        result = &mut engine_loop => {
            error!(engine_index, "mock engine loop exited unexpectedly");
            io_loop.abort();
            io_loop.await.ok();
            result??;
        }
    }

    info!(engine_index, "mock engine shut down");
    Ok(())
}

/// Run all requested mock engines until cancellation or one engine task fails.
pub async fn run(opt: Opt, shutdown: CancellationToken) -> Result<()> {
    info!(?opt, "starting mock engine");

    let mut engines = JoinSet::new();
    for engine_index in 0..opt.engine_count {
        engines.spawn(run_engine(
            engine_index as u32,
            opt.clone(),
            shutdown.clone(),
        ));
    }

    tokio::select! {
        biased;

        _ = shutdown.cancelled() => {
            engines.abort_all();
            engines.join_all().await;
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

#[cfg(test)]
mod tests;
