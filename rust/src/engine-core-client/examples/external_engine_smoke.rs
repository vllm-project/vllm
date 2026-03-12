use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use clap::Parser;
use tokio::time::timeout;
use tracing_subscriber::EnvFilter;
use vllm_engine_core_client::{
    EngineCoreClient, EngineCoreClientConfig, EngineCoreRequest, FinishReason, RequestBatchOutputs,
    RequestOutputKind, SamplingParams, StopReason,
};

const PROMPT_TOKEN_IDS: &[u32] = &[20841, 448, 6896, 25, 23811];

#[derive(Debug, Parser)]
#[command(about = "Smoke-test a Rust EngineCoreClient against an external vLLM engine.")]
struct Args {
    #[arg(long)]
    handshake_address: String,
    #[arg(long, default_value = "Qwen/Qwen3-0.6B")]
    model: String,
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 0)]
    client_index: u32,
    #[arg(long, default_value_t = 30)]
    ready_timeout_secs: u64,
    #[arg(long, default_value_t = 120)]
    output_timeout_secs: u64,
    #[arg(long, default_value_t = 5)]
    max_tokens: u32,
}

fn unix_timestamp_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is before unix epoch")
        .as_secs_f64()
}

fn unique_request_id() -> String {
    format!("rust-engine-core-smoke-{}", uuid::Uuid::new_v4())
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("vllm_engine_core_client=debug"));
    let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
}

fn build_request(request_id: String, max_tokens: u32) -> EngineCoreRequest {
    EngineCoreRequest {
        request_id,
        prompt_token_ids: Some(PROMPT_TOKEN_IDS.to_vec()),
        sampling_params: Some(SamplingParams {
            max_tokens: Some(max_tokens),
            // Raw EngineCore outputs remain step/increment based even if
            // `FinalOnly` is requested here; that behavior is enforced by the
            // higher-level Python frontend output processor instead.
            output_kind: RequestOutputKind::Delta,
            ..Default::default()
        }),
        arrival_time: unix_timestamp_secs(),
        ..Default::default()
    }
}

#[derive(Debug, Default)]
struct CompletedRequest {
    new_token_ids: Vec<u32>,
    finish_reason: Option<FinishReason>,
    stop_reason: Option<StopReason>,
    finished_via_finished_requests: bool,
}

async fn wait_for_request_completion(
    client: &mut EngineCoreClient,
    request_id: &str,
) -> vllm_engine_core_client::Result<CompletedRequest> {
    let mut completed = CompletedRequest::default();

    loop {
        let Ok(RequestBatchOutputs {
            outputs,
            finished_requests,
            ..
        }) = client.next_classified_output().await?.into_request_batch()
        else {
            continue;
        };

        let finished_via_finished_requests = finished_requests
            .as_ref()
            .is_some_and(|request_ids| request_ids.contains(request_id));

        if let Some(output) = outputs
            .into_iter()
            .find(|output| output.request_id == request_id)
        {
            let finished = output.finished();
            completed.new_token_ids.extend(output.new_token_ids);

            if finished {
                completed.finish_reason = output.finish_reason;
                completed.stop_reason = output.stop_reason;
                return Ok(completed);
            }
        }

        if finished_via_finished_requests {
            completed.finished_via_finished_requests = true;
            return Ok(completed);
        }
    }
}

async fn wait_for_timeout(
    client: &mut EngineCoreClient,
    request_id: &str,
    output_timeout: Duration,
) -> Result<CompletedRequest> {
    timeout(
        output_timeout,
        wait_for_request_completion(client, request_id),
    )
    .await
    .context("timed out waiting for request output")?
    .context("failed to receive request output")
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    init_tracing();
    let args = Args::parse();
    let ready_timeout = Duration::from_secs(args.ready_timeout_secs);
    let output_timeout = Duration::from_secs(args.output_timeout_secs);
    let request_id = unique_request_id();
    let mut client = EngineCoreClient::connect(EngineCoreClientConfig {
        handshake_address: args.handshake_address.clone(),
        local_host: args.host.clone(),
        ready_timeout,
        client_index: args.client_index,
    })
    .await
    .context("failed to connect to external vLLM engine")?;

    println!("model={}", args.model);
    println!("handshake_address={}", args.handshake_address);
    println!("input_address={}", client.input_address());
    println!("output_address={}", client.output_address());
    println!("engine_identity={:x?}", client.engine_identity());

    let ready_message = client.ready_message.clone();
    let request = build_request(request_id.clone(), args.max_tokens);
    println!("request_id={request_id}");
    println!("prompt_token_ids={PROMPT_TOKEN_IDS:?}");
    println!("ready_message={ready_message:?}");

    client
        .add_request(request)
        .await
        .context("failed to add request")?;

    let output = wait_for_timeout(&mut client, &request_id, output_timeout).await?;

    client
        .shutdown()
        .await
        .context("failed to shut down engine client")?;

    println!("new_token_ids={:?}", output.new_token_ids);
    println!("finish_reason={:?}", output.finish_reason);
    println!("stop_reason={:?}", output.stop_reason);
    println!(
        "finished_via_finished_requests={}",
        output.finished_via_finished_requests
    );

    Ok(())
}
