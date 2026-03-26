use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::Parser;
use futures::StreamExt as _;
use tokio::time::timeout;
use tracing_subscriber::EnvFilter;
use vllm_engine_core_client::protocol::{
    EngineCoreFinishReason, EngineCoreRequest, EngineCoreSamplingParams,
};
use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig, EngineCoreStreamOutput};

const BASE_PROMPT_TOKEN_IDS: &[u32] = &[20841, 448, 6896, 25, 23811];

#[derive(Debug, Parser)]
#[command(about = "Smoke-test engine-core sample logprobs against an external vLLM engine.")]
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
    #[arg(long, default_value_t = 1)]
    max_tokens: u32,
    #[arg(long, default_value_t = 2)]
    logprobs: i32,
    #[arg(long, default_value_t = 1)]
    prompt_logprobs: i32,
    #[arg(long, default_value_t = 96)]
    prompt_repeats: usize,
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("vllm_engine_core_client=debug"));
    let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
}

fn unique_request_id() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock should be after unix epoch")
        .as_nanos();
    format!("rust-engine-core-logprobs-{nanos}")
}

fn build_prompt_token_ids(prompt_repeats: usize) -> Vec<u32> {
    let repeats = prompt_repeats.max(1);
    BASE_PROMPT_TOKEN_IDS.repeat(repeats)
}

fn build_request(
    request_id: String,
    prompt_token_ids: Vec<u32>,
    max_tokens: u32,
    logprobs: i32,
    prompt_logprobs: i32,
    client_index: u32,
) -> EngineCoreRequest {
    EngineCoreRequest {
        request_id,
        prompt_token_ids: Some(prompt_token_ids),
        mm_features: None,
        sampling_params: Some(EngineCoreSamplingParams {
            max_tokens,
            logprobs: Some(logprobs),
            prompt_logprobs: Some(prompt_logprobs),
            ..EngineCoreSamplingParams::for_test()
        }),
        pooling_params: None,
        arrival_time: 0.0,
        lora_request: None,
        cache_salt: None,
        data_parallel_rank: None,
        prompt_embeds: None,
        client_index,
        current_wave: 0,
        priority: 0,
        trace_headers: None,
        resumable: false,
        external_req_id: None,
        reasoning_ended: None,
    }
}

async fn wait_for_final_output(
    mut stream: vllm_engine_core_client::EngineCoreOutputStream,
) -> Result<EngineCoreStreamOutput> {
    while let Some(output) = stream.next().await {
        let output = output.context("failed to receive engine-core output")?;
        if output.finished() {
            return Ok(output);
        }
    }
    bail!("request stream ended without a final output")
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    init_tracing();
    let args = Args::parse();
    let ready_timeout = Duration::from_secs(args.ready_timeout_secs);
    let output_timeout = Duration::from_secs(args.output_timeout_secs);
    let request_id = unique_request_id();
    let prompt_token_ids = build_prompt_token_ids(args.prompt_repeats);
    let client = EngineCoreClient::connect(EngineCoreClientConfig {
        handshake_address: args.handshake_address.clone(),
        model_name: args.model.clone(),
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
    println!("ready_message={:?}", client.ready_message);

    let request = build_request(
        request_id.clone(),
        prompt_token_ids.clone(),
        args.max_tokens,
        args.logprobs,
        args.prompt_logprobs,
        args.client_index,
    );
    println!("request_id={request_id}");
    println!("prompt_len={}", prompt_token_ids.len());
    println!("base_prompt_len={}", BASE_PROMPT_TOKEN_IDS.len());
    println!("prompt_repeats={}", args.prompt_repeats);
    println!("requested_logprobs={}", args.logprobs);
    println!("requested_prompt_logprobs={}", args.prompt_logprobs);

    let stream = client
        .call(request)
        .await
        .context("failed to submit engine-core request")?;
    let output = timeout(output_timeout, wait_for_final_output(stream))
        .await
        .context("timed out waiting for final output")??;

    let finish_reason = output.finish_reason;
    let token_ids = output.new_token_ids.clone();
    let logprobs = output
        .new_logprobs
        .as_ref()
        .and_then(|value| value.as_direct())
        .context("engine output did not include decoded sample logprobs")?;
    let prompt_logprobs = output
        .new_prompt_logprobs_tensors
        .as_ref()
        .and_then(|value| value.as_direct())
        .context("engine output did not include decoded prompt logprobs")?;

    println!("token_ids={token_ids:?}");
    println!("finish_reason={finish_reason:?}");
    println!("new_logprobs={logprobs:#?}");
    println!("new_prompt_logprobs_tensors={prompt_logprobs:#?}");

    client
        .shutdown()
        .await
        .context("failed to shut down engine-core client")?;

    if finish_reason != Some(EngineCoreFinishReason::Length) {
        bail!("unexpected finish_reason: expected Length, got {finish_reason:?}");
    }
    if token_ids.is_empty() {
        bail!("engine returned no generated token ids");
    }
    if logprobs.is_empty() {
        bail!("decoded logprobs payload is unexpectedly empty");
    }
    if prompt_logprobs.is_empty() {
        bail!("decoded prompt logprobs payload is unexpectedly empty");
    }
    if prompt_logprobs.len() + 1 < prompt_token_ids.len() {
        bail!(
            "prompt logprobs rows look too short: prompt_len={}, rows={}",
            prompt_token_ids.len(),
            prompt_logprobs.len()
        );
    }

    Ok(())
}
