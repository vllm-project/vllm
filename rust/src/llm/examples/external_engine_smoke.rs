use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::Parser;
use futures::StreamExt as _;
use tokio::time::timeout;
use tracing_subscriber::EnvFilter;
use vllm_engine_core_client::protocol::EngineCoreSamplingParams;
use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig, TransportMode};
use vllm_llm::{FinishReason, GenerateOutputStream, GenerateRequest, Llm};

const PROMPT_TOKEN_IDS: &[u32] = &[20841, 448, 6896, 25, 23811];

#[derive(Debug, Parser)]
#[command(about = "Smoke-test the Rust LLM facade against an external vLLM engine.")]
struct Args {
    #[arg(long)]
    handshake_address: String,
    #[arg(long, default_value_t = 1)]
    engine_count: usize,
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

fn unique_request_id() -> String {
    format!("rust-llm-smoke-{}", uuid::Uuid::new_v4())
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("vllm_engine_core_client=debug"));
    let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
}

fn build_request(request_id: String, max_tokens: u32) -> GenerateRequest {
    GenerateRequest {
        request_id,
        prompt_token_ids: PROMPT_TOKEN_IDS.to_vec(),
        sampling_params: EngineCoreSamplingParams {
            max_tokens,
            ..EngineCoreSamplingParams::for_test()
        },
        arrival_time: None,
        cache_salt: None,
        trace_headers: None,
        priority: 0,
        data_parallel_rank: None,
        reasoning_ended: None,
        lora_request: None,
    }
}

#[derive(Debug)]
struct CompletedRequest {
    token_ids: Vec<u32>,
    finish_reason: FinishReason,
}

async fn wait_for_request_completion(mut stream: GenerateOutputStream) -> Result<CompletedRequest> {
    let output = match stream.next().await {
        Some(output) => output.context("failed to receive request output")?,
        None => bail!("request stream ended without a final output"),
    };

    let none = stream.next().await;
    assert!(
        none.is_none(),
        "expected final-only stream to end after the final output"
    );

    let finish_reason = output
        .finish_reason
        .expect("final-only output must have a finish reason");
    let token_ids = output.token_ids;

    Ok(CompletedRequest {
        token_ids,
        finish_reason,
    })
}

async fn wait_for_timeout(
    stream: GenerateOutputStream,
    output_timeout: Duration,
) -> Result<CompletedRequest> {
    timeout(output_timeout, wait_for_request_completion(stream))
        .await
        .context("timed out waiting for request output")?
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    init_tracing();
    let args = Args::parse();
    let ready_timeout = Duration::from_secs(args.ready_timeout_secs);
    let output_timeout = Duration::from_secs(args.output_timeout_secs);
    let request_id = unique_request_id();
    let client = EngineCoreClient::connect(EngineCoreClientConfig {
        transport_mode: TransportMode::HandshakeOwner {
            handshake_address: args.handshake_address.clone(),
            advertised_host: args.host.clone(),
            engine_count: args.engine_count,
            ready_timeout,
            local_input_address: None,
            local_output_address: None,
        },
        coordinator_mode: None,
        model_name: args.model.clone(),
        client_index: args.client_index,
    })
    .await
    .context("failed to connect to external vLLM engine")?;

    println!("model={}", args.model);
    println!("handshake_address={}", args.handshake_address);
    println!("engine_count={}", args.engine_count);
    println!("input_address={}", client.input_address());
    println!("output_address={}", client.output_address());
    println!("engine_identities={:x?}", client.engine_identities());

    let llm = Llm::new(client);
    let request = build_request(request_id.clone(), args.max_tokens);
    println!("request_id={request_id}");
    println!("prompt_token_ids={PROMPT_TOKEN_IDS:?}");

    let stream = llm
        .generate(request)
        .await
        .context("failed to submit generate request")?;
    let output = wait_for_timeout(stream, output_timeout).await?;

    llm.shutdown()
        .await
        .context("failed to shut down llm client")?;

    println!("token_ids={:?}", output.token_ids);
    println!("finish_reason={:?}", output.finish_reason);

    Ok(())
}
