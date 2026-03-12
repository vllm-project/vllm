use std::io::{Error as IoError, ErrorKind};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use clap::Parser;
use tokio::time::timeout;
use vllm_engine_core_client::{
    EngineCoreClient, EngineCoreOutput, EngineCoreRequest, RequestOutputKind, SamplingParams,
    ZmqEngineCoreClient, ZmqEngineCoreClientConfig,
};

const MODEL_NAME: &str = "Qwen/Qwen3-0.6B";
const PROMPT_TOKEN_IDS: &[u32] = &[20841, 448, 6896, 25, 23811];

#[derive(Debug, Parser)]
#[command(about = "Smoke-test a Rust EngineCoreClient against an external vLLM engine.")]
struct Args {
    #[arg(long)]
    handshake_address: String,
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
}

fn invalid_input(message: String) -> IoError {
    IoError::new(ErrorKind::InvalidInput, message)
}

fn unix_timestamp_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is before unix epoch")
        .as_secs_f64()
}

fn unique_request_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is before unix epoch")
        .as_nanos();
    format!("rust-engine-core-smoke-{nanos}")
}

fn build_request(request_id: String, max_tokens: u32) -> EngineCoreRequest {
    EngineCoreRequest {
        request_id,
        prompt_token_ids: Some(PROMPT_TOKEN_IDS.to_vec()),
        mm_features: None,
        sampling_params: Some(SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            max_tokens: Some(max_tokens),
            output_kind: RequestOutputKind::FinalOnly,
            ..Default::default()
        }),
        pooling_params: None,
        arrival_time: unix_timestamp_secs(),
        lora_request: None,
        cache_salt: None,
        data_parallel_rank: None,
        prompt_embeds: None,
        client_index: 0,
        current_wave: 0,
        priority: 0,
        trace_headers: None,
        resumable: false,
        external_req_id: None,
        reasoning_ended: None,
    }
}

async fn wait_for_request_output(
    client: &mut ZmqEngineCoreClient,
    request_id: &str,
) -> vllm_engine_core_client::Result<EngineCoreOutput> {
    loop {
        let batch = client.next_output().await?;
        if let Some(output) = batch
            .outputs
            .into_iter()
            .find(|output| output.request_id == request_id)
        {
            return Ok(output);
        }
    }
}

async fn wait_for_timeout(
    client: &mut ZmqEngineCoreClient,
    request_id: &str,
    output_timeout: Duration,
) -> Result<EngineCoreOutput, Box<dyn std::error::Error>> {
    Ok(
        timeout(output_timeout, wait_for_request_output(client, request_id))
            .await
            .map_err(|_| {
                invalid_input(format!(
                    "timed out waiting {:?} for output of request {request_id}",
                    output_timeout
                ))
            })??,
    )
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let ready_timeout = Duration::from_secs(args.ready_timeout_secs);
    let output_timeout = Duration::from_secs(args.output_timeout_secs);
    let request_id = unique_request_id();
    let mut client = ZmqEngineCoreClient::connect(ZmqEngineCoreClientConfig {
        handshake_address: args.handshake_address.clone(),
        local_host: args.host.clone(),
        ready_timeout,
        client_index: args.client_index,
    })
    .await?;

    let ready_message = client.ready_message.clone();
    let request = build_request(request_id.clone(), args.max_tokens);
    client.add_request(request).await?;

    let output = wait_for_timeout(&mut client, &request_id, output_timeout).await?;
    let cleanup_output = if output.finished() {
        None
    } else {
        client
            .abort_requests(std::slice::from_ref(&request_id))
            .await?;
        Some(wait_for_timeout(&mut client, &request_id, output_timeout).await?)
    };

    client.shutdown().await?;

    println!("model={MODEL_NAME}");
    println!("handshake_address={}", args.handshake_address);
    println!("input_address={}", client.input_address());
    println!("output_address={}", client.output_address());
    println!(
        "engine_identity_hex={}",
        hex_string(client.engine_identity())
    );
    println!("request_id={request_id}");
    println!("prompt_token_ids={PROMPT_TOKEN_IDS:?}");
    println!("ready_message={ready_message:?}");
    println!("new_token_ids={:?}", output.new_token_ids);
    println!("finish_reason={:?}", output.finish_reason);
    println!("stop_reason={:?}", output.stop_reason);
    if let Some(cleanup_output) = cleanup_output {
        println!("cleanup_finish_reason={:?}", cleanup_output.finish_reason);
    }

    Ok(())
}

fn hex_string(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}
