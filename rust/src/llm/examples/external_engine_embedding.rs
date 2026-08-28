// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::time::Duration;

use anyhow::{Context, Result, ensure};
use clap::Parser;
use tokio::time::timeout;
use tracing_subscriber::EnvFilter;
use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig, TransportMode};
use vllm_llm::{EncodeRequest, EngineTask, Llm, PoolingParams, PoolingTask};

const PROMPT: &str = "Represent this sentence for semantic search: a small cat sleeps on a sofa.";
const PROMPT_TOKEN_IDS: &[u32] = &[
    65743, 419, 11652, 369, 41733, 2711, 25, 264, 2613, 8251, 71390, 389, 264, 31069, 13, 151643,
];

#[derive(Debug, Parser)]
#[command(about = "Run a Rust LLM embedding request against an external vLLM engine.")]
struct Args {
    #[arg(long)]
    handshake_address: String,
    #[arg(long, default_value_t = 1)]
    engine_count: usize,
    #[arg(long, default_value = "Qwen/Qwen3-Embedding-0.6B")]
    model: String,
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 0)]
    client_index: u32,
    #[arg(long, default_value_t = 120)]
    ready_timeout_secs: u64,
    #[arg(long, default_value_t = 120)]
    output_timeout_secs: u64,
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("vllm_engine_core_client=info"));
    let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
}

fn build_request() -> EncodeRequest {
    EncodeRequest {
        request_id: format!("rust-embedding-{}", uuid::Uuid::new_v4()),
        prompt_token_ids: PROMPT_TOKEN_IDS.to_vec(),
        task: PoolingTask::Embed,
        pooling_params: PoolingParams {
            use_activation: true,
            dimensions: None,
            step_tag_id: None,
            returned_token_ids: None,
        },
        arrival_time: None,
        cache_salt: None,
        trace_headers: None,
        priority: 0,
        data_parallel_rank: None,
        session_id: None,
        lora_request: None,
    }
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    init_tracing();
    let args = Args::parse();
    let client = EngineCoreClient::connect(EngineCoreClientConfig {
        transport_mode: TransportMode::HandshakeOwner {
            handshake_address: args.handshake_address,
            advertised_host: args.host,
            engine_count: args.engine_count,
            ready_timeout: Duration::from_secs(args.ready_timeout_secs),
            local_input_address: None,
            local_output_address: None,
        },
        coordinator_mode: None,
        model_name: args.model.clone(),
        client_index: args.client_index,
    })
    .await
    .context("failed to connect to external vLLM engine")?;
    let supported_tasks = client
        .get_supported_tasks()
        .await
        .context("failed to discover supported engine tasks")?
        .to_vec();
    ensure!(
        supported_tasks.contains(&EngineTask::Pooling(PoolingTask::Embed)),
        "model does not support the embedding task: {supported_tasks:?}"
    );
    let llm = Llm::new(client);

    let output = timeout(
        Duration::from_secs(args.output_timeout_secs),
        llm.encode(build_request()),
    )
    .await
    .context("timed out waiting for embedding output")??;
    llm.shutdown().await.context("failed to shut down LLM client")?;

    let l2_norm = output
        .output
        .data
        .iter()
        .map(|&value| f64::from(value).powi(2))
        .sum::<f64>()
        .sqrt();
    let preview_len = output.output.data.len().min(8);

    println!("model={}", args.model);
    println!("supported_tasks={supported_tasks:?}");
    println!("prompt={PROMPT:?}");
    println!("prompt_token_count={}", output.prompt_token_ids.len());
    println!("cached_token_count={}", output.cached_token_count);
    println!("embedding_shape={:?}", output.output.shape);
    println!("embedding_l2_norm={l2_norm:.8}");
    println!("embedding_preview={:?}", &output.output.data[..preview_len]);

    Ok(())
}
