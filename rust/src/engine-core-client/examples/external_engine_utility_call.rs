use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::Parser;
use tracing_subscriber::EnvFilter;
use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig};

#[derive(Debug, Parser)]
#[command(about = "Smoke-test EngineCoreClient::is_sleeping() against an external vLLM engine.")]
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
    #[arg(long, default_value_t = false)]
    expected_is_sleeping: bool,
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("vllm_engine_core_client=debug"));
    let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    init_tracing();
    let args = Args::parse();
    let client = EngineCoreClient::connect(EngineCoreClientConfig {
        handshake_address: args.handshake_address.clone(),
        model_name: args.model.clone(),
        local_host: args.host.clone(),
        ready_timeout: Duration::from_secs(args.ready_timeout_secs),
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

    let is_sleeping = client
        .is_sleeping()
        .await
        .context("failed to call is_sleeping utility")?;

    println!("is_sleeping={is_sleeping}");

    client
        .shutdown()
        .await
        .context("failed to shut down engine-core client")?;

    if is_sleeping != args.expected_is_sleeping {
        bail!(
            "unexpected is_sleeping state: expected {}, got {}",
            args.expected_is_sleeping,
            is_sleeping
        );
    }

    Ok(())
}
