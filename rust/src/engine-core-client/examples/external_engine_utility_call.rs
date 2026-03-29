use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::Parser;
use tracing_subscriber::EnvFilter;
use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig};

#[derive(Debug, Parser)]
#[command(about = "Smoke-test EngineCoreClient utility calls against an external vLLM engine.")]
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
    #[arg(
        long,
        default_value_t = false,
        help = "Expected initial result of is_sleeping() before running the smoke steps."
    )]
    expected_is_sleeping: bool,
    #[arg(long, default_value_t = false)]
    reset_running_requests: bool,
    #[arg(long, default_value_t = false)]
    reset_external: bool,
    #[arg(long, default_value_t = 1)]
    sleep_level: u32,
    #[arg(long, default_value = "abort")]
    sleep_mode: String,
    #[arg(
        long,
        default_value_t = false,
        help = "Skip sleep/wake_up calls when the engine was not started with sleep-mode support."
    )]
    skip_sleep_wake: bool,
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
        engine_count: args.engine_count,
        model_name: args.model.clone(),
        local_host: args.host.clone(),
        ready_timeout: Duration::from_secs(args.ready_timeout_secs),
        client_index: args.client_index,
        enable_inproc_coordinator: false,
    })
    .await
    .context("failed to connect to external vLLM engine")?;

    println!("model={}", args.model);
    println!("handshake_address={}", args.handshake_address);
    println!("engine_count={}", args.engine_count);
    println!("input_address={}", client.input_address());
    println!("output_address={}", client.output_address());
    println!("engine_identities={:x?}", client.engine_identities());

    let initial_is_sleeping = client
        .is_sleeping()
        .await
        .context("failed to call is_sleeping utility")?;

    println!("is_sleeping(initial)={initial_is_sleeping}");

    if initial_is_sleeping != args.expected_is_sleeping {
        bail!(
            "unexpected initial is_sleeping state: expected {}, got {}",
            args.expected_is_sleeping,
            initial_is_sleeping
        );
    }

    let reset_prefix_cache = client
        .reset_prefix_cache(args.reset_running_requests, args.reset_external)
        .await
        .context("failed to call reset_prefix_cache utility")?;
    println!("reset_prefix_cache={reset_prefix_cache}");

    client
        .reset_mm_cache()
        .await
        .context("failed to call reset_mm_cache utility")?;
    println!("reset_mm_cache=ok");

    client
        .reset_encoder_cache()
        .await
        .context("failed to call reset_encoder_cache utility")?;
    println!("reset_encoder_cache=ok");

    if args.skip_sleep_wake {
        println!("sleep_wake=skipped");
    } else {
        client
            .sleep(args.sleep_level, &args.sleep_mode)
            .await
            .with_context(|| {
                format!(
                    "failed to call sleep utility with level={} mode={}",
                    args.sleep_level, args.sleep_mode
                )
            })?;
        println!(
            "sleep=ok level={} mode={}",
            args.sleep_level, args.sleep_mode
        );

        let sleeping_after_sleep = client
            .is_sleeping()
            .await
            .context("failed to call is_sleeping after sleep")?;
        println!("is_sleeping(after_sleep)={sleeping_after_sleep}");

        if !sleeping_after_sleep {
            bail!("engine should report sleeping=true after sleep()");
        }

        client
            .wake_up(None)
            .await
            .context("failed to call wake_up utility")?;
        println!("wake_up=ok");

        let sleeping_after_wake = client
            .is_sleeping()
            .await
            .context("failed to call is_sleeping after wake_up")?;
        println!("is_sleeping(after_wake)={sleeping_after_wake}");

        if sleeping_after_wake {
            bail!("engine should report sleeping=false after wake_up()");
        }
    }

    client
        .shutdown()
        .await
        .context("failed to shut down engine-core client")?;

    Ok(())
}
