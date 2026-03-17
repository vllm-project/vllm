use futures::FutureExt as _;
use tokio::signal::ctrl_c;
use tracing_subscriber::EnvFilter;
use vllm_openai_server::{Config, serve};

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("debug"));
    let _ = tracing_subscriber::fmt()
        .with_file(true)
        .with_line_number(true)
        .with_env_filter(filter)
        .try_init();
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    init_tracing();
    let config = Config::parse();
    let shutdown = ctrl_c().map(|_| ());
    serve(config, shutdown).await
}
