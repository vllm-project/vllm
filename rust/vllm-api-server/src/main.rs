mod generated;
mod grpc;
mod openai;
mod routes;
mod server;

use clap::Parser;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use grpc::VllmClient;
use server::AppState;

#[derive(Parser, Debug)]
#[command(name = "vllm-api-server")]
#[command(about = "Rust API server for vLLM")]
struct Args {
    /// HTTP port to listen on
    #[arg(long, default_value = "8000")]
    port: u16,

    /// gRPC server address
    #[arg(long, default_value = "localhost:50051")]
    grpc_addr: String,

    /// Model name (for tokenizer and response metadata)
    #[arg(long)]
    model: String,

    /// Override chat template (Jinja2 format)
    #[arg(long)]
    chat_template: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let args = Args::parse();

    tracing::info!("Loading tokenizer for model: {}", args.model);
    let tokenizer = Tokenizer::from_pretrained(&args.model, None)
        .map_err(|e| format!("Failed to load tokenizer: {}", e))?;

    // Try to get chat template from tokenizer config
    let chat_template = args.chat_template.or_else(|| {
        // Note: tokenizers crate doesn't directly expose chat_template
        // For now, we use the fallback template
        tracing::warn!("Using fallback chat template - consider providing --chat-template");
        None
    });

    tracing::info!("Connecting to gRPC server at {}", args.grpc_addr);
    let grpc_client = VllmClient::connect(&args.grpc_addr).await?;

    let state = Arc::new(AppState::new(
        tokenizer,
        grpc_client,
        args.model,
        chat_template,
    ));

    let addr = format!("0.0.0.0:{}", args.port);
    server::run_server(&addr, state).await?;

    Ok(())
}
