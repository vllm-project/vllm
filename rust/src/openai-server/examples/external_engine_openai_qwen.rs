use std::time::Duration;

use anyhow::{Context, Result, bail};
use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_openai::types::chat::CreateChatCompletionStreamResponse;
use async_openai::types::models::ListModelResponse;
use clap::Parser;
use futures::StreamExt as _;
use serde_json::json;
use tokio::sync::oneshot;
use tracing_subscriber::EnvFilter;
use vllm_openai_server::{Config, serve};

#[derive(Debug, Parser)]
#[command(
    about = "Smoke-test the Rust OpenAI server with async-openai against an external Qwen vLLM engine."
)]
struct Args {
    #[arg(long)]
    handshake_address: String,
    #[arg(long, default_value = "Qwen/Qwen3-0.6B")]
    model: String,
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 30)]
    ready_timeout_secs: u64,
    #[arg(
        long,
        default_value = "Write exactly 4 short numbered facts about Paris. Each fact must be on its own line. Do not add any introduction or conclusion."
    )]
    prompt: String,
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    init_tracing();
    let args = Args::parse();
    let port = unique_local_port()?;
    let config = Config {
        handshake_address: args.handshake_address,
        model: args.model,
        bind_host: "127.0.0.1".to_string(),
        port,
        engine_local_host: args.host,
        ready_timeout: Duration::from_secs(args.ready_timeout_secs),
    };

    let bind_address = config.bind_address();
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let server_config = config.clone();
    let server_task = tokio::spawn(async move {
        serve(server_config, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_key("unused")
            .with_api_base(format!("http://{bind_address}/v1")),
    );

    print_models(&client).await?;
    let final_text = stream_completion(&client, &config.model, &args.prompt).await?;

    println!();
    println!("final_text={final_text:?}");

    shutdown(server_task, shutdown_tx).await
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
}

fn unique_local_port() -> Result<u16> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0")
        .context("failed to allocate local smoke-test port")?;
    let port = listener
        .local_addr()
        .context("failed to read local smoke-test port")?
        .port();
    drop(listener);
    Ok(port)
}

async fn print_models(client: &Client<OpenAIConfig>) -> Result<()> {
    let models = wait_for_models(client).await?;
    let model_ids = models
        .data
        .into_iter()
        .map(|model| model.id)
        .collect::<Vec<_>>();
    println!("models={model_ids:?}");
    Ok(())
}

async fn wait_for_models(client: &Client<OpenAIConfig>) -> Result<ListModelResponse> {
    let mut last_error = None;
    for _ in 0..240 {
        match client.models().list().await {
            Ok(models) => return Ok(models),
            Err(error) => {
                last_error = Some(error);
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
        }
    }

    match last_error {
        Some(error) => Err(error).context("OpenAI server did not become ready in time"),
        None => bail!("OpenAI server readiness loop finished without a result"),
    }
}

async fn stream_completion(
    client: &Client<OpenAIConfig>,
    model: &str,
    prompt: &str,
) -> Result<String> {
    // We have to pass a custom request here to set the `chat_template_kwargs` to disable thinking,
    // which is not yet supported by the minimal Rust frontend.
    let request = json!({
        "model": model,
        "stream": true,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": 0.0,
        "max_completion_tokens": 128,
        "chat_template_kwargs": {
            "enable_thinking": false
        }
    });

    let mut stream = client
        .chat()
        .create_stream_byot::<_, CreateChatCompletionStreamResponse>(request)
        .await
        .context("failed to create streaming chat completion")?;

    let mut final_text = String::new();
    let mut saw_role = false;
    let mut saw_finish_reason = false;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("streaming chat completion failed")?;
        for choice in chunk.choices {
            if choice.delta.role.is_some() {
                saw_role = true;
            }
            if let Some(delta) = choice.delta.content {
                print!("{delta}");
                final_text.push_str(&delta);
            }
            if choice.finish_reason.is_some() {
                saw_finish_reason = true;
            }
        }
    }

    if !saw_role {
        bail!("stream ended without an assistant role chunk");
    }
    if !saw_finish_reason {
        bail!("stream ended without a terminal finish reason");
    }
    if final_text.is_empty() {
        bail!("stream ended without any content deltas");
    }

    Ok(final_text)
}

async fn shutdown(
    server_task: tokio::task::JoinHandle<anyhow::Result<()>>,
    shutdown_tx: oneshot::Sender<()>,
) -> Result<()> {
    let _ = shutdown_tx.send(());
    server_task
        .await
        .context("server task join failed")?
        .context("server task failed")
}
