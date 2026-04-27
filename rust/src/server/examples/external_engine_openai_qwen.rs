use std::time::Duration;

use anyhow::{Context, Result, bail};
use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_openai::types::chat::{
    ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequest,
    CreateChatCompletionRequestArgs,
};
use async_openai::types::models::ListModelResponse;
use clap::Parser;
use futures::StreamExt as _;
use tokio_util::sync::CancellationToken;
use tracing_subscriber::EnvFilter;
use vllm_engine_core_client::TransportMode;
use vllm_server::{
    ChatTemplateContentFormatOption, Config, CoordinatorMode, HttpListenerMode, ParserSelection,
    RendererSelection, serve,
};

#[derive(Debug, Parser)]
#[command(
    about = "Smoke-test the Rust OpenAI server with async-openai against an external Qwen vLLM engine."
)]
struct Args {
    #[arg(long)]
    handshake_address: String,
    #[arg(long, default_value_t = 1)]
    engine_count: usize,
    #[arg(long, default_value = "Qwen/Qwen3-0.6B")]
    model: String,
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 30)]
    ready_timeout_secs: u64,
    #[arg(
        long,
        default_value = "What is the capital of France? Answer with one word."
    )]
    prompt: String,
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    init_tracing();
    let args = Args::parse();
    let port = unique_local_port()?;
    let config = Config {
        transport_mode: TransportMode::HandshakeOwner {
            handshake_address: args.handshake_address,
            advertised_host: args.host,
            engine_count: args.engine_count,
            ready_timeout: Duration::from_secs(args.ready_timeout_secs),
            local_input_address: None,
            local_output_address: None,
        },
        coordinator_mode: CoordinatorMode::MaybeInProc,
        model: args.model,
        listener_mode: HttpListenerMode::BindTcp {
            host: "127.0.0.1".to_string(),
            port,
        },
        tool_call_parser: ParserSelection::Auto,
        reasoning_parser: ParserSelection::Auto,
        renderer: RendererSelection::Auto,
        chat_template: None,
        default_chat_template_kwargs: None,
        chat_template_content_format: ChatTemplateContentFormatOption::Auto,
        enable_log_requests: false,
        disable_log_stats: false,
        grpc_port: None,
        shutdown_timeout: Duration::ZERO,
    };

    let bind_address = format!("127.0.0.1:{port}");
    let shutdown = CancellationToken::new();
    let server_config = config.clone();
    let server_shutdown = shutdown.clone();
    let server_task = tokio::spawn(async move { serve(server_config, server_shutdown).await });

    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_key("unused")
            .with_api_base(format!("http://{bind_address}/v1")),
    );

    print_models(&client).await?;
    let final_text = stream_completion(&client, &config.model, &args.prompt).await?;

    println!();
    println!("final_text={final_text:?}");

    shutdown_server(server_task, shutdown).await
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
    // Keep this smoke test on async-openai's standard `create_stream` path so it exercises
    // the ordinary typed chat-completions client without BYOT request/response types.
    //
    // The current async-openai chat-completions stream delta type does not expose our
    // OpenAI-compatible `reasoning_content` extension field, so this example only validates the
    // assistant role chunk, visible `content` deltas, and terminal finish chunk. Reasoning
    // coverage lives in our own route tests and in the `vllm-chat` smoke example.
    let request: CreateChatCompletionRequest = CreateChatCompletionRequestArgs::default()
        .model(model)
        .stream(true)
        .temperature(0.0)
        .max_completion_tokens(128u32)
        .messages([ChatCompletionRequestUserMessageArgs::default()
            .content(prompt)
            .build()
            .context("failed to build user chat message")?
            .into()])
        .build()
        .context("failed to build chat completion request")?;

    let mut stream = client
        .chat()
        .create_stream(request)
        .await
        .context("failed to create streaming chat completion")?;

    let mut final_text = String::new();
    let mut saw_role = false;
    let mut saw_finish_reason = false;
    let mut saw_text = false;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("streaming chat completion failed")?;
        for choice in chunk.choices {
            if choice.delta.role.is_some() {
                saw_role = true;
            }
            if let Some(delta) = choice.delta.content {
                if !saw_text {
                    print!("[answer] ");
                }
                print!("{delta}");
                final_text.push_str(&delta);
                saw_text = true;
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

async fn shutdown_server(
    server_task: tokio::task::JoinHandle<anyhow::Result<()>>,
    shutdown: CancellationToken,
) -> Result<()> {
    shutdown.cancel();
    server_task
        .await
        .context("server task join failed")?
        .context("server task failed")
}
