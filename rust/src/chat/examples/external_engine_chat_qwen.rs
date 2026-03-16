use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::Parser;
use futures::StreamExt as _;
use tracing_subscriber::EnvFilter;
use vllm_chat::{
    ChatEvent, ChatLlm, ChatMessage, ChatOptions, ChatRequest, ChatRole, SmgChatBackend,
};
use vllm_engine_core_client::protocol::SamplingParams;
use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig};
use vllm_llm::Llm;

#[derive(Debug, Parser)]
#[command(about = "Smoke-test the Rust chat facade against an external Qwen vLLM engine.")]
struct Args {
    #[arg(long)]
    handshake_address: String,
    #[arg(long, default_value = "Qwen/Qwen3-0.6B")]
    model: String,
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 30)]
    ready_timeout_secs: u64,
    #[arg(long)]
    prompt: String,
}

const CLIENT_INDEX: u32 = 0;
const OUTPUT_TIMEOUT_SECS: u64 = 120;
const MAX_TOKENS: u32 = 16;

fn unique_request_id() -> String {
    format!("rust-chat-smoke-{}", uuid::Uuid::new_v4())
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
    let backend = std::sync::Arc::new(
        SmgChatBackend::from_model_or_path(&args.model)
            .await
            .with_context(|| format!("failed to load chat backend for {}", args.model))?,
    );

    let ready_timeout = Duration::from_secs(args.ready_timeout_secs);
    let output_timeout = Duration::from_secs(OUTPUT_TIMEOUT_SECS);
    let request_id = unique_request_id();
    let client = EngineCoreClient::connect(EngineCoreClientConfig {
        handshake_address: args.handshake_address.clone(),
        local_host: args.host.clone(),
        ready_timeout,
        client_index: CLIENT_INDEX,
    })
    .await
    .context("failed to connect to external vLLM engine")?;

    println!("model={}", args.model);
    println!("tokenizer_source=smg_tokenizer (crates.io llm-tokenizer) auto");
    println!("chat_template_source=smg_tokenizer (crates.io llm-tokenizer) auto");
    println!("handshake_address={}", args.handshake_address);
    println!("input_address={}", client.input_address());
    println!("output_address={}", client.output_address());
    println!("engine_identity={:x?}", client.engine_identity());
    println!("ready_message={:?}", client.ready_message);

    let llm = Llm::new(client);
    let chat = ChatLlm::new(llm, backend);

    let request = ChatRequest {
        request_id: request_id.clone(),
        messages: vec![ChatMessage::text(ChatRole::User, args.prompt.clone())],
        sampling_params: SamplingParams {
            max_tokens: Some(MAX_TOKENS),
            temperature: 0.0,
            ..Default::default()
        },
        chat_options: ChatOptions::default(),
    };

    println!("request_id={request_id}");
    println!("prompt={}", args.prompt);

    let mut stream = chat
        .chat(request)
        .await
        .context("failed to submit chat request")?;
    let output = tokio::time::timeout(output_timeout, async {
        let mut final_text = String::new();
        let mut finish_reason = None;
        let mut saw_start = false;

        while let Some(event) = stream.next().await.transpose()? {
            match event {
                ChatEvent::Start => {
                    saw_start = true;
                }
                ChatEvent::TextDelta { delta, .. } => {
                    print!("{delta}");
                }
                ChatEvent::Done {
                    text,
                    finish_reason: reason,
                    ..
                } => {
                    final_text = text;
                    finish_reason = reason;
                    break;
                }
            }
        }

        println!();

        if !saw_start {
            bail!("chat stream ended without a start event");
        }
        Ok::<_, anyhow::Error>((final_text, finish_reason))
    })
    .await
    .context("timed out waiting for chat output")??;

    chat.shutdown()
        .await
        .context("failed to shut down chat client")?;

    println!("final_text={:?}", output.0);
    println!("finish_reason={:?}", output.1);

    Ok(())
}
