use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::Parser;
use futures::StreamExt as _;
use tracing_subscriber::EnvFilter;
use vllm_chat::{
    AssistantBlockKind, AssistantMessageExt as _, ChatEvent, ChatLlm, ChatMessage, ChatOptions,
    ChatRequest, ChatRole, ChatToolChoice, SamplingParams, load_model_backends,
};
use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig};
use vllm_llm::Llm;
use vllm_text::TextLlm;

#[derive(Debug, Parser)]
#[command(about = "Smoke-test the Rust chat facade against an external Qwen vLLM engine.")]
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
    #[arg(long)]
    prompt: String,
}

const CLIENT_INDEX: u32 = 0;
const OUTPUT_TIMEOUT_SECS: u64 = 120;

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
    let loaded = load_model_backends(&args.model)
        .await
        .with_context(|| format!("failed to load backends for {}", args.model))?;
    let text_backend = loaded.text_backend;
    let chat_backend = loaded.chat_backend;

    let ready_timeout = Duration::from_secs(args.ready_timeout_secs);
    let output_timeout = Duration::from_secs(OUTPUT_TIMEOUT_SECS);
    let request_id = unique_request_id();
    let client = EngineCoreClient::connect(EngineCoreClientConfig {
        handshake_address: args.handshake_address.clone(),
        engine_count: args.engine_count,
        model_name: args.model.clone(),
        local_host: args.host.clone(),
        ready_timeout,
        client_index: CLIENT_INDEX,
        enable_inproc_coordinator: false,
    })
    .await
    .context("failed to connect to external vLLM engine")?;

    println!("model={}", args.model);
    println!("tokenizer_source=tokenizers + hf-hub");
    println!("chat_template_source=tokenizer_config.json or adjacent chat template file");
    println!("handshake_address={}", args.handshake_address);
    println!("engine_count={}", args.engine_count);
    println!("input_address={}", client.input_address());
    println!("output_address={}", client.output_address());
    println!("engine_identities={:x?}", client.engine_identities());

    let llm = Llm::new(client);
    let chat = ChatLlm::new(TextLlm::new(llm, text_backend), chat_backend);

    let request = ChatRequest {
        request_id: request_id.clone(),
        messages: vec![ChatMessage::text(ChatRole::User, args.prompt.clone())],
        sampling_params: SamplingParams {
            temperature: Some(0.0),
            ..Default::default()
        },
        chat_options: ChatOptions::default(),
        tools: Vec::new(),
        tool_choice: ChatToolChoice::None,
        decode_options: Default::default(),
        intermediate: true,
        priority: 0,
    };

    println!("request_id={request_id}");
    println!("prompt={}", args.prompt);

    let mut stream = chat
        .chat(request)
        .await
        .context("failed to submit chat request")?;
    let output = tokio::time::timeout(output_timeout, async {
        let mut final_reasoning = String::new();
        let mut final_text = String::new();
        let mut final_output_token_count = 0usize;
        let mut finish_reason = None;
        let mut saw_start = false;
        let mut saw_stream_output = false;

        while let Some(event) = stream.next().await.transpose()? {
            match event {
                ChatEvent::Start { .. } => {
                    saw_start = true;
                }
                ChatEvent::BlockStart { kind, .. } => {
                    if saw_stream_output {
                        println!();
                    }
                    match kind {
                        AssistantBlockKind::Reasoning => print!("[reasoning] "),
                        AssistantBlockKind::Text => print!("[answer] "),
                        AssistantBlockKind::ToolCall => {}
                    }
                    saw_stream_output = true;
                }
                ChatEvent::ToolCallStart { name, .. } => {
                    if saw_stream_output {
                        println!();
                    }
                    print!("[tool:{name}] ");
                    saw_stream_output = true;
                }
                ChatEvent::LogprobsDelta { .. } => {}
                ChatEvent::Done {
                    message,
                    output_token_count,
                    finish_reason: reason,
                    ..
                } => {
                    final_reasoning = message.reasoning().unwrap_or_default();
                    final_text = message.text();
                    final_output_token_count = output_token_count;
                    finish_reason = Some(reason);
                    break;
                }
                ChatEvent::BlockDelta { kind, delta, .. } => match kind {
                    AssistantBlockKind::Reasoning | AssistantBlockKind::Text => {
                        print!("{delta}");
                    }
                    AssistantBlockKind::ToolCall => {}
                },
                ChatEvent::ToolCallArgumentsDelta { delta, .. } => print!("{delta}"),
                ChatEvent::BlockEnd { .. } | ChatEvent::ToolCallEnd { .. } => {}
            }
        }

        println!();

        if !saw_start {
            bail!("chat stream ended without a start event");
        }
        Ok::<_, anyhow::Error>((
            final_reasoning,
            final_text,
            final_output_token_count,
            finish_reason,
        ))
    })
    .await
    .context("timed out waiting for chat output")??;

    chat.shutdown()
        .await
        .context("failed to shut down chat client")?;

    println!("final_reasoning={:?}", output.0);
    println!("final_text={:?}", output.1);
    println!("final_output_token_count={:?}", output.2);
    println!("finish_reason={:?}", output.3);

    Ok(())
}
