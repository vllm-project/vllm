//! Integration tests that exercise the OpenAI-compatible HTTP API through a
//! real TCP connection using the `async-openai` client library, backed by a
//! mock engine.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_openai::types::chat::{
    ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs,
};
use futures::StreamExt as _;
use serial_test::serial;
use vllm_chat::{
    ChatBackend, ChatLlm, ChatRenderer, ChatRequest, ChatTextBackend, DefaultChatOutputProcessor,
    DynChatOutputProcessor, DynChatRenderer, NewChatOutputProcessorOptions, RenderedPrompt,
};
use vllm_engine_core_client::protocol::{
    EngineCoreFinishReason, EngineCoreOutput, EngineCoreOutputs, EngineCoreRequest,
};
use vllm_engine_core_client::test_utils::{IpcNamespace, spawn_mock_engine_task};
use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig, EngineId};
use vllm_llm::Llm;
use vllm_text::tokenizer::{DynTokenizer, Tokenizer};
use vllm_text::{Prompt, TextBackend};
use zeromq::prelude::{SocketRecv, SocketSend};
use zeromq::{DealerSocket, PushSocket, ZmqMessage};

use crate::routes::build_router;
use crate::state::AppState;

// ========================================================================================
// Test infrastructure (mirrors routes/tests.rs helpers)
// ========================================================================================

type TestFuture<'a> = Pin<Box<dyn Future<Output = ()> + Send + 'a>>;

fn boxed_test_future<'a>(future: impl Future<Output = ()> + Send + 'a) -> TestFuture<'a> {
    Box::pin(future)
}

struct MockEngineTask {
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
    join_handle: Option<tokio::task::JoinHandle<()>>,
}

impl MockEngineTask {
    fn new(
        (shutdown_tx, join_handle): (
            tokio::sync::oneshot::Sender<()>,
            tokio::task::JoinHandle<()>,
        ),
    ) -> Self {
        Self {
            shutdown_tx: Some(shutdown_tx),
            join_handle: Some(join_handle),
        }
    }
}

impl Future for MockEngineTask {
    type Output = Result<(), tokio::task::JoinError>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if let Some(shutdown_tx) = self.shutdown_tx.take() {
            let _ = shutdown_tx.send(());
        }
        match self.join_handle.as_mut() {
            Some(join_handle) => Pin::new(join_handle).poll(cx),
            None => Poll::Ready(Ok(())),
        }
    }
}

impl Drop for MockEngineTask {
    fn drop(&mut self) {
        if let Some(join_handle) = &self.join_handle {
            join_handle.abort();
        }
    }
}

fn request_output(
    request_id: &str,
    new_token_ids: Vec<u32>,
    finish_reason: Option<EngineCoreFinishReason>,
) -> EngineCoreOutput {
    EngineCoreOutput {
        request_id: request_id.to_string(),
        new_token_ids,
        new_logprobs: None,
        new_prompt_logprobs_tensors: None,
        pooling_output: None,
        finish_reason,
        stop_reason: None,
        events: None,
        kv_transfer_params: None,
        trace_headers: None,
        prefill_stats: None,
        routed_experts: None,
        num_nans_in_logits: 0,
    }
}

fn engine_outputs_for_request(
    request_id: &str,
    output_specs: Vec<(Vec<u32>, Option<EngineCoreFinishReason>)>,
) -> EngineCoreOutputs {
    EngineCoreOutputs {
        engine_index: 0,
        outputs: output_specs
            .into_iter()
            .map(|(token_ids, finish_reason)| request_output(request_id, token_ids, finish_reason))
            .collect(),
        scheduler_stats: None,
        timestamp: 0.0,
        utility_output: None,
        finished_requests: None,
        wave_complete: None,
        start_wave: None,
    }
}

fn default_stream_output_specs() -> Vec<(Vec<u32>, Option<EngineCoreFinishReason>)> {
    vec![
        (vec![b'h' as u32], None),
        (vec![b'i' as u32], None),
        (vec![b'!' as u32], Some(EngineCoreFinishReason::Stop)),
    ]
}

async fn send_outputs(push: &mut PushSocket, outputs: EngineCoreOutputs) {
    push.send(ZmqMessage::from(
        rmp_serde::to_vec_named(&outputs).expect("encode outputs"),
    ))
    .await
    .expect("send outputs");
}

async fn recv_engine_message(dealer: &mut DealerSocket) -> Vec<bytes::Bytes> {
    dealer.recv().await.expect("recv engine message").into_vec()
}

fn test_llm(client: EngineCoreClient) -> Llm {
    Llm::new(client).with_request_id_randomization(false)
}

#[derive(Clone, Debug)]
struct FakeChatBackend;

#[derive(Debug)]
struct FakeChatTokenizer;

impl Tokenizer for FakeChatTokenizer {
    fn encode(&self, text: &str, _add_special_tokens: bool) -> vllm_text::Result<Vec<u32>> {
        Ok(text.bytes().map(u32::from).collect())
    }

    fn decode(&self, token_ids: &[u32], _skip_special_tokens: bool) -> vllm_text::Result<String> {
        Ok(
            String::from_utf8_lossy(&token_ids.iter().map(|id| *id as u8).collect::<Vec<_>>())
                .into_owned(),
        )
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        token.bytes().next().map(u32::from)
    }
}

impl TextBackend for FakeChatBackend {
    fn tokenizer(&self) -> DynTokenizer {
        Arc::new(FakeChatTokenizer)
    }

    fn model_id(&self) -> &str {
        "test-model"
    }
}

impl ChatBackend for FakeChatBackend {
    fn chat_renderer(&self) -> DynChatRenderer {
        Arc::new(self.clone())
    }

    fn new_chat_output_processor(
        &self,
        request: &mut ChatRequest,
        options: NewChatOutputProcessorOptions<'_>,
    ) -> vllm_chat::Result<DynChatOutputProcessor> {
        Ok(Box::new(DefaultChatOutputProcessor::new(
            request,
            self.model_id(),
            options.tokenizer,
            options.tool_call_parser,
            options.reasoning_parser,
        )?))
    }
}

impl ChatRenderer for FakeChatBackend {
    fn render(&self, request: &ChatRequest) -> vllm_chat::Result<RenderedPrompt> {
        let mut prompt = String::new();
        for message in &request.messages {
            prompt.push_str(message.role().as_str());
            prompt.push_str(": ");
            prompt.push_str(&message.text_content()?);
            prompt.push('\n');
        }
        if request.chat_options.add_generation_prompt() {
            prompt.push_str("assistant:");
        }
        Ok(RenderedPrompt {
            prompt: Prompt::Text(prompt),
        })
    }
}

/// Spin up an HTTP server on a random port backed by a mock engine.
/// Returns the `async-openai` client, the HTTP server task, and the mock engine
/// task.
async fn http_test_server(
    engine_id: impl Into<EngineId>,
    output_specs: Vec<(Vec<u32>, Option<EngineCoreFinishReason>)>,
) -> (
    Client<OpenAIConfig>,
    tokio::task::JoinHandle<()>,
    MockEngineTask,
) {
    let ipc = IpcNamespace::new().expect("create ipc namespace");
    let handshake_address = ipc.handshake_endpoint();
    let engine_id = engine_id.into();

    let engine_task = MockEngineTask::new(spawn_mock_engine_task(
        handshake_address.clone(),
        engine_id.clone(),
        move |dealer, push| {
            boxed_test_future(async move {
                let add = recv_engine_message(dealer).await;
                let request: EngineCoreRequest =
                    rmp_serde::from_slice(&add[1]).expect("decode request");
                send_outputs(
                    push,
                    engine_outputs_for_request(&request.request_id, output_specs),
                )
                .await;
            })
        },
    ));

    let client = EngineCoreClient::connect_with_input_output_addresses(
        EngineCoreClientConfig::new_single(handshake_address).with_model_name("test-model"),
        Some(ipc.input_endpoint()),
        Some(ipc.output_endpoint()),
    )
    .await
    .expect("connect client");

    let chat = ChatLlm::from_shared_backend(
        test_llm(client),
        Arc::new(FakeChatBackend) as Arc<dyn ChatTextBackend>,
    );
    let state = Arc::new(AppState::new(vec!["test-model".to_string()], chat));
    let app = build_router(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind http listener");
    let addr = listener.local_addr().expect("local addr");

    let server_task = tokio::spawn(async move {
        axum::serve(listener, app).await.expect("http server");
    });

    let openai_client = Client::with_config(
        OpenAIConfig::new()
            .with_api_key("unused")
            .with_api_base(format!("http://{addr}/v1")),
    );

    (openai_client, server_task, engine_task)
}

// ========================================================================================
// Tests
// ========================================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn list_models_via_http_client() {
    let (client, server_task, _engine_task) =
        http_test_server(b"engine-http-models", default_stream_output_specs()).await;

    let models = client.models().list().await.expect("list models");
    let model_ids: Vec<&str> = models.data.iter().map(|m| m.id.as_str()).collect();
    assert_eq!(model_ids, vec!["test-model"]);

    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn non_streaming_chat_via_http_client() {
    let (client, server_task, engine_task) =
        http_test_server(b"engine-http-chat", default_stream_output_specs()).await;

    let request = CreateChatCompletionRequestArgs::default()
        .model("test-model")
        .stream(false)
        .max_completion_tokens(10u32)
        .messages([ChatCompletionRequestUserMessageArgs::default()
            .content("hello")
            .build()
            .expect("build user message")
            .into()])
        .build()
        .expect("build request");

    let response = client.chat().create(request).await.expect("chat completion");

    assert_eq!(response.model, "test-model");
    assert_eq!(response.choices.len(), 1);
    let choice = &response.choices[0];
    // The stop token `!` is suppressed from text.
    assert_eq!(choice.message.content.as_deref(), Some("hi"));
    assert_eq!(
        choice.finish_reason,
        Some(async_openai::types::chat::FinishReason::Stop)
    );

    engine_task.await.expect("mock engine task");
    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn streaming_chat_via_http_client() {
    let (client, server_task, engine_task) =
        http_test_server(b"engine-http-stream", default_stream_output_specs()).await;

    let request = CreateChatCompletionRequestArgs::default()
        .model("test-model")
        .stream(true)
        .max_completion_tokens(10u32)
        .messages([ChatCompletionRequestUserMessageArgs::default()
            .content("hello")
            .build()
            .expect("build user message")
            .into()])
        .build()
        .expect("build request");

    let mut stream = client.chat().create_stream(request).await.expect("streaming chat completion");

    let mut full_text = String::new();
    let mut saw_role = false;
    let mut saw_finish_reason = false;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.expect("stream chunk");
        for choice in &chunk.choices {
            if choice.delta.role.is_some() {
                saw_role = true;
            }
            if let Some(ref delta) = choice.delta.content {
                full_text.push_str(delta);
            }
            if choice.finish_reason.is_some() {
                saw_finish_reason = true;
            }
        }
    }

    assert!(saw_role, "expected an assistant role chunk");
    assert!(saw_finish_reason, "expected a terminal finish reason");
    assert_eq!(full_text, "hi");

    engine_task.await.expect("mock engine task");
    server_task.abort();
}
