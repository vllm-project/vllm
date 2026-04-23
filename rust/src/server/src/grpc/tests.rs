use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use futures::StreamExt as _;
use serial_test::serial;
use tonic::transport::Server as TonicServer;
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

use super::pb::generate_client::GenerateClient;
use super::{GenerateServer, GenerateServiceImpl, pb};
use crate::state::AppState;

// ========================================================================================
// Helpers (mirrors the patterns in routes/tests.rs)
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
        num_cached_tokens: 0,
        num_external_computed_tokens: 0,
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
struct FakeTextBackend;

#[derive(Debug)]
struct FakeTokenizer;

impl Tokenizer for FakeTokenizer {
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

impl TextBackend for FakeTextBackend {
    fn tokenizer(&self) -> DynTokenizer {
        Arc::new(FakeTokenizer)
    }

    fn model_id(&self) -> &str {
        "test-model"
    }
}

impl ChatBackend for FakeTextBackend {
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

impl ChatRenderer for FakeTextBackend {
    fn render(&self, _request: &ChatRequest) -> vllm_chat::Result<RenderedPrompt> {
        Ok(RenderedPrompt {
            prompt: Prompt::Text(String::new()),
        })
    }
}

/// Spin up a gRPC server backed by a mock engine that serves a single request with the
/// given output specs. Returns the client, the gRPC server task, and the mock engine task.
async fn grpc_test_server(
    engine_id: impl Into<EngineId>,
    output_specs: Vec<(Vec<u32>, Option<EngineCoreFinishReason>)>,
) -> (
    GenerateClient<tonic::transport::Channel>,
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
        Arc::new(FakeTextBackend) as Arc<dyn ChatTextBackend>,
    );
    let state = Arc::new(AppState::new("test-model", chat));
    let svc = GenerateServer::new(GenerateServiceImpl::new(state));

    // Bind to an OS-assigned port.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind grpc listener");
    let addr = listener.local_addr().expect("local addr");

    let server_task = tokio::spawn(async move {
        let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);
        TonicServer::builder()
            .add_service(svc)
            .serve_with_incoming(incoming)
            .await
            .expect("grpc server");
    });

    // Connect the client.
    let grpc_client = GenerateClient::connect(format!("http://{addr}"))
        .await
        .expect("connect grpc client");

    (grpc_client, server_task, engine_task)
}

// ========================================================================================
// Tests
// ========================================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn unary_generate_returns_collected_text() {
    let (mut client, server_task, engine_task) =
        grpc_test_server(b"engine-grpc-unary", default_stream_output_specs()).await;

    let response = client
        .generate(pb::GenerateRequest {
            request_id: "test-unary-1".to_string(),
            model: "test-model".to_string(),
            prompt: Some(pb::generate_request::Prompt::Text("hello".to_string())),
            stopping: Some(pb::StoppingCriteria {
                max_new_tokens: 10,
                ..Default::default()
            }),
            response: Some(pb::ResponseOptions {
                output_text: Some(true),
                ..Default::default()
            }),
            ..Default::default()
        })
        .await
        .expect("unary generate")
        .into_inner();

    // Unary collects all tokens into one response.
    let outputs = response.outputs.expect("outputs present");
    assert_eq!(outputs.text, "hi");

    let finish = outputs.finish_info.expect("finish_info present");
    assert_eq!(
        finish.finish_reason,
        pb::finish_info::FinishReason::Stop as i32
    );
    assert_eq!(finish.num_output_tokens, 3);

    let prompt = response.prompt_info.expect("prompt_info present");
    assert_eq!(prompt.num_prompt_tokens, 5); // "hello" = 5 bytes

    engine_task.await.expect("mock engine task");
    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn unary_generate_with_token_ids_prompt() {
    let (mut client, server_task, engine_task) =
        grpc_test_server(b"engine-grpc-token-ids", default_stream_output_specs()).await;

    let response = client
        .generate(pb::GenerateRequest {
            request_id: "test-token-ids".to_string(),
            model: "test-model".to_string(),
            prompt: Some(pb::generate_request::Prompt::TokenIds(pb::TokenIds {
                ids: vec![1, 2, 3],
            })),
            stopping: Some(pb::StoppingCriteria {
                max_new_tokens: 10,
                ..Default::default()
            }),
            ..Default::default()
        })
        .await
        .expect("unary generate with token ids")
        .into_inner();

    let outputs = response.outputs.expect("outputs present");
    assert_eq!(outputs.text, "hi");
    assert_eq!(
        response.prompt_info.expect("prompt_info").num_prompt_tokens,
        3
    );

    engine_task.await.expect("mock engine task");
    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn unary_generate_returns_token_ids_when_requested() {
    let (mut client, server_task, engine_task) =
        grpc_test_server(b"engine-grpc-tok-resp", default_stream_output_specs()).await;

    let response = client
        .generate(pb::GenerateRequest {
            request_id: "test-tok-resp".to_string(),
            model: "test-model".to_string(),
            prompt: Some(pb::generate_request::Prompt::Text("hi".to_string())),
            stopping: Some(pb::StoppingCriteria {
                max_new_tokens: 10,
                ..Default::default()
            }),
            response: Some(pb::ResponseOptions {
                output_text: Some(true),
                output_token_ids: true,
                prompt_token_ids: true,
                ..Default::default()
            }),
            ..Default::default()
        })
        .await
        .expect("unary generate")
        .into_inner();

    let outputs = response.outputs.expect("outputs present");
    assert_eq!(
        outputs.token_ids,
        vec![b'h' as u32, b'i' as u32, b'!' as u32]
    );

    let prompt = response.prompt_info.expect("prompt_info present");
    assert_eq!(prompt.token_ids, vec![b'h' as u32, b'i' as u32]);

    engine_task.await.expect("mock engine task");
    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn unary_generate_missing_prompt_returns_invalid_argument() {
    let (mut client, server_task, _engine_task) =
        grpc_test_server(b"engine-grpc-no-prompt", default_stream_output_specs()).await;

    let status = client
        .generate(pb::GenerateRequest {
            request_id: "test-no-prompt".to_string(),
            model: "test-model".to_string(),
            prompt: None,
            ..Default::default()
        })
        .await
        .expect_err("should fail without prompt");

    assert_eq!(status.code(), tonic::Code::InvalidArgument);
    assert!(status.message().contains("prompt"));

    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn streaming_generate_yields_incremental_responses() {
    let (mut client, server_task, engine_task) =
        grpc_test_server(b"engine-grpc-stream", default_stream_output_specs()).await;

    let stream = client
        .generate_stream(pb::GenerateRequest {
            request_id: "test-stream-1".to_string(),
            model: "test-model".to_string(),
            prompt: Some(pb::generate_request::Prompt::Text("hello".to_string())),
            stopping: Some(pb::StoppingCriteria {
                max_new_tokens: 10,
                ..Default::default()
            }),
            response: Some(pb::ResponseOptions {
                output_text: Some(true),
                ..Default::default()
            }),
            ..Default::default()
        })
        .await
        .expect("streaming generate")
        .into_inner();

    let responses: Vec<pb::GenerateResponse> =
        stream.map(|r| r.expect("stream item")).collect().await;

    // First response carries prompt info, subsequent ones carry output deltas.
    assert!(
        responses.len() >= 2,
        "expected at least 2 streamed responses, got {}",
        responses.len()
    );

    // First message should have prompt info.
    let first = &responses[0];
    let prompt_info = first
        .prompt_info
        .as_ref()
        .expect("first response has prompt_info");
    assert_eq!(prompt_info.num_prompt_tokens, 5); // "hello"

    // Collect all text deltas.
    let full_text: String = responses
        .iter()
        .filter_map(|r| r.outputs.as_ref())
        .map(|o| o.text.as_str())
        .collect();
    assert_eq!(full_text, "hi");

    // Last output response should have finish info.
    let last_output = responses
        .iter()
        .rev()
        .find_map(|r| r.outputs.as_ref())
        .expect("at least one output");
    let finish = last_output
        .finish_info
        .as_ref()
        .expect("finish_info on last output");
    assert_eq!(
        finish.finish_reason,
        pb::finish_info::FinishReason::Stop as i32
    );

    engine_task.await.expect("mock engine task");
    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn streaming_generate_missing_prompt_returns_invalid_argument() {
    let (mut client, server_task, _engine_task) = grpc_test_server(
        b"engine-grpc-stream-no-prompt",
        default_stream_output_specs(),
    )
    .await;

    let status = client
        .generate_stream(pb::GenerateRequest {
            request_id: "test-stream-no-prompt".to_string(),
            model: "test-model".to_string(),
            prompt: None,
            ..Default::default()
        })
        .await
        .expect_err("should fail without prompt");

    assert_eq!(status.code(), tonic::Code::InvalidArgument);

    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn unary_generate_with_sampling_params() {
    let (mut client, server_task, engine_task) =
        grpc_test_server(b"engine-grpc-sampling", default_stream_output_specs()).await;

    let response = client
        .generate(pb::GenerateRequest {
            request_id: "test-sampling".to_string(),
            model: "test-model".to_string(),
            prompt: Some(pb::generate_request::Prompt::Text("test".to_string())),
            temperature: Some(0.7),
            sampling: Some(pb::RandomSampling {
                top_k: 50,
                top_p: 0.9,
                seed: Some(42),
                ..Default::default()
            }),
            stopping: Some(pb::StoppingCriteria {
                max_new_tokens: 5,
                ..Default::default()
            }),
            ..Default::default()
        })
        .await
        .expect("generate with sampling params")
        .into_inner();

    // Verify the request was accepted and produced output.
    let outputs = response.outputs.expect("outputs present");
    assert_eq!(outputs.text, "hi");

    engine_task.await.expect("mock engine task");
    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn unary_generate_rejects_wrong_model() {
    let (mut client, server_task, _engine_task) =
        grpc_test_server(b"engine-grpc-wrong-model", default_stream_output_specs()).await;

    let status = client
        .generate(pb::GenerateRequest {
            request_id: "test-wrong-model".to_string(),
            model: "other-model".to_string(),
            prompt: Some(pb::generate_request::Prompt::Text("hi".to_string())),
            stopping: Some(pb::StoppingCriteria {
                max_new_tokens: 10,
                ..Default::default()
            }),
            ..Default::default()
        })
        .await
        .expect_err("should fail with wrong model");

    assert_eq!(status.code(), tonic::Code::NotFound);
    assert!(status.message().contains("other-model"));

    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn streaming_generate_rejects_wrong_model() {
    let (mut client, server_task, _engine_task) = grpc_test_server(
        b"engine-grpc-stream-wrong-model",
        default_stream_output_specs(),
    )
    .await;

    let status = client
        .generate_stream(pb::GenerateRequest {
            request_id: "test-stream-wrong-model".to_string(),
            model: "other-model".to_string(),
            prompt: Some(pb::generate_request::Prompt::Text("hi".to_string())),
            stopping: Some(pb::StoppingCriteria {
                max_new_tokens: 10,
                ..Default::default()
            }),
            ..Default::default()
        })
        .await
        .expect_err("should fail with wrong model");

    assert_eq!(status.code(), tonic::Code::NotFound);
    assert!(status.message().contains("other-model"));

    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn unary_generate_accepts_empty_model() {
    let (mut client, server_task, engine_task) =
        grpc_test_server(b"engine-grpc-empty-model", default_stream_output_specs()).await;

    // Empty `model` (proto3 default) is treated as "unset" and should be accepted.
    let response = client
        .generate(pb::GenerateRequest {
            request_id: "test-empty-model".to_string(),
            model: String::new(),
            prompt: Some(pb::generate_request::Prompt::Text("hi".to_string())),
            stopping: Some(pb::StoppingCriteria {
                max_new_tokens: 10,
                ..Default::default()
            }),
            ..Default::default()
        })
        .await
        .expect("unary generate with empty model")
        .into_inner();

    let outputs = response.outputs.expect("outputs present");
    assert_eq!(outputs.text, "hi");

    engine_task.await.expect("mock engine task");
    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn unary_generate_output_text_defaults_to_true() {
    let (mut client, server_task, engine_task) =
        grpc_test_server(b"engine-grpc-default-text", default_stream_output_specs()).await;

    // No response options at all — output_text should default to true.
    let response = client
        .generate(pb::GenerateRequest {
            request_id: "test-default-text".to_string(),
            model: "test-model".to_string(),
            prompt: Some(pb::generate_request::Prompt::Text("x".to_string())),
            stopping: Some(pb::StoppingCriteria {
                max_new_tokens: 10,
                ..Default::default()
            }),
            ..Default::default()
        })
        .await
        .expect("unary generate")
        .into_inner();

    let outputs = response.outputs.expect("outputs present");
    assert_eq!(outputs.text, "hi");

    engine_task.await.expect("mock engine task");
    server_task.abort();
}
