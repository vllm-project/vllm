use std::future::Future;
use std::io;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use futures::StreamExt as _;
use hyper_util::rt::TokioIo;
use openssl::ssl::{SslConnector, SslFiletype, SslMethod};
use serial_test::serial;
use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
use tokio::net::TcpStream;
use tokio_openssl::SslStream;
use tonic::transport::{Channel, Endpoint, Server as TonicServer, Uri};
use tower::service_fn;
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
use super::{GenerateServer, GenerateServiceImpl, incoming, pb, tls_incoming};
use crate::listener::Listener;
use crate::state::AppState;
use crate::tls;
use crate::tls_tests::{TestCerts, server_tls};

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
struct FakeTextBackend;

#[derive(Debug)]
struct FakeTokenizer;

impl Tokenizer for FakeTokenizer {
    fn encode(
        &self,
        text: &str,
        _add_special_tokens: bool,
    ) -> vllm_text::tokenizer::Result<Vec<u32>> {
        Ok(text.bytes().map(u32::from).collect())
    }

    fn decode(
        &self,
        token_ids: &[u32],
        _skip_special_tokens: bool,
    ) -> vllm_text::tokenizer::Result<String> {
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
            self.tokenizer(),
            options.tool_call_parser,
            options.reasoning_parser,
        )?))
    }
}

impl ChatRenderer for FakeTextBackend {
    fn render(&self, _request: &ChatRequest) -> vllm_chat::Result<RenderedPrompt> {
        Ok(RenderedPrompt {
            prompt: Prompt::Text(String::new()),
            effective_template_kwargs: Default::default(),
        })
    }
}

/// Build the gRPC service + mock engine that serves a single request with the
/// given output specs. Shared by the plaintext and TLS server fixtures.
async fn setup_grpc_service(
    engine_id: impl Into<EngineId>,
    output_specs: Vec<(Vec<u32>, Option<EngineCoreFinishReason>)>,
) -> (GenerateServer<GenerateServiceImpl>, MockEngineTask) {
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

    let client = EngineCoreClient::connect(
        EngineCoreClientConfig::new_single(handshake_address)
            .with_model_name("test-model")
            .with_local_input_output_addresses(
                Some(ipc.input_endpoint()),
                Some(ipc.output_endpoint()),
            ),
    )
    .await
    .expect("connect client");

    let chat = ChatLlm::from_shared_backend(
        test_llm(client),
        Arc::new(FakeTextBackend) as Arc<dyn ChatTextBackend>,
    );
    let state = Arc::new(AppState::new(vec!["test-model".to_string()], chat));
    (
        GenerateServer::new(GenerateServiceImpl::new(state)),
        engine_task,
    )
}

/// Spin up a plaintext gRPC server backed by a mock engine. Returns the client,
/// the gRPC server task, and the mock engine task.
async fn grpc_test_server(
    engine_id: impl Into<EngineId>,
    output_specs: Vec<(Vec<u32>, Option<EngineCoreFinishReason>)>,
) -> (
    GenerateClient<tonic::transport::Channel>,
    tokio::task::JoinHandle<()>,
    MockEngineTask,
) {
    let (svc, engine_task) = setup_grpc_service(engine_id, output_specs).await;

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind grpc listener");
    let addr = listener.local_addr().expect("local addr");

    let server_task = tokio::spawn(async move {
        let incoming = incoming(Listener::Tcp(listener));
        TonicServer::builder()
            .add_service(svc)
            .serve_with_incoming(incoming)
            .await
            .expect("grpc server");
    });

    let grpc_client = GenerateClient::connect(format!("http://{addr}"))
        .await
        .expect("connect grpc client");

    (grpc_client, server_task, engine_task)
}

/// Spin up a TLS gRPC server (server cert from `certs`, `cert_reqs` mTLS mode).
/// Returns the address, the server task, and the mock engine task.
async fn grpc_tls_test_server(
    engine_id: impl Into<EngineId>,
    output_specs: Vec<(Vec<u32>, Option<EngineCoreFinishReason>)>,
    certs: &TestCerts,
    cert_reqs: i32,
) -> (String, tokio::task::JoinHandle<()>, MockEngineTask) {
    let (svc, engine_task) = setup_grpc_service(engine_id, output_specs).await;
    let context = tls::build_grpc_server_config(&server_tls(certs, cert_reqs))
        .expect("build grpc tls config");

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind grpc listener");
    let addr = listener.local_addr().expect("local addr").to_string();

    let server_task = tokio::spawn(async move {
        let incoming = tls_incoming(Listener::Tcp(listener), context, tls::TLS_HANDSHAKE_TIMEOUT);
        TonicServer::builder()
            .add_service(svc)
            .serve_with_incoming(incoming)
            .await
            .expect("grpc tls server");
    });

    (addr, server_task, engine_task)
}

/// Build a tonic `Generate` client over a tokio-openssl connector, optionally
/// with a client identity for mTLS. Hand-rolled because tonic 0.14 ships no
/// OpenSSL transport.
async fn grpc_tls_client(
    certs: &TestCerts,
    addr: &str,
    identity: Option<&str>,
) -> Result<GenerateClient<Channel>, tonic::transport::Error> {
    let ca = certs.path("ca.pem");
    let identity = identity.map(|name| {
        (
            certs.path(&format!("{name}.pem")),
            certs.path(&format!("{name}.key")),
        )
    });
    let target = addr.to_string();

    let connector = service_fn(move |_: Uri| {
        let ca = ca.clone();
        let identity = identity.clone();
        let target = target.clone();
        async move {
            let tcp = TcpStream::connect(&target).await?;
            let mut builder =
                SslConnector::builder(SslMethod::tls_client()).map_err(io::Error::other)?;
            builder.set_ca_file(&ca).map_err(io::Error::other)?;
            if let Some((cert, key)) = &identity {
                builder.set_certificate_chain_file(cert).map_err(io::Error::other)?;
                builder.set_private_key_file(key, SslFiletype::PEM).map_err(io::Error::other)?;
            }
            let mut config = builder.build().configure().map_err(io::Error::other)?;
            config.set_verify_hostname(false);
            config.set_alpn_protos(b"\x02h2").map_err(io::Error::other)?;
            let ssl = config.into_ssl("127.0.0.1").map_err(io::Error::other)?;
            let mut stream = SslStream::new(ssl, tcp).map_err(io::Error::other)?;
            Pin::new(&mut stream).connect().await.map_err(io::Error::other)?;
            Ok::<_, io::Error>(TokioIo::new(stream))
        }
    });

    let channel = Endpoint::from_shared(format!("https://{addr}"))
        .expect("grpc endpoint")
        .connect_with_connector(connector)
        .await?;
    Ok(GenerateClient::new(channel))
}

/// Complete a raw TLS handshake against the gRPC port (offering ALPN `h2`) for
/// the ALPN-negotiation assertion.
async fn grpc_tls_handshake(
    certs: &TestCerts,
    addr: &str,
) -> io::Result<Pin<Box<SslStream<TcpStream>>>> {
    let tcp = TcpStream::connect(addr).await?;
    let mut builder = SslConnector::builder(SslMethod::tls_client()).map_err(io::Error::other)?;
    builder.set_ca_file(certs.path("ca.pem")).map_err(io::Error::other)?;
    let mut config = builder.build().configure().map_err(io::Error::other)?;
    config.set_verify_hostname(false);
    config.set_alpn_protos(b"\x02h2").map_err(io::Error::other)?;
    let ssl = config.into_ssl("127.0.0.1").map_err(io::Error::other)?;
    let mut stream = Box::pin(SslStream::new(ssl, tcp).map_err(io::Error::other)?);
    stream.as_mut().connect().await.map_err(io::Error::other)?;
    Ok(stream)
}

/// Spin up a plaintext gRPC server, optionally with HTTP/2 keepalive set to
/// `keepalive` for both the PING interval and the unanswered-PING timeout.
async fn grpc_server_with_keepalive(
    engine_id: impl Into<EngineId>,
    keepalive: Option<Duration>,
) -> (String, tokio::task::JoinHandle<()>, MockEngineTask) {
    let (svc, engine_task) = setup_grpc_service(engine_id, default_stream_output_specs()).await;

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind grpc listener");
    let addr = listener.local_addr().expect("local addr").to_string();

    let mut builder = TonicServer::builder();
    if let Some(interval) = keepalive {
        builder = builder
            .http2_keepalive_interval(Some(interval))
            .http2_keepalive_timeout(Some(interval));
    }

    let server_task = tokio::spawn(async move {
        let incoming = incoming(Listener::Tcp(listener));
        builder
            .add_service(svc)
            .serve_with_incoming(incoming)
            .await
            .expect("grpc server");
    });

    (addr, server_task, engine_task)
}

/// Establish an HTTP/2 connection (preface + SETTINGS exchange) then go silent,
/// ACKing the server's SETTINGS but never its keepalive PINGs. Returns whether
/// the SERVER closes the connection within `wait`. A minimal hand-rolled h2 peer
/// because a real client auto-ACKs PINGs and so can never be kept-alive-evicted.
async fn h2_unresponsive_peer_closed_within(addr: &str, wait: Duration) -> bool {
    let mut tcp = TcpStream::connect(addr).await.expect("connect");
    tcp.write_all(b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n").await.expect("preface");
    tcp.write_all(&[0, 0, 0, 0x4, 0, 0, 0, 0, 0]).await.expect("client settings");

    let closed = tokio::time::timeout(wait, async {
        let mut header = [0u8; 9];
        while tcp.read_exact(&mut header).await.is_ok() {
            let len = u32::from_be_bytes([0, header[0], header[1], header[2]]) as usize;
            let frame_type = header[3];
            let flags = header[4];
            let mut payload = vec![0u8; len];
            if tcp.read_exact(&mut payload).await.is_err() {
                return;
            }
            // ACK the server's SETTINGS so the only thing left unanswered is PINGs.
            if frame_type == 0x4 && flags & 0x1 == 0 {
                let _ = tcp.write_all(&[0, 0, 0, 0x4, 0x1, 0, 0, 0, 0]).await;
            }
        }
    })
    .await;

    closed.is_ok()
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
async fn unary_generate_min_tokens_above_max_tokens_returns_invalid_argument() {
    let (mut client, server_task, _engine_task) =
        grpc_test_server(b"engine-grpc-min-above-max", default_stream_output_specs()).await;

    let status = client
        .generate(pb::GenerateRequest {
            request_id: "test-min-above-max".to_string(),
            model: "test-model".to_string(),
            prompt: Some(pb::generate_request::Prompt::Text("hi".to_string())),
            stopping: Some(pb::StoppingCriteria {
                max_new_tokens: 4,
                min_new_tokens: 5,
                ..Default::default()
            }),
            ..Default::default()
        })
        .await
        .expect_err("should fail when min_new_tokens exceeds max_new_tokens");

    assert_eq!(status.code(), tonic::Code::InvalidArgument);
    assert!(status.message().contains("min_tokens=5"));
    assert!(status.message().contains("max_tokens=4"));

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
    let prompt_info = first.prompt_info.as_ref().expect("first response has prompt_info");
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
    let finish = last_output.finish_info.as_ref().expect("finish_info on last output");
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
async fn streaming_generate_min_tokens_above_max_tokens_returns_invalid_argument() {
    let (mut client, server_task, _engine_task) = grpc_test_server(
        b"engine-grpc-stream-min-above-max",
        default_stream_output_specs(),
    )
    .await;

    let status = client
        .generate_stream(pb::GenerateRequest {
            request_id: "test-stream-min-above-max".to_string(),
            model: "test-model".to_string(),
            prompt: Some(pb::generate_request::Prompt::Text("hi".to_string())),
            stopping: Some(pb::StoppingCriteria {
                max_new_tokens: 4,
                min_new_tokens: 5,
                ..Default::default()
            }),
            ..Default::default()
        })
        .await
        .expect_err("should fail when min_new_tokens exceeds max_new_tokens");

    assert_eq!(status.code(), tonic::Code::InvalidArgument);
    assert!(status.message().contains("min_tokens=5"));
    assert!(status.message().contains("max_tokens=4"));

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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn grpc_generate_succeeds_over_tls() {
    let certs = TestCerts::generate();
    let (addr, server_task, engine_task) = grpc_tls_test_server(
        b"engine-grpc-tls-unary",
        default_stream_output_specs(),
        &certs,
        0,
    )
    .await;

    let mut client = grpc_tls_client(&certs, &addr, None).await.expect("tls client");
    let response = client
        .generate(pb::GenerateRequest {
            request_id: "test-tls-unary".to_string(),
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
        .expect("unary generate over tls")
        .into_inner();

    assert_eq!(response.outputs.expect("outputs present").text, "hi");

    engine_task.await.expect("mock engine task");
    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn grpc_tls_negotiates_h2_alpn() {
    let certs = TestCerts::generate();
    let (addr, server_task, _engine_task) = grpc_tls_test_server(
        b"engine-grpc-tls-alpn",
        default_stream_output_specs(),
        &certs,
        0,
    )
    .await;

    let stream = grpc_tls_handshake(&certs, &addr).await.expect("handshake");
    assert_eq!(
        stream.ssl().selected_alpn_protocol(),
        Some(&b"h2"[..]),
        "server must negotiate h2 ALPN"
    );

    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn grpc_mtls_required_rejects_client_without_certificate() {
    let certs = TestCerts::generate();
    let (addr, server_task, _engine_task) = grpc_tls_test_server(
        b"engine-grpc-tls-mtls-reject",
        default_stream_output_specs(),
        &certs,
        2,
    )
    .await;

    // With TLS 1.3 the missing-client-cert rejection surfaces on first use, not
    // at the handshake, so drive an RPC and assert the call fails.
    let outcome = match grpc_tls_client(&certs, &addr, None).await {
        Err(_) => Err(()),
        Ok(mut client) => client
            .generate(pb::GenerateRequest {
                request_id: "test-tls-mtls-reject".to_string(),
                model: "test-model".to_string(),
                prompt: Some(pb::generate_request::Prompt::Text("hello".to_string())),
                stopping: Some(pb::StoppingCriteria {
                    max_new_tokens: 10,
                    ..Default::default()
                }),
                ..Default::default()
            })
            .await
            .map(|_| ())
            .map_err(|_| ()),
    };
    assert!(
        outcome.is_err(),
        "mTLS-required gRPC must reject a client without a certificate"
    );

    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn grpc_mtls_required_accepts_valid_client_certificate() {
    let certs = TestCerts::generate();
    let (addr, server_task, engine_task) = grpc_tls_test_server(
        b"engine-grpc-tls-mtls-accept",
        default_stream_output_specs(),
        &certs,
        2,
    )
    .await;

    let mut client = grpc_tls_client(&certs, &addr, Some("client")).await.expect("mtls client");
    let response = client
        .generate(pb::GenerateRequest {
            request_id: "test-tls-mtls".to_string(),
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
        .expect("mtls generate over tls")
        .into_inner();

    assert_eq!(response.outputs.expect("outputs present").text, "hi");

    engine_task.await.expect("mock engine task");
    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn grpc_keepalive_closes_unresponsive_connection() {
    let (addr, server_task, _engine_task) =
        grpc_server_with_keepalive(b"engine-grpc-keepalive", Some(Duration::from_millis(150)))
            .await;

    let closed = h2_unresponsive_peer_closed_within(&addr, Duration::from_secs(5)).await;
    assert!(
        closed,
        "keepalive must close a peer that stops answering PINGs"
    );

    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn grpc_without_keepalive_keeps_unresponsive_connection_open() {
    // Without keepalive the same unresponsive peer is NOT
    // closed, proving the close above is attributable to keepalive.
    let (addr, server_task, _engine_task) =
        grpc_server_with_keepalive(b"engine-grpc-no-keepalive", None).await;

    let closed = h2_unresponsive_peer_closed_within(&addr, Duration::from_secs(1)).await;
    assert!(
        !closed,
        "without keepalive an idle h2 connection must stay open"
    );

    server_task.abort();
}
