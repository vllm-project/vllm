// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::fs;
use std::future::Future;
use std::io;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use futures::StreamExt as _;
use hyper_util::rt::TokioIo;
use openssl::ssl::{SslConnector, SslFiletype, SslMethod};
use rmpv::Value;
use serial_test::serial;
use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
use tokio::net::TcpStream;
use tokio_openssl::SslStream;
use tonic::transport::server::Router;
use tonic::transport::{Channel, Endpoint, Server as TonicServer, Uri};
use tonic_health::pb::HealthCheckRequest;
use tonic_health::pb::health_check_response::ServingStatus as HealthServingStatus;
use tonic_health::pb::health_client::HealthClient;
use tonic_health::server::health_reporter;
use tower::service_fn;
use vllm_chat::{
    ChatBackend, ChatLlm, ChatRenderer, ChatRequest, ChatTextBackend, DefaultChatOutputProcessor,
    DynChatOutputProcessor, DynChatRenderer, NewChatOutputProcessorOptions, RenderedPrompt,
};
use vllm_engine_core_client::mock_engine::{
    DEFAULT_MOCK_BLOCK_SIZE, DEFAULT_MOCK_MAX_MODEL_LEN, DEFAULT_MOCK_NUM_GPU_BLOCKS,
    default_ready_response,
};
use vllm_engine_core_client::protocol::decode_value;
use vllm_engine_core_client::protocol::handshake::{
    EngineCoreReadyResponse, KvEventsConfig, KvTransferInfo,
};
use vllm_engine_core_client::protocol::kv_transfer::KvConnectorHandshakeEntry;
use vllm_engine_core_client::protocol::output::{
    EngineCoreFinishReason, EngineCoreOutput, EngineCoreOutputs, RequestBatchOutputs,
    UtilityCallOutput,
};
use vllm_engine_core_client::protocol::request::EngineCoreRequest;
use vllm_engine_core_client::protocol::sampling::EngineCoreSamplingParams;
use vllm_engine_core_client::protocol::utility::{UtilityOutput, UtilityResultEnvelope};
use vllm_engine_core_client::test_utils::{IpcNamespace, spawn_mock_engine_task_with_ready};
use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig, EngineId, TransportMode};
use vllm_llm::{GenerateRequest, Llm};
use vllm_text::tokenizer::DynTokenizer;
use vllm_text::{Prompt, TextBackend};
use vllm_tokenizer::test_utils::TestTokenizer;
use zeromq::prelude::{SocketRecv, SocketSend};
use zeromq::{DealerSocket, PushSocket, ZmqMessage};

use super::convert::json_to_proto_struct;
use super::kv_transfer::{kv_event_source, kv_transfer_engine};
use super::pb::control_client::ControlClient;
use super::pb::inference_client::InferenceClient;
use super::pb::kv_transfer_client::KvTransferClient;
use super::pb::rl_control_client::RlControlClient;
use super::{
    ControlGrpcService, ControlServiceImpl, InferenceGrpcService, InferenceServiceImpl,
    KvTransferGrpcService, KvTransferServiceImpl, RlControlGrpcService, RlControlServiceImpl, pb,
};
use crate::grpc_services::{GrpcServiceSelection, GrpcServices};
use crate::kv_peer::KvPeerHandshaker;
use crate::listener::{Listener, MaybeTlsListener};
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
        finish_reason,
        ..Default::default()
    }
}

fn engine_outputs_for_request(
    request_id: &str,
    output_specs: Vec<(Vec<u32>, Option<EngineCoreFinishReason>)>,
) -> EngineCoreOutputs {
    RequestBatchOutputs {
        outputs: output_specs
            .into_iter()
            .map(|(token_ids, finish_reason)| request_output(request_id, token_ids, finish_reason))
            .collect(),
        ..Default::default()
    }
    .into()
}

fn default_stream_output_specs() -> Vec<(Vec<u32>, Option<EngineCoreFinishReason>)> {
    vec![
        (vec![b'h' as u32], None),
        (vec![b'i' as u32], None),
        (vec![b'!' as u32], Some(EngineCoreFinishReason::Stop)),
    ]
}

fn ec_proto_struct(mm_hashes: &[&str]) -> prost_types::Struct {
    let ec_items: Vec<_> = mm_hashes
        .iter()
        .map(|mm_hash| {
            serde_json::json!({
                "image_grid_thw": [[1, 16, 16]],
                "mm_hash": mm_hash,
            })
        })
        .collect();
    json_to_proto_struct(&serde_json::json!({ "ec_items": ec_items }))
        .expect("valid EC proto struct")
}

fn decode_kv_proto_struct() -> prost_types::Struct {
    json_to_proto_struct(&serde_json::json!({
        "do_remote_prefill": true,
        "pp_size": 1,
        "remote_block_ids": [[7]],
    }))
    .expect("valid KV proto struct")
}

async fn send_outputs(push: &mut PushSocket, outputs: EngineCoreOutputs) {
    push.send(ZmqMessage::from(
        rmp_serde::to_vec_named(&outputs).expect("encode outputs"),
    ))
    .await
    .expect("send outputs");
}

async fn reply_utility_bool(
    dealer: &mut DealerSocket,
    push: &mut PushSocket,
    expected_method: &str,
    result: bool,
) {
    reply_utility_value(dealer, push, expected_method, Value::Boolean(result)).await;
}

async fn reply_utility_value(
    dealer: &mut DealerSocket,
    push: &mut PushSocket,
    expected_method: &str,
    result: Value,
) {
    let frames = recv_engine_message(dealer).await;
    assert_eq!(frames[0].as_ref(), &[0x03]);
    let payload = decode_value(&frames[1]).expect("decode utility payload");
    let fields = payload.as_array().expect("utility payload array");
    let call_id = fields[1].as_u64().expect("utility call id");
    assert_eq!(fields[2].as_str(), Some(expected_method));
    send_outputs(
        push,
        UtilityCallOutput {
            output: UtilityOutput {
                call_id: call_id.into(),
                failure_message: None,
                result: Some(UtilityResultEnvelope::without_type_info(result)),
            },
            ..Default::default()
        }
        .into(),
    )
    .await;
}

fn test_kv_transfer_info() -> KvTransferInfo {
    KvTransferInfo {
        engine_id: "prefill-0_dp0".to_string(),
        kv_connector: "NixlConnector".to_string(),
        kv_role: "kv_producer".to_string(),
    }
}

fn test_handshake_entries() -> Vec<KvConnectorHandshakeEntry> {
    vec![
        KvConnectorHandshakeEntry {
            pp_rank: 0,
            tp_rank: 0,
            payload: bytes::Bytes::from_static(b"agent-0"),
            compatibility_hash: Some("abc123".to_string()),
        },
        KvConnectorHandshakeEntry {
            pp_rank: 0,
            tp_rank: 1,
            payload: bytes::Bytes::from_static(b"agent-1"),
            compatibility_hash: Some("abc123".to_string()),
        },
    ]
}

async fn recv_engine_message(dealer: &mut DealerSocket) -> Vec<bytes::Bytes> {
    dealer.recv().await.expect("recv engine message").into_vec()
}

#[derive(Clone, Debug)]
struct FakeTextBackend;

impl TextBackend for FakeTextBackend {
    fn tokenizer(&self) -> DynTokenizer {
        Arc::new(TestTokenizer::new())
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

struct FakeMultimodalBackend {
    model_info: vllm_chat::multimodal::MultimodalModelInfo,
}

impl TextBackend for FakeMultimodalBackend {
    fn tokenizer(&self) -> DynTokenizer {
        Arc::new(TestTokenizer::new().with_regular_token("<|image_pad|>", QWEN_IMAGE_TOKEN_ID))
    }

    fn model_id(&self) -> &str {
        "test-model"
    }
}

impl ChatBackend for FakeMultimodalBackend {
    fn chat_renderer(&self) -> DynChatRenderer {
        Arc::new(FakeTextBackend)
    }

    fn multimodal_model_info(&self) -> Option<&vllm_chat::multimodal::MultimodalModelInfo> {
        Some(&self.model_info)
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

const QWEN_IMAGE_TOKEN_ID: u32 = 151655;
const TINY_PNG_DATA_URI: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";

fn multimodal_backend() -> Arc<dyn ChatTextBackend> {
    let config_path = std::env::temp_dir().join(format!(
        "vllm-grpc-qwen-config-{}.json",
        uuid::Uuid::new_v4()
    ));
    fs::write(
        &config_path,
        r#"{"model_type":"qwen2_vl","image_token_id":151655}"#,
    )
    .expect("write qwen test config");
    let model_info = vllm_chat::multimodal::MultimodalModelInfo::from_paths(
        "qwen2-vl-test".to_string(),
        Some("qwen2_vl".to_string()),
        vllm_chat::multimodal::MultimodalConfigFiles {
            config: Some(&config_path),
            ..Default::default()
        },
        Arc::new(TestTokenizer::new().with_regular_token("<|image_pad|>", QWEN_IMAGE_TOKEN_ID)),
        Default::default(),
    )
    .expect("load multimodal info")
    .expect("qwen multimodal info is registered");
    let _ = fs::remove_file(config_path);
    Arc::new(FakeMultimodalBackend { model_info })
}

/// The gRPC services one frontend mounts, kept together so fixtures can hand
/// them all to a test server.
struct TestServices {
    state: AppState,
    mounted: GrpcServices,
}

impl TestServices {
    fn new(state: AppState) -> Self {
        Self {
            state,
            mounted: GrpcServices::all(),
        }
    }

    fn mounting(mut self, mounted: GrpcServices) -> Self {
        self.mounted = mounted;
        self
    }

    fn install(self, builder: &mut TonicServer) -> Router {
        let Self { state, mounted } = self;
        let state = Arc::new(state.with_grpc_services(mounted));
        let mounted = state.grpc_services();
        let kv_transfer_impl = Arc::new(KvTransferServiceImpl::new(state.clone()));
        let rl_control_impl = Arc::new(RlControlServiceImpl::new(state.clone()));
        let control = mounted.contains(GrpcServices::CONTROL).then(|| {
            ControlGrpcService::new(ControlServiceImpl::new(
                state.clone(),
                kv_transfer_impl.clone(),
                rl_control_impl.clone(),
            ))
        });
        let kv_transfer = mounted
            .contains(GrpcServices::KV_TRANSFER)
            .then(|| KvTransferGrpcService::from_arc(kv_transfer_impl));
        let rl_control = mounted
            .contains(GrpcServices::RL_CONTROL)
            .then(|| RlControlGrpcService::from_arc(rl_control_impl));
        let inference = mounted.contains(GrpcServices::INFERENCE).then(|| {
            InferenceGrpcService::new(InferenceServiceImpl::new(state))
                .max_decoding_message_size(crate::DEFAULT_REQUEST_BODY_LIMIT_BYTES)
        });
        builder
            .add_optional_service(control)
            .add_optional_service(kv_transfer)
            .add_optional_service(rl_control)
            .add_optional_service(inference)
    }
}

/// Build the gRPC service + mock engine that serves a single request with the
/// given output specs. Shared by the plaintext and TLS server fixtures.
async fn setup_grpc_service(
    engine_id: impl Into<EngineId>,
    output_specs: Vec<(Vec<u32>, Option<EngineCoreFinishReason>)>,
) -> (
    TestServices,
    tokio::sync::watch::Receiver<bool>,
    MockEngineTask,
) {
    setup_grpc_service_with_backend(engine_id, output_specs, Arc::new(FakeTextBackend), |_| {})
        .await
}

async fn setup_grpc_service_with_backend<F>(
    engine_id: impl Into<EngineId>,
    output_specs: Vec<(Vec<u32>, Option<EngineCoreFinishReason>)>,
    backend: Arc<dyn ChatTextBackend>,
    check_request: F,
) -> (
    TestServices,
    tokio::sync::watch::Receiver<bool>,
    MockEngineTask,
)
where
    F: FnOnce(&EngineCoreRequest) + Send + 'static,
{
    setup_grpc_service_with_engine_script(
        engine_id,
        default_ready_response(),
        backend,
        move |dealer, push| {
            boxed_test_future(async move {
                let add = recv_engine_message(dealer).await;
                let request: EngineCoreRequest =
                    rmp_serde::from_slice(&add[1]).expect("decode request");
                check_request(&request);
                send_outputs(
                    push,
                    engine_outputs_for_request(&request.request_id, output_specs),
                )
                .await;
            })
        },
    )
    .await
}

async fn setup_grpc_service_with_engine_script<F>(
    engine_id: impl Into<EngineId>,
    ready: EngineCoreReadyResponse,
    backend: Arc<dyn ChatTextBackend>,
    script: F,
) -> (
    TestServices,
    tokio::sync::watch::Receiver<bool>,
    MockEngineTask,
)
where
    F: for<'a> FnOnce(&'a mut DealerSocket, &'a mut PushSocket) -> TestFuture<'a> + Send + 'static,
{
    let ipc = IpcNamespace::new().expect("create ipc namespace");
    let handshake_address = ipc.handshake_endpoint();
    let engine_id = engine_id.into();

    let engine_task = MockEngineTask::new(spawn_mock_engine_task_with_ready(
        handshake_address.clone(),
        engine_id.clone(),
        ready,
        script,
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
    let engine_health = client.subscribe_health();

    let chat = ChatLlm::from_shared_backend(Llm::new(client), backend);
    let state = AppState::new(vec!["test-model".to_string()], chat);
    (TestServices::new(state), engine_health, engine_task)
}

/// Spin up a plaintext gRPC server backed by a mock engine. Returns the client,
/// the gRPC server task, and the mock engine task.
async fn grpc_test_server(
    engine_id: impl Into<EngineId>,
    output_specs: Vec<(Vec<u32>, Option<EngineCoreFinishReason>)>,
) -> (
    InferenceClient<tonic::transport::Channel>,
    tokio::task::JoinHandle<()>,
    MockEngineTask,
) {
    let (services, engine_health, engine_task) = setup_grpc_service(engine_id, output_specs).await;
    let (channel, server_task) = start_grpc_test_server(
        services,
        engine_health,
        tokio_util::sync::CancellationToken::new(),
    )
    .await;
    (InferenceClient::new(channel), server_task, engine_task)
}

async fn start_grpc_test_server(
    services: TestServices,
    engine_health: tokio::sync::watch::Receiver<bool>,
    shutdown: tokio_util::sync::CancellationToken,
) -> (Channel, tokio::task::JoinHandle<()>) {
    let (channel, _addr, server_task) =
        start_grpc_test_server_with_addr(services, engine_health, shutdown).await;
    (channel, server_task)
}

async fn start_grpc_test_server_with_addr(
    services: TestServices,
    engine_health: tokio::sync::watch::Receiver<bool>,
    shutdown: tokio_util::sync::CancellationToken,
) -> (Channel, std::net::SocketAddr, tokio::task::JoinHandle<()>) {
    let (health_reporter, health_service) = health_reporter();
    let mounted = services.mounted;
    super::mark_serving(&health_reporter, mounted).await;

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind grpc listener");
    let addr = listener.local_addr().expect("local addr");

    let server_task = tokio::spawn(async move {
        let incoming = MaybeTlsListener::plain(Listener::Tcp(listener));
        let mut builder = TonicServer::builder();
        let server = services
            .install(&mut builder)
            .add_service(health_service)
            .serve_with_incoming_shutdown(incoming, shutdown.clone().cancelled_owned());
        let health_monitor =
            super::monitor_health(health_reporter, engine_health, shutdown.clone(), mounted);
        let server = async move {
            let result = server.await;
            shutdown.cancel();
            result
        };
        let (server_result, ()) = tokio::join!(server, health_monitor);
        server_result.expect("grpc server");
    });

    let channel = Endpoint::from_shared(format!("http://{addr}"))
        .expect("grpc endpoint")
        .connect()
        .await
        .expect("connect grpc channel");

    (channel, addr, server_task)
}

/// Spin up a TLS gRPC server (server cert from `certs`, `cert_reqs` mTLS mode).
/// Returns the address, the server task, and the mock engine task.
async fn grpc_tls_test_server(
    engine_id: impl Into<EngineId>,
    output_specs: Vec<(Vec<u32>, Option<EngineCoreFinishReason>)>,
    certs: &TestCerts,
    cert_reqs: i32,
) -> (String, tokio::task::JoinHandle<()>, MockEngineTask) {
    let (services, _engine_health, engine_task) = setup_grpc_service(engine_id, output_specs).await;
    let context = tls::build_grpc_server_config(&server_tls(certs, cert_reqs))
        .expect("build grpc tls config");

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind grpc listener");
    let addr = listener.local_addr().expect("local addr").to_string();

    let server_task = tokio::spawn(async move {
        let incoming = MaybeTlsListener::tls(Listener::Tcp(listener), context);
        let mut builder = TonicServer::builder();
        services
            .install(&mut builder)
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
) -> Result<InferenceClient<Channel>, tonic::transport::Error> {
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
    Ok(InferenceClient::new(channel))
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
    let (services, _engine_health, engine_task) =
        setup_grpc_service(engine_id, default_stream_output_specs()).await;

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind grpc listener");
    let addr = listener.local_addr().expect("local addr").to_string();

    let mut builder = TonicServer::builder();
    if let Some(interval) = keepalive {
        builder = builder
            .http2_keepalive_interval(Some(interval))
            .http2_keepalive_timeout(Some(interval));
    }

    let server_task = tokio::spawn(async move {
        let incoming = MaybeTlsListener::plain(Listener::Tcp(listener));
        services
            .install(&mut builder)
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
async fn unary_generate_prepares_multimodal_input_for_engine_core() {
    let (services, engine_health, engine_task) = setup_grpc_service_with_backend(
        b"engine-grpc-multimodal",
        default_stream_output_specs(),
        multimodal_backend(),
        |request| {
            let token_ids = request.prompt_token_ids.as_ref().expect("prompt token ids");
            let features = request.mm_features.as_ref().expect("multimodal features");
            assert_eq!(features.len(), 2);

            for (feature, identifier) in features.iter().zip(["image-1", "image-2"]) {
                assert_eq!(feature.modality.as_str(), "image");
                assert_eq!(feature.identifier, identifier);
                assert!(feature.mm_position.length > 1);
                assert_eq!(
                    feature
                        .data
                        .as_ref()
                        .expect("multimodal feature data")
                        .keys()
                        .map(String::as_str)
                        .collect::<Vec<_>>(),
                    vec!["image_grid_thw"]
                );
            }
            assert_eq!(features[0].mm_position.offset, 1);
            let xargs = request
                .sampling_params
                .as_ref()
                .and_then(|params| params.extra_args.as_ref())
                .expect("KV transfer args");
            let kv_transfer_params = xargs.get("kv_transfer_params").expect("KV transfer params");
            assert_eq!(kv_transfer_params["pp_size"].as_i64(), Some(1));
            assert_eq!(
                kv_transfer_params["remote_block_ids"][0][0].as_i64(),
                Some(7)
            );
            assert!(!xargs.contains_key("ec_transfer_params"));
            assert_eq!(
                token_ids.len(),
                features.iter().map(|feature| feature.mm_position.length).sum::<usize>() + 3
            );
            assert_eq!(token_ids[0], 11);
            assert_eq!(token_ids.last(), Some(&13));
            for feature in features {
                assert!(
                    token_ids[feature.mm_position.offset
                        ..feature.mm_position.offset + feature.mm_position.length]
                        .iter()
                        .all(|token_id| *token_id == QWEN_IMAGE_TOKEN_ID)
                );
            }
        },
    )
    .await;
    let (channel, server_task) = start_grpc_test_server(
        services,
        engine_health,
        tokio_util::sync::CancellationToken::new(),
    )
    .await;
    let mut client = InferenceClient::new(channel);

    client
        .generate(pb::GenerateRequest {
            request_id: "test-multimodal".to_string(),
            model: "test-model".to_string(),
            prompt: Some(pb::generate_request::Prompt::TokenIds(pb::TokenIds {
                ids: vec![11, QWEN_IMAGE_TOKEN_ID, 12, QWEN_IMAGE_TOKEN_ID, 13],
            })),
            media: ["image-1", "image-2"]
                .into_iter()
                .map(|uuid| pb::MediaItem {
                    modality: pb::Modality::Image as i32,
                    source: Some(pb::media_item::Source::DataUri(
                        TINY_PNG_DATA_URI.to_string(),
                    )),
                    mime_type: String::new(),
                    uuid: uuid.to_string(),
                })
                .collect(),
            kv: Some(pb::KvCacheParameters {
                kv_transfer_params: Some(decode_kv_proto_struct()),
                ec_transfer_params: Some(ec_proto_struct(&["image-2", "image-1"])),
                ..Default::default()
            }),
            stopping: Some(pb::StoppingCriteria {
                max_new_tokens: 10,
                ..Default::default()
            }),
            ..Default::default()
        })
        .await
        .expect("multimodal generate");

    engine_task.await.expect("mock engine task");
    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn streaming_generate_rejects_text_prompt_with_media() {
    let (mut client, server_task, _engine_task) = grpc_test_server(
        b"engine-grpc-stream-media-text",
        default_stream_output_specs(),
    )
    .await;

    let status = client
        .generate_stream(pb::GenerateRequest {
            request_id: "test-stream-media-text".to_string(),
            model: "test-model".to_string(),
            prompt: Some(pb::generate_request::Prompt::Text(
                "describe this".to_string(),
            )),
            media: vec![pb::MediaItem {
                modality: pb::Modality::Image as i32,
                source: Some(pb::media_item::Source::DataUri(
                    TINY_PNG_DATA_URI.to_string(),
                )),
                mime_type: String::new(),
                uuid: "image-1".to_string(),
            }],
            ..Default::default()
        })
        .await
        .expect_err("text prompts with media must be rejected");

    assert_eq!(status.code(), tonic::Code::InvalidArgument);
    assert_eq!(
        status.message(),
        "multimodal gRPC requests must provide token_ids input"
    );
    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn unary_generate_accepts_request_larger_than_tonic_default() {
    let (mut client, server_task, _engine_task) =
        grpc_test_server(b"engine-grpc-large-media", default_stream_output_specs()).await;

    let status = client
        .generate(pb::GenerateRequest {
            request_id: "test-large-media".to_string(),
            model: "test-model".to_string(),
            prompt: Some(pb::generate_request::Prompt::Text(
                "describe this".to_string(),
            )),
            media: vec![pb::MediaItem {
                modality: pb::Modality::Image as i32,
                source: Some(pb::media_item::Source::RawBytes(vec![0; 5 * 1024 * 1024])),
                mime_type: "image/png".to_string(),
                uuid: "image-1".to_string(),
            }],
            ..Default::default()
        })
        .await
        .expect_err("text prompts with media must be rejected after decoding");

    assert_eq!(status.code(), tonic::Code::InvalidArgument);
    assert_eq!(
        status.message(),
        "multimodal gRPC requests must provide token_ids input"
    );
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
async fn unary_generate_unconnected_data_parallel_rank_returns_invalid_argument() {
    let (mut client, server_task, _engine_task) = grpc_test_server(
        EngineId::from_engine_index(3),
        default_stream_output_specs(),
    )
    .await;

    let mut request = tonic::Request::new(pb::GenerateRequest {
        request_id: "test-unconnected-dp-rank".to_string(),
        model: "test-model".to_string(),
        prompt: Some(pb::generate_request::Prompt::Text("hello".to_string())),
        ..Default::default()
    });
    request.metadata_mut().insert(
        "x-data-parallel-rank",
        "0".parse().expect("valid metadata value"),
    );

    let status = client
        .generate(request)
        .await
        .expect_err("rank 0 should not select globally ranked engine 3");

    assert_eq!(status.code(), tonic::Code::InvalidArgument);
    assert!(status.message().contains("connected ranks: [3]"));

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
async fn unary_generate_empty_stop_string_returns_invalid_argument() {
    let (mut client, server_task, _engine_task) =
        grpc_test_server(b"engine-grpc-empty-stop", default_stream_output_specs()).await;

    let status = client
        .generate(pb::GenerateRequest {
            request_id: "test-empty-stop".to_string(),
            model: "test-model".to_string(),
            prompt: Some(pb::generate_request::Prompt::Text("hi".to_string())),
            stopping: Some(pb::StoppingCriteria {
                stop_strings: vec!["".to_string()],
                ..Default::default()
            }),
            ..Default::default()
        })
        .await
        .expect_err("should fail when stop_strings contains an empty string");

    assert_eq!(status.code(), tonic::Code::InvalidArgument);
    assert!(status.message().contains("stop strings cannot be empty"));

    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn unary_generate_invalid_sampling_params_returns_invalid_argument() {
    let (mut client, server_task, _engine_task) = grpc_test_server(
        b"engine-grpc-invalid-sampling",
        default_stream_output_specs(),
    )
    .await;

    let status = client
        .generate(pb::GenerateRequest {
            request_id: "test-invalid-sampling".to_string(),
            model: "test-model".to_string(),
            prompt: Some(pb::generate_request::Prompt::Text("hi".to_string())),
            sampling: Some(pb::RandomSampling {
                top_p: 2.0,
                ..Default::default()
            }),
            ..Default::default()
        })
        .await
        .expect_err("should fail when top_p is out of range");

    assert_eq!(status.code(), tonic::Code::InvalidArgument);
    assert!(status.message().contains("top_p"));

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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn control_abort_resolves_external_id_and_empty_is_noop() {
    let (services, engine_health, engine_task) =
        setup_grpc_service(b"engine-grpc-abort-active", vec![(vec![b'h' as u32], None)]).await;
    let (channel, server_task) = start_grpc_test_server(
        services,
        engine_health,
        tokio_util::sync::CancellationToken::new(),
    )
    .await;
    let mut inference_client = InferenceClient::new(channel.clone());
    let mut control_client = ControlClient::new(channel);
    let request_id = "test-abort-active";

    let mut stream = inference_client
        .generate_stream(pb::GenerateRequest {
            request_id: request_id.to_string(),
            model: "test-model".to_string(),
            prompt: Some(pb::generate_request::Prompt::Text("hello".to_string())),
            stopping: Some(pb::StoppingCriteria {
                max_new_tokens: 10,
                ..Default::default()
            }),
            ..Default::default()
        })
        .await
        .expect("start generation")
        .into_inner();

    loop {
        let response = tokio::time::timeout(Duration::from_secs(2), stream.message())
            .await
            .expect("timed out waiting for active generation output")
            .expect("read active generation output")
            .expect("generation ended before producing output");
        if let Some(output) = response.outputs {
            assert!(
                output.finish_info.is_none(),
                "generation finished before abort behavior was exercised"
            );
            break;
        }
    }

    control_client
        .abort(pb::AbortRequest::default())
        .await
        .expect("empty abort should be a no-op");
    assert!(
        tokio::time::timeout(Duration::from_millis(100), stream.message())
            .await
            .is_err(),
        "empty abort unexpectedly ended the active generation"
    );

    control_client
        .abort(pb::AbortRequest {
            request_ids: vec![
                request_id.to_string(),
                request_id.to_string(),
                "unknown".to_string(),
            ],
        })
        .await
        .expect("abort active generation");

    let finish_reason = loop {
        let response = tokio::time::timeout(Duration::from_secs(2), stream.message())
            .await
            .expect("timed out waiting for aborted generation")
            .expect("read aborted generation")
            .expect("generation ended without an aborted response");
        if let Some(finish_info) = response.outputs.and_then(|output| output.finish_info) {
            break finish_info.finish_reason;
        }
    };
    assert_eq!(finish_reason, pb::finish_info::FinishReason::Aborted as i32);

    control_client
        .abort(pb::AbortRequest {
            request_ids: vec![request_id.to_string()],
        })
        .await
        .expect("repeated abort should be idempotent");

    engine_task.await.expect("mock engine task");
    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn control_reports_server_and_model_info() {
    let (services, engine_health, _engine_task) =
        setup_grpc_service(b"engine-grpc-info", default_stream_output_specs()).await;
    let (channel, server_task) = start_grpc_test_server(
        services,
        engine_health,
        tokio_util::sync::CancellationToken::new(),
    )
    .await;
    let mut client = ControlClient::new(channel);

    let server = client
        .get_server_info(pb::GetServerInfoRequest {})
        .await
        .expect("get server info")
        .into_inner();
    assert_eq!(server.engine_version, "test-vllm-version");
    assert_eq!(server.api_version, "vllm");
    assert_eq!(server.instance_id, "test-instance");
    assert_eq!(server.max_model_len, DEFAULT_MOCK_MAX_MODEL_LEN as u32);
    assert_eq!(server.kv_block_size, DEFAULT_MOCK_BLOCK_SIZE as u32);
    assert_eq!(server.total_kv_blocks, DEFAULT_MOCK_NUM_GPU_BLOCKS);
    assert_eq!(server.max_running_requests, 256);
    assert_eq!(server.max_batched_tokens, 8_192);
    let parallelism = server.parallelism.expect("parallelism metadata");
    assert_eq!(parallelism.tensor_parallel_size, 1);
    assert_eq!(parallelism.pipeline_parallel_size, 1);
    assert_eq!(parallelism.data_parallel_size, 1);
    assert_eq!(parallelism.data_parallel_rank, 0);
    assert_eq!(parallelism.decode_context_parallel_size, 1);
    let rl = server.rl_capabilities.expect("RL capabilities");
    assert!(!rl.weight_transfer_enabled);
    assert!(rl.weight_transfer_backend.is_empty());
    assert!(!rl.sleep_mode_enabled);
    assert!(!rl.draft_weight_updates_enabled);

    let model = client
        .get_model_info(pb::GetModelInfoRequest {})
        .await
        .expect("get model info")
        .into_inner();
    assert_eq!(model.model_id, "test-model");
    assert_eq!(model.served_model_name, "test-model");
    assert!(model.served_model_aliases.is_empty());
    assert!(model.supports_text_input);
    assert!(model.supports_token_ids_input);
    assert!(!model.supports_multimodal);
    assert!(model.reasoning_parser.is_empty());
    assert!(model.tool_call_parser.is_empty());

    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn unmounted_services_answer_unimplemented() {
    let (services, engine_health, _engine_task) =
        setup_grpc_service(b"engine-grpc-service-set", default_stream_output_specs()).await;
    let mounted = GrpcServiceSelection::Configured
        .resolve(&services.state.engine_core_client().ready_responses())
        .expect("configured selection resolves");
    assert_eq!(mounted, GrpcServices::INFERENCE | GrpcServices::CONTROL);

    let (channel, server_task) = start_grpc_test_server(
        services.mounting(mounted),
        engine_health,
        tokio_util::sync::CancellationToken::new(),
    )
    .await;

    let server = ControlClient::new(channel.clone())
        .get_server_info(pb::GetServerInfoRequest {})
        .await
        .expect("get server info")
        .into_inner();
    assert_eq!(
        server.services,
        vec![
            pb::GrpcService::Inference as i32,
            pb::GrpcService::Control as i32
        ]
    );

    let kv_transfer = KvTransferClient::new(channel.clone())
        .get_kv_event_sources(pb::GetKvEventSourcesRequest {})
        .await
        .expect_err("KV transfer service is not mounted");
    assert_eq!(kv_transfer.code(), tonic::Code::Unimplemented);

    let rl_control = RlControlClient::new(channel)
        .is_paused(pb::IsPausedRequest {})
        .await
        .expect_err("RL control service is not mounted");
    assert_eq!(rl_control.code(), tonic::Code::Unimplemented);

    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
#[allow(deprecated)]
async fn deprecated_control_aliases_match_the_moved_services() {
    let mut ready = default_ready_response();
    ready.enable_sleep_mode = true;
    ready.kv_events_config = Some(KvEventsConfig {
        enable_kv_cache_events: true,
        publisher: "zmq".to_string(),
        endpoint: "tcp://*:5559".to_string(),
        replay_endpoint: Some("tcp://*:5560".to_string()),
        buffer_steps: 10_000,
        hwm: 100_000,
        max_queue_size: 100_000,
        topic: "kv".to_string(),
    });
    let (services, engine_health, engine_task) = setup_grpc_service_with_engine_script(
        b"engine-grpc-deprecated-aliases".to_vec(),
        ready,
        Arc::new(FakeTextBackend),
        |dealer, push| {
            boxed_test_future(async move {
                reply_utility_bool(dealer, push, "is_scheduler_paused", true).await;
                reply_utility_bool(dealer, push, "is_scheduler_paused", true).await;
            })
        },
    )
    .await;
    let mounted = GrpcServiceSelection::Configured
        .resolve(&services.state.engine_core_client().ready_responses())
        .expect("configured selection resolves");
    assert_eq!(mounted, GrpcServices::all());

    let (channel, server_task) = start_grpc_test_server(
        services.mounting(mounted),
        engine_health,
        tokio_util::sync::CancellationToken::new(),
    )
    .await;
    let mut control_client = ControlClient::new(channel.clone());

    let via_control = control_client
        .get_kv_event_sources(pb::GetKvEventSourcesRequest {})
        .await
        .expect("deprecated Control alias serves KV event sources")
        .into_inner();
    let via_kv_transfer = KvTransferClient::new(channel.clone())
        .get_kv_event_sources(pb::GetKvEventSourcesRequest {})
        .await
        .expect("KvTransfer serves KV event sources")
        .into_inner();
    assert_eq!(via_control.sources.len(), 1);
    assert_eq!(via_control, via_kv_transfer);

    let via_control = control_client
        .is_paused(pb::IsPausedRequest {})
        .await
        .expect("deprecated Control alias serves IsPaused")
        .into_inner();
    let via_rl_control = RlControlClient::new(channel)
        .is_paused(pb::IsPausedRequest {})
        .await
        .expect("RlControl serves IsPaused")
        .into_inner();
    assert!(via_control.paused);
    assert_eq!(via_control, via_rl_control);

    engine_task.await.expect("mock engine task");
    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
#[allow(deprecated)]
async fn deprecated_control_aliases_follow_the_mounted_set() {
    let (services, engine_health, _engine_task) = setup_grpc_service(
        b"engine-grpc-deprecated-unmounted",
        default_stream_output_specs(),
    )
    .await;
    let (channel, server_task) = start_grpc_test_server(
        services.mounting(GrpcServices::CONTROL | GrpcServices::INFERENCE),
        engine_health,
        tokio_util::sync::CancellationToken::new(),
    )
    .await;
    let mut control_client = ControlClient::new(channel);

    let kv_transfer = control_client
        .get_kv_event_sources(pb::GetKvEventSourcesRequest {})
        .await
        .expect_err("KvTransfer is not mounted");
    assert_eq!(kv_transfer.code(), tonic::Code::Unimplemented);
    assert!(
        kv_transfer.message().contains("vllm.KvTransfer")
            && kv_transfer.message().contains("--grpc-services"),
        "unexpected message: {}",
        kv_transfer.message()
    );

    let rl_control = control_client
        .is_paused(pb::IsPausedRequest {})
        .await
        .expect_err("RlControl is not mounted");
    assert_eq!(rl_control.code(), tonic::Code::Unimplemented);
    assert!(
        rl_control.message().contains("vllm.RlControl")
            && rl_control.message().contains("--grpc-services"),
        "unexpected message: {}",
        rl_control.message()
    );

    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn control_lora_lifecycle_selects_adapter_for_generation() {
    let mut ready = default_ready_response();
    ready.supports_lora = true;
    ready.max_loras = 4;
    let (services, engine_health, engine_task) = setup_grpc_service_with_engine_script(
        b"engine-grpc-lora".to_vec(),
        ready,
        Arc::new(FakeTextBackend),
        |dealer, push| {
            boxed_test_future(async move {
                reply_utility_bool(dealer, push, "add_lora", true).await;

                let frames = recv_engine_message(dealer).await;
                assert_eq!(frames[0].as_ref(), &[0x00]);
                let request: EngineCoreRequest =
                    rmp_serde::from_slice(&frames[1]).expect("decode generation request");
                let lora = request.lora_request.expect("generation LoRA request");
                assert_eq!(lora.lora_name, "adapter");
                assert_eq!(lora.lora_int_id, 1);
                send_outputs(
                    push,
                    engine_outputs_for_request(
                        &request.request_id,
                        vec![(vec![b'!' as u32], Some(EngineCoreFinishReason::Stop))],
                    ),
                )
                .await;

                reply_utility_bool(dealer, push, "remove_lora", true).await;
            })
        },
    )
    .await;
    let (channel, server_task) = start_grpc_test_server(
        services,
        engine_health,
        tokio_util::sync::CancellationToken::new(),
    )
    .await;
    let mut control_client = ControlClient::new(channel.clone());
    let mut inference_client = InferenceClient::new(channel);

    let denied_source_path =
        std::env::temp_dir().join(format!("vllm-grpc-lora-denied-{}", std::process::id()));
    let denied = control_client
        .load_lora(pb::LoadLoraRequest {
            lora_name: "denied-adapter".to_string(),
            source_path: denied_source_path.to_string_lossy().into_owned(),
        })
        .await
        .expect_err("local LoRA path should be validated before engine load");
    assert_eq!(denied.code(), tonic::Code::InvalidArgument);

    let loaded = control_client
        .load_lora(pb::LoadLoraRequest {
            lora_name: "adapter".to_string(),
            source_path: "adapter".to_string(),
        })
        .await
        .expect("load LoRA")
        .into_inner();
    assert_eq!(
        loaded.adapter.as_ref().map(|adapter| adapter.lora_id),
        Some(1)
    );

    let duplicate = control_client
        .load_lora(pb::LoadLoraRequest {
            lora_name: "adapter".to_string(),
            source_path: "adapter".to_string(),
        })
        .await
        .expect_err("duplicate LoRA name should be rejected");
    assert_eq!(duplicate.code(), tonic::Code::AlreadyExists);

    let listed = control_client
        .list_loras(pb::ListLorasRequest {})
        .await
        .expect("list LoRAs")
        .into_inner();
    assert_eq!(listed.adapters.len(), 1);
    assert_eq!(listed.adapters[0].lora_name, "adapter");

    inference_client
        .generate(pb::GenerateRequest {
            request_id: "grpc-lora-request".to_string(),
            model: "test-model".to_string(),
            prompt: Some(pb::generate_request::Prompt::Text("hello".to_string())),
            stopping: Some(pb::StoppingCriteria {
                max_new_tokens: 1,
                ..Default::default()
            }),
            lora_name: "adapter".to_string(),
            ..Default::default()
        })
        .await
        .expect("generate with LoRA");

    let unloaded = control_client
        .unload_lora(pb::UnloadLoraRequest {
            lora_name: "adapter".to_string(),
        })
        .await
        .expect("unload LoRA")
        .into_inner();
    assert_eq!(
        unloaded.adapter.as_ref().map(|adapter| adapter.lora_id),
        Some(1)
    );
    assert!(
        control_client
            .list_loras(pb::ListLorasRequest {})
            .await
            .expect("list LoRAs after unload")
            .into_inner()
            .adapters
            .is_empty()
    );

    engine_task.await.expect("mock engine task");
    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn control_list_loras_requires_lora_enabled_engine() {
    let (services, engine_health, engine_task) = setup_grpc_service_with_engine_script(
        b"engine-grpc-lora-disabled".to_vec(),
        default_ready_response(),
        Arc::new(FakeTextBackend),
        |_, _| boxed_test_future(async move {}),
    )
    .await;
    let (channel, server_task) = start_grpc_test_server(
        services,
        engine_health,
        tokio_util::sync::CancellationToken::new(),
    )
    .await;
    let mut control_client = ControlClient::new(channel);
    let status = control_client
        .list_loras(pb::ListLorasRequest {})
        .await
        .expect_err("list should require LoRA support");
    assert_eq!(status.code(), tonic::Code::FailedPrecondition);
    assert_eq!(status.message(), "engine was not started with LoRA enabled");

    engine_task.await.expect("mock engine task");
    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn rl_control_forwards_weight_update_without_pause_guard() {
    let mut ready = default_ready_response();
    ready.weight_transfer_backend = Some("nccl".to_string());
    let (services, engine_health, engine_task) = setup_grpc_service_with_engine_script(
        b"engine-grpc-rl".to_vec(),
        ready,
        Arc::new(FakeTextBackend),
        |dealer, push| {
            boxed_test_future(async move {
                let frames = recv_engine_message(dealer).await;
                assert_eq!(frames[0].as_ref(), &[0x03]);
                let payload = decode_value(&frames[1]).expect("decode utility payload");
                let fields = payload.as_array().expect("utility payload array");
                let call_id = fields[1].as_u64().expect("utility call id");
                assert_eq!(fields[2].as_str(), Some("collective_rpc"));
                let args = fields[3].as_array().expect("collective_rpc arguments");
                assert_eq!(args[0].as_str(), Some("update_weights"));
                send_outputs(
                    push,
                    UtilityCallOutput {
                        output: UtilityOutput {
                            call_id: call_id.into(),
                            failure_message: None,
                            result: Some(UtilityResultEnvelope::without_type_info(Value::Array(
                                vec![Value::Nil],
                            ))),
                        },
                        ..Default::default()
                    }
                    .into(),
                )
                .await;
            })
        },
    )
    .await;
    let (channel, server_task) = start_grpc_test_server(
        services,
        engine_health,
        tokio_util::sync::CancellationToken::new(),
    )
    .await;
    let mut client = RlControlClient::new(channel);

    client
        .update_weights(pb::UpdateWeightsRequest {
            update_info_json: br#"{"names":["model.weight"]}"#.to_vec(),
        })
        .await
        .expect("forward weight update without a pause probe");

    engine_task.await.expect("mock engine task");
    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn control_aggregates_multi_engine_capacity() {
    let ipc = IpcNamespace::new().expect("create ipc namespace");
    let handshake_address = ipc.handshake_endpoint();

    let mut ready_0 = default_ready_response();
    ready_0.max_model_len = 8_192;
    ready_0.num_gpu_blocks = 10;
    ready_0.effective_data_parallel_size = 2;
    ready_0.tensor_parallel_size = 2;
    ready_0.pipeline_parallel_size = 3;
    ready_0.world_size = 12;
    ready_0.weight_transfer_backend = Some("nccl".to_string());
    ready_0.enable_sleep_mode = true;
    ready_0.supports_draft_weight_updates = true;

    let mut ready_1 = default_ready_response();
    ready_1.max_model_len = 4_096;
    ready_1.num_gpu_blocks = 20;
    ready_1.effective_data_parallel_size = 2;
    ready_1.tensor_parallel_size = 2;
    ready_1.pipeline_parallel_size = 3;
    ready_1.world_size = 12;
    ready_1.data_parallel_rank = 1;

    let engine_tasks = [ready_0, ready_1].map(|ready| {
        let engine_id = EngineId::from_engine_index(
            ready.data_parallel_rank.try_into().expect("test rank fits engine identity"),
        );
        MockEngineTask::new(spawn_mock_engine_task_with_ready(
            handshake_address.clone(),
            engine_id,
            ready,
            |_, _| boxed_test_future(async {}),
        ))
    });

    let client = EngineCoreClient::connect(EngineCoreClientConfig {
        transport_mode: TransportMode::HandshakeOwner {
            handshake_address,
            advertised_host: "127.0.0.1".to_string(),
            engine_count: 2,
            ready_timeout: Duration::from_secs(2),
            local_input_address: Some(ipc.input_endpoint()),
            local_output_address: Some(ipc.output_endpoint()),
        },
        coordinator_mode: None,
        model_name: "test-model".to_string(),
        client_index: 0,
    })
    .await
    .expect("connect multi-engine client");
    let chat = ChatLlm::from_shared_backend(
        Llm::new(client),
        Arc::new(FakeTextBackend) as Arc<dyn ChatTextBackend>,
    );
    let state =
        AppState::new(vec!["test-model".to_string()], chat).with_grpc_services(GrpcServices::all());
    let state = Arc::new(state);
    let service = ControlServiceImpl::new(
        state.clone(),
        Arc::new(KvTransferServiceImpl::new(state.clone())),
        Arc::new(RlControlServiceImpl::new(state)),
    );

    let server = pb::control_server::Control::get_server_info(
        &service,
        tonic::Request::new(pb::GetServerInfoRequest {}),
    )
    .await
    .expect("get server info")
    .into_inner();
    assert_eq!(server.max_model_len, 4_096);
    assert_eq!(server.total_kv_blocks, 30);
    let rl = server.rl_capabilities.expect("RL capabilities");
    assert!(!rl.weight_transfer_enabled);
    assert!(rl.weight_transfer_backend.is_empty());
    assert!(!rl.sleep_mode_enabled);
    assert!(!rl.draft_weight_updates_enabled);
    let parallelism = server.parallelism.unwrap();
    assert_eq!(parallelism.data_parallel_size, 2);
    assert_eq!(parallelism.world_size, 12);

    drop(engine_tasks);
}

#[test]
fn kv_transfer_engine_requires_kv_transfer_info() {
    let mut ready = default_ready_response();
    assert!(kv_transfer_engine(&ready, &test_handshake_entries()).is_none());

    ready.kv_transfer_info = Some(test_kv_transfer_info());
    ready.data_parallel_rank = 1;
    ready.tensor_parallel_size = 2;
    ready.pipeline_parallel_size = 1;
    ready.block_size = 16;
    let engine = kv_transfer_engine(&ready, &test_handshake_entries()).expect("kv transfer engine");
    expect_test::expect![[r#"
        KvTransferEngine {
            engine_id: "prefill-0_dp0",
            connector: "NixlConnector",
            role: "kv_producer",
            data_parallel_rank: 1,
            tensor_parallel_size: 2,
            pipeline_parallel_size: 1,
            kv_block_size: 16,
            compatibility_hash: "abc123",
        }
    "#]]
    .assert_debug_eq(&engine);

    let without_hash = kv_transfer_engine(&ready, &[]).expect("kv transfer engine");
    assert!(without_hash.compatibility_hash.is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn kv_transfer_rpcs_serve_cached_handshake_entries() {
    let mut ready = default_ready_response();
    ready.tensor_parallel_size = 2;
    ready.kv_transfer_info = Some(test_kv_transfer_info());
    let entries = test_handshake_entries();
    // Named map encoding, matching how the engine's msgpack encoder emits dataclasses.
    let entries_value = decode_value(&rmp_serde::to_vec_named(&entries).expect("encode entries"))
        .expect("decode handshake entries");
    let (services, engine_health, engine_task) = setup_grpc_service_with_engine_script(
        b"engine-grpc-kv-transfer".to_vec(),
        ready,
        Arc::new(FakeTextBackend),
        move |dealer, push| {
            boxed_test_future(async move {
                // Answer exactly once: later RPCs must be served from cache.
                reply_utility_value(
                    dealer,
                    push,
                    "get_kv_connector_handshake_entries",
                    entries_value,
                )
                .await;
            })
        },
    )
    .await;
    let (channel, server_task) = start_grpc_test_server(
        services,
        engine_health,
        tokio_util::sync::CancellationToken::new(),
    )
    .await;
    let mut kv_client = KvTransferClient::new(channel);

    let info = tokio::time::timeout(
        Duration::from_secs(5),
        kv_client.get_kv_transfer_info(pb::GetKvTransferInfoRequest {}),
    )
    .await
    .expect("get kv transfer info timed out")
    .expect("get kv transfer info")
    .into_inner();
    expect_test::expect![[r#"
        GetKvTransferInfoResponse {
            engines: [
                KvTransferEngine {
                    engine_id: "prefill-0_dp0",
                    connector: "NixlConnector",
                    role: "kv_producer",
                    data_parallel_rank: 0,
                    tensor_parallel_size: 2,
                    pipeline_parallel_size: 1,
                    kv_block_size: 16,
                    compatibility_hash: "abc123",
                },
            ],
        }
    "#]]
    .assert_debug_eq(&info);

    let metadata = tokio::time::timeout(
        Duration::from_secs(5),
        kv_client.get_kv_handshake_metadata(pb::GetKvHandshakeMetadataRequest {
            engine_id: "prefill-0_dp0".to_string(),
        }),
    )
    .await
    .expect("get kv handshake metadata timed out")
    .expect("get kv handshake metadata")
    .into_inner();
    expect_test::expect![[r#"
        GetKvHandshakeMetadataResponse {
            ranks: [
                KvHandshakeRank {
                    pp_rank: 0,
                    tp_rank: 0,
                    compatibility_hash: "abc123",
                    encoding: "msgpack",
                    payload: [
                        97,
                        103,
                        101,
                        110,
                        116,
                        45,
                        48,
                    ],
                },
                KvHandshakeRank {
                    pp_rank: 0,
                    tp_rank: 1,
                    compatibility_hash: "abc123",
                    encoding: "msgpack",
                    payload: [
                        97,
                        103,
                        101,
                        110,
                        116,
                        45,
                        49,
                    ],
                },
            ],
        }
    "#]]
    .assert_debug_eq(&metadata);

    let unknown = kv_client
        .get_kv_handshake_metadata(pb::GetKvHandshakeMetadataRequest {
            engine_id: "nope".to_string(),
        })
        .await
        .expect_err("unknown engine id is rejected");
    assert_eq!(unknown.code(), tonic::Code::NotFound);

    server_task.abort();
    drop(engine_task);
}

#[test]
fn kv_event_source_filters_and_exposes_zmq_publisher() {
    let mut ready = default_ready_response();
    ready.data_parallel_rank = 2;
    ready.kv_events_config = Some(KvEventsConfig {
        enable_kv_cache_events: false,
        publisher: "null".to_string(),
        endpoint: "tcp://*:5559".to_string(),
        replay_endpoint: Some("tcp://*:5560".to_string()),
        buffer_steps: 10_000,
        hwm: 100_000,
        max_queue_size: 100_000,
        topic: "kv".to_string(),
    });

    assert!(kv_event_source(&ready).is_none());

    let config = ready.kv_events_config.as_mut().unwrap();
    config.enable_kv_cache_events = true;
    config.publisher = "zmq".to_string();
    let source = kv_event_source(&ready).expect("configured ZMQ event source");
    assert_eq!(source.transport, "zmq");
    assert_eq!(source.endpoint, "tcp://*:5559");
    assert_eq!(source.topic, "kv");
    assert_eq!(source.replay_endpoint, "tcp://*:5560");
    assert_eq!(source.data_parallel_rank, Some(2));
    assert_eq!(source.encoding, "msgpack");
    assert_eq!(source.schema_version, 1);
    assert_eq!(source.buffer_steps, 10_000);
    assert_eq!(source.hwm, 100_000);
    assert_eq!(source.max_queue_size, 100_000);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn grpc_health_transitions_to_not_serving_when_engine_becomes_unhealthy() {
    let (services, _connected_engine_health, _engine_task) =
        setup_grpc_service(b"engine-grpc-health-failure", default_stream_output_specs()).await;
    let (engine_health_tx, engine_health) = tokio::sync::watch::channel(true);
    let (channel, server_task) = start_grpc_test_server(
        services,
        engine_health,
        tokio_util::sync::CancellationToken::new(),
    )
    .await;
    let mut health_client = HealthClient::new(channel);

    let mut health_streams = Vec::new();
    for service in [
        "vllm.Inference",
        "vllm.Control",
        "vllm.KvTransfer",
        "vllm.RlControl",
        "",
    ] {
        let service_label = if service.is_empty() {
            "overall"
        } else {
            service
        };
        let mut stream = health_client
            .watch(HealthCheckRequest {
                service: service.to_string(),
            })
            .await
            .unwrap_or_else(|error| {
                panic!("failed to start health watch for {service_label}: {error}")
            })
            .into_inner();
        let initial = stream
            .message()
            .await
            .unwrap_or_else(|error| {
                panic!("failed to read initial health status for {service_label}: {error}")
            })
            .unwrap_or_else(|| {
                panic!("health watch for {service_label} ended before its initial status")
            });
        assert_eq!(
            initial.status,
            HealthServingStatus::Serving as i32,
            "unexpected initial health status for {service_label}"
        );
        health_streams.push((service_label, stream));
    }

    engine_health_tx.send(false).expect("publish unhealthy engine state");

    for (service_label, mut stream) in health_streams {
        let update = tokio::time::timeout(Duration::from_secs(2), stream.message())
            .await
            .unwrap_or_else(|_| panic!("timed out waiting for health update for {service_label}"))
            .unwrap_or_else(|error| {
                panic!("failed to read health update for {service_label}: {error}")
            })
            .unwrap_or_else(|| panic!("health watch for {service_label} ended before its update"));
        assert_eq!(
            update.status,
            HealthServingStatus::NotServing as i32,
            "unexpected health status for {service_label}"
        );
    }

    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn grpc_health_reports_only_mounted_services() {
    let (services, engine_health, _engine_task) =
        setup_grpc_service(b"engine-grpc-health-subset", default_stream_output_specs()).await;
    let (channel, server_task) = start_grpc_test_server(
        services.mounting(GrpcServices::INFERENCE | GrpcServices::CONTROL),
        engine_health,
        tokio_util::sync::CancellationToken::new(),
    )
    .await;
    let mut health_client = HealthClient::new(channel);

    let mounted = health_client
        .check(HealthCheckRequest {
            service: "vllm.Control".to_string(),
        })
        .await
        .expect("mounted service is registered")
        .into_inner();
    assert_eq!(mounted.status, HealthServingStatus::Serving as i32);

    for service in ["vllm.KvTransfer", "vllm.RlControl"] {
        let error = health_client
            .check(HealthCheckRequest {
                service: service.to_string(),
            })
            .await
            .expect_err("unmounted service is not registered");
        assert_eq!(error.code(), tonic::Code::NotFound, "{service}");
    }

    server_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn grpc_health_watch_closes_on_graceful_shutdown() {
    let (services, engine_health, _engine_task) = setup_grpc_service(
        b"engine-grpc-health-shutdown",
        default_stream_output_specs(),
    )
    .await;
    let shutdown = tokio_util::sync::CancellationToken::new();
    let (channel, server_task) =
        start_grpc_test_server(services, engine_health, shutdown.clone()).await;
    let mut health_client = HealthClient::new(channel);
    let mut stream = health_client
        .watch(HealthCheckRequest {
            service: "vllm.Inference".to_string(),
        })
        .await
        .expect("start health watch for vllm.Inference")
        .into_inner();

    let initial = stream
        .message()
        .await
        .expect("read initial health status for vllm.Inference")
        .expect("health watch ended before its initial status");
    assert_eq!(
        initial.status,
        HealthServingStatus::Serving as i32,
        "unexpected initial health status for vllm.Inference"
    );

    shutdown.cancel();

    let update = tokio::time::timeout(Duration::from_secs(2), stream.message())
        .await
        .expect("timed out waiting for shutdown health update for vllm.Inference")
        .expect("failed to read shutdown health update for vllm.Inference")
        .expect("health watch ended before its shutdown update");
    assert_eq!(
        update.status,
        HealthServingStatus::NotServing as i32,
        "unexpected shutdown health status for vllm.Inference"
    );

    let stream_end = tokio::time::timeout(Duration::from_secs(2), stream.message())
        .await
        .expect("timed out waiting for vllm.Inference health watch to close")
        .expect("failed while closing vllm.Inference health watch");
    assert!(
        stream_end.is_none(),
        "vllm.Inference health watch remained open"
    );

    tokio::time::timeout(Duration::from_secs(2), server_task)
        .await
        .expect("timed out waiting for gRPC server shutdown")
        .expect("gRPC server task failed");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn kv_peer_handshake_pushes_prefill_metadata_before_generate() {
    // Prefill side: a frontend whose control plane serves handshake entries.
    let mut prefill_ready = default_ready_response();
    prefill_ready.kv_transfer_info = Some(test_kv_transfer_info());
    let entries = test_handshake_entries();
    let entries_value =
        decode_value(&rmp_serde::to_vec_named(&entries).expect("encode entries")).expect("decode");
    let (prefill_services, prefill_health, _prefill_engine) =
        setup_grpc_service_with_engine_script(
            b"engine-kv-peer-prefill".to_vec(),
            prefill_ready,
            Arc::new(FakeTextBackend),
            move |dealer, push| {
                boxed_test_future(async move {
                    reply_utility_value(
                        dealer,
                        push,
                        "get_kv_connector_handshake_entries",
                        entries_value,
                    )
                    .await;
                })
            },
        )
        .await;
    let (_prefill_channel, prefill_addr, prefill_server) = start_grpc_test_server_with_addr(
        prefill_services,
        prefill_health,
        tokio_util::sync::CancellationToken::new(),
    )
    .await;

    // Decode side: the engine must see the pushed handshake before any request,
    // and only once across two requests naming the same peer.
    let ipc = IpcNamespace::new().expect("create ipc namespace");
    let handshake_address = ipc.handshake_endpoint();
    let expected_entries = entries.clone();
    let decode_engine = MockEngineTask::new(spawn_mock_engine_task_with_ready(
        handshake_address.clone(),
        b"engine-kv-peer-decode".to_vec(),
        default_ready_response(),
        move |dealer, push| {
            boxed_test_future(async move {
                let frames = recv_engine_message(dealer).await;
                assert_eq!(
                    frames[0].as_ref(),
                    &[0x03],
                    "handshake push must precede the request"
                );
                let payload = decode_value(&frames[1]).expect("decode utility payload");
                let fields = payload.as_array().expect("utility payload array");
                let call_id = fields[1].as_u64().expect("utility call id");
                assert_eq!(fields[2].as_str(), Some("add_remote_kv_handshake"));
                let args = fields[3].as_array().expect("utility args");
                assert_eq!(args[0].as_str(), Some("prefill-0_dp0"));
                // The engine converts these with msgspec into a dataclass, which
                // needs named maps, not the positional arrays rmp_serde emits
                // for structs by default.
                assert!(
                    args[1].as_array().expect("entries").iter().all(Value::is_map),
                    "entries must be msgpack maps: {:?}",
                    args[1]
                );
                let pushed: Vec<KvConnectorHandshakeEntry> =
                    rmpv::ext::from_value(args[1].clone()).expect("pushed entries");
                assert_eq!(pushed, expected_entries);
                send_outputs(
                    push,
                    UtilityCallOutput {
                        output: UtilityOutput {
                            call_id: call_id.into(),
                            failure_message: None,
                            result: Some(UtilityResultEnvelope::without_type_info(Value::Nil)),
                        },
                        ..Default::default()
                    }
                    .into(),
                )
                .await;
                for _ in 0..2 {
                    let add = recv_engine_message(dealer).await;
                    assert_eq!(add[0].as_ref(), &[0x00]);
                    let request: EngineCoreRequest =
                        rmp_serde::from_slice(&add[1]).expect("decode generation request");
                    send_outputs(
                        push,
                        engine_outputs_for_request(
                            &request.request_id,
                            vec![(vec![7], Some(EngineCoreFinishReason::Stop))],
                        ),
                    )
                    .await;
                }
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
    .expect("connect decode client");
    let llm = Llm::new(client).with_kv_peer_handshake(Arc::new(KvPeerHandshaker::new()));

    let request = |id: &str| {
        let mut sampling_params = EngineCoreSamplingParams::for_test();
        sampling_params.max_tokens = 1;
        sampling_params.extra_args = Some(std::collections::HashMap::from([(
            "kv_transfer_params".to_string(),
            serde_json::json!({
                "do_remote_prefill": true,
                "remote_engine_id": "prefill-0_dp0",
                "remote_host": prefill_addr.ip().to_string(),
                "remote_port": 5600,
                "remote_control_port": prefill_addr.port(),
            }),
        )]));
        GenerateRequest {
            request_id: id.to_string(),
            prompt_token_ids: vec![1, 2, 3],
            sampling_params,
            mm_features: None,
            arrival_time: None,
            cache_salt: None,
            trace_headers: None,
            priority: 0,
            data_parallel_rank: None,
            session_id: None,
            reasoning_parser_kwargs: None,
            lora_request: None,
        }
    };
    for id in ["req-1", "req-2"] {
        let mut stream = llm.generate(request(id)).await.expect("generate");
        let mut finished = false;
        while let Some(output) = stream.next().await {
            finished |= output.expect("generate output").finished();
        }
        assert!(finished, "{id} did not finish");
    }

    prefill_server.abort();
    drop(decode_engine);
}
