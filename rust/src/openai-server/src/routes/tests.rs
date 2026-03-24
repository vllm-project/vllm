use std::convert::TryFrom;
use std::net::TcpListener;
use std::sync::Arc;
use std::time::Duration;

use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode};
use bytes::Bytes;
use futures::StreamExt as _;
use serde_json::json;
use serial_test::serial;
use tower::util::ServiceExt as _;
use vllm_chat::{
    ChatBackend, ChatEvent, ChatLlm, ChatMessage, ChatRequest, ChatRole, ChatTextBackend,
    ChatToolChoice, SamplingParams,
};
use vllm_engine_core_client::protocol::handshake::{HandshakeInitMessage, ReadyMessage};
use vllm_engine_core_client::protocol::{
    EngineCoreOutput, EngineCoreOutputs, EngineCoreRequest, FinishReason, RequestOutputKind,
    StopReason,
};
use vllm_engine_core_client::{
    ENGINE_CORE_DEAD_SENTINEL, EngineCoreClient, EngineCoreClientConfig,
};
use vllm_llm::Llm;
use vllm_metrics::METRICS;
use vllm_text::TextBackend;
use zeromq::prelude::{Socket, SocketRecv, SocketSend};
use zeromq::util::PeerIdentity;
use zeromq::{DealerSocket, PushSocket, SocketOptions, ZmqMessage};

use super::build_router;
use crate::routes::chat_completions::convert::prepare_chat_request;
use crate::state::AppState;

fn unique_tcp_endpoint() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    let port = listener.local_addr().expect("read local addr").port();
    drop(listener);
    format!("tcp://127.0.0.1:{port}")
}

fn ready_message(status: &str) -> ReadyMessage {
    ReadyMessage {
        status: Some(status.to_string()),
        local: Some(true),
        headless: Some(true),
        num_gpu_blocks: None,
        dp_stats_address: None,
        parallel_config_hash: None,
    }
}

fn request_output(
    request_id: &str,
    new_token_ids: Vec<u32>,
    finish_reason: Option<FinishReason>,
) -> EngineCoreOutput {
    request_output_with_stop_reason(request_id, new_token_ids, finish_reason, None)
}

fn request_output_with_stop_reason(
    request_id: &str,
    new_token_ids: Vec<u32>,
    finish_reason: Option<FinishReason>,
    stop_reason: Option<StopReason>,
) -> EngineCoreOutput {
    EngineCoreOutput {
        request_id: request_id.to_string(),
        new_token_ids,
        new_logprobs: None,
        new_prompt_logprobs_tensors: None,
        pooling_output: None,
        finish_reason,
        stop_reason,
        events: None,
        kv_transfer_params: None,
        trace_headers: None,
        num_cached_tokens: 0,
        num_external_computed_tokens: 0,
        routed_experts: None,
        num_nans_in_logits: 0,
    }
}

fn bytes_to_token_ids(bytes: &[u8]) -> Vec<u32> {
    bytes.iter().map(|byte| u32::from(*byte)).collect()
}

fn default_stream_output_specs() -> Vec<(Vec<u32>, Option<FinishReason>)> {
    vec![
        (vec![b'h' as u32], None),
        (vec![b'i' as u32], None),
        (vec![b'!' as u32], Some(FinishReason::Stop)),
    ]
}

fn sse_data_payloads(text: &str) -> Vec<&str> {
    text.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .collect()
}

fn engine_outputs_for_request(
    request_id: &str,
    output_specs: Vec<(Vec<u32>, Option<FinishReason>)>,
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

async fn send_outputs(push: &mut PushSocket, outputs: EngineCoreOutputs) {
    push.send(ZmqMessage::from(
        rmp_serde::to_vec_named(&outputs).expect("encode outputs"),
    ))
    .await
    .expect("send outputs");
}

async fn recv_engine_message(dealer: &mut DealerSocket) -> Vec<Bytes> {
    dealer.recv().await.expect("recv engine message").into_vec()
}

async fn setup_mock_engine(
    engine_handshake: String,
    engine_identity: Vec<u8>,
) -> (DealerSocket, PushSocket) {
    tokio::time::sleep(Duration::from_millis(200)).await;

    let mut options = SocketOptions::default();
    options.peer_identity(PeerIdentity::try_from(engine_identity.clone()).expect("peer id"));
    let mut handshake = DealerSocket::with_options(options);
    handshake
        .connect(&engine_handshake)
        .await
        .expect("connect handshake");
    handshake
        .send(ZmqMessage::from(
            rmp_serde::to_vec_named(&ready_message("HELLO")).expect("encode ready"),
        ))
        .await
        .expect("send hello");

    let init_frames = handshake.recv().await.expect("recv init").into_vec();
    let init: HandshakeInitMessage =
        rmp_serde::from_slice(init_frames[0].as_ref()).expect("decode init");

    let mut input_options = SocketOptions::default();
    input_options.peer_identity(PeerIdentity::try_from(engine_identity).expect("peer id"));
    let mut dealer = DealerSocket::with_options(input_options);
    dealer
        .connect(&init.addresses.inputs[0])
        .await
        .expect("connect input");
    dealer
        .send(ZmqMessage::from(Vec::<u8>::new()))
        .await
        .expect("send ready frame");

    let mut push = PushSocket::new();
    push.connect(&init.addresses.outputs[0])
        .await
        .expect("connect output");

    handshake
        .send(ZmqMessage::from(
            rmp_serde::to_vec_named(&ready_message("READY")).expect("encode ready"),
        ))
        .await
        .expect("send ready");

    (dealer, push)
}

#[derive(Clone, Debug)]
struct FakeChatBackend {
    model_id: Option<String>,
}

impl FakeChatBackend {
    fn new() -> Self {
        Self { model_id: None }
    }

    fn with_model_id(model_id: impl Into<String>) -> Self {
        Self {
            model_id: Some(model_id.into()),
        }
    }
}

impl TextBackend for FakeChatBackend {
    fn encode(&self, text: &str) -> vllm_text::Result<Vec<u32>> {
        Ok(text.bytes().map(u32::from).collect())
    }

    fn decode(&self, token_ids: &[u32], _skip_special_tokens: bool) -> vllm_text::Result<String> {
        Ok(
            String::from_utf8_lossy(&token_ids.iter().map(|id| *id as u8).collect::<Vec<_>>())
                .into_owned(),
        )
    }

    fn model_id(&self) -> Option<&str> {
        self.model_id.as_deref()
    }
}

impl ChatBackend for FakeChatBackend {
    fn apply_chat_template(&self, request: &ChatRequest) -> vllm_chat::Result<String> {
        let mut prompt = String::new();
        for message in &request.messages {
            prompt.push_str(message.role().as_str());
            prompt.push_str(": ");
            prompt.push_str(&message.text_content()?);
            prompt.push('\n');
        }
        if request.chat_options.add_generation_prompt {
            prompt.push_str("assistant:");
        }
        Ok(prompt)
    }
}

#[derive(Clone, Debug)]
struct FailingDecodeChatBackend;

impl TextBackend for FailingDecodeChatBackend {
    fn encode(&self, text: &str) -> vllm_text::Result<Vec<u32>> {
        FakeChatBackend::new().encode(text)
    }

    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> vllm_text::Result<String> {
        if token_ids.contains(&(b'i' as u32)) {
            return Err(vllm_text::Error::Tokenizer(
                "forced decode failure for streaming test".to_string(),
            ));
        }

        FakeChatBackend::new().decode(token_ids, skip_special_tokens)
    }
}

impl ChatBackend for FailingDecodeChatBackend {
    fn apply_chat_template(&self, request: &ChatRequest) -> vllm_chat::Result<String> {
        FakeChatBackend::new().apply_chat_template(request)
    }
}

async fn test_models_with_engine_outputs_and_backend_inner(
    engine_identity: &[u8],
    output_specs: Vec<(Vec<u32>, Option<FinishReason>)>,
    expected_prompt_token_ids: Option<Vec<u32>>,
    backend: Arc<dyn ChatTextBackend>,
) -> (ChatLlm, tokio::task::JoinHandle<()>) {
    let handshake_address = unique_tcp_endpoint();
    let engine_identity = engine_identity.to_vec();

    let engine_task = tokio::spawn({
        let engine_handshake = handshake_address.clone();
        let engine_identity = engine_identity.clone();
        let expected_prompt_token_ids = expected_prompt_token_ids.clone();
        async move {
            let (mut dealer, mut push) = setup_mock_engine(engine_handshake, engine_identity).await;
            let add = recv_engine_message(&mut dealer).await;
            let request: EngineCoreRequest =
                rmp_serde::from_slice(&add[1]).expect("decode request");
            if let Some(expected_prompt_token_ids) = expected_prompt_token_ids {
                assert_eq!(
                    request.prompt_token_ids.as_deref(),
                    Some(expected_prompt_token_ids.as_slice())
                );
            }
            send_outputs(
                &mut push,
                engine_outputs_for_request(&request.request_id, output_specs),
            )
            .await;
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    });

    let client = EngineCoreClient::connect(EngineCoreClientConfig {
        handshake_address,
        model_name: "test-model".to_string(),
        local_host: "127.0.0.1".to_string(),
        ready_timeout: Duration::from_secs(2),
        client_index: 0,
    })
    .await
    .expect("connect client");

    (
        ChatLlm::from_shared_backend(Llm::new(client), backend),
        engine_task,
    )
}

async fn test_models_with_engine_outputs_and_backend(
    engine_identity: &[u8],
    output_specs: Vec<(Vec<u32>, Option<FinishReason>)>,
    backend: Arc<dyn ChatTextBackend>,
) -> (ChatLlm, tokio::task::JoinHandle<()>) {
    test_models_with_engine_outputs_and_backend_inner(engine_identity, output_specs, None, backend)
        .await
}

async fn test_chat_with_engine_outputs(
    engine_identity: &[u8],
    output_specs: Vec<(Vec<u32>, Option<FinishReason>)>,
) -> (ChatLlm, tokio::task::JoinHandle<()>) {
    test_models_with_engine_outputs_and_backend(
        engine_identity,
        output_specs,
        Arc::new(FakeChatBackend::new()),
    )
    .await
}

async fn test_app() -> axum::Router {
    let (chat, _engine_task) = test_models_with_engine_outputs_and_backend(
        b"engine-openai",
        default_stream_output_specs(),
        Arc::new(FakeChatBackend::new()),
    )
    .await;
    build_router(Arc::new(AppState::new("Qwen/Qwen1.5-0.5B-Chat", chat)))
}

async fn test_health_app_with_engine_script<F, Fut>(
    script: F,
) -> (axum::Router, Arc<AppState>, tokio::task::JoinHandle<()>)
where
    F: FnOnce(PushSocket) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = ()> + Send + 'static,
{
    let handshake_address = unique_tcp_endpoint();
    let engine_identity = b"engine-openai-health".to_vec();

    let engine_task = tokio::spawn({
        let engine_handshake = handshake_address.clone();
        let engine_identity = engine_identity.clone();
        async move {
            let (_dealer, push) = setup_mock_engine(engine_handshake, engine_identity).await;
            script(push).await;
        }
    });

    let client = EngineCoreClient::connect(EngineCoreClientConfig {
        handshake_address,
        model_name: "test-model".to_string(),
        local_host: "127.0.0.1".to_string(),
        ready_timeout: Duration::from_secs(2),
        client_index: 0,
    })
    .await
    .expect("connect client");

    let chat = ChatLlm::from_shared_backend(Llm::new(client), Arc::new(FakeChatBackend::new()));
    let state = Arc::new(AppState::new("Qwen/Qwen1.5-0.5B-Chat", chat));
    (build_router(state.clone()), state, engine_task)
}

async fn test_app_with_engine_handle() -> (axum::Router, tokio::task::JoinHandle<()>) {
    test_app_with_stream_output_specs(default_stream_output_specs()).await
}

async fn test_app_with_stream_output_specs(
    output_specs: Vec<(Vec<u32>, Option<FinishReason>)>,
) -> (axum::Router, tokio::task::JoinHandle<()>) {
    let (chat, engine_task) = test_models_with_engine_outputs_and_backend(
        b"engine-openai",
        output_specs,
        Arc::new(FakeChatBackend::new()),
    )
    .await;
    (
        build_router(Arc::new(AppState::new("Qwen/Qwen1.5-0.5B-Chat", chat))),
        engine_task,
    )
}

async fn test_app_with_backend_and_stream_output_specs(
    backend: Arc<dyn ChatTextBackend>,
    output_specs: Vec<(Vec<u32>, Option<FinishReason>)>,
) -> (axum::Router, tokio::task::JoinHandle<()>) {
    let (chat, engine_task) =
        test_models_with_engine_outputs_and_backend(b"engine-openai", output_specs, backend).await;
    (
        build_router(Arc::new(AppState::new("Qwen/Qwen1.5-0.5B-Chat", chat))),
        engine_task,
    )
}

async fn test_chat_with_engine_handle() -> (ChatLlm, tokio::task::JoinHandle<()>) {
    test_chat_with_engine_outputs(b"engine-openai-chat", default_stream_output_specs()).await
}

async fn server_load(app: &axum::Router) -> u64 {
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/load")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    let value: serde_json::Value = serde_json::from_slice(&body).expect("json body");
    value["server_load"].as_u64().expect("server_load")
}

async fn health_status(app: &axum::Router) -> (StatusCode, Bytes) {
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/health")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");

    let status = response.status();
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    (status, body)
}

fn metric_value(rendered: &str, metric: &str, labels: Option<&str>) -> Option<f64> {
    rendered.lines().find_map(|line| {
        let Some(rest) = line.strip_prefix(metric) else {
            return None;
        };

        match labels {
            Some(labels) => {
                let Some((encoded_labels, value)) = rest.split_once("} ") else {
                    return None;
                };
                if !encoded_labels.starts_with('{') {
                    return None;
                }
                let expected_parts = labels.split(',');
                if expected_parts
                    .into_iter()
                    .all(|part| encoded_labels.contains(part))
                {
                    value.parse::<f64>().ok()
                } else {
                    None
                }
            }
            None => rest
                .strip_prefix(' ')
                .and_then(|value| value.parse::<f64>().ok()),
        }
    })
}

fn metric_delta(
    rendered_before: &str,
    rendered_after: &str,
    metric: &str,
    labels: Option<&str>,
) -> f64 {
    metric_value(rendered_after, metric, labels).unwrap_or(0.0)
        - metric_value(rendered_before, metric, labels).unwrap_or(0.0)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn list_models_returns_configured_model() {
    let app = test_app().await;
    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");
    assert_eq!(json["data"][0]["id"], "Qwen/Qwen1.5-0.5B-Chat");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn http_metrics_record_list_models_requests() {
    let app = test_app().await;
    let before = METRICS.render().unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/models")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);

    let after = METRICS.render().unwrap();
    assert_eq!(
        metric_delta(
            &before,
            &after,
            "http_requests_total",
            Some("method=\"GET\",status=\"2xx\",handler=\"/v1/models\""),
        ),
        1.0
    );
    assert_eq!(
        metric_delta(
            &before,
            &after,
            "http_request_duration_seconds_count",
            Some("method=\"GET\",handler=\"/v1/models\""),
        ),
        1.0
    );
    assert_eq!(
        metric_delta(
            &before,
            &after,
            "http_request_duration_highr_seconds_count",
            None,
        ),
        1.0
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn wrong_model_returns_not_found() {
    let app = test_app().await;
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "wrong-model",
                        "stream": true,
                        "messages": [{"role": "user", "content": "hello"}]
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn invalid_request_returns_openai_error() {
    let app = test_app().await;
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "stream": false,
                        "stream_options": {"include_usage": true},
                        "messages": [{"role": "user", "content": "hello"}]
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");
    assert_eq!(json["error"]["type"], "invalid_request_error");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn non_stream_chat_returns_json_response() {
    let (app, engine_task) = test_app_with_engine_handle().await;
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "stream": false,
                        "messages": [{"role": "user", "content": "hello"}]
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    assert!(
        response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.starts_with("application/json"))
    );

    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    engine_task.await.expect("mock engine task");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");

    assert_eq!(json["object"], "chat.completion");
    assert_eq!(json["choices"][0]["message"]["role"], "assistant");
    assert_eq!(json["choices"][0]["message"]["content"], "hi");
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
    assert_eq!(json["usage"]["prompt_tokens"], 22);
    assert_eq!(json["usage"]["completion_tokens"], 3);
    assert_eq!(json["usage"]["total_tokens"], 25);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn happy_path_returns_sse_stream() {
    let (app, engine_task) = test_app_with_engine_handle().await;
    let before = METRICS.render().unwrap();
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "stream": true,
                        "messages": [{"role": "user", "content": "hello"}]
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok()),
        Some("text/event-stream")
    );

    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    engine_task.await.expect("mock engine task");
    let text = String::from_utf8(body.to_vec()).expect("utf8 body");
    let after = METRICS.render().unwrap();

    assert!(text.contains("\"role\":\"assistant\""), "{text}");
    assert!(text.starts_with("data: "), "{text}");
    assert_eq!(
        metric_delta(
            &before,
            &after,
            "http_requests_total",
            Some("method=\"POST\",status=\"2xx\",handler=\"/v1/chat/completions\""),
        ),
        1.0
    );
    assert_eq!(
        metric_delta(
            &before,
            &after,
            "http_request_duration_seconds_count",
            Some("method=\"POST\",handler=\"/v1/chat/completions\""),
        ),
        1.0
    );
    assert_eq!(
        metric_delta(
            &before,
            &after,
            "http_request_duration_highr_seconds_count",
            None,
        ),
        1.0
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn http_metrics_exclude_metrics_route() {
    let app = test_app().await;
    let before = METRICS.render().unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/metrics")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);

    let after = METRICS.render().unwrap();
    assert_eq!(
        metric_delta(
            &before,
            &after,
            "http_request_duration_highr_seconds_count",
            None,
        ),
        0.0
    );
    assert_eq!(
        metric_value(
            &after,
            "http_requests_total",
            Some("method=\"GET\",status=\"2xx\",handler=\"/metrics\""),
        ),
        None
    );
    assert_eq!(
        metric_value(
            &after,
            "http_request_duration_seconds_count",
            Some("method=\"GET\",handler=\"/metrics\""),
        ),
        None
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn http_metrics_group_error_statuses() {
    let app = test_app().await;
    let before = METRICS.render().unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "stream": false,
                        "stream_options": {"include_usage": true},
                        "messages": [{"role": "user", "content": "hello"}]
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    let after = METRICS.render().unwrap();
    assert_eq!(
        metric_delta(
            &before,
            &after,
            "http_requests_total",
            Some("method=\"POST\",status=\"4xx\",handler=\"/v1/chat/completions\""),
        ),
        1.0
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn load_endpoint_tracks_chat_stream_lifecycle() {
    let (app, engine_task) = test_app_with_engine_handle().await;

    assert_eq!(server_load(&app).await, 0);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "stream": true,
                        "messages": [{"role": "user", "content": "hello"}]
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(server_load(&app).await, 1);

    let _body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    engine_task.await.expect("mock engine task");

    assert_eq!(server_load(&app).await, 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn health_endpoint_returns_ok_with_empty_body_when_client_is_healthy() {
    let (app, _state, engine_task) = test_health_app_with_engine_script(|_push| async move {
        tokio::time::sleep(Duration::from_millis(200)).await;
    })
    .await;

    let (status, body) = health_status(&app).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.is_empty(), "expected empty body, got {:?}", body);

    engine_task.await.expect("mock engine task");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn health_endpoint_returns_503_after_engine_core_dead_sentinel() {
    let (app, state, engine_task) = test_health_app_with_engine_script(|mut push| async move {
        push.send(ZmqMessage::from(ENGINE_CORE_DEAD_SENTINEL.to_vec()))
            .await
            .expect("send sentinel");
    })
    .await;

    tokio::time::timeout(Duration::from_secs(2), async {
        while state.chat.engine_core_client().is_healthy() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("wait for unhealthy client");

    let (status, body) = health_status(&app).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert!(body.is_empty(), "expected empty body, got {:?}", body);
    assert!(matches!(
        state.chat.engine_core_client().health_error().as_deref(),
        Some(vllm_engine_core_client::Error::EngineCoreDead)
    ));

    engine_task.await.expect("mock engine task");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn load_endpoint_resets_when_stream_response_is_dropped() {
    let (app, engine_task) = test_app_with_engine_handle().await;

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "prompt": "hello",
                        "stream": true
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(server_load(&app).await, 1);

    drop(response);
    tokio::task::yield_now().await;
    engine_task.await.expect("mock engine task");

    assert_eq!(server_load(&app).await, 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn stream_error_is_returned_as_openai_error_sse() {
    let (app, engine_task) = test_app_with_backend_and_stream_output_specs(
        Arc::new(FailingDecodeChatBackend),
        default_stream_output_specs(),
    )
    .await;
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "stream": true,
                        "stream_options": {"include_usage": true},
                        "messages": [{"role": "user", "content": "hello"}]
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);

    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    engine_task.await.expect("mock engine task");
    let text = String::from_utf8(body.to_vec()).expect("utf8 body");

    assert!(text.contains("\"role\":\"assistant\""), "{text}");
    assert!(text.contains("\"type\":\"server_error\""), "{text}");
    assert!(
        text.contains("forced decode failure for streaming test"),
        "{text}"
    );
    assert!(!text.contains("\"usage\":"), "{text}");
    assert!(text.trim_end().ends_with("data: [DONE]"), "{text}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn invalid_terminal_finish_reason_is_returned_as_openai_error_sse() {
    let (app, engine_task) =
        test_app_with_stream_output_specs(vec![(vec![], Some(FinishReason::Error))]).await;
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "stream": true,
                        "stream_options": {"include_usage": true},
                        "messages": [{"role": "user", "content": "hello"}]
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);

    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    engine_task.await.expect("mock engine task");
    let text = String::from_utf8(body.to_vec()).expect("utf8 body");

    assert!(text.contains("\"role\":\"assistant\""), "{text}");
    assert!(text.contains("\"type\":\"server_error\""), "{text}");
    assert!(
        text.contains("stream terminated without a valid OpenAI finish reason"),
        "{text}"
    );
    assert!(!text.contains("\"usage\":"), "{text}");
    assert!(text.trim_end().ends_with("data: [DONE]"), "{text}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn include_usage_adds_final_usage_chunk_before_done() {
    let (app, engine_task) = test_app_with_stream_output_specs(default_stream_output_specs()).await;
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "stream": true,
                        "stream_options": {"include_usage": true},
                        "messages": [{"role": "user", "content": "hello"}]
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);

    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    engine_task.await.expect("mock engine task");
    let text = String::from_utf8(body.to_vec()).expect("utf8 body");

    let payloads = sse_data_payloads(&text);
    let finish_index = payloads
        .iter()
        .position(|payload| payload.contains("\"finish_reason\":\"stop\""))
        .expect("finish chunk");
    let usage_index = payloads
        .iter()
        .position(|payload| payload.contains("\"usage\":"))
        .expect("usage chunk");
    let done_index = payloads
        .iter()
        .position(|payload| *payload == "[DONE]")
        .expect("done sentinel");

    assert!(finish_index < usage_index, "{text}");
    assert!(usage_index < done_index, "{text}");

    let usage_chunk: serde_json::Value =
        serde_json::from_str(payloads[usage_index]).expect("usage chunk json");
    assert_eq!(usage_chunk["choices"], json!([]));
    assert_eq!(usage_chunk["usage"]["prompt_tokens"], 22);
    assert_eq!(usage_chunk["usage"]["completion_tokens"], 3);
    assert_eq!(usage_chunk["usage"]["total_tokens"], 25);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn stream_without_include_usage_keeps_existing_shape() {
    let (app, engine_task) = test_app_with_stream_output_specs(default_stream_output_specs()).await;
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "stream": true,
                        "messages": [{"role": "user", "content": "hello"}]
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);

    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    engine_task.await.expect("mock engine task");
    let text = String::from_utf8(body.to_vec()).expect("utf8 body");

    assert!(!text.contains("\"usage\":"), "{text}");
    assert!(text.contains("\"finish_reason\":\"stop\""), "{text}");
    assert!(text.trim_end().ends_with("data: [DONE]"), "{text}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn completions_invalid_request_returns_openai_error() {
    let app = test_app().await;
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "prompt": "hello",
                        "stream": false,
                        "stream_options": {"include_usage": true}
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");
    assert_eq!(json["error"]["type"], "invalid_request_error");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn non_stream_completions_return_json_response() {
    let (app, engine_task) = test_app_with_engine_handle().await;
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "prompt": "hello",
                        "stream": false
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    assert!(
        response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.starts_with("application/json"))
    );

    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    engine_task.await.expect("mock engine task");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");

    assert_eq!(json["object"], "text_completion");
    assert_eq!(json["choices"][0]["text"], "hi");
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
    assert_eq!(json["usage"]["completion_tokens"], 3);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn non_stream_completions_echo_prepends_prompt_text() {
    let (app, engine_task) = test_app_with_engine_handle().await;
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "prompt": "hello",
                        "echo": true,
                        "stream": false
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);

    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    engine_task.await.expect("mock engine task");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");

    assert_eq!(json["choices"][0]["text"], "hellohi");
    assert_eq!(json["usage"]["prompt_tokens"], 5);
    assert_eq!(json["usage"]["completion_tokens"], 3);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn non_stream_chat_uses_final_only_output_kind() {
    let handshake_address = unique_tcp_endpoint();
    let engine_identity = b"engine-openai-chat-final-only".to_vec();

    let engine_task = tokio::spawn({
        let engine_handshake = handshake_address.clone();
        let engine_identity = engine_identity.clone();
        async move {
            let (mut dealer, mut push) = setup_mock_engine(engine_handshake, engine_identity).await;
            let add = recv_engine_message(&mut dealer).await;
            let request: EngineCoreRequest =
                rmp_serde::from_slice(&add[1]).expect("decode request");
            assert_eq!(
                request
                    .sampling_params
                    .as_ref()
                    .expect("sampling params")
                    .output_kind,
                RequestOutputKind::FinalOnly
            );
            send_outputs(
                &mut push,
                engine_outputs_for_request(&request.request_id, default_stream_output_specs()),
            )
            .await;
        }
    });

    let client = EngineCoreClient::connect(EngineCoreClientConfig {
        handshake_address,
        model_name: "test-model".to_string(),
        local_host: "127.0.0.1".to_string(),
        ready_timeout: Duration::from_secs(2),
        client_index: 0,
    })
    .await
    .expect("connect client");
    let chat = ChatLlm::from_shared_backend(Llm::new(client), Arc::new(FakeChatBackend::new()));
    let app = build_router(Arc::new(AppState::new("Qwen/Qwen1.5-0.5B-Chat", chat)));

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "stream": false,
                        "messages": [{"role": "user", "content": "hello"}]
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    engine_task.await.expect("mock engine task");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn non_stream_completions_use_final_only_output_kind() {
    let handshake_address = unique_tcp_endpoint();
    let engine_identity = b"engine-openai-completion-final-only".to_vec();

    let engine_task = tokio::spawn({
        let engine_handshake = handshake_address.clone();
        let engine_identity = engine_identity.clone();
        async move {
            let (mut dealer, mut push) = setup_mock_engine(engine_handshake, engine_identity).await;
            let add = recv_engine_message(&mut dealer).await;
            let request: EngineCoreRequest =
                rmp_serde::from_slice(&add[1]).expect("decode request");
            assert_eq!(
                request
                    .sampling_params
                    .as_ref()
                    .expect("sampling params")
                    .output_kind,
                RequestOutputKind::FinalOnly
            );
            send_outputs(
                &mut push,
                engine_outputs_for_request(&request.request_id, default_stream_output_specs()),
            )
            .await;
        }
    });

    let client = EngineCoreClient::connect(EngineCoreClientConfig {
        handshake_address,
        model_name: "test-model".to_string(),
        local_host: "127.0.0.1".to_string(),
        ready_timeout: Duration::from_secs(2),
        client_index: 0,
    })
    .await
    .expect("connect client");
    let chat = ChatLlm::from_shared_backend(Llm::new(client), Arc::new(FakeChatBackend::new()));
    let app = build_router(Arc::new(AppState::new("Qwen/Qwen1.5-0.5B-Chat", chat)));

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "prompt": "hello",
                        "stream": false
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    engine_task.await.expect("mock engine task");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn completions_happy_path_returns_sse_stream() {
    let (app, engine_task) = test_app_with_engine_handle().await;
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "prompt": "hello",
                        "stream": true,
                        "stream_options": {"include_usage": true}
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok()),
        Some("text/event-stream")
    );

    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    engine_task.await.expect("mock engine task");
    let text = String::from_utf8(body.to_vec()).expect("utf8 body");
    let payloads = sse_data_payloads(&text);
    let usage_index = payloads
        .iter()
        .position(|payload| payload.contains("\"usage\":"))
        .expect("usage chunk");
    let done_index = payloads
        .iter()
        .position(|payload| *payload == "[DONE]")
        .expect("done sentinel");

    assert!(
        payloads
            .iter()
            .any(|payload| payload.contains("\"text\":\"h\"")),
        "{text}"
    );
    assert!(
        payloads
            .iter()
            .any(|payload| payload.contains("\"finish_reason\":\"stop\"")),
        "{text}"
    );
    assert!(usage_index < done_index, "{text}");

    let usage_chunk: serde_json::Value =
        serde_json::from_str(payloads[usage_index]).expect("usage chunk json");
    assert_eq!(usage_chunk["choices"], json!([]));
    assert_eq!(usage_chunk["usage"]["completion_tokens"], 3);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn completions_echo_stream_emits_separate_prompt_chunk() {
    let (app, engine_task) = test_app_with_engine_handle().await;
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "prompt": "hello",
                        "echo": true,
                        "stream": true,
                        "stream_options": {"include_usage": true}
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);

    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    engine_task.await.expect("mock engine task");
    let text = String::from_utf8(body.to_vec()).expect("utf8 body");
    let payloads = sse_data_payloads(&text);
    let hello_index = payloads
        .iter()
        .position(|payload| payload.contains("\"text\":\"hello\""))
        .expect("prompt echo chunk");
    let h_index = payloads
        .iter()
        .position(|payload| payload.contains("\"text\":\"h\""))
        .expect("first generation chunk");

    assert!(hello_index < h_index, "{text}");
    assert!(
        payloads
            .iter()
            .any(|payload| payload.contains("\"text\":\"i\"")),
        "{text}"
    );

    let usage_chunk: serde_json::Value = serde_json::from_str(
        payloads
            .iter()
            .find(|payload| payload.contains("\"usage\":"))
            .expect("usage chunk"),
    )
    .expect("usage chunk json");
    assert_eq!(usage_chunk["usage"]["prompt_tokens"], 5);
    assert_eq!(usage_chunk["usage"]["completion_tokens"], 3);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn chat_harness_streams_text_events() {
    let (chat, engine_task) = test_chat_with_engine_handle().await;
    let mut stream = chat
        .chat(ChatRequest {
            request_id: "chat-harness".to_string(),
            messages: vec![ChatMessage::text(ChatRole::User, "hello")],
            sampling_params: SamplingParams {
                max_tokens: Some(8),
                ..Default::default()
            },
            chat_options: Default::default(),
            tools: Vec::new(),
            tool_choice: ChatToolChoice::None,
            decode_options: Default::default(),
            intermediate: true,
        })
        .await
        .expect("submit chat request");

    let mut saw_text = false;
    let mut saw_done = false;
    while let Some(event) = stream.next().await {
        match event.expect("chat event") {
            ChatEvent::BlockDelta { .. } => saw_text = true,
            ChatEvent::Done { .. } => {
                saw_done = true;
                break;
            }
            ChatEvent::Start
            | ChatEvent::BlockStart { .. }
            | ChatEvent::BlockEnd { .. }
            | ChatEvent::ToolCallStart { .. }
            | ChatEvent::ToolCallArgumentsDelta { .. }
            | ChatEvent::ToolCallEnd { .. } => {}
        }
    }
    engine_task.await.expect("mock engine task");

    assert!(saw_text);
    assert!(saw_done);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn prepared_openai_request_streams_text_events() {
    let (chat, engine_task) = test_chat_with_engine_handle().await;
    let prepared = prepare_chat_request(
        &serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "stream": true,
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .expect("decode request"),
        "Qwen/Qwen1.5-0.5B-Chat",
    )
    .expect("prepare request");

    let mut stream = chat
        .chat(prepared.chat_request)
        .await
        .expect("submit chat request");

    let mut saw_text = false;
    let mut saw_done = false;
    while let Some(event) = stream.next().await {
        match event.expect("chat event") {
            ChatEvent::BlockDelta { .. } => saw_text = true,
            ChatEvent::Done { .. } => {
                saw_done = true;
                break;
            }
            ChatEvent::Start
            | ChatEvent::BlockStart { .. }
            | ChatEvent::BlockEnd { .. }
            | ChatEvent::ToolCallStart { .. }
            | ChatEvent::ToolCallArgumentsDelta { .. }
            | ChatEvent::ToolCallEnd { .. } => {}
        }
    }
    engine_task.await.expect("mock engine task");

    assert!(saw_text);
    assert!(saw_done);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn reasoning_blocks_are_mapped_to_reasoning_content_sse_chunks() {
    let (app, engine_task) = test_app_with_backend_and_stream_output_specs(
        Arc::new(FakeChatBackend::with_model_id("Qwen/Qwen3-0.6B")),
        vec![
            (bytes_to_token_ids(b"<think>"), None),
            (bytes_to_token_ids(b"think "), None),
            (bytes_to_token_ids(b"more</think>"), None),
            (bytes_to_token_ids(b"answer"), Some(FinishReason::Length)),
        ],
    )
    .await;

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "stream": true,
                        "messages": [{"role": "user", "content": "hello"}]
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    engine_task.await.expect("mock engine task");
    let text = String::from_utf8(body.to_vec()).expect("utf8 body");

    assert!(text.contains("\"reasoning_content\":\"think \""), "{text}");
    assert!(text.contains("\"reasoning_content\":\"more\""), "{text}");
    assert!(text.contains("\"content\":\"answer\""), "{text}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn tool_calls_are_mapped_to_tool_call_sse_chunks() {
    let (app, engine_task) = test_app_with_backend_and_stream_output_specs(
        Arc::new(FakeChatBackend::with_model_id("Qwen/Qwen3-0.6B")),
        vec![
            (
                bytes_to_token_ids(b"<tool_call>\n{\"name\":\"get_weather\", "),
                None,
            ),
            (
                bytes_to_token_ids(b"\"arguments\":{\"city\":\"Paris\"}}\n</tool_call>"),
                Some(FinishReason::Stop),
            ),
        ],
    )
    .await;

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "stream": true,
                        "messages": [{"role": "user", "content": "hello"}],
                        "tools": [{
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "description": "Get weather",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"city": {"type": "string"}}
                                }
                            }
                        }]
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    engine_task.await.expect("mock engine task");
    let text = String::from_utf8(body.to_vec()).expect("utf8 body");

    assert!(text.contains("\"tool_calls\":"), "{text}");
    assert!(text.contains("\"name\":\"get_weather\""), "{text}");
    assert!(
        text.contains("\"arguments\":\"{\\\"city\\\":\\\"Paris\\\"}\""),
        "{text}"
    );
    assert!(text.contains("\"finish_reason\":\"tool_calls\""), "{text}");
}
