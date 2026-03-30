// Route tests should use `Service::call` rather than `ServiceExt::oneshot`.
// `oneshot` consumes the router and can drop `AppState` before a streaming
// response body is fully drained, which closes the mock engine connection too
// early and causes flaky `closed unexpectedly` failures.

use std::collections::BTreeSet;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode};
use bytes::Bytes;
use futures::StreamExt as _;
use rmpv::Value;
use serde_json::json;
use serial_test::serial;
use tower::Service as _;
use vllm_chat::{
    ChatBackend, ChatEvent, ChatLlm, ChatMessage, ChatRequest, ChatRole, ChatTextBackend,
    ChatToolChoice, SamplingParams,
};
use vllm_engine_core_client::protocol::{
    EngineCoreFinishReason, EngineCoreOutput, EngineCoreOutputs, EngineCoreRequest, Logprobs,
    MaybeWireLogprobs, PositionLogprobs, StopReason, TokenLogprob, UtilityOutput,
    UtilityResultEnvelope, decode_value,
};
use vllm_engine_core_client::test_utils::{IpcNamespace, spawn_mock_engine_task};
use vllm_engine_core_client::{
    ENGINE_CORE_DEAD_SENTINEL, EngineCoreClient, EngineCoreClientConfig, EngineId,
};
use vllm_llm::Llm;
use vllm_metrics::METRICS;
use vllm_text::TextBackend;
use zeromq::prelude::{SocketRecv, SocketSend};
use zeromq::{DealerSocket, PushSocket, ZmqMessage};

use super::{build_router, build_router_with_dev_mode};
use crate::routes::chat_completions::convert::prepare_chat_request;
use crate::state::AppState;

fn request_output(
    request_id: &str,
    new_token_ids: Vec<u32>,
    finish_reason: Option<EngineCoreFinishReason>,
) -> EngineCoreOutput {
    request_output_with_stop_reason(request_id, new_token_ids, finish_reason, None)
}

fn request_output_with_stop_reason(
    request_id: &str,
    new_token_ids: Vec<u32>,
    finish_reason: Option<EngineCoreFinishReason>,
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

fn request_output_with_logprobs(
    request_id: &str,
    new_token_ids: Vec<u32>,
    finish_reason: Option<EngineCoreFinishReason>,
    stop_reason: Option<StopReason>,
    new_logprobs: Option<Logprobs>,
    new_prompt_logprobs_tensors: Option<Logprobs>,
) -> EngineCoreOutput {
    EngineCoreOutput {
        request_id: request_id.to_string(),
        new_token_ids,
        new_logprobs: new_logprobs.map(MaybeWireLogprobs::Direct),
        new_prompt_logprobs_tensors: new_prompt_logprobs_tensors.map(MaybeWireLogprobs::Direct),
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

fn default_stream_output_specs() -> Vec<(Vec<u32>, Option<EngineCoreFinishReason>)> {
    vec![
        (vec![b'h' as u32], None),
        (vec![b'i' as u32], None),
        (vec![b'!' as u32], Some(EngineCoreFinishReason::Stop)),
    ]
}

fn sse_data_payloads(text: &str) -> Vec<&str> {
    text.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .collect()
}

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

    async fn finish(self) {
        self.await.expect("mock engine task");
    }

    fn abort(&self) {
        if let Some(join_handle) = &self.join_handle {
            join_handle.abort();
        }
    }

    async fn abort_and_join(mut self) {
        if let Some(join_handle) = self.join_handle.take() {
            join_handle.abort();
            let _ = join_handle.await;
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

fn sample_logprobs_for_token(token_id: u32, alternate_token_id: u32) -> Logprobs {
    Logprobs {
        positions: vec![PositionLogprobs {
            entries: vec![
                TokenLogprob {
                    token_id,
                    logprob: -0.1,
                    rank: 1,
                },
                TokenLogprob {
                    token_id: alternate_token_id,
                    logprob: -0.2,
                    rank: 1,
                },
            ],
        }],
    }
}

fn sample_logprobs_for_tokens(token_ids: &[u32]) -> Logprobs {
    Logprobs {
        positions: token_ids
            .iter()
            .map(|&token_id| PositionLogprobs {
                entries: vec![
                    TokenLogprob {
                        token_id,
                        logprob: -0.1,
                        rank: 1,
                    },
                    TokenLogprob {
                        token_id: token_id.saturating_add(1),
                        logprob: -0.2,
                        rank: 2,
                    },
                ],
            })
            .collect(),
    }
}

fn prompt_logprobs_for_hello() -> Logprobs {
    Logprobs {
        positions: vec![
            PositionLogprobs {
                entries: vec![
                    TokenLogprob {
                        token_id: b'e' as u32,
                        logprob: -0.3,
                        rank: 1,
                    },
                    TokenLogprob {
                        token_id: b'a' as u32,
                        logprob: -0.5,
                        rank: 1,
                    },
                ],
            },
            PositionLogprobs {
                entries: vec![
                    TokenLogprob {
                        token_id: b'l' as u32,
                        logprob: -0.4,
                        rank: 1,
                    },
                    TokenLogprob {
                        token_id: b'r' as u32,
                        logprob: -0.6,
                        rank: 1,
                    },
                ],
            },
            PositionLogprobs {
                entries: vec![
                    TokenLogprob {
                        token_id: b'l' as u32,
                        logprob: -0.45,
                        rank: 1,
                    },
                    TokenLogprob {
                        token_id: b'i' as u32,
                        logprob: -0.65,
                        rank: 1,
                    },
                ],
            },
            PositionLogprobs {
                entries: vec![
                    TokenLogprob {
                        token_id: b'o' as u32,
                        logprob: -0.5,
                        rank: 1,
                    },
                    TokenLogprob {
                        token_id: b'u' as u32,
                        logprob: -0.7,
                        rank: 1,
                    },
                ],
            },
        ],
    }
}

fn prompt_logprobs_for_tokens(token_ids: &[u32]) -> Logprobs {
    Logprobs {
        positions: token_ids
            .iter()
            .skip(1)
            .map(|&token_id| PositionLogprobs {
                entries: vec![
                    TokenLogprob {
                        token_id,
                        logprob: -0.3,
                        rank: 1,
                    },
                    TokenLogprob {
                        token_id: token_id.saturating_add(1),
                        logprob: -0.5,
                        rank: 2,
                    },
                ],
            })
            .collect(),
    }
}

fn utility_result_value<T>(value: T) -> UtilityResultEnvelope
where
    T: serde::Serialize,
{
    UtilityResultEnvelope::without_type_info(rmpv::ext::to_value(value).expect("encode result"))
}

fn utility_none_result() -> UtilityResultEnvelope {
    UtilityResultEnvelope::without_type_info(Value::Nil)
}

fn utility_outputs(call_id: i64, result: UtilityResultEnvelope) -> EngineCoreOutputs {
    EngineCoreOutputs {
        utility_output: Some(UtilityOutput {
            call_id,
            failure_message: None,
            result: Some(result),
        }),
        ..Default::default()
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
    engine_id: impl Into<EngineId>,
    output_specs: Vec<(Vec<u32>, Option<EngineCoreFinishReason>)>,
    expected_prompt_token_ids: Option<Vec<u32>>,
    backend: Arc<dyn ChatTextBackend>,
) -> (ChatLlm, MockEngineTask) {
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
                if let Some(expected_prompt_token_ids) = expected_prompt_token_ids {
                    assert_eq!(
                        request.prompt_token_ids.as_deref(),
                        Some(expected_prompt_token_ids.as_slice())
                    );
                }
                send_outputs(
                    push,
                    engine_outputs_for_request(&request.request_id, output_specs),
                )
                .await;
            })
        },
    ));

    let client = EngineCoreClient::connect_with_input_output_addresses(
        EngineCoreClientConfig {
            handshake_address,
            engine_count: 1,
            model_name: "test-model".to_string(),
            local_host: "127.0.0.1".to_string(),
            ready_timeout: Duration::from_secs(2),
            client_index: 0,
            enable_inproc_coordinator: false,
        },
        Some(ipc.input_endpoint()),
        Some(ipc.output_endpoint()),
    )
    .await
    .expect("connect client");

    (
        ChatLlm::from_shared_backend(Llm::new(client), backend),
        engine_task,
    )
}

async fn test_models_with_engine_outputs_and_backend(
    engine_id: impl Into<EngineId>,
    output_specs: Vec<(Vec<u32>, Option<EngineCoreFinishReason>)>,
    backend: Arc<dyn ChatTextBackend>,
) -> (ChatLlm, MockEngineTask) {
    test_models_with_engine_outputs_and_backend_inner(engine_id, output_specs, None, backend).await
}

async fn test_chat_with_engine_outputs(
    engine_id: impl Into<EngineId>,
    output_specs: Vec<(Vec<u32>, Option<EngineCoreFinishReason>)>,
) -> (ChatLlm, MockEngineTask) {
    test_models_with_engine_outputs_and_backend(
        engine_id,
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

async fn test_health_app_with_engine_script<F>(
    script: F,
) -> (axum::Router, Arc<AppState>, MockEngineTask)
where
    F: for<'a> FnOnce(&'a mut PushSocket) -> TestFuture<'a> + Send + 'static,
{
    let ipc = IpcNamespace::new().expect("create ipc namespace");
    let handshake_address = ipc.handshake_endpoint();
    let engine_id = b"engine-openai-health".to_vec();

    let engine_task = MockEngineTask::new(spawn_mock_engine_task(
        handshake_address.clone(),
        engine_id.clone(),
        move |_dealer, push| script(push),
    ));

    let client = EngineCoreClient::connect_with_input_output_addresses(
        EngineCoreClientConfig {
            handshake_address,
            engine_count: 1,
            model_name: "test-model".to_string(),
            local_host: "127.0.0.1".to_string(),
            ready_timeout: Duration::from_secs(2),
            client_index: 0,
            enable_inproc_coordinator: false,
        },
        Some(ipc.input_endpoint()),
        Some(ipc.output_endpoint()),
    )
    .await
    .expect("connect client");

    let chat = ChatLlm::from_shared_backend(Llm::new(client), Arc::new(FakeChatBackend::new()));
    let state = Arc::new(AppState::new("Qwen/Qwen1.5-0.5B-Chat", chat));
    (build_router(state.clone()), state, engine_task)
}

async fn test_admin_app_with_engine_script<F>(script: F) -> (axum::Router, MockEngineTask)
where
    F: for<'a> FnOnce(&'a mut DealerSocket, &'a mut PushSocket) -> TestFuture<'a> + Send + 'static,
{
    let ipc = IpcNamespace::new().expect("create ipc namespace");
    let handshake_address = ipc.handshake_endpoint();
    let engine_id = b"engine-openai-admin".to_vec();

    let engine_task = MockEngineTask::new(spawn_mock_engine_task(
        handshake_address.clone(),
        engine_id.clone(),
        move |dealer, push| script(dealer, push),
    ));

    let client = EngineCoreClient::connect_with_input_output_addresses(
        EngineCoreClientConfig {
            handshake_address,
            engine_count: 1,
            model_name: "test-model".to_string(),
            local_host: "127.0.0.1".to_string(),
            ready_timeout: Duration::from_secs(2),
            client_index: 0,
            enable_inproc_coordinator: false,
        },
        Some(ipc.input_endpoint()),
        Some(ipc.output_endpoint()),
    )
    .await
    .expect("connect client");

    let chat = ChatLlm::from_shared_backend(Llm::new(client), Arc::new(FakeChatBackend::new()));
    (
        build_router_with_dev_mode(
            Arc::new(AppState::new("Qwen/Qwen1.5-0.5B-Chat", chat)),
            true,
        ),
        engine_task,
    )
}

async fn test_app_with_engine_handle() -> (axum::Router, MockEngineTask) {
    test_app_with_stream_output_specs(default_stream_output_specs()).await
}

async fn test_app_with_stream_output_specs(
    output_specs: Vec<(Vec<u32>, Option<EngineCoreFinishReason>)>,
) -> (axum::Router, MockEngineTask) {
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
    output_specs: Vec<(Vec<u32>, Option<EngineCoreFinishReason>)>,
) -> (axum::Router, MockEngineTask) {
    let (chat, engine_task) =
        test_models_with_engine_outputs_and_backend(b"engine-openai", output_specs, backend).await;
    (
        build_router(Arc::new(AppState::new("Qwen/Qwen1.5-0.5B-Chat", chat))),
        engine_task,
    )
}

async fn test_chat_with_engine_handle() -> (ChatLlm, MockEngineTask) {
    test_chat_with_engine_outputs(b"engine-openai-chat", default_stream_output_specs()).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn concurrent_non_stream_completions_are_distributed_across_two_engines() {
    let ipc = IpcNamespace::new().expect("create ipc namespace");
    let handshake_address = ipc.handshake_endpoint();
    let (engine_seen_tx, mut engine_seen_rx) = tokio::sync::mpsc::unbounded_channel();

    let engine_seen_tx_0 = engine_seen_tx.clone();
    let engine_task_0 = MockEngineTask::new(spawn_mock_engine_task(
        handshake_address.clone(),
        b"engine-openai-0".to_vec(),
        move |dealer, push| {
            boxed_test_future(async move {
                let add = recv_engine_message(dealer).await;
                let request: EngineCoreRequest =
                    rmp_serde::from_slice(&add[1]).expect("decode request");
                engine_seen_tx_0
                    .send("engine-openai-0".to_string())
                    .expect("record engine 0 request");
                send_outputs(
                    push,
                    EngineCoreOutputs {
                        engine_index: 0,
                        outputs: vec![
                            request_output(&request.request_id, vec![b'h' as u32], None),
                            request_output(
                                &request.request_id,
                                vec![b'i' as u32],
                                Some(EngineCoreFinishReason::Stop),
                            ),
                        ],
                        finished_requests: Some(BTreeSet::from([request.request_id.clone()])),
                        ..Default::default()
                    },
                )
                .await;
            })
        },
    ));
    let engine_seen_tx_1 = engine_seen_tx.clone();
    let engine_task_1 = MockEngineTask::new(spawn_mock_engine_task(
        handshake_address.clone(),
        b"engine-openai-1".to_vec(),
        move |dealer, push| {
            boxed_test_future(async move {
                let add = recv_engine_message(dealer).await;
                let request: EngineCoreRequest =
                    rmp_serde::from_slice(&add[1]).expect("decode request");
                engine_seen_tx_1
                    .send("engine-openai-1".to_string())
                    .expect("record engine 1 request");
                send_outputs(
                    push,
                    EngineCoreOutputs {
                        engine_index: 1,
                        outputs: vec![
                            request_output(&request.request_id, vec![b'h' as u32], None),
                            request_output(
                                &request.request_id,
                                vec![b'i' as u32],
                                Some(EngineCoreFinishReason::Stop),
                            ),
                        ],
                        finished_requests: Some(BTreeSet::from([request.request_id.clone()])),
                        ..Default::default()
                    },
                )
                .await;
            })
        },
    ));
    drop(engine_seen_tx);

    let client = EngineCoreClient::connect_with_input_output_addresses(
        EngineCoreClientConfig {
            handshake_address,
            engine_count: 2,
            model_name: "test-model".to_string(),
            local_host: "127.0.0.1".to_string(),
            ready_timeout: Duration::from_secs(2),
            client_index: 0,
            enable_inproc_coordinator: false,
        },
        Some(ipc.input_endpoint()),
        Some(ipc.output_endpoint()),
    )
    .await
    .expect("connect client");

    let chat = ChatLlm::from_shared_backend(Llm::new(client), Arc::new(FakeChatBackend::new()));
    let app = build_router(Arc::new(AppState::new("Qwen/Qwen1.5-0.5B-Chat", chat)));

    let request = || {
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
            .expect("build request")
    };

    let (response_1, response_2) =
        tokio::join!(app.clone().call(request()), app.clone().call(request()));
    let response_1 = response_1.expect("call first request");
    let response_2 = response_2.expect("call second request");
    assert_eq!(response_1.status(), StatusCode::OK);
    assert_eq!(response_2.status(), StatusCode::OK);

    let _body_1 = to_bytes(response_1.into_body(), usize::MAX)
        .await
        .expect("read first body");
    let _body_2 = to_bytes(response_2.into_body(), usize::MAX)
        .await
        .expect("read second body");
    engine_task_0.finish().await;
    engine_task_1.finish().await;

    let mut seen = vec![
        engine_seen_rx.recv().await.expect("first routed engine"),
        engine_seen_rx.recv().await.expect("second routed engine"),
    ];
    seen.sort();
    assert_eq!(
        seen,
        vec!["engine-openai-0".to_string(), "engine-openai-1".to_string()]
    );
}

async fn server_load(app: &axum::Router) -> u64 {
    let response = app
        .clone()
        .call(
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
        .call(
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
        let rest = line.strip_prefix(metric)?;

        match labels {
            Some(labels) => {
                let (encoded_labels, value) = rest.split_once("} ")?;
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
    let mut app = test_app().await;
    let response = app
        .call(
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
    let mut app = test_app().await;
    let before = METRICS.render().unwrap();

    let response = app
        .call(
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
    let mut app = test_app().await;
    let response = app
        .call(
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
    let mut app = test_app().await;
    let response = app
        .call(
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
        .call(
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
async fn non_stream_chat_includes_logprobs_and_prompt_logprobs() {
    let ipc = IpcNamespace::new().expect("create ipc namespace");
    let handshake_address = ipc.handshake_endpoint();
    let engine_id = b"engine-openai-chat-logprobs".to_vec();

    let engine_task = MockEngineTask::new(spawn_mock_engine_task(
        handshake_address.clone(),
        engine_id.clone(),
        |dealer, push| {
            boxed_test_future(async move {
                let add = recv_engine_message(dealer).await;
                let request: EngineCoreRequest =
                    rmp_serde::from_slice(&add[1]).expect("decode request");
                let prompt_token_ids = request.prompt_token_ids.clone().expect("prompt token ids");

                send_outputs(
                    push,
                    EngineCoreOutputs {
                        engine_index: 0,
                        outputs: vec![request_output_with_logprobs(
                            &request.request_id,
                            bytes_to_token_ids(b"hi"),
                            Some(EngineCoreFinishReason::Stop),
                            None,
                            Some(sample_logprobs_for_tokens(&bytes_to_token_ids(b"hi"))),
                            Some(prompt_logprobs_for_tokens(&prompt_token_ids)),
                        )],
                        scheduler_stats: None,
                        timestamp: 0.0,
                        utility_output: None,
                        finished_requests: None,
                        wave_complete: None,
                        start_wave: None,
                    },
                )
                .await;
            })
        },
    ));

    let client = EngineCoreClient::connect_with_input_output_addresses(
        EngineCoreClientConfig {
            handshake_address,
            engine_count: 1,
            model_name: "test-model".to_string(),
            local_host: "127.0.0.1".to_string(),
            ready_timeout: Duration::from_secs(2),
            client_index: 0,
            enable_inproc_coordinator: false,
        },
        Some(ipc.input_endpoint()),
        Some(ipc.output_endpoint()),
    )
    .await
    .expect("connect client");
    let chat = ChatLlm::from_shared_backend(Llm::new(client), Arc::new(FakeChatBackend::new()));
    let mut app = build_router(Arc::new(AppState::new("Qwen/Qwen1.5-0.5B-Chat", chat)));

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "stream": false,
                        "logprobs": true,
                        "prompt_logprobs": 1,
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
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");

    assert_eq!(
        json["choices"][0]["logprobs"]["content"][0]["token"],
        json!("h")
    );
    assert_eq!(
        json["choices"][0]["logprobs"]["content"][1]["token"],
        json!("i")
    );
    assert_eq!(json["prompt_logprobs"][0], serde_json::Value::Null);
    assert!(json["prompt_logprobs"][1].is_object());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn happy_path_returns_sse_stream() {
    let (app, engine_task) = test_app_with_engine_handle().await;
    let before = METRICS.render().unwrap();
    let response = app
        .clone()
        .call(
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
    let mut app = test_app().await;
    let before = METRICS.render().unwrap();

    let response = app
        .call(
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
    let mut app = test_app().await;
    let before = METRICS.render().unwrap();

    let response = app
        .call(
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
        .call(
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
    let (app, _state, engine_task) =
        test_health_app_with_engine_script(|_push| boxed_test_future(async move {})).await;

    let (status, body) = health_status(&app).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.is_empty(), "expected empty body, got {:?}", body);

    engine_task.await.expect("mock engine task");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn health_endpoint_returns_503_after_engine_core_dead_sentinel() {
    let (app, state, engine_task) = test_health_app_with_engine_script(|push| {
        boxed_test_future(async move {
            push.send(ZmqMessage::from(ENGINE_CORE_DEAD_SENTINEL.to_vec()))
                .await
                .expect("send sentinel");
        })
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
        .call(
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
        .call(
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
        test_app_with_stream_output_specs(vec![(vec![], Some(EngineCoreFinishReason::Error))])
            .await;
    let response = app
        .clone()
        .call(
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
        .call(
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
        .call(
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
    let mut app = test_app().await;
    let response = app
        .call(
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
        .call(
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
        .call(
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
async fn non_stream_completions_include_logprobs() {
    let ipc = IpcNamespace::new().expect("create ipc namespace");
    let handshake_address = ipc.handshake_endpoint();
    let engine_id = b"engine-openai-completion-logprobs".to_vec();

    let engine_task = MockEngineTask::new(spawn_mock_engine_task(
        handshake_address.clone(),
        engine_id.clone(),
        |dealer, push| {
            boxed_test_future(async move {
                let add = recv_engine_message(dealer).await;
                let request: EngineCoreRequest =
                    rmp_serde::from_slice(&add[1]).expect("decode request");
                send_outputs(
                    push,
                    EngineCoreOutputs {
                        engine_index: 0,
                        outputs: vec![
                            request_output_with_logprobs(
                                &request.request_id,
                                vec![b'h' as u32],
                                None,
                                None,
                                Some(sample_logprobs_for_token(b'h' as u32, b'H' as u32)),
                                None,
                            ),
                            request_output_with_logprobs(
                                &request.request_id,
                                vec![b'i' as u32],
                                Some(EngineCoreFinishReason::Stop),
                                None,
                                Some(sample_logprobs_for_token(b'i' as u32, b'I' as u32)),
                                None,
                            ),
                        ],
                        scheduler_stats: None,
                        timestamp: 0.0,
                        utility_output: None,
                        finished_requests: Some(BTreeSet::from([request.request_id.clone()])),
                        wave_complete: None,
                        start_wave: None,
                    },
                )
                .await;
            })
        },
    ));

    let client = EngineCoreClient::connect_with_input_output_addresses(
        EngineCoreClientConfig {
            handshake_address,
            engine_count: 1,
            model_name: "test-model".to_string(),
            local_host: "127.0.0.1".to_string(),
            ready_timeout: Duration::from_secs(2),
            client_index: 0,
            enable_inproc_coordinator: false,
        },
        Some(ipc.input_endpoint()),
        Some(ipc.output_endpoint()),
    )
    .await
    .expect("connect client");
    let chat = ChatLlm::from_shared_backend(Llm::new(client), Arc::new(FakeChatBackend::new()));
    let mut app = build_router(Arc::new(AppState::new("Qwen/Qwen1.5-0.5B-Chat", chat)));

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "prompt": "hello",
                        "stream": false,
                        "logprobs": 1
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

    assert_eq!(json["choices"][0]["logprobs"]["tokens"], json!(["h", "i"]));
    assert_eq!(
        json["choices"][0]["logprobs"]["token_logprobs"],
        json!([-0.1, -0.1])
    );
    assert_eq!(json["choices"][0]["logprobs"]["text_offset"], json!([0, 1]));
    assert_eq!(
        json["choices"][0]["logprobs"]["top_logprobs"][0],
        json!({"h": -0.1, "H": -0.2})
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn non_stream_completions_include_prompt_logprobs() {
    let ipc = IpcNamespace::new().expect("create ipc namespace");
    let handshake_address = ipc.handshake_endpoint();
    let engine_id = b"engine-openai-completion-prompt-logprobs".to_vec();

    let engine_task = MockEngineTask::new(spawn_mock_engine_task(
        handshake_address.clone(),
        engine_id.clone(),
        |dealer, push| {
            boxed_test_future(async move {
                let add = recv_engine_message(dealer).await;
                let request: EngineCoreRequest =
                    rmp_serde::from_slice(&add[1]).expect("decode request");
                send_outputs(
                    push,
                    EngineCoreOutputs {
                        engine_index: 0,
                        outputs: vec![request_output_with_logprobs(
                            &request.request_id,
                            vec![b'h' as u32, b'i' as u32, b'!' as u32],
                            Some(EngineCoreFinishReason::Stop),
                            None,
                            Some(Logprobs {
                                positions: vec![
                                    sample_logprobs_for_token(b'h' as u32, b'H' as u32).positions
                                        [0]
                                    .clone(),
                                    sample_logprobs_for_token(b'i' as u32, b'I' as u32).positions
                                        [0]
                                    .clone(),
                                    sample_logprobs_for_token(b'!' as u32, b'?' as u32).positions
                                        [0]
                                    .clone(),
                                ],
                            }),
                            Some(prompt_logprobs_for_hello()),
                        )],
                        scheduler_stats: None,
                        timestamp: 0.0,
                        utility_output: None,
                        finished_requests: None,
                        wave_complete: None,
                        start_wave: None,
                    },
                )
                .await;
            })
        },
    ));

    let client = EngineCoreClient::connect_with_input_output_addresses(
        EngineCoreClientConfig {
            handshake_address,
            engine_count: 1,
            model_name: "test-model".to_string(),
            local_host: "127.0.0.1".to_string(),
            ready_timeout: Duration::from_secs(2),
            client_index: 0,
            enable_inproc_coordinator: false,
        },
        Some(ipc.input_endpoint()),
        Some(ipc.output_endpoint()),
    )
    .await
    .expect("connect client");
    let chat = ChatLlm::from_shared_backend(Llm::new(client), Arc::new(FakeChatBackend::new()));
    let mut app = build_router(Arc::new(AppState::new("Qwen/Qwen1.5-0.5B-Chat", chat)));

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "prompt": "hello",
                        "stream": false,
                        "echo": true,
                        "logprobs": 1
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
    assert_eq!(
        json["choices"][0]["logprobs"]["tokens"],
        json!(["h", "e", "l", "l", "o", "h", "i", "!"])
    );
    assert_eq!(
        json["choices"][0]["logprobs"]["text_offset"],
        json!([0, 1, 2, 3, 4, 5, 6, 7])
    );
    assert_eq!(
        json["choices"][0]["logprobs"]["token_logprobs"],
        json!([null, -0.3, -0.4, -0.45, -0.5, -0.1, -0.1, -0.1])
    );
    assert_eq!(
        json["choices"][0]["prompt_logprobs"][0],
        serde_json::Value::Null
    );
    assert_eq!(
        json["choices"][0]["prompt_logprobs"][1],
        json!({"a": -0.5, "e": -0.3})
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn non_stream_chat_uses_final_only_output_kind() {
    let ipc = IpcNamespace::new().expect("create ipc namespace");
    let handshake_address = ipc.handshake_endpoint();
    let engine_id = b"engine-openai-chat-final-only".to_vec();

    let engine_task = MockEngineTask::new(spawn_mock_engine_task(
        handshake_address.clone(),
        engine_id.clone(),
        |dealer, push| {
            boxed_test_future(async move {
                let add = recv_engine_message(dealer).await;
                let request: EngineCoreRequest =
                    rmp_serde::from_slice(&add[1]).expect("decode request");
                send_outputs(
                    push,
                    engine_outputs_for_request(&request.request_id, default_stream_output_specs()),
                )
                .await;
            })
        },
    ));

    let client = EngineCoreClient::connect_with_input_output_addresses(
        EngineCoreClientConfig {
            handshake_address,
            engine_count: 1,
            model_name: "test-model".to_string(),
            local_host: "127.0.0.1".to_string(),
            ready_timeout: Duration::from_secs(2),
            client_index: 0,
            enable_inproc_coordinator: false,
        },
        Some(ipc.input_endpoint()),
        Some(ipc.output_endpoint()),
    )
    .await
    .expect("connect client");
    let chat = ChatLlm::from_shared_backend(Llm::new(client), Arc::new(FakeChatBackend::new()));
    let mut app = build_router(Arc::new(AppState::new("Qwen/Qwen1.5-0.5B-Chat", chat)));

    let response = app
        .call(
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
    let ipc = IpcNamespace::new().expect("create ipc namespace");
    let handshake_address = ipc.handshake_endpoint();
    let engine_id = b"engine-openai-completion-final-only".to_vec();

    let engine_task = MockEngineTask::new(spawn_mock_engine_task(
        handshake_address.clone(),
        engine_id.clone(),
        |dealer, push| {
            boxed_test_future(async move {
                let add = recv_engine_message(dealer).await;
                let request: EngineCoreRequest =
                    rmp_serde::from_slice(&add[1]).expect("decode request");
                send_outputs(
                    push,
                    engine_outputs_for_request(&request.request_id, default_stream_output_specs()),
                )
                .await;
            })
        },
    ));

    let client = EngineCoreClient::connect_with_input_output_addresses(
        EngineCoreClientConfig {
            handshake_address,
            engine_count: 1,
            model_name: "test-model".to_string(),
            local_host: "127.0.0.1".to_string(),
            ready_timeout: Duration::from_secs(2),
            client_index: 0,
            enable_inproc_coordinator: false,
        },
        Some(ipc.input_endpoint()),
        Some(ipc.output_endpoint()),
    )
    .await
    .expect("connect client");
    let chat = ChatLlm::from_shared_backend(Llm::new(client), Arc::new(FakeChatBackend::new()));
    let mut app = build_router(Arc::new(AppState::new("Qwen/Qwen1.5-0.5B-Chat", chat)));

    let response = app
        .call(
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
        .call(
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
        .call(
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
            ChatEvent::Start { .. }
            | ChatEvent::LogprobsDelta { .. }
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
            ChatEvent::Start { .. }
            | ChatEvent::LogprobsDelta { .. }
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
            (
                bytes_to_token_ids(b"answer"),
                Some(EngineCoreFinishReason::Length),
            ),
        ],
    )
    .await;

    let response = app
        .clone()
        .call(
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
                Some(EngineCoreFinishReason::Stop),
            ),
        ],
    )
    .await;

    let response = app
        .clone()
        .call(
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn tool_call_sse_chunks_can_carry_logprobs() {
    let ipc = IpcNamespace::new().expect("create ipc namespace");
    let handshake_address = ipc.handshake_endpoint();
    let engine_id = b"engine-openai-chat-tools-logprobs".to_vec();

    let engine_task = MockEngineTask::new(spawn_mock_engine_task(
        handshake_address.clone(),
        engine_id.clone(),
        |dealer, push| {
            Box::pin(async move {
                let add = recv_engine_message(dealer).await;
                let request: EngineCoreRequest =
                    rmp_serde::from_slice(&add[1]).expect("decode request");

                send_outputs(
                    push,
                    EngineCoreOutputs {
                        engine_index: 0,
                        outputs: vec![request_output_with_logprobs(
                            &request.request_id,
                            bytes_to_token_ids(b"<tool_call>\n{\"name\":\"get_weather\", "),
                            None,
                            None,
                            Some(sample_logprobs_for_tokens(&bytes_to_token_ids(
                                b"<tool_call>\n{\"name\":\"get_weather\", ",
                            ))),
                            None,
                        )],
                        scheduler_stats: None,
                        timestamp: 0.0,
                        utility_output: None,
                        finished_requests: None,
                        wave_complete: None,
                        start_wave: None,
                    },
                )
                .await;
                send_outputs(
                    push,
                    EngineCoreOutputs {
                        engine_index: 0,
                        outputs: vec![request_output_with_logprobs(
                            &request.request_id,
                            bytes_to_token_ids(
                                b"\"arguments\":{\"city\":\"Paris\"}}\n</tool_call>",
                            ),
                            Some(EngineCoreFinishReason::Stop),
                            None,
                            Some(sample_logprobs_for_tokens(&bytes_to_token_ids(
                                b"\"arguments\":{\"city\":\"Paris\"}}\n</tool_call>",
                            ))),
                            None,
                        )],
                        scheduler_stats: None,
                        timestamp: 0.0,
                        utility_output: None,
                        finished_requests: Some(BTreeSet::from([request.request_id.clone()])),
                        wave_complete: None,
                        start_wave: None,
                    },
                )
                .await;
            })
        },
    ));

    let client = EngineCoreClient::connect_with_input_output_addresses(
        EngineCoreClientConfig {
            handshake_address,
            engine_count: 1,
            model_name: "test-model".to_string(),
            local_host: "127.0.0.1".to_string(),
            ready_timeout: Duration::from_secs(2),
            client_index: 0,
            enable_inproc_coordinator: false,
        },
        Some(ipc.input_endpoint()),
        Some(ipc.output_endpoint()),
    )
    .await
    .expect("connect client");
    let chat = ChatLlm::from_shared_backend(
        Llm::new(client),
        Arc::new(FakeChatBackend::with_model_id("Qwen/Qwen3-0.6B")),
    );
    let app = build_router(Arc::new(AppState::new("Qwen/Qwen1.5-0.5B-Chat", chat)));

    let response = app
        .clone()
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "stream": true,
                        "logprobs": true,
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
    engine_task.finish().await;
    let text = String::from_utf8(body.to_vec()).expect("utf8 body");

    assert!(text.contains("\"tool_calls\":"), "{text}");
    assert!(text.contains("\"logprobs\":{\"content\":"), "{text}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn streaming_chat_prompt_logprobs_are_rejected() {
    let (app, engine_task) = test_app_with_engine_handle().await;
    let response = app
        .clone()
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "stream": true,
                        "prompt_logprobs": 1,
                        "messages": [{"role": "user", "content": "hello"}]
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    engine_task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn reset_prefix_cache_route_sends_expected_utility_call() {
    let (app, engine_task) = test_admin_app_with_engine_script(|dealer, push| {
        boxed_test_future(async move {
            let utility = recv_engine_message(dealer).await;
            assert_eq!(utility[0].as_ref(), &[0x03]);

            let payload = decode_value(&utility[1]).expect("decode utility payload");
            let array = payload.as_array().expect("utility payload array");
            let call_id = array[1].as_i64().expect("call id");

            assert_eq!(array[2], Value::from("reset_prefix_cache"));
            assert_eq!(
                array[3],
                Value::Array(vec![Value::from(true), Value::from(true)])
            );

            send_outputs(push, utility_outputs(call_id, utility_result_value(true))).await;
        })
    })
    .await;

    let response = app
        .clone()
        .call(
            Request::builder()
                .method("POST")
                .uri("/reset_prefix_cache?reset_running_requests=true&reset_external=true")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");

    let status = response.status();
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    assert!(body.is_empty());
    engine_task.await.expect("mock engine task");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn reset_mm_cache_route_sends_expected_utility_call() {
    let (app, engine_task) = test_admin_app_with_engine_script(|dealer, push| {
        boxed_test_future(async move {
            let utility = recv_engine_message(dealer).await;
            assert_eq!(utility[0].as_ref(), &[0x03]);

            let payload = decode_value(&utility[1]).expect("decode utility payload");
            let array = payload.as_array().expect("utility payload array");
            let call_id = array[1].as_i64().expect("call id");

            assert_eq!(array[2], Value::from("reset_mm_cache"));
            assert_eq!(array[3], Value::Array(Vec::new()));

            send_outputs(push, utility_outputs(call_id, utility_none_result())).await;
        })
    })
    .await;

    let response = app
        .clone()
        .call(
            Request::builder()
                .method("POST")
                .uri("/reset_mm_cache")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");

    let status = response.status();
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    assert!(body.is_empty());
    engine_task.await.expect("mock engine task");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn reset_encoder_cache_route_sends_expected_utility_call() {
    let (app, engine_task) = test_admin_app_with_engine_script(|dealer, push| {
        boxed_test_future(async move {
            let utility = recv_engine_message(dealer).await;
            assert_eq!(utility[0].as_ref(), &[0x03]);

            let payload = decode_value(&utility[1]).expect("decode utility payload");
            let array = payload.as_array().expect("utility payload array");
            let call_id = array[1].as_i64().expect("call id");

            assert_eq!(array[2], Value::from("reset_encoder_cache"));
            assert_eq!(array[3], Value::Array(Vec::new()));

            send_outputs(push, utility_outputs(call_id, utility_none_result())).await;
        })
    })
    .await;

    let response = app
        .clone()
        .call(
            Request::builder()
                .method("POST")
                .uri("/reset_encoder_cache")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");

    let status = response.status();
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    assert!(body.is_empty());
    engine_task.await.expect("mock engine task");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn sleep_route_uses_python_compatible_default_query_values() {
    let (app, engine_task) = test_admin_app_with_engine_script(|dealer, push| {
        boxed_test_future(async move {
            let utility = recv_engine_message(dealer).await;
            assert_eq!(utility[0].as_ref(), &[0x03]);

            let payload = decode_value(&utility[1]).expect("decode utility payload");
            let array = payload.as_array().expect("utility payload array");
            let call_id = array[1].as_i64().expect("call id");

            assert_eq!(array[2], Value::from("sleep"));
            assert_eq!(
                array[3],
                Value::Array(vec![Value::from(1_u64), Value::from("abort")])
            );

            send_outputs(push, utility_outputs(call_id, utility_none_result())).await;
        })
    })
    .await;

    let response = app
        .clone()
        .call(
            Request::builder()
                .method("POST")
                .uri("/sleep")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");

    let status = response.status();
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    assert!(body.is_empty());
    engine_task.await.expect("mock engine task");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn wake_up_route_without_tags_sends_none() {
    let (app, engine_task) = test_admin_app_with_engine_script(|dealer, push| {
        boxed_test_future(async move {
            let utility = recv_engine_message(dealer).await;
            assert_eq!(utility[0].as_ref(), &[0x03]);

            let payload = decode_value(&utility[1]).expect("decode utility payload");
            let array = payload.as_array().expect("utility payload array");
            let call_id = array[1].as_i64().expect("call id");

            assert_eq!(array[2], Value::from("wake_up"));
            assert_eq!(array[3], Value::Array(vec![Value::Nil]));

            send_outputs(push, utility_outputs(call_id, utility_none_result())).await;
        })
    })
    .await;

    let response = app
        .clone()
        .call(
            Request::builder()
                .method("POST")
                .uri("/wake_up")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");

    let status = response.status();
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    assert!(body.is_empty());
    engine_task.await.expect("mock engine task");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn is_sleeping_route_returns_json_payload() {
    let (app, engine_task) = test_admin_app_with_engine_script(|dealer, push| {
        boxed_test_future(async move {
            let utility = recv_engine_message(dealer).await;
            assert_eq!(utility[0].as_ref(), &[0x03]);

            let payload = decode_value(&utility[1]).expect("decode utility payload");
            let array = payload.as_array().expect("utility payload array");
            let call_id = array[1].as_i64().expect("call id");

            assert_eq!(array[2], Value::from("is_sleeping"));
            assert_eq!(array[3], Value::Array(Vec::new()));

            send_outputs(push, utility_outputs(call_id, utility_result_value(true))).await;
        })
    })
    .await;

    let response = app
        .clone()
        .call(
            Request::builder()
                .method("GET")
                .uri("/is_sleeping")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    engine_task.await.expect("mock engine task");

    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&body).expect("decode json"),
        json!({ "is_sleeping": true })
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn admin_routes_are_hidden_when_dev_mode_is_disabled() {
    let (chat, engine_task) = test_chat_with_engine_handle().await;
    let app = build_router_with_dev_mode(
        Arc::new(AppState::new("Qwen/Qwen1.5-0.5B-Chat", chat)),
        false,
    );

    for (method, uri) in [
        ("GET", "/is_sleeping"),
        ("POST", "/sleep"),
        ("POST", "/wake_up"),
        ("POST", "/reset_prefix_cache"),
        ("POST", "/reset_mm_cache"),
        ("POST", "/reset_encoder_cache"),
    ] {
        let response = app
            .clone()
            .call(
                Request::builder()
                    .method(method)
                    .uri(uri)
                    .body(Body::empty())
                    .expect("build request"),
            )
            .await
            .expect("call app");

        assert_eq!(response.status(), StatusCode::NOT_FOUND, "{method} {uri}");
    }

    engine_task.abort_and_join().await;
}
