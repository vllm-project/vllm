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
use std::{fmt, fs};

use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode};
use bytes::Bytes;
use rmpv::Value;
use serde_json::json;
use serial_test::serial;
use tower::{Service as _, ServiceExt as _};
use vllm_chat::{
    ChatBackend, ChatContent, ChatContentPart, ChatLlm, ChatMessage, ChatRenderer, ChatRequest,
    ChatTextBackend, DefaultChatOutputProcessor, DynChatOutputProcessor, DynChatRenderer,
    NewChatOutputProcessorOptions,
};
use vllm_engine_core_client::mock_engine::default_ready_response;
use vllm_engine_core_client::protocol::decode_value;
use vllm_engine_core_client::protocol::logprobs::{
    Logprobs, MaybeWireLogprobs, PositionLogprobs, TokenLogprob,
};
use vllm_engine_core_client::protocol::output::{
    EngineCoreFinishReason, EngineCoreOutput, EngineCoreOutputs, RequestBatchOutputs, StopReason,
    UtilityCallOutput,
};
use vllm_engine_core_client::protocol::request::EngineCoreRequest;
use vllm_engine_core_client::protocol::utility::{UtilityOutput, UtilityResultEnvelope};
use vllm_engine_core_client::test_utils::{
    IpcNamespace, spawn_mock_engine_task, spawn_mock_engine_task_with_ready,
};
use vllm_engine_core_client::{
    ENGINE_CORE_DEAD_SENTINEL, EngineCoreClient, EngineCoreClientConfig, EngineId,
};
use vllm_llm::Llm;
use vllm_metrics::METRICS;
use vllm_text::tokenizer::DynTokenizer;
use vllm_text::{Prompt, TextBackend};
use vllm_tokenizer::test_utils::TestTokenizer;
use zeromq::prelude::{SocketRecv, SocketSend};
use zeromq::{DealerSocket, PushSocket, ZmqMessage};

use super::{build_router, build_router_with_dev_mode, build_router_with_dev_mode_and_lora};
use crate::config::{ApiServerOptions, CorsConfig};
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
        prefill_stats: None,
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
        prefill_stats: None,
        routed_experts: None,
        num_nans_in_logits: 0,
    }
}

fn request_output_with_logprobs_and_kv(
    request_id: &str,
    new_token_ids: Vec<u32>,
    finish_reason: Option<EngineCoreFinishReason>,
    stop_reason: Option<StopReason>,
    new_logprobs: Option<Logprobs>,
    new_prompt_logprobs_tensors: Option<Logprobs>,
    kv_transfer_params: Option<serde_json::Value>,
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
        kv_transfer_params,
        trace_headers: None,
        prefill_stats: None,
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

fn assert_adapter_a_lora_request(request: &EngineCoreRequest) {
    let lora = request.lora_request.as_ref().expect("lora request");
    assert_eq!(lora.lora_name, "adapter-a");
    assert_eq!(lora.lora_int_id, 1);
    assert_eq!(lora.lora_path, "org/adapter-a");
}

fn sse_data_payloads(text: &str) -> Vec<&str> {
    text.lines().filter_map(|line| line.strip_prefix("data: ")).collect()
}

fn sse_json_payloads(text: &str) -> Vec<serde_json::Value> {
    sse_data_payloads(text)
        .into_iter()
        .filter(|payload| *payload != "[DONE]")
        .map(|payload| serde_json::from_str(payload).expect("sse json payload"))
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
    RequestBatchOutputs {
        outputs: output_specs
            .into_iter()
            .map(|(token_ids, finish_reason)| request_output(request_id, token_ids, finish_reason))
            .collect(),
        ..Default::default()
    }
    .into()
}

fn test_llm(client: EngineCoreClient) -> Llm {
    Llm::new(client).with_request_id_randomization(false)
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

fn utility_outputs(call_id: u64, result: UtilityResultEnvelope) -> EngineCoreOutputs {
    UtilityCallOutput {
        output: UtilityOutput {
            call_id: call_id.into(),
            failure_message: None,
            result: Some(result),
        },
        ..Default::default()
    }
    .into()
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

#[derive(Clone)]
struct FakeChatBackend {
    model_id: String,
    multimodal_model_info: Option<vllm_chat::multimodal::MultimodalModelInfo>,
}

/// Synthetic BOS id used when `add_special_tokens` is true in tests.
const FAKE_BOS_TOKEN_ID: u32 = 256;
const UNKNOWN_DECODE_TOKEN_ID: u32 = 10_000;

fn fake_chat_tokenizer() -> TestTokenizer {
    TestTokenizer::new()
        .with_bos_token("<bos>", FAKE_BOS_TOKEN_ID)
        .with_regular_token("<image>", 999)
        .with_regular_token("<|image_pad|>", 151655)
        .with_regular_token("<think>", 0xF001)
        .with_regular_token("</think>", 0xF002)
        .with_regular_token("<|START_THINKING|>", 0xF003)
        .with_regular_token("<|END_THINKING|>", 0xF004)
        .with_regular_token("◁think▷", 0xF005)
        .with_regular_token("◁/think▷", 0xF006)
}

impl FakeChatBackend {
    fn new() -> Self {
        Self {
            model_id: "test-model".to_string(),
            multimodal_model_info: None,
        }
    }

    fn with_model_id(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            multimodal_model_info: None,
        }
    }

    fn with_multimodal_model_info(
        multimodal_model_info: vllm_chat::multimodal::MultimodalModelInfo,
    ) -> Self {
        Self {
            model_id: "test-model".to_string(),
            multimodal_model_info: Some(multimodal_model_info),
        }
    }
}

impl fmt::Debug for FakeChatBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FakeChatBackend")
            .field("model_id", &self.model_id)
            .finish_non_exhaustive()
    }
}

impl TextBackend for FakeChatBackend {
    fn tokenizer(&self) -> DynTokenizer {
        Arc::new(fake_chat_tokenizer())
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

impl ChatBackend for FakeChatBackend {
    fn chat_renderer(&self) -> DynChatRenderer {
        Arc::new(self.clone())
    }

    fn multimodal_model_info(&self) -> Option<&vllm_chat::multimodal::MultimodalModelInfo> {
        self.multimodal_model_info.as_ref()
    }

    fn new_chat_output_processor(
        &self,
        request: &mut ChatRequest,
        options: NewChatOutputProcessorOptions<'_>,
    ) -> vllm_chat::Result<DynChatOutputProcessor> {
        Ok(Box::new(DefaultChatOutputProcessor::new(
            request,
            &self.model_id,
            self.tokenizer(),
            options.tool_call_parser,
            options.reasoning_parser,
        )?))
    }
}

impl ChatRenderer for FakeChatBackend {
    fn render(&self, request: &ChatRequest) -> vllm_chat::Result<vllm_chat::RenderedPrompt> {
        let placeholder = self
            .multimodal_model_info
            .as_ref()
            .map(|info| info.placeholder_token())
            .unwrap_or("<image>");
        let mut prompt = String::new();
        for message in &request.messages {
            prompt.push_str(message.role().as_str());
            prompt.push_str(": ");
            prompt.push_str(&render_fake_message_content(message, placeholder)?);
            prompt.push('\n');
        }
        if request.chat_options.add_generation_prompt() {
            prompt.push_str("assistant:");
        }
        Ok(vllm_chat::RenderedPrompt {
            prompt: Prompt::Text(prompt),
            effective_template_kwargs: Default::default(),
        })
    }
}

fn render_fake_message_content(
    message: &ChatMessage,
    placeholder: &str,
) -> vllm_chat::Result<String> {
    match message {
        ChatMessage::System { content }
        | ChatMessage::Developer { content, .. }
        | ChatMessage::User { content }
        | ChatMessage::ToolResponse { content, .. } => render_fake_content(content, placeholder),
        ChatMessage::Assistant { .. } => message.text_content(),
    }
}

fn render_fake_content(content: &ChatContent, placeholder: &str) -> vllm_chat::Result<String> {
    Ok(match content {
        ChatContent::Text(text) => text.clone(),
        ChatContent::Parts(parts) => {
            let mut out = String::new();
            for part in parts {
                match part {
                    ChatContentPart::Text { text } => out.push_str(text),
                    ChatContentPart::ImageUrl { .. } => out.push_str(placeholder),
                }
            }
            out
        }
    })
}

fn qwen_multimodal_model_info() -> vllm_chat::multimodal::MultimodalModelInfo {
    let config_path = std::env::temp_dir().join(format!(
        "vllm-server-qwen-config-{}.json",
        uuid::Uuid::new_v4()
    ));
    fs::write(
        &config_path,
        r#"{"model_type":"qwen2_vl","image_token_id":151655}"#,
    )
    .expect("write qwen test config");
    let info = vllm_chat::multimodal::MultimodalModelInfo::from_paths(
        "qwen2-vl-test".to_string(),
        Some("qwen2_vl".to_string()),
        Some(&config_path),
        None,
        Arc::new(fake_chat_tokenizer()),
    )
    .expect("load multimodal info")
    .expect("qwen multimodal info is registered");
    let _ = fs::remove_file(config_path);
    info
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

    (
        ChatLlm::from_shared_backend(test_llm(client), backend),
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
    test_app_with_dev_mode(false).await
}

async fn test_app_with_dev_mode(dev_mode_enabled: bool) -> axum::Router {
    let (chat, _engine_task) = test_models_with_engine_outputs_and_backend(
        b"engine-openai",
        default_stream_output_specs(),
        Arc::new(FakeChatBackend::new()),
    )
    .await;
    build_router_with_dev_mode(
        Arc::new(AppState::new(
            vec!["Qwen/Qwen1.5-0.5B-Chat".to_string()],
            chat,
        )),
        dev_mode_enabled,
    )
}

/// Build a dev-mode router backed by a mock engine using a custom ready
/// response, returning the router and the engine task handle so the engine
/// stays alive for the duration of the test.
async fn test_dev_mode_app_with_ready(
    ready_response: vllm_engine_core_client::protocol::handshake::EngineCoreReadyResponse,
) -> (axum::Router, MockEngineTask) {
    let ipc = IpcNamespace::new().expect("create ipc namespace");
    let handshake_address = ipc.handshake_endpoint();
    let engine_id = b"engine-world-size".to_vec();

    let engine_task = MockEngineTask::new(spawn_mock_engine_task_with_ready(
        handshake_address.clone(),
        engine_id.clone(),
        ready_response,
        |_dealer, _push| boxed_test_future(async {}),
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

    let chat = ChatLlm::from_shared_backend(test_llm(client), Arc::new(FakeChatBackend::new()));
    let app = build_router_with_dev_mode(
        Arc::new(AppState::new(
            vec!["Qwen/Qwen1.5-0.5B-Chat".to_string()],
            chat,
        )),
        true,
    );
    (app, engine_task)
}

async fn test_app_with_request_id_headers() -> (axum::Router, MockEngineTask) {
    let (chat, engine_task) = test_models_with_engine_outputs_and_backend(
        b"engine-openai-request-id",
        default_stream_output_specs(),
        Arc::new(FakeChatBackend::new()),
    )
    .await;
    let app = build_router(Arc::new(
        AppState::new(vec!["Qwen/Qwen1.5-0.5B-Chat".to_string()], chat).with_api_server_options(
            ApiServerOptions {
                enable_request_id_headers: true,
                ..Default::default()
            },
        ),
    ));
    (app, engine_task)
}

async fn test_app_with_api_keys(api_keys: Vec<String>) -> (axum::Router, MockEngineTask) {
    let (chat, engine_task) = test_models_with_engine_outputs_and_backend(
        b"engine-openai-api-key",
        default_stream_output_specs(),
        Arc::new(FakeChatBackend::new()),
    )
    .await;
    let app = build_router(Arc::new(
        AppState::new(vec!["Qwen/Qwen1.5-0.5B-Chat".to_string()], chat).with_api_keys(api_keys),
    ));
    (app, engine_task)
}

async fn test_app_with_cors_and_keys(
    cors: CorsConfig,
    api_keys: Vec<String>,
) -> (axum::Router, MockEngineTask) {
    let (chat, engine_task) = test_models_with_engine_outputs_and_backend(
        b"engine-openai-cors",
        default_stream_output_specs(),
        Arc::new(FakeChatBackend::new()),
    )
    .await;
    let app = build_router(Arc::new(
        AppState::new(vec!["Qwen/Qwen1.5-0.5B-Chat".to_string()], chat)
            .with_cors(cors)
            .with_api_keys(api_keys),
    ));
    (app, engine_task)
}

async fn test_app_with_cors(cors: CorsConfig) -> (axum::Router, MockEngineTask) {
    test_app_with_cors_and_keys(cors, vec![]).await
}

fn header_value<'a>(response: &'a axum::response::Response, name: &str) -> Option<&'a str> {
    response
        .headers()
        .get(name)
        .map(|value| value.to_str().expect("header is valid utf-8"))
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

    let chat = ChatLlm::from_shared_backend(test_llm(client), Arc::new(FakeChatBackend::new()));
    let state = Arc::new(AppState::new(
        vec!["Qwen/Qwen1.5-0.5B-Chat".to_string()],
        chat,
    ));
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

    let chat = ChatLlm::from_shared_backend(test_llm(client), Arc::new(FakeChatBackend::new()));
    (
        build_router_with_dev_mode_and_lora(
            Arc::new(AppState::new(
                vec!["Qwen/Qwen1.5-0.5B-Chat".to_string()],
                chat,
            )),
            true,
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
        build_router(Arc::new(AppState::new(
            vec!["Qwen/Qwen1.5-0.5B-Chat".to_string()],
            chat,
        ))),
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
        build_router(Arc::new(AppState::new(
            vec!["Qwen/Qwen1.5-0.5B-Chat".to_string()],
            chat,
        ))),
        engine_task,
    )
}

async fn test_app_with_backend_and_engine_request_check<F>(
    backend: Arc<dyn ChatTextBackend>,
    check_request: F,
) -> (axum::Router, MockEngineTask)
where
    F: FnOnce(&EngineCoreRequest) + Send + 'static,
{
    let ipc = IpcNamespace::new().expect("create ipc namespace");
    let handshake_address = ipc.handshake_endpoint();
    let engine_id = b"engine-openai-check-request".to_vec();

    let engine_task = MockEngineTask::new(spawn_mock_engine_task(
        handshake_address.clone(),
        engine_id.clone(),
        move |dealer, push| {
            boxed_test_future(async move {
                let add = recv_engine_message(dealer).await;
                let request: EngineCoreRequest =
                    rmp_serde::from_slice(&add[1]).expect("decode request");
                check_request(&request);
                send_outputs(
                    push,
                    engine_outputs_for_request(&request.request_id, default_stream_output_specs()),
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

    let chat = ChatLlm::from_shared_backend(test_llm(client), backend);
    (
        build_router(Arc::new(AppState::new(
            vec!["Qwen/Qwen1.5-0.5B-Chat".to_string()],
            chat,
        ))),
        engine_task,
    )
}

async fn test_chat_with_engine_handle() -> (ChatLlm, MockEngineTask) {
    test_chat_with_engine_outputs(b"engine-openai-chat", default_stream_output_specs()).await
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
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
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
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    (status, body)
}

async fn health_response(app: &axum::Router, request_id: Option<&str>) -> axum::response::Response {
    let mut builder = Request::builder().method("GET").uri("/health");
    if let Some(request_id) = request_id {
        builder = builder.header("X-Request-Id", request_id);
    }

    app.clone()
        .call(builder.body(Body::empty()).expect("build request"))
        .await
        .expect("call app")
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
                if expected_parts.into_iter().all(|part| encoded_labels.contains(part)) {
                    value.parse::<f64>().ok()
                } else {
                    None
                }
            }
            None => rest.strip_prefix(' ').and_then(|value| value.parse::<f64>().ok()),
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
        .call(Request::builder().uri("/v1/models").body(Body::empty()).expect("build request"))
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");
    assert_eq!(json["data"][0]["id"], "Qwen/Qwen1.5-0.5B-Chat");
    // No model path configured: `root` falls back to the served name.
    assert_eq!(json["data"][0]["root"], "Qwen/Qwen1.5-0.5B-Chat");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn list_models_base_card_includes_metadata() {
    let (chat, _engine_task) = test_models_with_engine_outputs_and_backend(
        b"engine-openai-models-meta",
        default_stream_output_specs(),
        Arc::new(FakeChatBackend::new()),
    )
    .await;
    // `id` is the served alias; `root` is the underlying model path.
    let mut app = build_router(Arc::new(
        AppState::new(vec!["public-alias".to_string()], chat)
            .with_model_path("org/backend-model".to_string()),
    ));

    let response = app
        .call(Request::builder().uri("/v1/models").body(Body::empty()).expect("build request"))
        .await
        .expect("call app");
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");

    let card = json["data"][0].as_object().expect("card object");
    assert_eq!(card["id"], "public-alias");
    assert_eq!(card["owned_by"], "vllm-frontend-rs");
    assert_eq!(card["root"], "org/backend-model");
    assert!(card["max_model_len"].as_u64().expect("max_model_len") > 0);
    assert!(card["created"].as_i64().expect("created") > 0);
    // `parent` must be emitted as null, not omitted.
    assert!(card.contains_key("parent") && card["parent"].is_null());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn list_models_lists_loras_in_load_order() {
    // Load out of lexicographic order; the list must preserve load order, not sort.
    let (mut app, _engine_task) = test_admin_app_with_engine_script(|dealer, push| {
        boxed_test_future(async move {
            for _ in 0..2 {
                let utility = recv_engine_message(dealer).await;
                let payload = decode_value(&utility[1]).expect("decode utility payload");
                let call_id =
                    payload.as_array().expect("utility array")[1].as_u64().expect("call id");
                send_outputs(push, utility_outputs(call_id, utility_result_value(true))).await;
            }
        })
    })
    .await;

    for name in ["zebra", "alpha"] {
        let path = format!("org/{name}");
        let response = app
            .call(
                Request::builder()
                    .method("POST")
                    .uri("/v1/load_lora_adapter")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({ "lora_name": name, "lora_path": path }).to_string(),
                    ))
                    .expect("build request"),
            )
            .await
            .expect("call app");
        assert_eq!(response.status(), StatusCode::OK);
    }

    let models = app
        .call(Request::builder().uri("/v1/models").body(Body::empty()).expect("build request"))
        .await
        .expect("call app");
    let body = to_bytes(models.into_body(), usize::MAX).await.expect("read body");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");

    assert_eq!(json["data"][0]["id"], "Qwen/Qwen1.5-0.5B-Chat");
    assert_eq!(json["data"][1]["id"], "zebra");
    assert_eq!(json["data"][2]["id"], "alpha");
    // `max_model_len` must be emitted as null on LoRA cards, not omitted.
    let lora_card = json["data"][1].as_object().expect("lora card object");
    assert_eq!(lora_card["root"], "org/zebra");
    assert_eq!(lora_card["parent"], "Qwen/Qwen1.5-0.5B-Chat");
    assert!(lora_card.contains_key("max_model_len") && lora_card["max_model_len"].is_null());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn request_id_header_is_absent_by_default() {
    let app = test_app().await;
    let response = health_response(&app, None).await;

    assert_eq!(response.status(), StatusCode::OK);
    assert!(!response.headers().contains_key("x-request-id"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn request_id_header_generates_uuid_hex_when_enabled() {
    let (app, _engine_task) = test_app_with_request_id_headers().await;
    let response = health_response(&app, None).await;

    assert_eq!(response.status(), StatusCode::OK);
    let request_id = response
        .headers()
        .get("x-request-id")
        .expect("x-request-id header")
        .to_str()
        .expect("header is ascii");
    assert_eq!(request_id.len(), 32);
    assert!(request_id.chars().all(|ch| ch.is_ascii_hexdigit()));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn request_id_header_echoes_incoming_header_when_enabled() {
    let (app, _engine_task) = test_app_with_request_id_headers().await;
    let response = health_response(&app, Some("req-123")).await;

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers().get("x-request-id").unwrap(), "req-123");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn api_key_auth_rejects_missing_token_on_guarded_route() {
    let (mut app, _engine_task) = test_app_with_api_keys(vec!["secret".to_string()]).await;
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

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("json body");
    assert_eq!(json, json!({ "error": "Unauthorized" }));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn api_key_auth_rejects_wrong_token_on_guarded_route() {
    let (mut app, _engine_task) = test_app_with_api_keys(vec!["secret".to_string()]).await;
    let response = app
        .call(
            Request::builder()
                .method("GET")
                .uri("/v1/models")
                .header("authorization", "Bearer wrong")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn api_key_auth_accepts_matching_bearer_token_on_guarded_route() {
    let (mut app, _engine_task) = test_app_with_api_keys(vec!["secret".to_string()]).await;
    let response = app
        .call(
            Request::builder()
                .method("GET")
                .uri("/v1/models")
                .header("authorization", "Bearer secret")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn api_key_auth_allows_options_without_token() {
    let (mut app, _engine_task) = test_app_with_api_keys(vec!["secret".to_string()]).await;
    let response = app
        .call(
            Request::builder()
                .method("OPTIONS")
                .uri("/v1/models")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_ne!(response.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn api_key_auth_allows_unguarded_route_without_token() {
    let (mut app, _engine_task) = test_app_with_api_keys(vec!["secret".to_string()]).await;
    let response = app
        .call(
            Request::builder()
                .method("GET")
                .uri("/health")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn cors_default_simple_request_allows_any_origin() {
    let (mut app, _engine_task) = test_app_with_cors(CorsConfig::default()).await;
    let response = app
        .call(
            Request::builder()
                .method("GET")
                .uri("/v1/models")
                .header("origin", "http://example.com")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        header_value(&response, "access-control-allow-origin"),
        Some("*")
    );
    // Wildcard origins without credentials emit no `Vary` (Starlette parity).
    assert_eq!(header_value(&response, "vary"), None);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn cors_default_preflight_returns_explicit_methods_and_max_age() {
    let (mut app, _engine_task) = test_app_with_cors(CorsConfig::default()).await;
    let response = app
        .call(
            Request::builder()
                .method("OPTIONS")
                .uri("/v1/chat/completions")
                .header("origin", "http://example.com")
                .header("access-control-request-method", "POST")
                .header("access-control-request-headers", "content-type")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    // `*` methods expand to the explicit method list, matching Starlette
    // (never the literal `*`).
    assert_eq!(
        header_value(&response, "access-control-allow-methods"),
        Some("DELETE,GET,HEAD,OPTIONS,PATCH,POST,PUT")
    );
    assert_eq!(
        header_value(&response, "access-control-max-age"),
        Some("600")
    );
    // `*` headers mirror the requested headers.
    assert_eq!(
        header_value(&response, "access-control-allow-headers"),
        Some("content-type")
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn cors_no_origin_request_has_no_cors_headers() {
    let (mut app, _engine_task) = test_app_with_cors(CorsConfig::default()).await;
    let response = app
        .call(
            Request::builder()
                .method("GET")
                .uri("/health")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(header_value(&response, "access-control-allow-origin"), None);
    assert_eq!(header_value(&response, "vary"), None);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn cors_explicit_origin_allowed_reflects_origin_with_vary() {
    let cors = CorsConfig {
        allow_origins: vec!["http://allowed.com".to_string()],
        ..CorsConfig::default()
    };
    let (mut app, _engine_task) = test_app_with_cors(cors).await;
    let response = app
        .call(
            Request::builder()
                .method("GET")
                .uri("/v1/models")
                .header("origin", "http://allowed.com")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(
        header_value(&response, "access-control-allow-origin"),
        Some("http://allowed.com")
    );
    assert_eq!(header_value(&response, "vary"), Some("origin"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn cors_explicit_origin_disallowed_omits_allow_origin() {
    let cors = CorsConfig {
        allow_origins: vec!["http://allowed.com".to_string()],
        ..CorsConfig::default()
    };
    let (mut app, _engine_task) = test_app_with_cors(cors).await;
    let response = app
        .call(
            Request::builder()
                .method("GET")
                .uri("/v1/models")
                .header("origin", "http://evil.com")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(header_value(&response, "access-control-allow-origin"), None);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn cors_wildcard_with_credentials_reflects_origin_without_panic() {
    let cors = CorsConfig {
        allow_credentials: true,
        ..CorsConfig::default()
    };
    let (mut app, _engine_task) = test_app_with_cors(cors).await;
    let response = app
        .call(
            Request::builder()
                .method("GET")
                .uri("/v1/models")
                .header("origin", "http://example.com")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");

    // `*` + credentials reflects the request origin instead of `*` (Starlette
    // parity, and avoids tower-http's wildcard+credentials panic).
    assert_eq!(
        header_value(&response, "access-control-allow-origin"),
        Some("http://example.com")
    );
    assert_eq!(
        header_value(&response, "access-control-allow-credentials"),
        Some("true")
    );
    assert_eq!(header_value(&response, "vary"), Some("origin"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn cors_unauthorized_response_has_no_cors_headers() {
    let (mut app, _engine_task) =
        test_app_with_cors_and_keys(CorsConfig::default(), vec!["secret".to_string()]).await;
    let response = app
        .call(
            Request::builder()
                .method("GET")
                .uri("/v1/models")
                .header("origin", "http://example.com")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");

    // Auth sits outside CORS, so a 401 carries no CORS headers.
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(header_value(&response, "access-control-allow-origin"), None);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn cors_preflight_bypasses_auth_and_returns_cors_headers() {
    let (mut app, _engine_task) =
        test_app_with_cors_and_keys(CorsConfig::default(), vec!["secret".to_string()]).await;
    let response = app
        .call(
            Request::builder()
                .method("OPTIONS")
                .uri("/v1/chat/completions")
                .header("origin", "http://example.com")
                .header("access-control-request-method", "POST")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_ne!(response.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(
        header_value(&response, "access-control-allow-origin"),
        Some("*")
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn cors_explicit_methods_preflight_returns_that_list() {
    let cors = CorsConfig {
        allow_methods: vec!["GET".to_string(), "POST".to_string()],
        ..CorsConfig::default()
    };
    let (mut app, _engine_task) = test_app_with_cors(cors).await;
    let response = app
        .call(
            Request::builder()
                .method("OPTIONS")
                .uri("/v1/chat/completions")
                .header("origin", "http://example.com")
                .header("access-control-request-method", "POST")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");

    // Explicit methods are emitted verbatim, not expanded and not `*`.
    assert_eq!(
        header_value(&response, "access-control-allow-methods"),
        Some("GET,POST")
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn cors_explicit_headers_union_safelisted_headers() {
    let cors = CorsConfig {
        allow_headers: vec!["X-Custom".to_string()],
        ..CorsConfig::default()
    };
    let (mut app, _engine_task) = test_app_with_cors(cors).await;
    let response = app
        .call(
            Request::builder()
                .method("OPTIONS")
                .uri("/v1/chat/completions")
                .header("origin", "http://example.com")
                .header("access-control-request-method", "POST")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");

    // Explicit headers are unioned with the safelisted set, lowercased + sorted.
    assert_eq!(
        header_value(&response, "access-control-allow-headers"),
        Some("accept,accept-language,content-language,content-type,x-custom")
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn version_returns_engine_vllm_version() {
    let mut app = test_app().await;
    let response = app
        .call(Request::builder().uri("/version").body(Body::empty()).expect("build request"))
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");
    assert_eq!(
        json,
        json!({
            "version": "test-vllm-version",
            "rust_frontend_version": env!("CARGO_PKG_VERSION"),
        })
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn server_info_endpoint_is_dev_mode_only() {
    let mut app = test_app().await;
    let response = app
        .call(
            Request::builder()
                .uri("/server_info")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn load_lora_adapter_registers_model_and_forwards_lora_request() {
    let (mut app, engine_task) = test_admin_app_with_engine_script(|dealer, push| {
        boxed_test_future(async move {
            let utility = recv_engine_message(dealer).await;
            assert_eq!(utility[0].as_ref(), &[0x03]);

            let payload = decode_value(&utility[1]).expect("decode utility payload");
            let array = payload.as_array().expect("utility payload array");
            let call_id = array[1].as_u64().expect("call id");
            assert_eq!(array[2], Value::from("add_lora"));

            let args = array[3].as_array().expect("utility args");
            let lora = args[0].as_array().expect("lora request tuple");
            assert_eq!(lora[0], Value::from("adapter-a"));
            assert_eq!(lora[1], Value::from(1));
            assert_eq!(lora[2], Value::from("org/adapter-a"));

            send_outputs(push, utility_outputs(call_id, utility_result_value(true))).await;

            let add = recv_engine_message(dealer).await;
            assert_eq!(add[0].as_ref(), &[0x00]);
            let request: EngineCoreRequest =
                rmp_serde::from_slice(&add[1]).expect("decode engine request");
            assert_adapter_a_lora_request(&request);

            send_outputs(
                push,
                engine_outputs_for_request(&request.request_id, default_stream_output_specs()),
            )
            .await;

            let add = recv_engine_message(dealer).await;
            assert_eq!(add[0].as_ref(), &[0x00]);
            let request: EngineCoreRequest =
                rmp_serde::from_slice(&add[1]).expect("decode engine request");
            assert_adapter_a_lora_request(&request);

            send_outputs(
                push,
                engine_outputs_for_request(&request.request_id, default_stream_output_specs()),
            )
            .await;

            let add = recv_engine_message(dealer).await;
            assert_eq!(add[0].as_ref(), &[0x00]);
            let request: EngineCoreRequest =
                rmp_serde::from_slice(&add[1]).expect("decode engine request");
            assert_eq!(request.prompt_token_ids.as_deref(), Some(&[11, 22][..]));
            assert_adapter_a_lora_request(&request);

            send_outputs(
                push,
                engine_outputs_for_request(&request.request_id, default_stream_output_specs()),
            )
            .await;

            let utility = recv_engine_message(dealer).await;
            assert_eq!(utility[0].as_ref(), &[0x03]);

            let payload = decode_value(&utility[1]).expect("decode utility payload");
            let array = payload.as_array().expect("utility payload array");
            let call_id = array[1].as_u64().expect("call id");
            assert_eq!(array[2], Value::from("remove_lora"));

            let args = array[3].as_array().expect("utility args");
            assert_eq!(args[0], Value::from(1));

            send_outputs(push, utility_outputs(call_id, utility_result_value(true))).await;
        })
    })
    .await;

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/load_lora_adapter")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "lora_name": "adapter-a",
                        "lora_path": "org/adapter-a"
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");
    assert_eq!(response.status(), StatusCode::OK);

    let models = app
        .call(Request::builder().uri("/v1/models").body(Body::empty()).expect("build request"))
        .await
        .expect("call app");
    let body = to_bytes(models.into_body(), usize::MAX).await.expect("read body");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");
    assert_eq!(json["data"][1]["id"], "adapter-a");

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "adapter-a",
                        "prompt": "hello",
                        "max_tokens": 2
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");
    assert_eq!(response.status(), StatusCode::OK);

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "adapter-a",
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

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/inference/v1/generate")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "adapter-a",
                        "token_ids": [11, 22],
                        "stream": false,
                        "sampling_params": {
                            "max_tokens": 2
                        }
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");
    assert_eq!(response.status(), StatusCode::OK);

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/unload_lora_adapter")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "lora_name": "adapter-a",
                        "lora_int_id": 1
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");
    assert_eq!(response.status(), StatusCode::OK);

    let models = app
        .call(Request::builder().uri("/v1/models").body(Body::empty()).expect("build request"))
        .await
        .expect("call app");
    let body = to_bytes(models.into_body(), usize::MAX).await.expect("read body");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");
    assert_eq!(json["data"].as_array().expect("model data").len(), 1);

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "adapter-a",
                        "prompt": "hello",
                        "max_tokens": 2
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");
    assert_eq!(response.status(), StatusCode::NOT_FOUND);

    drop(app);
    engine_task.finish().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn server_info_endpoint_returns_not_found_without_snapshot() {
    let mut app = test_app_with_dev_mode(true).await;
    let response = app
        .call(
            Request::builder()
                .uri("/server_info")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn unload_lora_adapter_rejects_mismatched_lora_int_id() {
    let (mut app, engine_task) = test_admin_app_with_engine_script(|dealer, push| {
        boxed_test_future(async move {
            let utility = recv_engine_message(dealer).await;
            assert_eq!(utility[0].as_ref(), &[0x03]);

            let payload = decode_value(&utility[1]).expect("decode utility payload");
            let array = payload.as_array().expect("utility payload array");
            let call_id = array[1].as_u64().expect("call id");
            assert_eq!(array[2], Value::from("add_lora"));

            send_outputs(push, utility_outputs(call_id, utility_result_value(true))).await;
        })
    })
    .await;

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/load_lora_adapter")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "lora_name": "adapter-a",
                        "lora_path": "org/adapter-a"
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");
    assert_eq!(response.status(), StatusCode::OK);

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/unload_lora_adapter")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "lora_name": "adapter-a",
                        "lora_int_id": 99
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    let models = app
        .call(Request::builder().uri("/v1/models").body(Body::empty()).expect("build request"))
        .await
        .expect("call app");
    let body = to_bytes(models.into_body(), usize::MAX).await.expect("read body");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");
    assert_eq!(json["data"][1]["id"], "adapter-a");

    drop(app);
    engine_task.finish().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn load_lora_adapter_rejects_engine_false_result() {
    let (mut app, engine_task) = test_admin_app_with_engine_script(|dealer, push| {
        boxed_test_future(async move {
            let utility = recv_engine_message(dealer).await;
            assert_eq!(utility[0].as_ref(), &[0x03]);

            let payload = decode_value(&utility[1]).expect("decode utility payload");
            let array = payload.as_array().expect("utility payload array");
            let call_id = array[1].as_u64().expect("call id");
            assert_eq!(array[2], Value::from("add_lora"));

            send_outputs(push, utility_outputs(call_id, utility_result_value(false))).await;
        })
    })
    .await;

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/load_lora_adapter")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "lora_name": "adapter-a",
                        "lora_path": "org/adapter-a"
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);

    let models = app
        .call(Request::builder().uri("/v1/models").body(Body::empty()).expect("build request"))
        .await
        .expect("call app");
    let body = to_bytes(models.into_body(), usize::MAX).await.expect("read body");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");
    assert_eq!(json["data"].as_array().expect("model data").len(), 1);

    drop(app);
    engine_task.finish().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn load_lora_adapter_rejects_base_model_name_collision() {
    let (mut app, engine_task) =
        test_admin_app_with_engine_script(|_, _| boxed_test_future(async move {})).await;

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/load_lora_adapter")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "lora_name": "Qwen/Qwen1.5-0.5B-Chat",
                        "lora_path": "org/adapter-a"
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    drop(app);
    engine_task.finish().await;
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
async fn request_metrics_use_served_model_name_label() {
    let ipc = IpcNamespace::new().expect("create ipc namespace");
    let handshake_address = ipc.handshake_endpoint();
    let engine_id = b"engine-openai-served-model-metrics".to_vec();

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

    let client = EngineCoreClient::connect(
        EngineCoreClientConfig::new_single(handshake_address)
            .with_model_name("served-model-metrics")
            .with_local_input_output_addresses(
                Some(ipc.input_endpoint()),
                Some(ipc.output_endpoint()),
            ),
    )
    .await
    .expect("connect client");
    let chat = ChatLlm::from_shared_backend(test_llm(client), Arc::new(FakeChatBackend::new()));
    let mut app = build_router(Arc::new(AppState::new(
        vec![
            "served-model-metrics".to_string(),
            "served-model-alias".to_string(),
        ],
        chat,
    )));
    let before = METRICS.render().unwrap();

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "served-model-alias",
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
    let _ = to_bytes(response.into_body(), usize::MAX).await.unwrap();

    let after = METRICS.render().unwrap();
    assert_eq!(
        metric_delta(
            &before,
            &after,
            "vllm:request_success_total",
            Some("model_name=\"served-model-metrics\",engine=\"0\",finished_reason=\"stop\""),
        ),
        1.0
    );
    engine_task.await.expect("mock engine task");
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
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");
    assert_eq!(json["error"]["type"], "invalid_request_error");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn chat_completions_empty_allowed_token_ids_returns_openai_error() {
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
                        "messages": [{"role": "user", "content": "hello"}],
                        "allowed_token_ids": []
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");
    assert_eq!(json["error"]["type"], "invalid_request_error");
    assert!(
        json["error"]["message"]
            .as_str()
            .expect("message string")
            .contains("allowed_token_ids should not be empty")
    );
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

    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    engine_task.await.expect("mock engine task");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");

    assert_eq!(json["object"], "chat.completion");
    assert_eq!(json["choices"][0]["message"]["role"], "assistant");
    assert_eq!(json["choices"][0]["message"]["content"], "hi");
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
    assert_eq!(json["usage"]["prompt_tokens"], 22);
    assert_eq!(json["usage"]["completion_tokens"], 3);
    assert_eq!(json["usage"]["total_tokens"], 25);

    // Unset optional fields are serialized as explicit `null` on
    // non-streaming responses...
    let response_object = json.as_object().expect("response object");
    let choice = json["choices"][0].as_object().expect("choice object");
    let message = choice["message"].as_object().expect("message object");
    for (object, key) in [
        (response_object, "system_fingerprint"),
        (response_object, "prompt_token_ids"),
        (response_object, "kv_transfer_params"),
        (choice, "logprobs"),
        (choice, "stop_reason"),
        (choice, "token_ids"),
        (message, "reasoning"),
    ] {
        assert!(
            object.contains_key(key) && object[key].is_null(),
            "expected explicit null `{key}`: {json}"
        );
    }
    // ...except `tool_calls`, which Python pops from the payload when empty.
    assert!(!message.contains_key("tool_calls"), "{json}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn non_stream_chat_image_url_reaches_engine_mm_features() {
    let (app, engine_task) = test_app_with_backend_and_engine_request_check(
        Arc::new(FakeChatBackend::with_multimodal_model_info(
            qwen_multimodal_model_info(),
        )),
        |request| {
            let prompt_token_ids = request.prompt_token_ids.as_ref().expect("prompt token ids");
            assert!(prompt_token_ids.contains(&151655));

            let features = request.mm_features.as_ref().expect("multimodal features");
            assert_eq!(features.len(), 1);
            assert_eq!(features[0].modality, "image");
            assert_eq!(features[0].identifier, "image-1");
            assert!(features[0].mm_position.length > 0);
            assert!(features[0].mm_position.is_embed.is_some());

            let data = features[0].data.as_ref().expect("feature data");
            assert!(data.contains_key("pixel_values"));
            assert!(data.contains_key("image_grid_thw"));
        },
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
                        "stream": false,
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "describe "},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
                                    },
                                    "uuid": "image-1"
                                }
                            ]
                        }]
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    engine_task.await.expect("mock engine task");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");

    assert_eq!(json["object"], "chat.completion");
    assert_eq!(json["choices"][0]["message"]["content"], "hi");
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
                    RequestBatchOutputs {
                        outputs: vec![request_output_with_logprobs(
                            &request.request_id,
                            bytes_to_token_ids(b"hi"),
                            Some(EngineCoreFinishReason::Stop),
                            None,
                            Some(sample_logprobs_for_tokens(&bytes_to_token_ids(b"hi"))),
                            Some(prompt_logprobs_for_tokens(&prompt_token_ids)),
                        )],
                        ..Default::default()
                    }
                    .into(),
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
    let chat = ChatLlm::from_shared_backend(test_llm(client), Arc::new(FakeChatBackend::new()));
    let mut app = build_router(Arc::new(AppState::new(
        vec!["Qwen/Qwen1.5-0.5B-Chat".to_string()],
        chat,
    )));

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
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
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
        response.headers().get("content-type").and_then(|value| value.to_str().ok()),
        Some("text/event-stream")
    );

    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
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

    let _body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
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
        Arc::new(FakeChatBackend::new()),
        vec![(vec![UNKNOWN_DECODE_TOKEN_ID], None)],
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

    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    engine_task.await.expect("mock engine task");
    let text = String::from_utf8(body.to_vec()).expect("utf8 body");

    assert!(text.contains("\"role\":\"assistant\""), "{text}");
    assert!(text.contains("\"type\":\"server_error\""), "{text}");
    assert!(
        text.contains(&format!(
            "test tokenizer cannot decode unknown token id {UNKNOWN_DECODE_TOKEN_ID}"
        )),
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

    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    engine_task.await.expect("mock engine task");
    let text = String::from_utf8(body.to_vec()).expect("utf8 body");

    assert!(text.contains("\"role\":\"assistant\""), "{text}");
    assert!(text.contains("\"type\":\"server_error\""), "{text}");
    assert!(text.contains("Internal server error"), "{text}");
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

    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
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
    let done_index =
        payloads.iter().position(|payload| *payload == "[DONE]").expect("done sentinel");

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
async fn stream_continuous_usage_stats_adds_usage_to_chat_chunks() {
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
                        "stream_options": {
                            "include_usage": true,
                            "continuous_usage_stats": true
                        },
                        "messages": [{"role": "user", "content": "hello"}]
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);

    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    engine_task.await.expect("mock engine task");
    let text = String::from_utf8(body.to_vec()).expect("utf8 body");
    let payloads = sse_json_payloads(&text);

    assert!(
        payloads.iter().all(|payload| payload.get("usage").is_some()),
        "{text}"
    );
    assert!(
        payloads.iter().any(|payload| {
            payload["choices"].as_array().is_some_and(|choices| !choices.is_empty())
                && payload["usage"]["completion_tokens"] == json!(1)
        }),
        "{text}"
    );
    let usage_chunk = payloads
        .iter()
        .find(|payload| payload["choices"] == json!([]))
        .expect("final usage chunk");
    assert_eq!(usage_chunk["usage"]["prompt_tokens"], 22);
    assert_eq!(usage_chunk["usage"]["completion_tokens"], 3);
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

    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
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
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");
    assert_eq!(json["error"]["type"], "invalid_request_error");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn completions_empty_allowed_token_ids_returns_openai_error() {
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
                        "allowed_token_ids": []
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");
    assert_eq!(json["error"]["type"], "invalid_request_error");
    assert!(
        json["error"]["message"]
            .as_str()
            .expect("message string")
            .contains("allowed_token_ids should not be empty")
    );
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

    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    engine_task.await.expect("mock engine task");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");

    assert_eq!(json["object"], "text_completion");
    assert_eq!(json["choices"][0]["text"], "hi");
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
    assert_eq!(json["usage"]["completion_tokens"], 3);

    // Unset optional fields are serialized as explicit `null` on
    // non-streaming responses.
    let response_object = json.as_object().expect("response object");
    let choice = json["choices"][0].as_object().expect("choice object");
    for (object, key) in [
        (response_object, "system_fingerprint"),
        (response_object, "kv_transfer_params"),
        (choice, "logprobs"),
        (choice, "stop_reason"),
        (choice, "prompt_logprobs"),
        (choice, "token_ids"),
        (choice, "prompt_token_ids"),
    ] {
        assert!(
            object.contains_key(key) && object[key].is_null(),
            "expected explicit null `{key}`: {json}"
        );
    }
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
                        "stream": false,
                        "add_special_tokens": false
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);

    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    engine_task.await.expect("mock engine task");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");

    assert_eq!(json["choices"][0]["text"], "hellohi");
    assert_eq!(json["usage"]["prompt_tokens"], 5);
    assert_eq!(json["usage"]["completion_tokens"], 3);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn non_stream_completions_echo_decodes_token_id_prompt_text() {
    let prompt_token_ids = bytes_to_token_ids(b"hello");
    let expected_prompt_token_ids = prompt_token_ids.clone();
    let (app, engine_task) = test_app_with_backend_and_engine_request_check(
        Arc::new(FakeChatBackend::new()),
        move |request| {
            assert_eq!(
                request.prompt_token_ids.as_deref(),
                Some(expected_prompt_token_ids.as_slice())
            );
        },
    )
    .await;
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
                        "prompt": prompt_token_ids,
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

    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    engine_task.await.expect("mock engine task");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");

    assert_eq!(json["choices"][0]["text"], "hellohi");
    assert_eq!(json["usage"]["prompt_tokens"], 5);
    assert_eq!(json["usage"]["completion_tokens"], 3);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn non_stream_completions_token_id_echo_return_token_ids_keeps_prompt_ids_separate() {
    let prompt_token_ids = bytes_to_token_ids(b"hello");
    let expected_prompt_token_ids = prompt_token_ids.clone();
    let (app, engine_task) = test_app_with_backend_and_engine_request_check(
        Arc::new(FakeChatBackend::new()),
        move |request| {
            assert_eq!(
                request.prompt_token_ids.as_deref(),
                Some(expected_prompt_token_ids.as_slice())
            );
        },
    )
    .await;
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
                        "prompt": prompt_token_ids,
                        "echo": true,
                        "return_token_ids": true,
                        "stream": false
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);

    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    engine_task.await.expect("mock engine task");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");

    assert_eq!(json["choices"][0]["text"], "hi");
    assert_eq!(
        json["choices"][0]["prompt_token_ids"],
        json!(bytes_to_token_ids(b"hello"))
    );
    assert_eq!(
        json["choices"][0]["token_ids"],
        json!(bytes_to_token_ids(b"hi!"))
    );
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
                    RequestBatchOutputs {
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
                        finished_requests: Some(BTreeSet::from([request.request_id.clone()])),
                        ..Default::default()
                    }
                    .into(),
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
    let chat = ChatLlm::from_shared_backend(test_llm(client), Arc::new(FakeChatBackend::new()));
    let mut app = build_router(Arc::new(AppState::new(
        vec!["Qwen/Qwen1.5-0.5B-Chat".to_string()],
        chat,
    )));

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
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
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
                    RequestBatchOutputs {
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
                        ..Default::default()
                    }
                    .into(),
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
    let chat = ChatLlm::from_shared_backend(test_llm(client), Arc::new(FakeChatBackend::new()));
    let mut app = build_router(Arc::new(AppState::new(
        vec!["Qwen/Qwen1.5-0.5B-Chat".to_string()],
        chat,
    )));

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
                        "logprobs": 1,
                        "add_special_tokens": false
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
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
async fn non_stream_chat_completions_still_succeed() {
    let ipc = IpcNamespace::new().expect("create ipc namespace");
    let handshake_address = ipc.handshake_endpoint();
    let engine_id = b"engine-openai-chat-non-stream".to_vec();

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
    let chat = ChatLlm::from_shared_backend(test_llm(client), Arc::new(FakeChatBackend::new()));
    let mut app = build_router(Arc::new(AppState::new(
        vec!["Qwen/Qwen1.5-0.5B-Chat".to_string()],
        chat,
    )));

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
async fn chat_completions_accepts_request_body_larger_than_axum_default() {
    let (chat, engine_task) = test_chat_with_engine_outputs(
        b"engine-openai-chat-large-body",
        default_stream_output_specs(),
    )
    .await;
    let mut app = build_router(Arc::new(AppState::new(
        vec!["Qwen/Qwen1.5-0.5B-Chat".to_string()],
        chat,
    )));

    let large_template_arg = "a".repeat(2 * 1024 * 1024);
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
                        "messages": [{"role": "user", "content": "hello"}],
                        "chat_template_kwargs": {"large": large_template_arg}
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    let status = response.status();
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    engine_task.await.expect("mock engine task");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn non_stream_completions_still_succeed() {
    let ipc = IpcNamespace::new().expect("create ipc namespace");
    let handshake_address = ipc.handshake_endpoint();
    let engine_id = b"engine-openai-completion-non-stream".to_vec();

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
    let chat = ChatLlm::from_shared_backend(test_llm(client), Arc::new(FakeChatBackend::new()));
    let mut app = build_router(Arc::new(AppState::new(
        vec!["Qwen/Qwen1.5-0.5B-Chat".to_string()],
        chat,
    )));

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
async fn chat_completions_header_request_id_takes_precedence() {
    let ipc = IpcNamespace::new().expect("create ipc namespace");
    let handshake_address = ipc.handshake_endpoint();
    let engine_id = b"engine-chat-request-id-precedence".to_vec();

    let engine_task = MockEngineTask::new(spawn_mock_engine_task(
        handshake_address.clone(),
        engine_id.clone(),
        |dealer, push| {
            boxed_test_future(async move {
                let add = recv_engine_message(dealer).await;
                let request: EngineCoreRequest =
                    rmp_serde::from_slice(&add[1]).expect("decode request");
                assert_eq!(
                    request.external_req_id.as_deref(),
                    Some("chatcmpl-header-req")
                );
                assert!(request.request_id.starts_with("chatcmpl-header-req-"));
                assert_ne!(request.request_id, "chatcmpl-header-req");

                send_outputs(
                    push,
                    engine_outputs_for_request(&request.request_id, default_stream_output_specs()),
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
    let chat = ChatLlm::from_shared_backend(Llm::new(client), Arc::new(FakeChatBackend::new()));
    let mut app = build_router(Arc::new(AppState::new(
        vec!["Qwen/Qwen1.5-0.5B-Chat".to_string()],
        chat,
    )));

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .header("X-Request-Id", "header-req")
                .body(Body::from(
                    json!({
                        "request_id": "body-req",
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "messages": [{"role": "user", "content": "hello"}]
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    engine_task.await.expect("mock engine task");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");

    assert_eq!(json["id"], "chatcmpl-header-req");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn non_stream_raw_generate_returns_token_output_envelope() {
    let ipc = IpcNamespace::new().expect("create ipc namespace");
    let handshake_address = ipc.handshake_endpoint();
    let engine_id = b"engine-raw-generate-non-stream".to_vec();

    let engine_task = MockEngineTask::new(spawn_mock_engine_task(
        handshake_address.clone(),
        engine_id.clone(),
        |dealer, push| {
            boxed_test_future(async move {
                let add = recv_engine_message(dealer).await;
                let request: EngineCoreRequest =
                    rmp_serde::from_slice(&add[1]).expect("decode request");
                assert_eq!(request.prompt_token_ids.as_deref(), Some(&[11, 22][..]));
                assert_eq!(request.external_req_id.as_deref(), Some("raw-req"));
                assert!(request.request_id.starts_with("raw-req-"));
                assert_ne!(request.request_id, "raw-req");

                send_outputs(
                    push,
                    RequestBatchOutputs {
                        outputs: vec![
                            request_output_with_logprobs(
                                &request.request_id,
                                vec![33],
                                None,
                                None,
                                Some(sample_logprobs_for_token(33, 34)),
                                Some(prompt_logprobs_for_tokens(&[11, 22])),
                            ),
                            request_output_with_logprobs_and_kv(
                                &request.request_id,
                                vec![44],
                                Some(EngineCoreFinishReason::Stop),
                                None,
                                Some(sample_logprobs_for_token(44, 45)),
                                None,
                                Some(json!({"connector": "x"})),
                            ),
                        ],
                        ..Default::default()
                    }
                    .into(),
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
    let chat = ChatLlm::from_shared_backend(Llm::new(client), Arc::new(FakeChatBackend::new()));
    let mut app = build_router(Arc::new(AppState::new(
        vec!["Qwen/Qwen1.5-0.5B-Chat".to_string()],
        chat,
    )));

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/inference/v1/generate")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "request_id": "raw-req",
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "token_ids": [11, 22],
                        "stream": false,
                        "sampling_params": {
                            "max_tokens": 2,
                            "logprobs": 1,
                            "prompt_logprobs": 1
                        }
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    engine_task.await.expect("mock engine task");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");

    assert_eq!(json["request_id"], "raw-req");
    assert_eq!(json["choices"][0]["index"], 0);
    assert_eq!(json["choices"][0]["token_ids"], json!([33, 44]));
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
    assert_eq!(
        json["choices"][0]["logprobs"]["content"][0]["token"],
        "token_id:33"
    );
    assert_eq!(
        json["choices"][0]["logprobs"]["content"][1]["top_logprobs"][0]["token"],
        "token_id:44"
    );
    assert_eq!(json["prompt_logprobs"][0], serde_json::Value::Null);
    assert_eq!(
        json["prompt_logprobs"][1]["22"]["decoded_token"],
        "token_id:22"
    );
    assert_eq!(json["kv_transfer_params"], json!({"connector": "x"}));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn stream_raw_generate_returns_sse_chunks_and_usage() {
    let ipc = IpcNamespace::new().expect("create ipc namespace");
    let handshake_address = ipc.handshake_endpoint();
    let engine_id = b"engine-raw-generate-stream".to_vec();

    let engine_task = MockEngineTask::new(spawn_mock_engine_task(
        handshake_address.clone(),
        engine_id.clone(),
        |dealer, push| {
            boxed_test_future(async move {
                let add = recv_engine_message(dealer).await;
                let request: EngineCoreRequest =
                    rmp_serde::from_slice(&add[1]).expect("decode request");
                assert_eq!(request.prompt_token_ids.as_deref(), Some(&[11, 22][..]));
                assert_eq!(request.external_req_id.as_deref(), Some("raw-stream"));

                send_outputs(
                    push,
                    RequestBatchOutputs {
                        outputs: vec![
                            request_output_with_logprobs(
                                &request.request_id,
                                vec![33],
                                None,
                                None,
                                Some(sample_logprobs_for_token(33, 34)),
                                None,
                            ),
                            request_output_with_logprobs(
                                &request.request_id,
                                vec![44],
                                Some(EngineCoreFinishReason::Stop),
                                None,
                                Some(sample_logprobs_for_token(44, 45)),
                                None,
                            ),
                        ],
                        ..Default::default()
                    }
                    .into(),
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
    let chat = ChatLlm::from_shared_backend(Llm::new(client), Arc::new(FakeChatBackend::new()));
    let mut app = build_router(Arc::new(AppState::new(
        vec!["Qwen/Qwen1.5-0.5B-Chat".to_string()],
        chat,
    )));

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/inference/v1/generate")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "request_id": "raw-stream",
                        "token_ids": [11, 22],
                        "stream": true,
                        "stream_options": {
                            "include_usage": true,
                            "continuous_usage_stats": true
                        },
                        "sampling_params": {
                            "max_tokens": 2,
                            "logprobs": 1
                        }
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get("content-type").and_then(|value| value.to_str().ok()),
        Some("text/event-stream")
    );

    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    engine_task.await.expect("mock engine task");
    let text = String::from_utf8(body.to_vec()).expect("utf8 body");
    let payloads = sse_data_payloads(&text);
    assert_eq!(payloads.len(), 4, "{text}");

    let first: serde_json::Value = serde_json::from_str(payloads[0]).expect("first chunk json");
    assert_eq!(first["request_id"], "raw-stream");
    assert_eq!(first["choices"][0]["index"], 0);
    assert_eq!(first["choices"][0]["token_ids"], json!([33]));
    assert_eq!(
        first["choices"][0]["logprobs"]["content"][0]["token"],
        "token_id:33"
    );
    assert_eq!(first["usage"]["prompt_tokens"], 2);
    assert_eq!(first["usage"]["completion_tokens"], 1);

    let second: serde_json::Value = serde_json::from_str(payloads[1]).expect("second chunk json");
    assert_eq!(second["choices"][0]["token_ids"], json!([44]));
    assert_eq!(second["choices"][0]["finish_reason"], "stop");
    assert_eq!(second["usage"]["completion_tokens"], 2);

    let usage: serde_json::Value = serde_json::from_str(payloads[2]).expect("usage chunk json");
    assert_eq!(usage["choices"], json!([]));
    assert_eq!(usage["usage"]["prompt_tokens"], 2);
    assert_eq!(usage["usage"]["completion_tokens"], 2);
    assert_eq!(usage["usage"]["total_tokens"], 4);
    assert_eq!(payloads[3], "[DONE]");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn stream_raw_generate_emits_final_usage_without_continuous_usage() {
    let (mut app, engine_task) = test_app_with_stream_output_specs(vec![
        (vec![33], None),
        (vec![44], Some(EngineCoreFinishReason::Stop)),
    ])
    .await;

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/inference/v1/generate")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "request_id": "raw-stream-final-usage",
                        "token_ids": [11, 22],
                        "stream": true,
                        "stream_options": {
                            "include_usage": true
                        },
                        "sampling_params": {
                            "max_tokens": 2
                        }
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);

    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    engine_task.await.expect("mock engine task");
    let text = String::from_utf8(body.to_vec()).expect("utf8 body");
    let payloads = sse_data_payloads(&text);
    assert_eq!(payloads.len(), 4, "{text}");

    let first: serde_json::Value = serde_json::from_str(payloads[0]).expect("first chunk json");
    assert_eq!(first["choices"][0]["token_ids"], json!([33]));
    assert!(first.get("usage").is_none());

    let second: serde_json::Value = serde_json::from_str(payloads[1]).expect("second chunk json");
    assert_eq!(second["choices"][0]["token_ids"], json!([44]));
    assert_eq!(second["choices"][0]["finish_reason"], "stop");
    assert!(second.get("usage").is_none());

    let usage: serde_json::Value = serde_json::from_str(payloads[2]).expect("usage chunk json");
    assert_eq!(usage["choices"], json!([]));
    assert_eq!(usage["usage"]["prompt_tokens"], 2);
    assert_eq!(usage["usage"]["completion_tokens"], 2);
    assert_eq!(usage["usage"]["total_tokens"], 4);
    assert_eq!(payloads[3], "[DONE]");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn stream_raw_generate_emits_empty_finish_chunk() {
    let (mut app, engine_task) = test_app_with_stream_output_specs(vec![
        (vec![33], None),
        (vec![], Some(EngineCoreFinishReason::Stop)),
    ])
    .await;

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/inference/v1/generate")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "request_id": "raw-stream-empty-finish",
                        "token_ids": [11, 22],
                        "stream": true,
                        "sampling_params": {
                            "max_tokens": 2
                        }
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);

    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    engine_task.await.expect("mock engine task");
    let text = String::from_utf8(body.to_vec()).expect("utf8 body");
    let payloads = sse_data_payloads(&text);
    assert_eq!(payloads.len(), 3, "{text}");

    let first: serde_json::Value = serde_json::from_str(payloads[0]).expect("first chunk json");
    assert_eq!(first["choices"][0]["token_ids"], json!([33]));
    assert!(first["choices"][0].get("finish_reason").is_none());

    let second: serde_json::Value = serde_json::from_str(payloads[1]).expect("second chunk json");
    assert_eq!(second["choices"][0]["token_ids"], json!([]));
    assert_eq!(second["choices"][0]["finish_reason"], "stop");
    assert_eq!(payloads[2], "[DONE]");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn stream_raw_generate_error_finish_returns_sse_error() {
    let (mut app, engine_task) =
        test_app_with_stream_output_specs(vec![(vec![], Some(EngineCoreFinishReason::Error))])
            .await;

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/inference/v1/generate")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "request_id": "raw-stream-error",
                        "token_ids": [11, 22],
                        "stream": true,
                        "stream_options": {
                            "include_usage": true
                        },
                        "sampling_params": {
                            "max_tokens": 2
                        }
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);

    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    engine_task.await.expect("mock engine task");
    let text = String::from_utf8(body.to_vec()).expect("utf8 body");

    assert!(text.contains("\"type\":\"server_error\""), "{text}");
    assert!(text.contains("Internal server error"), "{text}");
    assert!(!text.contains("\"finish_reason\":\"error\""), "{text}");
    assert!(!text.contains("\"usage\":"), "{text}");
    assert!(text.trim_end().ends_with("data: [DONE]"), "{text}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn raw_generate_rejects_empty_token_ids() {
    let mut app = test_app().await;

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/inference/v1/generate")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "Qwen/Qwen1.5-0.5B-Chat",
                        "token_ids": [],
                        "sampling_params": {}
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");
    assert_eq!(json["error"]["param"], "token_ids");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn raw_generate_rejects_streaming_prompt_logprobs() {
    let mut app = test_app().await;

    for prompt_logprobs in [0, 1] {
        let response = app
            .call(
                Request::builder()
                    .method("POST")
                    .uri("/inference/v1/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "model": "Qwen/Qwen1.5-0.5B-Chat",
                            "token_ids": [11, 22],
                            "stream": true,
                            "sampling_params": {
                                "prompt_logprobs": prompt_logprobs
                            }
                        })
                        .to_string(),
                    ))
                    .expect("build request"),
            )
            .await
            .expect("call app");

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
        let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");
        assert_eq!(json["error"]["param"], "sampling_params");
        assert_eq!(
            json["error"]["message"],
            "`prompt_logprobs` are not available when `stream=true`."
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn raw_generate_rejects_wrong_model() {
    let mut app = test_app().await;

    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri("/inference/v1/generate")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "wrong-model",
                        "token_ids": [11, 22],
                        "sampling_params": {}
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
        response.headers().get("content-type").and_then(|value| value.to_str().ok()),
        Some("text/event-stream")
    );

    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    engine_task.await.expect("mock engine task");
    let text = String::from_utf8(body.to_vec()).expect("utf8 body");
    let payloads = sse_data_payloads(&text);
    let usage_index = payloads
        .iter()
        .position(|payload| payload.contains("\"usage\":"))
        .expect("usage chunk");
    let done_index =
        payloads.iter().position(|payload| *payload == "[DONE]").expect("done sentinel");

    assert!(
        payloads.iter().any(|payload| payload.contains("\"text\":\"h\"")),
        "{text}"
    );
    assert!(
        payloads.iter().any(|payload| payload.contains("\"finish_reason\":\"stop\"")),
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
async fn completions_stream_continuous_usage_stats_adds_usage_to_chunks() {
    let (app, engine_task) = test_app_with_stream_output_specs(default_stream_output_specs()).await;
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
                        "stream_options": {
                            "include_usage": true,
                            "continuous_usage_stats": true
                        }
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);

    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    engine_task.await.expect("mock engine task");
    let text = String::from_utf8(body.to_vec()).expect("utf8 body");
    let payloads = sse_json_payloads(&text);

    assert!(
        payloads.iter().all(|payload| payload.get("usage").is_some()),
        "{text}"
    );
    assert!(
        payloads.iter().any(|payload| {
            payload["choices"].as_array().is_some_and(|choices| !choices.is_empty())
                && payload["usage"]["completion_tokens"] == json!(1)
        }),
        "{text}"
    );
    let usage_chunk = payloads
        .iter()
        .find(|payload| payload["choices"] == json!([]))
        .expect("final usage chunk");
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
                        "stream_options": {"include_usage": true},
                        "add_special_tokens": false
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);

    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
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
        payloads.iter().any(|payload| payload.contains("\"text\":\"i\"")),
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
async fn completions_echo_stream_decodes_token_id_prompt_chunk() {
    let prompt_token_ids = bytes_to_token_ids(b"hello");
    let expected_prompt_token_ids = prompt_token_ids.clone();
    let (app, engine_task) = test_app_with_backend_and_engine_request_check(
        Arc::new(FakeChatBackend::new()),
        move |request| {
            assert_eq!(
                request.prompt_token_ids.as_deref(),
                Some(expected_prompt_token_ids.as_slice())
            );
        },
    )
    .await;
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
                        "prompt": prompt_token_ids,
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

    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
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
async fn reasoning_blocks_are_mapped_to_reasoning_sse_chunks() {
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
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    engine_task.await.expect("mock engine task");
    let text = String::from_utf8(body.to_vec()).expect("utf8 body");

    assert!(text.contains("\"reasoning\":\"think \""), "{text}");
    assert!(text.contains("\"reasoning\":\"more\""), "{text}");
    assert!(text.contains("\"content\":\"answer\""), "{text}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn include_reasoning_false_suppresses_reasoning_in_non_stream_chat() {
    let (app, engine_task) = test_app_with_backend_and_stream_output_specs(
        Arc::new(FakeChatBackend::with_model_id("Qwen/Qwen3-0.6B")),
        vec![
            (bytes_to_token_ids(b"<think>think</think>"), None),
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
                        "stream": false,
                        "include_reasoning": false,
                        "messages": [{"role": "user", "content": "hello"}]
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    engine_task.await.expect("mock engine task");
    let text = String::from_utf8(body.to_vec()).expect("utf8 body");
    let json: serde_json::Value = serde_json::from_str(&text).expect("decode json");

    assert_eq!(json["choices"][0]["message"]["content"], "answer");
    // Suppressed fields are serialized as explicit `null` on non-streaming
    // responses.
    assert!(
        json["choices"][0]["message"]["reasoning"].is_null(),
        "{text}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn include_reasoning_false_suppresses_non_stream_output_metadata() {
    let ipc = IpcNamespace::new().expect("create ipc namespace");
    let handshake_address = ipc.handshake_endpoint();
    let engine_id = b"engine-openai-hidden-reasoning-logprobs".to_vec();

    let engine_task = MockEngineTask::new(spawn_mock_engine_task(
        handshake_address.clone(),
        engine_id.clone(),
        |dealer, push| {
            boxed_test_future(async move {
                let add = recv_engine_message(dealer).await;
                let request: EngineCoreRequest =
                    rmp_serde::from_slice(&add[1]).expect("decode request");
                let reasoning_token_ids = bytes_to_token_ids(b"<think>think</think>");
                let answer_token_ids = bytes_to_token_ids(b"answer");

                send_outputs(
                    push,
                    RequestBatchOutputs {
                        outputs: vec![
                            request_output_with_logprobs(
                                &request.request_id,
                                reasoning_token_ids.clone(),
                                None,
                                None,
                                Some(sample_logprobs_for_tokens(&reasoning_token_ids)),
                                None,
                            ),
                            request_output_with_logprobs(
                                &request.request_id,
                                answer_token_ids.clone(),
                                Some(EngineCoreFinishReason::Length),
                                None,
                                Some(sample_logprobs_for_tokens(&answer_token_ids)),
                                None,
                            ),
                        ],
                        ..Default::default()
                    }
                    .into(),
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
        Arc::new(FakeChatBackend::with_model_id("Qwen/Qwen3-0.6B")),
    );
    let mut app = build_router(Arc::new(AppState::new(
        vec!["Qwen/Qwen1.5-0.5B-Chat".to_string()],
        chat,
    )));

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
                        "include_reasoning": false,
                        "logprobs": true,
                        "return_token_ids": true,
                        "messages": [{"role": "user", "content": "hello"}]
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    engine_task.await.expect("mock engine task");
    let text = String::from_utf8(body.to_vec()).expect("utf8 body");
    let json: serde_json::Value = serde_json::from_str(&text).expect("decode json");
    let choice = json["choices"][0].as_object().expect("choice object");

    assert_eq!(json["choices"][0]["message"]["content"], "answer");
    // Suppressed fields are serialized as explicit `null` on non-streaming
    // responses.
    assert!(
        json["choices"][0]["message"]["reasoning"].is_null(),
        "{text}"
    );
    assert!(choice["logprobs"].is_null(), "{text}");
    assert!(choice["token_ids"].is_null(), "{text}");
    assert!(json["prompt_token_ids"].is_array(), "{text}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn tool_calls_are_mapped_to_tool_call_sse_chunks() {
    let (app, engine_task) = test_app_with_backend_and_stream_output_specs(
        Arc::new(FakeChatBackend::with_model_id("Qwen/Qwen3-0.6B")),
        vec![
            (bytes_to_token_ids(b"<think>Need tool.</think>"), None),
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
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
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
                    RequestBatchOutputs {
                        outputs: vec![request_output_with_logprobs(
                            &request.request_id,
                            bytes_to_token_ids(b"<think>Need tool.</think>"),
                            None,
                            None,
                            Some(sample_logprobs_for_tokens(&bytes_to_token_ids(
                                b"<think>Need tool.</think>",
                            ))),
                            None,
                        )],
                        ..Default::default()
                    }
                    .into(),
                )
                .await;
                send_outputs(
                    push,
                    RequestBatchOutputs {
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
                        ..Default::default()
                    }
                    .into(),
                )
                .await;
                send_outputs(
                    push,
                    RequestBatchOutputs {
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
                        finished_requests: Some(BTreeSet::from([request.request_id.clone()])),
                        ..Default::default()
                    }
                    .into(),
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
        Arc::new(FakeChatBackend::with_model_id("Qwen/Qwen3-0.6B")),
    );
    let app = build_router(Arc::new(AppState::new(
        vec!["Qwen/Qwen1.5-0.5B-Chat".to_string()],
        chat,
    )));

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
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
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
            let call_id = array[1].as_u64().expect("call id");

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
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
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
            let call_id = array[1].as_u64().expect("call id");

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
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
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
            let call_id = array[1].as_u64().expect("call id");

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
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    assert!(body.is_empty());
    engine_task.await.expect("mock engine task");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn collective_rpc_route_sends_expected_utility_call_and_returns_results() {
    let (app, engine_task) = test_admin_app_with_engine_script(|dealer, push| {
        boxed_test_future(async move {
            let utility = recv_engine_message(dealer).await;
            assert_eq!(utility[0].as_ref(), &[0x03]);

            let payload = decode_value(&utility[1]).expect("decode utility payload");
            let array = payload.as_array().expect("utility payload array");
            let call_id = array[1].as_u64().expect("call id");

            assert_eq!(array[2], Value::from("collective_rpc"));
            assert_eq!(
                array[3],
                Value::Array(vec![
                    Value::from("echo_args_kwargs"),
                    Value::from(1.5_f64),
                    Value::Array(vec![Value::from("arg1"), Value::from("arg2")]),
                    Value::Map(vec![
                        (Value::from("key1"), Value::from("value1")),
                        (Value::from("key2"), Value::from("value2")),
                    ]),
                ])
            );

            send_outputs(
                push,
                utility_outputs(
                    call_id,
                    UtilityResultEnvelope::without_type_info(Value::Array(vec![Value::Map(vec![
                        (
                            Value::from("args"),
                            Value::Array(vec![Value::from("arg1"), Value::from("arg2")]),
                        ),
                        (
                            Value::from("kwargs"),
                            Value::Map(vec![
                                (Value::from("key1"), Value::from("value1")),
                                (Value::from("key2"), Value::from("value2")),
                            ]),
                        ),
                        (Value::from("total_items"), Value::from(4_u64)),
                    ])])),
                ),
            )
            .await;
        })
    })
    .await;

    let response = app
        .clone()
        .call(
            Request::builder()
                .method("POST")
                .uri("/collective_rpc")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"method":"echo_args_kwargs","args":["arg1","arg2"],"kwargs":{"key1":"value1","key2":"value2"},"timeout":1.5}"#,
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    engine_task.await.expect("mock engine task");

    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&body).expect("decode json"),
        json!({
            "results": [{
                "args": ["arg1", "arg2"],
                "kwargs": {
                    "key1": "value1",
                    "key2": "value2"
                },
                "total_items": 4
            }]
        })
    );
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
            let call_id = array[1].as_u64().expect("call id");

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
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
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
            let call_id = array[1].as_u64().expect("call id");

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
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
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
            let call_id = array[1].as_u64().expect("call id");

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
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    engine_task.await.expect("mock engine task");

    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&body).expect("decode json"),
        json!({ "is_sleeping": true })
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn pause_route_uses_python_compatible_default_query_values() {
    let (app, engine_task) = test_admin_app_with_engine_script(|dealer, push| {
        boxed_test_future(async move {
            let utility = recv_engine_message(dealer).await;
            assert_eq!(utility[0].as_ref(), &[0x03]);

            let payload = decode_value(&utility[1]).expect("decode utility payload");
            let array = payload.as_array().expect("utility payload array");
            let call_id = array[1].as_u64().expect("call id");

            assert_eq!(array[2], Value::from("pause_scheduler"));
            assert_eq!(
                array[3],
                Value::Array(vec![Value::from("abort"), Value::from(true)])
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
                .uri("/pause")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    engine_task.await.expect("mock engine task");

    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&body).expect("decode json"),
        json!({ "status": "paused" })
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn pause_route_rejects_invalid_mode() {
    let (app, engine_task) =
        test_admin_app_with_engine_script(|_dealer, _push| boxed_test_future(async move {})).await;

    let response = app
        .clone()
        .call(
            Request::builder()
                .method("POST")
                .uri("/pause?mode=banana")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");
    assert_eq!(json["error"]["type"], "invalid_request_error");
    assert_eq!(json["error"]["param"], "mode");
    engine_task.abort_and_join().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn resume_route_sends_no_args() {
    let (app, engine_task) = test_admin_app_with_engine_script(|dealer, push| {
        boxed_test_future(async move {
            let utility = recv_engine_message(dealer).await;
            assert_eq!(utility[0].as_ref(), &[0x03]);

            let payload = decode_value(&utility[1]).expect("decode utility payload");
            let array = payload.as_array().expect("utility payload array");
            let call_id = array[1].as_u64().expect("call id");

            assert_eq!(array[2], Value::from("resume_scheduler"));
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
                .uri("/resume")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    engine_task.await.expect("mock engine task");

    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&body).expect("decode json"),
        json!({ "status": "resumed" })
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn is_paused_route_returns_json_payload() {
    let (app, engine_task) = test_admin_app_with_engine_script(|dealer, push| {
        boxed_test_future(async move {
            let utility = recv_engine_message(dealer).await;
            assert_eq!(utility[0].as_ref(), &[0x03]);

            let payload = decode_value(&utility[1]).expect("decode utility payload");
            let array = payload.as_array().expect("utility payload array");
            let call_id = array[1].as_u64().expect("call id");

            assert_eq!(array[2], Value::from("is_scheduler_paused"));
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
                .uri("/is_paused")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    engine_task.await.expect("mock engine task");

    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&body).expect("decode json"),
        json!({ "is_paused": true })
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn abort_requests_route_returns_ok_for_well_formed_body() {
    let (app, engine_task) =
        test_admin_app_with_engine_script(|_dealer, _push| boxed_test_future(async move {})).await;

    let response = app
        .clone()
        .call(
            Request::builder()
                .method("POST")
                .uri("/abort_requests")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"request_ids":["req-1","req-2"]}"#))
                .expect("build request"),
        )
        .await
        .expect("call app");

    let status = response.status();
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    assert!(body.is_empty());
    engine_task.abort_and_join().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn abort_requests_route_rejects_missing_request_ids() {
    let (app, engine_task) =
        test_admin_app_with_engine_script(|_dealer, _push| boxed_test_future(async move {})).await;

    let response = app
        .clone()
        .call(
            Request::builder()
                .method("POST")
                .uri("/abort_requests")
                .header("content-type", "application/json")
                .body(Body::from(r#"{}"#))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");
    assert_eq!(json["error"]["type"], "invalid_request_error");
    assert_eq!(json["error"]["param"], "request_ids");
    engine_task.abort_and_join().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn abort_requests_route_rejects_malformed_json() {
    let (app, engine_task) =
        test_admin_app_with_engine_script(|_dealer, _push| boxed_test_future(async move {})).await;

    let response = app
        .clone()
        .call(
            Request::builder()
                .method("POST")
                .uri("/abort_requests")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"request_ids": "#))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");
    assert_eq!(json["error"]["type"], "invalid_request_error");
    engine_task.abort_and_join().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn abort_requests_route_accepts_empty_id_list() {
    let (app, engine_task) =
        test_admin_app_with_engine_script(|_dealer, _push| boxed_test_future(async move {})).await;

    let response = app
        .clone()
        .call(
            Request::builder()
                .method("POST")
                .uri("/abort_requests")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"request_ids":[]}"#))
                .expect("build request"),
        )
        .await
        .expect("call app");

    let status = response.status();
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    assert!(body.is_empty());
    engine_task.abort_and_join().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn admin_routes_are_hidden_when_dev_mode_is_disabled() {
    let (chat, engine_task) = test_chat_with_engine_handle().await;
    let app = build_router_with_dev_mode(
        Arc::new(AppState::new(
            vec!["Qwen/Qwen1.5-0.5B-Chat".to_string()],
            chat,
        )),
        false,
    );

    for (method, uri) in [
        ("GET", "/is_sleeping"),
        ("POST", "/sleep"),
        ("POST", "/wake_up"),
        ("GET", "/is_paused"),
        ("POST", "/pause"),
        ("POST", "/resume"),
        ("POST", "/collective_rpc"),
        ("POST", "/abort_requests"),
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

// ========================= Stop string tests =========================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn non_stream_completions_stop_string_excluded_from_output() {
    // Engine generates "say world" but stop string "wor" truncates output to "say
    // ".
    let output_specs = vec![
        (bytes_to_token_ids(b"say"), None),
        (
            bytes_to_token_ids(b" world"),
            Some(EngineCoreFinishReason::Length),
        ),
    ];
    let (app, engine_task) = test_app_with_stream_output_specs(output_specs).await;

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
                        "stream": false,
                        "stop": ["wor"]
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);

    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    engine_task.await.expect("mock engine task");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");

    assert_eq!(json["choices"][0]["text"], "say ");
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
    assert_eq!(json["choices"][0]["stop_reason"], "wor");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn non_stream_completions_stop_string_included_in_output() {
    // Same tokens but include_stop_str_in_output=true includes the stop string in
    // the output.
    let output_specs = vec![
        (bytes_to_token_ids(b"say"), None),
        (
            bytes_to_token_ids(b" world"),
            Some(EngineCoreFinishReason::Length),
        ),
    ];
    let (app, engine_task) = test_app_with_stream_output_specs(output_specs).await;

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
                        "stream": false,
                        "stop": ["wor"],
                        "include_stop_str_in_output": true
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);

    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    engine_task.await.expect("mock engine task");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");

    assert_eq!(json["choices"][0]["text"], "say wor");
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
    assert_eq!(json["choices"][0]["stop_reason"], "wor");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn stream_completions_stop_string_excluded_from_output() {
    let output_specs = vec![
        (bytes_to_token_ids(b"say"), None),
        (
            bytes_to_token_ids(b" world"),
            Some(EngineCoreFinishReason::Length),
        ),
    ];
    let (app, engine_task) = test_app_with_stream_output_specs(output_specs).await;

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
                        "stop": ["wor"]
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);

    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    engine_task.await.expect("mock engine task");
    let text = String::from_utf8(body.to_vec()).expect("utf8 body");
    let payloads = sse_data_payloads(&text);

    // Collect all text deltas from the SSE chunks.
    let mut full_text = String::new();
    for payload in &payloads {
        if *payload == "[DONE]" {
            continue;
        }
        let chunk: serde_json::Value = serde_json::from_str(payload).expect("json chunk");
        if let Some(text) = chunk["choices"][0]["text"].as_str() {
            full_text.push_str(text);
        }
    }

    // The concatenated text deltas should equal "say " (stop string excluded).
    assert_eq!(full_text, "say ", "full streamed text: {text}");

    // The final chunk should have finish_reason "stop".
    assert!(
        payloads.iter().any(|p| p.contains("\"finish_reason\":\"stop\"")),
        "{text}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn stream_completions_stop_string_included_in_output() {
    let output_specs = vec![
        (bytes_to_token_ids(b"say"), None),
        (
            bytes_to_token_ids(b" world"),
            Some(EngineCoreFinishReason::Length),
        ),
    ];
    let (app, engine_task) = test_app_with_stream_output_specs(output_specs).await;

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
                        "stop": ["wor"],
                        "include_stop_str_in_output": true
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);

    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    engine_task.await.expect("mock engine task");
    let text = String::from_utf8(body.to_vec()).expect("utf8 body");
    let payloads = sse_data_payloads(&text);

    let mut full_text = String::new();
    for payload in &payloads {
        if *payload == "[DONE]" {
            continue;
        }
        let chunk: serde_json::Value = serde_json::from_str(payload).expect("json chunk");
        if let Some(text) = chunk["choices"][0]["text"].as_str() {
            full_text.push_str(text);
        }
    }

    // With include_stop_str_in_output, the stop string "wor" should be included.
    assert_eq!(full_text, "say wor", "full streamed text: {text}");

    assert!(
        payloads.iter().any(|p| p.contains("\"finish_reason\":\"stop\"")),
        "{text}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn non_stream_completions_no_stop_string_match_preserves_original_finish_reason() {
    // Stop string "xyz" does not appear in "hi!" so the original finish reason is
    // preserved.
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
                        "stream": false,
                        "stop": ["xyz"]
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);

    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    engine_task.await.expect("mock engine task");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");

    // Default output is "hi" (stop token '!' suppressed), finish_reason remains
    // "stop" from EOS.
    assert_eq!(json["choices"][0]["text"], "hi");
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
    // No text stop string matched — stop_reason should be absent.
    assert!(json["choices"][0]["stop_reason"].is_null());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn non_stream_completions_stop_string_array_matches_first_occurrence() {
    // Multiple stop strings: "rl" appears in "world" but " wo" appears earlier.
    let output_specs = vec![(
        bytes_to_token_ids(b"say world"),
        Some(EngineCoreFinishReason::Length),
    )];
    let (app, engine_task) = test_app_with_stream_output_specs(output_specs).await;

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
                        "stream": false,
                        "stop": [" wo", "rl"]
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);

    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    engine_task.await.expect("mock engine task");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");

    // " wo" is detected first (at byte 3), so output is truncated to "say".
    assert_eq!(json["choices"][0]["text"], "say");
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
    assert_eq!(json["choices"][0]["stop_reason"], " wo");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn completions_empty_stop_string_returns_validation_error() {
    let (app, _engine_task) = test_app_with_engine_handle().await;

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
                        "stream": false,
                        "stop": [""]
                    })
                    .to_string(),
                ))
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

async fn post_json(
    app: &mut axum::Router,
    uri: &str,
    body: serde_json::Value,
) -> (StatusCode, serde_json::Value) {
    let response = app
        .call(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .expect("build request"),
        )
        .await
        .expect("call app");
    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    let json: serde_json::Value = serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| json!({ "raw": String::from_utf8_lossy(&bytes) }));
    (status, json)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn tokenize_completion_round_trips_through_detokenize() {
    let mut app = test_app().await;
    let prompt = "Hello world";

    let (_, tokenize_json) = post_json(
        &mut app,
        "/tokenize",
        json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "prompt": prompt,
            "add_special_tokens": false,
        }),
    )
    .await;
    let tokens = tokenize_json["tokens"]
        .as_array()
        .expect("tokens array")
        .iter()
        .map(|v| v.as_u64().expect("token id") as u32)
        .collect::<Vec<_>>();

    let (status, detokenize_json) = post_json(
        &mut app,
        "/detokenize",
        json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "tokens": tokens,
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(detokenize_json["prompt"], prompt);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn tokenize_completion_add_special_tokens_changes_ids() {
    let mut app = test_app().await;

    let (_, with_special) = post_json(
        &mut app,
        "/tokenize",
        json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "prompt": "hi",
            "add_special_tokens": true,
        }),
    )
    .await;
    let (_, without_special) = post_json(
        &mut app,
        "/tokenize",
        json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "prompt": "hi",
            "add_special_tokens": false,
        }),
    )
    .await;

    let with_ids: Vec<u32> = with_special["tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let without_ids: Vec<u32> = without_special["tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();

    assert_ne!(with_ids, without_ids);
    assert_eq!(with_ids.first().copied(), Some(FAKE_BOS_TOKEN_ID));
    assert_eq!(without_ids.first().copied(), Some(b'h' as u32));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn tokenize_completion_return_token_strs_matches_tokens() {
    let mut app = test_app().await;

    let (status, json) = post_json(
        &mut app,
        "/tokenize",
        json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "prompt": "hi",
            "add_special_tokens": false,
            "return_token_strs": true,
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    let tokens = json["tokens"].as_array().expect("tokens");
    let token_strs = json["token_strs"].as_array().expect("token_strs");
    assert_eq!(tokens.len(), token_strs.len());
    assert_eq!(token_strs.len(), json["count"].as_u64().unwrap() as usize);
    assert!(!token_strs[0].as_str().unwrap().is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn tokenize_completion_count_and_max_model_len() {
    let mut app = test_app().await;

    let (status, json) = post_json(
        &mut app,
        "/tokenize",
        json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "prompt": "hello",
            "add_special_tokens": false,
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        json["count"].as_u64().unwrap() as usize,
        json["tokens"].as_array().unwrap().len()
    );
    assert!(json["max_model_len"].as_u64().unwrap() > 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn tokenize_chat_includes_generation_prompt_in_token_count() {
    let mut app = test_app().await;
    let messages = json!([{"role": "user", "content": "hi"}]);

    let (_, with_prompt) = post_json(
        &mut app,
        "/tokenize",
        json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "messages": messages,
            "add_generation_prompt": true,
            "add_special_tokens": false,
        }),
    )
    .await;
    let (_, without_prompt) = post_json(
        &mut app,
        "/tokenize",
        json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "messages": messages,
            "add_generation_prompt": false,
            "add_special_tokens": false,
        }),
    )
    .await;

    let with_len = with_prompt["tokens"].as_array().unwrap().len();
    let without_len = without_prompt["tokens"].as_array().unwrap().len();
    assert!(with_len > without_len);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn tokenize_chat_conflicting_generation_flags_returns_400() {
    let mut app = test_app().await;

    let (status, json) = post_json(
        &mut app,
        "/tokenize",
        json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "messages": [{"role": "user", "content": "hi"}],
            "add_generation_prompt": true,
            "continue_final_message": true,
        }),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["type"], "invalid_request_error");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn tokenize_chat_empty_messages_returns_400() {
    let mut app = test_app().await;

    let (status, json) = post_json(
        &mut app,
        "/tokenize",
        json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "messages": [],
        }),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["type"], "invalid_request_error");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn tokenize_chat_empty_message_content_returns_400() {
    let mut app = test_app().await;

    let (status, json) = post_json(
        &mut app,
        "/tokenize",
        json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "messages": [{"role": "user", "content": ""}],
        }),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["type"], "invalid_request_error");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn tokenize_unknown_model_returns_404() {
    let mut app = test_app().await;

    let (status, json) = post_json(
        &mut app,
        "/tokenize",
        json!({
            "model": "does-not-exist",
            "prompt": "hello",
        }),
    )
    .await;

    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(json["error"]["type"], "invalid_request_error");
    assert_eq!(json["error"]["code"], "model_not_found");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn detokenize_unknown_model_returns_404() {
    let mut app = test_app().await;

    let (status, json) = post_json(
        &mut app,
        "/detokenize",
        json!({
            "model": "does-not-exist",
            "tokens": [72, 101, 108, 108, 111],
        }),
    )
    .await;

    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(json["error"]["type"], "invalid_request_error");
    assert_eq!(json["error"]["code"], "model_not_found");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn detokenize_empty_tokens_returns_empty_prompt() {
    let mut app = test_app().await;

    let (status, json) = post_json(
        &mut app,
        "/detokenize",
        json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "tokens": [],
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["prompt"], "");
}

/// Decode an explicit token sequence — pins `/detokenize` independently of
/// `/tokenize` (the round-trip test alone would pass even if encode and decode
/// were both wrong in mirrored ways).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn detokenize_decodes_known_token_ids() {
    let mut app = test_app().await;

    let (status, json) = post_json(
        &mut app,
        "/detokenize",
        json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "tokens": [72, 101, 108, 108, 111],
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["prompt"], "Hello");
}

/// `continue_final_message` without a trailing assistant message must 400.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn tokenize_chat_continue_without_assistant_returns_400() {
    let mut app = test_app().await;

    let (status, json) = post_json(
        &mut app,
        "/tokenize",
        json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "messages": [{"role": "user", "content": "hi"}],
            "add_generation_prompt": false,
            "continue_final_message": true,
        }),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["type"], "invalid_request_error");
}

/// `continue_final_message` must not append a new generation suffix vs `add_generation_prompt`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn tokenize_chat_continue_final_vs_new_assistant_differs() {
    let mut app = test_app().await;
    let messages = json!([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "partial,"}
    ]);

    let (_, continue_final) = post_json(
        &mut app,
        "/tokenize",
        json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "messages": messages,
            "add_generation_prompt": false,
            "continue_final_message": true,
            "add_special_tokens": false,
        }),
    )
    .await;
    let (_, new_assistant) = post_json(
        &mut app,
        "/tokenize",
        json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "messages": messages,
            "add_generation_prompt": true,
            "continue_final_message": false,
            "add_special_tokens": false,
        }),
    )
    .await;

    let continue_len = continue_final["tokens"].as_array().unwrap().len();
    let new_len = new_assistant["tokens"].as_array().unwrap().len();
    assert!(new_len > continue_len);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn world_size_endpoint_is_dev_mode_only() {
    let mut app = test_app().await;
    let response = app
        .call(
            Request::builder()
                .uri("/get_world_size")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn world_size_includes_data_parallelism_by_default() {
    let ready = vllm_engine_core_client::protocol::handshake::EngineCoreReadyResponse {
        world_size: 2,
        data_parallel_size: 4,
        ..default_ready_response()
    };
    let (mut app, _engine_task) = test_dev_mode_app_with_ready(ready).await;

    let response = app
        .call(
            Request::builder()
                .uri("/get_world_size")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");
    assert_eq!(json, json!({"world_size": 8}));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn world_size_excludes_data_parallelism_when_include_dp_false() {
    let ready = vllm_engine_core_client::protocol::handshake::EngineCoreReadyResponse {
        world_size: 2,
        data_parallel_size: 4,
        ..default_ready_response()
    };
    let (mut app, _engine_task) = test_dev_mode_app_with_ready(ready).await;

    let response = app
        .call(
            Request::builder()
                .uri("/get_world_size?include_dp=false")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("decode json");
    assert_eq!(json, json!({"world_size": 2}));
}

// ========================= Profiler route tests =========================

async fn test_profiling_app_with_engine_script<F>(script: F) -> (axum::Router, MockEngineTask)
where
    F: for<'a> FnOnce(&'a mut DealerSocket, &'a mut PushSocket) -> TestFuture<'a> + Send + 'static,
{
    let ipc = IpcNamespace::new().expect("create ipc namespace");
    let handshake_address = ipc.handshake_endpoint();
    let engine_id = b"engine-openai-profiler".to_vec();

    let engine_task = MockEngineTask::new(spawn_mock_engine_task(
        handshake_address.clone(),
        engine_id.clone(),
        move |dealer, push| script(dealer, push),
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

    let chat = ChatLlm::from_shared_backend(test_llm(client), Arc::new(FakeChatBackend::new()));
    (
        build_router(Arc::new(
            AppState::new(vec!["test-model".to_string()], chat)
                .with_profiler(Some("torch".to_string())),
        )),
        engine_task,
    )
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn start_profile_route_sends_expected_utility_call() {
    let (app, engine_task) = test_profiling_app_with_engine_script(|dealer, push| {
        boxed_test_future(async move {
            let utility = recv_engine_message(dealer).await;
            assert_eq!(utility[0].as_ref(), &[0x03]);

            let payload = decode_value(&utility[1]).expect("decode utility payload");
            let array = payload.as_array().expect("utility payload array");
            let call_id = array[1].as_u64().expect("call id");

            assert_eq!(array[2], Value::from("profile"));
            assert_eq!(array[3], Value::Array(vec![Value::from(true), Value::Nil]));

            send_outputs(push, utility_outputs(call_id, utility_none_result())).await;
        })
    })
    .await;

    let response = app
        .clone()
        .call(
            Request::builder()
                .method("POST")
                .uri("/start_profile")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");

    let status = response.status();
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    assert!(body.is_empty());
    engine_task.await.expect("mock engine task");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn stop_profile_route_sends_expected_utility_call() {
    let (app, engine_task) = test_profiling_app_with_engine_script(|dealer, push| {
        boxed_test_future(async move {
            let utility = recv_engine_message(dealer).await;
            assert_eq!(utility[0].as_ref(), &[0x03]);

            let payload = decode_value(&utility[1]).expect("decode utility payload");
            let array = payload.as_array().expect("utility payload array");
            let call_id = array[1].as_u64().expect("call id");

            assert_eq!(array[2], Value::from("profile"));
            assert_eq!(array[3], Value::Array(vec![Value::from(false), Value::Nil]));

            send_outputs(push, utility_outputs(call_id, utility_none_result())).await;
        })
    })
    .await;

    let response = app
        .clone()
        .call(
            Request::builder()
                .method("POST")
                .uri("/stop_profile")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("call app");

    let status = response.status();
    let body = to_bytes(response.into_body(), usize::MAX).await.expect("read body");
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    assert!(body.is_empty());
    engine_task.await.expect("mock engine task");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn profile_routes_are_hidden_when_profiling_is_disabled() {
    let (chat, engine_task) = test_chat_with_engine_handle().await;
    let app = build_router(Arc::new(AppState::new(
        vec!["test-model".to_string()],
        chat,
    )));

    for (method, uri) in [("POST", "/start_profile"), ("POST", "/stop_profile")] {
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
