use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt as _;
use tokio::time::timeout;
use vllm_chat::{
    AssistantBlockKind, AssistantContentBlock, AssistantMessageExt as _, ChatBackend, ChatEvent,
    ChatLlm, ChatMessage, ChatOptions, ChatRequest, ChatRole, ChatTextBackend, ChatTool,
    ChatToolChoice, FinishReason, SamplingParams,
};
use vllm_engine_core_client::protocol::{
    EngineCoreFinishReason, EngineCoreOutput, EngineCoreOutputs, EngineCoreRequest, Logprobs,
    MaybeWireLogprobs, PositionLogprobs, StopReason, TokenLogprob,
};
use vllm_engine_core_client::test_utils::{IpcNamespace, spawn_mock_engine_task};
use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig};
use vllm_llm::Llm;
use vllm_text::tokenizers::{DynTokenizer, Tokenizer};
use vllm_text::{
    DecodedLogprobs, DecodedPositionLogprobs, DecodedPromptLogprobs, DecodedTokenLogprob,
    TextBackend,
};
use zeromq::prelude::{SocketRecv, SocketSend};
use zeromq::{DealerSocket, PushSocket, ZmqMessage};

const SPECIAL_STOP_TOKEN_ID: u32 = 256;

fn request_output(
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
                    rank: 2,
                },
            ],
        }],
    }
}

fn prompt_logprobs_for_hi() -> Logprobs {
    Logprobs {
        positions: vec![PositionLogprobs {
            entries: vec![
                TokenLogprob {
                    token_id: b'i' as u32,
                    logprob: -0.3,
                    rank: 1,
                },
                TokenLogprob {
                    token_id: b'!' as u32,
                    logprob: -0.4,
                    rank: 2,
                },
            ],
        }],
    }
}

fn bytes_to_token_ids(bytes: &[u8]) -> Vec<u32> {
    bytes.iter().map(|byte| u32::from(*byte)).collect()
}

fn bytes_with_special_stop_token(bytes: &[u8]) -> Vec<u32> {
    let mut token_ids = bytes_to_token_ids(bytes);
    token_ids.push(SPECIAL_STOP_TOKEN_ID);
    token_ids
}

async fn send_outputs(push: &mut PushSocket, outputs: EngineCoreOutputs) {
    push.send(ZmqMessage::from(rmp_serde::to_vec_named(&outputs).unwrap()))
        .await
        .unwrap();
}

async fn recv_engine_message(dealer: &mut DealerSocket) -> Vec<bytes::Bytes> {
    dealer.recv().await.unwrap().into_vec()
}

async fn connect_chat_llm_with_ipc(
    config: EngineCoreClientConfig,
    ipc: &IpcNamespace,
    backend: Arc<dyn ChatTextBackend>,
) -> ChatLlm {
    let client = EngineCoreClient::connect_with_input_output_addresses(
        config,
        Some(ipc.input_endpoint()),
        Some(ipc.output_endpoint()),
    )
    .await
    .unwrap();
    ChatLlm::from_shared_backend(Llm::new(client), backend)
}

#[derive(Clone)]
struct FakeChatBackend {
    has_template: bool,
    model_id: Option<String>,
}

#[derive(Debug)]
struct FakeChatTokenizer;

impl Tokenizer for FakeChatTokenizer {
    fn encode(&self, text: &str, _add_special_tokens: bool) -> vllm_text::Result<Vec<u32>> {
        Ok(text.bytes().map(u32::from).collect())
    }

    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> vllm_text::Result<String> {
        let bytes = token_ids
            .iter()
            .filter_map(|id| {
                if skip_special_tokens && *id == SPECIAL_STOP_TOKEN_ID {
                    None
                } else {
                    Some(*id as u8)
                }
            })
            .collect::<Vec<_>>();
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        token.bytes().next().map(u32::from)
    }
}

impl fmt::Debug for FakeChatBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FakeChatBackend").finish_non_exhaustive()
    }
}

impl FakeChatBackend {
    fn new() -> Self {
        Self {
            has_template: true,
            model_id: None,
        }
    }

    fn without_template() -> Self {
        Self {
            has_template: false,
            model_id: None,
        }
    }

    fn with_model_id(model_id: impl Into<String>) -> Self {
        Self {
            has_template: true,
            model_id: Some(model_id.into()),
        }
    }
}

impl TextBackend for FakeChatBackend {
    fn tokenizer(&self) -> DynTokenizer {
        Arc::new(FakeChatTokenizer)
    }

    fn model_id(&self) -> Option<&str> {
        self.model_id.as_deref()
    }
}

impl ChatBackend for FakeChatBackend {
    fn apply_chat_template(&self, request: &ChatRequest) -> vllm_chat::Result<String> {
        if !self.has_template {
            return Err(vllm_chat::Error::MissingChatTemplate);
        }

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

#[derive(Debug)]
struct FailingDecodeBackend {
    inner: FakeChatBackend,
}

#[derive(Debug)]
struct FailingDecodeTokenizer;

impl Tokenizer for FailingDecodeTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> vllm_text::Result<Vec<u32>> {
        FakeChatTokenizer.encode(text, add_special_tokens)
    }

    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> vllm_text::Result<String> {
        if token_ids.contains(&(b'i' as u32)) {
            return Err(vllm_text::Error::Tokenizer("decode failed".to_string()));
        }
        FakeChatTokenizer.decode(token_ids, skip_special_tokens)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        FakeChatTokenizer.token_to_id(token)
    }
}

impl TextBackend for FailingDecodeBackend {
    fn tokenizer(&self) -> DynTokenizer {
        Arc::new(FailingDecodeTokenizer)
    }
}

impl ChatBackend for FailingDecodeBackend {
    fn apply_chat_template(&self, request: &ChatRequest) -> vllm_chat::Result<String> {
        self.inner.apply_chat_template(request)
    }
}

fn sample_request(request_id: &str) -> ChatRequest {
    ChatRequest {
        request_id: request_id.to_string(),
        messages: vec![
            ChatMessage::text(ChatRole::System, "You are terse."),
            ChatMessage::text(ChatRole::User, "Say hi"),
        ],
        sampling_params: SamplingParams {
            max_tokens: Some(8),
            ..Default::default()
        },
        chat_options: ChatOptions::default(),
        tools: Vec::new(),
        tool_choice: ChatToolChoice::None,
        decode_options: Default::default(),
        intermediate: true,
        priority: 0,
        documents: None,
        cache_salt: None,
        add_special_tokens: false,
    }
}

fn sample_tool_request(request_id: &str) -> ChatRequest {
    let mut request = sample_request(request_id);
    request.tools = vec![ChatTool {
        name: "get_weather".to_string(),
        description: Some("Get weather".to_string()),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        }),
        strict: None,
    }];
    request.tool_choice = ChatToolChoice::Auto;
    request
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_streams_text_events() {
    let ipc = IpcNamespace::new().unwrap();
    let handshake_address = ipc.handshake_endpoint();
    let engine_id = b"engine-chat".to_vec();

    let (shutdown_tx, engine_task) = spawn_mock_engine_task(
        handshake_address.clone(),
        engine_id.clone(),
        |dealer, push| {
            Box::pin(async move {
                let add = recv_engine_message(dealer).await;
                assert_eq!(add[0].as_ref(), &[0x00]);
                let request: EngineCoreRequest = rmp_serde::from_slice(&add[1]).unwrap();
                assert_eq!(request.request_id, "chat-1");
                // more fields here in the future
                assert_eq!(
                    String::from_utf8(
                        request
                            .prompt_token_ids
                            .clone()
                            .unwrap()
                            .into_iter()
                            .map(|id| id as u8)
                            .collect()
                    )
                    .unwrap(),
                    "system: You are terse.\nuser: Say hi\nassistant:"
                );
                send_outputs(
                    push,
                    EngineCoreOutputs {
                        outputs: vec![
                            request_output("chat-1", vec![b'H' as u32], None, None),
                            request_output(
                                "chat-1",
                                vec![b'i' as u32, b'!' as u32],
                                Some(EngineCoreFinishReason::Stop),
                                Some(StopReason::TokenId(b'!' as u32)),
                            ),
                        ],
                        finished_requests: Some(BTreeSet::from(["chat-1".to_string()])),
                        ..Default::default()
                    },
                )
                .await;
            })
        },
    );

    let backend: Arc<dyn ChatTextBackend> = Arc::new(FakeChatBackend::new());
    let chat = connect_chat_llm_with_ipc(
        EngineCoreClientConfig::new_single(handshake_address).with_model_name("test-model"),
        &ipc,
        backend,
    )
    .await;

    let mut stream = chat.chat(sample_request("chat-1")).await.unwrap();

    assert_eq!(
        stream.next().await.unwrap().unwrap(),
        ChatEvent::Start {
            prompt_token_count: "system: You are terse.\nuser: Say hi\nassistant:".len(),
            prompt_logprobs: None,
        }
    );
    assert_eq!(
        stream.next().await.unwrap().unwrap(),
        ChatEvent::BlockStart {
            index: 0,
            kind: AssistantBlockKind::Text,
        }
    );
    assert_eq!(
        stream.next().await.unwrap().unwrap(),
        ChatEvent::BlockDelta {
            index: 0,
            kind: AssistantBlockKind::Text,
            delta: "H".to_string(),
        }
    );
    assert_eq!(
        stream.next().await.unwrap().unwrap(),
        ChatEvent::BlockDelta {
            index: 0,
            kind: AssistantBlockKind::Text,
            delta: "i".to_string(),
        }
    );
    assert_eq!(
        stream.next().await.unwrap().unwrap(),
        ChatEvent::BlockEnd {
            index: 0,
            block: AssistantContentBlock::Text {
                text: "Hi".to_string(),
            },
        }
    );

    match stream.next().await {
        Some(Ok(ChatEvent::Done {
            message,
            output_token_count,
            finish_reason,
            ..
        })) => {
            assert_eq!(message.text(), "Hi");
            assert_eq!(output_token_count, 3);
            assert_eq!(
                finish_reason,
                FinishReason::Stop(Some(StopReason::TokenId(b'!' as u32)))
            );
        }
        other => panic!("unexpected final event: {other:?}"),
    }
    assert!(stream.next().await.is_none());

    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    chat.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_stream_waits_for_complete_utf8_before_emitting() {
    let ipc = IpcNamespace::new().unwrap();
    let handshake_address = ipc.handshake_endpoint();
    let engine_id = b"engine-chat-utf8".to_vec();

    let (shutdown_tx, engine_task) = spawn_mock_engine_task(
        handshake_address.clone(),
        engine_id.clone(),
        |dealer, push| {
            Box::pin(async move {
                let _ = recv_engine_message(dealer).await;
                send_outputs(
                    push,
                    EngineCoreOutputs {
                        outputs: vec![
                            request_output("chat-utf8", bytes_to_token_ids(&[0xe4]), None, None),
                            request_output(
                                "chat-utf8",
                                bytes_to_token_ids(&[0xbd, 0xa0, b'!']),
                                Some(EngineCoreFinishReason::Stop),
                                Some(StopReason::TokenId(b'!' as u32)),
                            ),
                        ],
                        finished_requests: Some(BTreeSet::from(["chat-utf8".to_string()])),
                        ..Default::default()
                    },
                )
                .await;
            })
        },
    );

    let backend: Arc<dyn ChatTextBackend> = Arc::new(FakeChatBackend::new());
    let chat = connect_chat_llm_with_ipc(
        EngineCoreClientConfig::new_single(handshake_address),
        &ipc,
        backend,
    )
    .await;

    let mut stream = chat.chat(sample_request("chat-utf8")).await.unwrap();

    assert!(matches!(
        stream.next().await,
        Some(Ok(ChatEvent::Start {
            prompt_logprobs: None,
            ..
        }))
    ));
    assert_eq!(
        stream.next().await.unwrap().unwrap(),
        ChatEvent::BlockStart {
            index: 0,
            kind: AssistantBlockKind::Text,
        }
    );
    assert_eq!(
        stream.next().await.unwrap().unwrap(),
        ChatEvent::BlockDelta {
            index: 0,
            kind: AssistantBlockKind::Text,
            delta: "你".to_string(),
        }
    );
    assert_eq!(
        stream.next().await.unwrap().unwrap(),
        ChatEvent::BlockEnd {
            index: 0,
            block: AssistantContentBlock::Text {
                text: "你".to_string(),
            },
        }
    );

    match stream.next().await {
        Some(Ok(ChatEvent::Done {
            message,
            output_token_count,
            ..
        })) => {
            assert_eq!(message.text(), "你");
            assert_eq!(output_token_count, 4);
        }
        other => panic!("unexpected final event: {other:?}"),
    }

    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    chat.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_stream_flushes_held_text_on_finish() {
    let ipc = IpcNamespace::new().unwrap();
    let handshake_address = ipc.handshake_endpoint();
    let engine_id = b"engine-chat-final-flush".to_vec();

    let (shutdown_tx, engine_task) = spawn_mock_engine_task(
        handshake_address.clone(),
        engine_id.clone(),
        |dealer, push| {
            Box::pin(async move {
                let _ = recv_engine_message(dealer).await;
                send_outputs(
                    push,
                    EngineCoreOutputs {
                        outputs: vec![request_output(
                            "chat-final-flush",
                            bytes_to_token_ids(b"ok st"),
                            Some(EngineCoreFinishReason::Length),
                            None,
                        )],
                        finished_requests: Some(BTreeSet::from(["chat-final-flush".to_string()])),
                        ..Default::default()
                    },
                )
                .await;
            })
        },
    );

    let backend: Arc<dyn ChatTextBackend> = Arc::new(FakeChatBackend::new());
    let chat = connect_chat_llm_with_ipc(
        EngineCoreClientConfig::new_single(handshake_address),
        &ipc,
        backend,
    )
    .await;

    let mut stream = chat.chat(sample_request("chat-final-flush")).await.unwrap();

    assert!(matches!(
        stream.next().await,
        Some(Ok(ChatEvent::Start {
            prompt_logprobs: None,
            ..
        }))
    ));
    assert_eq!(
        stream.next().await.unwrap().unwrap(),
        ChatEvent::BlockStart {
            index: 0,
            kind: AssistantBlockKind::Text,
        }
    );
    assert_eq!(
        stream.next().await.unwrap().unwrap(),
        ChatEvent::BlockDelta {
            index: 0,
            kind: AssistantBlockKind::Text,
            delta: "ok st".to_string(),
        }
    );
    assert_eq!(
        stream.next().await.unwrap().unwrap(),
        ChatEvent::BlockEnd {
            index: 0,
            block: AssistantContentBlock::Text {
                text: "ok st".to_string(),
            },
        }
    );

    match stream.next().await {
        Some(Ok(ChatEvent::Done {
            message,
            output_token_count,
            finish_reason,
            ..
        })) => {
            assert_eq!(message.text(), "ok st");
            assert_eq!(output_token_count, 5);
            assert_eq!(finish_reason, FinishReason::Length);
        }
        other => panic!("unexpected final event: {other:?}"),
    }

    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    chat.shutdown().await.unwrap();
}

#[test]
fn chat_request_rejects_conflicting_generation_modes() {
    let mut request = sample_request("chat-2");
    request.chat_options.continue_final_message = true;
    let error = request.validate().unwrap_err();

    assert!(matches!(
        error,
        vllm_chat::Error::ConflictingGenerationPromptMode
    ));
}

#[test]
fn backend_requires_a_template() {
    let request = sample_request("chat-3");
    let backend = FakeChatBackend::without_template();
    let error = backend.apply_chat_template(&request).unwrap_err();
    assert!(matches!(error, vllm_chat::Error::MissingChatTemplate));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_stream_reports_decode_failure_as_error_event() {
    let ipc = IpcNamespace::new().unwrap();
    let handshake_address = ipc.handshake_endpoint();
    let engine_id = b"engine-chat-decode-fail".to_vec();

    let (shutdown_tx, engine_task) = spawn_mock_engine_task(
        handshake_address.clone(),
        engine_id.clone(),
        |dealer, push| {
            Box::pin(async move {
                let _ = recv_engine_message(dealer).await;
                send_outputs(
                    push,
                    EngineCoreOutputs {
                        outputs: vec![request_output("chat-4", vec![b'X' as u32], None, None)],
                        ..Default::default()
                    },
                )
                .await;
            })
        },
    );

    let backend: Arc<dyn ChatTextBackend> = Arc::new(FailingDecodeBackend {
        inner: FakeChatBackend::new(),
    });
    let chat = connect_chat_llm_with_ipc(
        EngineCoreClientConfig::new_single(handshake_address),
        &ipc,
        backend,
    )
    .await;

    let mut stream = chat.chat(sample_request("chat-4")).await.unwrap();
    assert_eq!(stream.request_id(), "chat-4");
    assert!(matches!(
        stream.next().await,
        Some(Ok(ChatEvent::Start {
            prompt_logprobs: None,
            ..
        }))
    ));

    match timeout(Duration::from_secs(2), stream.next())
        .await
        .unwrap()
    {
        Some(Err(vllm_chat::Error::Text(vllm_text::Error::Tokenizer(message)))) => {
            assert_eq!(message, "decode failed");
        }
        other => panic!("unexpected event after close: {other:?}"),
    }

    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    chat.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_stream_preserves_terminal_stop_token_when_requested() {
    let ipc = IpcNamespace::new().unwrap();
    let handshake_address = ipc.handshake_endpoint();
    let engine_id = b"engine-chat-include-stop".to_vec();

    let (shutdown_tx, engine_task) = spawn_mock_engine_task(
        handshake_address.clone(),
        engine_id.clone(),
        |dealer, push| {
            Box::pin(async move {
                let _ = recv_engine_message(dealer).await;
                send_outputs(
                    push,
                    EngineCoreOutputs {
                        outputs: vec![request_output(
                            "chat-include-stop",
                            vec![b'H' as u32, b'i' as u32, b'!' as u32],
                            Some(EngineCoreFinishReason::Stop),
                            Some(StopReason::TokenId(b'!' as u32)),
                        )],
                        finished_requests: Some(BTreeSet::from(["chat-include-stop".to_string()])),
                        ..Default::default()
                    },
                )
                .await;
            })
        },
    );

    let backend: Arc<dyn ChatTextBackend> = Arc::new(FakeChatBackend::new());
    let chat = connect_chat_llm_with_ipc(
        EngineCoreClientConfig::new_single(handshake_address),
        &ipc,
        backend,
    )
    .await;

    let mut request = sample_request("chat-include-stop");
    request.decode_options.include_stop_str_in_output = true;
    let mut stream = chat.chat(request).await.unwrap();

    assert!(matches!(
        stream.next().await,
        Some(Ok(ChatEvent::Start {
            prompt_logprobs: None,
            ..
        }))
    ));
    assert_eq!(
        stream.next().await.unwrap().unwrap(),
        ChatEvent::BlockStart {
            index: 0,
            kind: AssistantBlockKind::Text,
        }
    );
    assert_eq!(
        stream.next().await.unwrap().unwrap(),
        ChatEvent::BlockDelta {
            index: 0,
            kind: AssistantBlockKind::Text,
            delta: "Hi!".to_string(),
        }
    );
    assert_eq!(
        stream.next().await.unwrap().unwrap(),
        ChatEvent::BlockEnd {
            index: 0,
            block: AssistantContentBlock::Text {
                text: "Hi!".to_string(),
            },
        }
    );

    match stream.next().await {
        Some(Ok(ChatEvent::Done {
            message,
            output_token_count,
            ..
        })) => {
            assert_eq!(message.text(), "Hi!");
            assert_eq!(output_token_count, 3);
        }
        other => panic!("unexpected final event: {other:?}"),
    }

    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    chat.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_stream_separates_reasoning_blocks_automatically() {
    let ipc = IpcNamespace::new().unwrap();
    let handshake_address = ipc.handshake_endpoint();
    let engine_id = b"engine-chat-reasoning".to_vec();

    let (shutdown_tx, engine_task) = spawn_mock_engine_task(
        handshake_address.clone(),
        engine_id.clone(),
        |dealer, push| {
            Box::pin(async move {
                let _ = recv_engine_message(dealer).await;
                send_outputs(
                    push,
                    EngineCoreOutputs {
                        outputs: vec![
                            request_output(
                                "chat-reasoning",
                                bytes_to_token_ids(b"<think>"),
                                None,
                                None,
                            ),
                            request_output(
                                "chat-reasoning",
                                bytes_to_token_ids(b"reason "),
                                None,
                                None,
                            ),
                            request_output(
                                "chat-reasoning",
                                bytes_to_token_ids(b"more</think>"),
                                None,
                                None,
                            ),
                            request_output(
                                "chat-reasoning",
                                bytes_to_token_ids(b"answer"),
                                Some(EngineCoreFinishReason::Length),
                                None,
                            ),
                        ],
                        finished_requests: Some(BTreeSet::from(["chat-reasoning".to_string()])),
                        ..Default::default()
                    },
                )
                .await;
            })
        },
    );

    let backend: Arc<dyn ChatTextBackend> =
        Arc::new(FakeChatBackend::with_model_id("Qwen/Qwen3-0.6B"));
    let chat = connect_chat_llm_with_ipc(
        EngineCoreClientConfig::new_single(handshake_address),
        &ipc,
        backend,
    )
    .await;

    let mut stream = chat.chat(sample_request("chat-reasoning")).await.unwrap();

    assert!(matches!(
        stream.next().await,
        Some(Ok(ChatEvent::Start {
            prompt_logprobs: None,
            ..
        }))
    ));
    assert_eq!(
        stream.next().await.unwrap().unwrap(),
        ChatEvent::BlockStart {
            index: 0,
            kind: AssistantBlockKind::Reasoning,
        }
    );
    assert_eq!(
        stream.next().await.unwrap().unwrap(),
        ChatEvent::BlockDelta {
            index: 0,
            kind: AssistantBlockKind::Reasoning,
            delta: "reason ".to_string(),
        }
    );
    assert_eq!(
        stream.next().await.unwrap().unwrap(),
        ChatEvent::BlockDelta {
            index: 0,
            kind: AssistantBlockKind::Reasoning,
            delta: "more".to_string(),
        }
    );
    assert_eq!(
        stream.next().await.unwrap().unwrap(),
        ChatEvent::BlockEnd {
            index: 0,
            block: AssistantContentBlock::Reasoning {
                text: "reason more".to_string(),
            },
        }
    );
    assert_eq!(
        stream.next().await.unwrap().unwrap(),
        ChatEvent::BlockStart {
            index: 1,
            kind: AssistantBlockKind::Text,
        }
    );
    assert_eq!(
        stream.next().await.unwrap().unwrap(),
        ChatEvent::BlockDelta {
            index: 1,
            kind: AssistantBlockKind::Text,
            delta: "answer".to_string(),
        }
    );
    assert_eq!(
        stream.next().await.unwrap().unwrap(),
        ChatEvent::BlockEnd {
            index: 1,
            block: AssistantContentBlock::Text {
                text: "answer".to_string(),
            },
        }
    );

    match stream.next().await {
        Some(Ok(ChatEvent::Done {
            message,
            finish_reason,
            ..
        })) => {
            assert_eq!(message.reasoning().unwrap(), "reason more");
            assert_eq!(message.text(), "answer");
            assert_eq!(finish_reason, FinishReason::Length);
        }
        other => panic!("unexpected final event: {other:?}"),
    }

    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    chat.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_collectors_return_structured_message_and_visible_text() {
    let ipc = IpcNamespace::new().unwrap();
    let handshake_address = ipc.handshake_endpoint();
    let engine_id = b"engine-chat-collect".to_vec();

    let (shutdown_tx, engine_task) = spawn_mock_engine_task(
        handshake_address.clone(),
        engine_id.clone(),
        |dealer, push| {
            Box::pin(async move {
                let _ = recv_engine_message(dealer).await;
                send_outputs(
                    push,
                    EngineCoreOutputs {
                        outputs: vec![request_output(
                            "chat-collect",
                            bytes_to_token_ids(b"<think>inner</think>outer"),
                            Some(EngineCoreFinishReason::Length),
                            None,
                        )],
                        finished_requests: Some(BTreeSet::from(["chat-collect".to_string()])),
                        ..Default::default()
                    },
                )
                .await;
            })
        },
    );

    let backend: Arc<dyn ChatTextBackend> =
        Arc::new(FakeChatBackend::with_model_id("Qwen/Qwen3-0.6B"));
    let chat = connect_chat_llm_with_ipc(
        EngineCoreClientConfig::new_single(handshake_address.clone()),
        &ipc,
        backend.clone(),
    )
    .await;

    let message = chat
        .chat(sample_request("chat-collect"))
        .await
        .unwrap()
        .collect_message()
        .await
        .unwrap();
    assert_eq!(message.message.reasoning().unwrap(), "inner");
    assert_eq!(message.message.text(), "outer");
    assert_eq!(message.finish_reason, FinishReason::Length);
    assert_eq!(
        message.prompt_token_count,
        "system: You are terse.\nuser: Say hi\nassistant:".len()
    );
    assert_eq!(
        message.output_token_count,
        "<think>inner</think>outer".len()
    );

    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    chat.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_stream_parses_tool_calls_automatically() {
    let ipc = IpcNamespace::new().unwrap();
    let handshake_address = ipc.handshake_endpoint();
    let engine_id = b"engine-chat-tool".to_vec();

    let (shutdown_tx, engine_task) = spawn_mock_engine_task(
        handshake_address.clone(),
        engine_id.clone(),
        |dealer, push| {
            Box::pin(async move {
                let _ = recv_engine_message(dealer).await;
                send_outputs(
                    push,
                    EngineCoreOutputs {
                        outputs: vec![
                            request_output(
                                "chat-tool",
                                bytes_to_token_ids(b"<tool_call>\n{\"name\":\"get_weather\", "),
                                None,
                                None,
                            ),
                            request_output(
                                "chat-tool",
                                bytes_to_token_ids(
                                    b"\"arguments\":{\"city\":\"Paris\"}}\n</tool_call>",
                                ),
                                Some(EngineCoreFinishReason::Stop),
                                Some(StopReason::TokenId(SPECIAL_STOP_TOKEN_ID)),
                            ),
                        ],
                        finished_requests: Some(BTreeSet::from(["chat-tool".to_string()])),
                        ..Default::default()
                    },
                )
                .await;
            })
        },
    );

    let backend: Arc<dyn ChatTextBackend> =
        Arc::new(FakeChatBackend::with_model_id("Qwen/Qwen3-0.6B"));
    let chat = connect_chat_llm_with_ipc(
        EngineCoreClientConfig::new_single(handshake_address),
        &ipc,
        backend,
    )
    .await;
    let mut stream = chat.chat(sample_tool_request("chat-tool")).await.unwrap();

    let mut saw_tool_start = false;
    let mut saw_tool_args = false;
    let mut saw_tool_end = false;

    while let Some(event) = stream.next().await {
        match event.unwrap() {
            ChatEvent::Start { .. } => {}
            ChatEvent::LogprobsDelta { .. } => {}
            ChatEvent::ToolCallStart { name, .. } => {
                saw_tool_start = true;
                assert_eq!(name, "get_weather");
            }
            ChatEvent::ToolCallArgumentsDelta { delta, .. } => {
                saw_tool_args = true;
                assert!(delta.contains("Paris"), "{delta}");
            }
            ChatEvent::ToolCallEnd { call, .. } => {
                saw_tool_end = true;
                assert_eq!(call.name, "get_weather");
                assert_eq!(call.arguments, r#"{"city":"Paris"}"#);
            }
            ChatEvent::Done {
                message,
                finish_reason,
                ..
            } => {
                assert_eq!(
                    finish_reason,
                    FinishReason::Stop(Some(StopReason::TokenId(SPECIAL_STOP_TOKEN_ID)))
                );
                assert_eq!(message.text(), "");
                let tool_calls = message.tool_calls().collect::<Vec<_>>();
                assert_eq!(tool_calls.len(), 1);
                assert_eq!(tool_calls[0].name, "get_weather");
                assert_eq!(tool_calls[0].arguments, r#"{"city":"Paris"}"#);
                break;
            }
            ChatEvent::BlockStart { .. }
            | ChatEvent::BlockDelta { .. }
            | ChatEvent::BlockEnd { .. } => {}
        }
    }

    assert!(saw_tool_start);
    assert!(saw_tool_args);
    assert!(saw_tool_end);

    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    chat.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_collect_message_preserves_tool_call_arguments_in_final_only_mode() {
    let ipc = IpcNamespace::new().unwrap();
    let handshake_address = ipc.handshake_endpoint();
    let engine_id = b"engine-chat-final-only-tool".to_vec();

    let (shutdown_tx, engine_task) = spawn_mock_engine_task(
        handshake_address.clone(),
        engine_id.clone(),
        |dealer, push| {
            Box::pin(async move {
                let _ = recv_engine_message(dealer).await;
                send_outputs(
                    push,
                    EngineCoreOutputs {
                        outputs: vec![
                            request_output(
                                "chat-final-only-tool",
                                bytes_to_token_ids(b"<tool_call>\n{\"name\":\"get_weather\", "),
                                None,
                                None,
                            ),
                            request_output(
                                "chat-final-only-tool",
                                bytes_with_special_stop_token(
                                    b"\"arguments\":{\"city\":\"Paris\"}}\n</tool_call>",
                                ),
                                Some(EngineCoreFinishReason::Stop),
                                Some(StopReason::TokenId(SPECIAL_STOP_TOKEN_ID)),
                            ),
                        ],
                        finished_requests: Some(BTreeSet::from([
                            "chat-final-only-tool".to_string()
                        ])),
                        ..Default::default()
                    },
                )
                .await;
            })
        },
    );

    let backend: Arc<dyn ChatTextBackend> =
        Arc::new(FakeChatBackend::with_model_id("Qwen/Qwen3-0.6B"));
    let chat = connect_chat_llm_with_ipc(
        EngineCoreClientConfig::new_single(handshake_address),
        &ipc,
        backend,
    )
    .await;
    let mut request = sample_tool_request("chat-final-only-tool");
    request.intermediate = false;

    let message = chat
        .chat(request)
        .await
        .unwrap()
        .collect_message()
        .await
        .unwrap();

    assert_eq!(
        message.finish_reason,
        FinishReason::Stop(Some(StopReason::TokenId(SPECIAL_STOP_TOKEN_ID)))
    );
    assert_eq!(message.message.tool_calls().count(), 1);
    assert_eq!(
        message.message.tool_calls().next().unwrap().arguments,
        r#"{"city":"Paris"}"#
    );

    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    chat.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_stream_and_collect_preserve_prompt_and_sample_logprobs() {
    let ipc = IpcNamespace::new().unwrap();
    let handshake_address = ipc.handshake_endpoint();
    let engine_id = b"engine-chat-logprobs".to_vec();

    let (shutdown_tx, engine_task) = spawn_mock_engine_task(
        handshake_address.clone(),
        engine_id.clone(),
        |dealer, push| {
            Box::pin(async move {
                for _ in 0..2 {
                    let add = recv_engine_message(dealer).await;
                    let request: EngineCoreRequest = rmp_serde::from_slice(&add[1]).unwrap();
                    send_outputs(
                        push,
                        EngineCoreOutputs {
                            outputs: vec![
                                request_output_with_logprobs(
                                    &request.request_id,
                                    vec![b'H' as u32],
                                    None,
                                    None,
                                    Some(sample_logprobs_for_token(b'H' as u32, b'h' as u32)),
                                    Some(prompt_logprobs_for_hi()),
                                ),
                                request_output_with_logprobs(
                                    &request.request_id,
                                    vec![b'i' as u32],
                                    Some(EngineCoreFinishReason::Length),
                                    None,
                                    Some(sample_logprobs_for_token(b'i' as u32, b'I' as u32)),
                                    None,
                                ),
                            ],
                            finished_requests: Some(BTreeSet::from([request.request_id])),
                            ..Default::default()
                        },
                    )
                    .await;
                }
            })
        },
    );

    let backend: Arc<dyn ChatTextBackend> = Arc::new(FakeChatBackend::new());
    let chat = connect_chat_llm_with_ipc(
        EngineCoreClientConfig::new_single(handshake_address.clone()),
        &ipc,
        backend,
    )
    .await;

    let mut request = sample_request("chat-logprobs");
    request.sampling_params.logprobs = Some(1);
    request.sampling_params.prompt_logprobs = Some(1);

    let mut stream = chat.chat(request.clone()).await.unwrap();
    assert_eq!(
        stream.next().await.unwrap().unwrap(),
        ChatEvent::Start {
            prompt_token_count: "system: You are terse.\nuser: Say hi\nassistant:".len(),
            prompt_logprobs: Some(DecodedPromptLogprobs {
                first_token: "s".to_string(),
                scored_positions: vec![DecodedPositionLogprobs {
                    entries: vec![
                        DecodedTokenLogprob {
                            token: "i".to_string(),
                            logprob: -0.3,
                            rank: 1,
                        },
                        DecodedTokenLogprob {
                            token: "!".to_string(),
                            logprob: -0.4,
                            rank: 1,
                        },
                    ],
                }],
            }),
        }
    );
    assert_eq!(
        stream.next().await.unwrap().unwrap(),
        ChatEvent::BlockStart {
            index: 0,
            kind: AssistantBlockKind::Text,
        }
    );
    assert_eq!(
        stream.next().await.unwrap().unwrap(),
        ChatEvent::BlockDelta {
            index: 0,
            kind: AssistantBlockKind::Text,
            delta: "H".to_string(),
        }
    );
    assert_eq!(
        stream.next().await.unwrap().unwrap(),
        ChatEvent::LogprobsDelta {
            logprobs: DecodedLogprobs {
                positions: vec![DecodedPositionLogprobs {
                    entries: vec![
                        DecodedTokenLogprob {
                            token: "H".to_string(),
                            logprob: -0.1,
                            rank: 1,
                        },
                        DecodedTokenLogprob {
                            token: "h".to_string(),
                            logprob: -0.2,
                            rank: 1,
                        },
                    ],
                }],
            },
        }
    );
    while !matches!(stream.next().await, Some(Ok(ChatEvent::Done { .. }))) {}

    request.request_id = "chat-logprobs-collect".to_string();
    let collected = chat
        .chat(request)
        .await
        .unwrap()
        .collect_message()
        .await
        .unwrap();
    assert_eq!(collected.message.text(), "Hi");
    assert_eq!(
        collected.prompt_logprobs,
        Some(DecodedPromptLogprobs {
            first_token: "s".to_string(),
            scored_positions: vec![DecodedPositionLogprobs {
                entries: vec![
                    DecodedTokenLogprob {
                        token: "i".to_string(),
                        logprob: -0.3,
                        rank: 1,
                    },
                    DecodedTokenLogprob {
                        token: "!".to_string(),
                        logprob: -0.4,
                        rank: 1,
                    },
                ],
            }],
        })
    );
    assert_eq!(
        collected.logprobs,
        Some(DecodedLogprobs {
            positions: vec![
                DecodedPositionLogprobs {
                    entries: vec![
                        DecodedTokenLogprob {
                            token: "H".to_string(),
                            logprob: -0.1,
                            rank: 1,
                        },
                        DecodedTokenLogprob {
                            token: "h".to_string(),
                            logprob: -0.2,
                            rank: 1,
                        },
                    ],
                },
                DecodedPositionLogprobs {
                    entries: vec![
                        DecodedTokenLogprob {
                            token: "i".to_string(),
                            logprob: -0.1,
                            rank: 1,
                        },
                        DecodedTokenLogprob {
                            token: "I".to_string(),
                            logprob: -0.2,
                            rank: 1,
                        },
                    ],
                },
            ],
        })
    );

    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    chat.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_rejects_tool_parsing_without_model_hint() {
    let ipc = IpcNamespace::new().unwrap();
    let handshake_address = ipc.handshake_endpoint();
    let engine_id = b"engine-chat-tool-no-model".to_vec();
    let (shutdown_tx, engine_task) =
        spawn_mock_engine_task(handshake_address.clone(), engine_id, |_, _| {
            Box::pin(async move {})
        });

    let backend: Arc<dyn ChatTextBackend> = Arc::new(FakeChatBackend::new());
    let chat = connect_chat_llm_with_ipc(
        EngineCoreClientConfig::new_single(handshake_address),
        &ipc,
        backend,
    )
    .await;
    let error = match chat.chat(sample_tool_request("chat-tool-no-model")).await {
        Ok(_) => panic!("tool parsing without model hint should fail"),
        Err(error) => error,
    };

    assert!(matches!(error, vllm_chat::Error::ToolParserRequiresModelId));

    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    chat.shutdown().await.unwrap();
}
