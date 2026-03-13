use std::collections::BTreeSet;
use std::convert::TryFrom;
use std::net::TcpListener;
use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt as _;
use smg_tokenizer::{ChatTemplateState, Decoder, Encoder, Encoding, SpecialTokens, TokenizerTrait};
use tokio::time::timeout;
use vllm_chat::{
    ChatEvent, ChatLlm, ChatMessage, ChatOptions, ChatRenderer, ChatRequest, ChatRole,
    SmgTokenizer, SmgTokenizerChatRenderer, Tokenizer,
};
use vllm_engine_core_client::protocol::handshake::{HandshakeInitMessage, ReadyMessage};
use vllm_engine_core_client::protocol::{
    EngineCoreOutput, EngineCoreOutputs, EngineCoreRequest, FinishReason, SamplingParams,
};
use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig};
use vllm_llm::Llm;
use zeromq::prelude::{Socket, SocketRecv, SocketSend};
use zeromq::util::PeerIdentity;
use zeromq::{DealerSocket, PushSocket, SocketOptions, ZmqMessage};

fn unique_tcp_endpoint() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
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

async fn send_outputs(push: &mut PushSocket, outputs: EngineCoreOutputs) {
    push.send(ZmqMessage::from(rmp_serde::to_vec_named(&outputs).unwrap()))
        .await
        .unwrap();
}

async fn recv_engine_message(dealer: &mut DealerSocket) -> Vec<bytes::Bytes> {
    dealer.recv().await.unwrap().into_vec()
}

async fn setup_mock_engine(
    engine_handshake: String,
    engine_identity: Vec<u8>,
) -> (DealerSocket, PushSocket) {
    tokio::time::sleep(Duration::from_millis(200)).await;

    let mut options = SocketOptions::default();
    options.peer_identity(PeerIdentity::try_from(engine_identity.clone()).unwrap());
    let mut handshake = DealerSocket::with_options(options);
    handshake.connect(&engine_handshake).await.unwrap();
    handshake
        .send(ZmqMessage::from(
            rmp_serde::to_vec_named(&ready_message("HELLO")).unwrap(),
        ))
        .await
        .unwrap();

    let init_frames = handshake.recv().await.unwrap().into_vec();
    assert_eq!(init_frames.len(), 1);
    let init: HandshakeInitMessage = rmp_serde::from_slice(init_frames[0].as_ref()).unwrap();

    let engine_input = init.addresses.inputs[0].clone();
    let engine_output = init.addresses.outputs[0].clone();

    let mut input_options = SocketOptions::default();
    input_options.peer_identity(PeerIdentity::try_from(engine_identity).unwrap());
    let mut dealer = DealerSocket::with_options(input_options);
    dealer.connect(&engine_input).await.unwrap();
    dealer
        .send(ZmqMessage::from(Vec::<u8>::new()))
        .await
        .unwrap();

    let mut push = PushSocket::new();
    push.connect(&engine_output).await.unwrap();

    handshake
        .send(ZmqMessage::from(
            rmp_serde::to_vec_named(&ready_message("READY")).unwrap(),
        ))
        .await
        .unwrap();

    (dealer, push)
}

async fn connect_chat_llm(
    handshake_address: String,
    renderer: Arc<dyn ChatRenderer>,
    tokenizer: Arc<dyn Tokenizer>,
) -> ChatLlm {
    let client = EngineCoreClient::connect(EngineCoreClientConfig {
        handshake_address,
        local_host: "127.0.0.1".to_string(),
        ready_timeout: Duration::from_secs(2),
        client_index: 0,
    })
    .await
    .unwrap();
    ChatLlm::new(Llm::new(client), renderer, tokenizer)
}

struct FakeTemplateTokenizer {
    chat_template: ChatTemplateState,
    special_tokens: SpecialTokens,
}

impl FakeTemplateTokenizer {
    fn new(template: &str) -> anyhow::Result<Self> {
        Ok(Self {
            chat_template: ChatTemplateState::new(Some(template.to_string()))?,
            special_tokens: SpecialTokens::default(),
        })
    }

    fn empty() -> Self {
        Self {
            chat_template: ChatTemplateState::empty(),
            special_tokens: SpecialTokens::default(),
        }
    }
}

impl Encoder for FakeTemplateTokenizer {
    fn encode(&self, _input: &str, _add_special_tokens: bool) -> anyhow::Result<Encoding> {
        Ok(Encoding::Plain(Vec::new()))
    }

    fn encode_batch(
        &self,
        inputs: &[&str],
        _add_special_tokens: bool,
    ) -> anyhow::Result<Vec<Encoding>> {
        Ok(inputs.iter().map(|_| Encoding::Plain(Vec::new())).collect())
    }
}

impl Decoder for FakeTemplateTokenizer {
    fn decode(&self, _token_ids: &[u32], _skip_special_tokens: bool) -> anyhow::Result<String> {
        Ok(String::new())
    }
}

impl TokenizerTrait for FakeTemplateTokenizer {
    fn vocab_size(&self) -> usize {
        0
    }

    fn get_special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }

    fn token_to_id(&self, _token: &str) -> Option<u32> {
        None
    }

    fn id_to_token(&self, _id: u32) -> Option<String> {
        None
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn apply_chat_template(
        &self,
        messages: &[serde_json::Value],
        params: smg_tokenizer::chat_template::ChatTemplateParams,
    ) -> anyhow::Result<String> {
        self.chat_template.apply(messages, params)
    }

    fn chat_template_content_format(
        &self,
    ) -> smg_tokenizer::chat_template::ChatTemplateContentFormat {
        self.chat_template.content_format()
    }
}

#[derive(Debug)]
struct ByteTokenizer;

impl Tokenizer for ByteTokenizer {
    fn encode(&self, text: &str, _add_special_tokens: bool) -> vllm_chat::Result<Vec<u32>> {
        Ok(text.bytes().map(u32::from).collect())
    }

    fn decode(&self, token_ids: &[u32], _skip_special_tokens: bool) -> vllm_chat::Result<String> {
        let bytes = token_ids.iter().map(|id| *id as u8).collect::<Vec<_>>();
        Ok(String::from_utf8(bytes).unwrap())
    }
}

fn sample_request(request_id: &str) -> ChatRequest {
    ChatRequest {
        request_id: request_id.to_string(),
        messages: vec![
            ChatMessage {
                role: ChatRole::System,
                content: "You are terse.".to_string(),
            },
            ChatMessage {
                role: ChatRole::User,
                content: "Say hi".to_string(),
            },
        ],
        sampling_params: SamplingParams {
            max_tokens: Some(8),
            ..Default::default()
        },
        chat_options: ChatOptions::default(),
        cache_salt: Some("salt".to_string()),
        trace_headers: None,
        priority: 2,
        data_parallel_rank: Some(3),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_streams_text_events() {
    let handshake_address = unique_tcp_endpoint();
    let engine_identity = b"engine-chat".to_vec();

    let engine_task = tokio::spawn({
        let engine_handshake = handshake_address.clone();
        let engine_identity = engine_identity.clone();
        async move {
            let (mut dealer, mut push) = setup_mock_engine(engine_handshake, engine_identity).await;

            let add = recv_engine_message(&mut dealer).await;
            assert_eq!(add[0].as_ref(), &[0x00]);
            let request: EngineCoreRequest = rmp_serde::from_slice(&add[1]).unwrap();
            assert_eq!(request.request_id, "chat-1");
            assert_eq!(request.cache_salt.as_deref(), Some("salt"));
            assert_eq!(request.priority, 2);
            assert_eq!(request.data_parallel_rank, Some(3));
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
            assert_eq!(
                request.sampling_params.unwrap().output_kind,
                vllm_engine_core_client::protocol::RequestOutputKind::Cumulative
            );

            send_outputs(
                &mut push,
                EngineCoreOutputs {
                    outputs: vec![
                        request_output("chat-1", vec![b'H' as u32], None),
                        request_output(
                            "chat-1",
                            vec![b'i' as u32, b'!' as u32],
                            Some(FinishReason::Stop),
                        ),
                    ],
                    finished_requests: Some(BTreeSet::from(["chat-1".to_string()])),
                    ..Default::default()
                },
            )
            .await;
        }
    });

    let renderer: Arc<dyn ChatRenderer> = Arc::new(SmgTokenizerChatRenderer::new(
        SmgTokenizer::from_arc(Arc::new(
            FakeTemplateTokenizer::new(
                "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}",
            )
            .unwrap(),
        )),
    ));
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(ByteTokenizer);
    let chat = connect_chat_llm(handshake_address, renderer, tokenizer).await;

    let mut stream = chat.chat(sample_request("chat-1")).await.unwrap();

    assert_eq!(stream.next().await.unwrap().unwrap(), ChatEvent::Start);
    assert_eq!(
        stream.next().await.unwrap().unwrap(),
        ChatEvent::TextDelta {
            delta: "H".to_string(),
            text: "H".to_string(),
        }
    );
    assert_eq!(
        stream.next().await.unwrap().unwrap(),
        ChatEvent::TextDelta {
            delta: "i!".to_string(),
            text: "Hi!".to_string(),
        }
    );

    match stream.next().await {
        Some(Ok(ChatEvent::Done {
            text,
            finish_reason,
            ..
        })) => {
            assert_eq!(text, "Hi!");
            assert_eq!(finish_reason, Some(FinishReason::Stop));
        }
        other => panic!("unexpected final event: {other:?}"),
    }
    assert!(stream.next().await.is_none());

    chat.shutdown().await.unwrap();
    engine_task.await.unwrap();
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
fn renderer_requires_a_template() {
    let request = sample_request("chat-3");
    let renderer = SmgTokenizerChatRenderer::new(SmgTokenizer::from_arc(Arc::new(
        FakeTemplateTokenizer::empty(),
    )));
    let error = renderer.render(&request).unwrap_err();
    assert!(matches!(error, vllm_chat::Error::MissingChatTemplate));
}

#[derive(Debug)]
struct FailingDecodeTokenizer;

impl Tokenizer for FailingDecodeTokenizer {
    fn encode(&self, text: &str, _add_special_tokens: bool) -> vllm_chat::Result<Vec<u32>> {
        Ok(text.bytes().map(u32::from).collect())
    }

    fn decode(&self, _token_ids: &[u32], _skip_special_tokens: bool) -> vllm_chat::Result<String> {
        Err(vllm_chat::Error::Tokenizer("decode failed".to_string()))
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_stream_reports_decode_failure_as_error_event() {
    let handshake_address = unique_tcp_endpoint();
    let engine_identity = b"engine-chat-decode-fail".to_vec();

    let engine_task = tokio::spawn({
        let engine_handshake = handshake_address.clone();
        let engine_identity = engine_identity.clone();
        async move {
            let (mut dealer, mut push) = setup_mock_engine(engine_handshake, engine_identity).await;
            let _ = recv_engine_message(&mut dealer).await;
            send_outputs(
                &mut push,
                EngineCoreOutputs {
                    outputs: vec![request_output("chat-4", vec![b'X' as u32], None)],
                    ..Default::default()
                },
            )
            .await;
        }
    });

    let renderer: Arc<dyn ChatRenderer> = Arc::new(SmgTokenizerChatRenderer::new(
        SmgTokenizer::from_arc(Arc::new(
            FakeTemplateTokenizer::new("{{ messages[0].role }}: {{ messages[0].content }}")
                .unwrap(),
        )),
    ));
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(FailingDecodeTokenizer);
    let chat = connect_chat_llm(handshake_address, renderer, tokenizer).await;

    let mut stream = chat.chat(sample_request("chat-4")).await.unwrap();
    assert_eq!(stream.request_id(), "chat-4");
    assert!(matches!(stream.next().await, Some(Ok(ChatEvent::Start))));

    match timeout(Duration::from_secs(2), stream.next())
        .await
        .unwrap()
    {
        Some(Err(vllm_chat::Error::Tokenizer(message))) => {
            assert_eq!(message, "decode failed");
        }
        other => panic!("unexpected event after close: {other:?}"),
    }

    chat.shutdown().await.unwrap();
    engine_task.await.unwrap();
}
