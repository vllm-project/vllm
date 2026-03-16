use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::StreamExt as _;
use openai_protocol::chat::ChatCompletionRequest;
use openai_protocol::models::{ListModelsResponse, ModelObject};
use openai_protocol::validated::ValidatedJson;
use tracing::{error, info};
use vllm_chat::{ChatEvent, ChatEventStream};

use crate::convert::{final_chunk, prepare_chat_request, start_chunk, text_chunk};
use crate::error::ApiError;
use crate::state::AppState;

pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
}

async fn list_models(State(state): State<Arc<AppState>>) -> Json<ListModelsResponse> {
    Json(ListModelsResponse {
        object: "list".to_string(),
        data: vec![ModelObject {
            id: state.model_id.clone(),
            object: "model".to_string(),
            created: 0,
            owned_by: "vllm-frontend-rs".to_string(),
        }],
    })
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    ValidatedJson(body): ValidatedJson<ChatCompletionRequest>,
) -> Response {
    let prepared = match prepare_chat_request(&body, &state.model_id) {
        Ok(prepared) => prepared,
        Err(error) => return error.into_response(),
    };

    let response_id = prepared.response_id.clone();
    let response_model = prepared.response_model.clone();
    let created = unix_timestamp();
    info!(request_id = %response_id, model = %response_model, "streaming chat completion");

    let chat_stream = match state.chat.chat(prepared.chat_request).await {
        Ok(stream) => stream,
        Err(error) => {
            return ApiError::server_error(format!("failed to submit chat request: {error}"))
                .into_response();
        }
    };
    let sse_stream = event_stream(chat_stream, response_id, response_model, created);

    Sse::new(sse_stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

fn event_stream(
    stream: ChatEventStream,
    response_id: String,
    response_model: String,
    created: u64,
) -> impl futures::Stream<Item = Result<Event, Infallible>> + Send {
    futures::stream::unfold(
        Some(SseState::Start {
            stream,
            response_id,
            response_model,
            created,
        }),
        |state| async move {
            let state = state?;
            next_sse_item(state).await
        },
    )
}

enum SseState {
    Start {
        stream: ChatEventStream,
        response_id: String,
        response_model: String,
        created: u64,
    },
    Stream {
        stream: ChatEventStream,
        response_id: String,
        response_model: String,
        created: u64,
    },
    DoneMarker,
}

async fn next_sse_item(state: SseState) -> Option<(Result<Event, Infallible>, Option<SseState>)> {
    match state {
        SseState::Start {
            stream,
            response_id,
            response_model,
            created,
        } => {
            let chunk = start_chunk(&response_id, &response_model, created);
            Some((
                Ok(to_sse_event(&chunk)),
                Some(SseState::Stream {
                    stream,
                    response_id,
                    response_model,
                    created,
                }),
            ))
        }
        SseState::Stream {
            mut stream,
            response_id,
            response_model,
            created,
        } => loop {
            match stream.next().await {
                Some(Ok(ChatEvent::Start)) => continue,
                Some(Ok(ChatEvent::TextDelta { delta, .. })) => {
                    let chunk = text_chunk(&response_id, &response_model, created, delta);
                    return Some((
                        Ok(to_sse_event(&chunk)),
                        Some(SseState::Stream {
                            stream,
                            response_id,
                            response_model,
                            created,
                        }),
                    ));
                }
                Some(Ok(ChatEvent::Done {
                    finish_reason: Some(finish_reason),
                    stop_reason,
                    ..
                })) => match final_chunk(
                    &response_id,
                    &response_model,
                    created,
                    finish_reason,
                    stop_reason,
                ) {
                    Ok(chunk) => {
                        return Some((Ok(to_sse_event(&chunk)), Some(SseState::DoneMarker)));
                    }
                    Err(error) => {
                        error!(request_id = %response_id, ?error, "invalid terminal finish reason");
                        return None;
                    }
                },
                Some(Ok(ChatEvent::Done { .. })) => {
                    error!(request_id = %response_id, "missing terminal finish reason");
                    return None;
                }
                Some(Err(error)) => {
                    error!(request_id = %response_id, ?error, "chat stream failed");
                    return None;
                }
                None => return None,
            }
        },
        SseState::DoneMarker => Some((Ok(Event::default().data("[DONE]")), None)),
    }
}

fn to_sse_event(chunk: &openai_protocol::chat::ChatCompletionStreamResponse) -> Event {
    let payload =
        serde_json::to_string(chunk).expect("ChatCompletionStreamResponse must serialize to JSON");
    Event::default().data(payload)
}

#[cfg(test)]
mod tests {
    use std::convert::TryFrom;
    use std::net::TcpListener;
    use std::sync::Arc;
    use std::time::Duration;

    use axum::body::{Body, to_bytes};
    use axum::http::{Request, StatusCode};
    use bytes::Bytes;
    use futures::StreamExt as _;
    use serde_json::json;
    use tower::util::ServiceExt as _;
    use vllm_chat::{
        ChatBackend, ChatEvent, ChatLlm, ChatMessage, ChatRequest, ChatRole, UserSamplingParams,
    };
    use vllm_engine_core_client::protocol::handshake::{HandshakeInitMessage, ReadyMessage};
    use vllm_engine_core_client::protocol::{
        EngineCoreOutput, EngineCoreOutputs, EngineCoreRequest, FinishReason,
    };
    use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig};
    use vllm_llm::Llm;
    use zeromq::prelude::{Socket, SocketRecv, SocketSend};
    use zeromq::util::PeerIdentity;
    use zeromq::{DealerSocket, PushSocket, SocketOptions, ZmqMessage};

    use super::build_router;
    use crate::convert::prepare_chat_request;
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
    struct FakeChatBackend;

    impl ChatBackend for FakeChatBackend {
        fn apply_chat_template(&self, request: &ChatRequest) -> vllm_chat::Result<String> {
            let mut prompt = String::new();
            for message in &request.messages {
                prompt.push_str(message.role.as_str());
                prompt.push_str(": ");
                prompt.push_str(&message.content.try_flatten_to_text()?);
                prompt.push('\n');
            }
            if request.chat_options.add_generation_prompt {
                prompt.push_str("assistant:");
            }
            Ok(prompt)
        }

        fn encode(&self, text: &str) -> vllm_chat::Result<Vec<u32>> {
            Ok(text.bytes().map(u32::from).collect())
        }

        fn decode(
            &self,
            token_ids: &[u32],
            _skip_special_tokens: bool,
        ) -> vllm_chat::Result<String> {
            Ok(
                String::from_utf8_lossy(&token_ids.iter().map(|id| *id as u8).collect::<Vec<_>>())
                    .into_owned(),
            )
        }
    }

    async fn test_app() -> axum::Router {
        let handshake_address = unique_tcp_endpoint();
        let engine_identity = b"engine-openai".to_vec();

        #[expect(
            clippy::disallowed_methods,
            reason = "test harness background mock engine"
        )]
        tokio::spawn({
            let engine_handshake = handshake_address.clone();
            let engine_identity = engine_identity.clone();
            async move {
                let (mut dealer, mut push) =
                    setup_mock_engine(engine_handshake, engine_identity).await;
                let add = recv_engine_message(&mut dealer).await;
                let request: EngineCoreRequest =
                    rmp_serde::from_slice(&add[1]).expect("decode request");
                let response_id = request.request_id.clone();
                send_outputs(
                    &mut push,
                    EngineCoreOutputs {
                        engine_index: 0,
                        outputs: vec![
                            request_output(&response_id, vec![b'h' as u32], None),
                            request_output(&response_id, vec![b'i' as u32], None),
                            request_output(
                                &response_id,
                                vec![b'!' as u32],
                                Some(FinishReason::Stop),
                            ),
                        ],
                        scheduler_stats: None,
                        timestamp: 0.0,
                        utility_output: None,
                        finished_requests: None,
                        wave_complete: None,
                        start_wave: None,
                    },
                )
                .await;
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        });

        let client = EngineCoreClient::connect(EngineCoreClientConfig {
            handshake_address,
            local_host: "127.0.0.1".to_string(),
            ready_timeout: Duration::from_secs(2),
            client_index: 0,
        })
        .await
        .expect("connect client");

        let chat = Arc::new(ChatLlm::new(Llm::new(client), Arc::new(FakeChatBackend)));
        build_router(Arc::new(AppState::new("Qwen/Qwen1.5-0.5B-Chat", chat)))
    }

    async fn test_app_with_engine_handle() -> (axum::Router, tokio::task::JoinHandle<()>) {
        let handshake_address = unique_tcp_endpoint();
        let engine_identity = b"engine-openai".to_vec();

        let engine_task = tokio::spawn({
            let engine_handshake = handshake_address.clone();
            let engine_identity = engine_identity.clone();
            async move {
                let (mut dealer, mut push) =
                    setup_mock_engine(engine_handshake, engine_identity).await;
                let add = recv_engine_message(&mut dealer).await;
                let request: EngineCoreRequest =
                    rmp_serde::from_slice(&add[1]).expect("decode request");
                let response_id = request.request_id.clone();
                send_outputs(
                    &mut push,
                    EngineCoreOutputs {
                        engine_index: 0,
                        outputs: vec![
                            request_output(&response_id, vec![b'h' as u32], None),
                            request_output(&response_id, vec![b'i' as u32], None),
                            request_output(
                                &response_id,
                                vec![b'!' as u32],
                                Some(FinishReason::Stop),
                            ),
                        ],
                        scheduler_stats: None,
                        timestamp: 0.0,
                        utility_output: None,
                        finished_requests: None,
                        wave_complete: None,
                        start_wave: None,
                    },
                )
                .await;
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        });

        let client = EngineCoreClient::connect(EngineCoreClientConfig {
            handshake_address,
            local_host: "127.0.0.1".to_string(),
            ready_timeout: Duration::from_secs(2),
            client_index: 0,
        })
        .await
        .expect("connect client");

        let chat = Arc::new(ChatLlm::new(Llm::new(client), Arc::new(FakeChatBackend)));
        (
            build_router(Arc::new(AppState::new("Qwen/Qwen1.5-0.5B-Chat", chat))),
            engine_task,
        )
    }

    async fn test_chat_with_engine_handle() -> (Arc<ChatLlm>, tokio::task::JoinHandle<()>) {
        let handshake_address = unique_tcp_endpoint();
        let engine_identity = b"engine-openai-chat".to_vec();

        let engine_task = tokio::spawn({
            let engine_handshake = handshake_address.clone();
            let engine_identity = engine_identity.clone();
            async move {
                let (mut dealer, mut push) =
                    setup_mock_engine(engine_handshake, engine_identity).await;
                let add = recv_engine_message(&mut dealer).await;
                let request: EngineCoreRequest =
                    rmp_serde::from_slice(&add[1]).expect("decode request");
                let response_id = request.request_id.clone();
                send_outputs(
                    &mut push,
                    EngineCoreOutputs {
                        engine_index: 0,
                        outputs: vec![
                            request_output(&response_id, vec![b'h' as u32], None),
                            request_output(&response_id, vec![b'i' as u32], None),
                            request_output(
                                &response_id,
                                vec![b'!' as u32],
                                Some(FinishReason::Stop),
                            ),
                        ],
                        scheduler_stats: None,
                        timestamp: 0.0,
                        utility_output: None,
                        finished_requests: None,
                        wave_complete: None,
                        start_wave: None,
                    },
                )
                .await;
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        });

        let client = EngineCoreClient::connect(EngineCoreClientConfig {
            handshake_address,
            local_host: "127.0.0.1".to_string(),
            ready_timeout: Duration::from_secs(2),
            client_index: 0,
        })
        .await
        .expect("connect client");

        (
            Arc::new(ChatLlm::new(Llm::new(client), Arc::new(FakeChatBackend))),
            engine_task,
        )
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
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
    async fn happy_path_returns_sse_stream() {
        let (app, engine_task) = test_app_with_engine_handle().await;
        let response = app
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

        assert!(text.contains("\"role\":\"assistant\""));
        assert!(text.starts_with("data: "));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn chat_harness_streams_text_events() {
        let (chat, engine_task) = test_chat_with_engine_handle().await;
        let mut stream = chat
            .chat(ChatRequest {
                request_id: "chat-harness".to_string(),
                messages: vec![ChatMessage::text(ChatRole::User, "hello")],
                sampling_params: UserSamplingParams {
                    max_tokens: 8,
                    ..Default::default()
                },
                chat_options: Default::default(),
            })
            .await
            .expect("submit chat request");

        let mut saw_text = false;
        let mut saw_done = false;
        while let Some(event) = stream.next().await {
            match event.expect("chat event") {
                ChatEvent::TextDelta { .. } => saw_text = true,
                ChatEvent::Done { .. } => {
                    saw_done = true;
                    break;
                }
                ChatEvent::Start => {}
            }
        }
        engine_task.await.expect("mock engine task");

        assert!(saw_text);
        assert!(saw_done);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
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
                ChatEvent::TextDelta { .. } => saw_text = true,
                ChatEvent::Done { .. } => {
                    saw_done = true;
                    break;
                }
                ChatEvent::Start => {}
            }
        }
        engine_task.await.expect("mock engine task");

        assert!(saw_text);
        assert!(saw_done);
    }
}
