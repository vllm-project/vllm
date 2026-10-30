// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use serde_json::json;
use vllm_text::{Prompt, SamplingHints, TextBackend, TextRequest, TextRequestProcessor};
use vllm_tokenizer::test_utils::TestTokenizer;

use super::stream::{output_logprobs, request_body};
use super::*;
use futures::StreamExt as _;
use vllm_llm::{FinishReason, Llm};

struct Backend;

impl TextBackend for Backend {
    fn model_id(&self) -> &str {
        "test-model"
    }
    fn tokenizer(&self) -> vllm_tokenizer::DynTokenizer {
        Arc::new(TestTokenizer::new())
    }
    fn sampling_hints(&self) -> vllm_text::Result<SamplingHints> {
        Ok(SamplingHints {
            default_temperature: Some(0.7),
            default_max_tokens: Some(12),
            primary_eos_token_id: Some(2),
            ..Default::default()
        })
    }
}

fn prepared() -> vllm_llm::GenerateRequest {
    let mut request = TextRequest {
        request_id: "chat-test".into(),
        prompt: Prompt::TokenIds(vec![10, 20]),
        ..TextRequest::for_test()
    };
    request.sampling_params.max_tokens = None;
    request.sampling_params.ignore_eos = true;
    request.sampling_params.vllm_xargs = Some(
        [
            ("custom".into(), json!(7)),
            ("kv_transfer_params".into(), json!({"remote_decode": true})),
        ]
        .into(),
    );
    TextRequestProcessor::new(Arc::new(Backend), 128)
        .prepare(request)
        .unwrap()
        .generate_request
}

#[test]
fn generation_url_joins_generate_path_and_rejects_non_base_urls() {
    assert_eq!(
        generate_url("http://gateway:8080").unwrap().as_str(),
        "http://gateway:8080/inference/v1/generate"
    );
    assert_eq!(
        generate_url("https://gateway/prefix/").unwrap().as_str(),
        "https://gateway/prefix/inference/v1/generate"
    );
    for url in [
        "ftp://gateway",
        "gateway:8080",
        "http://gateway?x=1",
        "http://gateway#frag",
    ] {
        assert!(generate_url(url).is_err(), "{url}");
    }
}

#[test]
fn http_request_preserves_resolved_defaults_and_public_sampling_semantics() {
    let p = prepared();
    let body = request_body("test-model", &engine_request(p)).unwrap();
    assert_eq!(body["token_ids"], json!([10, 20]));
    assert_eq!(body["sampling_params"]["max_tokens"], 12);
    assert!((body["sampling_params"]["temperature"].as_f64().unwrap() - 0.7).abs() < 1e-6);
    assert_eq!(body["sampling_params"]["ignore_eos"], true);
    assert!(body["sampling_params"].get("bad_words_token_ids").is_none());
    assert_eq!(body["sampling_params"]["extra_args"], json!({"custom": 7}));
    assert_eq!(body["kv_transfer_params"], json!({"remote_decode": true}));
    assert!(body["sampling_params"].get("_eos_token_id").is_none());
    // The Rust HTTP endpoint accepts the same Python-compatible extension spelling.
    let wire: crate::routes::inference::generate::types::GenerateRequest =
        serde_json::from_value(body).unwrap();
    assert_eq!(wire.sampling_params.inner.vllm_xargs.unwrap()["custom"], 7);
}

#[test]
fn logprobs_keep_sampled_token_first_without_duplicating_it() {
    let wire = serde_json::from_value(json!({"content": [{
        "token_id": 20, "logprob": -1.7, "rank": 2,
        "top_logprobs": [
            {"token_id": 10, "logprob": -0.3, "rank": 1},
            {"token_id": 20, "logprob": -1.7, "rank": 2}
        ]
    }]}))
    .unwrap();
    let probabilities = output_logprobs(wire);
    let entries = &probabilities.positions[0].entries;
    assert_eq!(
        entries.iter().map(|p| (p.token_id, p.logprob, p.rank)).collect::<Vec<_>>(),
        vec![(20, -1.7, 2), (10, -0.3, 1)]
    );
}

#[test]
fn selective_logprobs_are_rejected_before_submission() {
    let mut request = TextRequest::for_test();
    request.prompt = Prompt::TokenIds(vec![10, 20]);
    request.sampling_params.logprob_token_ids = Some(vec![30, 40]);
    let p = TextRequestProcessor::new(Arc::new(Backend), 128)
        .prepare(request)
        .unwrap()
        .generate_request;
    let Err(Error::Http {
        status: 400,
        message,
    }) = request_body("test-model", &engine_request(p))
    else {
        panic!("expected selective logprobs rejection")
    };
    assert!(message.contains("selective logprobs"), "{message}");
}

async fn endpoint(
    body: axum::body::Body,
    status: u16,
) -> (
    String,
    tokio::sync::oneshot::Receiver<Value>,
    tokio::task::JoinHandle<()>,
) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let (sent, received) = tokio::sync::oneshot::channel();
    let response = Arc::new(Mutex::new(Some((sent, body))));
    let app = discovery_routes().route(
        "/inference/v1/generate",
        axum::routing::post(move |axum::Json(request): axum::Json<Value>| {
            let (sent, body) = response.lock().unwrap().take().unwrap();
            async move {
                sent.send(request).unwrap();
                axum::http::Response::builder()
                    .status(status)
                    .header("content-type", "text/event-stream")
                    .body(body)
                    .unwrap()
            }
        }),
    );
    let task = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (url, received, task)
}

fn fragmented_body(data: &str) -> axum::body::Body {
    let pieces: Vec<_> = data
        .as_bytes()
        .chunks(3)
        .map(|bytes| Ok::<_, std::convert::Infallible>(bytes.to_vec()))
        .collect();
    axum::body::Body::from_stream(futures::stream::iter(pieces))
}

#[tokio::test]
async fn fragmented_stream_keeps_final_usage_and_token_metadata() {
    let data = concat!(
        "data: {\"choices\":[{\"index\":0,\"token_ids\":[30]}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"token_ids\":[40],\"finish_reason\":\"stop\",\"stop_reason\":2,\"weight_versions\":[{\"version\":\"v1\",\"start\":0,\"end\":1},{\"version\":\"v2\",\"start\":1,\"end\":2}]}],\"kv_transfer_params\":{\"remote_decode\":true}}\n\n",
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens_details\":{\"cached_tokens\":1}}}\n\n",
        "data: [DONE]\n\n",
    );
    let (url, received, server) = endpoint(fragmented_body(data), 200).await;
    let client = EngineHTTPClient::connect(&url, "test-model".into(), 128).await.unwrap();
    let active = client.active.clone();
    let llm = Llm::new(client).with_request_id_randomization(false);
    let p = prepared();
    let mut stream = llm.generate(p).await.unwrap();
    let first = stream.next().await.unwrap().unwrap();
    assert_eq!(first.token_ids, vec![30]);
    assert_eq!(&*first.prompt_info.unwrap().prompt_token_ids, &[10, 20]);
    let last = stream.next().await.unwrap().unwrap();
    assert_eq!(last.token_ids, vec![40]);
    assert!(matches!(last.finish_reason, Some(FinishReason::Stop(_))));
    assert_eq!(last.prompt_cache_usage.cached_token_count, 1);
    assert_eq!(
        last.weight_versions,
        Some(vec![
            vllm_llm::WeightVersionSpan {
                version: "v1".into(),
                start: 0,
                end: 1
            },
            vllm_llm::WeightVersionSpan {
                version: "v2".into(),
                start: 1,
                end: 2
            },
        ])
    );
    assert_eq!(
        last.kv_transfer_params.as_deref(),
        Some(&json!({"remote_decode": true}))
    );
    assert!(stream.next().await.is_none());
    let request = received.await.unwrap();
    assert_eq!(request["output_mode"], "tokens");
    assert_eq!(request["stream"], true);
    assert!(active.lock().unwrap().is_empty());
    server.abort();
}

#[tokio::test]
async fn stream_truncation_and_upstream_errors_are_not_success() {
    for data in [
        "data: {\"choices\":[{\"index\":0,\"token_ids\":[],\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]\n\n",
        "data: {\"error\":{\"message\":\"engine failed\"}}\n\n",
    ] {
        let (url, received, server) = endpoint(fragmented_body(data), 200).await;
        let client = EngineHTTPClient::connect(&url, "test-model".into(), 128).await.unwrap();
        let llm = Llm::new(client).with_request_id_randomization(false);
        let p = prepared();
        let mut stream = llm.generate(p).await.unwrap();
        let error = stream.next().await.unwrap().unwrap_err();
        assert_eq!(http_error(&error).0, 502);
        if data.contains("engine failed") {
            assert!(http_error(&error).1.contains("engine failed"));
        }
        received.await.unwrap();
        server.abort();
    }
    let (url, received, server) = endpoint(axum::body::Body::from("overloaded"), 429).await;
    let client = EngineHTTPClient::connect(&url, "test-model".into(), 128).await.unwrap();
    let llm = Llm::new(client).with_request_id_randomization(false);
    let p = prepared();
    let Err(error) = llm.generate(p).await else {
        panic!("expected rejection")
    };
    let error =
        crate::error::chat_submit_error("chat failed", vllm_chat::Error::Text(error.into()));
    assert_eq!(
        error.status_code(),
        axum::http::StatusCode::TOO_MANY_REQUESTS
    );
    received.await.unwrap();
    server.abort();
}

#[tokio::test]
async fn upstream_validation_preserves_error_details() {
    let body = json!({"error": {
        "message": "top_p must be within (0, 1]", "type": "BadRequestError",
        "param": "top_p", "code": 400
    }});
    let (url, received, server) = endpoint(axum::body::Body::from(body.to_string()), 400).await;
    let client = EngineHTTPClient::connect(&url, "test-model".into(), 128).await.unwrap();
    let llm = Llm::new(client).with_request_id_randomization(false);
    let p = prepared();
    let Err(error) = llm.generate(p).await else {
        panic!("expected validation failure")
    };
    let error = crate::error::text_submit_error("submit", error.into());
    assert_eq!(error.status_code(), axum::http::StatusCode::BAD_REQUEST);
    let response = error.to_error_response();
    assert_eq!(response.error.message, "top_p must be within (0, 1]");
    assert_eq!(response.error.error_type, "BadRequestError");
    assert_eq!(response.error.param.as_deref(), Some("top_p"));
    assert_eq!(response.error.code, 400);
    received.await.unwrap();
    server.abort();
}

#[tokio::test]
async fn abort_and_stream_drop_close_the_upstream_connection() {
    struct Dropped(Option<tokio::sync::oneshot::Sender<()>>);
    impl Drop for Dropped {
        fn drop(&mut self) {
            if let Some(sent) = self.0.take() {
                let _ = sent.send(());
            }
        }
    }
    for abort in [false, true] {
        let (sent, dropped) = tokio::sync::oneshot::channel();
        let guard = Dropped(Some(sent));
        let first = futures::stream::once(async {
            Ok::<_, std::convert::Infallible>(
                "data: {\"choices\":[{\"index\":0,\"token_ids\":[30]}]}\n\n",
            )
        });
        let pending = futures::stream::once(async move {
            let _guard = guard;
            std::future::pending().await
        });
        let (url, received, server) =
            endpoint(axum::body::Body::from_stream(first.chain(pending)), 200).await;
        let client = EngineHTTPClient::connect(&url, "test-model".into(), 128).await.unwrap();
        let active = client.active.clone();
        let llm = Llm::new(client).with_request_id_randomization(false);
        let p = prepared();
        let mut stream = llm.generate(p).await.unwrap();
        assert_eq!(stream.next().await.unwrap().unwrap().token_ids, vec![30]);
        assert_eq!(active.lock().unwrap().len(), 1);
        if abort {
            llm.abort(&["chat-test".into()]).await.unwrap();
            let output = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
                .await
                .unwrap()
                .unwrap()
                .unwrap();
            assert!(matches!(output.finish_reason, Some(FinishReason::Abort)));
        }
        drop(stream);
        tokio::time::timeout(std::time::Duration::from_secs(5), dropped)
            .await
            .unwrap()
            .unwrap();
        assert!(active.lock().unwrap().is_empty());
        received.await.unwrap();
        server.abort();
    }
}

fn http_error(error: &vllm_llm::Error) -> (u16, &str) {
    let vllm_llm::Error::EngineCoreClient(Error::Http { status, message }) = error else {
        panic!("expected HTTP failure, got {error:?}")
    };
    (*status, message)
}

fn engine_request(request: vllm_llm::GenerateRequest) -> EngineCoreRequest {
    EngineCoreRequest {
        request_id: request.request_id,
        prompt_token_ids: Some(request.prompt_token_ids),
        sampling_params: Some(request.sampling_params),
        ..Default::default()
    }
}

pub(crate) fn discovery_routes() -> axum::Router {
    axum::Router::new()
        .route(
            "/server_info",
            axum::routing::get(|| async { axum::Json(server_info_response()) }),
        )
        .route(
            "/version",
            axum::routing::get(|| async { axum::Json(json!({"version": "test-vllm-version"})) }),
        )
        .route(
            "/kv_event_sources",
            axum::routing::get(|| async { axum::Json(json!({})) }),
        )
        .route(
            "/health",
            axum::routing::get(|| async { axum::http::StatusCode::OK }),
        )
}

pub(crate) fn server_info_response() -> Value {
    json!({"vllm_config": {
        "model_config": {"max_model_len": 1048576, "dtype": "torch.float32", "enable_sleep_mode": false},
        "parallel_config": {"data_parallel_size_local": 1, "data_parallel_size": 1, "data_parallel_rank": 0,
            "tensor_parallel_size": 1, "pipeline_parallel_size": 1, "decode_context_parallel_size": 1, "world_size": 1},
        "cache_config": {"num_gpu_blocks": 128, "block_size": 16, "kv_cache_size_tokens": null,
            "kv_cache_max_concurrency": null, "effective_attention_block_size": null},
        "scheduler_config": {"max_num_seqs": 256, "max_num_batched_tokens": 8192}, "instance_id": "test-instance",
        "lora_config": null, "weight_transfer_config": null
    }})
}

#[test]
fn pretokenized_bad_words_are_rejected_instead_of_silently_dropped() {
    let mut request = engine_request(prepared());
    request.sampling_params.as_mut().unwrap().bad_words_token_ids = Some(vec![vec![104, 105]]);
    assert!(matches!(
        request_body("test-model", &request),
        Err(Error::Http { status: 400, .. })
    ));
}

#[tokio::test]
async fn http_discovery_rejects_internal_dp_across_nodes() {
    let mut info = server_info_response();
    let parallel = &mut info["vllm_config"]["parallel_config"];
    parallel["data_parallel_size"] = json!(2);
    parallel["data_parallel_external_lb"] = json!(false);
    parallel["data_parallel_hybrid_lb"] = json!(false);
    // One local engine does not imply one managed engine in internal DP.
    let app = axum::Router::new()
        .route(
            "/server_info",
            axum::routing::get(move || {
                let info = info.clone();
                async move { axum::Json(info) }
            }),
        )
        .route(
            "/version",
            axum::routing::get(|| async { axum::Json(json!({"version": "test"})) }),
        )
        .route(
            "/kv_event_sources",
            axum::routing::get(|| async { axum::Json(json!({})) }),
        );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let Err(error) = EngineHTTPClient::connect(&url, "test-model".into(), 128).await else {
        panic!("aggregated metadata must not be treated as a single engine");
    };
    assert!(format!("{error:#}").contains("one managed DP engine"));
    server.abort();
}

#[tokio::test]
async fn controls_preserve_http_payloads_and_operation_outcomes() {
    use axum::http::{Method, Uri};
    use axum::response::IntoResponse as _;
    let (sent, mut received) = tokio::sync::mpsc::unbounded_channel();
    let app =
        discovery_routes().fallback(move |method: Method, uri: Uri, body: axum::body::Bytes| {
            let body: Value = if body.is_empty() {
                Value::Null
            } else {
                serde_json::from_slice(&body).unwrap()
            };
            sent.send((method, uri.clone(), body)).unwrap();
            async move {
                let (status, body) = match uri.path() {
                    "/reset_prefix_cache" => (200, json!({"success": false})),
                    "/wake_up" => (200, json!({"status": "sleeping"})),
                    "/is_sleeping" => (200, json!({"is_sleeping": true})),
                    "/is_paused" => (200, json!({"is_paused": false})),
                    "/weight_info" => (200, json!({"weight_version": "step-2"})),
                    "/collective_rpc" => (200, json!({"results": [null, "True", {"rank": 0}]})),
                    "/checkpoint/status" => (503, json!({"state": "preparing"})),
                    "/checkpoint/prepare" => {
                        (400, json!({"state": "running", "error": "still busy"}))
                    }
                    "/start_profile" => (409, json!({"error": "already profiling"})),
                    "/reset_mm_cache" => (503, json!({"error": "engine unavailable"})),
                    _ => (200, json!({})),
                };
                (
                    axum::http::StatusCode::from_u16(status).unwrap(),
                    axum::Json(body),
                )
                    .into_response()
            }
        });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let client = EngineHTTPClient::connect(&url, "test-model".into(), 128).await.unwrap();
    client.pause_generation(PauseMode::Keep, false).await.unwrap();
    assert!(!client.reset_prefix_cache(true, true).await.unwrap());
    client.sleep(SleepLevel::DiscardAll, PauseMode::Wait).await.unwrap();
    assert!(!client.wake_up(Some(vec!["weights".into(), "kv_cache".into()])).await.unwrap());
    assert_eq!(client.get_weight_version().await.unwrap(), "step-2");
    assert!(client.is_sleeping().await.unwrap());
    assert!(!client.is_scheduler_paused().await.unwrap());
    assert_eq!(
        client.checkpoint_state().await.unwrap(),
        (CheckpointState::Preparing, None)
    );
    assert_eq!(
        client.checkpoint_prepare(PauseMode::Wait).await.unwrap().as_deref(),
        Some("still busy")
    );
    assert!(matches!(
        client.start_profile(None, None, None, None).await,
        Err(Error::ProfileAlreadyActive)
    ));
    assert!(matches!(
        client.reset_mm_cache().await,
        Err(Error::Http { status: 503, .. })
    ));
    let results = client
        .collective_rpc("method", Some(2.0), vec![json!(42)], BTreeMap::new())
        .await
        .unwrap();
    assert_eq!(
        results,
        vec![
            OpaqueValue::Nil,
            OpaqueValue::from("True"),
            rmpv::ext::to_value(json!({"rank": 0})).unwrap()
        ]
    );
    let lora = LoraRequest {
        lora_name: "adapter".into(),
        lora_path: "/shared/adapter".into(),
        lora_int_id: 17,
        base_model_name: None,
        load_inplace: false,
        is_3d_lora_weight: false,
    };
    assert!(client.add_lora(&lora).await.unwrap());
    assert!(client.remove_lora(17).await.unwrap());
    assert!(!client.remove_lora(999).await.unwrap());
    assert!(matches!(
        client.stop_profile(Some("prefix")).await,
        Err(Error::Http { status: 400, .. })
    ));
    let calls: Vec<_> = std::iter::from_fn(|| received.try_recv().ok()).collect();
    let uris: Vec<_> = calls.iter().map(|(_, uri, _)| uri.to_string()).collect();
    assert_eq!(
        uris,
        [
            "/pause?mode=keep&clear_cache=false",
            "/reset_prefix_cache?reset_running_requests=true&reset_external=true",
            "/sleep?level=2&mode=wait",
            "/wake_up?tags=weights&tags=kv_cache",
            "/weight_info",
            "/is_sleeping",
            "/is_paused",
            "/checkpoint/status",
            "/checkpoint/prepare?mode=wait",
            "/start_profile",
            "/reset_mm_cache",
            "/collective_rpc",
            "/v1/load_lora_adapter",
            "/v1/unload_lora_adapter",
        ]
    );
    assert_eq!(calls[4].0, Method::GET);
    assert_eq!(calls[12].2["lora_name"], "adapter");
    assert_eq!(calls[13].2, json!({"lora_name": "adapter"}));
    server.abort();
}
