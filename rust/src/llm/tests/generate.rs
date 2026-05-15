use std::collections::BTreeSet;
use std::sync::Once;
use std::time::Duration;

use futures::StreamExt as _;
use tokio::time::timeout;
use tracing_subscriber::EnvFilter;
use uuid::Uuid;
use vllm_engine_core_client::protocol::logprobs::{
    Logprobs, MaybeWireLogprobs, PositionLogprobs, TokenLogprob,
};
use vllm_engine_core_client::protocol::stats::PrefillStats;
use vllm_engine_core_client::protocol::{
    EngineCoreEvent, EngineCoreEventType, EngineCoreFinishReason, EngineCoreOutput,
    EngineCoreOutputs, EngineCoreRequest, EngineCoreSamplingParams,
};
use vllm_engine_core_client::test_utils::{IpcNamespace, spawn_mock_engine_task};
use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig};
use vllm_llm::{
    Error, FinishReason, GenerateOutputStreamExt as _, GeneratePromptInfo, GenerateRequest, Llm,
};
use vllm_metrics::METRICS;
use zeromq::prelude::{SocketRecv, SocketSend};
use zeromq::{DealerSocket, PushSocket, ZmqMessage};

static TRACING: Once = Once::new();

fn request_output(
    request_id: &str,
    new_token_ids: Vec<u32>,
    finish_reason: Option<EngineCoreFinishReason>,
) -> EngineCoreOutput {
    request_output_with_events(request_id, new_token_ids, finish_reason, None)
}

fn request_output_with_events(
    request_id: &str,
    new_token_ids: Vec<u32>,
    finish_reason: Option<EngineCoreFinishReason>,
    events: Option<Vec<EngineCoreEvent>>,
) -> EngineCoreOutput {
    EngineCoreOutput {
        request_id: request_id.to_string(),
        new_token_ids,
        new_logprobs: None,
        new_prompt_logprobs_tensors: None,
        pooling_output: None,
        finish_reason,
        stop_reason: None,
        events,
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
    new_logprobs: Option<Logprobs>,
    prompt_logprobs: Option<Logprobs>,
) -> EngineCoreOutput {
    EngineCoreOutput {
        request_id: request_id.to_string(),
        new_token_ids,
        new_logprobs: new_logprobs.map(MaybeWireLogprobs::Direct),
        new_prompt_logprobs_tensors: prompt_logprobs.map(MaybeWireLogprobs::Direct),
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

fn request_output_with_logprobs_and_kv(
    request_id: &str,
    new_token_ids: Vec<u32>,
    finish_reason: Option<EngineCoreFinishReason>,
    new_logprobs: Option<Logprobs>,
    prompt_logprobs: Option<Logprobs>,
    kv_transfer_params: Option<serde_json::Value>,
) -> EngineCoreOutput {
    EngineCoreOutput {
        request_id: request_id.to_string(),
        new_token_ids,
        new_logprobs: new_logprobs.map(MaybeWireLogprobs::Direct),
        new_prompt_logprobs_tensors: prompt_logprobs.map(MaybeWireLogprobs::Direct),
        pooling_output: None,
        finish_reason,
        stop_reason: None,
        events: None,
        kv_transfer_params,
        trace_headers: None,
        prefill_stats: None,
        routed_experts: None,
        num_nans_in_logits: 0,
    }
}

fn logprobs_for_position(
    sampled_token_id: u32,
    sampled_logprob: f32,
    sampled_rank: u32,
    top_token_id: u32,
    top_logprob: f32,
) -> Logprobs {
    Logprobs {
        positions: vec![PositionLogprobs {
            entries: vec![
                TokenLogprob {
                    token_id: sampled_token_id,
                    logprob: sampled_logprob,
                    rank: sampled_rank,
                },
                TokenLogprob {
                    token_id: top_token_id,
                    logprob: top_logprob,
                    rank: 1,
                },
            ],
        }],
    }
}

fn prompt_logprobs() -> Logprobs {
    Logprobs {
        positions: vec![
            PositionLogprobs {
                entries: vec![
                    TokenLogprob {
                        token_id: 11,
                        logprob: -0.1,
                        rank: 2,
                    },
                    TokenLogprob {
                        token_id: 7,
                        logprob: -0.05,
                        rank: 1,
                    },
                ],
            },
            PositionLogprobs {
                entries: vec![
                    TokenLogprob {
                        token_id: 22,
                        logprob: -0.2,
                        rank: 3,
                    },
                    TokenLogprob {
                        token_id: 8,
                        logprob: -0.1,
                        rank: 1,
                    },
                ],
            },
        ],
    }
}

fn sample_generate_request(request_id: &str, max_tokens: u32) -> GenerateRequest {
    GenerateRequest {
        request_id: request_id.to_string(),
        prompt_token_ids: vec![11, 22],
        sampling_params: EngineCoreSamplingParams {
            max_tokens,
            ..EngineCoreSamplingParams::for_test()
        },
        arrival_time: Some(42.5),
        cache_salt: None,
        trace_headers: None,
        priority: 0,
        data_parallel_rank: None,
        reasoning_ended: None,
        lora_request: None,
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

async fn connect_async_llm_with_ipc(
    handshake_address: String,
    client_index: u32,
    model_name: &str,
    ipc: &IpcNamespace,
) -> Llm {
    let client = EngineCoreClient::connect(
        EngineCoreClientConfig::new_single(handshake_address)
            .with_model_name(model_name)
            .with_client_index(client_index)
            .with_local_input_output_addresses(
                Some(ipc.input_endpoint()),
                Some(ipc.output_endpoint()),
            ),
    )
    .await
    .unwrap();
    Llm::new(client)
}

fn request_metrics_model_name(prefix: &str) -> String {
    format!("{prefix}-{}", Uuid::new_v4().simple())
}

fn init_tracing() {
    TRACING.call_once(|| {
        let filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("vllm_engine_core_client=debug"));
        let _ = tracing_subscriber::fmt().with_test_writer().with_env_filter(filter).try_init();
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn generate_streams_outputs() {
    init_tracing();
    let ipc = IpcNamespace::new().unwrap();
    let handshake_address = ipc.handshake_endpoint();
    let engine_id = b"engine-delta".to_vec();

    let (shutdown_tx, engine_task) = spawn_mock_engine_task(
        handshake_address.clone(),
        engine_id.clone(),
        |dealer, push| {
            Box::pin(async move {
                let add = recv_engine_message(dealer).await;
                assert_eq!(add[0].as_ref(), &[0x00]);
                let request: EngineCoreRequest = rmp_serde::from_slice(&add[1]).unwrap();
                assert_eq!(request.external_req_id.as_deref(), Some("req-delta"));
                assert!(request.request_id.starts_with("req-delta-"));
                assert_ne!(request.request_id, "req-delta");
                assert_eq!(request.client_index, 7);
                assert_eq!(request.prompt_token_ids, Some(vec![11, 22]));

                send_outputs(
                    push,
                    EngineCoreOutputs {
                        outputs: vec![
                            request_output_with_logprobs(
                                &request.request_id,
                                vec![1, 2],
                                None,
                                Some(logprobs_for_position(1, -0.3, 4, 9, -0.1)),
                                Some(prompt_logprobs()),
                            ),
                            request_output_with_logprobs(
                                &request.request_id,
                                vec![3],
                                Some(EngineCoreFinishReason::Length),
                                Some(logprobs_for_position(3, -0.4, 5, 10, -0.2)),
                                None,
                            ),
                        ],
                        finished_requests: Some(BTreeSet::from([request.request_id.clone()])),
                        ..Default::default()
                    },
                )
                .await;
            })
        },
    );

    let llm = connect_async_llm_with_ipc(handshake_address, 7, "test-model", &ipc).await;
    let mut stream = llm.generate(sample_generate_request("req-delta", 3)).await.unwrap();
    let internal_id = stream.request_id().to_string();

    let first = stream.next().await.unwrap().unwrap();
    assert_eq!(first.request_id, internal_id);
    assert_eq!(
        first.prompt_info,
        Some(GeneratePromptInfo {
            prompt_token_ids: vec![11, 22].into(),
            prompt_logprobs: Some(prompt_logprobs()),
        })
    );
    assert_eq!(first.token_ids, vec![1, 2]);
    assert_eq!(
        first.logprobs,
        Some(logprobs_for_position(1, -0.3, 4, 9, -0.1))
    );
    assert_eq!(first.finish_reason, None);

    let second = stream.next().await.unwrap().unwrap();
    assert_eq!(second.prompt_info, None);
    assert_eq!(second.token_ids, vec![3]);
    assert_eq!(
        second.logprobs,
        Some(logprobs_for_position(3, -0.4, 5, 10, -0.2))
    );
    assert_eq!(second.finish_reason, Some(FinishReason::Length));
    assert!(stream.next().await.is_none());

    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    llm.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn collect_output_aggregates_raw_tokens_logprobs_and_terminal_metadata() {
    init_tracing();
    let ipc = IpcNamespace::new().unwrap();
    let handshake_address = ipc.handshake_endpoint();
    let engine_id = b"engine-collect-output".to_vec();

    let (shutdown_tx, engine_task) = spawn_mock_engine_task(
        handshake_address.clone(),
        engine_id.clone(),
        |dealer, push| {
            Box::pin(async move {
                let add = recv_engine_message(dealer).await;
                let request: EngineCoreRequest = rmp_serde::from_slice(&add[1]).unwrap();
                assert_eq!(request.external_req_id.as_deref(), Some("req-collect"));
                assert!(request.request_id.starts_with("req-collect-"));

                send_outputs(
                    push,
                    EngineCoreOutputs {
                        engine_index: 0,
                        outputs: vec![
                            request_output_with_logprobs(
                                &request.request_id,
                                vec![33],
                                None,
                                Some(logprobs_for_position(33, -0.1, 1, 99, -0.2)),
                                Some(prompt_logprobs()),
                            ),
                            request_output_with_logprobs_and_kv(
                                &request.request_id,
                                vec![44],
                                Some(EngineCoreFinishReason::Stop),
                                Some(logprobs_for_position(44, -0.3, 1, 88, -0.4)),
                                None,
                                Some(serde_json::json!({"connector": "x"})),
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
            })
        },
    );

    let llm = connect_async_llm_with_ipc(handshake_address, 7, "test-model", &ipc).await;
    let stream = llm.generate(sample_generate_request("req-collect", 4)).await.unwrap();
    let internal_id = stream.request_id().to_string();
    let collected = stream.collect_output().await.unwrap();

    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();

    assert_eq!(collected.request_id, internal_id);
    assert_eq!(collected.prompt_token_ids, vec![11, 22]);
    assert_eq!(collected.token_ids, vec![33, 44]);
    assert_eq!(collected.finish_reason, FinishReason::stop_eos());
    assert_eq!(collected.prompt_logprobs, Some(prompt_logprobs()));
    assert_eq!(
        collected.logprobs.as_ref().map(|lp| lp.positions.len()),
        Some(2)
    );
    assert_eq!(
        collected.kv_transfer_params,
        Some(serde_json::json!({"connector": "x"}))
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn generate_propagates_unexpected_close_errors() {
    let ipc = IpcNamespace::new().unwrap();
    let handshake_address = ipc.handshake_endpoint();
    let engine_id = b"engine-close".to_vec();

    let (shutdown_tx, engine_task) = spawn_mock_engine_task(
        handshake_address.clone(),
        engine_id.clone(),
        |dealer, push| {
            Box::pin(async move {
                let add = recv_engine_message(dealer).await;
                assert_eq!(add[0].as_ref(), &[0x00]);
                let request: EngineCoreRequest = rmp_serde::from_slice(&add[1]).unwrap();

                send_outputs(
                    push,
                    EngineCoreOutputs {
                        finished_requests: Some(BTreeSet::from([request.request_id])),
                        ..Default::default()
                    },
                )
                .await;
            })
        },
    );

    let llm = connect_async_llm_with_ipc(handshake_address, 0, "test-model", &ipc).await;
    let mut stream = llm.generate(sample_generate_request("req-close", 1)).await.unwrap();
    let internal_id = stream.request_id().to_string();

    let error = stream.next().await.unwrap().unwrap_err();
    assert!(matches!(
        error,
        Error::EngineCoreClient(vllm_engine_core_client::Error::RequestStreamClosed {
            request_id
        }) if request_id == internal_id
    ));
    assert!(stream.next().await.is_none());

    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    llm.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dropping_a_live_generate_stream_triggers_abort() {
    let ipc = IpcNamespace::new().unwrap();
    let handshake_address = ipc.handshake_endpoint();
    let engine_id = b"engine-drop".to_vec();

    let (shutdown_tx, engine_task) = spawn_mock_engine_task(
        handshake_address.clone(),
        engine_id.clone(),
        |dealer, push| {
            Box::pin(async move {
                let add = recv_engine_message(dealer).await;
                assert_eq!(add[0].as_ref(), &[0x00]);
                let request: EngineCoreRequest = rmp_serde::from_slice(&add[1]).unwrap();
                assert_eq!(request.external_req_id.as_deref(), Some("req-drop"));
                assert!(request.request_id.starts_with("req-drop-"));

                send_outputs(
                    push,
                    EngineCoreOutputs {
                        outputs: vec![request_output(&request.request_id, vec![99], None)],
                        ..Default::default()
                    },
                )
                .await;

                let abort =
                    timeout(Duration::from_secs(1), recv_engine_message(dealer)).await.unwrap();
                assert_eq!(abort[0].as_ref(), &[0x01]);
                let aborted_ids: Vec<String> = rmp_serde::from_slice(&abort[1]).unwrap();
                assert_eq!(aborted_ids, vec![request.request_id]);
            })
        },
    );

    let llm = connect_async_llm_with_ipc(handshake_address, 0, "test-model", &ipc).await;
    let mut stream = llm.generate(sample_generate_request("req-drop", 4)).await.unwrap();

    let output = stream.next().await.unwrap().unwrap();
    assert_eq!(output.token_ids, vec![99]);
    drop(stream);

    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    llm.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn duplicate_external_request_ids_are_randomized_before_reaching_engine_core_client() {
    let ipc = IpcNamespace::new().unwrap();
    let handshake_address = ipc.handshake_endpoint();
    let engine_id = b"engine-dup".to_vec();

    let (shutdown_tx, engine_task) = spawn_mock_engine_task(
        handshake_address.clone(),
        engine_id.clone(),
        |dealer, push| {
            Box::pin(async move {
                let add_1 = recv_engine_message(dealer).await;
                assert_eq!(add_1[0].as_ref(), &[0x00]);
                let request_1: EngineCoreRequest = rmp_serde::from_slice(&add_1[1]).unwrap();
                assert_eq!(request_1.external_req_id.as_deref(), Some("req-dup"));
                assert!(request_1.request_id.starts_with("req-dup-"));

                let add_2 = recv_engine_message(dealer).await;
                assert_eq!(add_2[0].as_ref(), &[0x00]);
                let request_2: EngineCoreRequest = rmp_serde::from_slice(&add_2[1]).unwrap();
                assert_eq!(request_2.external_req_id.as_deref(), Some("req-dup"));
                assert!(request_2.request_id.starts_with("req-dup-"));
                assert_ne!(request_1.request_id, request_2.request_id);

                send_outputs(
                    push,
                    EngineCoreOutputs {
                        outputs: vec![request_output(
                            &request_1.request_id,
                            vec![],
                            Some(EngineCoreFinishReason::Length),
                        )],
                        finished_requests: Some(BTreeSet::from([request_1.request_id.clone()])),
                        ..Default::default()
                    },
                )
                .await;

                send_outputs(
                    push,
                    EngineCoreOutputs {
                        outputs: vec![request_output(
                            &request_2.request_id,
                            vec![],
                            Some(EngineCoreFinishReason::Length),
                        )],
                        finished_requests: Some(BTreeSet::from([request_2.request_id])),
                        ..Default::default()
                    },
                )
                .await;
            })
        },
    );

    let llm = connect_async_llm_with_ipc(handshake_address, 0, "test-model", &ipc).await;
    let stream_1 = llm.generate(sample_generate_request("req-dup", 1)).await.unwrap();
    let stream_2 = llm.generate(sample_generate_request("req-dup", 1)).await.unwrap();
    let internal_id_1 = stream_1.request_id().to_string();
    let internal_id_2 = stream_2.request_id().to_string();
    let collected_1 = stream_1.collect_output().await.unwrap();
    let collected_2 = stream_2.collect_output().await.unwrap();
    assert_eq!(collected_1.request_id, internal_id_1);
    assert_eq!(collected_2.request_id, internal_id_2);
    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    llm.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn generate_records_request_metrics_in_prometheus_output() {
    let ipc = IpcNamespace::new().unwrap();
    let handshake_address = ipc.handshake_endpoint();
    let engine_id = b"engine-metrics".to_vec();
    let model_name = request_metrics_model_name("metrics-model");

    let (shutdown_tx, engine_task) = spawn_mock_engine_task(
        handshake_address.clone(),
        engine_id.clone(),
        |dealer, push| {
            Box::pin(async move {
                let add = recv_engine_message(dealer).await;
                assert_eq!(add[0].as_ref(), &[0x00]);
                let request: EngineCoreRequest = rmp_serde::from_slice(&add[1]).unwrap();

                send_outputs(
                    push,
                    EngineCoreOutputs {
                        engine_index: 4,
                        timestamp: 10.0,
                        outputs: vec![EngineCoreOutput {
                            prefill_stats: Some(PrefillStats {
                                num_prompt_tokens: 2,
                                num_computed_tokens: 2,
                                ..Default::default()
                            }),
                            ..request_output_with_events(
                                &request.request_id,
                                vec![1],
                                None,
                                Some(vec![
                                    EngineCoreEvent {
                                        r#type: EngineCoreEventType::Queued,
                                        timestamp: 8.0,
                                    },
                                    EngineCoreEvent {
                                        r#type: EngineCoreEventType::Scheduled,
                                        timestamp: 9.0,
                                    },
                                ]),
                            )
                        }],
                        ..Default::default()
                    },
                )
                .await;

                send_outputs(
                    push,
                    EngineCoreOutputs {
                        engine_index: 4,
                        timestamp: 11.5,
                        outputs: vec![request_output_with_events(
                            &request.request_id,
                            vec![2, 3],
                            Some(EngineCoreFinishReason::Length),
                            Some(vec![EngineCoreEvent {
                                r#type: EngineCoreEventType::Preempted,
                                timestamp: 10.5,
                            }]),
                        )],
                        finished_requests: Some(BTreeSet::from([request.request_id])),
                        ..Default::default()
                    },
                )
                .await;
            })
        },
    );

    let llm = connect_async_llm_with_ipc(handshake_address, 0, &model_name, &ipc).await;
    let mut request = sample_generate_request("req-metrics", 8);
    request.arrival_time = None;
    let mut stream = llm.generate(request).await.unwrap();

    assert_eq!(stream.next().await.unwrap().unwrap().token_ids, vec![1]);
    let final_output = stream.next().await.unwrap().unwrap();
    assert_eq!(final_output.token_ids, vec![2, 3]);
    assert_eq!(final_output.finish_reason, Some(FinishReason::Length));
    assert!(stream.next().await.is_none());

    let rendered = METRICS.render().unwrap();
    assert!(rendered.contains(&format!(
        "vllm:request_success_total{{model_name=\"{model_name}\",engine=\"4\",finished_reason=\"length\"}} 1"
    )));
    assert!(rendered.contains(&format!(
        "vllm:prompt_tokens_total{{model_name=\"{model_name}\",engine=\"4\"}} 2"
    )));
    assert!(rendered.contains(&format!(
        "vllm:prompt_tokens_by_source_total{{model_name=\"{model_name}\",engine=\"4\",source=\"local_compute\"}} 2"
    )));
    assert!(rendered.contains(&format!(
        "vllm:prompt_tokens_by_source_total{{model_name=\"{model_name}\",engine=\"4\",source=\"local_cache_hit\"}} 0"
    )));
    assert!(rendered.contains(&format!(
        "vllm:prompt_tokens_by_source_total{{model_name=\"{model_name}\",engine=\"4\",source=\"external_kv_transfer\"}} 0"
    )));
    assert!(rendered.contains(&format!(
        "vllm:prompt_tokens_cached_total{{model_name=\"{model_name}\",engine=\"4\"}} 0"
    )));
    assert!(rendered.contains(&format!(
        "vllm:generation_tokens_total{{model_name=\"{model_name}\",engine=\"4\"}} 3"
    )));
    assert!(rendered.contains(&format!(
        "vllm:num_preemptions_total{{model_name=\"{model_name}\",engine=\"4\"}} 1"
    )));
    assert!(rendered.contains(&format!(
        "vllm:time_to_first_token_seconds_count{{model_name=\"{model_name}\",engine=\"4\"}} 1"
    )));
    assert!(rendered.contains(&format!(
        "vllm:inter_token_latency_seconds_count{{model_name=\"{model_name}\",engine=\"4\"}} 1"
    )));
    assert!(rendered.contains(&format!(
        "vllm:e2e_request_latency_seconds_count{{model_name=\"{model_name}\",engine=\"4\"}} 1"
    )));
    assert!(rendered.contains(&format!(
        "vllm:request_prompt_tokens_count{{model_name=\"{model_name}\",engine=\"4\"}} 1"
    )));
    assert!(rendered.contains(&format!(
        "vllm:request_generation_tokens_count{{model_name=\"{model_name}\",engine=\"4\"}} 1"
    )));
    assert!(rendered.contains(&format!(
        "vllm:request_prefill_kv_computed_tokens_count{{model_name=\"{model_name}\",engine=\"4\"}} 1"
    )));

    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    llm.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dropping_stream_records_abort_terminal_request_metrics() {
    let ipc = IpcNamespace::new().unwrap();
    let handshake_address = ipc.handshake_endpoint();
    let engine_id = b"engine-metrics-drop".to_vec();
    let model_name = request_metrics_model_name("metrics-drop-model");

    let (shutdown_tx, engine_task) = spawn_mock_engine_task(
        handshake_address.clone(),
        engine_id.clone(),
        |dealer, push| {
            Box::pin(async move {
                let add = recv_engine_message(dealer).await;
                assert_eq!(add[0].as_ref(), &[0x00]);
                let request: EngineCoreRequest = rmp_serde::from_slice(&add[1]).unwrap();
                assert_eq!(request.external_req_id.as_deref(), Some("req-metrics-drop"));
                assert!(request.request_id.starts_with("req-metrics-drop-"));

                send_outputs(
                    push,
                    EngineCoreOutputs {
                        engine_index: 5,
                        timestamp: 10.0,
                        outputs: vec![request_output_with_events(
                            &request.request_id,
                            vec![99],
                            None,
                            Some(vec![
                                EngineCoreEvent {
                                    r#type: EngineCoreEventType::Queued,
                                    timestamp: 8.0,
                                },
                                EngineCoreEvent {
                                    r#type: EngineCoreEventType::Scheduled,
                                    timestamp: 9.0,
                                },
                            ]),
                        )],
                        ..Default::default()
                    },
                )
                .await;

                let abort =
                    timeout(Duration::from_secs(1), recv_engine_message(dealer)).await.unwrap();
                assert_eq!(abort[0].as_ref(), &[0x01]);
                let aborted_ids: Vec<String> = rmp_serde::from_slice(&abort[1]).unwrap();
                assert_eq!(aborted_ids, vec![request.request_id]);
            })
        },
    );

    let llm = connect_async_llm_with_ipc(handshake_address, 0, &model_name, &ipc).await;
    let mut request = sample_generate_request("req-metrics-drop", 8);
    request.arrival_time = None;
    let mut stream = llm.generate(request).await.unwrap();
    assert_eq!(stream.next().await.unwrap().unwrap().token_ids, vec![99]);
    drop(stream);

    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    let rendered = METRICS.render().unwrap();
    assert!(rendered.contains(&format!(
        "vllm:request_success_total{{model_name=\"{model_name}\",engine=\"5\",finished_reason=\"abort\"}} 1"
    )));
    assert!(rendered.contains(&format!(
        "vllm:e2e_request_latency_seconds_count{{model_name=\"{model_name}\",engine=\"5\"}} 1"
    )));

    llm.shutdown().await.unwrap();
}
