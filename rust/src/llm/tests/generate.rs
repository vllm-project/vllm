use std::collections::BTreeSet;
use std::sync::Once;
use std::time::Duration;

use futures::StreamExt as _;
use tokio::time::timeout;
use tracing_subscriber::EnvFilter;
use uuid::Uuid;
use vllm_engine_core_client::protocol::{
    EngineCoreEvent, EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs, EngineCoreRequest,
    EngineCoreSamplingParams, FinishReason, RequestOutputKind,
};
use vllm_engine_core_client::test_utils::{IpcNamespace, spawn_mock_engine_task};
use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig};
use vllm_llm::{Error, GenerateRequest, Llm};
use vllm_metrics::METRICS;
use zeromq::prelude::{SocketRecv, SocketSend};
use zeromq::{DealerSocket, PushSocket, ZmqMessage};

static TRACING: Once = Once::new();

fn request_output(
    request_id: &str,
    new_token_ids: Vec<u32>,
    finish_reason: Option<FinishReason>,
) -> EngineCoreOutput {
    request_output_with_events(request_id, new_token_ids, finish_reason, None)
}

fn request_output_with_events(
    request_id: &str,
    new_token_ids: Vec<u32>,
    finish_reason: Option<FinishReason>,
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
        num_cached_tokens: 0,
        num_external_computed_tokens: 0,
        routed_experts: None,
        num_nans_in_logits: 0,
    }
}

fn sample_generate_request(
    request_id: &str,
    output_kind: RequestOutputKind,
    max_tokens: u32,
) -> GenerateRequest {
    GenerateRequest {
        request_id: request_id.to_string(),
        prompt_token_ids: vec![11, 22],
        sampling_params: EngineCoreSamplingParams {
            output_kind,
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
    let client = EngineCoreClient::connect_with_input_output_addresses(
        EngineCoreClientConfig {
            handshake_address,
            model_name: model_name.to_string(),
            local_host: "127.0.0.1".to_string(),
            ready_timeout: Duration::from_secs(2),
            client_index,
        },
        Some(ipc.input_endpoint()),
        Some(ipc.output_endpoint()),
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
        let _ = tracing_subscriber::fmt()
            .with_test_writer()
            .with_env_filter(filter)
            .try_init();
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn generate_streams_delta_outputs() {
    init_tracing();
    let ipc = IpcNamespace::new().unwrap();
    let handshake_address = ipc.handshake_endpoint();
    let engine_identity = b"engine-delta".to_vec();

    let (shutdown_tx, engine_task) = spawn_mock_engine_task(
        handshake_address.clone(),
        engine_identity.clone(),
        |dealer, push| {
            Box::pin(async move {
                let add = recv_engine_message(dealer).await;
                assert_eq!(add[0].as_ref(), &[0x00]);
                let request: EngineCoreRequest = rmp_serde::from_slice(&add[1]).unwrap();
                assert_eq!(request.request_id, "req-delta");
                assert_eq!(request.client_index, 7);
                assert_eq!(request.prompt_token_ids, Some(vec![11, 22]));

                send_outputs(
                    push,
                    EngineCoreOutputs {
                        outputs: vec![
                            request_output("req-delta", vec![1, 2], None),
                            request_output("req-delta", vec![3], Some(FinishReason::Length)),
                        ],
                        finished_requests: Some(BTreeSet::from(["req-delta".to_string()])),
                        ..Default::default()
                    },
                )
                .await;
            })
        },
    );

    let llm = connect_async_llm_with_ipc(handshake_address, 7, "test-model", &ipc).await;
    let mut stream = llm
        .generate(sample_generate_request(
            "req-delta",
            RequestOutputKind::Delta,
            3,
        ))
        .await
        .unwrap();

    let first = stream.next().await.unwrap().unwrap();
    assert_eq!(first.request_id, "req-delta");
    assert_eq!(first.prompt_token_ids.as_ref(), &[11, 22]);
    assert_eq!(first.token_ids, vec![1, 2]);
    assert_eq!(first.raw.finish_reason, None);

    let second = stream.next().await.unwrap().unwrap();
    assert_eq!(second.token_ids, vec![3]);
    assert_eq!(second.raw.finish_reason, Some(FinishReason::Length));
    assert!(stream.next().await.is_none());

    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    llm.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn generate_streams_cumulative_outputs() {
    let ipc = IpcNamespace::new().unwrap();
    let handshake_address = ipc.handshake_endpoint();
    let engine_identity = b"engine-cumulative".to_vec();

    let (shutdown_tx, engine_task) = spawn_mock_engine_task(
        handshake_address.clone(),
        engine_identity.clone(),
        |dealer, push| {
            Box::pin(async move {
                let add = recv_engine_message(dealer).await;
                assert_eq!(add[0].as_ref(), &[0x00]);

                send_outputs(
                    push,
                    EngineCoreOutputs {
                        outputs: vec![
                            request_output("req-cumulative", vec![9], None),
                            request_output(
                                "req-cumulative",
                                vec![10, 11],
                                Some(FinishReason::Length),
                            ),
                        ],
                        finished_requests: Some(BTreeSet::from(["req-cumulative".to_string()])),
                        ..Default::default()
                    },
                )
                .await;
            })
        },
    );

    let llm = connect_async_llm_with_ipc(handshake_address, 0, "test-model", &ipc).await;
    let mut stream = llm
        .generate(sample_generate_request(
            "req-cumulative",
            RequestOutputKind::Cumulative,
            3,
        ))
        .await
        .unwrap();

    let first = stream.next().await.unwrap().unwrap();
    assert_eq!(first.token_ids, vec![9]);

    let second = stream.next().await.unwrap().unwrap();
    assert_eq!(second.token_ids, vec![9, 10, 11]);
    assert_eq!(second.raw.finish_reason, Some(FinishReason::Length));
    assert!(stream.next().await.is_none());

    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    llm.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn generate_streams_final_only_outputs() {
    let ipc = IpcNamespace::new().unwrap();
    let handshake_address = ipc.handshake_endpoint();
    let engine_identity = b"engine-final".to_vec();

    let (shutdown_tx, engine_task) = spawn_mock_engine_task(
        handshake_address.clone(),
        engine_identity.clone(),
        |dealer, push| {
            Box::pin(async move {
                let add = recv_engine_message(dealer).await;
                assert_eq!(add[0].as_ref(), &[0x00]);

                send_outputs(
                    push,
                    EngineCoreOutputs {
                        outputs: vec![
                            request_output("req-final", vec![4], None),
                            request_output("req-final", vec![5, 6], Some(FinishReason::Length)),
                        ],
                        finished_requests: Some(BTreeSet::from(["req-final".to_string()])),
                        ..Default::default()
                    },
                )
                .await;
            })
        },
    );

    let llm = connect_async_llm_with_ipc(handshake_address, 0, "test-model", &ipc).await;
    let mut stream = llm
        .generate(sample_generate_request(
            "req-final",
            RequestOutputKind::FinalOnly,
            3,
        ))
        .await
        .unwrap();

    let final_output = stream.next().await.unwrap().unwrap();
    assert_eq!(final_output.token_ids, vec![4, 5, 6]);
    assert_eq!(final_output.raw.finish_reason, Some(FinishReason::Length));
    assert!(stream.next().await.is_none());

    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    llm.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn generate_propagates_unexpected_close_errors() {
    let ipc = IpcNamespace::new().unwrap();
    let handshake_address = ipc.handshake_endpoint();
    let engine_identity = b"engine-close".to_vec();

    let (shutdown_tx, engine_task) = spawn_mock_engine_task(
        handshake_address.clone(),
        engine_identity.clone(),
        |dealer, push| {
            Box::pin(async move {
                let add = recv_engine_message(dealer).await;
                assert_eq!(add[0].as_ref(), &[0x00]);

                send_outputs(
                    push,
                    EngineCoreOutputs {
                        finished_requests: Some(BTreeSet::from(["req-close".to_string()])),
                        ..Default::default()
                    },
                )
                .await;
            })
        },
    );

    let llm = connect_async_llm_with_ipc(handshake_address, 0, "test-model", &ipc).await;
    let mut stream = llm
        .generate(sample_generate_request(
            "req-close",
            RequestOutputKind::Cumulative,
            1,
        ))
        .await
        .unwrap();

    let error = stream.next().await.unwrap().unwrap_err();
    assert!(matches!(
        error,
        Error::EngineCoreClient(vllm_engine_core_client::Error::RequestStreamClosed {
            request_id
        }) if request_id == "req-close"
    ));
    assert!(stream.next().await.is_none());

    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    llm.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn abort_forwards_to_engine_core_client() {
    let ipc = IpcNamespace::new().unwrap();
    let handshake_address = ipc.handshake_endpoint();
    let engine_identity = b"engine-abort".to_vec();

    let (shutdown_tx, engine_task) = spawn_mock_engine_task(
        handshake_address.clone(),
        engine_identity.clone(),
        |dealer, push| {
            Box::pin(async move {
                let add = recv_engine_message(dealer).await;
                assert_eq!(add[0].as_ref(), &[0x00]);

                let abort = timeout(Duration::from_secs(1), recv_engine_message(dealer))
                    .await
                    .unwrap();
                assert_eq!(abort[0].as_ref(), &[0x01]);
                let aborted_ids: Vec<String> = rmp_serde::from_slice(&abort[1]).unwrap();
                assert_eq!(aborted_ids, vec!["req-abort".to_string()]);
                let _ = push;
            })
        },
    );

    let llm = connect_async_llm_with_ipc(handshake_address, 0, "test-model", &ipc).await;
    let stream = llm
        .generate(sample_generate_request(
            "req-abort",
            RequestOutputKind::Delta,
            4,
        ))
        .await
        .unwrap();
    drop(stream);

    llm.abort("req-abort").await.unwrap();
    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    llm.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dropping_a_live_generate_stream_triggers_abort() {
    let ipc = IpcNamespace::new().unwrap();
    let handshake_address = ipc.handshake_endpoint();
    let engine_identity = b"engine-drop".to_vec();

    let (shutdown_tx, engine_task) = spawn_mock_engine_task(
        handshake_address.clone(),
        engine_identity.clone(),
        |dealer, push| {
            Box::pin(async move {
                let add = recv_engine_message(dealer).await;
                assert_eq!(add[0].as_ref(), &[0x00]);

                send_outputs(
                    push,
                    EngineCoreOutputs {
                        outputs: vec![request_output("req-drop", vec![99], None)],
                        ..Default::default()
                    },
                )
                .await;

                let abort = timeout(Duration::from_secs(1), recv_engine_message(dealer))
                    .await
                    .unwrap();
                assert_eq!(abort[0].as_ref(), &[0x01]);
                let aborted_ids: Vec<String> = rmp_serde::from_slice(&abort[1]).unwrap();
                assert_eq!(aborted_ids, vec!["req-drop".to_string()]);
            })
        },
    );

    let llm = connect_async_llm_with_ipc(handshake_address, 0, "test-model", &ipc).await;
    let mut stream = llm
        .generate(sample_generate_request(
            "req-drop",
            RequestOutputKind::Delta,
            4,
        ))
        .await
        .unwrap();

    let output = stream.next().await.unwrap().unwrap();
    assert_eq!(output.token_ids, vec![99]);
    drop(stream);

    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    llm.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn duplicate_request_ids_bubble_up_from_engine_core_client() {
    let ipc = IpcNamespace::new().unwrap();
    let handshake_address = ipc.handshake_endpoint();
    let engine_identity = b"engine-dup".to_vec();

    let (shutdown_tx, engine_task) = spawn_mock_engine_task(
        handshake_address.clone(),
        engine_identity.clone(),
        |dealer, push| {
            Box::pin(async move {
                let add = recv_engine_message(dealer).await;
                assert_eq!(add[0].as_ref(), &[0x00]);

                assert!(
                    timeout(Duration::from_millis(200), dealer.recv())
                        .await
                        .is_err()
                );

                send_outputs(
                    push,
                    EngineCoreOutputs {
                        outputs: vec![request_output(
                            "req-dup",
                            vec![],
                            Some(FinishReason::Length),
                        )],
                        finished_requests: Some(BTreeSet::from(["req-dup".to_string()])),
                        ..Default::default()
                    },
                )
                .await;
            })
        },
    );

    let llm = connect_async_llm_with_ipc(handshake_address, 0, "test-model", &ipc).await;
    let stream_1 = llm
        .generate(sample_generate_request(
            "req-dup",
            RequestOutputKind::FinalOnly,
            1,
        ))
        .await
        .unwrap();
    let error = match llm
        .generate(sample_generate_request(
            "req-dup",
            RequestOutputKind::FinalOnly,
            1,
        ))
        .await
    {
        Ok(_) => panic!("expected duplicate request id error"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        Error::EngineCoreClient(vllm_engine_core_client::Error::DuplicateRequestId {
            request_id
        }) if request_id == "req-dup"
    ));
    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    drop(stream_1);
    llm.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn generate_records_request_metrics_in_prometheus_output() {
    let ipc = IpcNamespace::new().unwrap();
    let handshake_address = ipc.handshake_endpoint();
    let engine_identity = b"engine-metrics".to_vec();
    let model_name = request_metrics_model_name("metrics-model");

    let (shutdown_tx, engine_task) = spawn_mock_engine_task(
        handshake_address.clone(),
        engine_identity.clone(),
        |dealer, push| {
            Box::pin(async move {
                let add = recv_engine_message(dealer).await;
                assert_eq!(add[0].as_ref(), &[0x00]);

                send_outputs(
                    push,
                    EngineCoreOutputs {
                        engine_index: 4,
                        timestamp: 10.0,
                        outputs: vec![request_output_with_events(
                            "req-metrics",
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
                        )],
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
                            "req-metrics",
                            vec![2, 3],
                            Some(FinishReason::Length),
                            Some(vec![EngineCoreEvent {
                                r#type: EngineCoreEventType::Preempted,
                                timestamp: 10.5,
                            }]),
                        )],
                        finished_requests: Some(BTreeSet::from(["req-metrics".to_string()])),
                        ..Default::default()
                    },
                )
                .await;
            })
        },
    );

    let llm = connect_async_llm_with_ipc(handshake_address, 0, &model_name, &ipc).await;
    let mut request = sample_generate_request("req-metrics", RequestOutputKind::Delta, 8);
    request.arrival_time = None;
    let mut stream = llm.generate(request).await.unwrap();

    assert_eq!(stream.next().await.unwrap().unwrap().token_ids, vec![1]);
    let final_output = stream.next().await.unwrap().unwrap();
    assert_eq!(final_output.token_ids, vec![2, 3]);
    assert_eq!(final_output.raw.finish_reason, Some(FinishReason::Length));
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
        "vllm:prompt_tokens_recomputed_total{{model_name=\"{model_name}\",engine=\"4\"}} 0"
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
    let engine_identity = b"engine-metrics-drop".to_vec();
    let model_name = request_metrics_model_name("metrics-drop-model");

    let (shutdown_tx, engine_task) = spawn_mock_engine_task(
        handshake_address.clone(),
        engine_identity.clone(),
        |dealer, push| {
            Box::pin(async move {
                let add = recv_engine_message(dealer).await;
                assert_eq!(add[0].as_ref(), &[0x00]);

                send_outputs(
                    push,
                    EngineCoreOutputs {
                        engine_index: 5,
                        timestamp: 10.0,
                        outputs: vec![request_output_with_events(
                            "req-metrics-drop",
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

                let abort = timeout(Duration::from_secs(1), recv_engine_message(dealer))
                    .await
                    .unwrap();
                assert_eq!(abort[0].as_ref(), &[0x01]);
            })
        },
    );

    let llm = connect_async_llm_with_ipc(handshake_address, 0, &model_name, &ipc).await;
    let mut request = sample_generate_request("req-metrics-drop", RequestOutputKind::Delta, 8);
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
