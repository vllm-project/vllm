use std::collections::BTreeSet;
use std::convert::TryFrom;
use std::io::Cursor;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Once;
use std::time::Duration;

use futures::StreamExt;
use rmpv::Value;
use thiserror_ext::AsReport as _;
use tokio::time::timeout;
use tracing_subscriber::EnvFilter;
use zeromq::prelude::{Socket, SocketRecv, SocketSend};
use zeromq::util::PeerIdentity;
use zeromq::{DealerSocket, PushSocket, SocketOptions, ZmqMessage};

use crate::protocol::handshake::ReadyMessage;
use crate::protocol::{
    EngineCoreOutput, EngineCoreOutputs, EngineCoreRequest, EngineCoreSamplingParams, FinishReason,
    MaybeWireLogprobs, UtilityOutput, UtilityResultEnvelope, decode_engine_core_outputs,
};
use crate::test_utils::{IpcNamespace, spawn_mock_engine_task};
use crate::{ENGINE_CORE_DEAD_SENTINEL, EngineCoreClient, EngineCoreClientConfig, Error};

static TRACING: Once = Once::new();

fn expect_sample_logprobs(actual: &MaybeWireLogprobs) {
    expect_test::expect![[r#"
        Logprobs {
            positions: [
                PositionLogprobs {
                    entries: [
                        TokenLogprob {
                            token_id: 1,
                            logprob: 1.0,
                            rank: 1,
                        },
                        TokenLogprob {
                            token_id: 2,
                            logprob: 2.0,
                            rank: 1,
                        },
                        TokenLogprob {
                            token_id: 3,
                            logprob: 3.0,
                            rank: 2,
                        },
                    ],
                },
                PositionLogprobs {
                    entries: [
                        TokenLogprob {
                            token_id: 4,
                            logprob: 4.0,
                            rank: 2,
                        },
                        TokenLogprob {
                            token_id: 5,
                            logprob: 5.0,
                            rank: 1,
                        },
                        TokenLogprob {
                            token_id: 6,
                            logprob: 6.0,
                            rank: 2,
                        },
                    ],
                },
            ],
        }
    "#]]
    .assert_debug_eq(actual.as_direct().expect("logprobs resolved"));
}

fn expect_prompt_logprobs(actual: &MaybeWireLogprobs) {
    expect_test::expect![[r#"
        Logprobs {
            positions: [
                PositionLogprobs {
                    entries: [
                        TokenLogprob {
                            token_id: 10,
                            logprob: 10.0,
                            rank: 3,
                        },
                        TokenLogprob {
                            token_id: 11,
                            logprob: 11.0,
                            rank: 1,
                        },
                        TokenLogprob {
                            token_id: 12,
                            logprob: 12.0,
                            rank: 2,
                        },
                    ],
                },
                PositionLogprobs {
                    entries: [
                        TokenLogprob {
                            token_id: 13,
                            logprob: 13.0,
                            rank: 4,
                        },
                        TokenLogprob {
                            token_id: 14,
                            logprob: 14.0,
                            rank: 1,
                        },
                        TokenLogprob {
                            token_id: 15,
                            logprob: 15.0,
                            rank: 2,
                        },
                    ],
                },
            ],
        }
    "#]]
    .assert_debug_eq(actual.as_direct().expect("prompt logprobs resolved"));
}

fn sample_request() -> EngineCoreRequest {
    sample_request_with_id("req-1")
}

fn sample_request_with_id(request_id: &str) -> EngineCoreRequest {
    EngineCoreRequest {
        request_id: request_id.to_string(),
        prompt_token_ids: Some(vec![11, 22]),
        mm_features: None,
        sampling_params: Some(EngineCoreSamplingParams {
            temperature: 0.8,
            top_p: 0.9,
            top_k: 8,
            max_tokens: 32,
            min_tokens: 1,
            stop_token_ids: vec![151643],
            eos_token_id: Some(151645),
            all_stop_token_ids: BTreeSet::from([151643, 151645]),
            ..EngineCoreSamplingParams::for_test()
        }),
        pooling_params: None,
        arrival_time: 42.5,
        lora_request: None,
        cache_salt: None,
        data_parallel_rank: None,
        prompt_embeds: None,
        client_index: 0,
        current_wave: 0,
        priority: 0,
        trace_headers: None,
        resumable: false,
        external_req_id: None,
        reasoning_ended: None,
    }
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

async fn send_output_frames(push: &mut PushSocket, frames: Vec<bytes::Bytes>) {
    push.send(ZmqMessage::try_from(frames).unwrap())
        .await
        .unwrap();
}

async fn recv_engine_message(dealer: &mut DealerSocket) -> Vec<bytes::Bytes> {
    dealer.recv().await.unwrap().into_vec()
}

async fn connect_client_with_ipc(
    config: EngineCoreClientConfig,
    ipc: &IpcNamespace,
) -> EngineCoreClient {
    EngineCoreClient::connect_with_input_output_addresses(
        config,
        Some(ipc.input_endpoint()),
        Some(ipc.output_endpoint()),
    )
    .await
    .unwrap()
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

fn is_dispatcher_closed(error: &Error) -> bool {
    match error {
        Error::DispatcherClosed { .. } => true,
        Error::Shared(error) => is_dispatcher_closed(error),
        _ => false,
    }
}

fn is_engine_core_dead(error: &Error) -> bool {
    match error {
        Error::EngineCoreDead => true,
        Error::Shared(error) => is_engine_core_dead(error),
        _ => false,
    }
}

fn is_decode_error(error: &Error) -> bool {
    match error {
        Error::Decode(_) | Error::ValueDecodeExt(_) => true,
        Error::Shared(error) => is_decode_error(error),
        _ => false,
    }
}

fn decode_value(bytes: &[u8]) -> Value {
    rmpv::decode::read_value(&mut Cursor::new(bytes)).unwrap()
}

fn encode_value(value: &Value) -> Vec<u8> {
    let mut out = Vec::new();
    rmpv::encode::write_value(&mut out, value).unwrap();
    out
}

fn ndarray_value(dtype: &str, shape: &[usize], data: Value) -> Value {
    Value::Array(vec![
        Value::from(dtype),
        Value::Array(shape.iter().copied().map(Value::from).collect()),
        data,
    ])
}

fn multipart_logprob_output_frames(request_id: &str) -> Vec<bytes::Bytes> {
    let main = Value::Array(vec![
        Value::from(0),
        Value::Array(vec![Value::Array(vec![
            Value::from(request_id),
            Value::Array(vec![Value::from(7), Value::from(8)]),
            Value::Array(vec![
                ndarray_value("<i4", &[2, 3], Value::from(1)),
                ndarray_value("<f4", &[2, 3], Value::from(2)),
                ndarray_value("<i4", &[2], Value::from(3)),
                Value::Nil,
            ]),
            Value::Nil,
            Value::Nil,
            Value::from(FinishReason::Length as u8),
        ])]),
        Value::Nil,
        Value::from(0.0),
        Value::Nil,
        Value::Array(vec![Value::from(request_id)]),
    ]);

    vec![
        bytes::Bytes::from(encode_value(&main)),
        bytes::Bytes::from_static(&[
            1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0, 6, 0, 0, 0,
        ]),
        bytes::Bytes::from_static(&[
            0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64, 0, 0, 160, 64, 0, 0, 192, 64,
        ]),
        bytes::Bytes::from_static(&[1, 0, 0, 0, 2, 0, 0, 0]),
    ]
}

fn utility_result_value<T>(value: T) -> UtilityResultEnvelope
where
    T: serde::Serialize,
{
    UtilityResultEnvelope::without_type_info(rmpv::ext::to_value(value).unwrap())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn client_streams_outputs_per_request_and_ignores_other_messages() {
    init_tracing();
    let ipc = IpcNamespace::new().unwrap();
    let handshake_address = ipc.handshake_endpoint();
    let engine_identity = b"engine-0".to_vec();

    let (shutdown_tx, engine_task) = spawn_mock_engine_task(
        handshake_address.clone(),
        engine_identity.clone(),
        |dealer, push| {
            Box::pin(async move {
                let add_1 = recv_engine_message(dealer).await;
                assert_eq!(add_1[0].as_ref(), &[0x00]);
                let request_1: EngineCoreRequest = rmp_serde::from_slice(&add_1[1]).unwrap();
                assert_eq!(request_1.client_index, 7);
                assert_eq!(request_1.request_id, "req-1");

                let add_2 = recv_engine_message(dealer).await;
                assert_eq!(add_2[0].as_ref(), &[0x00]);
                let request_2: EngineCoreRequest = rmp_serde::from_slice(&add_2[1]).unwrap();
                assert_eq!(request_2.client_index, 7);
                assert_eq!(request_2.request_id, "req-2");

                send_outputs(
                    push,
                    EngineCoreOutputs {
                        utility_output: Some(UtilityOutput {
                            call_id: 1,
                            failure_message: None,
                            result: None,
                        }),
                        ..Default::default()
                    },
                )
                .await;
                send_outputs(
                    push,
                    EngineCoreOutputs {
                        start_wave: Some(3),
                        ..Default::default()
                    },
                )
                .await;
                send_outputs(
                    push,
                    EngineCoreOutputs {
                        outputs: vec![request_output("req-1", vec![999], None)],
                        utility_output: Some(UtilityOutput {
                            call_id: 2,
                            failure_message: None,
                            result: None,
                        }),
                        ..Default::default()
                    },
                )
                .await;

                send_outputs(
                    push,
                    EngineCoreOutputs {
                        outputs: vec![
                            request_output("req-2", vec![22], None),
                            request_output("req-1", vec![11], None),
                        ],
                        ..Default::default()
                    },
                )
                .await;

                let abort = recv_engine_message(dealer).await;
                assert_eq!(abort[0].as_ref(), &[0x01]);
                let aborted_ids: Vec<String> = rmp_serde::from_slice(&abort[1]).unwrap();
                assert_eq!(aborted_ids, vec!["req-1".to_string()]);

                send_outputs(
                    push,
                    EngineCoreOutputs {
                        outputs: vec![request_output("req-2", vec![], Some(FinishReason::Length))],
                        finished_requests: Some(BTreeSet::from(["req-2".to_string()])),
                        ..Default::default()
                    },
                )
                .await;

                send_outputs(
                    push,
                    EngineCoreOutputs {
                        outputs: vec![request_output("req-1", vec![], Some(FinishReason::Abort))],
                        finished_requests: Some(BTreeSet::from(["req-1".to_string()])),
                        ..Default::default()
                    },
                )
                .await;
            })
        },
    );

    let client = connect_client_with_ipc(
        EngineCoreClientConfig {
            handshake_address,
            model_name: "test-model".to_string(),
            local_host: "127.0.0.1".to_string(),
            ready_timeout: Duration::from_secs(2),
            client_index: 7,
        },
        &ipc,
    )
    .await;
    assert_eq!(client.engine_identity(), b"engine-0");
    assert_eq!(
        client
            .ready_message
            .as_ref()
            .and_then(|msg| msg.status.as_deref()),
        Some("READY")
    );

    let mut stream_1 = client.call(sample_request_with_id("req-1")).await.unwrap();
    let mut stream_2 = client.call(sample_request_with_id("req-2")).await.unwrap();

    let first_2 = timeout(Duration::from_secs(1), stream_2.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(first_2.engine_index, 0);
    assert_eq!(first_2.timestamp, 0.0);
    assert_eq!(first_2.request_id, "req-2");
    assert_eq!(first_2.new_token_ids, vec![22]);

    let first_1 = timeout(Duration::from_secs(1), stream_1.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(first_1.engine_index, 0);
    assert_eq!(first_1.timestamp, 0.0);
    assert_eq!(first_1.request_id, "req-1");
    assert_eq!(first_1.new_token_ids, vec![11]);

    client
        .abort(&["req-1".to_string(), "unknown".to_string()])
        .await
        .unwrap();

    let final_2 = timeout(Duration::from_secs(1), stream_2.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(final_2.request_id, "req-2");
    assert_eq!(final_2.finish_reason, Some(FinishReason::Length));
    assert!(
        timeout(Duration::from_secs(1), stream_2.next())
            .await
            .unwrap()
            .is_none()
    );

    let final_1 = timeout(Duration::from_secs(1), stream_1.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(final_1.request_id, "req-1");
    assert_eq!(final_1.finish_reason, Some(FinishReason::Abort));
    assert!(
        timeout(Duration::from_secs(1), stream_1.next())
            .await
            .unwrap()
            .is_none()
    );

    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    client.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn duplicate_request_ids_are_rejected_without_sending_a_second_add() {
    init_tracing();
    let ipc = IpcNamespace::new().unwrap();
    let handshake_address = ipc.handshake_endpoint();
    let engine_identity = b"engine-dup".to_vec();

    let (shutdown_tx, engine_task) = spawn_mock_engine_task(
        handshake_address.clone(),
        engine_identity.clone(),
        |dealer, push| {
            Box::pin(async move {
                let add_1 = recv_engine_message(dealer).await;
                assert_eq!(add_1[0].as_ref(), &[0x00]);
                let request_1: EngineCoreRequest = rmp_serde::from_slice(&add_1[1]).unwrap();
                assert_eq!(request_1.request_id, "req-1");

                assert!(
                    timeout(Duration::from_millis(200), dealer.recv())
                        .await
                        .is_err()
                );

                send_outputs(
                    push,
                    EngineCoreOutputs {
                        outputs: vec![request_output("req-1", vec![], Some(FinishReason::Length))],
                        finished_requests: Some(BTreeSet::from(["req-1".to_string()])),
                        ..Default::default()
                    },
                )
                .await;
            })
        },
    );

    let client = connect_client_with_ipc(
        EngineCoreClientConfig {
            handshake_address,
            model_name: "test-model".to_string(),
            local_host: "127.0.0.1".to_string(),
            ready_timeout: Duration::from_secs(2),
            client_index: 0,
        },
        &ipc,
    )
    .await;

    let mut stream = client.call(sample_request()).await.unwrap();
    let error = match client.call(sample_request()).await {
        Ok(_) => panic!("expected duplicate request error"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        Error::DuplicateRequestId { request_id } if request_id == "req-1"
    ));

    let final_output = timeout(Duration::from_secs(1), stream.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(final_output.finish_reason, Some(FinishReason::Length));
    assert!(
        timeout(Duration::from_secs(1), stream.next())
            .await
            .unwrap()
            .is_none()
    );
    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    client.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn finished_requests_without_final_output_is_treated_as_unexpected_close() {
    init_tracing();
    let ipc = IpcNamespace::new().unwrap();
    let handshake_address = ipc.handshake_endpoint();
    let engine_identity = b"engine-finished-only".to_vec();

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
                        finished_requests: Some(BTreeSet::from(["req-1".to_string()])),
                        ..Default::default()
                    },
                )
                .await;

                assert!(
                    timeout(Duration::from_millis(200), dealer.recv())
                        .await
                        .is_err()
                );
                let _ = push;
            })
        },
    );

    let client = connect_client_with_ipc(
        EngineCoreClientConfig {
            handshake_address,
            model_name: "test-model".to_string(),
            local_host: "127.0.0.1".to_string(),
            ready_timeout: Duration::from_secs(2),
            client_index: 0,
        },
        &ipc,
    )
    .await;

    let mut stream = client.call(sample_request()).await.unwrap();
    let error = timeout(Duration::from_secs(1), stream.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap_err();
    assert!(matches!(
        error,
        Error::RequestStreamClosed { request_id } if request_id == "req-1"
    ));
    assert!(
        timeout(Duration::from_secs(1), stream.next())
            .await
            .unwrap()
            .is_none()
    );

    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    client.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dropping_a_live_stream_triggers_abort() {
    init_tracing();
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
                        outputs: vec![request_output("req-1", vec![99], None)],
                        ..Default::default()
                    },
                )
                .await;

                let abort = timeout(Duration::from_secs(1), recv_engine_message(dealer))
                    .await
                    .unwrap();
                assert_eq!(abort[0].as_ref(), &[0x01]);
                let aborted_ids: Vec<String> = rmp_serde::from_slice(&abort[1]).unwrap();
                assert_eq!(aborted_ids, vec!["req-1".to_string()]);
            })
        },
    );

    let client = connect_client_with_ipc(
        EngineCoreClientConfig {
            handshake_address,
            model_name: "test-model".to_string(),
            local_host: "127.0.0.1".to_string(),
            ready_timeout: Duration::from_secs(2),
            client_index: 0,
        },
        &ipc,
    )
    .await;

    let mut stream = client.call(sample_request()).await.unwrap();
    let first = timeout(Duration::from_secs(1), stream.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(first.new_token_ids, vec![99]);
    drop(stream);

    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    client.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dispatcher_failure_propagates_to_streams_and_future_calls() {
    init_tracing();
    let ipc = IpcNamespace::new().unwrap();
    let handshake_address = ipc.handshake_endpoint();
    let engine_identity = b"engine-fail".to_vec();

    let (shutdown_tx, engine_task) = spawn_mock_engine_task(
        handshake_address.clone(),
        engine_identity.clone(),
        |dealer, push| {
            Box::pin(async move {
                let _ = recv_engine_message(dealer).await;
                let _ = recv_engine_message(dealer).await;

                push.send(ZmqMessage::from(vec![0xc1])).await.unwrap();
            })
        },
    );

    let client = connect_client_with_ipc(
        EngineCoreClientConfig {
            handshake_address,
            model_name: "test-model".to_string(),
            local_host: "127.0.0.1".to_string(),
            ready_timeout: Duration::from_secs(2),
            client_index: 0,
        },
        &ipc,
    )
    .await;

    let mut stream_1 = client.call(sample_request_with_id("req-1")).await.unwrap();
    let mut stream_2 = client.call(sample_request_with_id("req-2")).await.unwrap();

    let error_1 = timeout(Duration::from_secs(1), stream_1.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap_err();
    let error_2 = timeout(Duration::from_secs(1), stream_2.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap_err();
    assert!(is_decode_error(&error_1));
    assert!(is_decode_error(&error_2));
    assert!(is_decode_error(
        client
            .health_error()
            .as_deref()
            .expect("health error recorded")
    ));

    let abort_error = client.abort(&["req-1".to_string()]).await.unwrap_err();
    assert!(is_decode_error(&abort_error));

    let add_error = match client.call(sample_request_with_id("req-3")).await {
        Ok(_) => panic!("expected dispatcher closed error"),
        Err(error) => error,
    };
    assert!(is_decode_error(&add_error));

    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    client.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn is_sleeping_wrapper_sends_typed_request_and_returns_typed_response() {
    init_tracing();
    let ipc = IpcNamespace::new().unwrap();
    let handshake_address = ipc.handshake_endpoint();
    let engine_identity = b"engine-utility-success".to_vec();

    let (shutdown_tx, engine_task) = spawn_mock_engine_task(
        handshake_address.clone(),
        engine_identity.clone(),
        |dealer, push| {
            Box::pin(async move {
                let utility = recv_engine_message(dealer).await;
                assert_eq!(utility[0].as_ref(), &[0x03]);

                let payload = decode_value(&utility[1]);
                let array = match payload {
                    Value::Array(array) => array,
                    other => panic!("expected utility payload array, got {other:?}"),
                };
                assert_eq!(array.len(), 4);
                assert_eq!(array[0], Value::from(5));
                let call_id = array[1].as_i64().expect("call_id");
                assert_eq!(array[2], Value::from("is_sleeping"));
                assert_eq!(array[3], Value::Array(Vec::new()));

                send_outputs(
                    push,
                    EngineCoreOutputs {
                        utility_output: Some(UtilityOutput {
                            call_id,
                            failure_message: None,
                            result: Some(utility_result_value(true)),
                        }),
                        ..Default::default()
                    },
                )
                .await;
            })
        },
    );

    let client = connect_client_with_ipc(
        EngineCoreClientConfig {
            handshake_address,
            model_name: "test-model".to_string(),
            local_host: "127.0.0.1".to_string(),
            ready_timeout: Duration::from_secs(2),
            client_index: 5,
        },
        &ipc,
    )
    .await;

    let result = client.is_sleeping().await.unwrap();
    assert!(result);

    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    client.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn call_utility_failure_message_surfaces_as_error() {
    init_tracing();
    let ipc = IpcNamespace::new().unwrap();
    let handshake_address = ipc.handshake_endpoint();
    let engine_identity = b"engine-utility-fail".to_vec();

    let (shutdown_tx, engine_task) = spawn_mock_engine_task(
        handshake_address.clone(),
        engine_identity.clone(),
        |dealer, push| {
            Box::pin(async move {
                let utility = recv_engine_message(dealer).await;
                assert_eq!(utility[0].as_ref(), &[0x03]);
                let payload = decode_value(&utility[1]);
                let call_id = payload
                    .as_array()
                    .and_then(|array| array[1].as_i64())
                    .expect("call_id");

                send_outputs(
                    push,
                    EngineCoreOutputs {
                        utility_output: Some(UtilityOutput {
                            call_id,
                            failure_message: Some("boom".to_string()),
                            result: None,
                        }),
                        ..Default::default()
                    },
                )
                .await;
            })
        },
    );

    let client = connect_client_with_ipc(
        EngineCoreClientConfig {
            handshake_address,
            model_name: "test-model".to_string(),
            local_host: "127.0.0.1".to_string(),
            ready_timeout: Duration::from_secs(2),
            client_index: 0,
        },
        &ipc,
    )
    .await;

    let error = client
        .call_utility::<bool, _>("is_sleeping", ())
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        Error::UtilityCallFailed {
            method,
            message,
            ..
        } if method == "is_sleeping" && message == "boom"
    ));

    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    client.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dispatcher_failure_propagates_to_waiting_utility_calls() {
    init_tracing();
    let ipc = IpcNamespace::new().unwrap();
    let handshake_address = ipc.handshake_endpoint();
    let engine_identity = b"engine-utility-dispatcher-fail".to_vec();

    let (shutdown_tx, engine_task) = spawn_mock_engine_task(
        handshake_address.clone(),
        engine_identity.clone(),
        |dealer, push| {
            Box::pin(async move {
                let utility = recv_engine_message(dealer).await;
                assert_eq!(utility[0].as_ref(), &[0x03]);

                push.send(ZmqMessage::from(vec![0xc1])).await.unwrap();
            })
        },
    );

    let client = connect_client_with_ipc(
        EngineCoreClientConfig {
            handshake_address,
            model_name: "test-model".to_string(),
            local_host: "127.0.0.1".to_string(),
            ready_timeout: Duration::from_secs(2),
            client_index: 0,
        },
        &ipc,
    )
    .await;

    let error = client
        .call_utility::<bool, _>("is_sleeping", ())
        .await
        .unwrap_err();
    assert!(is_decode_error(&error));
    assert!(is_decode_error(
        client
            .health_error()
            .as_deref()
            .expect("health error recorded")
    ));

    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    client.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn connect_times_out_without_ready_message() {
    init_tracing();
    let ipc = IpcNamespace::new().unwrap();
    let handshake_address = ipc.handshake_endpoint();
    let engine_handshake = handshake_address.clone();
    let engine_task = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(50)).await;

        let mut options = SocketOptions::default();
        options.peer_identity(PeerIdentity::try_from(b"engine-timeout".to_vec()).unwrap());
        let mut handshake = DealerSocket::with_options(options);
        handshake.connect(&engine_handshake).await.unwrap();
        handshake
            .send(ZmqMessage::from(
                rmp_serde::to_vec_named(&ready_message("HELLO")).unwrap(),
            ))
            .await
            .unwrap();

        let _ = handshake.recv().await.unwrap();
    });

    let result = EngineCoreClient::connect_with_input_output_addresses(
        EngineCoreClientConfig {
            handshake_address,
            model_name: "test-model".to_string(),
            local_host: "127.0.0.1".to_string(),
            ready_timeout: Duration::from_millis(100),
            client_index: 0,
        },
        Some(ipc.input_endpoint()),
        Some(ipc.output_endpoint()),
    )
    .await;

    let error = match result {
        Ok(_) => panic!("expected ready timeout"),
        Err(error) => error,
    };

    let message = error.to_report_string();
    assert!(message.contains("timed out"));
    assert!(message.contains("READY"));
    engine_task.await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn engine_core_dead_sentinel_marks_client_unhealthy_and_sticks() {
    init_tracing();
    let ipc = IpcNamespace::new().unwrap();
    let handshake_address = ipc.handshake_endpoint();
    let engine_identity = b"engine-dead".to_vec();

    let (shutdown_tx, engine_task) = spawn_mock_engine_task(
        handshake_address.clone(),
        engine_identity.clone(),
        |_dealer, push| {
            Box::pin(async move {
                push.send(ZmqMessage::from(ENGINE_CORE_DEAD_SENTINEL.to_vec()))
                    .await
                    .unwrap();
            })
        },
    );

    let client = connect_client_with_ipc(
        EngineCoreClientConfig {
            handshake_address,
            model_name: "test-model".to_string(),
            local_host: "127.0.0.1".to_string(),
            ready_timeout: Duration::from_secs(2),
            client_index: 0,
        },
        &ipc,
    )
    .await;

    timeout(Duration::from_secs(2), async {
        while client.is_healthy() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("wait for unhealthy client");

    assert!(!client.is_healthy());
    assert!(matches!(
        client.health_error().as_deref(),
        Some(Error::EngineCoreDead)
    ));

    let error = client
        .call_utility::<bool, _>("is_sleeping", ())
        .await
        .unwrap_err();
    assert!(
        is_dispatcher_closed(&error) || is_engine_core_dead(&error),
        "unexpected error: {error:?}"
    );
    assert!(matches!(
        client.health_error().as_deref(),
        Some(Error::EngineCoreDead)
    ));

    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    client.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn output_loop_failure_marks_client_unhealthy_and_records_first_error() {
    init_tracing();
    let ipc = IpcNamespace::new().unwrap();
    let handshake_address = ipc.handshake_endpoint();
    let engine_identity = b"engine-output-failure".to_vec();

    let (shutdown_tx, engine_task) = spawn_mock_engine_task(
        handshake_address.clone(),
        engine_identity.clone(),
        |_dealer, push| {
            Box::pin(async move {
                send_output_frames(
                    push,
                    vec![
                        bytes::Bytes::from_static(b"frame-1"),
                        bytes::Bytes::from_static(b"frame-2"),
                    ],
                )
                .await;
            })
        },
    );

    let client = connect_client_with_ipc(
        EngineCoreClientConfig {
            handshake_address,
            model_name: "test-model".to_string(),
            local_host: "127.0.0.1".to_string(),
            ready_timeout: Duration::from_secs(2),
            client_index: 0,
        },
        &ipc,
    )
    .await;

    timeout(Duration::from_secs(2), async {
        while client.is_healthy() {
            let _ = client.call_utility::<bool, _>("is_sleeping", ()).await;
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("wait for unhealthy client");

    assert!(!client.is_healthy());
    assert!(is_decode_error(
        client
            .health_error()
            .as_deref()
            .expect("health error recorded")
    ));

    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    client.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn client_decodes_multipart_logprob_outputs() {
    init_tracing();
    let ipc = IpcNamespace::new().unwrap();
    let handshake_address = ipc.handshake_endpoint();
    let engine_identity = b"engine-multipart-logprobs".to_vec();

    let (shutdown_tx, engine_task) = spawn_mock_engine_task(
        handshake_address.clone(),
        engine_identity.clone(),
        |dealer, push| {
            Box::pin(async move {
                let add = recv_engine_message(dealer).await;
                assert_eq!(add[0].as_ref(), &[0x00]);
                let request: EngineCoreRequest = rmp_serde::from_slice(&add[1]).unwrap();
                assert_eq!(request.request_id, "req-1");

                send_output_frames(push, multipart_logprob_output_frames("req-1")).await;
            })
        },
    );

    let client = connect_client_with_ipc(
        EngineCoreClientConfig {
            handshake_address,
            model_name: "test-model".to_string(),
            local_host: "127.0.0.1".to_string(),
            ready_timeout: Duration::from_secs(2),
            client_index: 0,
        },
        &ipc,
    )
    .await;

    let stream = client.call(sample_request()).await.unwrap();
    let outputs = stream.collect::<Vec<_>>().await;
    assert_eq!(outputs.len(), 1);

    let output = outputs.into_iter().next().unwrap().unwrap();
    assert_eq!(output.output.new_token_ids, vec![7, 8]);
    assert_eq!(output.output.finish_reason, Some(FinishReason::Length));
    expect_sample_logprobs(
        output
            .output
            .new_logprobs
            .as_ref()
            .expect("logprobs decoded"),
    );

    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    client.shutdown().await.unwrap();
}

#[test]
fn python_msgpack_fixtures_match_rust_encoding() {
    init_tracing();
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/tests/python_compat.py");
    let output = Command::new(&script)
        .output()
        .unwrap_or_else(|error| panic!("failed to execute {:?}: {error}", script));
    assert!(
        output.status.success(),
        "python fixture script failed: status={:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );

    let stdout = String::from_utf8(output.stdout).unwrap();
    let mut lines = stdout.lines();
    let request_hex = lines.next().expect("missing request fixture line");
    let outputs_hex = lines.next().expect("missing outputs fixture line");
    let inline_logprobs_frames = lines.next().expect("missing inline logprobs fixture line");
    let multipart_logprobs_frames = lines
        .next()
        .expect("missing multipart logprobs fixture line");
    let inline_prompt_frames = lines
        .next()
        .expect("missing inline prompt logprobs fixture line");
    let multipart_prompt_frames = lines
        .next()
        .expect("missing multipart prompt logprobs fixture line");

    let request_bytes = hex::decode(request_hex).unwrap();
    let outputs_bytes = hex::decode(outputs_hex).unwrap();

    let decoded_request: EngineCoreRequest = rmp_serde::from_slice(&request_bytes).unwrap();
    let expected_request = sample_request();
    assert_eq!(decoded_request, expected_request);

    let decoded_outputs: EngineCoreOutputs = rmp_serde::from_slice(&outputs_bytes).unwrap();
    expect_test::expect![[r#"
        EngineCoreOutputs {
            engine_index: 0,
            outputs: [
                EngineCoreOutput {
                    request_id: "req-1",
                    new_token_ids: [
                        7,
                        8,
                    ],
                    new_logprobs: None,
                    new_prompt_logprobs_tensors: None,
                    pooling_output: None,
                    finish_reason: Some(
                        Length,
                    ),
                    stop_reason: None,
                    events: None,
                    kv_transfer_params: None,
                    trace_headers: None,
                    num_cached_tokens: 0,
                    num_external_computed_tokens: 0,
                    routed_experts: None,
                    num_nans_in_logits: 0,
                },
            ],
            scheduler_stats: None,
            timestamp: 0.0,
            utility_output: None,
            finished_requests: Some(
                {
                    "req-1",
                },
            ),
            wave_complete: None,
            start_wave: None,
        }
    "#]]
    .assert_debug_eq(&decoded_outputs);

    let decode_frames = |line: &str| {
        line.split_whitespace()
            .map(|frame| bytes::Bytes::from(hex::decode(frame).unwrap()))
            .collect::<Vec<_>>()
    };

    let inline_logprobs =
        decode_engine_core_outputs(&decode_frames(inline_logprobs_frames)).unwrap();
    expect_sample_logprobs(
        inline_logprobs.outputs[0]
            .new_logprobs
            .as_ref()
            .expect("inline logprobs decoded"),
    );

    let multipart_logprobs =
        decode_engine_core_outputs(&decode_frames(multipart_logprobs_frames)).unwrap();
    expect_sample_logprobs(
        multipart_logprobs.outputs[0]
            .new_logprobs
            .as_ref()
            .expect("multipart logprobs decoded"),
    );

    let inline_prompt = decode_engine_core_outputs(&decode_frames(inline_prompt_frames)).unwrap();
    expect_prompt_logprobs(
        inline_prompt.outputs[0]
            .new_prompt_logprobs_tensors
            .as_ref()
            .expect("inline prompt logprobs decoded"),
    );

    let multipart_prompt =
        decode_engine_core_outputs(&decode_frames(multipart_prompt_frames)).unwrap();
    expect_prompt_logprobs(
        multipart_prompt.outputs[0]
            .new_prompt_logprobs_tensors
            .as_ref()
            .expect("multipart prompt logprobs decoded"),
    );
}
