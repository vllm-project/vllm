use std::collections::BTreeSet;
use std::convert::TryFrom;
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Once;
use std::time::Duration;

use futures::StreamExt;
use thiserror_ext::AsReport as _;
use tokio::time::timeout;
use tracing_subscriber::EnvFilter;
use vllm_engine_core_client::protocol::handshake::{HandshakeInitMessage, ReadyMessage};
use vllm_engine_core_client::protocol::{
    EngineCoreOutput, EngineCoreOutputs, EngineCoreRequest, EngineCoreSamplingParams, FinishReason,
    RequestOutputKind, UtilityOutput,
};
use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig, Error};
use zeromq::prelude::{Socket, SocketRecv, SocketSend};
use zeromq::util::PeerIdentity;
use zeromq::{DealerSocket, PushSocket, SocketOptions, ZmqMessage};

static TRACING: Once = Once::new();

fn unique_tcp_endpoint() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    format!("tcp://127.0.0.1:{port}")
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
            output_kind: RequestOutputKind::FinalOnly,
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
    assert_eq!(init.addresses.inputs.len(), 1);
    assert_eq!(init.addresses.outputs.len(), 1);

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

fn is_shared_dispatcher_closed(error: &Error) -> bool {
    matches!(error, Error::Shared(inner) if matches!(inner.as_ref(), Error::DispatcherClosed { .. }))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn client_streams_outputs_per_request_and_ignores_other_messages() {
    init_tracing();
    let handshake_address = unique_tcp_endpoint();
    let engine_identity = b"engine-0".to_vec();

    let engine_task = tokio::spawn({
        let engine_handshake = handshake_address.clone();
        let engine_identity = engine_identity.clone();
        async move {
            let (mut dealer, mut push) = setup_mock_engine(engine_handshake, engine_identity).await;

            let add_1 = recv_engine_message(&mut dealer).await;
            assert_eq!(add_1[0].as_ref(), &[0x00]);
            let request_1: EngineCoreRequest = rmp_serde::from_slice(&add_1[1]).unwrap();
            assert_eq!(request_1.client_index, 7);
            assert_eq!(request_1.request_id, "req-1");

            let add_2 = recv_engine_message(&mut dealer).await;
            assert_eq!(add_2[0].as_ref(), &[0x00]);
            let request_2: EngineCoreRequest = rmp_serde::from_slice(&add_2[1]).unwrap();
            assert_eq!(request_2.client_index, 7);
            assert_eq!(request_2.request_id, "req-2");

            send_outputs(
                &mut push,
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
                &mut push,
                EngineCoreOutputs {
                    start_wave: Some(3),
                    ..Default::default()
                },
            )
            .await;
            send_outputs(
                &mut push,
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
                &mut push,
                EngineCoreOutputs {
                    outputs: vec![
                        request_output("req-2", vec![22], None),
                        request_output("req-1", vec![11], None),
                    ],
                    ..Default::default()
                },
            )
            .await;

            let abort = recv_engine_message(&mut dealer).await;
            assert_eq!(abort[0].as_ref(), &[0x01]);
            let aborted_ids: Vec<String> = rmp_serde::from_slice(&abort[1]).unwrap();
            assert_eq!(aborted_ids, vec!["req-1".to_string()]);

            send_outputs(
                &mut push,
                EngineCoreOutputs {
                    outputs: vec![request_output("req-2", vec![], Some(FinishReason::Length))],
                    finished_requests: Some(BTreeSet::from(["req-2".to_string()])),
                    ..Default::default()
                },
            )
            .await;

            send_outputs(
                &mut push,
                EngineCoreOutputs {
                    outputs: vec![request_output("req-1", vec![], Some(FinishReason::Abort))],
                    finished_requests: Some(BTreeSet::from(["req-1".to_string()])),
                    ..Default::default()
                },
            )
            .await;
        }
    });

    let client = EngineCoreClient::connect(EngineCoreClientConfig {
        handshake_address,
        model_name: "test-model".to_string(),
        local_host: "127.0.0.1".to_string(),
        ready_timeout: Duration::from_secs(2),
        client_index: 7,
    })
    .await
    .unwrap();
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

    client.shutdown().await.unwrap();
    engine_task.await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn duplicate_request_ids_are_rejected_without_sending_a_second_add() {
    init_tracing();
    let handshake_address = unique_tcp_endpoint();
    let engine_identity = b"engine-dup".to_vec();

    let engine_task = tokio::spawn({
        let engine_handshake = handshake_address.clone();
        let engine_identity = engine_identity.clone();
        async move {
            let (mut dealer, mut push) = setup_mock_engine(engine_handshake, engine_identity).await;

            let add_1 = recv_engine_message(&mut dealer).await;
            assert_eq!(add_1[0].as_ref(), &[0x00]);
            let request_1: EngineCoreRequest = rmp_serde::from_slice(&add_1[1]).unwrap();
            assert_eq!(request_1.request_id, "req-1");

            assert!(
                timeout(Duration::from_millis(200), dealer.recv())
                    .await
                    .is_err()
            );

            send_outputs(
                &mut push,
                EngineCoreOutputs {
                    outputs: vec![request_output("req-1", vec![], Some(FinishReason::Length))],
                    finished_requests: Some(BTreeSet::from(["req-1".to_string()])),
                    ..Default::default()
                },
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
    .unwrap();

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
    client.shutdown().await.unwrap();
    engine_task.await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn finished_requests_without_final_output_is_treated_as_unexpected_close() {
    init_tracing();
    let handshake_address = unique_tcp_endpoint();
    let engine_identity = b"engine-finished-only".to_vec();

    let engine_task = tokio::spawn({
        let engine_handshake = handshake_address.clone();
        let engine_identity = engine_identity.clone();
        async move {
            let (mut dealer, mut push) = setup_mock_engine(engine_handshake, engine_identity).await;

            let add = recv_engine_message(&mut dealer).await;
            assert_eq!(add[0].as_ref(), &[0x00]);

            send_outputs(
                &mut push,
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
    .unwrap();

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

    client.shutdown().await.unwrap();
    engine_task.await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dropping_a_live_stream_triggers_abort() {
    init_tracing();
    let handshake_address = unique_tcp_endpoint();
    let engine_identity = b"engine-drop".to_vec();

    let engine_task = tokio::spawn({
        let engine_handshake = handshake_address.clone();
        let engine_identity = engine_identity.clone();
        async move {
            let (mut dealer, mut push) = setup_mock_engine(engine_handshake, engine_identity).await;

            let add = recv_engine_message(&mut dealer).await;
            assert_eq!(add[0].as_ref(), &[0x00]);
            send_outputs(
                &mut push,
                EngineCoreOutputs {
                    outputs: vec![request_output("req-1", vec![99], None)],
                    ..Default::default()
                },
            )
            .await;

            let abort = timeout(Duration::from_secs(1), recv_engine_message(&mut dealer))
                .await
                .unwrap();
            assert_eq!(abort[0].as_ref(), &[0x01]);
            let aborted_ids: Vec<String> = rmp_serde::from_slice(&abort[1]).unwrap();
            assert_eq!(aborted_ids, vec!["req-1".to_string()]);
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
    .unwrap();

    let mut stream = client.call(sample_request()).await.unwrap();
    let first = timeout(Duration::from_secs(1), stream.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(first.new_token_ids, vec![99]);
    drop(stream);

    engine_task.await.unwrap();
    client.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dispatcher_failure_propagates_to_streams_and_future_calls() {
    init_tracing();
    let handshake_address = unique_tcp_endpoint();
    let engine_identity = b"engine-fail".to_vec();

    let engine_task = tokio::spawn({
        let engine_handshake = handshake_address.clone();
        let engine_identity = engine_identity.clone();
        async move {
            let (mut dealer, mut push) = setup_mock_engine(engine_handshake, engine_identity).await;

            let _ = recv_engine_message(&mut dealer).await;
            let _ = recv_engine_message(&mut dealer).await;

            push.send(ZmqMessage::from(vec![0xc1])).await.unwrap();
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
    .unwrap();

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
    assert!(is_shared_dispatcher_closed(&error_1));
    assert!(is_shared_dispatcher_closed(&error_2));

    let abort_error = client.abort(&["req-1".to_string()]).await.unwrap_err();
    assert!(matches!(abort_error, Error::DispatcherClosed { .. }));

    let add_error = match client.call(sample_request_with_id("req-3")).await {
        Ok(_) => panic!("expected dispatcher closed error"),
        Err(error) => error,
    };
    assert!(matches!(add_error, Error::DispatcherClosed { .. }));

    client.shutdown().await.unwrap();
    engine_task.await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn connect_times_out_without_ready_message() {
    init_tracing();
    let handshake_address = unique_tcp_endpoint();
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

    let result = EngineCoreClient::connect(EngineCoreClientConfig {
        handshake_address,
        model_name: "test-model".to_string(),
        local_host: "127.0.0.1".to_string(),
        ready_timeout: Duration::from_millis(100),
        client_index: 0,
    })
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

#[test]
fn python_msgpack_fixtures_match_rust_encoding() {
    init_tracing();
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/python_compat.py");
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
}
