use std::collections::BTreeSet;
use std::convert::TryFrom;
use std::net::TcpListener;
use std::sync::Once;
use std::time::Duration;

use futures::StreamExt as _;
use tokio::time::timeout;
use tracing_subscriber::EnvFilter;
use vllm_engine_core_client::protocol::handshake::{HandshakeInitMessage, ReadyMessage};
use vllm_engine_core_client::protocol::{
    EngineCoreOutput, EngineCoreOutputs, EngineCoreRequest, EngineCoreSamplingParams, FinishReason,
    RequestOutputKind,
};
use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig};
use vllm_llm::{Error, GenerateRequest, Llm};
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

async fn connect_async_llm(handshake_address: String, client_index: u32) -> Llm {
    let client = EngineCoreClient::connect(EngineCoreClientConfig {
        handshake_address,
        local_host: "127.0.0.1".to_string(),
        ready_timeout: Duration::from_secs(2),
        client_index,
    })
    .await
    .unwrap();
    Llm::new(client)
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
    let handshake_address = unique_tcp_endpoint();
    let engine_identity = b"engine-delta".to_vec();

    let engine_task = tokio::spawn({
        let engine_handshake = handshake_address.clone();
        let engine_identity = engine_identity.clone();
        async move {
            let (mut dealer, mut push) = setup_mock_engine(engine_handshake, engine_identity).await;

            let add = recv_engine_message(&mut dealer).await;
            assert_eq!(add[0].as_ref(), &[0x00]);
            let request: EngineCoreRequest = rmp_serde::from_slice(&add[1]).unwrap();
            assert_eq!(request.request_id, "req-delta");
            assert_eq!(request.client_index, 7);
            assert_eq!(request.prompt_token_ids, Some(vec![11, 22]));

            send_outputs(
                &mut push,
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
        }
    });

    let llm = connect_async_llm(handshake_address, 7).await;
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

    llm.shutdown().await.unwrap();
    engine_task.await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn generate_streams_cumulative_outputs() {
    let handshake_address = unique_tcp_endpoint();
    let engine_identity = b"engine-cumulative".to_vec();

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
                    outputs: vec![
                        request_output("req-cumulative", vec![9], None),
                        request_output("req-cumulative", vec![10, 11], Some(FinishReason::Length)),
                    ],
                    finished_requests: Some(BTreeSet::from(["req-cumulative".to_string()])),
                    ..Default::default()
                },
            )
            .await;
        }
    });

    let llm = connect_async_llm(handshake_address, 0).await;
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

    llm.shutdown().await.unwrap();
    engine_task.await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn generate_streams_final_only_outputs() {
    let handshake_address = unique_tcp_endpoint();
    let engine_identity = b"engine-final".to_vec();

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
                    outputs: vec![
                        request_output("req-final", vec![4], None),
                        request_output("req-final", vec![5, 6], Some(FinishReason::Length)),
                    ],
                    finished_requests: Some(BTreeSet::from(["req-final".to_string()])),
                    ..Default::default()
                },
            )
            .await;
        }
    });

    let llm = connect_async_llm(handshake_address, 0).await;
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

    llm.shutdown().await.unwrap();
    engine_task.await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn generate_propagates_unexpected_close_errors() {
    let handshake_address = unique_tcp_endpoint();
    let engine_identity = b"engine-close".to_vec();

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
                    finished_requests: Some(BTreeSet::from(["req-close".to_string()])),
                    ..Default::default()
                },
            )
            .await;
        }
    });

    let llm = connect_async_llm(handshake_address, 0).await;
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

    llm.shutdown().await.unwrap();
    engine_task.await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn abort_forwards_to_engine_core_client() {
    let handshake_address = unique_tcp_endpoint();
    let engine_identity = b"engine-abort".to_vec();

    let engine_task = tokio::spawn({
        let engine_handshake = handshake_address.clone();
        let engine_identity = engine_identity.clone();
        async move {
            let (mut dealer, _push) = setup_mock_engine(engine_handshake, engine_identity).await;

            let add = recv_engine_message(&mut dealer).await;
            assert_eq!(add[0].as_ref(), &[0x00]);

            let abort = timeout(Duration::from_secs(1), recv_engine_message(&mut dealer))
                .await
                .unwrap();
            assert_eq!(abort[0].as_ref(), &[0x01]);
            let aborted_ids: Vec<String> = rmp_serde::from_slice(&abort[1]).unwrap();
            assert_eq!(aborted_ids, vec!["req-abort".to_string()]);
        }
    });

    let llm = connect_async_llm(handshake_address, 0).await;
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
    llm.shutdown().await.unwrap();
    engine_task.await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dropping_a_live_generate_stream_triggers_abort() {
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
                    outputs: vec![request_output("req-drop", vec![99], None)],
                    ..Default::default()
                },
            )
            .await;

            let abort = timeout(Duration::from_secs(1), recv_engine_message(&mut dealer))
                .await
                .unwrap();
            assert_eq!(abort[0].as_ref(), &[0x01]);
            let aborted_ids: Vec<String> = rmp_serde::from_slice(&abort[1]).unwrap();
            assert_eq!(aborted_ids, vec!["req-drop".to_string()]);
        }
    });

    let llm = connect_async_llm(handshake_address, 0).await;
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

    engine_task.await.unwrap();
    llm.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn duplicate_request_ids_bubble_up_from_engine_core_client() {
    let handshake_address = unique_tcp_endpoint();
    let engine_identity = b"engine-dup".to_vec();

    let engine_task = tokio::spawn({
        let engine_handshake = handshake_address.clone();
        let engine_identity = engine_identity.clone();
        async move {
            let (mut dealer, mut push) = setup_mock_engine(engine_handshake, engine_identity).await;

            let add = recv_engine_message(&mut dealer).await;
            assert_eq!(add[0].as_ref(), &[0x00]);

            assert!(
                timeout(Duration::from_millis(200), dealer.recv())
                    .await
                    .is_err()
            );

            send_outputs(
                &mut push,
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
        }
    });

    let llm = connect_async_llm(handshake_address, 0).await;
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
    drop(stream_1);

    llm.shutdown().await.unwrap();
    engine_task.await.unwrap();
}
