use std::net::TcpListener;
use std::time::Duration;

use anyhow::Result;
use futures::StreamExt as _;
use tokio::time::timeout;
use tokio_util::sync::CancellationToken;
use vllm_engine_core_client::protocol::{
    EngineCoreFinishReason, EngineCoreRequest, EngineCoreSamplingParams,
};
use vllm_engine_core_client::test_utils::IpcNamespace;
use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig, TransportMode};

use crate::{Opt, run};

fn free_tcp_address() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind free port");
    let port = listener.local_addr().expect("local addr").port();
    drop(listener);
    format!("tcp://127.0.0.1:{port}")
}

fn client_config(handshake_address: String, engine_count: usize) -> EngineCoreClientConfig {
    EngineCoreClientConfig {
        transport_mode: TransportMode::HandshakeOwner {
            handshake_address,
            advertised_host: "127.0.0.1".to_string(),
            engine_count,
            ready_timeout: Duration::from_secs(5),
            local_input_address: None,
            local_output_address: None,
        },
        coordinator_mode: None,
        model_name: "mock-model".to_string(),
        client_index: 0,
    }
}

async fn connect_with_mock(
    handshake_address: String,
    engine_count: usize,
    output_token_chunk_size: usize,
) -> (
    EngineCoreClient,
    CancellationToken,
    tokio::task::JoinHandle<Result<()>>,
) {
    let shutdown = CancellationToken::new();
    let task = tokio::spawn(run(
        Opt {
            handshake_address: handshake_address.clone(),
            engine_count,
            output_token_chunk_size,
            vocab_size: 32_000,
            seed: 0,
            log_requests: false,
        },
        shutdown.clone(),
    ));

    let client = timeout(
        Duration::from_secs(5),
        EngineCoreClient::connect(client_config(handshake_address, engine_count)),
    )
    .await
    .expect("client connect timeout")
    .expect("connect client");

    (client, shutdown, task)
}

fn sample_request(request_id: &str, max_tokens: u32) -> EngineCoreRequest {
    EngineCoreRequest {
        request_id: request_id.to_string(),
        prompt_token_ids: Some(vec![1, 2, 3]),
        sampling_params: Some(EngineCoreSamplingParams {
            max_tokens,
            ..EngineCoreSamplingParams::for_test()
        }),
        arrival_time: 0.0,
        ..Default::default()
    }
}

async fn shutdown_mock(
    client: EngineCoreClient,
    shutdown: CancellationToken,
    task: tokio::task::JoinHandle<Result<()>>,
) {
    client.shutdown().await.expect("client shutdown");
    shutdown.cancel();
    task.await.expect("mock join").expect("mock run");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mock_engine_connects_over_tcp() {
    let handshake_address = free_tcp_address();
    let (client, shutdown, task) = connect_with_mock(handshake_address, 1, 1).await;
    assert_eq!(client.engine_count(), 1);
    assert_eq!(client.engine_identities()[0], &[0, 0]);
    assert_eq!(client.max_model_len(), 1024 * 1024);
    assert_eq!(client.vllm_version(), "test-vllm-version");
    shutdown_mock(client, shutdown, task).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mock_engine_connects_over_ipc() {
    let ipc = IpcNamespace::new().expect("ipc namespace");
    let handshake_address = ipc.handshake_endpoint();
    let (client, shutdown, task) = connect_with_mock(handshake_address, 1, 1).await;
    assert_eq!(client.engine_count(), 1);
    assert_eq!(client.engine_identities()[0], &[0, 0]);
    shutdown_mock(client, shutdown, task).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mock_engine_registers_multiple_identities() {
    let handshake_address = free_tcp_address();
    let (client, shutdown, task) = connect_with_mock(handshake_address, 2, 1).await;
    assert_eq!(client.engine_count(), 2);
    assert_eq!(client.engine_identities(), vec![&[0, 0][..], &[1, 0][..]]);
    shutdown_mock(client, shutdown, task).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chunk_size_one_outputs_one_token_per_update() {
    let handshake_address = free_tcp_address();
    let (client, shutdown, task) = connect_with_mock(handshake_address, 1, 1).await;
    let mut stream = client.call(sample_request("req-1", 3)).await.expect("call");

    let first = stream.next().await.expect("first").expect("first ok");
    assert_eq!(first.new_token_ids.len(), 1);
    assert_eq!(first.finish_reason, None);
    let second = stream.next().await.expect("second").expect("second ok");
    assert_eq!(second.new_token_ids.len(), 1);
    assert_eq!(second.finish_reason, None);
    let third = stream.next().await.expect("third").expect("third ok");
    assert_eq!(third.new_token_ids.len(), 1);
    assert_eq!(third.finish_reason, Some(EngineCoreFinishReason::Length));
    assert!(stream.next().await.is_none());

    shutdown_mock(client, shutdown, task).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chunk_size_clips_final_output_to_max_tokens() {
    let handshake_address = free_tcp_address();
    let (client, shutdown, task) = connect_with_mock(handshake_address, 1, 4).await;
    let mut stream = client.call(sample_request("req-clip", 6)).await.expect("call");

    let first = stream.next().await.expect("first").expect("first ok");
    assert_eq!(first.new_token_ids.len(), 4);
    assert_eq!(first.finish_reason, None);
    let second = stream.next().await.expect("second").expect("second ok");
    assert_eq!(second.new_token_ids.len(), 2);
    assert_eq!(second.finish_reason, Some(EngineCoreFinishReason::Length));
    assert!(stream.next().await.is_none());

    shutdown_mock(client, shutdown, task).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn abort_cancels_active_request_and_emits_terminal_output() {
    let handshake_address = free_tcp_address();
    let (client, shutdown, task) = connect_with_mock(handshake_address, 1, 1).await;
    let mut stream = client.call(sample_request("req-abort", 1_000_000)).await.expect("call");
    let first = stream.next().await.expect("first").expect("first ok");
    assert_eq!(first.finish_reason, None);

    client.abort(&["req-abort".to_string()]).await.expect("abort");

    loop {
        let output = timeout(Duration::from_secs(5), stream.next())
            .await
            .expect("stream timeout")
            .expect("terminal output")
            .expect("output ok");
        if output.finish_reason.is_some() {
            assert_eq!(output.finish_reason, Some(EngineCoreFinishReason::Abort));
            break;
        }
    }

    shutdown_mock(client, shutdown, task).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn utility_requests_return_minimal_success_responses() {
    let handshake_address = free_tcp_address();
    let (client, shutdown, task) = connect_with_mock(handshake_address, 1, 1).await;

    assert!(!client.is_sleeping().await.expect("is sleeping"));
    assert!(client.reset_prefix_cache(false, false).await.expect("reset prefix cache"));
    client.reset_mm_cache().await.expect("reset mm cache");
    client.reset_encoder_cache().await.expect("reset encoder cache");

    shutdown_mock(client, shutdown, task).await;
}
