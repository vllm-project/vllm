use std::convert::TryFrom;
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Once;
use std::time::Duration;

use thiserror_ext::AsReport as _;
use tracing_subscriber::EnvFilter;
use vllm_engine_core_client::protocol::handshake::HandshakeInitMessage;
use vllm_engine_core_client::{
    EngineCoreClient, EngineCoreOutput, EngineCoreOutputs, EngineCoreRequest, FinishReason,
    ReadyMessage, RequestOutputKind, SamplingParams, ZmqEngineCoreClient,
    ZmqEngineCoreClientConfig,
};
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
    EngineCoreRequest {
        request_id: "req-1".to_string(),
        prompt_token_ids: Some(vec![11, 22]),
        mm_features: None,
        sampling_params: Some(SamplingParams {
            n: 2,
            temperature: 0.8,
            top_p: 0.9,
            top_k: 8,
            max_tokens: Some(32),
            min_tokens: 1,
            stop: vec!["stop".to_string()],
            stop_token_ids: vec![151643],
            ignore_eos: true,
            output_kind: RequestOutputKind::FinalOnly,
            structured_outputs: None,
            logit_bias: None,
            allowed_token_ids: Some(vec![1, 2, 3]),
            extra_args: None,
            repetition_detection: None,
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
async fn client_roundtrip_add_abort_and_finish() {
    init_tracing();
    let handshake_address = unique_tcp_endpoint();
    let engine_identity = b"engine-0".to_vec();

    let engine_handshake = handshake_address.clone();
    let engine_id = engine_identity.clone();
    let engine_task = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(200)).await;

        let mut options = SocketOptions::default();
        options.peer_identity(PeerIdentity::try_from(engine_id.clone()).unwrap());
        let mut handshake = DealerSocket::with_options(options);
        handshake.connect(&engine_handshake).await.unwrap();

        let hello = rmp_serde::to_vec_named(&ReadyMessage {
            status: Some("HELLO".to_string()),
            local: Some(true),
            headless: Some(true),
            num_gpu_blocks: None,
            dp_stats_address: None,
            parallel_config_hash: None,
        })
        .unwrap();
        handshake.send(ZmqMessage::from(hello)).await.unwrap();

        let init_frames = handshake.recv().await.unwrap().into_vec();
        assert_eq!(init_frames.len(), 1);
        let init: HandshakeInitMessage = rmp_serde::from_slice(init_frames[0].as_ref()).unwrap();
        assert_eq!(init.addresses.inputs.len(), 1);
        assert_eq!(init.addresses.outputs.len(), 1);

        let engine_input = init.addresses.inputs[0].clone();
        let engine_output = init.addresses.outputs[0].clone();

        let mut input_options = SocketOptions::default();
        input_options.peer_identity(PeerIdentity::try_from(engine_id).unwrap());
        let mut dealer = DealerSocket::with_options(input_options);
        dealer.connect(&engine_input).await.unwrap();
        dealer
            .send(ZmqMessage::from(Vec::<u8>::new()))
            .await
            .unwrap();

        let mut push = PushSocket::new();
        push.connect(&engine_output).await.unwrap();

        let ready = rmp_serde::to_vec_named(&ReadyMessage {
            status: Some("READY".to_string()),
            local: Some(true),
            headless: Some(true),
            num_gpu_blocks: None,
            dp_stats_address: None,
            parallel_config_hash: None,
        })
        .unwrap();
        handshake.send(ZmqMessage::from(ready)).await.unwrap();

        let add_message = dealer.recv().await.unwrap().into_vec();
        assert_eq!(add_message[0].as_ref(), &[0x00]);
        let request: EngineCoreRequest = rmp_serde::from_slice(&add_message[1]).unwrap();
        assert_eq!(request.client_index, 7);
        assert_eq!(request.request_id, "req-1");

        let partial = EngineCoreOutputs {
            outputs: vec![EngineCoreOutput {
                request_id: "req-1".to_string(),
                new_token_ids: vec![99],
                new_logprobs: None,
                new_prompt_logprobs_tensors: None,
                pooling_output: None,
                finish_reason: None,
                stop_reason: None,
                events: None,
                kv_transfer_params: None,
                trace_headers: None,
                num_cached_tokens: 0,
                num_external_computed_tokens: 0,
                routed_experts: None,
                num_nans_in_logits: 0,
            }],
            ..Default::default()
        };
        push.send(ZmqMessage::from(rmp_serde::to_vec_named(&partial).unwrap()))
            .await
            .unwrap();

        let abort_message = dealer.recv().await.unwrap().into_vec();
        assert_eq!(abort_message[0].as_ref(), &[0x01]);
        let aborted_ids: Vec<String> = rmp_serde::from_slice(&abort_message[1]).unwrap();
        assert_eq!(aborted_ids, vec!["req-1".to_string()]);

        let finished = EngineCoreOutputs {
            outputs: vec![EngineCoreOutput {
                request_id: "req-1".to_string(),
                new_token_ids: vec![],
                new_logprobs: None,
                new_prompt_logprobs_tensors: None,
                pooling_output: None,
                finish_reason: Some(FinishReason::Abort),
                stop_reason: None,
                events: None,
                kv_transfer_params: None,
                trace_headers: None,
                num_cached_tokens: 0,
                num_external_computed_tokens: 0,
                routed_experts: None,
                num_nans_in_logits: 0,
            }],
            finished_requests: Some(["req-1".to_string()].into_iter().collect()),
            ..Default::default()
        };
        push.send(ZmqMessage::from(
            rmp_serde::to_vec_named(&finished).unwrap(),
        ))
        .await
        .unwrap();
    });

    let mut client = ZmqEngineCoreClient::connect(ZmqEngineCoreClientConfig {
        handshake_address,
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
    assert!(client.input_address().starts_with("tcp://127.0.0.1:"));
    assert!(client.output_address().starts_with("tcp://127.0.0.1:"));

    client.add_request(sample_request()).await.unwrap();
    let first = client.next_output().await.unwrap();
    assert_eq!(first.outputs[0].new_token_ids, vec![99]);
    assert_eq!(first.outputs[0].finish_reason, None);

    client
        .abort_requests(&["req-1".to_string(), "unknown".to_string()])
        .await
        .unwrap();
    let final_output = client.next_output().await.unwrap();
    assert_eq!(
        final_output.outputs[0].finish_reason,
        Some(FinishReason::Abort)
    );

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

        let hello = rmp_serde::to_vec_named(&ReadyMessage {
            status: Some("HELLO".to_string()),
            local: Some(true),
            headless: Some(true),
            num_gpu_blocks: None,
            dp_stats_address: None,
            parallel_config_hash: None,
        })
        .unwrap();
        handshake.send(ZmqMessage::from(hello)).await.unwrap();

        let _ = handshake.recv().await.unwrap();
    });

    let result = ZmqEngineCoreClient::connect(ZmqEngineCoreClientConfig {
        handshake_address,
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
    assert_eq!(decoded_request.request_id, expected_request.request_id);
    assert_eq!(
        decoded_request.prompt_token_ids,
        expected_request.prompt_token_ids
    );
    assert_eq!(
        decoded_request.sampling_params,
        expected_request.sampling_params
    );
    assert_eq!(decoded_request.arrival_time, expected_request.arrival_time);

    let decoded_outputs: EngineCoreOutputs = rmp_serde::from_slice(&outputs_bytes).unwrap();
    assert_eq!(decoded_outputs.outputs.len(), 1);
    assert_eq!(decoded_outputs.outputs[0].request_id, "req-1");
    assert_eq!(decoded_outputs.outputs[0].new_token_ids, vec![7, 8]);
    assert_eq!(
        decoded_outputs.outputs[0].finish_reason,
        Some(FinishReason::Length)
    );
    assert_eq!(
        decoded_outputs.finished_requests,
        Some(["req-1".to_string()].into_iter().collect())
    );
}
