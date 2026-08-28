// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::oneshot;
use tokio::time::timeout;
use uuid::Uuid;
use vllm_engine_core_client::protocol::output::{
    EngineCoreFinishReason, EngineCoreOutput, EngineCoreOutputs, RequestBatchOutputs,
};
use vllm_engine_core_client::protocol::pooling::PoolingTask;
use vllm_engine_core_client::protocol::request::EngineCoreRequest;
use vllm_engine_core_client::protocol::stats::PrefillStats;
use vllm_engine_core_client::protocol::tensor::WireTensor;
use vllm_engine_core_client::test_utils::{IpcNamespace, spawn_mock_engine_task};
use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig};
use vllm_llm::{EncodeRequest, Error, Llm, PoolingParams};
use vllm_metrics::METRICS;
use zeromq::prelude::{SocketRecv, SocketSend};
use zeromq::{DealerSocket, PushSocket, ZmqMessage};

fn sample_request() -> EncodeRequest {
    EncodeRequest {
        request_id: "embed-1".to_string(),
        prompt_token_ids: vec![11, 22],
        task: PoolingTask::Embed,
        pooling_params: PoolingParams {
            use_activation: false,
            dimensions: Some(2),
            step_tag_id: None,
            returned_token_ids: None,
        },
        arrival_time: Some(42.5),
        cache_salt: None,
        trace_headers: None,
        priority: 0,
        data_parallel_rank: None,
        session_id: None,
        lora_request: None,
    }
}

async fn recv_engine_request(dealer: &mut DealerSocket) -> EngineCoreRequest {
    let frames = dealer.recv().await.unwrap().into_vec();
    assert_eq!(frames[0].as_ref(), &[0x00]);
    rmp_serde::from_slice(&frames[1]).unwrap()
}

async fn send_outputs(push: &mut PushSocket, outputs: EngineCoreOutputs) {
    push.send(ZmqMessage::from(rmp_serde::to_vec_named(&outputs).unwrap()))
        .await
        .unwrap();
}

async fn connect_llm(handshake_address: String, model_name: &str, ipc: &IpcNamespace) -> Llm {
    let client = EngineCoreClient::connect(
        EngineCoreClientConfig::new_single(handshake_address)
            .with_model_name(model_name)
            .with_local_input_output_addresses(
                Some(ipc.input_endpoint()),
                Some(ipc.output_endpoint()),
            ),
    )
    .await
    .unwrap();
    Llm::new(client).with_request_id_randomization(false)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn encode_lowers_request_and_collects_final_pooling_tensor() {
    let ipc = IpcNamespace::new().unwrap();
    let handshake_address = ipc.handshake_endpoint();
    let model_name = format!("encode-metrics-{}", Uuid::new_v4().simple());

    let (shutdown_tx, engine_task) = spawn_mock_engine_task(
        handshake_address.clone(),
        b"engine-encode".to_vec(),
        |dealer, push| {
            Box::pin(async move {
                let request = recv_engine_request(dealer).await;
                let params = request.pooling_params.as_ref().unwrap();
                assert_eq!(request.external_req_id.as_deref(), Some("embed-1"));
                assert_eq!(request.prompt_token_ids.as_deref(), Some(&[11, 22][..]));
                assert_eq!(params.task, PoolingTask::Embed);
                assert!(!params.use_activation);
                assert_eq!(params.dimensions, Some(2));

                send_outputs(
                    push,
                    RequestBatchOutputs {
                        outputs: vec![EngineCoreOutput {
                            request_id: request.request_id.clone(),
                            pooling_output: Some(
                                WireTensor::from_f32(vec![2], vec![0.25, -0.5]).unwrap(),
                            ),
                            finish_reason: Some(EngineCoreFinishReason::Stop),
                            prefill_stats: Some(PrefillStats {
                                num_prompt_tokens: 2,
                                num_computed_tokens: 1,
                                num_cached_tokens: 1,
                                num_local_cached_tokens: 1,
                                ..Default::default()
                            }),
                            ..Default::default()
                        }],
                        finished_requests: Some(BTreeSet::from([request.request_id])),
                        ..Default::default()
                    }
                    .into(),
                )
                .await;
            })
        },
    );

    let llm = connect_llm(handshake_address, &model_name, &ipc).await;
    let mut request = sample_request();
    request.arrival_time = None;
    let output = llm.encode(request).await.unwrap();

    assert_eq!(output.request_id, "embed-1");
    assert_eq!(output.prompt_token_ids, vec![11, 22]);
    assert_eq!(output.output.shape, vec![2]);
    assert_eq!(output.output.data, vec![0.25, -0.5]);
    assert_eq!(output.cached_token_count, 1);

    let rendered = METRICS.render().unwrap();
    assert!(rendered.contains(&format!(
        "vllm:request_success_total{{model_name=\"{model_name}\",engine=\"0\",finished_reason=\"stop\"}} 1"
    )));
    assert!(rendered.contains(&format!(
        "vllm:prompt_tokens_total{{model_name=\"{model_name}\",engine=\"0\"}} 2"
    )));
    assert!(rendered.contains(&format!(
        "vllm:prompt_tokens_cached_total{{model_name=\"{model_name}\",engine=\"0\"}} 1"
    )));
    assert!(rendered.contains(&format!(
        "vllm:time_to_first_token_seconds_count{{model_name=\"{model_name}\",engine=\"0\"}} 0"
    )));

    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    llm.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn encode_reports_terminal_output_without_pooling_tensor() {
    let ipc = IpcNamespace::new().unwrap();
    let handshake_address = ipc.handshake_endpoint();

    let (shutdown_tx, engine_task) = spawn_mock_engine_task(
        handshake_address.clone(),
        b"engine-missing-pooling".to_vec(),
        |dealer, push| {
            Box::pin(async move {
                let request = recv_engine_request(dealer).await;
                send_outputs(
                    push,
                    RequestBatchOutputs {
                        outputs: vec![EngineCoreOutput {
                            request_id: request.request_id.clone(),
                            finish_reason: Some(EngineCoreFinishReason::Stop),
                            ..Default::default()
                        }],
                        finished_requests: Some(BTreeSet::from([request.request_id])),
                        ..Default::default()
                    }
                    .into(),
                )
                .await;
            })
        },
    );

    let llm = connect_llm(handshake_address, "test-model", &ipc).await;
    let error = llm.encode(sample_request()).await.unwrap_err();
    assert!(matches!(
        error,
        Error::PoolingRequest { request_id, message }
            if request_id == "embed-1"
                && message == "request finished without a pooling output"
    ));

    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    llm.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancelling_encode_future_aborts_engine_request() {
    let ipc = IpcNamespace::new().unwrap();
    let handshake_address = ipc.handshake_endpoint();
    let (request_received_tx, request_received_rx) = oneshot::channel();

    let (shutdown_tx, engine_task) = spawn_mock_engine_task(
        handshake_address.clone(),
        b"engine-cancel-encode".to_vec(),
        move |dealer, _push| {
            Box::pin(async move {
                let request = recv_engine_request(dealer).await;
                request_received_tx.send(()).unwrap();

                let abort = timeout(Duration::from_secs(1), dealer.recv()).await.unwrap().unwrap();
                let frames = abort.into_vec();
                assert_eq!(frames[0].as_ref(), &[0x01]);
                let aborted_ids: Vec<String> = rmp_serde::from_slice(&frames[1]).unwrap();
                assert_eq!(aborted_ids, vec![request.request_id]);
            })
        },
    );

    let llm = Arc::new(connect_llm(handshake_address, "test-model", &ipc).await);
    let encode_task = tokio::spawn({
        let llm = Arc::clone(&llm);
        async move { llm.encode(sample_request()).await }
    });
    request_received_rx.await.unwrap();
    encode_task.abort();
    assert!(encode_task.await.unwrap_err().is_cancelled());

    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    Arc::into_inner(llm)
        .expect("encode task released its Llm handle")
        .shutdown()
        .await
        .unwrap();
}
