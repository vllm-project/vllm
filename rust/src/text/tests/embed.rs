// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeSet;
use std::sync::Arc;

use vllm_engine_core_client::protocol::output::{
    EngineCoreFinishReason, EngineCoreOutput, EngineCoreOutputs, RequestBatchOutputs,
    UtilityCallOutput,
};
use vllm_engine_core_client::protocol::request::EngineCoreRequest;
use vllm_engine_core_client::protocol::stats::PrefillStats;
use vllm_engine_core_client::protocol::tensor::WireTensor;
use vllm_engine_core_client::protocol::utility::{
    EngineCoreUtilityRequest, UtilityOutput, UtilityResultEnvelope,
};
use vllm_engine_core_client::test_utils::{IpcNamespace, spawn_mock_engine_task};
use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig};
use vllm_llm::{Llm, PoolingTask};
use vllm_text::tokenizer::DynTokenizer;
use vllm_text::tokenizer::test_utils::TestTokenizer;
use vllm_text::{EmbeddingRequest, TextBackend, TextLlm};
use zeromq::prelude::{SocketRecv, SocketSend};
use zeromq::{DealerSocket, PushSocket, ZmqMessage};

struct FakeEmbeddingBackend {
    tokenizer: DynTokenizer,
}

impl TextBackend for FakeEmbeddingBackend {
    fn tokenizer(&self) -> DynTokenizer {
        self.tokenizer.clone()
    }

    fn model_id(&self) -> &str {
        "test-embedding-model"
    }

    fn model_vocab_size(&self) -> usize {
        512
    }

    fn tokenizer_vocab_size(&self) -> usize {
        512
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

async fn answer_supported_tasks(dealer: &mut DealerSocket, push: &mut PushSocket) {
    let frames = dealer.recv().await.unwrap().into_vec();
    assert_eq!(frames[0].as_ref(), &[0x03]);
    let request: EngineCoreUtilityRequest = rmp_serde::from_slice(&frames[1]).unwrap();
    assert_eq!(request.method_name, "get_supported_tasks");

    send_outputs(
        push,
        UtilityCallOutput {
            engine_index: 0,
            timestamp: 0.0,
            output: UtilityOutput {
                call_id: request.call_id,
                failure_message: None,
                result: Some(UtilityResultEnvelope::without_type_info(
                    rmpv::ext::to_value(vec!["embed"]).unwrap(),
                )),
            },
        }
        .into(),
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn text_llm_embeds_text_through_the_token_level_encode_boundary() {
    let ipc = IpcNamespace::new().unwrap();
    let handshake_address = ipc.handshake_endpoint();

    let (shutdown_tx, engine_task) = spawn_mock_engine_task(
        handshake_address.clone(),
        b"engine-text-embed".to_vec(),
        |dealer, push| {
            Box::pin(async move {
                answer_supported_tasks(dealer, push).await;
                let request = recv_engine_request(dealer).await;
                assert_eq!(request.external_req_id.as_deref(), Some("text-embed-1"));
                assert!(request.request_id.starts_with("text-embed-1-"));
                assert_eq!(
                    request.prompt_token_ids.as_deref(),
                    Some(&[300, 97, 98, 99][..])
                );
                let params = request.pooling_params.as_ref().unwrap();
                assert_eq!(params.task, PoolingTask::Embed);
                assert_eq!(params.use_activation, None);
                assert_eq!(params.dimensions, None);

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
                                num_prompt_tokens: 4,
                                num_computed_tokens: 3,
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

    let client = EngineCoreClient::connect(
        EngineCoreClientConfig::new_single(handshake_address)
            .with_model_name("test-embedding-model")
            .with_local_input_output_addresses(
                Some(ipc.input_endpoint()),
                Some(ipc.output_endpoint()),
            ),
    )
    .await
    .unwrap();
    let tokenizer = TestTokenizer::new().with_bos_token("<s>", 300).with_vocab_size(512);
    let text = TextLlm::new(
        Llm::new(client),
        Arc::new(FakeEmbeddingBackend {
            tokenizer: Arc::new(tokenizer),
        }),
    );

    let mut request = EmbeddingRequest::for_test();
    request.request_id = "text-embed-1".to_string();
    request.prompt = vllm_text::Prompt::Text("abc".to_string());
    request.arrival_time = Some(42.5);
    let output = text.embed(request).await.unwrap();

    assert_eq!(output.request_id, "text-embed-1");
    assert_eq!(output.prompt_token_ids, vec![300, 97, 98, 99]);
    assert_eq!(output.embedding, vec![0.25, -0.5]);
    assert_eq!(output.cached_token_count, 1);

    let _ = shutdown_tx.send(());
    engine_task.await.unwrap();
    text.shutdown().await.unwrap();
}
