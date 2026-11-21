// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! A router built around a caller-supplied `Llm` (`serve_with_llm`) instead of
//! an engine-core connection.

use std::num::NonZeroU32;
use std::sync::Arc;

use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode};
use futures::stream;
use serde_json::json;
use serial_test::serial;
use tokio::sync::watch;
use tower::ServiceExt as _;
use vllm_chat::ChatLlm;
use vllm_llm::{
    BoxGenerateOutputStream, FinishReason, GenerateOutput, GeneratePromptInfo, GenerateRequest,
    Llm, LlmInfo, PromptCacheUsage, RequestTimestamps,
};

use super::super::build_router;
use super::FakeChatBackend;
use crate::state::AppState;

/// Answers every request with the token IDs `[b'o', b'k']`, then stops at its length limit.
struct FixedLlm {
    info: LlmInfo,
    health: watch::Sender<bool>,
}

impl FixedLlm {
    fn new() -> Self {
        Self {
            info: LlmInfo {
                max_model_len: 1024,
                model_dtype: None,
                supports_lora: false,
                vllm_version: "custom".to_string(),
            },
            health: watch::channel(true).0,
        }
    }
}

#[async_trait::async_trait]
impl Llm for FixedLlm {
    fn info(&self) -> &LlmInfo {
        &self.info
    }

    fn subscribe_health(&self) -> watch::Receiver<bool> {
        self.health.subscribe()
    }

    fn subscribe_readiness(&self) -> watch::Receiver<bool> {
        self.health.subscribe()
    }

    async fn generate_n(
        &self,
        request: GenerateRequest,
        n: NonZeroU32,
    ) -> vllm_llm::Result<Vec<BoxGenerateOutputStream>> {
        Ok((0..n.get())
            .map(|_| {
                let output = GenerateOutput {
                    request_id: request.request_id.clone(),
                    prompt_info: Some(Box::new(GeneratePromptInfo {
                        prompt_token_ids: request.prompt_token_ids.as_slice().into(),
                        prompt_logprobs: None,
                        prompt_token_id_logprobs: None,
                    })),
                    token_ids: vec![u32::from(b'o'), u32::from(b'k')],
                    logprobs: None,
                    finish_reason: Some(FinishReason::Length),
                    prompt_cache_usage: PromptCacheUsage::default(),
                    kv_transfer_params: None,
                    ec_transfer_params: None,
                    sampling_mask: None,
                    spec_decode_metrics: None,
                    weight_versions: None,
                    routed_experts: Default::default(),
                    timestamps: RequestTimestamps::default(),
                };
                let stream: BoxGenerateOutputStream = Box::pin(stream::iter([Ok(output)]));
                stream
            })
            .collect())
    }

    async fn abort(&self, _external_ids: &[String]) -> vllm_llm::Result<()> {
        Ok(())
    }

    async fn shutdown(self: Box<Self>) -> vllm_llm::Result<()> {
        Ok(())
    }
}

fn custom_llm_app() -> axum::Router {
    let chat = ChatLlm::from_shared_backend(FixedLlm::new(), Arc::new(FakeChatBackend::new()))
        .with_served_model_name("model");
    build_router(Arc::new(AppState::new(vec!["model".to_string()], chat)))
}

#[tokio::test]
#[serial]
async fn custom_llm_serves_completions() {
    let request = Request::post("/v1/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({"model": "model", "prompt": "hi", "max_tokens": 4}).to_string(),
        ))
        .unwrap();
    let response = custom_llm_app().oneshot(request).await.unwrap();
    let status = response.status();
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(body["choices"][0]["text"], "ok");
    assert_eq!(body["choices"][0]["finish_reason"], "length");
}
