// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use futures::{StreamExt as _, future::BoxFuture, stream::BoxStream};
use vllm_engine_core_client::EngineCoreClient;
use vllm_engine_core_client::protocol::dtype::ModelDtype;
use vllm_llm::{GenerateOutput, GenerateRequest, Llm, Result};

pub type TokenStream = BoxStream<'static, Result<GenerateOutput>>;

/// Generation, lifecycle, and model metadata used by the frontend.
pub trait EngineClient: Send + Sync {
    /// Submit a tokenized request and return its output stream.
    fn generate<'a>(&'a self, request: GenerateRequest) -> BoxFuture<'a, Result<TokenStream>>;

    fn abort<'a>(&'a self, request_ids: &'a [String]) -> BoxFuture<'a, Result<()>>;
    fn shutdown(self: Box<Self>) -> BoxFuture<'static, Result<()>>;
    fn max_model_len(&self) -> u32;
    fn model_dtype(&self) -> Option<ModelDtype>;
    fn is_healthy(&self) -> bool;
    fn supports_lora(&self) -> bool;

    /// Access native administration for routes that require the ZMQ client.
    fn as_engine_core_client(&self) -> Option<&EngineCoreClient> {
        None
    }
}

impl EngineClient for Llm {
    fn generate<'a>(&'a self, request: GenerateRequest) -> BoxFuture<'a, Result<TokenStream>> {
        Box::pin(async move { Ok(Llm::generate(self, request).await?.boxed()) })
    }

    fn abort<'a>(&'a self, request_ids: &'a [String]) -> BoxFuture<'a, Result<()>> {
        Box::pin(Llm::abort(self, request_ids))
    }

    fn shutdown(self: Box<Self>) -> BoxFuture<'static, Result<()>> {
        Box::pin(Llm::shutdown(*self))
    }

    fn max_model_len(&self) -> u32 {
        self.engine_core_client().max_model_len()
    }

    fn model_dtype(&self) -> Option<ModelDtype> {
        Some(self.engine_core_client().model_dtype())
    }

    fn is_healthy(&self) -> bool {
        self.engine_core_client().is_healthy()
    }

    fn supports_lora(&self) -> bool {
        self.engine_core_client().ready_response().supports_lora
    }

    fn as_engine_core_client(&self) -> Option<&EngineCoreClient> {
        Some(self.engine_core_client())
    }
}
