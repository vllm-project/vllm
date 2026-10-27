// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeMap;
use std::sync::Arc;

use futures::{StreamExt as _, future::BoxFuture, stream::BoxStream};
use serde_json::Value as JsonValue;
use tokio::sync::watch;
use vllm_engine_core_client::protocol::OpaqueValue;
use vllm_engine_core_client::protocol::dtype::ModelDtype;
use vllm_engine_core_client::protocol::handshake::EngineCoreReadyResponse;
use vllm_engine_core_client::protocol::lora::LoraRequest;
use vllm_engine_core_client::protocol::utility::PauseMode;
use vllm_engine_core_client::{EngineCoreClient, Error as EngineCoreError};
use vllm_llm::{Error, GenerateOutput, GenerateRequest, Llm, Result};

pub type TokenStream = BoxStream<'static, Result<GenerateOutput>>;

/// Generation, lifecycle, metadata, and engine operations used by the frontend.
/// Backends return `UnsupportedOperation` for operations they do not implement.
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

    fn health_error(&self) -> Option<Arc<EngineCoreError>> {
        None
    }

    fn ready_response(&self) -> Result<&EngineCoreReadyResponse> {
        unsupported_operation("ready_response")
    }

    fn ready_responses(&self) -> Result<Vec<&EngineCoreReadyResponse>> {
        unsupported_operation("ready_responses")
    }

    fn data_parallel_size(&self) -> Result<usize> {
        unsupported_operation("data_parallel_size")
    }

    fn engine_count(&self) -> Result<usize> {
        unsupported_operation("engine_count")
    }

    fn world_size(&self) -> Result<u64> {
        unsupported_operation("world_size")
    }

    fn vllm_version(&self) -> Result<&str> {
        unsupported_operation("vllm_version")
    }

    fn total_num_gpu_blocks(&self) -> Result<u64> {
        unsupported_operation("total_num_gpu_blocks")
    }

    fn effective_attention_block_size(&self) -> Result<Option<u64>> {
        unsupported_operation("effective_attention_block_size")
    }

    fn subscribe_health(&self) -> Result<watch::Receiver<bool>> {
        unsupported_operation("subscribe_health")
    }

    fn collective_rpc<'a>(
        &'a self,
        _method: &'a str,
        _timeout: Option<f64>,
        _args: Vec<JsonValue>,
        _kwargs: BTreeMap<String, JsonValue>,
    ) -> BoxFuture<'a, Result<Vec<OpaqueValue>>> {
        Box::pin(async { unsupported_operation("collective_rpc") })
    }

    fn init_weight_transfer_engine<'a>(
        &'a self,
        _init_info: JsonValue,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async { unsupported_operation("init_weight_transfer_engine") })
    }

    fn shutdown_weight_transfer_engine<'a>(&'a self) -> BoxFuture<'a, Result<()>> {
        Box::pin(async { unsupported_operation("shutdown_weight_transfer_engine") })
    }

    fn start_weight_update<'a>(&'a self) -> BoxFuture<'a, Result<()>> {
        Box::pin(async { unsupported_operation("start_weight_update") })
    }

    fn start_draft_weight_update<'a>(&'a self) -> BoxFuture<'a, Result<()>> {
        Box::pin(async { unsupported_operation("start_draft_weight_update") })
    }

    fn update_weights<'a>(&'a self, _update_info: JsonValue) -> BoxFuture<'a, Result<()>> {
        Box::pin(async { unsupported_operation("update_weights") })
    }

    fn finish_weight_update<'a>(&'a self) -> BoxFuture<'a, Result<()>> {
        Box::pin(async { unsupported_operation("finish_weight_update") })
    }

    fn set_weight_version<'a>(&'a self, _weight_version: &'a str) -> BoxFuture<'a, Result<()>> {
        Box::pin(async { unsupported_operation("set_weight_version") })
    }

    fn get_weight_version<'a>(&'a self) -> BoxFuture<'a, Result<String>> {
        Box::pin(async { unsupported_operation("get_weight_version") })
    }

    fn is_sleeping<'a>(&'a self) -> BoxFuture<'a, Result<bool>> {
        Box::pin(async { unsupported_operation("is_sleeping") })
    }

    fn reset_mm_cache<'a>(&'a self) -> BoxFuture<'a, Result<()>> {
        Box::pin(async { unsupported_operation("reset_mm_cache") })
    }

    fn reset_encoder_cache<'a>(&'a self) -> BoxFuture<'a, Result<()>> {
        Box::pin(async { unsupported_operation("reset_encoder_cache") })
    }

    fn reset_prefix_cache<'a>(
        &'a self,
        _reset_running_requests: bool,
        _reset_connector: bool,
    ) -> BoxFuture<'a, Result<bool>> {
        Box::pin(async { unsupported_operation("reset_prefix_cache") })
    }

    fn add_lora<'a>(&'a self, _lora_request: &'a LoraRequest) -> BoxFuture<'a, Result<bool>> {
        Box::pin(async { unsupported_operation("add_lora") })
    }

    fn remove_lora<'a>(&'a self, _lora_id: u64) -> BoxFuture<'a, Result<bool>> {
        Box::pin(async { unsupported_operation("remove_lora") })
    }

    fn sleep<'a>(&'a self, _level: u32, _mode: PauseMode) -> BoxFuture<'a, Result<()>> {
        Box::pin(async { unsupported_operation("sleep") })
    }

    fn release_kv_cache_memory<'a>(&'a self) -> BoxFuture<'a, Result<()>> {
        Box::pin(async { unsupported_operation("release_kv_cache_memory") })
    }

    fn wake_up<'a>(&'a self, _tags: Option<Vec<String>>) -> BoxFuture<'a, Result<bool>> {
        Box::pin(async { unsupported_operation("wake_up") })
    }

    fn pause_generation<'a>(
        &'a self,
        _mode: PauseMode,
        _clear_cache: bool,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async { unsupported_operation("pause_generation") })
    }

    fn resume_scheduler<'a>(&'a self) -> BoxFuture<'a, Result<()>> {
        Box::pin(async { unsupported_operation("resume_scheduler") })
    }

    fn is_scheduler_paused<'a>(&'a self) -> BoxFuture<'a, Result<bool>> {
        Box::pin(async { unsupported_operation("is_scheduler_paused") })
    }

    fn start_profile<'a>(
        &'a self,
        _profile_prefix: Option<&'a str>,
        _delay_iterations: Option<u64>,
        _max_iterations: Option<u64>,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async { unsupported_operation("start_profile") })
    }

    fn stop_profile<'a>(&'a self, _profile_prefix: Option<&'a str>) -> BoxFuture<'a, Result<()>> {
        Box::pin(async { unsupported_operation("stop_profile") })
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

    fn health_error(&self) -> Option<Arc<EngineCoreError>> {
        self.engine_core_client().health_error()
    }

    fn ready_response(&self) -> Result<&EngineCoreReadyResponse> {
        Ok(self.engine_core_client().ready_response())
    }

    fn ready_responses(&self) -> Result<Vec<&EngineCoreReadyResponse>> {
        Ok(self.engine_core_client().ready_responses())
    }

    fn data_parallel_size(&self) -> Result<usize> {
        Ok(self.engine_core_client().data_parallel_size())
    }

    fn engine_count(&self) -> Result<usize> {
        Ok(self.engine_core_client().engine_count())
    }

    fn world_size(&self) -> Result<u64> {
        Ok(self.engine_core_client().world_size())
    }

    fn vllm_version(&self) -> Result<&str> {
        Ok(self.engine_core_client().vllm_version())
    }

    fn total_num_gpu_blocks(&self) -> Result<u64> {
        Ok(self.engine_core_client().total_num_gpu_blocks())
    }

    fn effective_attention_block_size(&self) -> Result<Option<u64>> {
        Ok(self.engine_core_client().effective_attention_block_size())
    }

    fn subscribe_health(&self) -> Result<watch::Receiver<bool>> {
        Ok(self.engine_core_client().subscribe_health())
    }

    fn collective_rpc<'a>(
        &'a self,
        method: &'a str,
        timeout: Option<f64>,
        args: Vec<JsonValue>,
        kwargs: BTreeMap<String, JsonValue>,
    ) -> BoxFuture<'a, Result<Vec<OpaqueValue>>> {
        Box::pin(async move {
            Ok(self.engine_core_client().collective_rpc(method, timeout, args, kwargs).await?)
        })
    }

    fn init_weight_transfer_engine<'a>(
        &'a self,
        init_info: JsonValue,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            Ok(self.engine_core_client().init_weight_transfer_engine(init_info).await?)
        })
    }

    fn shutdown_weight_transfer_engine<'a>(&'a self) -> BoxFuture<'a, Result<()>> {
        Box::pin(
            async move { Ok(self.engine_core_client().shutdown_weight_transfer_engine().await?) },
        )
    }

    fn start_weight_update<'a>(&'a self) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move { Ok(self.engine_core_client().start_weight_update().await?) })
    }

    fn start_draft_weight_update<'a>(&'a self) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move { Ok(self.engine_core_client().start_draft_weight_update().await?) })
    }

    fn update_weights<'a>(&'a self, update_info: JsonValue) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move { Ok(self.engine_core_client().update_weights(update_info).await?) })
    }

    fn finish_weight_update<'a>(&'a self) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move { Ok(self.engine_core_client().finish_weight_update().await?) })
    }

    fn set_weight_version<'a>(&'a self, weight_version: &'a str) -> BoxFuture<'a, Result<()>> {
        Box::pin(
            async move { Ok(self.engine_core_client().set_weight_version(weight_version).await?) },
        )
    }

    fn get_weight_version<'a>(&'a self) -> BoxFuture<'a, Result<String>> {
        Box::pin(async move { Ok(self.engine_core_client().get_weight_version().await?) })
    }

    fn is_sleeping<'a>(&'a self) -> BoxFuture<'a, Result<bool>> {
        Box::pin(async move { Ok(self.engine_core_client().is_sleeping().await?) })
    }

    fn reset_mm_cache<'a>(&'a self) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move { Ok(self.engine_core_client().reset_mm_cache().await?) })
    }

    fn reset_encoder_cache<'a>(&'a self) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move { Ok(self.engine_core_client().reset_encoder_cache().await?) })
    }

    fn reset_prefix_cache<'a>(
        &'a self,
        reset_running_requests: bool,
        reset_connector: bool,
    ) -> BoxFuture<'a, Result<bool>> {
        Box::pin(async move {
            Ok(self
                .engine_core_client()
                .reset_prefix_cache(reset_running_requests, reset_connector)
                .await?)
        })
    }

    fn add_lora<'a>(&'a self, lora_request: &'a LoraRequest) -> BoxFuture<'a, Result<bool>> {
        Box::pin(async move { Ok(self.engine_core_client().add_lora(lora_request).await?) })
    }

    fn remove_lora<'a>(&'a self, lora_id: u64) -> BoxFuture<'a, Result<bool>> {
        Box::pin(async move { Ok(self.engine_core_client().remove_lora(lora_id).await?) })
    }

    fn sleep<'a>(&'a self, level: u32, mode: PauseMode) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move { Ok(self.engine_core_client().sleep(level, mode).await?) })
    }

    fn release_kv_cache_memory<'a>(&'a self) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move { Ok(self.engine_core_client().release_kv_cache_memory().await?) })
    }

    fn wake_up<'a>(&'a self, tags: Option<Vec<String>>) -> BoxFuture<'a, Result<bool>> {
        Box::pin(async move { Ok(self.engine_core_client().wake_up(tags).await?) })
    }

    fn pause_generation<'a>(
        &'a self,
        mode: PauseMode,
        clear_cache: bool,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(
            async move { Ok(self.engine_core_client().pause_generation(mode, clear_cache).await?) },
        )
    }

    fn resume_scheduler<'a>(&'a self) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move { Ok(self.engine_core_client().resume_scheduler().await?) })
    }

    fn is_scheduler_paused<'a>(&'a self) -> BoxFuture<'a, Result<bool>> {
        Box::pin(async move { Ok(self.engine_core_client().is_scheduler_paused().await?) })
    }

    fn start_profile<'a>(
        &'a self,
        profile_prefix: Option<&'a str>,
        delay_iterations: Option<u64>,
        max_iterations: Option<u64>,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            Ok(self
                .engine_core_client()
                .start_profile(profile_prefix, delay_iterations, max_iterations)
                .await?)
        })
    }

    fn stop_profile<'a>(&'a self, profile_prefix: Option<&'a str>) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move { Ok(self.engine_core_client().stop_profile(profile_prefix).await?) })
    }
}

fn unsupported_operation<T>(operation: &'static str) -> Result<T> {
    Err(Error::UnsupportedOperation { operation })
}
