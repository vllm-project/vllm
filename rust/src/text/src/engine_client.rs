// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeMap;
use std::sync::Arc;

use futures::{StreamExt as _, future::BoxFuture, stream::BoxStream};
use serde_json::Value as JsonValue;
use tokio::sync::watch;
use vllm_engine_core_client::Error as EngineCoreError;
use vllm_engine_core_client::protocol::OpaqueValue;
use vllm_engine_core_client::protocol::dtype::ModelDtype;
use vllm_engine_core_client::protocol::handshake::EngineCoreReadyResponse;
use vllm_engine_core_client::protocol::lora::LoraRequest;
use vllm_engine_core_client::protocol::utility::{CheckpointState, PauseMode, SleepLevel};
use vllm_llm::{GenerateOutput, GenerateRequest, Llm, Result};

pub type TokenStream = BoxStream<'static, Result<GenerateOutput>>;

/// Generation, lifecycle, metadata, and engine operations used by the frontend.
pub trait EngineClient: Send + Sync {
    /// Submit a tokenized request and return its output stream.
    fn generate<'a>(&'a self, request: GenerateRequest) -> BoxFuture<'a, Result<TokenStream>>;

    fn abort<'a>(&'a self, request_ids: &'a [String]) -> BoxFuture<'a, Result<()>>;
    fn shutdown(self: Box<Self>) -> BoxFuture<'static, Result<()>>;
    fn max_model_len(&self) -> u32;
    fn model_dtype(&self) -> ModelDtype;
    fn is_healthy(&self) -> bool;
    fn supports_lora(&self) -> bool;

    fn health_error(&self) -> Option<Arc<EngineCoreError>>;

    fn ready_response(&self) -> &EngineCoreReadyResponse;

    fn ready_responses(&self) -> Vec<&EngineCoreReadyResponse>;

    fn data_parallel_size(&self) -> usize;

    fn engine_count(&self) -> usize;

    fn world_size(&self) -> u64;

    fn vllm_version(&self) -> &str;

    fn total_num_gpu_blocks(&self) -> u64;

    fn effective_attention_block_size(&self) -> Option<u64>;

    fn subscribe_health(&self) -> watch::Receiver<bool>;

    fn collective_rpc<'a>(
        &'a self,
        method: &'a str,
        timeout: Option<f64>,
        args: Vec<JsonValue>,
        kwargs: BTreeMap<String, JsonValue>,
    ) -> BoxFuture<'a, Result<Vec<OpaqueValue>>>;

    fn init_weight_transfer_engine<'a>(&'a self, init_info: JsonValue)
    -> BoxFuture<'a, Result<()>>;

    fn shutdown_weight_transfer_engine<'a>(&'a self) -> BoxFuture<'a, Result<()>>;

    fn start_weight_update<'a>(&'a self) -> BoxFuture<'a, Result<()>>;

    fn start_draft_weight_update<'a>(&'a self) -> BoxFuture<'a, Result<()>>;

    fn update_weights<'a>(&'a self, update_info: JsonValue) -> BoxFuture<'a, Result<()>>;

    fn finish_weight_update<'a>(&'a self) -> BoxFuture<'a, Result<()>>;

    fn set_weight_version<'a>(&'a self, weight_version: &'a str) -> BoxFuture<'a, Result<()>>;

    fn get_weight_version<'a>(&'a self) -> BoxFuture<'a, Result<String>>;

    fn is_sleeping<'a>(&'a self) -> BoxFuture<'a, Result<bool>>;

    fn reset_mm_cache<'a>(&'a self) -> BoxFuture<'a, Result<()>>;

    fn reset_encoder_cache<'a>(&'a self) -> BoxFuture<'a, Result<()>>;

    fn reset_prefix_cache<'a>(
        &'a self,
        reset_running_requests: bool,
        reset_connector: bool,
    ) -> BoxFuture<'a, Result<bool>>;

    fn add_lora<'a>(&'a self, lora_request: &'a LoraRequest) -> BoxFuture<'a, Result<bool>>;

    fn remove_lora<'a>(&'a self, lora_id: u64) -> BoxFuture<'a, Result<bool>>;

    fn sleep<'a>(&'a self, level: SleepLevel, mode: PauseMode) -> BoxFuture<'a, Result<()>>;

    fn release_kv_cache_memory<'a>(&'a self) -> BoxFuture<'a, Result<()>>;

    fn wake_up<'a>(&'a self, tags: Option<Vec<String>>) -> BoxFuture<'a, Result<bool>>;

    fn pause_generation<'a>(
        &'a self,
        mode: PauseMode,
        clear_cache: bool,
    ) -> BoxFuture<'a, Result<()>>;

    fn resume_scheduler<'a>(&'a self) -> BoxFuture<'a, Result<()>>;

    fn checkpoint_prepare<'a>(&'a self, mode: PauseMode) -> BoxFuture<'a, Result<Option<String>>>;

    fn checkpoint_restore<'a>(&'a self) -> BoxFuture<'a, Result<()>>;

    fn checkpoint_state<'a>(&'a self) -> BoxFuture<'a, Result<(CheckpointState, Option<String>)>>;

    fn is_scheduler_paused<'a>(&'a self) -> BoxFuture<'a, Result<bool>>;

    fn start_profile<'a>(
        &'a self,
        profile_prefix: Option<&'a str>,
        delay_iterations: Option<u64>,
        max_iterations: Option<u64>,
        profiler_kwargs: Option<&'a serde_json::Map<String, JsonValue>>,
    ) -> BoxFuture<'a, Result<()>>;

    fn stop_profile<'a>(&'a self, profile_prefix: Option<&'a str>) -> BoxFuture<'a, Result<()>>;
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

    fn model_dtype(&self) -> ModelDtype {
        self.engine_core_client().model_dtype()
    }

    fn is_healthy(&self) -> bool {
        self.engine_core_client().is_healthy()
    }

    fn supports_lora(&self) -> bool {
        self.engine_core_client().ready_response().supports_lora
    }

    fn health_error(&self) -> Option<Arc<EngineCoreError>> {
        self.engine_core_client().health_error()
    }

    fn ready_response(&self) -> &EngineCoreReadyResponse {
        self.engine_core_client().ready_response()
    }

    fn ready_responses(&self) -> Vec<&EngineCoreReadyResponse> {
        self.engine_core_client().ready_responses()
    }

    fn data_parallel_size(&self) -> usize {
        self.engine_core_client().data_parallel_size()
    }

    fn engine_count(&self) -> usize {
        self.engine_core_client().engine_count()
    }

    fn world_size(&self) -> u64 {
        self.engine_core_client().world_size()
    }

    fn vllm_version(&self) -> &str {
        self.engine_core_client().vllm_version()
    }

    fn total_num_gpu_blocks(&self) -> u64 {
        self.engine_core_client().total_num_gpu_blocks()
    }

    fn effective_attention_block_size(&self) -> Option<u64> {
        self.engine_core_client().effective_attention_block_size()
    }

    fn subscribe_health(&self) -> watch::Receiver<bool> {
        self.engine_core_client().subscribe_health()
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

    fn sleep<'a>(&'a self, level: SleepLevel, mode: PauseMode) -> BoxFuture<'a, Result<()>> {
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

    fn checkpoint_prepare<'a>(&'a self, mode: PauseMode) -> BoxFuture<'a, Result<Option<String>>> {
        Box::pin(async move { Ok(self.engine_core_client().checkpoint_prepare(mode).await?) })
    }

    fn checkpoint_restore<'a>(&'a self) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move { Ok(self.engine_core_client().checkpoint_restore().await?) })
    }

    fn checkpoint_state<'a>(&'a self) -> BoxFuture<'a, Result<(CheckpointState, Option<String>)>> {
        Box::pin(async move { Ok(self.engine_core_client().checkpoint_state().await?) })
    }

    fn is_scheduler_paused<'a>(&'a self) -> BoxFuture<'a, Result<bool>> {
        Box::pin(async move { Ok(self.engine_core_client().is_scheduler_paused().await?) })
    }

    fn start_profile<'a>(
        &'a self,
        profile_prefix: Option<&'a str>,
        delay_iterations: Option<u64>,
        max_iterations: Option<u64>,
        profiler_kwargs: Option<&'a serde_json::Map<String, JsonValue>>,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            Ok(self
                .engine_core_client()
                .start_profile(
                    profile_prefix,
                    delay_iterations,
                    max_iterations,
                    profiler_kwargs,
                )
                .await?)
        })
    }

    fn stop_profile<'a>(&'a self, profile_prefix: Option<&'a str>) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move { Ok(self.engine_core_client().stop_profile(profile_prefix).await?) })
    }
}
