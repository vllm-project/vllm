// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeMap;

use futures::{future::BoxFuture, stream::FusedStream};
use serde_json::Value as JsonValue;
use tokio::sync::watch;

use crate::protocol::OpaqueValue;
use crate::protocol::dtype::ModelDtype;
use crate::protocol::handshake::EngineCoreReadyResponse;
use crate::protocol::lora::LoraRequest;
use crate::protocol::request::EngineCoreRequest;
use crate::protocol::utility::{CheckpointState, PauseMode, SleepLevel};
use crate::{EngineCoreClient, EngineCoreOutputStream, EngineCoreStreamDelivery, Result};

/// Per-request engine output. Dropping a live stream must abort its request.
/// Mark the stream terminated when delivering its terminal output, so dropping
/// it immediately afterward does not abort a completed request.
/// Outputs must belong to this request. An unexpected close must yield an
/// error, not end the stream without a terminal output. Errors terminate the stream.
pub trait EngineOutputStream:
    FusedStream<Item = Result<EngineCoreStreamDelivery>> + Send + Unpin
{
    /// Internal engine request ID, matching the submitted request.
    fn request_id(&self) -> &str;
    fn engine_index(&self) -> u32;
}

impl EngineOutputStream for EngineCoreOutputStream {
    fn request_id(&self) -> &str {
        self.request_id()
    }

    fn engine_index(&self) -> u32 {
        self.engine_index()
    }
}

/// Request submission, lifecycle, metadata, and engine operations used by the frontend.
///
/// Metadata is available when the client is constructed. Submitted utility
/// calls wait for every connected engine's outcome before reporting failure.
pub trait EngineClient: Send + Sync {
    /// Submit an engine request and return its output stream.
    /// The stream's engine index must be one of [`Self::engine_indices`].
    /// Cancelling submission must clean up any registered request, including
    /// aborting it if it may already have reached the engine.
    fn call(
        &self,
        request: EngineCoreRequest,
    ) -> BoxFuture<'_, Result<Box<dyn EngineOutputStream>>>;

    /// Abort by internal engine IDs and deliver terminal abort outputs to the
    /// affected streams. Unknown IDs and an empty list are no-ops.
    fn abort<'a>(&'a self, request_ids: &'a [String]) -> BoxFuture<'a, Result<()>>;
    /// Close client resources and background tasks, not the engine processes.
    fn shutdown(self: Box<Self>) -> BoxFuture<'static, Result<()>>;
    fn model_name(&self) -> &str;
    /// Engine indices used for request metrics, fixed for this client's lifetime.
    fn engine_indices(&self) -> Vec<u32>;
    fn engine_stats_enabled(&self) -> bool;
    /// Minimum effective context limit across connected engines.
    fn max_model_len(&self) -> u32;
    /// Effective model dtype reported by the first connected engine.
    fn model_dtype(&self) -> ModelDtype;
    fn is_healthy(&self) -> bool;
    fn supports_lora(&self) -> bool;

    /// First connected engine's cached metadata; per-engine values require
    /// [`Self::ready_responses`].
    fn ready_response(&self) -> &EngineCoreReadyResponse;

    /// Cached metadata for every connected engine, in order; never empty.
    fn ready_responses(&self) -> Vec<&EngineCoreReadyResponse>;

    /// Deployment-wide DP size, which can differ from the connected-engine count
    /// and the effective DP size in an individual engine's metadata.
    fn data_parallel_size(&self) -> usize;

    /// Number of engines connected to this client.
    fn engine_count(&self) -> usize;

    /// First engine's world size, excluding deployment-wide data parallelism.
    fn world_size(&self) -> u64;

    /// Python vLLM version reported by the first connected engine.
    fn vllm_version(&self) -> &str;

    /// GPU blocks summed across connected engines.
    fn total_num_gpu_blocks(&self) -> u64;

    /// Attention block size only when every engine reports the same value.
    fn effective_attention_block_size(&self) -> Option<u64>;

    /// Current health and subsequent changes, consistent with [`Self::is_healthy`].
    fn subscribe_health(&self) -> watch::Receiver<bool>;

    /// Execute on every engine and flatten their worker results into one list.
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

    /// Return the version shared by all engines; disagreement is an error.
    fn get_weight_version<'a>(&'a self) -> BoxFuture<'a, Result<String>>;

    /// Return the sleep state shared by all engines; disagreement is an error.
    fn is_sleeping<'a>(&'a self) -> BoxFuture<'a, Result<bool>>;

    fn reset_mm_cache<'a>(&'a self) -> BoxFuture<'a, Result<()>>;

    fn reset_encoder_cache<'a>(&'a self) -> BoxFuture<'a, Result<()>>;

    /// Return true only when every engine confirms the reset.
    fn reset_prefix_cache<'a>(
        &'a self,
        reset_running_requests: bool,
        reset_connector: bool,
    ) -> BoxFuture<'a, Result<bool>>;

    /// Load on every engine; return true only when all confirm success.
    fn add_lora<'a>(&'a self, lora_request: &'a LoraRequest) -> BoxFuture<'a, Result<bool>>;

    /// Remove from every engine; return true only when all confirm success.
    fn remove_lora<'a>(&'a self, lora_id: u64) -> BoxFuture<'a, Result<bool>>;

    fn sleep<'a>(&'a self, level: SleepLevel, mode: PauseMode) -> BoxFuture<'a, Result<()>>;

    fn release_kv_cache_memory<'a>(&'a self) -> BoxFuture<'a, Result<()>>;

    /// Return true only when every engine is fully awake after applying the tags.
    fn wake_up<'a>(&'a self, tags: Option<Vec<String>>) -> BoxFuture<'a, Result<bool>>;

    fn pause_generation<'a>(
        &'a self,
        mode: PauseMode,
        clear_cache: bool,
    ) -> BoxFuture<'a, Result<()>>;

    fn resume_scheduler<'a>(&'a self) -> BoxFuture<'a, Result<()>>;

    /// Return the first refusal, if any; other engines may already have started.
    fn checkpoint_prepare<'a>(&'a self, mode: PauseMode) -> BoxFuture<'a, Result<Option<String>>>;

    fn checkpoint_restore<'a>(&'a self) -> BoxFuture<'a, Result<()>>;

    /// Prefer any failed state, then unanimous state, then preparing if any
    /// engine is preparing, otherwise restoring. Include the failure's error.
    fn checkpoint_state<'a>(&'a self) -> BoxFuture<'a, Result<(CheckpointState, Option<String>)>>;

    /// Return the pause state shared by all engines; disagreement is an error.
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

impl EngineClient for EngineCoreClient {
    fn call(
        &self,
        request: EngineCoreRequest,
    ) -> BoxFuture<'_, Result<Box<dyn EngineOutputStream>>> {
        Box::pin(async move {
            Ok(Box::new(EngineCoreClient::call(self, request).await?)
                as Box<dyn EngineOutputStream>)
        })
    }

    fn abort<'a>(&'a self, request_ids: &'a [String]) -> BoxFuture<'a, Result<()>> {
        Box::pin(EngineCoreClient::abort(self, request_ids))
    }

    fn shutdown(self: Box<Self>) -> BoxFuture<'static, Result<()>> {
        Box::pin(EngineCoreClient::shutdown(*self))
    }

    fn model_name(&self) -> &str {
        self.model_name()
    }

    fn engine_indices(&self) -> Vec<u32> {
        self.engine_indices()
    }

    fn engine_stats_enabled(&self) -> bool {
        self.engine_stats_enabled()
    }

    fn max_model_len(&self) -> u32 {
        self.max_model_len()
    }

    fn model_dtype(&self) -> ModelDtype {
        self.model_dtype()
    }

    fn is_healthy(&self) -> bool {
        self.is_healthy()
    }

    fn supports_lora(&self) -> bool {
        self.ready_response().supports_lora
    }

    fn ready_response(&self) -> &EngineCoreReadyResponse {
        self.ready_response()
    }

    fn ready_responses(&self) -> Vec<&EngineCoreReadyResponse> {
        self.ready_responses()
    }

    fn data_parallel_size(&self) -> usize {
        self.data_parallel_size()
    }

    fn engine_count(&self) -> usize {
        self.engine_count()
    }

    fn world_size(&self) -> u64 {
        self.world_size()
    }

    fn vllm_version(&self) -> &str {
        self.vllm_version()
    }

    fn total_num_gpu_blocks(&self) -> u64 {
        self.total_num_gpu_blocks()
    }

    fn effective_attention_block_size(&self) -> Option<u64> {
        self.effective_attention_block_size()
    }

    fn subscribe_health(&self) -> watch::Receiver<bool> {
        self.subscribe_health()
    }

    fn collective_rpc<'a>(
        &'a self,
        method: &'a str,
        timeout: Option<f64>,
        args: Vec<JsonValue>,
        kwargs: BTreeMap<String, JsonValue>,
    ) -> BoxFuture<'a, Result<Vec<OpaqueValue>>> {
        Box::pin(self.collective_rpc(method, timeout, args, kwargs))
    }

    fn init_weight_transfer_engine<'a>(
        &'a self,
        init_info: JsonValue,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(self.init_weight_transfer_engine(init_info))
    }

    fn shutdown_weight_transfer_engine<'a>(&'a self) -> BoxFuture<'a, Result<()>> {
        Box::pin(self.shutdown_weight_transfer_engine())
    }

    fn start_weight_update<'a>(&'a self) -> BoxFuture<'a, Result<()>> {
        Box::pin(self.start_weight_update())
    }

    fn start_draft_weight_update<'a>(&'a self) -> BoxFuture<'a, Result<()>> {
        Box::pin(self.start_draft_weight_update())
    }

    fn update_weights<'a>(&'a self, update_info: JsonValue) -> BoxFuture<'a, Result<()>> {
        Box::pin(self.update_weights(update_info))
    }

    fn finish_weight_update<'a>(&'a self) -> BoxFuture<'a, Result<()>> {
        Box::pin(self.finish_weight_update())
    }

    fn set_weight_version<'a>(&'a self, weight_version: &'a str) -> BoxFuture<'a, Result<()>> {
        Box::pin(self.set_weight_version(weight_version))
    }

    fn get_weight_version<'a>(&'a self) -> BoxFuture<'a, Result<String>> {
        Box::pin(self.get_weight_version())
    }

    fn is_sleeping<'a>(&'a self) -> BoxFuture<'a, Result<bool>> {
        Box::pin(self.is_sleeping())
    }

    fn reset_mm_cache<'a>(&'a self) -> BoxFuture<'a, Result<()>> {
        Box::pin(self.reset_mm_cache())
    }

    fn reset_encoder_cache<'a>(&'a self) -> BoxFuture<'a, Result<()>> {
        Box::pin(self.reset_encoder_cache())
    }

    fn reset_prefix_cache<'a>(
        &'a self,
        reset_running_requests: bool,
        reset_connector: bool,
    ) -> BoxFuture<'a, Result<bool>> {
        Box::pin(self.reset_prefix_cache(reset_running_requests, reset_connector))
    }

    fn add_lora<'a>(&'a self, lora_request: &'a LoraRequest) -> BoxFuture<'a, Result<bool>> {
        Box::pin(self.add_lora(lora_request))
    }

    fn remove_lora<'a>(&'a self, lora_id: u64) -> BoxFuture<'a, Result<bool>> {
        Box::pin(self.remove_lora(lora_id))
    }

    fn sleep<'a>(&'a self, level: SleepLevel, mode: PauseMode) -> BoxFuture<'a, Result<()>> {
        Box::pin(self.sleep(level, mode))
    }

    fn release_kv_cache_memory<'a>(&'a self) -> BoxFuture<'a, Result<()>> {
        Box::pin(self.release_kv_cache_memory())
    }

    fn wake_up<'a>(&'a self, tags: Option<Vec<String>>) -> BoxFuture<'a, Result<bool>> {
        Box::pin(self.wake_up(tags))
    }

    fn pause_generation<'a>(
        &'a self,
        mode: PauseMode,
        clear_cache: bool,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(self.pause_generation(mode, clear_cache))
    }

    fn resume_scheduler<'a>(&'a self) -> BoxFuture<'a, Result<()>> {
        Box::pin(self.resume_scheduler())
    }

    fn checkpoint_prepare<'a>(&'a self, mode: PauseMode) -> BoxFuture<'a, Result<Option<String>>> {
        Box::pin(self.checkpoint_prepare(mode))
    }

    fn checkpoint_restore<'a>(&'a self) -> BoxFuture<'a, Result<()>> {
        Box::pin(self.checkpoint_restore())
    }

    fn checkpoint_state<'a>(&'a self) -> BoxFuture<'a, Result<(CheckpointState, Option<String>)>> {
        Box::pin(self.checkpoint_state())
    }

    fn is_scheduler_paused<'a>(&'a self) -> BoxFuture<'a, Result<bool>> {
        Box::pin(self.is_scheduler_paused())
    }

    fn start_profile<'a>(
        &'a self,
        profile_prefix: Option<&'a str>,
        delay_iterations: Option<u64>,
        max_iterations: Option<u64>,
        profiler_kwargs: Option<&'a serde_json::Map<String, JsonValue>>,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(self.start_profile(
            profile_prefix,
            delay_iterations,
            max_iterations,
            profiler_kwargs,
        ))
    }

    fn stop_profile<'a>(&'a self, profile_prefix: Option<&'a str>) -> BoxFuture<'a, Result<()>> {
        Box::pin(self.stop_profile(profile_prefix))
    }
}
