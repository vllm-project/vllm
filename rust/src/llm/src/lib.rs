// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::any::Any;
use std::collections::BTreeMap;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt as _;
use futures::stream::BoxStream;
use tokio::sync::watch;
use tracing::Span;
use vllm_engine_core_client::EngineCoreClient;
use vllm_engine_core_client::protocol::dtype::ModelDtype;
use vllm_engine_core_client::protocol::request::EngineCoreRequest;
use vllm_metrics::METRICS;

mod error;
mod inflight;
mod log_stats;
mod output;
mod request;
mod request_metrics;

pub use error::{Error, Result};
pub use output::{
    CollectedGenerateOutput, FinishReason, GenerateOutput, GenerateOutputStream,
    GenerateOutputStreamExt, GeneratePromptInfo, PromptCacheUsage, TokenUsage, WeightVersionSpan,
};
pub use request::GenerateRequest;
pub use request_metrics::{RequestTimestamps, current_unix_timestamp_secs};
pub use vllm_engine_core_client::protocol::logprobs::{Logprobs, PositionLogprobs, TokenLogprob};

use crate::inflight::InflightRequests;
use crate::log_stats::StatsLogger;
use crate::request_metrics::{
    ParallelSamplingStats, RequestMetricHandles, RequestMetricsTracker,
    resolve_request_metric_handles,
};

/// Boxed output stream returned by [`Llm::generate`].
pub type BoxGenerateOutputStream = BoxStream<'static, Result<GenerateOutput>>;

/// Facts about the model and engine fixed when an [`Llm`] is created.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlmInfo {
    /// Effective context length, after the engine auto-fitted its KV cache.
    pub max_model_len: u32,
    /// Effective model dtype, used to encode multimodal tensors. `None` when
    /// it is not reported, which disables multimodal inputs.
    pub model_dtype: Option<ModelDtype>,
    /// Whether the engine was started with LoRA support.
    pub supports_lora: bool,
    /// vLLM version of the engine.
    pub vllm_version: String,
}

/// Token-level generation: a [`GenerateRequest`] in, a [`GenerateOutput`]
/// stream out.
///
/// This is the Rust counterpart of Python's `EngineClient` generate/abort
/// surface, and the layer below `TextLlm`. [`EngineCoreLlm`] implements it
/// over an engine-core client; other implementations reach a model over the
/// network or route among several. Requests arrive fully lowered (sampling
/// defaults resolved, stop tokens set), and outputs carry raw tokens, so
/// detokenization and stop strings stay with the caller.
///
/// Callers that need engine-core specifics, such as admin routes, downcast to
/// [`EngineCoreLlm`] through [`Any`].
#[async_trait::async_trait]
pub trait Llm: Any + Send + Sync {
    /// Facts fixed when this `Llm` was created.
    fn info(&self) -> &LlmInfo;

    /// Subscribe to health. It changes permanently to `false` once this `Llm`
    /// fails and can never serve again; restarting the process is the remedy.
    fn subscribe_health(&self) -> watch::Receiver<bool>;

    /// Subscribe to whether new requests can be accepted and processed now.
    /// It may change either way, such as while engines are paused or asleep,
    /// or while a remote one is unreachable.
    fn subscribe_readiness(&self) -> watch::Receiver<bool>;

    /// Whether this `Llm` has not failed.
    fn is_healthy(&self) -> bool {
        *self.subscribe_health().borrow()
    }

    /// Submit one tokenized request and stream its outputs. Dropping the
    /// stream before the terminal output aborts the request.
    async fn generate(&self, request: GenerateRequest) -> Result<BoxGenerateOutputStream> {
        let mut streams = self.generate_n(request, NonZeroU32::MIN).await?;
        Ok(streams.pop().expect("one output stream per sample"))
    }

    /// Submit one tokenized request as `n` parallel samples, returning exactly
    /// `n` output streams in index order. Dropping a stream before its
    /// terminal output aborts that sample; an abort by the request's external
    /// ID reaches every sample.
    async fn generate_n(
        &self,
        request: GenerateRequest,
        n: NonZeroU32,
    ) -> Result<Vec<BoxGenerateOutputStream>>;

    /// Abort in-flight requests by their external (user-supplied) request ids.
    /// An empty slice aborts every in-flight request.
    async fn abort(&self, external_ids: &[String]) -> Result<()>;

    /// Shut down and stop background tasks.
    async fn shutdown(self: Box<Self>) -> Result<()>;
}

/// Thin generate-and-abort facade over [`EngineCoreClient`].
///
/// This mirrors the narrow public shape of Python `AsyncLLM.generate()` and
/// `abort()`, but keeps the boundary close to raw engine-core requests and
/// outputs. It tracks an in-flight external→internal request-id index (see
/// [`InflightRequests`]) so that aborts issued against external (user-supplied)
/// ids can be resolved to the internal engine ids that engine-core understands.
pub struct EngineCoreLlm {
    client: EngineCoreClient,
    /// Facts from the engine-core startup handshake.
    info: LlmInfo,
    randomize_request_id: bool,
    stream_interval: NonZeroU32,
    stats_logger: Option<StatsLogger>,
    inflight: InflightRequests,
    /// Request metric handles per connected engine index, resolved at
    /// construction so the request families export zero-valued series before
    /// the first request.
    request_metric_handles: BTreeMap<u32, Arc<RequestMetricHandles>>,
}

impl EngineCoreLlm {
    /// Create a new minimal LLM facade from an already connected engine-core
    /// client.
    pub fn new(client: EngineCoreClient) -> Self {
        METRICS.request.set_max_model_len(client.max_model_len());
        let request_metric_handles = (client.engine_indices().into_iter())
            .map(|engine| {
                let handles = resolve_request_metric_handles(
                    client.model_name(),
                    engine,
                    client.engine_stats_enabled(),
                );
                (engine, Arc::new(handles))
            })
            .collect();
        // The engine-reported value reflects the post-profiling, auto-fitted
        // KV cache limit used at runtime.
        let info = LlmInfo {
            max_model_len: client.max_model_len(),
            model_dtype: Some(client.model_dtype()),
            supports_lora: client.ready_response().supports_lora,
            vllm_version: client.vllm_version().to_string(),
        };
        Self {
            client,
            info,
            randomize_request_id: true,
            stream_interval: NonZeroU32::MIN,
            stats_logger: None,
            inflight: InflightRequests::new(),
            request_metric_handles,
        }
    }

    /// Enable periodic stats logging at the given interval, or disable it with
    /// `None`. A zero interval logs as often as the timer allows.
    pub fn with_log_stats(mut self, interval: Option<Duration>) -> Self {
        self.stats_logger = interval.map(|interval| {
            StatsLogger::start(
                self.client.model_name().to_string(),
                self.client.engine_indices(),
                interval,
            )
        });
        self
    }

    /// Control whether external request ids are randomized before reaching
    /// engine-core.
    pub fn with_request_id_randomization(mut self, enabled: bool) -> Self {
        self.randomize_request_id = enabled;
        self
    }

    /// Set the frontend-level stream interval: the minimum number of newly
    /// generated tokens batched into each streamed output after the first one.
    /// A request's own `stream_interval` can only raise it.
    pub fn with_stream_interval(mut self, stream_interval: NonZeroU32) -> Self {
        self.stream_interval = stream_interval;
        self
    }

    /// Expose the underlying engine-core client for low-level utility/admin
    /// calls.
    pub fn engine_core_client(&self) -> &EngineCoreClient {
        &self.client
    }

    /// Submit one tokenized generate request and return a per-request output
    /// stream.
    pub async fn generate(&self, req: GenerateRequest) -> Result<GenerateOutputStream> {
        let mut streams = self.generate_n(req, NonZeroU32::MIN).await?;
        Ok(streams.pop().expect("one output stream per sample"))
    }

    /// Submit one request as `n` parallel samples, returning one output stream
    /// per sample in index order.
    ///
    /// Like Python's `ParentRequest`, each sample is its own engine request:
    /// child `i` gets the internal ID `{i}_{request_id}` and, when the request
    /// is seeded, the seed `seed + i`, while all of them keep the caller's
    /// external ID, so an abort by that ID reaches every sample. With `n = 1`
    /// the request is submitted unchanged.
    ///
    /// Original Python definition:
    /// <https://github.com/vllm-project/vllm/blob/913fccd11847f5b4d4769e3a69c8925f955bc29e/vllm/v1/engine/parallel_sampling.py#L56-L96>
    pub async fn generate_n(
        &self,
        mut req: GenerateRequest,
        n: NonZeroU32,
    ) -> Result<Vec<GenerateOutputStream>> {
        // Clamp the request's stream interval to the frontend-level one, like
        // Python. Engine-core ignores it; the client batches deliveries by it.
        let stream_interval = (req.sampling_params.stream_interval)
            .map_or(self.stream_interval, |interval| {
                interval.max(self.stream_interval)
            });
        req.sampling_params.stream_interval =
            (stream_interval > NonZeroU32::MIN).then_some(stream_interval);

        let prepared = req.prepare(self.randomize_request_id)?;
        let remote_prefill_cached_tokens = prepared.remote_prefill_cached_tokens();
        let prompt_token_ids = prepared.prompt_token_ids;
        let engine_request = prepared.engine_request;

        if n == NonZeroU32::MIN {
            let stream = self
                .submit(
                    engine_request,
                    &prompt_token_ids,
                    remote_prefill_cached_tokens,
                    None,
                )
                .await?;
            return Ok(vec![stream]);
        }

        let parallel_sampling = Arc::new(ParallelSamplingStats::new(n.get()));
        let mut streams = Vec::with_capacity(n.get() as usize);
        for index in 0..n.get() {
            let mut child = engine_request.clone();
            child.request_id = format!("{index}_{}", engine_request.request_id);
            // A seed of `-1` means unseeded, like Python's `SamplingParams`, and
            // stays so for every child.
            if let Some(sampling_params) = child.sampling_params.as_mut()
                && let Some(seed) = sampling_params.seed.filter(|&seed| seed != -1)
            {
                sampling_params.seed = Some(seed.wrapping_add(index.into()));
            }
            // A failed submission drops the earlier streams, which aborts them.
            streams.push(
                self.submit(
                    child,
                    &prompt_token_ids,
                    remote_prefill_cached_tokens,
                    Some(parallel_sampling.clone()),
                )
                .await?,
            );
        }
        Ok(streams)
    }

    /// Submit one prepared engine request and track it until its stream drops.
    async fn submit(
        &self,
        engine_request: EngineCoreRequest,
        prompt_token_ids: &Arc<[u32]>,
        remote_prefill_cached_tokens: Option<usize>,
        parallel_sampling: Option<Arc<ParallelSamplingStats>>,
    ) -> Result<GenerateOutputStream> {
        let external_request_id = engine_request
            .external_req_id
            .clone()
            .expect("prepare always sets external_req_id");
        let internal_request_id = engine_request.request_id.clone();

        // Record internal engine-core request ID in the current tracing span.
        Span::current().record("engine_request_id", &internal_request_id);

        let arrival_time = engine_request.arrival_time;
        let max_tokens_param = (engine_request.sampling_params.as_ref()).map(|p| p.max_tokens);
        let prompt_len = prompt_token_ids.len() as u32;
        let watermark_requested =
            (engine_request.sampling_params.as_ref()).is_some_and(|p| p.watermarking);

        let stream = self.client.call(engine_request).await?;

        let handles = (self.request_metric_handles.get(&stream.engine_index()))
            .expect("requests are routed only to connected engines")
            .clone();
        let request_metrics = RequestMetricsTracker::new(
            handles,
            arrival_time,
            prompt_len,
            max_tokens_param,
            parallel_sampling,
            watermark_requested,
        );
        let guard = self.inflight.track(external_request_id, internal_request_id);

        Ok(GenerateOutputStream::new(
            prompt_token_ids.clone(),
            remote_prefill_cached_tokens,
            stream,
            request_metrics,
            guard,
        ))
    }

    /// Abort in-flight requests by their external (user-supplied) request ids.
    ///
    /// External ids are resolved to the internal engine ids actually known to
    /// engine-core (one external id may map to several internal ids). Unknown
    /// or already-finished ids resolve to nothing and are a safe no-op. The
    /// tracking entries themselves are removed when the corresponding output
    /// streams are dropped, not here.
    pub async fn abort(&self, external_ids: &[String]) -> Result<()> {
        // Empty `external_ids` means abort every in-flight request.
        let internal_ids = if external_ids.is_empty() {
            self.inflight.all_internal_ids()
        } else {
            self.inflight.resolve(external_ids)
        };
        if internal_ids.is_empty() {
            return Ok(());
        }
        self.client.abort(&internal_ids).await?;
        Ok(())
    }

    /// Shut down the underlying engine-core client and its background tasks.
    pub async fn shutdown(self) -> Result<()> {
        self.client.shutdown().await?;
        Ok(())
    }
}

#[async_trait::async_trait]
impl Llm for EngineCoreLlm {
    fn info(&self) -> &LlmInfo {
        &self.info
    }

    fn subscribe_health(&self) -> watch::Receiver<bool> {
        self.client.subscribe_health()
    }

    fn subscribe_readiness(&self) -> watch::Receiver<bool> {
        self.client.subscribe_readiness()
    }

    async fn generate_n(
        &self,
        request: GenerateRequest,
        n: NonZeroU32,
    ) -> Result<Vec<BoxGenerateOutputStream>> {
        let streams = EngineCoreLlm::generate_n(self, request, n).await?;
        Ok(streams.into_iter().map(|stream| stream.boxed()).collect())
    }

    async fn abort(&self, external_ids: &[String]) -> Result<()> {
        EngineCoreLlm::abort(self, external_ids).await
    }

    async fn shutdown(self: Box<Self>) -> Result<()> {
        EngineCoreLlm::shutdown(*self).await
    }
}
