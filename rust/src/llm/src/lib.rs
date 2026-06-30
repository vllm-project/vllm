use tracing::Span;
use vllm_engine_core_client::EngineCoreClient;

mod error;
mod inflight;
mod log_stats;
mod output;
mod request;
mod request_metrics;

pub use error::{Error, Result};
pub use output::{
    CollectedGenerateOutput, FinishReason, GenerateOutput, GenerateOutputStream,
    GenerateOutputStreamExt, GeneratePromptInfo, TokenUsage,
};
pub use request::GenerateRequest;
pub use vllm_engine_core_client::protocol::logprobs::{Logprobs, PositionLogprobs, TokenLogprob};

use crate::inflight::InflightRequests;
use crate::log_stats::StatsLogger;
use crate::request_metrics::RequestMetricsTracker;

/// Thin generate-and-abort facade over [`EngineCoreClient`].
///
/// This mirrors the narrow public shape of Python `AsyncLLM.generate()` and
/// `abort()`, but keeps the boundary close to raw engine-core requests and
/// outputs. It tracks an in-flight external→internal request-id index (see
/// [`InflightRequests`]) so that aborts issued against external (user-supplied)
/// ids can be resolved to the internal engine ids that engine-core understands.
pub struct Llm {
    client: EngineCoreClient,
    randomize_request_id: bool,
    stats_logger: Option<StatsLogger>,
    inflight: InflightRequests,
}

impl Llm {
    /// Create a new minimal LLM facade from an already connected engine-core
    /// client.
    pub fn new(client: EngineCoreClient) -> Self {
        Self {
            client,
            randomize_request_id: true,
            stats_logger: None,
            inflight: InflightRequests::new(),
        }
    }

    /// Enable or disable periodic stats logging.
    pub fn with_log_stats(mut self, enabled: bool) -> Self {
        if enabled {
            let stats_logger = StatsLogger::start(
                self.client.model_name().to_string(),
                self.client.engine_count(),
            );
            self.stats_logger = Some(stats_logger);
        } else {
            self.stats_logger = None;
        }
        self
    }

    /// Control whether external request ids are randomized before reaching
    /// engine-core.
    pub fn with_request_id_randomization(mut self, enabled: bool) -> Self {
        self.randomize_request_id = enabled;
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
        let prepared = req.prepare(self.randomize_request_id)?;
        let prompt_token_ids = prepared.prompt_token_ids().into();
        let external_request_id = prepared
            .engine_request
            .external_req_id
            .clone()
            .expect("prepare always sets external_req_id");
        let internal_request_id = prepared.engine_request.request_id.clone();

        // Record internal engine-core request ID in the current tracing span.
        Span::current().record("engine_request_id", &internal_request_id);

        let request_metrics = RequestMetricsTracker::new(
            self.client.model_name().to_string(),
            prepared.engine_request.arrival_time,
            prepared.prompt_token_ids().len() as u32,
            (prepared.engine_request.sampling_params.as_ref()).map(|p| p.max_tokens),
            1,
        );
        let stream = self.client.call(prepared.engine_request).await?;
        let guard = self.inflight.track(external_request_id, internal_request_id);

        Ok(GenerateOutputStream::new(
            prompt_token_ids,
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
