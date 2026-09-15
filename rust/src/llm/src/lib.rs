// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
pub use request_metrics::current_unix_timestamp_secs;
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
                self.client.engine_indices(),
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

        let arrival_time = prepared.engine_request.arrival_time;
        let max_tokens_param =
            (prepared.engine_request.sampling_params.as_ref()).map(|p| p.max_tokens);
        let prompt_len = prepared.prompt_token_ids().len() as u32;

        let stream = self.client.call(prepared.engine_request).await?;

        let request_metrics = RequestMetricsTracker::new(
            self.client.model_name().to_string(),
            stream.engine_index(),
            arrival_time,
            prompt_len,
            max_tokens_param,
            1,
        );
        let guard = self.inflight.track(external_request_id, internal_request_id);

        Ok(GenerateOutputStream::new(
            prompt_token_ids,
            stream,
            request_metrics,
            guard,
        ))
    }

    /// Open a resumable streaming-input session: one logical request fed many
    /// prompt chunks over time, with a single continuous output stream.
    ///
    /// `first_segment` is the opening ADD and must have `resumable: true`.
    /// Further chunks go through [`SegmentSink::push`], and
    /// [`SegmentSink::finish`] sends the sentinel that ends the session — at
    /// which point the returned stream terminates. Engine-core retains the
    /// request's KV between chunks, so later segments continue the generation
    /// instead of re-prefilling.
    ///
    /// This is the Rust analog of Python's `_add_streaming_input_request`.
    ///
    /// # Streaming-input validation parity
    ///
    /// Python's `_add_streaming_input_request` rejects `n > 1`, pooling
    /// params, `FINAL_ONLY` output kind, and stop strings before opening a
    /// session (`_validate_streaming_input_sampling_params`,
    /// `vllm/v1/engine/async_llm.py:506-520`). This method has no runtime
    /// check for any of the four — each is structurally unreachable through
    /// this signature, not merely unvalidated:
    ///
    /// - `n` and `FINAL_ONLY` (`output_kind`) have no field on
    ///   [`vllm_engine_core_client::protocol::sampling::EngineCoreSamplingParams`],
    ///   the type user sampling params are lowered into before they can reach
    ///   here.
    /// - Pooling params have no variant on [`GenerateRequest`]: this is the
    ///   generate-only facade, and pooling requests go through a separate one
    ///   that never calls `generate_streaming`.
    /// - Stop *strings* (as opposed to the already-tokenized
    ///   `stop_token_ids` this API does accept) exist only as `vllm_text`'s
    ///   `TextDecodeOptions::stop_strings`, matched frontend-side against
    ///   decoded text — a layer this API never sees. `vllm_text::lower_text_request`
    ///   additionally hardcodes `resumable: false` on every request it
    ///   lowers, so a caller carrying stop strings could not reach a
    ///   resumable session through that path either.
    ///
    /// A no-op validator was deliberately not written: there is nothing left
    /// for it to check.
    pub async fn generate_streaming(
        &self,
        first_segment: GenerateRequest,
    ) -> Result<(SegmentSink<'_>, GenerateOutputStream)> {
        if !first_segment.resumable {
            return Err(Error::NotResumable {
                request_id: first_segment.request_id,
            });
        }
        let prepared = first_segment.prepare(self.randomize_request_id)?;
        let prompt_token_ids = prepared.prompt_token_ids().into();
        let external_request_id = prepared
            .engine_request
            .external_req_id
            .clone()
            .expect("prepare always sets external_req_id");
        let internal_request_id = prepared.engine_request.request_id.clone();

        Span::current().record("engine_request_id", &internal_request_id);

        let arrival_time = prepared.engine_request.arrival_time;
        let max_tokens_param =
            (prepared.engine_request.sampling_params.as_ref()).map(|p| p.max_tokens);
        let prompt_len = prepared.prompt_token_ids().len() as u32;

        let stream = self.client.call(prepared.engine_request).await?;

        let request_metrics = RequestMetricsTracker::new(
            self.client.model_name().to_string(),
            stream.engine_index(),
            arrival_time,
            prompt_len,
            max_tokens_param,
            1,
        );
        let guard = self.inflight.track(external_request_id, internal_request_id.clone());

        let sink = SegmentSink {
            client: &self.client,
            request_id: internal_request_id,
        };
        Ok((
            sink,
            GenerateOutputStream::new(prompt_token_ids, stream, request_metrics, guard),
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

/// Feeds successive prompt chunks into one open resumable request.
///
/// Returned by [`Llm::generate_streaming`] alongside the session's output
/// stream. Every chunk is an ADD under the same engine-side `request_id`; the
/// stream stays open across all of them and terminates on the finish reason
/// triggered by [`Self::finish`].
///
/// Borrows the client rather than owning it, so a sink cannot outlive the
/// [`Llm`] whose session it feeds.
pub struct SegmentSink<'a> {
    client: &'a EngineCoreClient,
    /// Engine-side id of the open session. Every chunk is pinned to it.
    request_id: String,
}

impl SegmentSink<'_> {
    /// The engine-side request id this session runs under.
    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    /// Feed one more segment into the open session.
    ///
    /// Its output continues on the stream `generate_streaming` returned; there
    /// is no new stream. `segment.request_id` is ignored — the chunk is pinned
    /// to the session's id.
    ///
    /// A segment with `resumable: false` is rejected: engine-core reads that
    /// flag as the closing ADD, so it would end the session instead of
    /// extending it, and discard the segment's own prompt. Use [`Self::finish`].
    pub async fn push(&self, segment: GenerateRequest) -> Result<()> {
        if !segment.resumable {
            return Err(Error::NotResumable {
                request_id: self.request_id.clone(),
            });
        }
        self.send(segment).await
    }

    /// Close the session with the sentinel ADD, ending the output stream.
    ///
    /// Mirrors Python's `final_req`: a dummy one-token prompt with
    /// `resumable: false`, whose finish reason terminates the stream and frees
    /// the engine's KV.
    pub async fn finish(&self) -> Result<()> {
        self.send(GenerateRequest::sentinel(self.request_id.clone())).await
    }

    /// Prepare one chunk and send it as a continuation of the session.
    async fn send(&self, segment: GenerateRequest) -> Result<()> {
        // Chunks are never id-randomized: they must land on the session's
        // engine-side id, which `generate_streaming` already resolved.
        let mut prepared = segment.prepare(false)?;
        prepared.engine_request.request_id = self.request_id.clone();
        self.client.call_continuation(prepared.engine_request).await?;
        Ok(())
    }
}
