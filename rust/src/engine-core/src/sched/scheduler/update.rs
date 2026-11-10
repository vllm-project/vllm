// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Processing model runner outputs: `update_from_output` and the draft token
//! updates.

use indexmap::{IndexMap, IndexSet};
use rustc_hash::{FxHashMap, FxHashSet};

use super::connector::{KvConnectorOutput, KvTransferParams};
use super::{ErrorRequestIds, FailedKvLoads, Scheduler};
use crate::kv_cache::KvCacheEvent;
use crate::sched::output::Opaque;
use crate::sched::output::SchedulerOutput;
use crate::sched::request::{
    EngineCoreEvent, EngineCoreEventType, FinishReason, PrefillStats, RequestStatus, StopReason,
    StreamingUpdate,
};
use crate::sched::stats::{
    RequestSpecDecodeMetrics, SchedulerStats, SpecDecodeMetricsLevel, SpecDecodingStats,
};
use crate::sched::utils::{check_stop, remove_all};

/// `ModelRunnerOutput` as the scheduler reads it. Tensors stay with the
/// binding layer, which attaches them to the [`EngineCoreOutput`]s by
/// [`EngineCoreOutput::req_index`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ModelRunnerOutput {
    /// [num_reqs]
    pub req_ids: Vec<String>,
    /// num_reqs x num_generated_tokens. Empty when nothing was sampled.
    pub sampled_token_ids: Vec<Vec<i32>>,
    /// Whether `logprobs` is set.
    pub has_logprobs: bool,
    /// Whether `sampling_masks` is set.
    pub has_sampling_masks: bool,
    /// Whether each `pooler_output` entry is not `None`; empty when the
    /// list is empty.
    pub pooler_output: Vec<bool>,
    /// req_id -> num_nans_in_logits
    pub num_nans_in_logits: Option<FxHashMap<String, usize>>,
    pub kv_connector_output: Option<KvConnectorOutput>,
    /// The worker-side `ECConnectorOutput`, when truthy.
    pub ec_connector_output: Option<Opaque>,
    /// The workers' `aux_output_connector_output`.
    pub aux_output_connector_output: Option<Opaque>,
}

/// `vllm.outputs.RequestError`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RequestError {
    pub code: String,
    pub message: String,
}

/// An output of one request in a step (`EngineCoreOutput`).
///
/// Payloads that stay with the binding layer are flagged instead: it slices
/// them from the step's `ModelRunnerOutput` at [`Self::req_index`]
/// (`logprobs`, `sampling_masks`, `pooler_output`, and
/// `prompt_logprobs_dict` / `prompt_token_id_logprobs_dict` by request id),
/// and attaches the request's `trace_headers`.
#[derive(Clone, Debug, PartialEq)]
pub struct EngineCoreOutput {
    pub request_id: String,
    pub new_token_ids: Vec<i32>,
    /// The request's index in `ModelRunnerOutput.req_ids`; `None` for
    /// outputs not produced by the model (finished with an error).
    pub req_index: Option<usize>,
    /// `logprobs.slice_request(req_index, len(new_token_ids))`.
    pub new_logprobs: bool,
    /// `sampling_masks.slice_request(req_index, len(new_token_ids))`.
    pub new_sampling_mask: bool,
    /// `pooler_output[req_index]`.
    pub pooling_output: bool,
    pub finish_reason: Option<FinishReason>,
    pub stop_reason: Option<StopReason>,
    pub events: Option<Vec<EngineCoreEvent>>,
    pub kv_transfer_params: Option<KvTransferParams>,
    /// The EC connector's `request_finished` params.
    pub ec_transfer_params: Option<Opaque>,
    /// The aux output connector's rows for the request.
    pub routed_experts: Option<Opaque>,
    pub prefill_stats: Option<PrefillStats>,
    pub num_nans_in_logits: usize,
    /// Per-request spec-decode acceptance; attached only on the final output.
    pub spec_decode_metrics: Option<RequestSpecDecodeMetrics>,
    pub error: Option<RequestError>,
    pub watermarking: Option<bool>,
}

impl EngineCoreOutput {
    fn new(request_id: String, new_token_ids: Vec<i32>) -> Self {
        Self {
            request_id,
            new_token_ids,
            req_index: None,
            new_logprobs: false,
            new_sampling_mask: false,
            pooling_output: false,
            finish_reason: None,
            stop_reason: None,
            events: None,
            kv_transfer_params: None,
            ec_transfer_params: None,
            routed_experts: None,
            prefill_stats: None,
            num_nans_in_logits: 0,
            spec_decode_metrics: None,
            error: None,
            watermarking: None,
        }
    }
}

/// The outputs of one client in a step (`EngineCoreOutputs`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EngineCoreOutputs {
    /// [num_reqs]
    pub outputs: Vec<EngineCoreOutput>,
    pub scheduler_stats: Option<SchedulerStats>,
    pub finished_requests: Option<IndexSet<String>>,
}

/// What `update_from_output` returns: the outputs by client index, plus the
/// side products the binding layer forwards to Python objects.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct UpdateOutput {
    /// client_index -> outputs, in Python's dict order.
    pub engine_core_outputs: IndexMap<usize, EngineCoreOutputs>,
    /// KV cache events of this step from the KV cache manager, to publish
    /// (with the connector's events) as one `KVEventBatch`.
    pub kv_cache_events: Vec<KvCacheEvent>,
    /// KV cache events from the connector.
    pub connector_kv_cache_events: Option<Opaque>,
    /// Scheduler-side connector stats, to aggregate with the worker-side
    /// `kv_connector_stats` into `SchedulerStats.kv_connector_stats`.
    pub scheduler_kv_connector_stats: Option<Opaque>,
    /// Scheduler-side EC connector stats, to aggregate with the worker-side
    /// `ec_connector_stats` into `SchedulerStats.ec_connector_stats`.
    pub scheduler_ec_connector_stats: Option<Opaque>,
}

/// `DraftTokenIds`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DraftTokenIds {
    /// [num_reqs]
    pub req_ids: Vec<String>,
    /// num_reqs x num_draft_tokens
    pub draft_token_ids: Vec<Vec<i32>>,
}

impl Scheduler {
    pub fn update_from_output(
        &mut self,
        scheduler_output: &SchedulerOutput,
        model_runner_output: &ModelRunnerOutput,
    ) -> UpdateOutput {
        let sampled_token_ids = &model_runner_output.sampled_token_ids;
        let num_scheduled_tokens = &scheduler_output.num_scheduled_tokens;
        let pooler_outputs = &model_runner_output.pooler_output;
        let num_nans_in_logits = &model_runner_output.num_nans_in_logits;
        let kv_connector_output = &model_runner_output.kv_connector_output;

        // Every GPU write enqueued by this and earlier steps has completed, so it is
        // safe to return deferred-free blocks to the pool.
        if self.defer_block_free && scheduler_output.total_num_scheduled_tokens > 0 {
            self.processed_step_seq += 1;
            self.drain_deferred_frees();
            self.drain_deferred_loads();
        }

        let mut outputs: IndexMap<usize, Vec<EngineCoreOutput>> = IndexMap::new();
        let mut spec_decoding_stats: Option<SpecDecodingStats> = None;

        let mut failed_kv_loads = FailedKvLoads::default();
        if let Some(kv_output) = kv_connector_output {
            if !kv_output.failed_recving.is_empty() {
                failed_kv_loads.recving = self.handle_failed_recving(&kv_output.failed_recving);
            }
            if !kv_output.invalid_block_ids.is_empty() {
                // These blocks contain externally computed tokens that failed to
                // load. Identify affected requests and adjust their computed token
                // count to trigger recomputation of the invalid blocks.
                (failed_kv_loads.invalid_async, failed_kv_loads.invalid_sync) =
                    self.handle_invalid_blocks(&kv_output.invalid_block_ids, num_scheduled_tokens);
            }
        }
        let failed_kv_load_req_ids: IndexSet<String> = failed_kv_loads
            .recving
            .iter()
            .chain(&failed_kv_loads.invalid_async)
            .chain(&failed_kv_loads.invalid_sync)
            .cloned()
            .collect();
        // NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        // the below loop can be a performance bottleneck. We should do our best
        // to avoid expensive operations inside the loop.
        let mut stopped_running_reqs: FxHashSet<usize> = FxHashSet::default();
        let mut stopped_preempted_reqs: FxHashSet<usize> = FxHashSet::default();
        // Stopped resumable requests to re-enqueue, in output order.
        let mut continuing_reqs: Vec<usize> = Vec::new();
        // The worker's output order follows its batch, not num_scheduled_tokens.
        let req_indices: FxHashMap<&str, usize> = model_runner_output
            .req_ids
            .iter()
            .enumerate()
            .map(|(i, req_id)| (req_id.as_str(), i))
            .collect();
        for (req_id, &num_tokens_scheduled) in num_scheduled_tokens {
            assert!(num_tokens_scheduled > 0);
            let h = self.requests.handle(req_id);
            let mut output_is_stale = false;
            if let Some(h) = h {
                let request = &mut self.requests[h];
                request.num_in_flight_tokens -= num_tokens_scheduled;
                // Drain any stale share (see _preempt_request) in lockstep.
                if request.num_stale_output_tokens > 0 {
                    output_is_stale = true;
                    request.num_stale_output_tokens = request
                        .num_stale_output_tokens
                        .checked_sub(num_tokens_scheduled)
                        .expect("num_stale_output_tokens >= 0");
                }
            }
            if !failed_kv_load_req_ids.is_empty() && failed_kv_load_req_ids.contains(req_id) {
                // skip failed or rescheduled requests from KV load failure
                continue;
            }
            let Some(h) = h.filter(|&h| !self.requests[h].is_finished()) else {
                // The request is already finished. This can happen if the
                // request is aborted while the model is executing it (e.g.,
                // in pipeline parallelism or in async scheduling).
                // NOTE(Kuntai): When delay_free_blocks=True (for async KV
                // cache transfer in KV connector), the aborted request will not
                // be set to None (in order to finish async KV transfer).
                // In this case, we use is_finished() to check.
                continue;
            };

            // Drop-mode stale output (same-step resume) is discarded entirely.
            if output_is_stale && self.requests[h].drop_stale_output {
                continue;
            }

            let req_index = req_indices[req_id.as_str()];
            let mut generated_token_ids: Vec<i32> = if sampled_token_ids.is_empty() {
                Vec::new()
            } else {
                sampled_token_ids[req_index].clone()
            };

            let scheduled_spec_token_ids =
                scheduler_output.scheduled_spec_decode_tokens.get(req_id);
            if let Some(scheduled_spec_token_ids) = scheduled_spec_token_ids
                && !scheduled_spec_token_ids.is_empty()
                && (!generated_token_ids.is_empty() || self.num_sampled_tokens_per_step == 0)
            {
                let num_draft_tokens = scheduled_spec_token_ids.len();
                let num_sampled = self.num_sampled_tokens_per_step;
                let num_accepted = generated_token_ids.len().saturating_sub(num_sampled);
                let num_rejected = num_draft_tokens as i64 - num_accepted as i64;
                // Rejections roll back num_computed_tokens (and, under async
                // scheduling, num_output_placeholders, which covers the spec
                // tokens). A stale rejection count predates the preemption
                // rollback and must not apply.
                let request = &mut self.requests[h];
                if !output_is_stale {
                    if request.num_computed_tokens > 0 {
                        request.num_computed_tokens =
                            (request.num_computed_tokens as i64 - num_rejected) as usize;
                    }
                    if request.num_output_placeholders > 0 {
                        request.num_output_placeholders =
                            (request.num_output_placeholders as i64 - num_rejected) as usize;
                    }
                }
                let num_invalid_spec_tokens = scheduler_output.num_invalid_spec_tokens.as_ref();
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens,
                    num_accepted,
                    num_invalid_spec_tokens,
                    req_id,
                );
                let request = &mut self.requests[h];
                if let Some(metrics) = &mut request.spec_decode_metrics {
                    // Exclude grammar-invalidated drafts from the proposed
                    // count, mirroring make_spec_decoding_stats; the accepted
                    // bucket (j) is unaffected.
                    let mut adj_draft_tokens = num_draft_tokens;
                    if let Some(invalid) = num_invalid_spec_tokens
                        && !invalid.is_empty()
                    {
                        adj_draft_tokens -= invalid.get(req_id).copied().unwrap_or(0);
                    }
                    metrics.observe(
                        adj_draft_tokens,
                        num_accepted,
                        self.spec_decode_metrics_level == SpecDecodeMetricsLevel::Detailed,
                    );
                }
            }

            // Free encoder inputs only after the step has actually executed.
            if self.requests[h].has_encoder_inputs() {
                self.free_encoder_inputs(h);
            }

            let mut stopped = false;
            let pooler_output = !pooler_outputs.is_empty() && pooler_outputs[req_index];
            let mut kv_transfer_params = None;
            let mut ec_transfer_params = None;
            let mut prefill_stats = None;
            let mut watermarking = None;
            let status_before_stop = self.requests[h].status;

            // Check for stop and update request status.
            let mut new_token_ids = std::mem::take(&mut generated_token_ids);
            if !new_token_ids.is_empty() {
                let has_output_placeholders = self.requests[h].num_output_placeholders > 0;
                stopped = self.update_request_with_output(h, &mut new_token_ids, output_is_stale);
                if !output_is_stale
                    && status_before_stop == RequestStatus::Running
                    && (has_output_placeholders
                        || self.num_spec_tokens > 0
                        || (self.has_mamba_layers
                            && !self.async_scheduling
                            && self.enable_prefix_caching))
                {
                    // Publish finalized states before retirement or completion.
                    // A failed KV load can roll num_computed_tokens back below
                    // the in-flight count; Python then passes a negative count,
                    // which every manager treats like 0.
                    let request = &self.requests[h];
                    self.kv_cache_manager.cache_blocks(
                        request,
                        request.num_computed_tokens.saturating_sub(request.num_in_flight_tokens),
                    );
                }
            } else if self.requests[h].pooling_params.is_some() && pooler_output {
                // Pooling stops as soon as there is output.
                self.requests[h].status = RequestStatus::FinishedStopped;
                stopped = true;
            } else if self.is_mm_encoder_only
                && self.requests[h].num_computed_tokens >= self.requests[h].num_prompt_tokens
            {
                // An encoder instance runs the encoder and publishes the
                // embeddings instead of sampling, so it stops as soon as the
                // whole prompt is consumed. Encoder inputs are never scheduled
                // past a multi-modal item the encoder cache could not admit, so
                // a consumed prompt also means every item in it was encoded.
                self.requests[h].status = RequestStatus::FinishedStopped;
                stopped = true;
            }

            if !new_token_ids.is_empty()
                && self.requests[h].use_structured_output()
                && !self.structured_output.accept_tokens(&self.requests[h], &new_token_ids)
            {
                // TODO: Python logs "Unexpected: grammar rejected tokens"
                // at error level.
                let request = &mut self.requests[h];
                request.status = RequestStatus::FinishedError;
                request.resumable = false;
                stopped = true;
            }

            let should_emit_output = !new_token_ids.is_empty() || pooler_output || stopped;
            let mut routed_experts = None;
            if let Some(aux) = self.aux_output_connector.as_mut()
                && should_emit_output
            {
                routed_experts = aux.take_output(
                    &self.requests[h],
                    model_runner_output.aux_output_connector_output.as_ref(),
                );
            }
            if should_emit_output {
                prefill_stats = self.requests[h].take_prefill_stats();
                if let Some(stats) = &mut prefill_stats {
                    stats.finalize(self.kv_cache_manager.estimate_cached_tokens(&self.requests[h]));
                    if let Some(params) = &self.requests[h].sampling_params
                        && self.report_watermarking
                    {
                        // Report the flag after enforcement; the sampler skips
                        // greedy and trace replay rows.
                        watermarking = Some(
                            params.watermarking
                                && params.temperature != 0.0
                                && !params.has_trace_decode_token_ids,
                        );
                    }
                }
            }

            // Extract sample logprobs before stop handling can replace the
            // sampling parameters for a streaming continuation.
            let new_logprobs = self.requests[h]
                .sampling_params
                .as_ref()
                .is_some_and(|sp| sp.num_logprobs.is_some())
                && model_runner_output.has_logprobs;

            let mut finish_reason = None;
            if stopped {
                // Capture finish_reason BEFORE _handle_stopped_request, which may
                // reset the status to WAITING for streaming requests that continue.
                finish_reason = self.requests[h].get_finished_reason();
                let finished = self.handle_stopped_request(h);
                if finished {
                    (kv_transfer_params, ec_transfer_params) = self.free_request(h, false);
                    if let (Some(prefill_stats), Some(params)) =
                        (&prefill_stats, &mut kv_transfer_params)
                        && params.do_remote_prefill
                    {
                        // P-side cache hits, so D can report them in
                        // prompt_tokens_details instead of its own (~100%)
                        // hit rate from the KV transfer.
                        params.remote_prefill_cached_tokens = Some(prefill_stats.num_cached_tokens);
                    }
                } else {
                    continuing_reqs.push(h);
                }

                if status_before_stop == RequestStatus::Running {
                    stopped_running_reqs.insert(h);
                } else {
                    stopped_preempted_reqs.insert(h);
                }
            }

            let new_sampling_mask = self.return_sampling_mask
                && !new_token_ids.is_empty()
                && model_runner_output.has_sampling_masks;

            let request = &mut self.requests[h];
            if let Some(nans) = num_nans_in_logits
                && let Some(&n) = nans.get(req_id)
            {
                request.num_nans_in_logits = n;
            }

            if should_emit_output {
                // Add EngineCoreOutput for this Request.
                let spec_decode_metrics = if finish_reason.is_some() {
                    request.spec_decode_metrics.clone()
                } else {
                    None
                };
                outputs.entry(request.client_index).or_default().push(EngineCoreOutput {
                    req_index: Some(req_index),
                    new_logprobs,
                    new_sampling_mask,
                    pooling_output: pooler_output,
                    finish_reason,
                    stop_reason: request.stop_reason.clone(),
                    events: request.take_events(),
                    kv_transfer_params,
                    ec_transfer_params,
                    routed_experts,
                    prefill_stats,
                    num_nans_in_logits: request.num_nans_in_logits,
                    spec_decode_metrics,
                    watermarking,
                    ..EngineCoreOutput::new(req_id.clone(), new_token_ids)
                });
            }
            // Invariant: EngineCore returns no partial prefill outputs (no
            // prompt logprobs without an output); the binding layer owns the
            // prompt logprobs and checks it.
        }

        // Remove stopped requests from their queues, then enqueue the ones
        // that continue, so each ends up in exactly one queue.
        if !stopped_running_reqs.is_empty() {
            remove_all(&mut self.running, &stopped_running_reqs);
        }
        if !stopped_preempted_reqs.is_empty() {
            let stopped: Vec<usize> = stopped_preempted_reqs.iter().copied().collect();
            self.waiting.remove_requests(stopped.iter().copied(), &self.requests);
            self.kv_holding_waiting.remove_requests(stopped.iter().copied(), &self.requests);
            self.async_kv_waiting.remove_requests(stopped.iter().copied(), &self.requests);
            for h in &stopped {
                self.deferred_waiting.remove(h);
            }
        }
        for h in continuing_reqs {
            self.enqueue_waiting_request(h);
        }

        let request_errors = std::mem::take(&mut self.grammar_compile_error_reqs);
        let mut error_ids = ErrorRequestIds {
            grammar: request_errors.keys().cloned().collect(),
            encoder_cache_mismatch: std::mem::take(&mut self.encoder_cache_mismatch_reqs),
            failed_kv_loads: FailedKvLoads::default(),
            ec_unavailable: IndexSet::new(),
        };
        if !failed_kv_load_req_ids.is_empty() && !self.recompute_kv_load_failures {
            error_ids.failed_kv_loads = failed_kv_loads;
        }
        if let Some(ec_connector) = self.ec_connector.as_mut() {
            // An encoder input the connector can no longer obtain. Failing is
            // retryable: re-issuing the request re-runs the encode.
            error_ids.ec_unavailable = ec_connector.take_unavailable_requests();
        }

        if !error_ids.is_empty() {
            // Python finishes them iterating a set.
            let error_req_ids = self.error_request_order(&error_ids);
            let error_reqs =
                self.finish_request_handles(Some(error_req_ids), RequestStatus::FinishedError);
            for h in error_reqs {
                let request = &mut self.requests[h];
                let request_id = request.request_id.clone();
                outputs.entry(request.client_index).or_default().push(EngineCoreOutput {
                    finish_reason: request.get_finished_reason(),
                    events: request.take_events(),
                    error: request_errors.get(&request_id).cloned(),
                    ..EngineCoreOutput::new(request_id, Vec::new())
                });
            }
        }

        // KV Connector: update state for finished KV Transfers.
        if let Some(kv_output) = kv_connector_output {
            self.update_from_kv_xfer_finished(kv_output);
        }

        // EC Connector: update state from worker-side EC connector output.
        if let (Some(ec_connector), Some(ec_output)) = (
            self.ec_connector.as_mut(),
            &model_runner_output.ec_connector_output,
        ) {
            ec_connector.update_connector_output(ec_output);
        }

        // Scheduler-side KV connector stats collected after connector update.
        let scheduler_kv_connector_stats = self
            .connector
            .as_mut()
            .and_then(|c| c.get_kv_connector_stats(&mut self.kv_cache_manager));

        // Scheduler-side EC connector stats collected after connector update.
        let scheduler_ec_connector_stats =
            self.ec_connector.as_mut().and_then(|c| c.get_ec_connector_stats());

        // collect KV cache events from KV cache manager
        let kv_cache_events = self.kv_cache_manager.take_events();

        // collect KV cache events from connector
        let connector_kv_cache_events =
            self.connector.as_mut().and_then(|c| c.take_events(&mut self.kv_cache_manager));

        // Create EngineCoreOutputs for all clients that have requests with
        // outputs in this step.
        let mut engine_core_outputs: IndexMap<usize, EngineCoreOutputs> = outputs
            .into_iter()
            .map(|(client_index, outs)| {
                (
                    client_index,
                    EngineCoreOutputs {
                        outputs: outs,
                        ..Default::default()
                    },
                )
            })
            .collect();

        if let Some(finished_req_ids) = &mut self.finished_req_ids_dict
            && !finished_req_ids.is_empty()
        {
            // Include ids of requests that finished since last outputs
            // were sent.
            for (client_index, finished_set) in finished_req_ids.drain(..) {
                // Set finished request set in EngineCoreOutputs for this client.
                engine_core_outputs.entry(client_index).or_default().finished_requests =
                    Some(finished_set);
            }
        }

        if let Some(stats) = self.make_stats(spec_decoding_stats) {
            // Return stats to only one of the front-ends.
            if engine_core_outputs.is_empty() {
                // We must return the stats even if there are no request
                // outputs this step.
                engine_core_outputs.insert(0, EngineCoreOutputs::default());
            }
            engine_core_outputs.get_index_mut(0).unwrap().1.scheduler_stats = Some(stats);
        }

        self.flush_removed();
        UpdateOutput {
            engine_core_outputs,
            kv_cache_events,
            connector_kv_cache_events,
            scheduler_kv_connector_stats,
            scheduler_ec_connector_stats,
        }
    }

    /// Return True if finished (can be False for resumable requests).
    fn handle_stopped_request(&mut self, h: usize) -> bool {
        let request = &mut self.requests[h];
        if !request.resumable {
            return true;
        }

        // Drop the finished turn's in-flight work and resume from the
        // materialized frontier.
        let safe_frontier = request
            .num_computed_tokens
            .checked_sub(request.num_output_placeholders)
            .expect("safe_frontier >= 0");
        request.num_computed_tokens = safe_frontier;
        request.spec_token_ids = Vec::new();
        request.drop_stale_output = true;
        request.num_stale_output_tokens = request.num_in_flight_tokens;
        request.num_output_placeholders = 0;

        if let Some(queue) = request.streaming_queue.as_mut()
            && !queue.is_empty()
        {
            let update = queue.pop_front().unwrap();
            let Some(update) = update else {
                // Streaming request finished.
                return true;
            };
            self.update_request_as_session(h, update);
        } else {
            request.status = RequestStatus::WaitingForStreamingReq;
            self.num_waiting_for_streaming_input += 1;
        }

        false
    }

    /// Updates the waiting session with the next streaming update.
    ///
    /// Discards the last sampled output token from the prior input chunk.
    pub(super) fn update_request_as_session(&mut self, h: usize, update: StreamingUpdate) {
        let hash_block_size = self.hash_block_size;
        let session = &mut self.requests[h];
        // Keep all output tokens but the final sampled one.
        let keep_end = session.num_tokens() - 1;
        assert!(keep_end >= session.num_computed_tokens);
        session.truncate_all_token_ids(keep_end);
        session.block_hashes.truncate(keep_end / hash_block_size);
        session.clear_output_token_ids();
        let num_prompt_tokens = session.num_prompt_tokens;
        let kept = session.all_token_ids()[num_prompt_tokens..keep_end].to_vec();
        let prompt = session.prompt_token_ids.as_mut().expect("prompt_token_ids is set");
        // Extend prompt with kept output tokens.
        prompt.extend_from_slice(&kept);

        if let Some(mut mm_features) = update.mm_features.filter(|f| !f.is_empty()) {
            let base = session.num_tokens();
            for mm_feature in &mut mm_features {
                mm_feature.mm_position =
                    mm_feature.mm_position.with_offset(mm_feature.mm_position.offset + base);
            }
            session.mm_features.extend(mm_features);
        }

        let new_tokens = update.prompt_token_ids.unwrap_or_default();
        session.extend_all_token_ids(&new_tokens);
        let prompt = session.prompt_token_ids.as_mut().unwrap();
        prompt.extend_from_slice(&new_tokens);
        // Update block hashes for the new tokens.
        session.update_block_hashes();
        session.num_prompt_tokens = session.prompt_token_ids.as_ref().unwrap().len();
        session.arrival_time = update.arrival_time;
        session.sampling_params = update.sampling_params;
        session.prefix_cache_hit_limit = session.get_prefix_cache_hit_limit();
        session.max_tokens = update.max_tokens;
        if session.status == RequestStatus::WaitingForStreamingReq {
            self.num_waiting_for_streaming_input -= 1;
        }
        session.status = RequestStatus::Waiting;

        if self.log_stats {
            let now = (self.clock)();
            self.requests[h].record_event(EngineCoreEventType::Queued, now);
        }
    }

    /// Returns whether the request stopped; trims `new_token_ids` at the stop.
    fn update_request_with_output(
        &mut self,
        h: usize,
        new_token_ids: &mut Vec<i32>,
        is_stale: bool,
    ) -> bool {
        let max_model_len = self.max_model_len;
        let request = &mut self.requests[h];
        let num_new_tokens = new_token_ids.len();
        // Append generated tokens and check for stop. Note that if
        // a request is still being prefilled, we expect the model runner
        // to return empty token ids for the request.
        let mut stopped = false;
        for num_new in 1..=num_new_tokens {
            let output_token_id = new_token_ids[num_new - 1];
            request.append_output_token_ids(&[output_token_id]);

            // Check for stop and update request state.
            // This must be called before we make the EngineCoreOutput.
            stopped = check_stop(request, max_model_len);
            if stopped {
                // num_computed_tokens covers all new tokens but the last one.
                // Keep that true after trimming: resumable requests resume from it.
                if !is_stale {
                    request.num_computed_tokens -= num_new_tokens - num_new;
                }
                new_token_ids.truncate(num_new); // Trim new tokens if needed.
                break;
            }
        }

        if !is_stale && request.num_output_placeholders > 0 {
            // Resolve all returned tokens, including those trimmed at a stop.
            request.num_output_placeholders = request
                .num_output_placeholders
                .checked_sub(num_new_tokens)
                .expect("num_output_placeholders >= 0");
        }
        stopped
    }

    fn free_encoder_inputs(&mut self, h: usize) {
        let request = &self.requests[h];
        let cached_encoder_input_ids = self.encoder_cache_manager.get_cached_input_ids(request);
        // OPTIMIZATION: Avoid list(set) if the set is empty.
        if cached_encoder_input_ids.is_empty() {
            return;
        }

        let mut to_free = Vec::new();
        for input_id in cached_encoder_input_ids {
            let mm_feature = &request.mm_features[input_id];
            let start_pos = mm_feature.mm_position.offset;
            let num_tokens = mm_feature.mm_position.length;
            if self.is_encoder_decoder && request.num_computed_tokens > 0 {
                // With Whisper, as soon as we've generated a single token,
                // we know we're done with the encoder input. Cross Attention
                // KVs have been calculated and cached already.
                to_free.push(input_id);
            } else if (start_pos + num_tokens) as i64
                <= request.num_computed_tokens as i64 - request.num_output_placeholders as i64
            {
                // Processed, stored in the decoder KV cache, and past the
                // placeholder range with no pending rejection that could roll
                // back into it. The drafter only reads ahead of
                // num_computed_tokens, so it cannot reference this input either.
                to_free.push(input_id);
            }
        }
        for input_id in to_free {
            self.free_encoder_input(h, input_id);
        }
    }

    fn free_encoder_input(&mut self, h: usize, input_id: usize) {
        let request = &self.requests[h];
        self.encoder_cache_manager.free_encoder_input(request, input_id);
        if let Some(ec_connector) = self.ec_connector.as_mut() {
            ec_connector.update_state_after_free(request, input_id);
        }
    }

    pub fn update_draft_token_ids(&mut self, draft_token_ids: DraftTokenIds) {
        for (req_id, spec_token_ids) in
            draft_token_ids.req_ids.iter().zip(draft_token_ids.draft_token_ids)
        {
            let Some(h) = self.requests.handle(req_id) else {
                // The request may have been finished. Skip.
                continue;
            };
            let request = &mut self.requests[h];
            if request.is_finished() {
                continue;
            }

            if request.is_prefill_chunk {
                // Ignore draft tokens for prefill chunks.
                if !request.spec_token_ids.is_empty() {
                    request.spec_token_ids = Vec::new();
                }
                continue;
            }

            // Add newly generated spec token ids to the request.
            request.spec_token_ids = if request.use_structured_output() {
                self.structured_output.trim_spec_tokens(request, spec_token_ids)
            } else {
                spec_token_ids
            };
        }
    }

    pub fn update_draft_token_ids_in_output(
        &mut self,
        draft_token_ids: DraftTokenIds,
        scheduler_output: &mut SchedulerOutput,
    ) {
        let mut num_invalid_spec_tokens: IndexMap<String, usize> = IndexMap::new();

        let sched_spec_tokens = &mut scheduler_output.scheduled_spec_decode_tokens;
        for (req_id, mut spec_token_ids) in
            draft_token_ids.req_ids.into_iter().zip(draft_token_ids.draft_token_ids)
        {
            let Some(h) = self.requests.handle(&req_id) else {
                // The request may have been finished. Skip.
                continue;
            };
            if self.requests[h].is_finished() {
                continue;
            }

            let Some(placeholder_spec_tokens) = sched_spec_tokens.get(&req_id) else {
                continue;
            };
            if placeholder_spec_tokens.is_empty() {
                continue;
            }

            let orig_num_spec_tokens = placeholder_spec_tokens.len();
            // Trim drafts to scheduled number of spec tokens
            // (needed for chunked prefill case for example).
            spec_token_ids.truncate(orig_num_spec_tokens);
            // Preserve draft slots; rejection sampling handles grammar violations.
            // Pad to original number of spec tokens.
            let num_invalid_tokens = orig_num_spec_tokens - spec_token_ids.len();
            if num_invalid_tokens > 0 {
                spec_token_ids.resize(orig_num_spec_tokens, -1);
                num_invalid_spec_tokens.insert(req_id.clone(), num_invalid_tokens);
            }

            sched_spec_tokens.insert(req_id, spec_token_ids);
        }

        scheduler_output.num_invalid_spec_tokens = Some(num_invalid_spec_tokens);
    }

    fn make_spec_decoding_stats(
        &self,
        spec_decoding_stats: Option<SpecDecodingStats>,
        mut num_draft_tokens: usize,
        num_accepted_tokens: usize,
        num_invalid_spec_tokens: Option<&IndexMap<String, usize>>,
        request_id: &str,
    ) -> Option<SpecDecodingStats> {
        if !self.log_stats || num_draft_tokens == 0 {
            return None;
        }
        let mut spec_decoding_stats =
            spec_decoding_stats.unwrap_or_else(|| SpecDecodingStats::new(self.num_spec_tokens));
        if let Some(invalid) = num_invalid_spec_tokens {
            num_draft_tokens -= invalid.get(request_id).copied().unwrap_or(0);
        }
        spec_decoding_stats.observe_draft(num_draft_tokens, num_accepted_tokens);
        Some(spec_decoding_stats)
    }

    pub fn make_stats(
        &mut self,
        spec_decoding_stats: Option<SpecDecodingStats>,
    ) -> Option<SchedulerStats> {
        if !self.log_stats {
            return None;
        }
        let prefix_cache_stats = self
            .kv_cache_manager
            .make_prefix_cache_stats()
            .expect("prefix cache stats exist when log_stats is set");
        let connector_prefix_cache_stats =
            self.connector_prefix_cache_stats.as_mut().map(std::mem::take);
        let kv_cache_eviction_events = self
            .kv_cache_manager
            .block_pool_mut()
            .metrics_collector
            .as_mut()
            .map(|c| c.drain_events())
            .unwrap_or_default();
        let (num_running, num_waiting) = self.get_request_counts();
        let num_deferred = self.deferred_waiting.len();
        Some(SchedulerStats {
            num_running_reqs: num_running,
            num_waiting_reqs: num_waiting - num_deferred,
            num_skipped_waiting_reqs: num_deferred,
            num_kv_fetch_reqs_by_stage: self.kv_fetch_counts,
            kv_cache_usage: self.kv_cache_manager.usage(),
            prefix_cache_stats,
            connector_prefix_cache_stats,
            kv_cache_eviction_events,
            spec_decoding_stats,
        })
    }
}
