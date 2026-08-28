// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::time::{SystemTime, UNIX_EPOCH};

use vllm_engine_core_client::protocol::output::{
    EngineCoreEvent, EngineCoreEventType, EngineCoreOutput,
};
use vllm_engine_core_client::protocol::stats::PrefillStats;
use vllm_metrics::{
    EngineLabels, Family, FinishedReasonLabels, HistogramMetric, METRICS, PromptTokenSourceLabels,
    U64Counter,
};

use crate::FinishReason;

const PROMPT_TOKEN_SOURCE_LOCAL_COMPUTE: &str = "local_compute";
const PROMPT_TOKEN_SOURCE_LOCAL_CACHE_HIT: &str = "local_cache_hit";
const PROMPT_TOKEN_SOURCE_EXTERNAL_KV_TRANSFER: &str = "external_kv_transfer";

/// Request-scoped metrics state tracked across streamed engine-core updates.
///
/// This is the Rust-side counterpart of the Python frontend's request-lifecycle
/// bookkeeping, centered on `RequestStateStats` and the per-output/per-finished
/// update flow.
///
/// Original Python definitions:
/// <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/v1/metrics/stats.py#L200-L237>
///
/// Original Python update flow:
/// <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/v1/engine/output_processor.py#L600-L677>
#[derive(Clone)]
pub(crate) struct RequestMetricsTracker {
    /// Cached request metric handles for this request's model and engine index.
    handles: RequestMetricHandles,

    arrival_time: f64,
    prompt_len: u32,
    max_tokens_param: Option<u32>,
    n_param: u32,
    is_prefilling: bool,
    queued_ts: f64,
    scheduled_ts: f64,
    first_token_ts: f64,
    last_token_ts: f64,
    first_token_latency: f64,
    num_generation_tokens: u32,
    latest_num_cached_tokens: u32,
}

/// Cached request metric handles for one model and engine index.
#[derive(Clone)]
struct RequestMetricHandles {
    labels: EngineLabels,

    // Request-derived counters.
    num_preemptions: U64Counter,
    prompt_tokens: U64Counter,
    prompt_tokens_local_compute: U64Counter,
    prompt_tokens_local_cache_hit: U64Counter,
    prompt_tokens_external_kv_transfer: U64Counter,
    prompt_tokens_cached: U64Counter,
    generation_tokens: U64Counter,

    // Request lifecycle counters and histograms.
    request_success: Family<FinishedReasonLabels, U64Counter>,
    request_prompt_tokens: HistogramMetric,
    request_generation_tokens: HistogramMetric,
    request_max_num_generation_tokens: HistogramMetric,
    request_params_max_tokens: HistogramMetric,
    request_params_n: HistogramMetric,
    request_prefill_kv_computed_tokens: HistogramMetric,
    time_to_first_token_seconds: HistogramMetric,
    inter_token_latency_seconds: HistogramMetric,
    e2e_request_latency_seconds: HistogramMetric,
    request_queue_time_seconds: HistogramMetric,
    request_prefill_time_seconds: HistogramMetric,
    request_decode_time_seconds: HistogramMetric,
    request_inference_time_seconds: HistogramMetric,
    request_time_per_output_token_seconds: HistogramMetric,
}

impl RequestMetricsTracker {
    /// Create the per-request tracker from the normalized `llm`-layer request
    /// context.
    pub(crate) fn new(
        model_name: String,
        engine_index: u32,
        arrival_time: f64,
        prompt_len: u32,
        max_tokens_param: Option<u32>,
        n_param: u32,
    ) -> Self {
        Self {
            handles: resolve_request_metric_handles(&model_name, engine_index),
            arrival_time,
            prompt_len,
            max_tokens_param,
            n_param,
            is_prefilling: true,
            queued_ts: 0.0,
            scheduled_ts: 0.0,
            first_token_ts: 0.0,
            last_token_ts: 0.0,
            first_token_latency: 0.0,
            num_generation_tokens: 0,
            latest_num_cached_tokens: 0,
        }
    }

    /// Update request-lifecycle state from one engine-core output item.
    ///
    /// Original Python stats logic:
    /// <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/v1/metrics/stats.py#L331-L384>
    pub(crate) fn observe_output(
        &mut self,
        batch_timestamp: f64,
        received_at: f64,
        output: &EngineCoreOutput,
    ) {
        if let Some(prefill_stats) = &output.prefill_stats {
            self.latest_num_cached_tokens = prefill_stats.num_cached_tokens;
        }
        self.num_generation_tokens += output.new_token_ids.len() as u32;
        self.handles.generation_tokens.inc_by(output.new_token_ids.len() as u64);

        if let Some(events) = &output.events {
            self.observe_events(events);
        }

        // Only outputs that actually carry tokens drive token-timing metrics.
        // A terminal output with no new tokens (e.g. the synthesized abort
        // output) must not log a stray time-to-first-token or inter-token
        // sample.
        if !output.new_token_ids.is_empty() {
            if self.is_prefilling {
                if let Some(prefill_stats) = &output.prefill_stats {
                    self.record_prompt_tokens(prefill_stats);
                }
                self.first_token_latency = received_at - self.arrival_time;
                self.handles.time_to_first_token_seconds.observe(self.first_token_latency);
                self.first_token_ts = batch_timestamp;
                self.is_prefilling = false;
            } else if self.last_token_ts > 0.0 {
                self.handles
                    .inter_token_latency_seconds
                    .observe(batch_timestamp - self.last_token_ts);
            }

            self.last_token_ts = batch_timestamp;
        }
    }

    /// Emit the terminal request metrics once a finished output has been
    /// observed.
    ///
    /// Original Python finished-request stats:
    /// <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/v1/metrics/stats.py#L222-L237>
    pub(crate) fn record_finished(&self, received_at: f64, finish_reason: FinishReason) {
        let prefill_kv_computed_tokens =
            self.prompt_len.saturating_sub(self.latest_num_cached_tokens);
        let e2e_latency_seconds = received_at - self.arrival_time;
        let queue_time_seconds = diff_or_zero(self.scheduled_ts, self.queued_ts);
        let prefill_time_seconds = diff_or_zero(self.first_token_ts, self.scheduled_ts);
        let decode_time_seconds = diff_or_zero(self.last_token_ts, self.first_token_ts);
        let inference_time_seconds = diff_or_zero(self.last_token_ts, self.scheduled_ts);
        let time_per_output_token_seconds = if self.num_generation_tokens > 1 {
            diff_or_zero(self.last_token_ts, self.first_token_ts)
                / (self.num_generation_tokens - 1) as f64
        } else {
            0.0
        };

        self.record_request_success(finish_reason);

        self.handles.request_prompt_tokens.observe(self.prompt_len as f64);
        self.handles
            .request_generation_tokens
            .observe(self.num_generation_tokens as f64);
        self.handles
            .request_max_num_generation_tokens
            .observe(self.num_generation_tokens as f64);
        if let Some(max_tokens_param) = self.max_tokens_param {
            self.handles.request_params_max_tokens.observe(max_tokens_param as f64);
        }
        self.handles.request_params_n.observe(self.n_param as f64);
        self.handles
            .request_prefill_kv_computed_tokens
            .observe(prefill_kv_computed_tokens as f64);
        self.handles.e2e_request_latency_seconds.observe(e2e_latency_seconds);
        self.handles.request_queue_time_seconds.observe(queue_time_seconds);
        self.handles.request_prefill_time_seconds.observe(prefill_time_seconds);
        self.handles.request_decode_time_seconds.observe(decode_time_seconds);
        self.handles.request_inference_time_seconds.observe(inference_time_seconds);
        self.handles
            .request_time_per_output_token_seconds
            .observe(time_per_output_token_seconds);
    }

    /// Record prompt token counters through cached metric handles.
    fn record_prompt_tokens(&self, prefill_stats: &PrefillStats) {
        let computed = prefill_stats.num_computed_tokens as u64;
        let local_cache_hit = prefill_stats.num_local_cached_tokens as u64;
        let external_kv_transfer = prefill_stats.num_external_cached_tokens as u64;

        self.handles.prompt_tokens.inc_by(prefill_stats.num_prompt_tokens as u64);
        self.handles.prompt_tokens_local_compute.inc_by(computed);
        self.handles.prompt_tokens_local_cache_hit.inc_by(local_cache_hit);
        self.handles.prompt_tokens_external_kv_transfer.inc_by(external_kv_transfer);
        self.handles.prompt_tokens_cached.inc_by(prefill_stats.num_cached_tokens as u64);
    }

    /// Record request event counters through cached metric handles.
    fn observe_events(&mut self, events: &[EngineCoreEvent]) {
        for event in events {
            match event.r#type {
                EngineCoreEventType::Queued => {
                    self.queued_ts = event.timestamp;
                }
                EngineCoreEventType::Scheduled => {
                    if self.scheduled_ts == 0.0 {
                        self.scheduled_ts = event.timestamp;
                    }
                }
                EngineCoreEventType::Preempted => {
                    self.handles.num_preemptions.inc();
                }
            }
        }
    }

    /// Increment the request-success counter for the terminal finish reason.
    fn record_request_success(&self, finish_reason: FinishReason) {
        self.handles
            .request_success
            .get_or_create(&FinishedReasonLabels {
                model_name: self.handles.labels.model_name.clone(),
                engine: self.handles.labels.engine,
                finished_reason: finish_reason.as_str(),
            })
            .inc();
    }
}

/// Resolve fixed request metric handles for one model and engine index.
fn resolve_request_metric_handles(model_name: &str, engine: u32) -> RequestMetricHandles {
    let metrics = &METRICS.request;
    let labels = EngineLabels {
        model_name: model_name.to_string(),
        engine,
    };

    RequestMetricHandles {
        num_preemptions: metrics.num_preemptions.get_or_create_owned(&labels),
        prompt_tokens: metrics.prompt_tokens.get_or_create_owned(&labels),
        prompt_tokens_local_compute: metrics.prompt_tokens_by_source.get_or_create_owned(
            &prompt_token_source_labels(model_name, engine, PROMPT_TOKEN_SOURCE_LOCAL_COMPUTE),
        ),
        prompt_tokens_local_cache_hit: metrics.prompt_tokens_by_source.get_or_create_owned(
            &prompt_token_source_labels(model_name, engine, PROMPT_TOKEN_SOURCE_LOCAL_CACHE_HIT),
        ),
        prompt_tokens_external_kv_transfer: metrics.prompt_tokens_by_source.get_or_create_owned(
            &prompt_token_source_labels(
                model_name,
                engine,
                PROMPT_TOKEN_SOURCE_EXTERNAL_KV_TRANSFER,
            ),
        ),
        prompt_tokens_cached: metrics.prompt_tokens_cached.get_or_create_owned(&labels),
        generation_tokens: metrics.generation_tokens.get_or_create_owned(&labels),
        request_success: metrics.request_success.clone(),
        request_prompt_tokens: metrics.request_prompt_tokens.get_or_create_owned(&labels),
        request_generation_tokens: metrics.request_generation_tokens.get_or_create_owned(&labels),
        request_max_num_generation_tokens: metrics
            .request_max_num_generation_tokens
            .get_or_create_owned(&labels),
        request_params_max_tokens: metrics.request_params_max_tokens.get_or_create_owned(&labels),
        request_params_n: metrics.request_params_n.get_or_create_owned(&labels),
        request_prefill_kv_computed_tokens: metrics
            .request_prefill_kv_computed_tokens
            .get_or_create_owned(&labels),
        time_to_first_token_seconds: metrics
            .time_to_first_token_seconds
            .get_or_create_owned(&labels),
        inter_token_latency_seconds: metrics
            .inter_token_latency_seconds
            .get_or_create_owned(&labels),
        e2e_request_latency_seconds: metrics
            .e2e_request_latency_seconds
            .get_or_create_owned(&labels),
        request_queue_time_seconds: metrics.request_queue_time_seconds.get_or_create_owned(&labels),
        request_prefill_time_seconds: metrics
            .request_prefill_time_seconds
            .get_or_create_owned(&labels),
        request_decode_time_seconds: metrics
            .request_decode_time_seconds
            .get_or_create_owned(&labels),
        request_inference_time_seconds: metrics
            .request_inference_time_seconds
            .get_or_create_owned(&labels),
        request_time_per_output_token_seconds: metrics
            .request_time_per_output_token_seconds
            .get_or_create_owned(&labels),
        labels,
    }
}

fn prompt_token_source_labels(
    model_name: &str,
    engine: u32,
    source: &'static str,
) -> PromptTokenSourceLabels {
    PromptTokenSourceLabels {
        model_name: model_name.to_string(),
        engine,
        source,
    }
}

fn diff_or_zero(end: f64, start: f64) -> f64 {
    if end > 0.0 && start > 0.0 && end >= start {
        end - start
    } else {
        0.0
    }
}

/// Return the current wall-clock time in seconds since the Unix epoch.
///
/// This is used for frontend-side latency measurements such as TTFT and E2E,
/// matching the Python frontend's use of wall-clock request arrival/iteration
/// timestamps rather than engine-core's monotonic scheduler timestamps.
///
/// Original Python request timestamp source:
/// <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/v1/metrics/stats.py#L206-L216>
pub fn current_unix_timestamp_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is before unix epoch")
        .as_secs_f64()
}

#[cfg(test)]
mod tests {
    use vllm_engine_core_client::protocol::output::{EngineCoreEvent, EngineCoreEventType};
    use vllm_engine_core_client::protocol::stats::PrefillStats;

    use super::{RequestMetricsTracker, diff_or_zero};

    #[test]
    fn tracker_updates_timing_state_across_prefill_decode_and_finish() {
        let mut tracker =
            RequestMetricsTracker::new("model".to_string(), 2, 100.0, 64, Some(128), 1);

        tracker.observe_output(
            10.0,
            100.2,
            &vllm_engine_core_client::protocol::output::EngineCoreOutput {
                request_id: "req-1".to_string(),
                new_token_ids: vec![1],
                finish_reason: None,
                events: Some(vec![
                    EngineCoreEvent {
                        r#type: EngineCoreEventType::Queued,
                        timestamp: 8.0,
                    },
                    EngineCoreEvent {
                        r#type: EngineCoreEventType::Scheduled,
                        timestamp: 9.0,
                    },
                ]),
                prefill_stats: Some(PrefillStats {
                    num_prompt_tokens: 64,
                    num_computed_tokens: 60,
                    num_cached_tokens: 4,
                    num_local_cached_tokens: 4,
                    num_external_cached_tokens: 0,
                    ..Default::default()
                }),
                ..Default::default()
            },
        );
        tracker.observe_output(
            11.5,
            100.4,
            &vllm_engine_core_client::protocol::output::EngineCoreOutput {
                request_id: "req-1".to_string(),
                new_token_ids: vec![2, 3],
                finish_reason: None,
                events: Some(vec![EngineCoreEvent {
                    r#type: EngineCoreEventType::Preempted,
                    timestamp: 10.5,
                }]),
                ..Default::default()
            },
        );

        assert!(!tracker.is_prefilling);
        assert_eq!(tracker.handles.labels.engine, 2);
        assert_eq!(tracker.num_generation_tokens, 3);
        assert_eq!(tracker.queued_ts, 8.0);
        assert_eq!(tracker.scheduled_ts, 9.0);
        assert_eq!(tracker.first_token_ts, 10.0);
        assert_eq!(tracker.last_token_ts, 11.5);
        assert!((tracker.first_token_latency - 0.2).abs() < 1e-9);
        assert_eq!(
            diff_or_zero(tracker.last_token_ts, tracker.first_token_ts),
            1.5
        );
    }
}
