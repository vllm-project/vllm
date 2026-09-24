// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::time::Instant;

use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::histogram::Histogram;
use prometheus_client::registry::Registry;

use crate::{U64Counter, U64Gauge};

const HTTP_REQUEST_DURATION_BUCKETS: [f64; 3] = [0.1, 0.5, 1.0];
const HTTP_REQUEST_DURATION_HIGHR_BUCKETS: [f64; 21] = [
    0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
    7.5, 10.0, 30.0, 60.0,
];

fn http_request_duration_histogram() -> Histogram {
    Histogram::new(HTTP_REQUEST_DURATION_BUCKETS.iter().copied())
}

fn http_request_duration_highr_histogram() -> Histogram {
    Histogram::new(HTTP_REQUEST_DURATION_HIGHR_BUCKETS.iter().copied())
}

fn sleep_mode_operation_duration_histogram() -> Histogram {
    Histogram::new([0.01, 0.1, 1.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0])
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct HttpRequestLabels {
    pub method: String,
    pub status: &'static str,
    pub handler: String,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct HttpHandlerLabels {
    pub method: String,
    pub handler: String,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct SleepModeOperationLabels {
    pub operation: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct SleepModeOperationResultLabels {
    pub operation: &'static str,
    pub status: &'static str,
}

pub(crate) type HttpRequestCounterFamily = Family<HttpRequestLabels, U64Counter>;
pub(crate) type HttpHandlerHistogramFamily =
    Family<HttpHandlerLabels, Histogram, fn() -> Histogram>;
pub(crate) type SleepModeOperationCounterFamily =
    Family<SleepModeOperationResultLabels, U64Counter>;
pub(crate) type SleepModeOperationHistogramFamily =
    Family<SleepModeOperationLabels, Histogram, fn() -> Histogram>;
pub(crate) type SleepModeOperationGaugeFamily = Family<SleepModeOperationLabels, U64Gauge>;

/// API-server Prometheus families exported from the HTTP middleware layer.
pub struct ApiServerMetrics {
    pub http_requests: HttpRequestCounterFamily,
    pub http_request_duration_seconds: HttpHandlerHistogramFamily,
    pub http_request_duration_highr_seconds: Histogram,
    pub sleep_mode_operations: SleepModeOperationCounterFamily,
    pub sleep_mode_operation_duration_seconds: SleepModeOperationHistogramFamily,
    pub sleep_mode_operations_in_flight: SleepModeOperationGaugeFamily,
}

impl ApiServerMetrics {
    /// Register the API-server metric families into the shared registry.
    pub(crate) fn register(registry: &mut Registry) -> Self {
        let http_requests = HttpRequestCounterFamily::default();
        registry.register(
            "http_requests",
            "Total number of HTTP requests by method, status, and handler.",
            http_requests.clone(),
        );

        let http_request_duration_seconds = HttpHandlerHistogramFamily::new_with_constructor(
            http_request_duration_histogram as fn() -> Histogram,
        );
        registry.register(
            "http_request_duration_seconds",
            "Duration of HTTP requests in seconds grouped by method and handler.",
            http_request_duration_seconds.clone(),
        );

        let http_request_duration_highr_seconds = http_request_duration_highr_histogram();
        registry.register(
            "http_request_duration_highr_seconds",
            "High-resolution duration of HTTP requests in seconds.",
            http_request_duration_highr_seconds.clone(),
        );

        let sleep_mode_operations = SleepModeOperationCounterFamily::default();
        registry.register(
            "vllm:rl_sleep_mode_operations",
            "Dispatched sleep-mode operations by outcome.",
            sleep_mode_operations.clone(),
        );

        let sleep_mode_operation_duration_seconds = Family::new_with_constructor(
            sleep_mode_operation_duration_histogram as fn() -> Histogram,
        );
        registry.register(
            "vllm:rl_sleep_mode_operation_duration_seconds",
            "Duration of one sleep-mode operation.",
            sleep_mode_operation_duration_seconds.clone(),
        );

        let sleep_mode_operations_in_flight = SleepModeOperationGaugeFamily::default();
        registry.register(
            "vllm:rl_sleep_mode_operations_in_flight",
            "Sleep-mode operations currently awaited.",
            sleep_mode_operations_in_flight.clone(),
        );

        Self {
            http_requests,
            http_request_duration_seconds,
            http_request_duration_highr_seconds,
            sleep_mode_operations,
            sleep_mode_operation_duration_seconds,
            sleep_mode_operations_in_flight,
        }
    }

    /// Record one dispatched sleep-mode engine operation.
    pub fn record_sleep_mode_operation(
        &self,
        operation: &'static str,
    ) -> SleepModeOperationRecorder {
        let labels = SleepModeOperationLabels { operation };
        let in_flight = self.sleep_mode_operations_in_flight.get_or_create_owned(&labels);
        in_flight.inc();
        SleepModeOperationRecorder {
            operation,
            started_at: Instant::now(),
            operations: self.sleep_mode_operations.clone(),
            duration: self
                .sleep_mode_operation_duration_seconds
                .get_or_create_owned(&labels),
            in_flight,
            status: "error",
        }
    }
}

/// Records the outcome when the operation completes or its future is dropped.
pub struct SleepModeOperationRecorder {
    operation: &'static str,
    started_at: Instant,
    operations: SleepModeOperationCounterFamily,
    duration: Histogram,
    in_flight: U64Gauge,
    status: &'static str,
}

impl SleepModeOperationRecorder {
    pub fn success(mut self) {
        self.status = "success";
    }
}

impl Drop for SleepModeOperationRecorder {
    fn drop(&mut self) {
        self.in_flight.dec();
        self.duration.observe(self.started_at.elapsed().as_secs_f64());
        self.operations
            .get_or_create(&SleepModeOperationResultLabels {
                operation: self.operation,
                status: self.status,
            })
            .inc();
    }
}

#[cfg(test)]
mod tests {
    use prometheus_client::encoding::text::encode;
    use prometheus_client::registry::Registry;

    use super::ApiServerMetrics;

    fn rendered_metrics(registry: &Registry) -> String {
        let mut rendered = String::new();
        encode(&mut rendered, registry).expect("encode metrics");
        rendered
    }

    #[test]
    fn sleep_mode_recorder_tracks_outcomes_and_concurrency() {
        let mut registry = Registry::default();
        let metrics = ApiServerMetrics::register(&mut registry);
        for operation in ["sleep", "release_kv_cache_memory", "wake"] {
            let recorder = metrics.record_sleep_mode_operation(operation);
            let active = rendered_metrics(&registry);
            assert!(active.contains(&format!(
                "vllm:rl_sleep_mode_operations_in_flight{{operation=\"{operation}\"}} 1"
            )));

            recorder.success();
            let succeeded = rendered_metrics(&registry);
            assert!(succeeded.contains(&format!(
                "vllm:rl_sleep_mode_operations_in_flight{{operation=\"{operation}\"}} 0"
            )));
            assert!(succeeded.contains(&format!(
                "vllm:rl_sleep_mode_operation_duration_seconds_count{{operation=\"{operation}\"}} 1"
            )));
            assert!(succeeded.contains(&format!(
                "vllm:rl_sleep_mode_operations_total{{operation=\"{operation}\",status=\"success\"}} 1"
            )));

            let recorder = metrics.record_sleep_mode_operation(operation);
            drop(recorder);
            let rendered = rendered_metrics(&registry);
            assert!(rendered.contains(&format!(
                "vllm:rl_sleep_mode_operations_total{{operation=\"{operation}\",status=\"error\"}} 1"
            )));
            assert!(rendered.contains(&format!(
                "vllm:rl_sleep_mode_operations_in_flight{{operation=\"{operation}\"}} 0"
            )));
            assert!(rendered.contains(&format!(
                "vllm:rl_sleep_mode_operation_duration_seconds_count{{operation=\"{operation}\"}} 2"
            )));
        }
    }
}
