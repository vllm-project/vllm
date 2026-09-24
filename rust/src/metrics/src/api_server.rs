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

fn weight_operation_duration_histogram() -> Histogram {
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
pub struct WeightOperationLabels {
    pub operation: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct WeightOperationResultLabels {
    pub operation: &'static str,
    pub status: &'static str,
}

pub(crate) type HttpRequestCounterFamily = Family<HttpRequestLabels, U64Counter>;
pub(crate) type HttpHandlerHistogramFamily =
    Family<HttpHandlerLabels, Histogram, fn() -> Histogram>;
pub(crate) type WeightOperationCounterFamily = Family<WeightOperationResultLabels, U64Counter>;
pub(crate) type WeightOperationHistogramFamily =
    Family<WeightOperationLabels, Histogram, fn() -> Histogram>;
pub(crate) type WeightOperationGaugeFamily = Family<WeightOperationLabels, U64Gauge>;

/// API-server Prometheus families exported from the HTTP middleware layer.
pub struct ApiServerMetrics {
    pub http_requests: HttpRequestCounterFamily,
    pub http_request_duration_seconds: HttpHandlerHistogramFamily,
    pub http_request_duration_highr_seconds: Histogram,
    pub weight_operations: WeightOperationCounterFamily,
    pub weight_operation_duration_seconds: WeightOperationHistogramFamily,
    pub weight_operations_in_flight: WeightOperationGaugeFamily,
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

        let weight_operations = WeightOperationCounterFamily::default();
        registry.register(
            "vllm:rl_weight_update_operations",
            "Logical frontend weight operations by outcome, one observation per \
             dispatched engine call. 'finish' covers only finish_weight_update(); the \
             weight-version handshake is counted separately as 'set_version'.",
            weight_operations.clone(),
        );

        let weight_operation_duration_seconds =
            Family::new_with_constructor(weight_operation_duration_histogram as fn() -> Histogram);
        registry.register(
            "vllm:rl_weight_update_operation_duration_seconds",
            "Duration of one logical frontend weight operation.",
            weight_operation_duration_seconds.clone(),
        );

        let weight_operations_in_flight = WeightOperationGaugeFamily::default();
        registry.register(
            "vllm:rl_weight_update_operations_in_flight",
            "Logical frontend weight operations currently awaited.",
            weight_operations_in_flight.clone(),
        );

        Self {
            http_requests,
            http_request_duration_seconds,
            http_request_duration_highr_seconds,
            weight_operations,
            weight_operation_duration_seconds,
            weight_operations_in_flight,
        }
    }

    /// Record one dispatched logical frontend weight operation.
    pub fn record_weight_operation(&self, operation: &'static str) -> WeightOperationRecorder {
        let labels = WeightOperationLabels { operation };
        let in_flight = self.weight_operations_in_flight.get_or_create_owned(&labels);
        in_flight.inc();
        WeightOperationRecorder {
            operation,
            started_at: Instant::now(),
            operations: self.weight_operations.clone(),
            duration: self.weight_operation_duration_seconds.get_or_create_owned(&labels),
            in_flight,
            status: "error",
        }
    }
}

/// Completes a weight-operation observation when it succeeds or is dropped.
pub struct WeightOperationRecorder {
    operation: &'static str,
    started_at: Instant,
    operations: WeightOperationCounterFamily,
    duration: Histogram,
    in_flight: U64Gauge,
    status: &'static str,
}

impl WeightOperationRecorder {
    /// Mark the observed operation as successful.
    pub fn success(mut self) {
        // `self` is dropped when this method returns, which records the outcome.
        self.status = "success";
    }
}

impl Drop for WeightOperationRecorder {
    fn drop(&mut self) {
        self.in_flight.dec();
        self.duration.observe(self.started_at.elapsed().as_secs_f64());
        self.operations
            .get_or_create(&WeightOperationResultLabels {
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

    #[test]
    fn weight_operation_recorder_tracks_success_and_error() {
        let mut registry = Registry::default();
        let metrics = ApiServerMetrics::register(&mut registry);

        metrics.record_weight_operation("update").success();
        drop(metrics.record_weight_operation("update"));

        let mut rendered = String::new();
        encode(&mut rendered, &registry).expect("encode metrics");
        assert!(rendered.contains(
            "vllm:rl_weight_update_operations_total{operation=\"update\",status=\"success\"} 1"
        ));
        assert!(rendered.contains(
            "vllm:rl_weight_update_operations_total{operation=\"update\",status=\"error\"} 1"
        ));
        assert!(
            rendered.contains("vllm:rl_weight_update_operations_in_flight{operation=\"update\"} 0")
        );
    }
}
