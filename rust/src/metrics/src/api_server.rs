use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::histogram::Histogram;
use prometheus_client::registry::Registry;

use crate::U64Counter;

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

pub(crate) type HttpRequestCounterFamily = Family<HttpRequestLabels, U64Counter>;
pub(crate) type HttpHandlerHistogramFamily =
    Family<HttpHandlerLabels, Histogram, fn() -> Histogram>;

/// API-server Prometheus families exported from the HTTP middleware layer.
pub struct ApiServerMetrics {
    pub http_requests: HttpRequestCounterFamily,
    pub http_request_duration_seconds: HttpHandlerHistogramFamily,
    pub http_request_duration_highr_seconds: Histogram,
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

        Self {
            http_requests,
            http_request_duration_seconds,
            http_request_duration_highr_seconds,
        }
    }
}
