use std::fmt;
use std::sync::LazyLock;
use std::sync::atomic::AtomicU64;

use prometheus_client::encoding::text::encode;
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::metrics::histogram::Histogram;
use prometheus_client::registry::Registry;

mod api_server;
mod request;
mod scheduler;

pub use api_server::*;
pub use request::*;
pub use scheduler::*;

// Note: `prometheus-client` appends the `_total` suffix automatically when encoding counters, so
// all counter family registration names in this crate must use the base metric name without a
// trailing `_total`.
pub type U64Counter = Counter<u64, AtomicU64>;
pub type U64Gauge = Gauge<u64, AtomicU64>;
pub type F64Gauge = Gauge<f64, AtomicU64>;
pub(crate) type HistogramFamily = Family<EngineLabels, Histogram, fn() -> Histogram>;

/// Shared Prometheus registry for frontend metrics.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/v1/metrics/loggers.py#L389-L1004>
pub struct Metrics {
    registry: Registry,
    pub scheduler: SchedulerMetrics,
    pub request: RequestMetrics,
    pub api_server: ApiServerMetrics,
}

impl Metrics {
    /// Construct a new metrics registry.
    pub fn new() -> Self {
        let mut registry = Registry::default();
        let scheduler = SchedulerMetrics::register(&mut registry);
        let request = RequestMetrics::register(&mut registry);
        let api_server = ApiServerMetrics::register(&mut registry);

        Self {
            registry,
            scheduler,
            request,
            api_server,
        }
    }

    /// Render the current metrics registry into Prometheus/OpenMetrics text format.
    pub fn render(&self) -> Result<String, fmt::Error> {
        let mut output = String::new();
        encode(&mut output, &self.registry)?;
        Ok(output)
    }

    /// Return the registry owned by this metrics object.
    pub fn registry(&self) -> &Registry {
        &self.registry
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Process-global metrics registry shared by the frontend crates.
pub static METRICS: LazyLock<Metrics> = LazyLock::new(Metrics::new);
