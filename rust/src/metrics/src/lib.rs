use std::fmt;
use std::sync::LazyLock;
use std::sync::atomic::AtomicU64;

use prometheus_client::encoding::text::encode;
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::metrics::histogram::Histogram;
use prometheus_client::registry::Registry;
use prometheus_client_derive_encode::EncodeLabelSet;

mod request;
mod scheduler;

pub use request::RequestMetrics;
pub use scheduler::SchedulerMetrics;

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct EngineLabels {
    pub model_name: String,
    pub engine: u32,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct EnginePositionLabels {
    pub model_name: String,
    pub engine: u32,
    pub position: u32,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct FinishedReasonLabels {
    pub model_name: String,
    pub engine: u32,
    pub finished_reason: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct PromptTokenSourceLabels {
    pub model_name: String,
    pub engine: u32,
    pub source: &'static str,
}

pub(crate) type U64Counter = Counter<u64, AtomicU64>;
pub(crate) type U64Gauge = Gauge<u64, AtomicU64>;
pub(crate) type F64Gauge = Gauge<f64, AtomicU64>;
pub(crate) type HistogramFamily = Family<EngineLabels, Histogram, fn() -> Histogram>;
pub(crate) type FinishedReasonCounterFamily = Family<FinishedReasonLabels, U64Counter>;
pub(crate) type PromptTokenSourceCounterFamily = Family<PromptTokenSourceLabels, U64Counter>;

/// Shared Prometheus registry for frontend metrics.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/v1/metrics/loggers.py#L389-L1004>
pub struct Metrics {
    registry: Registry,
    pub scheduler: SchedulerMetrics,
    pub request: RequestMetrics,
}

impl Metrics {
    /// Construct a new metrics registry.
    pub fn new() -> Self {
        let mut registry = Registry::default();
        let scheduler = SchedulerMetrics::register(&mut registry);
        let request = RequestMetrics::register(&mut registry);

        Self {
            registry,
            scheduler,
            request,
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
