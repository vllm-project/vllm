use std::fmt;
use std::sync::LazyLock;
use std::sync::atomic::AtomicU64;

use prometheus_client::encoding::text::encode;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::registry::Registry;
use prometheus_client_derive_encode::EncodeLabelSet;

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct EngineLabels {
    pub model_name: String,
    pub engine: u32,
}

/// Shared Prometheus registry for frontend metrics.
///
/// This currently owns a minimal subset of scheduler gauges. More metric families can be
/// registered here over time as the Rust frontend closes the remaining observability gap.
pub struct Metrics {
    registry: Registry,

    pub scheduler_running: Family<EngineLabels, Gauge<u64, AtomicU64>>,
    pub scheduler_waiting: Family<EngineLabels, Gauge<u64, AtomicU64>>,
    pub kv_cache_usage: Family<EngineLabels, Gauge<f64, AtomicU64>>,
}

impl Metrics {
    /// Construct a new metrics registry.
    pub fn new() -> Self {
        let mut registry = Registry::default();

        let scheduler_running = Family::default();
        registry.register(
            "vllm:num_requests_running",
            "Number of requests in model execution batches",
            scheduler_running.clone(),
        );

        let scheduler_waiting = Family::default();
        registry.register(
            "vllm:num_requests_waiting",
            "Number of requests waiting to be processed",
            scheduler_waiting.clone(),
        );

        let kv_cache_usage = Family::default();
        registry.register(
            "vllm:kv_cache_usage_perc",
            "KV-cache usage. 1 means 100 percent usage",
            kv_cache_usage.clone(),
        );

        Self {
            registry,
            scheduler_running,
            scheduler_waiting,
            kv_cache_usage,
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
