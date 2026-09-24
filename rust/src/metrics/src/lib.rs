// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::fmt;
use std::sync::LazyLock;
use std::sync::Mutex;
use std::sync::atomic::AtomicU64;

use prometheus_client::encoding::text::{encode_eof, encode_registry};
use prometheus_client::metrics::counter::Counter;
pub use prometheus_client::metrics::family::{Family, MetricConstructor};
use prometheus_client::metrics::gauge::Gauge;
pub use prometheus_client::metrics::histogram::Histogram;
use prometheus_client::registry::Registry;

mod api_server;
mod request;
mod scheduler;

pub use api_server::*;
pub use request::*;
pub use scheduler::*;

// Note: `prometheus-client` appends the `_total` suffix automatically when
// encoding counters, so all counter family registration names in this crate
// must use the base metric name without a trailing `_total`.
pub type U64Counter = Counter<u64, AtomicU64>;
pub type F64Counter = Counter<f64, AtomicU64>;
pub type U64Gauge = Gauge<u64, AtomicU64>;
pub type F64Gauge = Gauge<f64, AtomicU64>;
/// Histogram metric handle cloned out of a Prometheus family.
pub type HistogramMetric = Histogram;
pub(crate) type HistogramFamily = Family<EngineLabels, Histogram, fn() -> Histogram>;

/// Label pairs for descriptor-driven KV connector metrics (`model_name` /
/// `engine` plus optional descriptor `const_labels`).
pub type ConnectorMetricLabels = Vec<(String, String)>;

/// Build sorted connector metric labels for one engine plus optional extras.
pub fn connector_metric_labels(
    model_name: &str,
    engine: u32,
    const_labels: &[(String, String)],
) -> ConnectorMetricLabels {
    let mut labels = Vec::with_capacity(2 + const_labels.len());
    labels.push(("model_name".to_string(), model_name.to_string()));
    labels.push(("engine".to_string(), engine.to_string()));
    labels.extend(const_labels.iter().cloned());
    labels.sort_by(|a, b| a.0.cmp(&b.0));
    labels
}

/// Shared Prometheus registry for frontend metrics.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/v1/metrics/loggers.py#L389-L1004>
pub struct Metrics {
    registry: Registry,
    /// Families registered at runtime from third-party connector descriptors.
    dynamic_registry: Mutex<Registry>,
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
            dynamic_registry: Mutex::new(Registry::default()),
            scheduler,
            request,
            api_server,
        }
    }

    /// Render the current metrics registry into Prometheus/OpenMetrics text
    /// format.
    pub fn render(&self) -> Result<String, fmt::Error> {
        let mut output = String::new();
        encode_registry(&mut output, &self.registry)?;
        {
            let dynamic = self.dynamic_registry.lock().expect("metrics dynamic registry");
            encode_registry(&mut output, &dynamic)?;
        }
        encode_eof(&mut output)?;
        Ok(output)
    }

    /// Return the registry owned by this metrics object.
    pub fn registry(&self) -> &Registry {
        &self.registry
    }

    /// Mutate the dynamic (descriptor-driven) registry.
    ///
    /// Used by the KV connector metrics plugin for third-party connectors.
    /// Panics if the registry mutex is poisoned.
    pub fn with_dynamic_registry<R>(&self, f: impl FnOnce(&mut Registry) -> R) -> R {
        let mut registry = self.dynamic_registry.lock().expect("metrics dynamic registry");
        f(&mut registry)
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Process-global metrics registry shared by the frontend crates.
pub static METRICS: LazyLock<Metrics> = LazyLock::new(Metrics::new);
