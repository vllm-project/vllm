use std::fmt;
use std::sync::LazyLock;

use prometheus_client::encoding::text::encode;
use prometheus_client::registry::Registry;

pub mod stats;

/// Shared Prometheus registry for frontend metrics.
///
/// This starts intentionally empty. Callers can use the global [`METRICS`] instance and
/// progressively add metric families and recording logic over time.
pub struct Metrics {
    registry: Registry,
}

impl Metrics {
    /// Construct a new metrics registry.
    pub fn new() -> Self {
        Self {
            registry: Registry::default(),
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
