// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeMap;

use serde::Deserialize;

/// Wire / config descriptor version supported by this frontend.
pub(crate) const DESCRIPTOR_VERSION_V1: u32 = 1;

/// Reserved stats key carrying a lazy metrics descriptor (stripped before observe).
pub(crate) const METRICS_DESCRIPTOR_KEY: &str = "_metrics_descriptor";

/// How to map one payload field into a Prometheus observation.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SampleKind {
    IncByU64,
    IncBySumU64,
    /// Increment a float counter (e.g. Offloading transfer time seconds).
    IncByF64,
    SetU64,
    SetF64,
    ObserveEachF64,
    ObserveEachU64AsF64,
}

/// Prometheus metric type declared in the descriptor.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum MetricType {
    Counter,
    Gauge,
    Histogram,
}

/// One metric family definition for a connector.
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct MetricDef {
    pub name: String,
    #[serde(rename = "type")]
    pub metric_type: MetricType,
    #[serde(default)]
    pub documentation: String,
    #[serde(default)]
    pub buckets: Option<Vec<f64>>,
    pub samples_path: String,
    pub sample_kind: SampleKind,
    /// Multiply histogram observations (e.g. `1e-6` for µs → seconds).
    #[serde(default = "default_scale")]
    pub scale: f64,
    /// Extra labels always applied with engine labels (e.g. `outcome=local`).
    #[serde(default)]
    pub const_labels: BTreeMap<String, String>,
}

fn default_scale() -> f64 {
    1.0
}

/// Descriptor v1 document for one connector class name.
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct MetricsDescriptorV1 {
    pub descriptor_version: u32,
    pub connector_id: String,
    pub metrics: Vec<MetricDef>,
}

impl MetricsDescriptorV1 {
    pub(crate) fn validate(&self) -> Result<(), String> {
        if self.descriptor_version != DESCRIPTOR_VERSION_V1 {
            return Err(format!(
                "unsupported descriptor_version {} (want {DESCRIPTOR_VERSION_V1})",
                self.descriptor_version
            ));
        }
        if self.connector_id.is_empty() {
            return Err("connector_id must be non-empty".to_string());
        }
        for metric in &self.metrics {
            if metric.name.is_empty() || metric.samples_path.is_empty() {
                return Err("metric name and samples_path must be non-empty".to_string());
            }
            if metric.metric_type == MetricType::Histogram && metric.buckets.is_none() {
                return Err(format!(
                    "histogram metric '{}' requires buckets",
                    metric.name
                ));
            }
            match (&metric.metric_type, &metric.sample_kind) {
                (
                    MetricType::Counter,
                    SampleKind::IncByU64 | SampleKind::IncBySumU64 | SampleKind::IncByF64,
                ) => {}
                (MetricType::Gauge, SampleKind::SetU64 | SampleKind::SetF64) => {}
                (
                    MetricType::Histogram,
                    SampleKind::ObserveEachF64 | SampleKind::ObserveEachU64AsF64,
                ) => {}
                (ty, kind) => {
                    return Err(format!(
                        "metric '{}' type {ty:?} incompatible with sample_kind {kind:?}",
                        metric.name
                    ));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{MetricDef, MetricType, MetricsDescriptorV1, SampleKind};

    fn metric(
        metric_type: MetricType,
        sample_kind: SampleKind,
        buckets: Option<Vec<f64>>,
    ) -> MetricDef {
        MetricDef {
            name: "example_metric".to_string(),
            metric_type,
            documentation: "example".to_string(),
            buckets,
            samples_path: "value".to_string(),
            sample_kind,
            scale: 1.0,
            const_labels: Default::default(),
        }
    }

    fn descriptor(version: u32, metrics: Vec<MetricDef>) -> MetricsDescriptorV1 {
        MetricsDescriptorV1 {
            descriptor_version: version,
            connector_id: "ExampleConnector".to_string(),
            metrics,
        }
    }

    #[test]
    fn validate_rejects_unknown_descriptor_version() {
        let err = descriptor(
            99,
            vec![metric(MetricType::Counter, SampleKind::IncByU64, None)],
        )
        .validate()
        .expect_err("bad version");
        assert!(err.contains("unsupported descriptor_version 99"), "{err}");
    }

    #[test]
    fn validate_rejects_histogram_without_buckets() {
        let err = descriptor(
            1,
            vec![metric(
                MetricType::Histogram,
                SampleKind::ObserveEachF64,
                None,
            )],
        )
        .validate()
        .expect_err("missing buckets");
        assert!(err.contains("requires buckets"), "{err}");
    }

    #[test]
    fn validate_rejects_type_sample_kind_mismatch() {
        let err = descriptor(
            1,
            vec![metric(MetricType::Counter, SampleKind::SetU64, None)],
        )
        .validate()
        .expect_err("incompatible sample kind");
        assert!(err.contains("incompatible with sample_kind"), "{err}");
    }
}
