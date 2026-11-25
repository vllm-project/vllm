// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Per-engine performance / MFU counters.

use std::collections::BTreeMap;

use prometheus_client::metrics::family::Family;
use prometheus_client::registry::Registry;
use serde::{Deserialize, Serialize};

use crate::{EngineLabels, U64Counter};

/// Stats for debugging the metrics calculation.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/v1/metrics/perf.py#L46-L55>
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DebugPerfStats {
    /// Time spent calculating these stats.
    pub calc_duration: f64,
    /// Number of prefill requests included in the sampled batch.
    pub num_prefill_requests: u64,
    /// Number of decode requests included in the sampled batch.
    pub num_decode_requests: u64,
    /// Optional execution-context breakdown used for debugging.
    pub context_breakdown: Option<BTreeMap<String, u64>>,
    /// Optional per-component FLOPs breakdown.
    pub num_flops_per_gpu_breakdown: Option<BTreeMap<String, u64>>,
    /// Optional per-component memory-read breakdown.
    pub num_read_bytes_per_gpu_breakdown: Option<BTreeMap<String, u64>>,
    /// Optional per-component memory-write breakdown.
    pub num_write_bytes_per_gpu_breakdown: Option<BTreeMap<String, u64>>,
}

/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/v1/metrics/perf.py#L58-L63>
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PerfStats {
    /// Estimated floating point operations per GPU.
    pub num_flops_per_gpu: u64,
    /// Estimated bytes read from memory per GPU.
    pub num_read_bytes_per_gpu: u64,
    /// Estimated bytes written to memory per GPU.
    pub num_write_bytes_per_gpu: u64,
    /// Optional debug-only perf derivation details.
    pub debug_stats: Option<DebugPerfStats>,
}

/// Per-engine performance / MFU counters.
pub struct PerfMetrics {
    pub estimated_flops_per_gpu: Family<EngineLabels, U64Counter>,
    pub estimated_read_bytes_per_gpu: Family<EngineLabels, U64Counter>,
    pub estimated_write_bytes_per_gpu: Family<EngineLabels, U64Counter>,
}

impl PerfMetrics {
    pub(super) fn register(registry: &mut Registry) -> Self {
        let estimated_flops_per_gpu = Family::default();
        registry.register(
            "vllm:estimated_flops_per_gpu",
            "Estimated number of floating point operations per GPU (for Model Flops Utilization calculations).",
            estimated_flops_per_gpu.clone(),
        );

        let estimated_read_bytes_per_gpu = Family::default();
        registry.register(
            "vllm:estimated_read_bytes_per_gpu",
            "Estimated number of bytes read from memory per GPU (for Model Flops Utilization calculations).",
            estimated_read_bytes_per_gpu.clone(),
        );

        let estimated_write_bytes_per_gpu = Family::default();
        registry.register(
            "vllm:estimated_write_bytes_per_gpu",
            "Estimated number of bytes written to memory per GPU (for Model Flops Utilization calculations).",
            estimated_write_bytes_per_gpu.clone(),
        );

        Self {
            estimated_flops_per_gpu,
            estimated_read_bytes_per_gpu,
            estimated_write_bytes_per_gpu,
        }
    }

    pub(super) fn resolve(&self, labels: &EngineLabels) -> PerfHandles {
        PerfHandles {
            estimated_flops_per_gpu: self.estimated_flops_per_gpu.get_or_create_owned(labels),
            estimated_read_bytes_per_gpu: self
                .estimated_read_bytes_per_gpu
                .get_or_create_owned(labels),
            estimated_write_bytes_per_gpu: self
                .estimated_write_bytes_per_gpu
                .get_or_create_owned(labels),
        }
    }
}

pub struct PerfHandles {
    estimated_flops_per_gpu: U64Counter,
    estimated_read_bytes_per_gpu: U64Counter,
    estimated_write_bytes_per_gpu: U64Counter,
}

impl PerfHandles {
    pub fn record(&self, stats: &PerfStats) {
        if stats.num_flops_per_gpu != 0
            || stats.num_read_bytes_per_gpu != 0
            || stats.num_write_bytes_per_gpu != 0
        {
            self.estimated_flops_per_gpu.inc_by(stats.num_flops_per_gpu);
            self.estimated_read_bytes_per_gpu.inc_by(stats.num_read_bytes_per_gpu);
            self.estimated_write_bytes_per_gpu.inc_by(stats.num_write_bytes_per_gpu);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{EngineLabels, Metrics};

    #[test]
    fn perf_counters_render_with_a_single_total_suffix() {
        let metrics = Metrics::new();
        let labels = EngineLabels {
            model_name: "model".to_string(),
            engine: 0,
        };

        metrics.scheduler.perf.estimated_flops_per_gpu.get_or_create(&labels).inc();
        metrics.scheduler.perf.estimated_read_bytes_per_gpu.get_or_create(&labels).inc();
        metrics
            .scheduler
            .perf
            .estimated_write_bytes_per_gpu
            .get_or_create(&labels)
            .inc();

        let rendered = metrics.render().unwrap();
        assert!(
            rendered.contains(
                "vllm:estimated_flops_per_gpu_total{model_name=\"model\",engine=\"0\"} 1"
            )
        );
        assert!(rendered.contains(
            "vllm:estimated_read_bytes_per_gpu_total{model_name=\"model\",engine=\"0\"} 1"
        ));
        assert!(rendered.contains(
            "vllm:estimated_write_bytes_per_gpu_total{model_name=\"model\",engine=\"0\"} 1"
        ));
        assert!(!rendered.contains("vllm:estimated_flops_per_gpu_total_total"));
        assert!(!rendered.contains("vllm:estimated_read_bytes_per_gpu_total_total"));
        assert!(!rendered.contains("vllm:estimated_write_bytes_per_gpu_total_total"));
    }
}
