// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Internal, non-Prometheus accumulators for the periodic text logs.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

/// CUDA graph sample key used for periodic text-log aggregation.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct CudagraphLogKey {
    pub num_unpadded_tokens: u64,
    pub num_padded_tokens: u64,
    pub num_paddings: u64,
    pub runtime_mode: String,
}

/// Raw scheduler stats accumulated for one periodic text-log interval.
#[derive(Default)]
pub struct SchedulerLogStatsInterval {
    pub spec_num_drafts: u64,
    pub spec_accepted_tokens_per_pos: Vec<u64>,
    pub cudagraph_counts: BTreeMap<CudagraphLogKey, u64>,
}

impl SchedulerLogStatsInterval {
    /// Merge another drained interval into this one.
    pub fn merge(&mut self, other: Self) {
        self.spec_num_drafts += other.spec_num_drafts;

        if self.spec_accepted_tokens_per_pos.len() < other.spec_accepted_tokens_per_pos.len() {
            self.spec_accepted_tokens_per_pos
                .resize(other.spec_accepted_tokens_per_pos.len(), 0);
        }
        for (position, accepted_tokens) in
            other.spec_accepted_tokens_per_pos.into_iter().enumerate()
        {
            self.spec_accepted_tokens_per_pos[position] += accepted_tokens;
        }

        for (key, count) in other.cudagraph_counts {
            *self.cudagraph_counts.entry(key).or_default() += count;
        }
    }
}

/// Internal, non-Prometheus accumulator for periodic text logs that need raw
/// scheduler DTOs.
#[derive(Clone, Default)]
pub struct SchedulerLogStatsAccumulator {
    inner: Arc<Mutex<SchedulerLogStatsInterval>>,
}

impl SchedulerLogStatsAccumulator {
    /// Observe spec-decoding fields needed for per-position text-log rates.
    pub fn observe_spec_decode(&self, num_drafts: u64, accepted_tokens_per_pos: &[u64]) {
        let mut inner = self.inner.lock().expect("scheduler log stats accumulator poisoned");
        inner.spec_num_drafts += num_drafts;

        if inner.spec_accepted_tokens_per_pos.len() < accepted_tokens_per_pos.len() {
            inner.spec_accepted_tokens_per_pos.resize(accepted_tokens_per_pos.len(), 0);
        }
        for (position, accepted_tokens) in accepted_tokens_per_pos.iter().copied().enumerate() {
            inner.spec_accepted_tokens_per_pos[position] += accepted_tokens;
        }
    }

    /// Observe one CUDA graph runtime sample for the interval table.
    pub fn observe_cudagraph(
        &self,
        num_unpadded_tokens: u64,
        num_padded_tokens: u64,
        num_paddings: u64,
        runtime_mode: &str,
    ) {
        let mut inner = self.inner.lock().expect("scheduler log stats accumulator poisoned");
        let key = CudagraphLogKey {
            num_unpadded_tokens,
            num_padded_tokens,
            num_paddings,
            runtime_mode: runtime_mode.to_string(),
        };
        *inner.cudagraph_counts.entry(key).or_default() += 1;
    }

    /// Drain and reset the current text-log interval.
    pub fn drain(&self) -> SchedulerLogStatsInterval {
        let mut inner = self.inner.lock().expect("scheduler log stats accumulator poisoned");
        std::mem::take(&mut *inner)
    }
}

/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/compilation/cuda_graph.py#L28-L33>
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CudagraphStats {
    /// Number of real tokens in the captured batch before padding.
    pub num_unpadded_tokens: u64,
    /// Number of padded tokens in the captured batch.
    pub num_padded_tokens: u64,
    /// Number of padding positions added for capture/runtime shape alignment.
    pub num_paddings: u64,
    /// Runtime mode string associated with this CUDA graph sample.
    pub runtime_mode: String,
}

#[cfg(test)]
mod tests {
    use super::{CudagraphLogKey, SchedulerLogStatsAccumulator};

    #[test]
    fn log_stats_accumulator_drains_interval_data() {
        let accumulator = SchedulerLogStatsAccumulator::default();

        accumulator.observe_spec_decode(2, &[1, 2]);
        accumulator.observe_spec_decode(3, &[3, 4, 5]);
        accumulator.observe_cudagraph(8, 16, 8, "FULL");
        accumulator.observe_cudagraph(8, 16, 8, "FULL");

        let interval = accumulator.drain();

        assert_eq!(interval.spec_num_drafts, 5);
        assert_eq!(interval.spec_accepted_tokens_per_pos, vec![4, 6, 5]);
        assert_eq!(
            interval
                .cudagraph_counts
                .get(&CudagraphLogKey {
                    num_unpadded_tokens: 8,
                    num_padded_tokens: 16,
                    num_paddings: 8,
                    runtime_mode: "FULL".to_string(),
                })
                .copied(),
            Some(2)
        );
        assert_eq!(accumulator.drain().spec_num_drafts, 0);
    }
}
