// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Scheduler state gauges and the preemption counter.

use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::family::Family;
use prometheus_client::registry::Registry;
use serde::{Deserialize, Serialize};

use crate::{EngineLabels, F64Gauge, U64Counter, U64Gauge};

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct WaitingReasonLabels {
    pub model_name: String,
    pub engine: u32,
    pub reason: &'static str,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct KvFetchStageLabels {
    pub model_name: String,
    pub engine: u32,
    pub stage: &'static str,
}

const WAITING_REASON_CAPACITY: &str = "capacity";
const WAITING_REASON_DEFERRED: &str = "deferred";

/// Waiting requests with an async KV load, per load stage. Python sends a
/// `dict[str, int]` keyed by stage, so absent stages default to zero.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct KvFetchReqsByStage {
    /// Needs an async KV load that has not started.
    pub waiting_to_start: u64,
    /// Load started and not yet reported finished.
    pub in_progress: u64,
    /// Load finished, request not running yet.
    pub completed_waiting: u64,
}

impl KvFetchReqsByStage {
    /// Label values, in the same order as [`Self::counts`].
    pub const STAGES: [&'static str; 3] = ["waiting_to_start", "in_progress", "completed_waiting"];

    pub fn counts(&self) -> [u64; 3] {
        [
            self.waiting_to_start,
            self.in_progress,
            self.completed_waiting,
        ]
    }
}

/// Scheduler state gauges and the preemption counter.
pub struct SchedulerStateMetrics {
    pub running: Family<EngineLabels, U64Gauge>,
    pub waiting: Family<EngineLabels, U64Gauge>,
    pub waiting_by_reason: Family<WaitingReasonLabels, U64Gauge>,
    pub kv_fetch_by_stage: Family<KvFetchStageLabels, U64Gauge>,
    pub kv_cache_usage: Family<EngineLabels, F64Gauge>,
    pub num_preemptions: Family<EngineLabels, U64Counter>,
}

impl SchedulerStateMetrics {
    pub(super) fn register(registry: &mut Registry) -> Self {
        let running = Family::default();
        registry.register(
            "vllm:num_requests_running",
            "Number of requests in model execution batches",
            running.clone(),
        );

        let waiting = Family::default();
        registry.register(
            "vllm:num_requests_waiting",
            "Number of requests waiting to be processed",
            waiting.clone(),
        );

        let waiting_by_reason = Family::default();
        registry.register(
            "vllm:num_requests_waiting_by_reason",
            "Number of waiting requests by reason. \
             Reason labels: 'capacity' = waiting for scheduling capacity; \
             'deferred' = deferred by transient constraints (LoRA budget, KV transfer, \
             blocked status). Sum of all reasons equals vllm:num_requests_waiting.",
            waiting_by_reason.clone(),
        );

        let kv_fetch_by_stage = Family::default();
        registry.register(
            "vllm:num_requests_kv_fetch_by_stage",
            "Number of waiting requests by async KV load stage. \
             Stage labels: 'waiting_to_start' = needs an async KV load that has not started; \
             'in_progress' = load started and not yet reported finished; \
             'completed_waiting' = load finished, request not running yet.",
            kv_fetch_by_stage.clone(),
        );

        let kv_cache_usage = Family::default();
        registry.register(
            "vllm:kv_cache_usage_perc",
            "KV-cache usage. 1 means 100 percent usage",
            kv_cache_usage.clone(),
        );

        // Counted by the scheduler when a request is preempted, so a request
        // preempted and then aborted counts too; the PREEMPTED request event
        // reaches the frontend only with the request's next output.
        let num_preemptions = Family::default();
        registry.register(
            "vllm:num_preemptions",
            "Cumulative number of preemption events.",
            num_preemptions.clone(),
        );

        Self {
            running,
            waiting,
            waiting_by_reason,
            kv_fetch_by_stage,
            kv_cache_usage,
            num_preemptions,
        }
    }

    pub(super) fn resolve(&self, labels: &EngineLabels) -> SchedulerStateHandles {
        let reason = |reason| WaitingReasonLabels {
            model_name: labels.model_name.clone(),
            engine: labels.engine,
            reason,
        };
        SchedulerStateHandles {
            running: self.running.get_or_create_owned(labels),
            waiting: self.waiting.get_or_create_owned(labels),
            waiting_capacity: self
                .waiting_by_reason
                .get_or_create_owned(&reason(WAITING_REASON_CAPACITY)),
            waiting_deferred: self
                .waiting_by_reason
                .get_or_create_owned(&reason(WAITING_REASON_DEFERRED)),
            kv_fetch_by_stage: KvFetchReqsByStage::STAGES.map(|stage| {
                self.kv_fetch_by_stage.get_or_create_owned(&KvFetchStageLabels {
                    model_name: labels.model_name.clone(),
                    engine: labels.engine,
                    stage,
                })
            }),
            kv_cache_usage: self.kv_cache_usage.get_or_create_owned(labels),
            num_preemptions: self.num_preemptions.get_or_create_owned(labels),
        }
    }
}

pub struct SchedulerStateHandles {
    running: U64Gauge,
    waiting: U64Gauge,
    waiting_capacity: U64Gauge,
    waiting_deferred: U64Gauge,
    /// One handle per `KvFetchReqsByStage::STAGES` entry.
    kv_fetch_by_stage: [U64Gauge; 3],
    kv_cache_usage: F64Gauge,
    num_preemptions: U64Counter,
}

impl SchedulerStateHandles {
    pub fn record(
        &self,
        num_running_reqs: u64,
        num_waiting_reqs: u64,
        num_skipped_waiting_reqs: u64,
        num_kv_fetch_reqs_by_stage: &KvFetchReqsByStage,
        kv_cache_usage: f64,
        num_preempted_reqs: u64,
    ) {
        self.running.set(num_running_reqs);
        self.waiting.set(num_waiting_reqs + num_skipped_waiting_reqs);
        self.waiting_capacity.set(num_waiting_reqs);
        self.waiting_deferred.set(num_skipped_waiting_reqs);
        for (handle, count) in
            self.kv_fetch_by_stage.iter().zip(num_kv_fetch_reqs_by_stage.counts())
        {
            handle.set(count);
        }
        self.kv_cache_usage.set(kv_cache_usage);
        self.num_preemptions.inc_by(num_preempted_reqs);
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use itertools::Itertools as _;

    use super::KvFetchReqsByStage;
    use crate::Metrics;

    /// Exports every engine's KV fetch stage gauges from startup and sets them
    /// from each `SchedulerStats` update.
    #[test]
    fn kv_fetch_reqs_by_stage_are_recorded_into_stage_gauges() {
        let metrics = Metrics::new();
        let handles = metrics.scheduler.resolve_handles("model", 0);
        // Series of one family render in hash order, so sort them.
        let kv_fetch_series = || {
            metrics
                .render()
                .unwrap()
                .lines()
                .filter(|line| line.starts_with("vllm:num_requests_kv_fetch_by_stage{"))
                .sorted()
                .join("\n")
        };

        expect![[r#"
            vllm:num_requests_kv_fetch_by_stage{model_name="model",engine="0",stage="completed_waiting"} 0
            vllm:num_requests_kv_fetch_by_stage{model_name="model",engine="0",stage="in_progress"} 0
            vllm:num_requests_kv_fetch_by_stage{model_name="model",engine="0",stage="waiting_to_start"} 0"#]]
        .assert_eq(&kv_fetch_series());

        handles.state.record(
            0,
            0,
            0,
            &KvFetchReqsByStage {
                waiting_to_start: 3,
                in_progress: 1,
                completed_waiting: 2,
            },
            0.0,
            0,
        );

        expect![[r#"
            vllm:num_requests_kv_fetch_by_stage{model_name="model",engine="0",stage="completed_waiting"} 2
            vllm:num_requests_kv_fetch_by_stage{model_name="model",engine="0",stage="in_progress"} 1
            vllm:num_requests_kv_fetch_by_stage{model_name="model",engine="0",stage="waiting_to_start"} 3"#]]
        .assert_eq(&kv_fetch_series());
    }
}
