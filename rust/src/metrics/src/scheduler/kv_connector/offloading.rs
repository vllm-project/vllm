// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! `OffloadingConnector` telemetry: the `vllm:kv_offload_*` families.
//!
//! Python declares these families as metadata tables, the connector's own
//! (`get_connector_metric_definitions`) and each offloading spec's
//! (`build_metric_definitions`), and keys its stats payload by metric name.
//! [`DEFINITIONS`] mirrors the tables of the built-in specs, so a family
//! reviews against its Python definition row by row.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use prometheus_client::metrics::family::{Family, MetricConstructor};
use prometheus_client::registry::Registry;
use serde::{Deserialize, Serialize};

use crate::{EngineLabels, F64Counter, F64Gauge, Histogram};

/// OffloadingConnector telemetry.
///
/// The `types`/`data` envelope of [`SimpleCpuOffloadStats`], plus `info`.
/// The families depend on the configured offloading spec, so `data` stays
/// keyed by metric name and each name is resolved against [`DEFINITIONS`];
/// `types` is not read.
///
/// Original Python definition: `OffloadingConnectorStats` in
/// `vllm/distributed/kv_transfer/kv_connector/v1/offloading/metrics.py`.
///
/// [`SimpleCpuOffloadStats`]: super::SimpleCpuOffloadStats
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct OffloadingStats {
    /// Samples of one interval, keyed by metric name and then by the
    /// metric's label values (a Python tuple, empty when unlabeled).
    pub data: BTreeMap<String, BTreeMap<Vec<String>, OffloadingSample>>,
    /// `vllm:kv_offload_config_info` facts, one mapping of label name to
    /// value per series, with the values already stringified by Python. The
    /// scheduler sends them once; `None` otherwise.
    #[serde(default)]
    pub info: Option<Vec<BTreeMap<String, String>>>,
}

/// One OffloadingConnector sample: a counter increment or gauge value, or
/// the observations of a histogram.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OffloadingSample {
    Value(f64),
    Observations(Vec<f64>),
}

const LOAD_BYTES: &str = "vllm:kv_offload_load_bytes";
const LOAD_TIME: &str = "vllm:kv_offload_load_time";
const LOAD_SIZE: &str = "vllm:kv_offload_load_size";
const STORE_BYTES: &str = "vllm:kv_offload_store_bytes";
const STORE_TIME: &str = "vllm:kv_offload_store_time";
const STORE_SIZE: &str = "vllm:kv_offload_store_size";

const TRANSFER_SIZE_BUCKETS: &[f64] =
    &[1e6, 5e6, 10e6, 20e6, 40e6, 60e6, 80e6, 100e6, 150e6, 200e6];
const LOOKUP_SYNC_DELAY_BUCKETS: &[f64] = &[
    0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0,
];
const LOOKUP_ASYNC_DELAY_BUCKETS: &[f64] = &[
    0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0,
];
const DEFERRED_RELEASE_HOLD_BUCKETS: &[f64] = &[
    0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0,
];
const PROMOTION_LATENCY_BUCKETS: &[f64] = &[
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
];
const CPU_ALLOCATION_SIZE_BUCKETS: &[f64] = &[
    1.0, 4.0, 16.0, 64.0, 256.0, 1024.0, 4096.0, 16384.0, 65536.0, 262144.0,
];

/// Kind of one family, which fixes the shape of its samples.
#[derive(Clone, Copy, Debug)]
enum Kind {
    Counter,
    Gauge,
    Histogram(&'static [f64]),
}

/// One family, as Python's `OffloadingMetricMetadata` declares it.
#[derive(Debug)]
struct Definition {
    name: &'static str,
    help: &'static str,
    /// Label names after `model_name` and `engine`.
    labels: &'static [&'static str],
    kind: Kind,
}

/// The families of `get_connector_metric_definitions` and of the built-in
/// specs' `build_metric_definitions`.
const DEFINITIONS: &[Definition] = &[
    // `get_connector_metric_definitions`.
    Definition {
        name: LOAD_BYTES,
        help: "Total bytes loaded from offload storage to GPU",
        labels: &[],
        kind: Kind::Counter,
    },
    Definition {
        name: LOAD_TIME,
        help: "Total load time from offload storage to GPU, in seconds",
        labels: &[],
        kind: Kind::Counter,
    },
    Definition {
        name: LOAD_SIZE,
        help: "Histogram of KV offload load operation size, in bytes",
        labels: &[],
        kind: Kind::Histogram(TRANSFER_SIZE_BUCKETS),
    },
    Definition {
        name: STORE_BYTES,
        help: "Total bytes stored from GPU to offload storage",
        labels: &[],
        kind: Kind::Counter,
    },
    Definition {
        name: STORE_TIME,
        help: "Total store time from GPU to offload storage, in seconds",
        labels: &[],
        kind: Kind::Counter,
    },
    Definition {
        name: STORE_SIZE,
        help: "Histogram of KV offload store operation size, in bytes",
        labels: &[],
        kind: Kind::Histogram(TRANSFER_SIZE_BUCKETS),
    },
    Definition {
        name: "vllm:kv_offload_lookup_sync_delay_seconds",
        help: "Histogram of the time spent in a single offload lookup call, in seconds",
        labels: &[],
        kind: Kind::Histogram(LOOKUP_SYNC_DELAY_BUCKETS),
    },
    Definition {
        name: "vllm:kv_offload_lookup_async_delay_seconds",
        help: "Histogram of time between a request's offload lookup first deferring and the \
               following lookup resolving, or request finish, in seconds",
        labels: &[],
        kind: Kind::Histogram(LOOKUP_ASYNC_DELAY_BUCKETS),
    },
    Definition {
        name: "vllm:kv_offload_allocation_failure",
        help: "Number of KV offload store allocation attempts that failed",
        labels: &[],
        kind: Kind::Counter,
    },
    // `CPUOffloadingSpec.build_metric_definitions`.
    Definition {
        name: "vllm:kv_offload_cpu_cache_usage_perc",
        help: "Fraction of CPU KV-cache space currently pinned by active transfers (0.0 = idle, \
               1.0 = saturated). Sustained high values indicate transfers (stores or promotions) \
               may be dropped due to insufficient capacity",
        labels: &[],
        kind: Kind::Gauge,
    },
    Definition {
        name: "vllm:kv_offload_cpu_cache_write_usage_perc",
        help: "Fraction of CPU KV-cache space currently pinned by in-flight stores that have not \
               yet completed (0.0 = idle, 1.0 = saturated)",
        labels: &[],
        kind: Kind::Gauge,
    },
    Definition {
        name: "vllm:kv_offload_cpu_cache_read_usage_perc",
        help: "Fraction of CPU KV-cache space currently pinned by in-flight loads that have not \
               yet completed (0.0 = idle, 1.0 = saturated)",
        labels: &[],
        kind: Kind::Gauge,
    },
    Definition {
        name: "vllm:kv_offload_cpu_allocation_size",
        help: "Histogram of the number of CPU chunks requested by each KV offload prepare_store \
               call",
        labels: &[],
        kind: Kind::Histogram(CPU_ALLOCATION_SIZE_BUCKETS),
    },
    // Declared only with `store_threshold >= 2`, which only then reports it.
    Definition {
        name: "vllm:kv_offload_stores_skipped",
        help: "Number of KV offload stores skipped because the reuse threshold was not reached",
        labels: &[],
        kind: Kind::Counter,
    },
    // `TieringOffloadingSpec.build_metric_definitions`, on top of the CPU
    // spec's. Secondary tiers may add their own families, which are not here.
    Definition {
        name: "vllm:kv_offload_tiering_lookup_sync_delay_seconds",
        help: "Histogram of blocking time spent in a per-chunk tier lookup that resolved as a \
               hit or miss, labeled by tier, in seconds",
        labels: &["tier"],
        kind: Kind::Histogram(LOOKUP_SYNC_DELAY_BUCKETS),
    },
    Definition {
        name: "vllm:kv_offload_tiering_lookup_async_delay_seconds",
        help: "Histogram of wall-clock time from a per-chunk tier lookup first returning \
               retry until that same tier lookup resolves as a hit or miss, labeled by tier, \
               in seconds",
        labels: &["tier"],
        kind: Kind::Histogram(LOOKUP_ASYNC_DELAY_BUCKETS),
    },
    Definition {
        name: "vllm:kv_offload_tiering_read_bytes",
        help: "Total bytes read from secondary tiers into the primary tier, labeled by tier",
        labels: &["tier"],
        kind: Kind::Counter,
    },
    Definition {
        name: "vllm:kv_offload_tiering_read_time",
        help: "Total time spent reading from secondary tiers into the primary tier, in \
               seconds, labeled by tier. Tiers that do not report per-job transfer time have \
               no series here. Use vllm:kv_offload_tiering_promotion_latency_seconds for \
               tier-independent promotion timing",
        labels: &["tier"],
        kind: Kind::Counter,
    },
    Definition {
        name: "vllm:kv_offload_tiering_write_bytes",
        help: "Total bytes written from the primary tier to secondary tiers, labeled by tier",
        labels: &["tier"],
        kind: Kind::Counter,
    },
    Definition {
        name: "vllm:kv_offload_tiering_write_time",
        help: "Total time spent writing from the primary tier to secondary tiers, in seconds, \
               labeled by tier. Tiers that do not report per-job transfer time have no series \
               here",
        labels: &["tier"],
        kind: Kind::Counter,
    },
    Definition {
        name: "vllm:kv_offload_tiering_promotion_job_failures",
        help: "Number of failed secondary-tier promotion jobs, labeled by tier",
        labels: &["tier"],
        kind: Kind::Counter,
    },
    Definition {
        name: "vllm:kv_offload_tiering_cascade_job_failures",
        help: "Number of failed secondary-tier cascade jobs, labeled by tier",
        labels: &["tier"],
        kind: Kind::Counter,
    },
    Definition {
        name: "vllm:kv_offload_tiering_chunk_queries",
        help: "Number of chunk lookup queries sent to a tier, labeled by tier",
        labels: &["tier"],
        kind: Kind::Counter,
    },
    Definition {
        name: "vllm:kv_offload_tiering_chunk_hits",
        help: "Number of chunk lookup hits in a tier, labeled by tier",
        labels: &["tier"],
        kind: Kind::Counter,
    },
    Definition {
        name: "vllm:kv_offload_tiering_promotion_allocation_failures",
        help: "Number of promotion attempts that failed because the primary tier could not \
               allocate space",
        labels: &[],
        kind: Kind::Counter,
    },
    Definition {
        name: "vllm:kv_offload_tiering_primary_write_usage_perc",
        help: "Current fraction of primary-tier space used by writes from secondary tiers, \
               labeled by tier",
        labels: &["tier"],
        kind: Kind::Gauge,
    },
    Definition {
        name: "vllm:kv_offload_tiering_primary_read_usage_perc",
        help: "Current fraction of primary-tier space used by reads to secondary tiers, \
               labeled by tier",
        labels: &["tier"],
        kind: Kind::Gauge,
    },
    Definition {
        name: "vllm:kv_offload_tiering_active_promotion_jobs",
        help: "Number of active secondary-tier promotion jobs, labeled by tier",
        labels: &["tier"],
        kind: Kind::Gauge,
    },
    Definition {
        name: "vllm:kv_offload_tiering_active_cascade_jobs",
        help: "Number of active secondary-tier cascade jobs, labeled by tier",
        labels: &["tier"],
        kind: Kind::Gauge,
    },
    Definition {
        name: "vllm:kv_offload_tiering_deferred_release_chunks",
        help: "Number of primary-tier chunks of failed promotions whose release to the free \
               pool is deferred until the tier confirms its I/O can no longer write to them, \
               labeled by tier",
        labels: &["tier"],
        kind: Kind::Gauge,
    },
    Definition {
        name: "vllm:kv_offload_tiering_deferred_release_hold_seconds",
        help: "Histogram of how long a failed promotion's primary-tier chunks were held \
               before their deferred release completed, observed at release, labeled by tier, \
               in seconds",
        labels: &["tier"],
        kind: Kind::Histogram(DEFERRED_RELEASE_HOLD_BUCKETS),
    },
    Definition {
        name: "vllm:kv_offload_tiering_promotion_latency_seconds",
        help: "Histogram of per-job latency for secondary-tier promotion attempts, measured \
               from job creation until the tiering manager observes completion so that tier \
               queueing is included, labeled by tier and result, in seconds. Includes failed \
               attempts",
        labels: &["tier", "result"],
        kind: Kind::Histogram(PROMOTION_LATENCY_BUCKETS),
    },
    Definition {
        name: "vllm:kv_offload_tiering_backpressure_store_latency_ema",
        help: "Exponential moving average of store latency for back-pressure detection, in s/MiB",
        labels: &["tier"],
        kind: Kind::Gauge,
    },
    Definition {
        name: "vllm:kv_offload_tiering_backpressure_stores_dropped",
        help: "Number of store operations dropped due to back-pressure on a secondary tier",
        labels: &["tier"],
        kind: Kind::Counter,
    },
    Definition {
        name: "vllm:kv_offload_tiering_backpressure_blocks_dropped",
        help: "Number of blocks dropped due to back-pressure on a secondary tier",
        labels: &["tier"],
        kind: Kind::Counter,
    },
];

/// Constructor of histograms with fixed bucket bounds.
#[derive(Clone, Copy, Debug)]
struct FixedBuckets(&'static [f64]);

impl MetricConstructor<Histogram> for FixedBuckets {
    fn new_metric(&self) -> Histogram {
        Histogram::new(self.0.iter().copied())
    }
}

/// Label set of these families, whose label names vary by family.
type Labels = Vec<(String, String)>;

#[derive(Clone, Debug)]
enum OffloadFamily {
    Counter(Family<Labels, F64Counter>),
    Gauge(Family<Labels, F64Gauge>),
    Histogram(Family<Labels, Histogram, FixedBuckets>),
}

/// The `vllm:kv_offload_*` families. Mirrors `OffloadPromMetrics`.
#[derive(Clone, Debug)]
pub(super) struct OffloadingMetrics {
    families: HashMap<&'static str, (&'static Definition, OffloadFamily)>,
    // Deprecated predecessors of the flat transfer families, labeled by
    // `transfer_type`.
    total_bytes: Family<Labels, F64Counter>,
    total_time: Family<Labels, F64Counter>,
    size: Family<Labels, Histogram, FixedBuckets>,
    config_info: Family<Labels, F64Gauge>,
}

impl OffloadingMetrics {
    pub(super) fn register(registry: &mut Registry) -> Self {
        let families = DEFINITIONS
            .iter()
            .map(|definition| {
                let family = match definition.kind {
                    Kind::Counter => {
                        let family = Family::default();
                        registry.register(definition.name, definition.help, family.clone());
                        OffloadFamily::Counter(family)
                    }
                    Kind::Gauge => {
                        let family = Family::default();
                        registry.register(definition.name, definition.help, family.clone());
                        OffloadFamily::Gauge(family)
                    }
                    Kind::Histogram(buckets) => {
                        let family = Family::new_with_constructor(FixedBuckets(buckets));
                        registry.register(definition.name, definition.help, family.clone());
                        OffloadFamily::Histogram(family)
                    }
                };
                (definition.name, (definition, family))
            })
            .collect();

        let total_bytes = Family::default();
        registry.register(
            "vllm:kv_offload_total_bytes",
            "Number of bytes offloaded by KV connector",
            total_bytes.clone(),
        );
        let total_time = Family::default();
        registry.register(
            "vllm:kv_offload_total_time",
            "Total time measured by all KV offloading operations",
            total_time.clone(),
        );
        let size = Family::new_with_constructor(FixedBuckets(TRANSFER_SIZE_BUCKETS));
        registry.register(
            "vllm:kv_offload_size",
            "Histogram of KV offload transfer size, in bytes",
            size.clone(),
        );

        let config_info = Family::default();
        registry.register(
            "vllm:kv_offload_config_info",
            "Static configuration of the KV offload managers of this engine instance. The \
             configured offloading spec declares the label names, and each manager fills the \
             values, so a series appears from the first scheduler step of its engine. A manager \
             reports one series for each configuration it holds, and a label of its own tells \
             the series apart. Each engine reports its own configuration, not the instance total",
            config_info.clone(),
        );

        Self {
            families,
            total_bytes,
            total_time,
            size,
            config_info,
        }
    }

    pub(super) fn record(&self, engine: &EngineLabels, stats: &OffloadingStats) {
        if let Some(info) = &stats.info {
            self.record_config_info(engine, info);
        }
        for (name, samples) in &stats.data {
            for (label_values, sample) in samples {
                self.record_sample(engine, name, label_values, sample);
            }
        }
    }

    /// Record one family's sample. A family Rust does not define, a label
    /// count other than its definition's, or a sample of the wrong shape is
    /// dropped; Python asserts on all three.
    fn record_sample(
        &self,
        engine: &EngineLabels,
        name: &str,
        label_values: &[String],
        sample: &OffloadingSample,
    ) {
        let Some((definition, family)) = self.families.get(name) else {
            return;
        };
        if label_values.len() != definition.labels.len() {
            return;
        }
        let labels = engine_labels(engine)
            .chain(
                definition
                    .labels
                    .iter()
                    .zip(label_values)
                    .map(|(name, value)| ((*name).to_string(), value.clone())),
            )
            .collect();
        match (family, sample) {
            (OffloadFamily::Counter(family), OffloadingSample::Value(value)) => {
                family.get_or_create(&labels).inc_by(*value);
            }
            (OffloadFamily::Gauge(family), OffloadingSample::Value(value)) => {
                family.get_or_create(&labels).set(*value);
            }
            (OffloadFamily::Histogram(family), OffloadingSample::Observations(values)) => {
                let histogram = family.get_or_create(&labels);
                for value in values {
                    histogram.observe(*value);
                }
            }
            _ => return,
        }
        self.record_deprecated(engine, name, sample);
    }

    /// Feed the deprecated `transfer_type` families from the flat transfer
    /// families, as Python does for `CPUOffloadingSpec` and its subclasses,
    /// which cover every built-in spec.
    fn record_deprecated(&self, engine: &EngineLabels, name: &str, sample: &OffloadingSample) {
        let transfer_type = match name {
            LOAD_BYTES | LOAD_TIME | LOAD_SIZE => "CPU_to_GPU",
            STORE_BYTES | STORE_TIME | STORE_SIZE => "GPU_to_CPU",
            _ => return,
        };
        let labels = engine_labels(engine)
            .chain([("transfer_type".to_string(), transfer_type.to_string())])
            .collect();
        match (name, sample) {
            (LOAD_BYTES | STORE_BYTES, OffloadingSample::Value(value)) => {
                self.total_bytes.get_or_create(&labels).inc_by(*value);
            }
            (LOAD_TIME | STORE_TIME, OffloadingSample::Value(value)) => {
                self.total_time.get_or_create(&labels).inc_by(*value);
            }
            (LOAD_SIZE | STORE_SIZE, OffloadingSample::Observations(values)) => {
                let histogram = self.size.get_or_create(&labels);
                for value in values {
                    histogram.observe(*value);
                }
            }
            _ => {}
        }
    }

    /// Publish one engine's `vllm:kv_offload_config_info` series, one per
    /// mapping of label name to value, each set to 1.
    ///
    /// Python labels every series with the names the offloading spec declares,
    /// empty where a series fills none. Rust does not know the spec, so it
    /// takes the names the series fill; the two agree unless a declared name
    /// is filled by no series, which Python warns about.
    fn record_config_info(&self, engine: &EngineLabels, series: &[BTreeMap<String, String>]) {
        let names: BTreeSet<&String> = series.iter().flat_map(BTreeMap::keys).collect();
        for info in series {
            let labels = engine_labels(engine)
                .chain(names.iter().map(|name| {
                    let value = info.get(*name).cloned().unwrap_or_default();
                    ((*name).clone(), value)
                }))
                .collect();
            self.config_info.get_or_create(&labels).set(1.0);
        }
    }
}

fn engine_labels(engine: &EngineLabels) -> impl Iterator<Item = (String, String)> {
    [
        ("model_name".to_string(), engine.model_name.clone()),
        ("engine".to_string(), engine.engine.to_string()),
    ]
    .into_iter()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use expect_test::expect;
    use itertools::Itertools as _;

    use super::{OffloadingSample, OffloadingStats};
    use crate::{KvConnectorStats, Metrics, MultiConnectorStats};

    /// Records `OffloadingConnectorStats` payloads into the `vllm:kv_offload_*`
    /// families: the flat transfer families also feed their deprecated
    /// `transfer_type` predecessors, the config facts become
    /// `vllm:kv_offload_config_info`, and a family Rust does not define or a
    /// sample with the wrong label count is dropped.
    #[test]
    fn offloading_stats_are_recorded_into_kv_offload_metrics() {
        let metrics = Metrics::new();
        let handles = metrics.scheduler.resolve_handles("model", 0);
        // Series of one family render in hash order, so sort them.
        let kv_offload_series = || {
            metrics
                .render()
                .unwrap()
                .lines()
                .filter(|line| line.starts_with("vllm:kv_offload_") && !line.contains("_bucket{"))
                .sorted()
                .join("\n")
        };

        // Like Python, no series before the connector reports.
        expect![[""]].assert_eq(&kv_offload_series());

        let value = |value| BTreeMap::from([(vec![], OffloadingSample::Value(value))]);
        let interval = |usage, info| OffloadingStats {
            data: BTreeMap::from([
                ("vllm:kv_offload_load_bytes".to_string(), value(4096.0)),
                ("vllm:kv_offload_store_time".to_string(), value(0.25)),
                (
                    "vllm:kv_offload_store_size".to_string(),
                    BTreeMap::from([(vec![], OffloadingSample::Observations(vec![2e6, 3e6]))]),
                ),
                (
                    "vllm:kv_offload_cpu_cache_usage_perc".to_string(),
                    value(usage),
                ),
                // `TieringOffloadingSpec` families, labeled by tier.
                (
                    "vllm:kv_offload_tiering_chunk_hits".to_string(),
                    BTreeMap::from([(vec!["1:kvcr".to_string()], OffloadingSample::Value(3.0))]),
                ),
                (
                    "vllm:kv_offload_tiering_promotion_latency_seconds".to_string(),
                    BTreeMap::from([(
                        vec!["1:kvcr".to_string(), "success".to_string()],
                        OffloadingSample::Observations(vec![0.02]),
                    )]),
                ),
                // Wrong label count for an unlabeled family.
                (
                    "vllm:kv_offload_allocation_failure".to_string(),
                    BTreeMap::from([(vec!["x".to_string()], OffloadingSample::Value(1.0))]),
                ),
                // A family of a spec Rust does not know.
                ("vllm:kv_offload_custom_spec_total".to_string(), value(1.0)),
            ]),
            info,
        };
        let info = vec![BTreeMap::from([
            ("cpu_num_chunks".to_string(), "64".to_string()),
            ("cpu_kv_cache_size_tokens".to_string(), "None".to_string()),
        ])];
        // Once as the only connector, once as a `MultiConnector` child:
        // counters and histograms add up, gauges keep the latest value.
        for kv_connector_stats in [
            KvConnectorStats::Offloading(interval(0.5, Some(info))),
            KvConnectorStats::Multi(Box::new(MultiConnectorStats {
                offloading: Some(interval(0.75, None)),
                ..Default::default()
            })),
        ] {
            handles.kv_connector.record(&kv_connector_stats);
        }

        expect![[r#"
            vllm:kv_offload_config_info{model_name="model",engine="0",cpu_kv_cache_size_tokens="None",cpu_num_chunks="64"} 1.0
            vllm:kv_offload_cpu_cache_usage_perc{model_name="model",engine="0"} 0.75
            vllm:kv_offload_load_bytes_total{model_name="model",engine="0"} 8192.0
            vllm:kv_offload_size_count{model_name="model",engine="0",transfer_type="GPU_to_CPU"} 4
            vllm:kv_offload_size_sum{model_name="model",engine="0",transfer_type="GPU_to_CPU"} 10000000.0
            vllm:kv_offload_store_size_count{model_name="model",engine="0"} 4
            vllm:kv_offload_store_size_sum{model_name="model",engine="0"} 10000000.0
            vllm:kv_offload_store_time_total{model_name="model",engine="0"} 0.5
            vllm:kv_offload_tiering_chunk_hits_total{model_name="model",engine="0",tier="1:kvcr"} 6.0
            vllm:kv_offload_tiering_promotion_latency_seconds_count{model_name="model",engine="0",tier="1:kvcr",result="success"} 2
            vllm:kv_offload_tiering_promotion_latency_seconds_sum{model_name="model",engine="0",tier="1:kvcr",result="success"} 0.04
            vllm:kv_offload_total_bytes_total{model_name="model",engine="0",transfer_type="CPU_to_GPU"} 8192.0
            vllm:kv_offload_total_time_total{model_name="model",engine="0",transfer_type="GPU_to_CPU"} 0.5"#]].assert_eq(&kv_offload_series());
    }
}
