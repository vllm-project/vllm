// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Bucket boundaries of the core request and KV-cache histograms, grouped into
//! the families of Python's `vllm/v1/metrics/buckets.py` so that
//! `--custom-histogram-buckets` can replace a family's defaults. Histograms
//! owned by other subsystems, such as the KV connector metrics, keep their own
//! boundaries.

use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock};

use itertools::Itertools as _;
use prometheus_client::metrics::family::MetricConstructor;
use serde::{Deserialize, Serialize};

use crate::Histogram;

const REQUEST_LATENCY_BUCKETS: [f64; 21] = [
    0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0,
    480.0, 960.0, 1920.0, 7680.0,
];
const TTFT_BUCKETS: [f64; 22] = [
    0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0,
    20.0, 40.0, 80.0, 160.0, 640.0, 2560.0,
];
const ITL_BUCKETS: [f64; 19] = [
    0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0,
    40.0, 80.0,
];
const ITERATION_TOKENS_BUCKETS: [f64; 13] = [
    1.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0, 16384.0,
];
const REQUEST_PARAMS_N_BUCKETS: [f64; 5] = [1.0, 2.0, 5.0, 10.0, 20.0];
const REQUEST_NUM_PREEMPTIONS_BUCKETS: [f64; 7] = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0];
const KV_CACHE_RESIDENCY_BUCKETS: [f64; 21] = [
    0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0,
    120.0, 300.0, 600.0, 1200.0, 1800.0,
];

/// Histogram bucket family, keyed like Python's `BucketFamilyKey`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BucketFamily {
    RequestLatency,
    TimeToFirstToken,
    InterTokenLatency,
    IterationTokens,
    RequestParamsN,
    RequestNumPreemptions,
    RequestTokens,
    KvCacheResidency,
}

/// Replacement bucket bounds per family, from `--custom-histogram-buckets`.
///
/// Deserialization validates the bounds like Python's
/// `ObservabilityConfig.custom_histogram_buckets`: each list is non-empty,
/// finite, greater than 0, and strictly increasing.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "BTreeMap<BucketFamily, Vec<f64>>")]
pub struct CustomHistogramBuckets(BTreeMap<BucketFamily, Vec<f64>>);

// Validated bounds are finite, so equality is reflexive.
impl Eq for CustomHistogramBuckets {}

impl TryFrom<BTreeMap<BucketFamily, Vec<f64>>> for CustomHistogramBuckets {
    type Error = String;

    fn try_from(families: BTreeMap<BucketFamily, Vec<f64>>) -> Result<Self, Self::Error> {
        for (family, buckets) in &families {
            let family = serde_json::to_string(family).expect("family key serializes");
            if buckets.is_empty() {
                return Err(format!(
                    "custom_histogram_buckets[{family}]: bucket list must not be empty"
                ));
            }
            if let Some(bound) = buckets.iter().find(|bound| !bound.is_finite() || **bound <= 0.0) {
                return Err(format!(
                    "custom_histogram_buckets[{family}]: bound {bound} must be finite and greater than 0"
                ));
            }
            if buckets.iter().tuple_windows().any(|(a, b)| a >= b) {
                return Err(format!(
                    "custom_histogram_buckets[{family}]: bounds {buckets:?} must be strictly increasing"
                ));
            }
        }
        Ok(Self(families))
    }
}

/// Bucket settings fixed at startup and shared by every family histogram
/// constructor.
///
/// The registry is built before the server reads its config or connects an
/// engine, so both settings are filled in later, before the first histogram
/// of a family is created.
#[derive(Debug, Default)]
pub(crate) struct BucketConfig {
    custom: OnceLock<CustomHistogramBuckets>,
    /// Cap of the `request_tokens` 1-2-5 series.
    max_model_len: OnceLock<u32>,
}

impl BucketConfig {
    pub(crate) fn set_custom(&self, custom: CustomHistogramBuckets) {
        let _ = self.custom.set(custom);
    }

    pub(crate) fn set_max_model_len(&self, max_model_len: u32) {
        let _ = self.max_model_len.set(max_model_len);
    }

    /// Return the bucket bounds of `family`: the custom ones if configured,
    /// else the defaults.
    fn buckets(&self, family: BucketFamily) -> Vec<f64> {
        if let Some(buckets) = self.custom.get().and_then(|custom| custom.0.get(&family)) {
            return buckets.clone();
        }
        match family {
            BucketFamily::RequestLatency => REQUEST_LATENCY_BUCKETS.to_vec(),
            BucketFamily::TimeToFirstToken => TTFT_BUCKETS.to_vec(),
            BucketFamily::InterTokenLatency => ITL_BUCKETS.to_vec(),
            BucketFamily::IterationTokens => ITERATION_TOKENS_BUCKETS.to_vec(),
            BucketFamily::RequestParamsN => REQUEST_PARAMS_N_BUCKETS.to_vec(),
            BucketFamily::RequestNumPreemptions => REQUEST_NUM_PREEMPTIONS_BUCKETS.to_vec(),
            BucketFamily::RequestTokens => build_1_2_5_buckets(
                *self
                    .max_model_len
                    .get()
                    .expect("max_model_len must be set before recording request token metrics"),
            ),
            BucketFamily::KvCacheResidency => KV_CACHE_RESIDENCY_BUCKETS.to_vec(),
        }
    }
}

/// Constructor of the histograms of one bucket family.
#[derive(Clone, Debug)]
pub struct FamilyHistogram {
    family: BucketFamily,
    config: Arc<BucketConfig>,
}

impl FamilyHistogram {
    pub(crate) fn new(family: BucketFamily, config: &Arc<BucketConfig>) -> Self {
        Self {
            family,
            config: config.clone(),
        }
    }
}

impl MetricConstructor<Histogram> for FamilyHistogram {
    fn new_metric(&self) -> Histogram {
        Histogram::new(self.config.buckets(self.family))
    }
}

/// Build a 1-2-5 series of buckets capped at `max_value`, like Python's
/// `build_1_2_5_buckets`: `100` gives `[1, 2, 5, 10, 20, 50, 100]`.
fn build_1_2_5_buckets(max_value: u32) -> Vec<f64> {
    let max_value = u64::from(max_value);
    let mut buckets = Vec::new();
    let mut exponent = 0;
    loop {
        for mantissa in [1_u64, 2, 5] {
            let value = mantissa * 10_u64.pow(exponent);
            if value > max_value {
                return buckets;
            }
            buckets.push(value as f64);
        }
        exponent += 1;
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use itertools::Itertools as _;

    use super::*;
    use crate::{EngineLabels, Metrics};

    /// A family listed in `--custom-histogram-buckets` replaces the defaults of
    /// every histogram in it, including `request_tokens` without a
    /// `max_model_len`; other families keep their defaults.
    #[test]
    fn custom_buckets_replace_listed_family_defaults() {
        let metrics = Metrics::new();
        metrics.set_custom_histogram_buckets(
            serde_json::from_str(
                r#"{
                    "time_to_first_token": [0.5, 1.0],
                    "inter_token_latency": [0.2],
                    "request_tokens": [10, 100],
                    "kv_cache_residency": [60]
                }"#,
            )
            .unwrap(),
        );
        let labels = EngineLabels {
            model_name: "model".to_string(),
            engine: 0,
        };
        metrics.request.time_to_first_token_seconds.get_or_create_owned(&labels);
        metrics
            .request
            .request_time_per_output_token_seconds
            .get_or_create_owned(&labels);
        metrics.request.request_prompt_tokens.get_or_create_owned(&labels);
        metrics.request.request_params_n.get_or_create_owned(&labels);
        metrics.scheduler.kv_block_lifetime_seconds.get_or_create_owned(&labels);

        let output = metrics.render().unwrap();
        let bounds = [
            "vllm:time_to_first_token_seconds",
            "vllm:request_time_per_output_token_seconds",
            "vllm:request_prompt_tokens",
            "vllm:request_params_n",
            "vllm:kv_block_lifetime_seconds",
        ]
        .map(|name| {
            let bounds = output
                .lines()
                .filter(|line| line.starts_with(&format!("{name}_bucket")))
                .filter_map(|line| line.split_once("le=\"")?.1.split_once('"'))
                .map(|(bound, _)| bound)
                .join(" ");
            format!("{name}: {bounds}")
        })
        .join("\n");
        expect![[r#"
            vllm:time_to_first_token_seconds: 0.5 1.0 +Inf
            vllm:request_time_per_output_token_seconds: 0.2 +Inf
            vllm:request_prompt_tokens: 10.0 100.0 +Inf
            vllm:request_params_n: 1.0 2.0 5.0 10.0 20.0 +Inf
            vllm:kv_block_lifetime_seconds: 60.0 +Inf"#]]
        .assert_eq(&bounds);
    }

    #[test]
    fn custom_buckets_are_validated_like_python() {
        let errors = [
            r#"{"request_latency": []}"#,
            r#"{"request_latency": [0, 1]}"#,
            r#"{"request_latency": [1, 1]}"#,
            r#"{"request_latency": [true]}"#,
            r#"{"e2e_latency": [1]}"#,
        ]
        .map(|json| {
            let error = serde_json::from_str::<CustomHistogramBuckets>(json).unwrap_err();
            format!("{json} -> {error}")
        })
        .join("\n");
        expect![[r#"
            {"request_latency": []} -> custom_histogram_buckets["request_latency"]: bucket list must not be empty
            {"request_latency": [0, 1]} -> custom_histogram_buckets["request_latency"]: bound 0 must be finite and greater than 0
            {"request_latency": [1, 1]} -> custom_histogram_buckets["request_latency"]: bounds [1.0, 1.0] must be strictly increasing
            {"request_latency": [true]} -> invalid type: boolean `true`, expected f64 at line 1 column 25
            {"e2e_latency": [1]} -> unknown variant `e2e_latency`, expected one of `request_latency`, `time_to_first_token`, `inter_token_latency`, `iteration_tokens`, `request_params_n`, `request_num_preemptions`, `request_tokens`, `kv_cache_residency` at line 1 column 14"#]].assert_eq(&errors);
    }
}
