// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::fmt;
use std::iter::once;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use prometheus_client::encoding::{EncodeMetric, MetricEncoder, NoLabelSet};
use prometheus_client::metrics::{MetricType, TypedMetric};

/// Prometheus histogram with lock-free observations.
///
/// `prometheus_client::metrics::histogram::Histogram` takes an exclusive lock for every
/// observation, which serializes all Tokio workers on hot histograms such as inter-token
/// latency. This histogram updates atomic cells instead, so observers never block each other
/// or a concurrent scrape.
///
/// A scrape may see an observation's bucket count before its sum. `_count` is derived from the
/// bucket counts, so it always equals the `+Inf` bucket.
#[derive(Clone, Debug)]
pub struct Histogram {
    inner: Arc<Inner>,
}

#[derive(Debug)]
struct Inner {
    /// Upper bounds, ending with `f64::MAX`, which is encoded as `+Inf`.
    upper_bounds: Box<[f64]>,
    /// Non-cumulative observation count of each bucket.
    bucket_counts: Box<[AtomicU64]>,
    /// Bits of the `f64` sum of all observations.
    sum_bits: AtomicU64,
}

impl Histogram {
    /// Create a histogram with the given finite bucket upper bounds, in ascending order.
    pub fn new(buckets: impl IntoIterator<Item = f64>) -> Self {
        let upper_bounds: Box<[f64]> = buckets.into_iter().chain(once(f64::MAX)).collect();
        let bucket_counts = upper_bounds.iter().map(|_| AtomicU64::new(0)).collect();
        Self {
            inner: Arc::new(Inner {
                upper_bounds,
                bucket_counts,
                sum_bits: AtomicU64::new(0.0_f64.to_bits()),
            }),
        }
    }

    pub fn observe(&self, value: f64) {
        let inner = &*self.inner;
        // Unlike `prometheus_client`, which counts `+inf` and NaN in `_count` but in no bucket,
        // put them in the `+Inf` bucket as the Go client does.
        let index = inner
            .upper_bounds
            .iter()
            .position(|bound| value <= *bound)
            .unwrap_or(inner.upper_bounds.len() - 1);
        inner.bucket_counts[index].fetch_add(1, Ordering::Relaxed);
        // The closure always returns `Some`, so the update cannot fail.
        let _ = inner.sum_bits.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |bits| {
            Some((f64::from_bits(bits) + value).to_bits())
        });
    }
}

impl TypedMetric for Histogram {
    const TYPE: MetricType = MetricType::Histogram;
}

impl EncodeMetric for Histogram {
    fn encode(&self, mut encoder: MetricEncoder) -> Result<(), fmt::Error> {
        let inner = &*self.inner;
        let buckets: Vec<(f64, u64)> = inner
            .upper_bounds
            .iter()
            .zip(&inner.bucket_counts)
            .map(|(bound, count)| (*bound, count.load(Ordering::Relaxed)))
            .collect();
        let count = buckets.iter().map(|(_, count)| count).sum();
        let sum = f64::from_bits(inner.sum_bits.load(Ordering::Relaxed));
        encoder.encode_histogram::<NoLabelSet>(sum, count, &buckets, None)
    }

    fn metric_type(&self) -> MetricType {
        Self::TYPE
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use prometheus_client::encoding::text::encode;
    use prometheus_client::registry::Registry;

    use super::*;

    const BUCKETS: [f64; 3] = [0.1, 1.0, 10.0];

    fn render(metric: impl EncodeMetric + fmt::Debug + Send + Sync + 'static) -> String {
        let mut registry = Registry::default();
        registry.register("latency", "Latency", metric);
        let mut output = String::new();
        encode(&mut output, &registry).unwrap();
        output
    }

    #[test]
    #[expect(
        clippy::disallowed_types,
        reason = "compare against the replaced implementation"
    )]
    fn finite_observations_match_prometheus_client() {
        let baseline = prometheus_client::metrics::histogram::Histogram::new(BUCKETS);
        let histogram = Histogram::new(BUCKETS);
        let mut values = vec![-1.0, 0.0, 100.0, f64::MAX];
        for bound in BUCKETS {
            values.extend([bound.next_down(), bound, bound.next_up()]);
        }
        for value in values {
            baseline.observe(value);
            histogram.observe(value);
        }
        assert_eq!(render(histogram), render(baseline));
    }

    #[test]
    fn concurrent_observations_are_not_lost() {
        let histogram = Histogram::new(BUCKETS);
        std::thread::scope(|scope| {
            // Dyadic values keep the sum exact in any addition order.
            for value in [0.0625, 0.5, 4.0, 64.0] {
                let histogram = histogram.clone();
                scope.spawn(move || (0..10_000).for_each(|_| histogram.observe(value)));
            }
        });
        expect![[r##"
            # HELP latency Latency.
            # TYPE latency histogram
            latency_sum 685625.0
            latency_count 40000
            latency_bucket{le="0.1"} 10000
            latency_bucket{le="1.0"} 20000
            latency_bucket{le="10.0"} 30000
            latency_bucket{le="+Inf"} 40000
            # EOF
        "##]]
        .assert_eq(&render(histogram));
    }

    #[test]
    fn non_finite_observations_count_in_inf_bucket() {
        let histogram = Histogram::new(BUCKETS);
        histogram.observe(f64::INFINITY);
        histogram.observe(f64::NAN);
        expect![[r##"
            # HELP latency Latency.
            # TYPE latency histogram
            latency_sum NaN
            latency_count 2
            latency_bucket{le="0.1"} 0
            latency_bucket{le="1.0"} 0
            latency_bucket{le="10.0"} 0
            latency_bucket{le="+Inf"} 2
            # EOF
        "##]]
        .assert_eq(&render(histogram));
    }
}
