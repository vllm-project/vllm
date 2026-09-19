// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use parking_lot::RwLock;
use prometheus_client::encoding::{EncodeMetric, MetricEncoder, NoLabelSet};
use prometheus_client::metrics::{MetricType, TypedMetric};

use crate::request::ITL_BUCKETS;

const BUCKET_COUNT: usize = ITL_BUCKETS.len() + 1;

/// Request-local ITL observations, published at token intervals and stream end.
#[derive(Debug, Default)]
pub struct InterTokenLatencyObservations {
    sum: f64,
    count: u64,
    buckets: [u64; BUCKET_COUNT],
}

impl InterTokenLatencyObservations {
    /// Record one interval using the existing histogram's bucket rules.
    pub fn observe(&mut self, value: f64) {
        self.sum += value;
        self.count += 1;
        // Match prometheus-client's implicit final bucket, including nonfinite values.
        if let Some(index) = ITL_BUCKETS
            .iter()
            .copied()
            .chain(std::iter::once(f64::MAX))
            .position(|bound| bound >= value)
        {
            self.buckets[index] += 1;
        }
    }
}

/// ITL histogram that merges a batch of observations under one shared lock.
///
/// Active streams retain observations locally between token-based flushes.
/// Completion, error, and drop flush the remainder; process failure can lose it.
#[derive(Clone, Debug, Default)]
pub struct InterTokenLatencyHistogram {
    inner: Arc<RwLock<InterTokenLatencyObservations>>,
}

impl InterTokenLatencyHistogram {
    /// Merge and clear pending observations. Repeated empty flushes are no-ops.
    pub fn flush(&self, pending: &mut InterTokenLatencyObservations) {
        if pending.count == 0 {
            return;
        }
        let mut inner = self.inner.write();
        let pending = std::mem::take(pending);
        inner.sum += pending.sum;
        inner.count += pending.count;
        for (total, count) in inner.buckets.iter_mut().zip(pending.buckets) {
            *total += count;
        }
    }
}

impl TypedMetric for InterTokenLatencyHistogram {
    const TYPE: MetricType = MetricType::Histogram;
}

impl EncodeMetric for InterTokenLatencyHistogram {
    fn encode(&self, mut encoder: MetricEncoder) -> Result<(), std::fmt::Error> {
        let inner = self.inner.read();
        let buckets: [_; BUCKET_COUNT] = std::array::from_fn(|index| {
            (
                ITL_BUCKETS.get(index).copied().unwrap_or(f64::MAX),
                inner.buckets[index],
            )
        });
        encoder.encode_histogram::<NoLabelSet>(inner.sum, inner.count, &buckets, None)
    }

    fn metric_type(&self) -> MetricType {
        Self::TYPE
    }
}

#[cfg(test)]
mod tests {
    use prometheus_client::encoding::text::encode;
    use prometheus_client::metrics::histogram::Histogram;
    use prometheus_client::registry::Registry;

    use super::*;

    fn render(metric: impl EncodeMetric + std::fmt::Debug + Send + Sync + 'static) -> String {
        let mut registry = Registry::default();
        registry.register("itl", "Intervals", metric);
        let mut output = String::new();
        encode(&mut output, &registry).unwrap();
        output
    }

    #[test]
    fn completed_batches_match_existing_histogram() {
        let mut boundaries = vec![-1.0, 0.0, 81.0];
        for bound in ITL_BUCKETS {
            boundaries.extend([bound.next_down(), bound, bound.next_up()]);
        }
        for values in [
            vec![],
            boundaries,
            vec![f64::MAX],
            vec![f64::INFINITY],
            vec![f64::NEG_INFINITY],
            vec![f64::NAN],
        ] {
            let baseline = Histogram::new(ITL_BUCKETS);
            let buffered = InterTokenLatencyHistogram::default();
            for batch in values.chunks(3) {
                let mut pending = InterTokenLatencyObservations::default();
                for value in batch {
                    baseline.observe(*value);
                    pending.observe(*value);
                }
                buffered.flush(&mut pending);
                buffered.flush(&mut pending);
            }
            let expected_text = render(baseline);
            let actual_text = render(buffered);
            assert_eq!(expected_text.lines().count(), actual_text.lines().count());
            for (expected, actual) in expected_text.lines().zip(actual_text.lines()) {
                if expected.starts_with("itl_sum ") {
                    let expected: f64 =
                        expected.split_whitespace().last().unwrap().parse().unwrap();
                    let actual: f64 = actual.split_whitespace().last().unwrap().parse().unwrap();
                    assert!(
                        expected == actual
                            || (expected.is_nan() && actual.is_nan())
                            || (expected - actual).abs() <= 1e-12 * expected.abs().max(1.0)
                    );
                } else {
                    assert_eq!(actual, expected);
                }
            }
        }
    }

    #[test]
    fn concurrent_flushes_and_scrapes_keep_complete_observations() {
        let histogram = InterTokenLatencyHistogram::default();
        let barrier = std::sync::Barrier::new(3);
        std::thread::scope(|scope| {
            for _ in 0..2 {
                let histogram = &histogram;
                let barrier = &barrier;
                scope.spawn(move || {
                    for _ in 0..16 {
                        let mut pending = InterTokenLatencyObservations::default();
                        pending.observe(0.5);
                        pending.observe(0.5);
                        barrier.wait();
                        histogram.flush(&mut pending);
                        barrier.wait();
                    }
                });
            }
            for _ in 0..16 {
                barrier.wait();
                let output = render(histogram.clone());
                let value = |name: &str| {
                    output
                        .lines()
                        .find_map(|line| line.strip_prefix(name))
                        .unwrap()
                        .parse::<f64>()
                        .unwrap()
                };
                assert_eq!(value("itl_sum ") * 2.0, value("itl_count "));
                assert_eq!(value("itl_bucket{le=\"+Inf\"} "), value("itl_count "));
                barrier.wait();
            }
        });
        assert_eq!(histogram.inner.read().count, 64);
    }
}
