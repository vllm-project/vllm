// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Summary statistics over one variant's timed runs.

/// Summary statistics for a variant's `ready_secs` samples.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Summary {
    pub samples: usize,
    pub failures: usize,
    pub mean_secs: f64,
    pub median_secs: f64,
    pub min_secs: f64,
    pub max_secs: f64,
    pub stddev_secs: f64,
}

impl Summary {
    /// Compute summary statistics from successful-run samples; `failures`
    /// is the count of runs that never became ready.
    ///
    /// Returns `None` if there are no successful samples to summarize.
    pub fn from_samples(mut samples: Vec<f64>, failures: usize) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }
        samples.sort_by(|a, b| a.total_cmp(b));

        let n = samples.len();
        let mean = samples.iter().sum::<f64>() / n as f64;
        let median = if n.is_multiple_of(2) {
            (samples[n / 2 - 1] + samples[n / 2]) / 2.0
        } else {
            samples[n / 2]
        };
        let variance = samples.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;

        Some(Self {
            samples: n,
            failures,
            mean_secs: mean,
            median_secs: median,
            min_secs: samples[0],
            max_secs: samples[n - 1],
            stddev_secs: variance.sqrt(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::Summary;

    #[test]
    fn computes_mean_median_min_max() {
        let summary = Summary::from_samples(vec![1.0, 2.0, 3.0, 4.0], 0).unwrap();
        assert_eq!(summary.samples, 4);
        assert_eq!(summary.failures, 0);
        assert_eq!(summary.mean_secs, 2.5);
        assert_eq!(summary.median_secs, 2.5);
        assert_eq!(summary.min_secs, 1.0);
        assert_eq!(summary.max_secs, 4.0);
    }

    #[test]
    fn odd_count_median_is_middle_sample() {
        let summary = Summary::from_samples(vec![5.0, 1.0, 3.0], 0).unwrap();
        assert_eq!(summary.median_secs, 3.0);
    }

    #[test]
    fn no_successful_samples_returns_none() {
        assert!(Summary::from_samples(vec![], 3).is_none());
    }
}
