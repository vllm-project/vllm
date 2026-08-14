// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Gamma};

use crate::cli::RampUpStrategy;
use crate::config::RampUpConfig;

/// Pre-computed request schedule: cumulative delays from start and per-request rates.
pub struct RequestSchedule {
    /// Cumulative absolute delay from start (seconds) for each request.
    pub delays: Vec<f64>,
    /// Instantaneous rate for each request (for ramp-up logging).
    #[allow(dead_code)]
    pub rates: Vec<f64>,
}

/// Compute the per-request rate, accounting for optional ramp-up.
///
/// Mirrors Python's `_get_current_request_rate()` from serve.py:218-240.
fn get_current_request_rate(
    ramp_up: Option<&RampUpConfig>,
    request_index: usize,
    total_requests: usize,
    base_rate: f64,
) -> f64 {
    let config = match ramp_up {
        Some(c) => c,
        None => return base_rate,
    };

    let progress = request_index as f64 / (total_requests - 1).max(1) as f64;

    match config.strategy {
        RampUpStrategy::Linear => {
            let increase = (config.end_rps - config.start_rps) * progress;
            config.start_rps + increase
        }
        RampUpStrategy::Exponential => {
            let ratio = config.end_rps / config.start_rps;
            config.start_rps * ratio.powf(progress)
        }
    }
}

/// Compute the request schedule for all requests.
///
/// Ports the Python `get_request()` / `_generate_request_timestamps()` logic
/// from serve.py:243-340.
pub fn compute_schedule(
    num_requests: usize,
    request_rate: f64,
    burstiness: f64,
    seed: u64,
    ramp_up: Option<&RampUpConfig>,
) -> RequestSchedule {
    assert!(burstiness > 0.0, "burstiness must be positive");
    assert!(num_requests > 0, "must have at least one request");

    let mut rng = StdRng::seed_from_u64(seed);
    let mut delay_ts = Vec::with_capacity(num_requests);
    let mut rates = Vec::with_capacity(num_requests);

    for i in 0..num_requests {
        let current_rate = get_current_request_rate(ramp_up, i, num_requests, request_rate);
        rates.push(current_rate);

        if current_rate.is_infinite() {
            delay_ts.push(0.0);
        } else if burstiness.is_infinite() {
            // When burstiness → ∞, delay becomes constant = 1/rate
            delay_ts.push(1.0 / current_rate);
        } else {
            let theta = 1.0 / (current_rate * burstiness);
            let gamma = Gamma::new(burstiness, theta).unwrap();
            delay_ts.push(gamma.sample(&mut rng));
        }
    }

    // Compute cumulative delays
    for i in 1..delay_ts.len() {
        delay_ts[i] += delay_ts[i - 1];
    }

    // Normalize: scale cumulative delays so total matches target.
    // Only for fixed-rate (no ramp-up) mode, matching Python behavior.
    if ramp_up.is_none()
        && let Some(&last) = delay_ts.last()
        && last > 0.0
        && !request_rate.is_infinite()
    {
        let target_total = num_requests as f64 / request_rate;
        let factor = target_total / last;
        for d in &mut delay_ts {
            *d *= factor;
        }
    }

    RequestSchedule {
        delays: delay_ts,
        rates,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infinite_rate_all_zero_delays() {
        let sched = compute_schedule(100, f64::INFINITY, 1.0, 42, None);
        assert_eq!(sched.delays.len(), 100);
        for d in &sched.delays {
            assert_eq!(*d, 0.0);
        }
    }

    #[test]
    fn test_fixed_rate_monotonic() {
        let sched = compute_schedule(50, 10.0, 1.0, 42, None);
        for i in 1..sched.delays.len() {
            assert!(sched.delays[i] >= sched.delays[i - 1]);
        }
        // Total should be close to num_requests / rate = 5.0
        let last = *sched.delays.last().unwrap();
        assert!((last - 5.0).abs() < 0.01, "last delay = {last}");
    }

    #[test]
    fn test_infinite_burstiness_constant_delay() {
        let sched = compute_schedule(10, 5.0, f64::INFINITY, 42, None);
        // Each delay should be 0.2s apart (1/5)
        for i in 1..sched.delays.len() {
            let delta = sched.delays[i] - sched.delays[i - 1];
            assert!((delta - 0.2).abs() < 1e-10);
        }
    }

    #[test]
    fn test_linear_ramp_up() {
        let ramp = RampUpConfig {
            strategy: RampUpStrategy::Linear,
            start_rps: 1.0,
            end_rps: 10.0,
        };
        let sched = compute_schedule(10, 5.0, 1.0, 42, Some(&ramp));
        // Rates should increase linearly from 1.0 to 10.0
        assert!((sched.rates[0] - 1.0).abs() < 1e-10);
        assert!((sched.rates[9] - 10.0).abs() < 1e-10);
        // Middle should be ~5.5
        assert!((sched.rates[4] - 5.0).abs() < 0.5);
    }

    #[test]
    fn test_exponential_ramp_up() {
        let ramp = RampUpConfig {
            strategy: RampUpStrategy::Exponential,
            start_rps: 1.0,
            end_rps: 100.0,
        };
        let sched = compute_schedule(10, 5.0, 1.0, 42, Some(&ramp));
        assert!((sched.rates[0] - 1.0).abs() < 1e-10);
        assert!((sched.rates[9] - 100.0).abs() < 1e-10);
        // Exponential: rates should be monotonically increasing
        for i in 1..sched.rates.len() {
            assert!(sched.rates[i] >= sched.rates[i - 1]);
        }
    }
}
