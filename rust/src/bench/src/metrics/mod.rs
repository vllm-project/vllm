// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

pub mod calculator;
pub mod steady_state;

use serde::{Deserialize, Serialize};
pub use steady_state::SteadyStateMetrics;

/// Multi-turn benchmark metrics with overall and per-turn breakdown.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiTurnMetrics {
    pub overall: BenchmarkMetrics,
    pub per_turn: Vec<BenchmarkMetrics>,
    pub conversations_completed: usize,
    pub conversations_failed: usize,
    pub avg_turns_completed: f64,
    pub avg_conversation_duration_ms: f64,
}

/// Full benchmark metrics for generation tasks.
/// Matches Python's BenchmarkMetrics dataclass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMetrics {
    pub completed: usize,
    pub failed: usize,
    pub total_input: usize,
    pub total_output: usize,
    pub request_throughput: f64,
    pub request_goodput: f64,
    pub input_throughput: f64,
    pub output_throughput: f64,
    pub total_token_throughput: f64,
    pub mean_ttft_ms: f64,
    pub median_ttft_ms: f64,
    pub std_ttft_ms: f64,
    pub percentiles_ttft_ms: Vec<(f64, f64)>,
    pub mean_tpot_ms: f64,
    pub median_tpot_ms: f64,
    pub std_tpot_ms: f64,
    pub percentiles_tpot_ms: Vec<(f64, f64)>,
    pub mean_itl_ms: f64,
    pub median_itl_ms: f64,
    pub std_itl_ms: f64,
    pub percentiles_itl_ms: Vec<(f64, f64)>,
    pub mean_e2el_ms: f64,
    pub median_e2el_ms: f64,
    pub std_e2el_ms: f64,
    pub percentiles_e2el_ms: Vec<(f64, f64)>,
    pub max_output_tokens_per_s: f64,
    pub max_concurrent_requests: usize,
    #[serde(default)]
    pub steady_state: Option<SteadyStateMetrics>,
}
