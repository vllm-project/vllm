use std::fmt::Write;
use std::time::{Duration, Instant};

use tokio_util::task::AbortOnDropHandle;
use tracing::{debug, info};
use vllm_metrics::{
    EngineLabels, F64Gauge, METRICS, PromptTokenSourceLabels, SchedulerLogStatsAccumulator,
    SchedulerLogStatsInterval, U64Counter, U64Gauge, WaitingReasonLabels,
};

const LOG_STATS_INTERVAL: Duration = Duration::from_secs(10);
const WAITING_REASON_DEFERRED: &str = "deferred";

/// Cached, cloned metric handles for one engine. Each clone shares the same
/// underlying `Arc<Atomic*>` as the prometheus `Family` entry, so reads go
/// straight to the atomic with no lock.
struct EngineMetrics {
    // Counters for throughput deltas.
    prompt_tokens_computed: U64Counter,
    generation_tokens: U64Counter,
    prefix_cache_queries: U64Counter,
    prefix_cache_hits: U64Counter,
    external_prefix_cache_queries: U64Counter,
    external_prefix_cache_hits: U64Counter,
    num_preemptions: U64Counter,
    spec_decode_num_drafts: U64Counter,
    spec_decode_num_draft_tokens: U64Counter,
    spec_decode_num_accepted_tokens: U64Counter,
    estimated_flops_per_gpu: U64Counter,
    estimated_read_bytes_per_gpu: U64Counter,
    estimated_write_bytes_per_gpu: U64Counter,
    log_stats: SchedulerLogStatsAccumulator,

    // Gauges for instantaneous scheduler state.
    scheduler_running: U64Gauge,
    scheduler_waiting: U64Gauge,
    scheduler_deferred: U64Gauge,
    kv_cache_usage: F64Gauge,
}

/// Accumulated snapshot values from the last logging interval, used to compute
/// deltas.
#[derive(Default)]
struct CounterSnapshot {
    prompt_tokens: u64,
    generation_tokens: u64,
    prefix_cache_queries: u64,
    prefix_cache_hits: u64,
    external_prefix_cache_queries: u64,
    external_prefix_cache_hits: u64,
    num_preemptions: u64,
    spec_decode_num_drafts: u64,
    spec_decode_num_draft_tokens: u64,
    spec_decode_num_accepted_tokens: u64,
    estimated_flops_per_gpu: u64,
    estimated_read_bytes_per_gpu: u64,
    estimated_write_bytes_per_gpu: u64,
}

/// Derived spec-decoding values for one logging interval.
struct SpecDecodingLogStats {
    mean_acceptance_length: f64,
    accepted_throughput: f64,
    draft_throughput: f64,
    accepted_tokens: u64,
    draft_tokens: u64,
    per_position_acceptance_rates: Vec<f64>,
    draft_acceptance_rate: f64,
}

/// Derived MFU values for one logging interval.
struct MfuLogStats {
    tflops_per_gpu: f64,
    gbps_per_gpu: f64,
}

/// Periodic stats logger that mirrors Python vLLM's `LoggingStatLogger`.
///
/// Spawns a background task that logs throughput and scheduler state at a fixed
/// interval. When idle (both current and previous throughputs are zero), logs
/// at DEBUG level. When load drops to zero, emits one final INFO-level line
/// before going quiet.
pub(crate) struct StatsLogger {
    _task: AbortOnDropHandle<()>,
}

impl StatsLogger {
    /// Start the background stats logging task.
    pub(crate) fn start(model_name: String, engine_indices: Vec<u32>) -> Self {
        let task = AbortOnDropHandle::new(tokio::spawn(async move {
            run_stats_logger(model_name, engine_indices).await;
        }));
        Self { _task: task }
    }
}

/// Resolve and clone all metric handles once so the hot path is lock-free.
fn resolve_engine_metrics(model_name: &str, engine_indices: &[u32]) -> Vec<EngineMetrics> {
    let m = &METRICS;
    engine_indices
        .iter()
        .copied()
        .map(|engine| {
            let el = EngineLabels {
                model_name: model_name.to_string(),
                engine,
            };
            let pt = PromptTokenSourceLabels {
                model_name: model_name.to_string(),
                engine,
                source: "local_compute",
            };
            let deferred = WaitingReasonLabels {
                model_name: model_name.to_string(),
                engine,
                reason: WAITING_REASON_DEFERRED,
            };
            EngineMetrics {
                // Use "local_compute" source for prompt throughput (excludes
                // cached/transferred tokens), matching Python's
                // `iteration_stats.prompt_token_stats.computed`.
                prompt_tokens_computed: m.request.prompt_tokens_by_source.get_or_create_owned(&pt),
                generation_tokens: m.request.generation_tokens.get_or_create_owned(&el),
                prefix_cache_queries: m.scheduler.prefix_cache_queries.get_or_create_owned(&el),
                prefix_cache_hits: m.scheduler.prefix_cache_hits.get_or_create_owned(&el),
                external_prefix_cache_queries: m
                    .scheduler
                    .external_prefix_cache_queries
                    .get_or_create_owned(&el),
                external_prefix_cache_hits: m
                    .scheduler
                    .external_prefix_cache_hits
                    .get_or_create_owned(&el),
                num_preemptions: m.request.num_preemptions.get_or_create_owned(&el),
                spec_decode_num_drafts: m.scheduler.spec_decode_num_drafts.get_or_create_owned(&el),
                spec_decode_num_draft_tokens: m
                    .scheduler
                    .spec_decode_num_draft_tokens
                    .get_or_create_owned(&el),
                spec_decode_num_accepted_tokens: m
                    .scheduler
                    .spec_decode_num_accepted_tokens
                    .get_or_create_owned(&el),
                estimated_flops_per_gpu: m
                    .scheduler
                    .estimated_flops_per_gpu
                    .get_or_create_owned(&el),
                estimated_read_bytes_per_gpu: m
                    .scheduler
                    .estimated_read_bytes_per_gpu
                    .get_or_create_owned(&el),
                estimated_write_bytes_per_gpu: m
                    .scheduler
                    .estimated_write_bytes_per_gpu
                    .get_or_create_owned(&el),
                log_stats: m.scheduler.log_stats.get_or_create_owned(&el),
                scheduler_running: m.scheduler.scheduler_running.get_or_create_owned(&el),
                scheduler_waiting: m.scheduler.scheduler_waiting.get_or_create_owned(&el),
                scheduler_deferred: m
                    .scheduler
                    .scheduler_waiting_by_reason
                    .get_or_create_owned(&deferred),
                kv_cache_usage: m.scheduler.kv_cache_usage.get_or_create_owned(&el),
            }
        })
        .collect()
}

async fn run_stats_logger(model_name: String, engine_indices: Vec<u32>) {
    let engines = resolve_engine_metrics(&model_name, &engine_indices);

    let mut interval = tokio::time::interval(LOG_STATS_INTERVAL);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    // The first tick fires immediately; skip it so the first log is after one full
    // interval.
    interval.tick().await;

    let mut prev = read_counters(&engines);
    let mut last_log_time = Instant::now();
    let mut last_prompt_throughput: f64 = 0.0;
    let mut last_generation_throughput: f64 = 0.0;

    let mut msg = String::new();
    loop {
        interval.tick().await;

        let now = Instant::now();
        let elapsed = now.duration_since(last_log_time).as_secs_f64();
        if elapsed <= 0.0 {
            continue;
        }

        let curr = read_counters(&engines);
        let raw_log_stats = drain_scheduler_log_stats(&engines);

        let prompt_throughput =
            curr.prompt_tokens.wrapping_sub(prev.prompt_tokens) as f64 / elapsed;
        let generation_throughput =
            curr.generation_tokens.wrapping_sub(prev.generation_tokens) as f64 / elapsed;

        // Idle = both current and previous throughputs are zero.
        let is_idle = prompt_throughput == 0.0
            && generation_throughput == 0.0
            && last_prompt_throughput == 0.0
            && last_generation_throughput == 0.0;

        /// Emit one stats line at DEBUG while idle and INFO while active.
        macro_rules! log_stats_line {
            ($($arg:tt)*) => {
                if is_idle {
                    debug!($($arg)*);
                } else {
                    info!($($arg)*);
                }
            };
        }

        // Read scheduler gauges (aggregate across engines).
        let (num_running, num_waiting, kv_cache_usage) = read_scheduler_gauges(&engines);
        let num_deferred = read_deferred_waiting(&engines);
        let delta_preemptions = curr.num_preemptions.wrapping_sub(prev.num_preemptions);

        // Compute prefix cache hit rate over this interval.
        let delta_queries = curr.prefix_cache_queries.wrapping_sub(prev.prefix_cache_queries);
        let delta_hits = curr.prefix_cache_hits.wrapping_sub(prev.prefix_cache_hits);
        let prefix_cache_hit_rate = cache_hit_rate(delta_hits, delta_queries);

        let delta_external_queries = curr
            .external_prefix_cache_queries
            .wrapping_sub(prev.external_prefix_cache_queries);
        let delta_external_hits =
            curr.external_prefix_cache_hits.wrapping_sub(prev.external_prefix_cache_hits);
        let external_prefix_cache_hit_rate =
            cache_hit_rate(delta_external_hits, delta_external_queries);
        let spec_decoding_log_stats =
            spec_decoding_log_stats(&curr, &prev, elapsed, &raw_log_stats);
        let mfu_log_stats = mfu_log_stats(&curr, &prev, elapsed, engines.len());

        // Build the log line.
        msg.clear();
        write!(
            msg,
            "Avg prompt tput: {prompt_throughput:.1} toks/s, \
             Avg generation tput: {generation_throughput:.1} toks/s, \
             Reqs Running: {num_running}, \
             Waiting: {num_waiting}"
        )
        .unwrap();
        if num_deferred > 0 {
            write!(msg, ", Deferred: {num_deferred} reqs").unwrap();
        }
        if delta_preemptions > 0 {
            write!(msg, ", Preemptions: {delta_preemptions}").unwrap();
        }
        write!(
            msg,
            ", GPU KV cache used: {:.1}%, \
             Prefix cache hit rate: {prefix_cache_hit_rate:.1}%",
            kv_cache_usage * 100.0,
        )
        .unwrap();
        if delta_external_queries > 0 {
            write!(
                msg,
                ", External prefix cache hit rate: {external_prefix_cache_hit_rate:.1}%"
            )
            .unwrap();
        }

        log_stats_line!("{msg}");

        if let Some(spec_stats) = spec_decoding_log_stats {
            msg.clear();
            write!(
                msg,
                "SpecDecoding metrics: \
                 Mean acceptance length: {:.2}, \
                 Accepted throughput: {:.2} tokens/s, \
                 Drafted throughput: {:.2} tokens/s, \
                 Accepted: {} tokens, \
                 Drafted: {} tokens",
                spec_stats.mean_acceptance_length,
                spec_stats.accepted_throughput,
                spec_stats.draft_throughput,
                spec_stats.accepted_tokens,
                spec_stats.draft_tokens,
            )
            .unwrap();
            if !spec_stats.per_position_acceptance_rates.is_empty() {
                msg.push_str(", Per-position acceptance rate: ");
                format_position_rates(&mut msg, &spec_stats.per_position_acceptance_rates);
            }
            write!(
                msg,
                ", Avg Draft acceptance rate: {:.1}%",
                spec_stats.draft_acceptance_rate,
            )
            .unwrap();
            log_stats_line!("{msg}");
        }

        // TODO: Decide on best way to surface CUDAGraph interval samples.

        if let Some(mfu_stats) = mfu_log_stats {
            log_stats_line!(
                "MFU: {:.1} TF/s/GPU {:.1} GB/s/GPU",
                mfu_stats.tflops_per_gpu,
                mfu_stats.gbps_per_gpu,
            );
        }

        last_prompt_throughput = prompt_throughput;
        last_generation_throughput = generation_throughput;
        last_log_time = now;
        prev = curr;
    }
}

/// Read the current cumulative counter values for throughput computation.
fn read_counters(engines: &[EngineMetrics]) -> CounterSnapshot {
    let mut snap = CounterSnapshot::default();
    for e in engines {
        snap.prompt_tokens += e.prompt_tokens_computed.get();
        snap.generation_tokens += e.generation_tokens.get();
        snap.prefix_cache_queries += e.prefix_cache_queries.get();
        snap.prefix_cache_hits += e.prefix_cache_hits.get();
        snap.external_prefix_cache_queries += e.external_prefix_cache_queries.get();
        snap.external_prefix_cache_hits += e.external_prefix_cache_hits.get();
        snap.num_preemptions += e.num_preemptions.get();
        snap.spec_decode_num_drafts += e.spec_decode_num_drafts.get();
        snap.spec_decode_num_draft_tokens += e.spec_decode_num_draft_tokens.get();
        snap.spec_decode_num_accepted_tokens += e.spec_decode_num_accepted_tokens.get();
        snap.estimated_flops_per_gpu += e.estimated_flops_per_gpu.get();
        snap.estimated_read_bytes_per_gpu += e.estimated_read_bytes_per_gpu.get();
        snap.estimated_write_bytes_per_gpu += e.estimated_write_bytes_per_gpu.get();
    }
    snap
}

/// Read the current scheduler gauge values, aggregated across engines.
fn read_scheduler_gauges(engines: &[EngineMetrics]) -> (u64, u64, f64) {
    let mut num_running = 0u64;
    let mut num_waiting = 0u64;
    let mut kv_cache_usage_sum = 0.0f64;

    for e in engines {
        num_running += e.scheduler_running.get();
        num_waiting += e.scheduler_waiting.get();
        kv_cache_usage_sum += e.kv_cache_usage.get();
    }

    let kv_cache_usage = if !engines.is_empty() {
        kv_cache_usage_sum / engines.len() as f64
    } else {
        0.0
    };

    (num_running, num_waiting, kv_cache_usage)
}

/// Read deferred waiting requests across all engines.
fn read_deferred_waiting(engines: &[EngineMetrics]) -> u64 {
    engines.iter().map(|e| e.scheduler_deferred.get()).sum()
}

/// Return the cache hit rate as a percentage for a counter delta.
fn cache_hit_rate(hits: u64, queries: u64) -> f64 {
    if queries > 0 {
        hits as f64 / queries as f64 * 100.0
    } else {
        0.0
    }
}

/// Compute aggregate spec-decoding stats for one logging interval.
fn spec_decoding_log_stats(
    curr: &CounterSnapshot,
    prev: &CounterSnapshot,
    elapsed: f64,
    raw_log_stats: &SchedulerLogStatsInterval,
) -> Option<SpecDecodingLogStats> {
    let num_drafts = curr.spec_decode_num_drafts.wrapping_sub(prev.spec_decode_num_drafts);
    if num_drafts == 0 {
        return None;
    }

    let draft_tokens = curr
        .spec_decode_num_draft_tokens
        .wrapping_sub(prev.spec_decode_num_draft_tokens);
    let accepted_tokens = curr
        .spec_decode_num_accepted_tokens
        .wrapping_sub(prev.spec_decode_num_accepted_tokens);

    let (accepted_throughput, draft_throughput) = if elapsed > 0.0 {
        (
            accepted_tokens as f64 / elapsed,
            draft_tokens as f64 / elapsed,
        )
    } else {
        (0.0, 0.0)
    };
    let draft_acceptance_rate = if draft_tokens > 0 {
        accepted_tokens as f64 / draft_tokens as f64 * 100.0
    } else {
        f64::NAN
    };
    let per_position_acceptance_rates = if raw_log_stats.spec_num_drafts > 0 {
        raw_log_stats
            .spec_accepted_tokens_per_pos
            .iter()
            .map(|accepted_tokens| *accepted_tokens as f64 / raw_log_stats.spec_num_drafts as f64)
            .collect()
    } else {
        Vec::new()
    };

    Some(SpecDecodingLogStats {
        mean_acceptance_length: 1.0 + accepted_tokens as f64 / num_drafts as f64,
        accepted_throughput,
        draft_throughput,
        accepted_tokens,
        draft_tokens,
        per_position_acceptance_rates,
        draft_acceptance_rate,
    })
}

/// Compute average per-GPU MFU rates for one logging interval.
fn mfu_log_stats(
    curr: &CounterSnapshot,
    prev: &CounterSnapshot,
    elapsed: f64,
    engine_count: usize,
) -> Option<MfuLogStats> {
    let flops = curr.estimated_flops_per_gpu.wrapping_sub(prev.estimated_flops_per_gpu);
    let read_bytes = curr
        .estimated_read_bytes_per_gpu
        .wrapping_sub(prev.estimated_read_bytes_per_gpu);
    let write_bytes = curr
        .estimated_write_bytes_per_gpu
        .wrapping_sub(prev.estimated_write_bytes_per_gpu);

    if flops == 0 && read_bytes == 0 && write_bytes == 0 {
        return None;
    }

    let denominator = elapsed * engine_count.max(1) as f64;
    let (tflops_per_gpu, gbps_per_gpu) = if denominator > 0.0 {
        (
            flops as f64 / denominator / 1e12,
            (read_bytes as f64 + write_bytes as f64) / denominator / 1e9,
        )
    } else {
        (0.0, 0.0)
    };

    Some(MfuLogStats {
        tflops_per_gpu,
        gbps_per_gpu,
    })
}

/// Drain raw scheduler DTO stats for the configured model and engines.
fn drain_scheduler_log_stats(engines: &[EngineMetrics]) -> SchedulerLogStatsInterval {
    let mut interval = SchedulerLogStatsInterval::default();
    for engine in engines {
        interval.merge(engine.log_stats.drain());
    }
    interval
}

/// Append spec-decoding per-position acceptance rates like Python's logger.
fn format_position_rates(output: &mut String, rates: &[f64]) {
    for (position, rate) in rates.iter().enumerate() {
        if position > 0 {
            output.push_str(", ");
        }
        write!(output, "{rate:.3}").unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_hit_rate_returns_percent_for_non_empty_queries() {
        assert_eq!(cache_hit_rate(25, 100), 25.0);
        assert_eq!(cache_hit_rate(0, 0), 0.0);
    }

    #[test]
    fn spec_decoding_log_stats_uses_interval_deltas() {
        let raw_log_stats = SchedulerLogStatsInterval {
            spec_num_drafts: 4,
            spec_accepted_tokens_per_pos: vec![4, 2, 1],
            ..Default::default()
        };
        let prev = CounterSnapshot {
            spec_decode_num_drafts: 10,
            spec_decode_num_draft_tokens: 100,
            spec_decode_num_accepted_tokens: 40,
            ..Default::default()
        };
        let curr = CounterSnapshot {
            spec_decode_num_drafts: 14,
            spec_decode_num_draft_tokens: 120,
            spec_decode_num_accepted_tokens: 52,
            ..Default::default()
        };

        let stats = spec_decoding_log_stats(&curr, &prev, 2.0, &raw_log_stats).unwrap();

        assert_eq!(stats.mean_acceptance_length, 4.0);
        assert_eq!(stats.accepted_throughput, 6.0);
        assert_eq!(stats.draft_throughput, 10.0);
        assert_eq!(stats.accepted_tokens, 12);
        assert_eq!(stats.draft_tokens, 20);
        assert_eq!(stats.per_position_acceptance_rates, vec![1.0, 0.5, 0.25]);
        assert_eq!(stats.draft_acceptance_rate, 60.0);
    }

    #[test]
    fn mfu_log_stats_averages_per_gpu_across_engines() {
        let prev = CounterSnapshot {
            estimated_flops_per_gpu: 10,
            estimated_read_bytes_per_gpu: 10,
            estimated_write_bytes_per_gpu: 10,
            ..Default::default()
        };
        let curr = CounterSnapshot {
            estimated_flops_per_gpu: 4_000_000_000_010,
            estimated_read_bytes_per_gpu: 2_000_000_010,
            estimated_write_bytes_per_gpu: 2_000_000_010,
            ..Default::default()
        };

        let stats = mfu_log_stats(&curr, &prev, 2.0, 2).unwrap();

        assert_eq!(stats.tflops_per_gpu, 1.0);
        assert_eq!(stats.gbps_per_gpu, 1.0);
    }

    #[test]
    fn format_position_rates_uses_three_decimal_places() {
        let mut output = String::new();

        format_position_rates(&mut output, &[1.0, 0.5, 0.25]);

        assert_eq!(output, "1.000, 0.500, 0.250");
    }
}
