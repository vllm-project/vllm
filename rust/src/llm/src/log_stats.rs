use std::fmt::Write;
use std::time::{Duration, Instant};

use tokio_util::task::AbortOnDropHandle;
use tracing::{debug, info};
use vllm_metrics::{
    EngineLabels, F64Gauge, METRICS, PromptTokenSourceLabels, U64Counter, U64Gauge,
    WaitingReasonLabels,
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
}

/// Derived spec-decoding values for one logging interval.
struct SpecDecodingLogStats {
    mean_acceptance_length: f64,
    accepted_throughput: f64,
    draft_throughput: f64,
    accepted_tokens: u64,
    draft_tokens: u64,
    draft_acceptance_rate: f64,
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
    pub(crate) fn start(model_name: String, engine_count: usize) -> Self {
        let task = AbortOnDropHandle::new(tokio::spawn(async move {
            run_stats_logger(model_name, engine_count).await;
        }));
        Self { _task: task }
    }
}

/// Resolve and clone all metric handles once so the hot path is lock-free.
fn resolve_engine_metrics(model_name: &str, engine_count: usize) -> Vec<EngineMetrics> {
    let m = &METRICS;
    (0..engine_count as u32)
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

async fn run_stats_logger(model_name: String, engine_count: usize) {
    let engines = resolve_engine_metrics(&model_name, engine_count);

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
        let spec_decoding_log_stats = spec_decoding_log_stats(&curr, &prev, elapsed);

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
                 Drafted: {} tokens, \
                 Avg Draft acceptance rate: {:.1}%",
                spec_stats.mean_acceptance_length,
                spec_stats.accepted_throughput,
                spec_stats.draft_throughput,
                spec_stats.accepted_tokens,
                spec_stats.draft_tokens,
                spec_stats.draft_acceptance_rate,
            )
            .unwrap();
            log_stats_line!("{msg}");
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

    Some(SpecDecodingLogStats {
        mean_acceptance_length: 1.0 + accepted_tokens as f64 / num_drafts as f64,
        accepted_throughput,
        draft_throughput,
        accepted_tokens,
        draft_tokens,
        draft_acceptance_rate,
    })
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

        let stats = spec_decoding_log_stats(&curr, &prev, 2.0).unwrap();

        assert_eq!(stats.mean_acceptance_length, 4.0);
        assert_eq!(stats.accepted_throughput, 6.0);
        assert_eq!(stats.draft_throughput, 10.0);
        assert_eq!(stats.accepted_tokens, 12);
        assert_eq!(stats.draft_tokens, 20);
        assert_eq!(stats.draft_acceptance_rate, 60.0);
    }
}
