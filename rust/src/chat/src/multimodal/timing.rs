// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Request-scoped multimodal preprocessing timing.
//!
//! Mirrors the Python frontend's `TimingContext` / `MultiModalTimingRegistry`
//! (`vllm/multimodal/processing/context.py`, `vllm/multimodal/registry.py`) so
//! the Rust chat frontend can expose the same per-stage preprocessing latencies
//! that `vllm bench mm-processor` reports for the Python renderer.
//!
//! Recording is opt-in: a disabled context turns [`TimingContext::record`]
//! into a no-op, and a disabled registry returns an empty map from
//! [`MultiModalTimingRegistry::stat`], matching the Python `enabled` semantics.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

/// Per-request stats dictionary returned by [`TimingContext::stats_dict`] and
/// [`MultiModalTimingRegistry::stat`]: `"{stage}_secs" => seconds`.
pub type StageStats = HashMap<String, f64>;

/// Shared no-op context returned by a disabled registry so call sites can
/// invoke [`TimingContext::record`] unconditionally.
static DISABLED_CONTEXT: LazyLock<Arc<TimingContext>> =
    LazyLock::new(|| Arc::new(TimingContext::disabled()));

/// Per-stage timings recorded for one multimodal request.
///
/// Stage times live behind a [`Mutex`] that is only locked when a
/// [`StageTimer`] is dropped, so individual stages can span `await` points
/// without pinning the lock for their whole duration.
pub struct TimingContext {
    enabled: bool,
    stage_secs: Mutex<HashMap<&'static str, Duration>>,
}

impl TimingContext {
    /// Create an enabled timing context.
    pub fn new() -> Self {
        Self {
            enabled: true,
            stage_secs: Mutex::new(HashMap::new()),
        }
    }

    /// Create a disabled context whose [`TimingContext::record`] calls are
    /// no-ops.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            stage_secs: Mutex::new(HashMap::new()),
        }
    }

    /// Whether recording is enabled for this context.
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Start timing `stage`; the elapsed wall-clock time is accumulated when
    /// the returned [`StageTimer`] is dropped.
    pub fn record(&self, stage: &'static str) -> StageTimer<'_> {
        StageTimer {
            ctx: self,
            stage,
            start: self.enabled.then(Instant::now),
        }
    }

    /// Total accumulated time across all recorded stages, in seconds.
    pub fn total_secs(&self) -> f64 {
        self.stage_secs.lock().unwrap().values().map(Duration::as_secs_f64).sum()
    }

    /// Snapshot the recorded stages keyed by `"{stage}_secs"`, matching
    /// Python's `TimingContext.get_stats_dict`.
    pub fn stats_dict(&self) -> StageStats {
        self.stage_secs
            .lock()
            .unwrap()
            .iter()
            .map(|(stage, duration)| (format!("{stage}_secs"), duration.as_secs_f64()))
            .collect()
    }
}

impl Default for TimingContext {
    fn default() -> Self {
        Self::new()
    }
}

/// RAII guard that records the elapsed time of a [`TimingContext::record`]
/// scope into its owning context when dropped.
pub struct StageTimer<'a> {
    ctx: &'a TimingContext,
    stage: &'static str,
    start: Option<Instant>,
}

impl Drop for StageTimer<'_> {
    fn drop(&mut self) {
        if let Some(start) = self.start {
            let elapsed = start.elapsed();
            *self.ctx.stage_secs.lock().unwrap().entry(self.stage).or_default() += elapsed;
        }
    }
}

/// Request-id-keyed registry of [`TimingContext`]s, mirroring Python's
/// `MultiModalTimingRegistry` (`vllm/multimodal/registry.py`).
///
/// [`MultiModalTimingRegistry::stat`] drains and returns every recorded
/// request's stats at once, matching the Python clear-on-read behavior so a
/// benchmark can read a batch and immediately start a clean one.
pub struct MultiModalTimingRegistry {
    enabled: bool,
    contexts: Mutex<HashMap<String, Arc<TimingContext>>>,
}

impl MultiModalTimingRegistry {
    /// Create a registry; `enabled` mirrors `enable_mm_processor_stats`.
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            contexts: Mutex::new(HashMap::new()),
        }
    }

    /// Whether the registry records new timings.
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Return the [`TimingContext`] for `request_id`, creating an enabled one
    /// on first access. When disabled, returns a shared no-op context.
    pub fn context(&self, request_id: &str) -> Arc<TimingContext> {
        if !self.enabled {
            return Arc::clone(&DISABLED_CONTEXT);
        }
        self.contexts
            .lock()
            .unwrap()
            .entry(request_id.to_string())
            .or_insert_with(|| Arc::new(TimingContext::new()))
            .clone()
    }

    /// Drain and return `{request_id: {stage_secs}}` records, clearing the
    /// registry. Returns an empty map when disabled.
    pub fn stat(&self) -> HashMap<String, StageStats> {
        if !self.enabled {
            return HashMap::new();
        }
        self.contexts
            .lock()
            .unwrap()
            .drain()
            .map(|(request_id, ctx)| (request_id, ctx.stats_dict()))
            .collect()
    }
}

impl Default for MultiModalTimingRegistry {
    fn default() -> Self {
        Self::new(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_context_record_is_noop() {
        let ctx = TimingContext::disabled();
        {
            let _timer = ctx.record("media_fetch");
        }
        assert!(ctx.stats_dict().is_empty());
        assert_eq!(ctx.total_secs(), 0.0);
    }

    #[test]
    fn enabled_context_accumulates_stage() {
        let ctx = TimingContext::new();
        {
            let _timer = ctx.record("media_fetch");
        }
        let stats = ctx.stats_dict();
        assert!(stats.contains_key("media_fetch_secs"));
        assert!(stats["media_fetch_secs"] >= 0.0);
        assert!(ctx.total_secs() >= stats["media_fetch_secs"]);
    }

    #[test]
    fn registry_stat_drains() {
        let registry = MultiModalTimingRegistry::new(true);
        {
            let ctx = registry.context("req-1");
            let _timer = ctx.record("media_fetch");
        }
        let stats = registry.stat();
        assert!(stats.contains_key("req-1"));
        assert!(stats["req-1"].contains_key("media_fetch_secs"));
        // `stat` drains: a second read is empty.
        assert!(registry.stat().is_empty());
    }

    #[test]
    fn disabled_registry_is_noop() {
        let registry = MultiModalTimingRegistry::new(false);
        assert!(registry.stat().is_empty());
        let ctx = registry.context("req-1");
        assert!(!ctx.enabled());
        {
            let _timer = ctx.record("media_fetch");
        }
        assert!(ctx.stats_dict().is_empty());
        assert!(registry.stat().is_empty());
    }
}
