// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Request-scoped span timing.
//!
//! Pipelines emit stage spans (target = the layer's configured target, a
//! `stage` field naming the stage) inside a request span carrying a
//! `request_id` field; [`RequestTimingLayer`] attributes each stage span's
//! wall-clock time to the nearest ancestor `request_id` and accumulates per
//! request. Callers that want timings install the layer on their subscriber;
//! everyone else pays only the cost of untracked spans.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use tracing::field::{Field, Visit};
use tracing::span::Attributes;
use tracing::{Id, Subscriber};
use tracing_subscriber::filter::{Filtered, LevelFilter};
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::registry::LookupSpan;

/// `"{stage}_secs" => seconds` maps returned by [`RequestTimingStats::stat`].
pub type StageStats = HashMap<String, f64>;

/// `request_id` extracted from a span, propagated to stage timings.
struct RequestId(String);

/// Start timestamp of one in-flight stage span.
struct StageStart {
    stage: String,
    start: Instant,
}

/// Extracts the fields the layer cares about from span attributes.
#[derive(Default)]
struct SpanFields {
    request_id: Option<String>,
    stage: Option<String>,
}

impl Visit for SpanFields {
    fn record_str(&mut self, field: &Field, value: &str) {
        match field.name() {
            "request_id" => self.request_id = Some(value.to_string()),
            "stage" => self.stage = Some(value.to_string()),
            _ => {}
        }
    }

    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        match field.name() {
            "request_id" => self.request_id = Some(format!("{value:?}")),
            "stage" => self.stage = Some(format!("{value:?}")),
            _ => {}
        }
    }
}

/// `tracing` layer that aggregates stage-span timings per request, keyed by
/// the nearest ancestor span with a `request_id` field.
///
/// # Span contract
///
/// Timed spans must have the configured target and a `stage` field. The
/// `request_id` is taken from the timed span itself or its nearest ancestor
/// carrying that field, regardless of the ancestor's target. Spans without
/// a request ID are ignored. Span names are unrestricted. Both the timed spans
/// and their `request_id` ancestors must be at `INFO` level or above.
///
/// Both fields are read at span creation; subsequent [`tracing::Span::record`]
/// calls are ignored. String values are used verbatim; other values use their
/// debug representation. Prefer string fields for stable keys.
///
/// # Timing semantics
///
/// Each sample measures wall-clock time from span creation to final close,
/// including async waits and any time the span remains alive through cloned
/// handles. Nested and concurrent spans are measured independently, so their
/// durations can overlap. Repeated stages accumulate under `"{stage}_secs"`
/// for each request. Closing spans are recorded regardless of the operation's
/// success, failure, or cancellation; callers select the samples to report.
pub struct RequestTimingLayer {
    target: &'static str,
    stats: Arc<Mutex<HashMap<String, StageStats>>>,
}

/// Drain handle for the timings collected by [`RequestTimingLayer`].
pub struct RequestTimingStats {
    stats: Arc<Mutex<HashMap<String, StageStats>>>,
}

impl RequestTimingLayer {
    /// Create the layer and its stats handle. Only spans whose target equals
    /// `target` and that carry a `stage` field are timed.
    ///
    /// The layer is filtered to `INFO` so that, composed with a filtered log
    /// layer, it does not raise the subscriber's max level hint and enable
    /// `DEBUG`/`TRACE` callsites that the log layer would discard anyway.
    pub fn new<S>(target: &'static str) -> (Filtered<Self, LevelFilter, S>, RequestTimingStats) {
        let stats = Arc::new(Mutex::new(HashMap::new()));
        let layer = Self {
            target,
            stats: Arc::clone(&stats),
        };
        (
            Filtered::new(layer, LevelFilter::INFO),
            RequestTimingStats { stats },
        )
    }
}

impl RequestTimingStats {
    /// Drain and return `{request_id: {stage_secs}}` records.
    ///
    /// Returns accumulated timings since the previous drain. Spans still open
    /// contribute to a later drain; finish the measured work before calling
    /// this method to collect complete request timings.
    pub fn stat(&self) -> HashMap<String, StageStats> {
        std::mem::take(&mut *self.stats.lock().unwrap())
    }
}

impl<S> Layer<S> for RequestTimingLayer
where
    S: Subscriber + for<'lookup> LookupSpan<'lookup>,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(id) else {
            return;
        };
        let mut fields = SpanFields::default();
        attrs.record(&mut fields);
        let mut extensions = span.extensions_mut();
        if let Some(request_id) = fields.request_id {
            extensions.insert(RequestId(request_id));
        }
        if span.metadata().target() == self.target
            && let Some(stage) = fields.stage
        {
            extensions.insert(StageStart {
                stage,
                start: Instant::now(),
            });
        }
    }

    fn on_close(&self, id: Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(&id) else {
            return;
        };
        let (stage, start) = {
            let extensions = span.extensions();
            let Some(stage_start) = extensions.get::<StageStart>() else {
                return;
            };
            (stage_start.stage.clone(), stage_start.start)
        };
        let Some(request_id) = span
            .scope()
            .find_map(|ancestor| ancestor.extensions().get::<RequestId>().map(|id| id.0.clone()))
        else {
            return;
        };
        let mut stats = self.stats.lock().unwrap();
        *stats.entry(request_id).or_default().entry(format!("{stage}_secs")).or_default() +=
            start.elapsed().as_secs_f64();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use tracing::{debug, debug_span, info_span};
    use tracing_subscriber::layer::SubscriberExt as _;

    const TARGET: &str = "test_request_timing";

    fn stage_span(stage: &'static str) -> tracing::Span {
        info_span!(target: TARGET, "stage", stage)
    }

    #[test]
    fn records_stage_timings_per_request() {
        let (layer, stats) = RequestTimingLayer::new(TARGET);
        let subscriber = tracing_subscriber::registry().with(layer);
        tracing::subscriber::with_default(subscriber, || {
            info_span!("request", request_id = "req-1").in_scope(|| {
                drop(stage_span("media_fetch"));
                stage_span("prompt_expansion").in_scope(|| {});
            });
        });
        let drained = stats.stat();
        assert!(drained["req-1"].contains_key("media_fetch_secs"));
        assert!(drained["req-1"].contains_key("prompt_expansion_secs"));
        // `stat` drains: a second read is empty.
        assert!(stats.stat().is_empty());
    }

    #[test]
    fn attributes_to_nearest_request_id_ancestor() {
        let (layer, stats) = RequestTimingLayer::new(TARGET);
        let subscriber = tracing_subscriber::registry().with(layer);
        tracing::subscriber::with_default(subscriber, || {
            info_span!("request", request_id = "req-outer").in_scope(|| {
                info_span!("request", request_id = "req-inner").in_scope(|| {
                    drop(stage_span("media_fetch"));
                });
            });
        });
        let drained = stats.stat();
        assert!(drained["req-inner"].contains_key("media_fetch_secs"));
        assert!(!drained.contains_key("req-outer"));
    }

    #[test]
    fn stage_without_request_id_is_dropped() {
        let (layer, stats) = RequestTimingLayer::new(TARGET);
        let subscriber = tracing_subscriber::registry().with(layer);
        tracing::subscriber::with_default(subscriber, || {
            drop(stage_span("media_fetch"));
        });
        assert!(stats.stat().is_empty());
    }

    #[test]
    fn does_not_enable_debug_at_info() {
        let (timing_layer, _) = RequestTimingLayer::new(TARGET);
        let info_layer = tracing_subscriber::fmt::layer()
            .with_writer(std::io::sink)
            .with_filter(LevelFilter::INFO);
        let subscriber = tracing_subscriber::registry().with(timing_layer).with(info_layer);
        let evaluated_debug_fields = AtomicUsize::new(0);

        tracing::subscriber::with_default(subscriber, || {
            assert_eq!(LevelFilter::current(), LevelFilter::INFO);
            debug!(
                value = evaluated_debug_fields.fetch_add(1, Ordering::Relaxed),
                "debug event"
            );
            assert!(debug_span!("debug span").is_disabled());
        });

        assert_eq!(evaluated_debug_fields.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn does_not_disable_explicit_debug_logging() {
        let (timing_layer, _) = RequestTimingLayer::new(TARGET);
        let debug_layer = tracing_subscriber::fmt::layer()
            .with_writer(std::io::sink)
            .with_filter(LevelFilter::DEBUG);
        let subscriber = tracing_subscriber::registry().with(timing_layer).with(debug_layer);
        let evaluated_debug_fields = AtomicUsize::new(0);

        tracing::subscriber::with_default(subscriber, || {
            assert_eq!(LevelFilter::current(), LevelFilter::DEBUG);
            debug!(
                value = evaluated_debug_fields.fetch_add(1, Ordering::Relaxed),
                "debug event"
            );
            assert!(!debug_span!("debug span").is_disabled());
        });

        assert_eq!(evaluated_debug_fields.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn records_stage_timings_when_logging_at_warn() {
        let (timing_layer, stats) = RequestTimingLayer::new(TARGET);
        let warn_layer = tracing_subscriber::fmt::layer()
            .with_writer(std::io::sink)
            .with_filter(LevelFilter::WARN);
        let subscriber = tracing_subscriber::registry().with(timing_layer).with(warn_layer);
        tracing::subscriber::with_default(subscriber, || {
            info_span!("request", request_id = "req-1").in_scope(|| {
                drop(stage_span("media_fetch"));
            });
        });
        assert!(stats.stat()["req-1"].contains_key("media_fetch_secs"));
    }

    #[test]
    fn other_targets_are_ignored() {
        let (layer, stats) = RequestTimingLayer::new(TARGET);
        let subscriber = tracing_subscriber::registry().with(layer);
        tracing::subscriber::with_default(subscriber, || {
            info_span!("request", request_id = "req-1").in_scope(|| {
                drop(info_span!("unrelated", stage = "media_fetch"));
            });
        });
        assert!(stats.stat().is_empty());
    }
}
