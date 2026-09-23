// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Multimodal preprocessing spans and their timing collector configuration.

use tracing::{Span, info_span};
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::{Layer, Registry};
use vllm_tracing::timing::{RequestTimingLayer, RequestTimingStats};

/// Target of the multimodal preprocessing stage spans.
pub(super) const MM_STAGE_TARGET: &str = "mm_processor_timing";

/// Create the timing layer and its stats handle for multimodal preprocessing
/// stage spans (`vllm-bench mm-processor`), mirroring the Python
/// `TimingContext` / `MultiModalTimingRegistry`.
pub fn mm_timing_layer() -> (
    impl Layer<Registry> + Send + Sync + 'static,
    RequestTimingStats,
) {
    let (layer, stats) = RequestTimingLayer::new(MM_STAGE_TARGET);
    (layer.with_filter(LevelFilter::INFO), stats)
}

/// Span carrying the `request_id` used to attribute multimodal stage timings.
pub fn mm_request_span(request_id: &str) -> Span {
    info_span!("mm_request", request_id)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use tracing::{debug, debug_span};
    use tracing_subscriber::layer::SubscriberExt as _;

    use super::*;

    #[test]
    fn timing_layer_does_not_enable_debug_at_info() {
        let (timing_layer, _) = mm_timing_layer();
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
    fn timing_layer_does_not_disable_explicit_debug_logging() {
        let (timing_layer, _) = mm_timing_layer();
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
}
