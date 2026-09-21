// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Request-scoped multimodal preprocessing timing.
//! Mirrors the Python `TimingContext` / `MultiModalTimingRegistry`.
//!
//! The preprocessing pipeline only emits `tracing` spans; aggregation lives in
//! [`RequestTimingLayer`] from `vllm-tracing`, which attributes each stage
//! span to the nearest ancestor span carrying a `request_id` field. Callers
//! that want timings (e.g. `vllm-bench mm-processor`) install the layer on
//! their subscriber; everyone else pays only the cost of untracked spans.

use tracing::info_span;
use tracing::span::Span;

pub use vllm_tracing::timing::{RequestTimingLayer, RequestTimingStats, StageStats};

/// Target of the per-stage timing spans.
const STAGE_TARGET: &str = "mm_processor_timing";

/// Create the timing layer and its stats handle for multimodal stage spans.
pub fn mm_timing_layer() -> (RequestTimingLayer, RequestTimingStats) {
    RequestTimingLayer::new(STAGE_TARGET)
}

/// Span carrying the `request_id` used to attribute stage timings.
pub fn mm_request_span(request_id: &str) -> Span {
    info_span!("mm_request", request_id)
}

/// Span for one preprocessing stage; the elapsed time is recorded on close.
pub(crate) fn mm_stage_span(stage: &'static str) -> Span {
    info_span!(target: STAGE_TARGET, "mm_stage", stage)
}
