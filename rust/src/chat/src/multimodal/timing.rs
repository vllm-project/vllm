// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Multimodal preprocessing spans and their timing collector configuration.

use tracing::{Span, info_span};
use vllm_tracing::timing::{RequestTimingLayer, RequestTimingStats};

/// Target of the multimodal preprocessing stage spans.
pub(super) const MM_STAGE_TARGET: &str = "mm_processor_timing";

/// Create the timing layer and its stats handle for multimodal preprocessing
/// stage spans (`vllm-bench mm-processor`), mirroring the Python
/// `TimingContext` / `MultiModalTimingRegistry`.
pub fn mm_timing_layer() -> (RequestTimingLayer, RequestTimingStats) {
    RequestTimingLayer::new(MM_STAGE_TARGET)
}

/// Span carrying the `request_id` used to attribute multimodal stage timings.
pub fn mm_request_span(request_id: &str) -> Span {
    info_span!("mm_request", request_id)
}
