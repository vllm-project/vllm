// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The request accessors the KV cache stack reads (`vllm.v1.request.Request`).

use super::events::EventExtraKeys;
use crate::hash::BlockHash;

/// The view of a request that the KV cache manager needs.
///
/// Implemented by the engine's request type and by the Python binding layer.
/// Token counts are non-negative Python ints.
pub trait KvRequest {
    fn request_id(&self) -> &str;

    /// `request.block_hashes`: chained prefix hashes at `hash_block_size`
    /// granularity.
    fn block_hashes(&self) -> &[BlockHash];

    /// `request.num_tokens`: prompt plus output tokens (including unverified
    /// draft tokens).
    fn num_tokens(&self) -> usize;

    fn num_prompt_tokens(&self) -> usize;

    fn num_computed_tokens(&self) -> usize;

    /// `request.num_in_flight_tokens`: tokens scheduled but not yet processed.
    fn num_in_flight_tokens(&self) -> usize;

    fn num_preemptions(&self) -> usize;

    /// `request.shared_prefix_boundary` (0 when none).
    fn shared_prefix_boundary(&self) -> usize;

    /// `request.prefix_cache_hit_limit`: number of leading tokens that may be
    /// read from the prefix cache, or `None` when unbounded.
    fn prefix_cache_hit_limit(&self) -> Option<usize>;

    /// `request.status in (RequestStatus.WAITING, RequestStatus.PREEMPTED)`.
    fn is_waiting_or_preempted(&self) -> bool;

    /// Longest prefix cache hit allowed, optionally tightened by bound.
    fn get_max_cache_hit_length(&self, bound: Option<usize>) -> usize {
        // The last token is always recomputed to produce logits.
        // Python would go negative for an empty request; this saturates at 0.
        let mut limit = self.num_tokens().saturating_sub(1);
        if let Some(l) = self.prefix_cache_hit_limit() {
            limit = limit.min(l);
        }
        if let Some(b) = bound {
            limit = limit.min(b);
        }
        limit
    }

    // ---- KV events only. Not called unless KV cache events are enabled. ----

    /// `request.kv_cache_report_mode == "full"`.
    fn kv_cache_report_mode_full(&self) -> bool {
        false
    }

    /// `request.all_token_ids[start:end]` (Python slice semantics: clamped to
    /// the number of tokens).
    fn token_ids(&self, start: usize, end: usize) -> Vec<i32>;

    /// `request.lora_request.adapter_id`, if any.
    fn lora_id(&self) -> Option<i64> {
        None
    }

    /// `request.lora_request.name`, if any.
    fn lora_name(&self) -> Option<String> {
        None
    }

    fn session_id(&self) -> Option<String> {
        None
    }

    /// `generate_block_hash_extra_keys(request, start, end, start_mm_idx)`,
    /// with the keys converted to the untagged event format (one entry of
    /// `to_event_extra_keys`).
    ///
    /// Returns the event keys of the block (`None` without extra keys) and the
    /// next multi-modal index.
    // TODO: generate the keys in Rust once multi-modal features, LoRA requests
    // and prompt embeds are part of the Rust request.
    fn event_extra_keys(
        &self,
        start: usize,
        end: usize,
        start_mm_idx: isize,
    ) -> (Option<EventExtraKeys>, isize) {
        let _ = (start, end);
        (None, start_mm_idx)
    }

    /// `len(request.mm_features)`.
    fn num_mm_features(&self) -> usize {
        0
    }

    /// `(offset, length)` of `request.mm_features[idx].mm_position`. Features
    /// are sorted by offset and non-overlapping.
    fn mm_feature_range(&self, idx: usize) -> (usize, usize) {
        let _ = idx;
        unreachable!("request has no multi-modal features")
    }
}

/// `get_mm_features_in_window(request.mm_features, start, end)[0]`: the index
/// of the first feature ending after `start`.
pub(crate) fn mm_features_in_window_lo(request: &impl KvRequest, start: usize) -> usize {
    // bisect_left on `offset + length` for `start + 1`.
    let (mut lo, mut hi) = (0, request.num_mm_features());
    while lo < hi {
        let mid = (lo + hi) / 2;
        let (offset, length) = request.mm_feature_range(mid);
        if offset + length < start + 1 {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}
