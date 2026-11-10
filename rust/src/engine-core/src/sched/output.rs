// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Port of `vllm/v1/core/sched/output.py`: what the scheduler produces each
//! step.
//!
//! Fields carrying opaque Python objects (sampling / pooling params, LoRA
//! request, prompt embeddings, multimodal data, connector metadata) are left
//! out; the binding layer resolves them from the request id.

use std::any::Any;
use std::fmt;
use std::sync::{Arc, OnceLock};

use indexmap::{IndexMap, IndexSet};
use rustc_hash::FxHashMap;

use super::request::Request;

/// Block IDs of a request, one list per KV cache group
/// (`tuple[list[int], ...]` in Python).
pub type GroupedBlockIds = Vec<Vec<u32>>;

/// A request scheduled for the first time.
///
/// Not mirrored (resolved by the binding layer from `req_id`):
/// `mm_features` (built with `strip_covered_mm_data` from the request's
/// features and `num_computed_tokens`), `sampling_params`, `pooling_params`,
/// `lora_request`, and `prompt_embeds`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NewRequestData {
    pub req_id: String,
    pub prompt_token_ids: Option<Vec<i32>>,
    pub block_ids: GroupedBlockIds,
    pub num_computed_tokens: usize,
    /// All the tokens to prefill: the prompt, plus any generated tokens on resume.
    pub prefill_token_ids: Vec<i32>,
    pub prompt_is_token_ids: Option<Vec<bool>>,
    /// DeepSeek-V4.1 only: SWA bounded replay; see `Request::replay_start`.
    pub replay_start: usize,
}

impl NewRequestData {
    pub fn from_request(
        request: &Request,
        block_ids: GroupedBlockIds,
        prefill_token_ids: Vec<i32>,
    ) -> Self {
        Self {
            req_id: request.request_id.clone(),
            prompt_token_ids: request.prompt_token_ids.clone(),
            block_ids,
            num_computed_tokens: request.num_computed_tokens,
            prompt_is_token_ids: request.prompt_is_token_ids.clone(),
            prefill_token_ids,
            replay_start: request.replay_start,
        }
    }

    /// The prompt length; Python falls back to `prompt_embeds.shape[0]`,
    /// which the binding layer must use when `prompt_token_ids` is `None`.
    pub fn prompt_len(&self) -> Option<usize> {
        self.prompt_token_ids.as_ref().map(Vec::len)
    }
}

#[derive(Clone, Debug, Default)]
pub struct CachedRequestData {
    pub req_ids: Vec<String>,
    // NOTE(woosuk): new_token_ids is only used for pipeline parallelism.
    // When PP is not used, new_token_ids will be empty.
    pub new_token_ids: Vec<Vec<i32>>,
    /// Blocks to append to the request's existing block IDs.
    pub new_block_ids: Vec<Option<GroupedBlockIds>>,
    pub num_computed_tokens: Vec<usize>,
    pub num_output_tokens: Vec<usize>,
    /// Cache mapping of req_id to num_output_tokens for O(1) lookup.
    ///
    /// This cached property is safe because CachedRequestData instances
    /// are created fresh each scheduling iteration and not mutated during
    /// computation of iteration details.
    req_id_to_num_output_tokens: OnceLock<FxHashMap<String, usize>>,
}

impl PartialEq for CachedRequestData {
    fn eq(&self, other: &Self) -> bool {
        self.req_ids == other.req_ids
            && self.new_token_ids == other.new_token_ids
            && self.new_block_ids == other.new_block_ids
            && self.num_computed_tokens == other.num_computed_tokens
            && self.num_output_tokens == other.num_output_tokens
    }
}

impl CachedRequestData {
    pub fn new(
        req_ids: Vec<String>,
        new_token_ids: Vec<Vec<i32>>,
        new_block_ids: Vec<Option<GroupedBlockIds>>,
        num_computed_tokens: Vec<usize>,
        num_output_tokens: Vec<usize>,
    ) -> Self {
        Self {
            req_ids,
            new_token_ids,
            new_block_ids,
            num_computed_tokens,
            num_output_tokens,
            req_id_to_num_output_tokens: OnceLock::new(),
        }
    }

    pub fn make_empty() -> Self {
        Self::default()
    }

    pub fn num_reqs(&self) -> usize {
        self.req_ids.len()
    }

    pub fn is_context_phase(&self, req_id: &str) -> bool {
        let map = self.req_id_to_num_output_tokens.get_or_init(|| {
            // `dict(zip(...))`: a repeated id keeps its last value.
            self.req_ids
                .iter()
                .cloned()
                .zip(self.num_output_tokens.iter().copied())
                .collect()
        });
        map.get(req_id) == Some(&0)
    }
}

/// Scheduled prefill/decode request and input counts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IterationDetails {
    pub num_prefill_requests: usize,
    pub num_prefill_tokens: usize,
    pub num_decode_requests: usize,
    pub num_decode_tokens: usize,
    pub num_encoder_inputs: usize,
    pub num_encoder_output_tokens: usize,
}

/// A request-local CPU payload update of one encoder input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncoderInputUpdate {
    /// Release a consumed input (`None` in Python).
    Release,
    /// Restore the input's payload for re-encoding after KV-load rollback
    /// (the `MultiModalKwargsItem` in Python, which the binding layer fills
    /// in from the request's multimodal data).
    Restore,
}

pub use crate::kv_cache::KvCacheBlockCopy;

/// A Python object the scheduler passes through without reading, such as
/// connector metadata, stats and events. Compared by identity.
#[derive(Clone)]
pub struct Opaque(pub Arc<dyn Any + Send + Sync>);

impl Opaque {
    pub fn new(value: impl Any + Send + Sync) -> Self {
        Self(Arc::new(value))
    }
}

impl fmt::Debug for Opaque {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Opaque(..)")
    }
}

impl PartialEq for Opaque {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

/// Python dicts and sets keep insertion order, mirrored with `IndexMap` /
/// `IndexSet`.
///
/// Not mirrored: the scheduler-local `kv_connector_block_state`, which is
/// passed to the connector directly.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SchedulerOutput {
    /// list of the requests that are scheduled for the first time.
    /// We cache the request's data in each worker process, so that we don't
    /// need to re-send it every scheduling step.
    pub scheduled_new_reqs: Vec<NewRequestData>,
    /// list of the requests that have been scheduled before.
    /// Since the request's data is already cached in the worker processes,
    /// we only send the diff to minimize the communication cost.
    pub scheduled_cached_reqs: CachedRequestData,

    /// req_id -> num_scheduled_tokens
    /// Number of tokens scheduled for each request.
    pub num_scheduled_tokens: IndexMap<String, usize>,
    /// Total number of tokens scheduled for all requests.
    /// Equal to sum(num_scheduled_tokens.values())
    pub total_num_scheduled_tokens: usize,
    /// req_id -> spec_token_ids
    /// If a request does not have any spec decode tokens, it will not be
    /// included in the dictionary.
    pub scheduled_spec_decode_tokens: IndexMap<String, Vec<i32>>,
    /// req_id -> encoder input indices that need processing.
    /// E.g., if a request has [0, 1], it could mean the vision encoder needs
    /// to process that the request's 0-th and 1-th images in the current step.
    pub scheduled_encoder_inputs: IndexMap<String, Vec<usize>>,

    /// Request IDs that are finished in between the previous and the current
    /// steps. This is used to notify the workers about the finished requests
    /// so that they can free the cached states for those requests.
    ///
    /// Python's `set[str]` order depends on the string hash seed, so no
    /// consumer relies on it.
    pub finished_req_ids: IndexSet<String>,
    /// list of mm_hash strings associated with the encoder outputs to be
    /// freed from the encoder cache.
    pub free_encoder_mm_hashes: Vec<String>,

    /// Request-local CPU payload updates: None releases a consumed input; a
    /// payload restores it for re-encoding after KV-load rollback. GPU embeddings
    /// and the feature's identifier/placeholder metadata are unaffected.
    pub encoder_input_updates: IndexMap<String, IndexMap<usize, EncoderInputUpdate>>,

    /// Optional scheduling-time snapshot, before worker-side token adjustments.
    pub iteration_details: Option<IterationDetails>,

    /// Request IDs that are preempted in this step.
    /// Only used for v2 model runner.
    pub preempted_req_ids: Option<IndexSet<String>>,

    /// Whether any of the scheduled requests use structured output.
    /// Set only in async scheduling case.
    pub has_structured_output_requests: bool,

    /// Whether any scheduled request uses structured output with speculative
    /// decoding. Set only in async scheduling case.
    pub has_structured_output_spec_decode: bool,

    /// Whether the scheduled requests have all the output tokens they
    /// need to perform grammar bitmask computation.
    pub pending_structured_output_tokens: bool,

    /// Used for adjusting acceptance rate calculation.
    pub num_invalid_spec_tokens: Option<IndexMap<String, usize>>,

    /// Whether any scheduled request consumes KV that the connector loads
    /// synchronously during this step (load_async=False).
    pub has_sync_kv_loads: bool,

    /// Block IDs freshly allocated from the pool during this scheduling step.
    /// The worker zeros the corresponding GPU memory before the blocks are used,
    /// preventing stale NaN/data from corrupting attention or SSM computation.
    pub new_block_ids_to_zero: Option<Vec<u32>>,

    /// CoW copies to apply after zeroing new blocks and before forward.
    pub kv_cache_block_copies: Option<Vec<KvCacheBlockCopy>>,

    /// Dynamic speculative decoding: optimal K chosen by scheduler.
    /// Number of spec tokens to schedule for the next step.
    pub num_spec_tokens_to_schedule: usize,

    /// Whether no scheduled request can use the drafts proposed after this step
    /// (non-final prefill chunks, or requests sure to reach max_tokens), so the
    /// drafter can skip proposing them.
    pub skip_drafts: bool,

    /// KV Cache Connector metadata (the connector's `build_connector_meta`).
    pub kv_connector_metadata: Option<Opaque>,
    /// The aux output connector's metadata for this step.
    pub aux_output_connector_metadata: Option<Opaque>,
    /// EC Connector metadata.
    pub ec_connector_metadata: Option<Opaque>,
}

impl SchedulerOutput {
    pub fn make_empty() -> Self {
        Self::default()
    }
}

/// The structured output requests of a step and their grammar bitmask.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GrammarOutput {
    /// ids of structured output requests.
    pub structured_output_request_ids: Vec<String>,
    /// Bitmask ordered as structured_output_request_ids, one row per
    /// request (an `int32` ndarray in Python).
    pub grammar_bitmask: Vec<Vec<i32>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sched::request::NewRequest;

    #[test]
    fn cached_request_data_context_phase() {
        let data = CachedRequestData::new(
            vec!["a".into(), "b".into(), "a".into()],
            vec![],
            vec![None, Some(vec![vec![1, 2]]), None],
            vec![3, 4, 5],
            vec![0, 0, 2],
        );
        assert_eq!(data.num_reqs(), 3);
        // A repeated id keeps its last value, as `dict(zip(...))`.
        assert!(!data.is_context_phase("a"));
        assert!(data.is_context_phase("b"));
        assert!(!data.is_context_phase("missing"));
        assert_eq!(CachedRequestData::make_empty().num_reqs(), 0);
    }

    #[test]
    fn new_request_data_from_request() {
        let mut request = Request::new(NewRequest::for_test("r", vec![1, 2, 3])).unwrap();
        request.num_computed_tokens = 2;
        request.replay_start = 1;
        let data = NewRequestData::from_request(
            &request,
            vec![vec![7], vec![]],
            request.prefill_token_ids().to_vec(),
        );
        expect_test::expect![[r#"
            NewRequestData {
                req_id: "r",
                prompt_token_ids: Some(
                    [
                        1,
                        2,
                        3,
                    ],
                ),
                block_ids: [
                    [
                        7,
                    ],
                    [],
                ],
                num_computed_tokens: 2,
                prefill_token_ids: [
                    1,
                    2,
                    3,
                ],
                prompt_is_token_ids: None,
                replay_start: 1,
            }
        "#]]
        .assert_debug_eq(&data);
        assert_eq!(data.prompt_len(), Some(3));
        assert_eq!(SchedulerOutput::make_empty().total_num_scheduled_tokens, 0);
    }
}
