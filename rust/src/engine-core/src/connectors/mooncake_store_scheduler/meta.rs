// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The scheduler-to-worker metadata of the `MooncakeStoreConnector`
//! (`mooncake_store/data.py`) and the scheduler's per-request
//! `RequestTracker`.
//!
//! The binding layer builds the Python objects from these for the (Python)
//! worker; their content mirrors the Python objects field by field. Where
//! Python shares one object between metadata entries (a request's live
//! `block_hashes` list, a tracker's block-id tuple), the Rust values say so
//! (see [`ReqBlockHashes`] and [`ReqMeta::block_ids`]), so that the Python
//! objects built from them pickle to the same bytes.

use std::sync::Arc;

use indexmap::IndexSet;

use crate::connectors::mooncake_store::TailKeyBoundary;
use crate::hash::BlockHash;
use crate::kv_cache::{BlockId, NULL_BLOCK_ID};
use crate::sched::output::GroupedBlockIds;

/// `LoadSpec`: specification for loading KV cache from external store.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadSpec {
    pub vllm_cached_tokens: usize,
    pub kvpool_cached_tokens: usize,
    pub can_load: bool,
    pub token_len: usize,
    pub tail_key_boundaries: Vec<TailKeyBoundary>,
}

/// `BoundaryPut`: a block stored under the hash of the prefix ending at
/// `num_tokens`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoundaryPut {
    pub group_id: usize,
    pub block_id: BlockId,
    pub num_tokens: usize,
}

/// `ReqMeta.block_hashes`: which list of the request's block hashes an entry
/// references.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReqBlockHashes {
    /// `request.block_hashes` of a request the scheduler still holds: the
    /// request's own (live) list, which has `len` hashes when the metadata is
    /// built.
    Request { len: usize },
    /// The hashes of a request that finished before the metadata is built
    /// (the finish-time hand-offs). `copy` distinguishes
    /// `list(request.block_hashes)` (a copy sharing the hash objects) from
    /// the request's list itself.
    Finished {
        hashes: Arc<[BlockHash]>,
        copy: bool,
    },
}

impl ReqBlockHashes {
    pub fn len(&self) -> usize {
        match self {
            Self::Request { len } => *len,
            Self::Finished { hashes, .. } => hashes.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// `ReqMeta`: per-request metadata for store put/get operations.
///
/// Not mirrored: `current_event` (set by the worker), and `token_ids` /
/// `num_hash_lookahead_tokens`, which only KV cache events populate.
// TODO: KV cache events (`token_ids`, `num_hash_lookahead_tokens`) once the
// native connector supports them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReqMeta {
    pub req_id: String,
    pub token_len_chunk: usize,
    /// One list per store group. Entries holding the same `Arc` reference one
    /// Python tuple (Python shares a tracker's `allocated_block_ids`).
    pub block_ids: Arc<GroupedBlockIds>,
    pub block_hashes: ReqBlockHashes,

    pub can_save: bool,
    pub load_spec: Option<LoadSpec>,

    /// Absolute request offset represented by token_ids[0].
    pub token_ids_start: usize,
    pub num_prompt_tokens: Option<usize>,
    /// Total prefill tokens, including generated tokens replayed after preemption.
    pub prefill_end_tokens: Option<usize>,
    /// Identifies this store job for the engine's lifetime. A request id cannot
    /// serve that purpose: it is reused once a preempted request resumes, so it
    /// would release the wrong job's blocks.
    pub store_job_id: Option<u64>,
    /// Blocks stored under an explicit prefix hash rather than by position:
    /// core-provided mamba "align" states (a block-aligned entry is a committed
    /// snapshot, a non-aligned one the sub-block CoW prompt tail) and the
    /// scheduler-resolved non-Mamba blocks of the prompt's partial tail. The
    /// store-job reference keeps each block alive until every worker rank
    /// finishes the job.
    pub boundary_puts: Option<Vec<BoundaryPut>>,
    /// Total computed prefix length at the end of this step.
    pub completed_token_len: Option<usize>,
    /// Set on the save that first covers the whole prompt, so the scheduler
    /// publishes the prompt's partial tail once rather than on every save.
    pub publish_partial_tail: bool,
}

impl ReqMeta {
    /// An offload-only entry: `token_len_chunk=0` skips the normal save,
    /// `can_save=True` takes the normal enqueue and store-job pinning path.
    pub(super) fn offload_only(
        req_id: String,
        block_ids: Arc<GroupedBlockIds>,
        block_hashes: ReqBlockHashes,
        num_prompt_tokens: usize,
        prefill_end_tokens: usize,
        boundary_puts: Vec<BoundaryPut>,
        completed_token_len: usize,
    ) -> Self {
        Self {
            req_id,
            token_len_chunk: 0,
            block_ids,
            block_hashes,
            can_save: true,
            load_spec: None,
            token_ids_start: 0,
            num_prompt_tokens: Some(num_prompt_tokens),
            prefill_end_tokens: Some(prefill_end_tokens),
            store_job_id: None,
            boundary_puts: Some(boundary_puts),
            completed_token_len: Some(completed_token_len),
            publish_partial_tail: false,
        }
    }
}

/// An operation on the scheduler's `_unfinished_request_ids` set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SetOp {
    Add(String),
    Discard(String),
}

/// `MooncakeStoreConnectorMetadata`: metadata passed from scheduler to worker.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MooncakeStoreConnectorMetadata {
    pub requests: Vec<ReqMeta>,
    /// Python hands every metadata the scheduler's own (live)
    /// `_unfinished_request_ids` set. These are the changes to that set since
    /// the previous metadata, in order: replaying them on one persistent
    /// Python set reproduces it, down to its iteration order.
    pub unfinished_request_id_ops: Vec<SetOp>,
    /// `scheduler_output.preempted_req_ids or set()`.
    pub preempted_req_ids: IndexSet<String>,
}

/// `MooncakeStoreWorkerMetadata`, aggregated over the worker ranks: maps
/// `ReqMeta.store_job_id` to the number of ranks done with that job, in dict
/// order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MooncakeStoreWorkerMetadata {
    pub completed_saves: Vec<(u64, usize)>,
    /// Opt-in save pinning protocol ([`super::SavePinPolicy::SkipConfirmed`]):
    /// per finished store job, the lowest token offset below which every
    /// rank holds the job's KV in the store. Empty under the default
    /// protocol; the current worker does not report it.
    pub saved_offsets: Vec<(u64, usize)>,
}

/// `RequestTracker`: tracks per-request state across scheduler ticks.
///
/// Not mirrored: `token_ids`, which only KV cache events read.
#[derive(Clone, Debug)]
pub(super) struct RequestTracker {
    pub req_id: String,
    pub token_len: usize,
    pub allocated_block_ids: Arc<GroupedBlockIds>,
    pub num_saved_tokens: usize,
    pub has_pending_offload: bool,
    /// Snapshot of the prefill range length at tracker creation time.
    /// For a fresh request this is len(prompt). For a resumed-from-preemption
    /// request it includes previously-generated tokens, which are re-prefilled.
    pub prefill_end_tokens: usize,
    pub partial_tail_sent: bool,
}

impl RequestTracker {
    pub fn new(
        req_id: String,
        token_len: usize,
        allocated_block_ids: Arc<GroupedBlockIds>,
        prefill_end_tokens: usize,
    ) -> Self {
        Self {
            req_id,
            token_len,
            allocated_block_ids,
            num_saved_tokens: 0,
            has_pending_offload: false,
            prefill_end_tokens,
            partial_tail_sent: false,
        }
    }

    pub fn update(&mut self, new_block_ids: GroupedBlockIds) {
        assert_eq!(
            new_block_ids.len(),
            self.allocated_block_ids.len(),
            "Group count mismatch: tracker has {} groups, update has {}",
            self.allocated_block_ids.len(),
            new_block_ids.len()
        );
        if new_block_ids.iter().all(Vec::is_empty) {
            return;
        }
        // Copies only while a metadata entry built earlier still shares it.
        let allocated = Arc::make_mut(&mut self.allocated_block_ids);
        for (existing, new) in allocated.iter_mut().zip(new_block_ids) {
            existing.extend(new);
        }
    }
}

/// The arguments of `ReqMeta.from_request_tracker` besides the tracker.
pub(super) struct FromTracker {
    pub block_size: usize,
    pub hash_block_size: usize,
    pub load_spec: Option<LoadSpec>,
    pub skip_save: bool,
    pub block_hashes: ReqBlockHashes,
    pub save_partial_tail: bool,
    pub num_prompt_tokens: Option<usize>,
}

/// `ReqMeta.from_request_tracker`: create ReqMeta from a RequestTracker.
pub(super) fn from_request_tracker(
    tracker: &mut RequestTracker,
    args: FromTracker,
) -> Option<ReqMeta> {
    let FromTracker {
        block_size,
        hash_block_size,
        mut load_spec,
        skip_save,
        block_hashes,
        save_partial_tail,
        num_prompt_tokens,
    } = args;
    if save_partial_tail {
        assert!(num_prompt_tokens.is_some());
    }
    let input_token_len = tracker.token_len;

    let token_ids_start = tracker.num_saved_tokens;
    let chunk_boundary = (token_ids_start + 1).div_ceil(block_size) * block_size;
    let num_tokens_to_save = (input_token_len / block_size * block_size)
        .min(block_hashes.len() * hash_block_size / block_size * block_size);

    let publish_tail = save_partial_tail
        && !tracker.partial_tail_sent
        && num_prompt_tokens.is_some_and(|n| 0 < n && n <= input_token_len);
    let mut skip_save = skip_save || (num_tokens_to_save < chunk_boundary && !publish_tail);
    // A ReqMeta must never carry both a save AND a load.
    // The save would also be wasted work — the bytes are being looked up
    // in the store right now. Later cached_reqs steps save new tokens
    // normally.
    let can_load = load_spec.as_ref().is_some_and(|s| s.can_load);
    if can_load {
        skip_save = true;
    }
    if skip_save && load_spec.is_none() {
        return None;
    }

    if !skip_save {
        tracker.num_saved_tokens = num_tokens_to_save;
        tracker.partial_tail_sent |= publish_tail;
    }

    if !can_load {
        load_spec = None;
    }

    Some(ReqMeta {
        req_id: tracker.req_id.clone(),
        token_len_chunk: num_tokens_to_save,
        block_ids: tracker.allocated_block_ids.clone(),
        block_hashes,
        can_save: !skip_save,
        load_spec,
        token_ids_start,
        num_prompt_tokens,
        prefill_end_tokens: Some(tracker.prefill_end_tokens),
        store_job_id: None,
        boundary_puts: None,
        completed_token_len: Some(input_token_len),
        publish_partial_tail: publish_tail && !skip_save,
    })
}

/// The `MooncakeStoreCoordinator` state the scheduler half reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StoreGeometry {
    /// `kv_cache_spec.block_size` of each store group (DCP-resolved), indexed
    /// like `ReqMeta.block_ids`.
    pub group_block_sizes: Vec<usize>,
    /// Store groups whose (unwrapped) spec is a `MambaSpec`.
    pub mamba_group_ids: Vec<usize>,
    pub hash_block_size: usize,
    pub lcm_block_size: usize,
    /// `coord.use_eagle`: EAGLE block drop with no lookahead hashes.
    pub use_eagle: bool,
    pub num_hash_lookahead_tokens: usize,
    /// `(group_id, margin)` pairs of `eagle_proof_margin_by_group`.
    pub eagle_proof_margin_by_group: Vec<(usize, usize)>,
    pub enable_partial_hash_hits: bool,
}

impl StoreGeometry {
    fn eagle_proof_margin(&self, group_id: usize) -> usize {
        self.eagle_proof_margin_by_group
            .iter()
            .find(|&&(g, _)| g == group_id)
            .map_or(0, |&(_, margin)| margin)
    }
}

/// `get_mamba_prefill_checkpoint_position`: the reusable Mamba checkpoint
/// boundary for a prefill.
fn mamba_prefill_checkpoint_position(
    num_tokens: usize,
    hash_block_size: usize,
    drop_eagle_block: bool,
    num_hash_lookahead_tokens: usize,
) -> usize {
    // Leave tokens for logits recomputation and the checkpoint's hash.
    // EAGLE's block drop already leaves tokens to recompute.
    let proof_limit = if drop_eagle_block {
        num_tokens as i64
    } else {
        num_tokens as i64 - num_hash_lookahead_tokens.max(1) as i64
    };
    let mut checkpoint_position =
        proof_limit.div_euclid(hash_block_size as i64) * hash_block_size as i64;
    if drop_eagle_block {
        checkpoint_position -= hash_block_size as i64;
    }
    checkpoint_position.max(0) as usize
}

/// `_partial_tail_non_mamba_puts`: locate the non-Mamba blocks a partial-tail
/// save publishes.
///
/// A later request resumes at the prompt's Mamba checkpoint (`boundary`) only
/// if both pieces of KV are stored:
///
/// - Mamba groups: the state at `boundary`, handed off by the core as a
///   boundary put of its own, so Mamba groups are not listed here.
/// - Other groups: the KV from the last LCM-aligned normal save up to
///   `boundary` plus the group's EAGLE proof margin (`proof_end`). Each
///   block is keyed by the hash at its end, clipped to `proof_end`.
///
/// Groups whose proof is not computed yet are omitted. For example, with LCM
/// 16, blocks of 4 tokens, and a checkpoint at 44, the full-attention blocks
/// are 8-10, keyed at 36, 40 and 44.
///
/// A Mamba state handed off at any other position, i.e. a shared-prefix
/// junction (`--enable-mamba-shared-prefix-checkpoint`), adds no puts here.
pub(super) fn partial_tail_non_mamba_puts(
    coord: &StoreGeometry,
    req_meta: &ReqMeta,
) -> Vec<BoundaryPut> {
    let block_sizes = &coord.group_block_sizes;
    let mamba_tails: Vec<usize> = req_meta
        .boundary_puts
        .iter()
        .flatten()
        .filter(|put| {
            coord.mamba_group_ids.contains(&put.group_id)
                && !put.num_tokens.is_multiple_of(block_sizes[put.group_id])
        })
        .map(|put| put.num_tokens)
        .collect();
    // The tail is due on the save that first covers the prompt, or when Mamba
    // hands off its checkpoint state.
    if mamba_tails.is_empty() && !req_meta.publish_partial_tail {
        return Vec::new();
    }
    let prompt_tokens = req_meta.num_prompt_tokens.expect("num_prompt_tokens is set");
    let completed = req_meta.completed_token_len.expect("completed_token_len is set");
    let hash_block_size = coord.hash_block_size;
    // Core drops the Mamba checkpoint whenever EAGLE block drop is on, whichever
    // groups carry the eagle flag.
    let boundary = mamba_prefill_checkpoint_position(
        prompt_tokens,
        hash_block_size,
        coord.use_eagle,
        coord.num_hash_lookahead_tokens,
    );
    // Any other sub-block Mamba hand-off is a shared-prefix junction. Its
    // attention KV, EAGLE proof included, ends on an attention block end no later
    // than this checkpoint's proof, so the normal and prompt-tail saves store it.
    if !req_meta.publish_partial_tail && !mamba_tails.contains(&boundary) {
        return Vec::new();
    }
    let num_hashes = req_meta.block_hashes.len();
    if boundary == 0 || boundary / hash_block_size > num_hashes {
        return Vec::new();
    }
    let start = boundary / coord.lcm_block_size * coord.lcm_block_size;
    let mut puts = Vec::new();
    for (group_id, &block_size) in block_sizes.iter().enumerate() {
        if coord.mamba_group_ids.contains(&group_id) {
            continue;
        }
        let proof_end = boundary + coord.eagle_proof_margin(group_id);
        if proof_end > completed || proof_end / hash_block_size > num_hashes {
            continue;
        }
        let group_blocks = &req_meta.block_ids[group_id];
        for block_idx in start / block_size..proof_end.div_ceil(block_size) {
            let Some(&block_id) = group_blocks.get(block_idx) else {
                break;
            };
            if block_id == NULL_BLOCK_ID {
                continue;
            }
            let num_tokens = ((block_idx + 1) * block_size).min(proof_end);
            puts.push(BoundaryPut {
                group_id,
                block_id,
                num_tokens,
            });
        }
    }
    puts
}
