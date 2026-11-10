// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Unit tests of the pieces the scheduler differential test
//! (`sched/scheduler/tests/mooncake_store.rs`) reaches only indirectly.

use std::sync::Arc;

use super::meta::{FromTracker, RequestTracker, from_request_tracker, partial_tail_non_mamba_puts};
use super::*;
use crate::kv_cache::{BlockPool, MEDIUM_GPU};

fn tracker(token_len: usize, blocks: Vec<u32>) -> RequestTracker {
    RequestTracker::new("r".into(), token_len, Arc::new(vec![blocks]), 48)
}

fn args(num_hashes: usize) -> FromTracker {
    FromTracker {
        block_size: 16,
        hash_block_size: 16,
        load_spec: None,
        skip_save: false,
        block_hashes: ReqBlockHashes::Request { len: num_hashes },
        save_partial_tail: false,
        num_prompt_tokens: Some(48),
    }
}

/// Saves advance by whole blocks and are capped by the hashes available.
#[test]
fn from_request_tracker_saves_whole_hashed_blocks() {
    let mut t = tracker(40, vec![1, 2, 3]);
    let meta = from_request_tracker(&mut t, args(1)).unwrap();
    assert_eq!((meta.token_len_chunk, meta.token_ids_start), (16, 0));
    assert_eq!(t.num_saved_tokens, 16);
    // Nothing new to save until the next block boundary.
    assert!(from_request_tracker(&mut t, args(1)).is_none());
    let meta = from_request_tracker(&mut t, args(3)).unwrap();
    assert_eq!((meta.token_len_chunk, meta.token_ids_start), (32, 16));
}

/// A load always suppresses the save; a load that cannot proceed is dropped
/// from the entry but the save still happens.
#[test]
fn from_request_tracker_load_and_save_are_exclusive() {
    let spec = |can_load| LoadSpec {
        vllm_cached_tokens: 0,
        kvpool_cached_tokens: 32,
        can_load,
        token_len: 0,
        tail_key_boundaries: vec![],
    };
    let mut t = tracker(48, vec![1, 2, 3]);
    let meta = from_request_tracker(
        &mut t,
        FromTracker {
            load_spec: Some(spec(true)),
            ..args(3)
        },
    )
    .unwrap();
    assert!(!meta.can_save && meta.load_spec.is_some());
    assert_eq!(t.num_saved_tokens, 0);
    let meta = from_request_tracker(
        &mut t,
        FromTracker {
            load_spec: Some(spec(false)),
            ..args(3)
        },
    )
    .unwrap();
    assert!(meta.can_save && meta.load_spec.is_none());
    assert_eq!(meta.token_len_chunk, 48);
}

/// The worked example of `_partial_tail_non_mamba_puts`: LCM 16, blocks of 4
/// tokens, a checkpoint at 44: blocks 8-10, keyed at 36, 40 and 44.
#[test]
fn partial_tail_puts_cover_lcm_save_to_checkpoint() {
    let coord = StoreGeometry {
        group_block_sizes: vec![4, 16],
        mamba_group_ids: vec![1],
        hash_block_size: 4,
        lcm_block_size: 16,
        use_eagle: false,
        num_hash_lookahead_tokens: 0,
        eagle_proof_margin_by_group: vec![],
        enable_partial_hash_hits: true,
    };
    let meta = ReqMeta {
        publish_partial_tail: true,
        ..ReqMeta::offload_only(
            "r".into(),
            Arc::new(vec![(100..112).collect(), vec![7, 8, 9]]),
            ReqBlockHashes::Request { len: 11 },
            45,
            45,
            vec![],
            45,
        )
    };
    let puts = partial_tail_non_mamba_puts(&coord, &meta);
    let got: Vec<_> = puts.iter().map(|p| (p.group_id, p.block_id, p.num_tokens)).collect();
    assert_eq!(got, vec![(0, 108, 36), (0, 109, 40), (0, 110, 44)]);
}

struct NoLookups;

impl LookupClient for NoLookups {
    fn prefetch(&mut self, _request: &Request) -> bool {
        false
    }

    fn lookup(&mut self, _request: &Request, _non_block: bool) -> Option<LookupResult> {
        Some(LookupResult::default())
    }

    fn discard(&mut self, _request_id: &str) {}

    fn reset(&mut self) -> bool {
        true
    }
}

fn scheduler(policy: SavePinPolicy) -> MooncakeStoreScheduler {
    MooncakeStoreScheduler::new(
        MooncakeStoreSchedulerConfig {
            kv_role: KvRole::Both,
            load_async: false,
            lookup_async: false,
            enable_lookup: true,
            save_decode_cache: false,
            store_group_ids: vec![0],
            block_size: 16,
            hash_block_size: 16,
            boundary_state_group_ids: vec![],
            store: Some(StoreGeometry {
                group_block_sizes: vec![16],
                mamba_group_ids: vec![],
                hash_block_size: 16,
                lcm_block_size: 16,
                use_eagle: false,
                num_hash_lookahead_tokens: 0,
                eagle_proof_margin_by_group: vec![],
                enable_partial_hash_hits: false,
            }),
            num_workers: 2,
            save_pin_policy: policy,
        },
        Box::new(NoLookups),
    )
}

fn save_meta(req_id: &str, start: usize, end: usize, blocks: Vec<u32>) -> ReqMeta {
    ReqMeta {
        token_len_chunk: end,
        token_ids_start: start,
        ..ReqMeta::offload_only(
            req_id.into(),
            Arc::new(vec![blocks]),
            ReqBlockHashes::Request { len: end / 16 },
            end,
            end,
            vec![],
            end,
        )
    }
    .with_boundary_puts(None)
}

impl ReqMeta {
    fn with_boundary_puts(mut self, puts: Option<Vec<BoundaryPut>>) -> Self {
        self.boundary_puts = puts;
        self
    }
}

fn ref_cnts(pool: &BlockPool, blocks: &[u32]) -> Vec<i32> {
    blocks.iter().map(|&b| pool.block(b).ref_cnt).collect()
}

/// Every save job pins the whole table, and its references go once every
/// rank reported the job, most recent block first in the free queue.
#[test]
fn whole_table_pins_until_every_rank_reports() {
    let mut s = scheduler(SavePinPolicy::WholeTable);
    let mut pool = BlockPool::new(8, true, 16, false, None, 0, MEDIUM_GPU);
    let mut jobs = vec![
        save_meta("r", 0, 32, vec![1, 2]),
        save_meta("r", 32, 48, vec![1, 2, 3]),
    ];
    s.reference_save_blocks(&mut jobs, &mut pool);
    assert_eq!(ref_cnts(&pool, &[1, 2, 3]), vec![2, 2, 1]);
    let done = |job, count| MooncakeStoreWorkerMetadata {
        completed_saves: vec![(job, count)],
        saved_offsets: vec![(job, 32)],
    };
    s.update_worker_meta(&done(0, 1), &mut pool);
    assert_eq!(ref_cnts(&pool, &[1, 2, 3]), vec![2, 2, 1]);
    s.update_worker_meta(&done(0, 1), &mut pool);
    assert_eq!(ref_cnts(&pool, &[1, 2, 3]), vec![1, 1, 1]);
    // A later job still pins the prefix the first one confirmed.
    let mut jobs = vec![save_meta("r", 48, 64, vec![1, 2, 3, 4])];
    s.reference_save_blocks(&mut jobs, &mut pool);
    assert_eq!(ref_cnts(&pool, &[1, 2, 3, 4]), vec![2, 2, 2, 1]);
}

/// The opt-in protocol skips the blocks below a confirmed save offset.
#[test]
fn skip_confirmed_pins_only_unconfirmed_blocks() {
    let mut s = scheduler(SavePinPolicy::SkipConfirmed);
    let mut pool = BlockPool::new(8, true, 16, false, None, 0, MEDIUM_GPU);
    let mut jobs = vec![save_meta("r", 0, 32, vec![1, 2])];
    s.reference_save_blocks(&mut jobs, &mut pool);
    let done = MooncakeStoreWorkerMetadata {
        completed_saves: vec![(0, 2)],
        saved_offsets: vec![(0, 32)],
    };
    s.update_worker_meta(&done, &mut pool);
    let mut jobs = vec![save_meta("r", 32, 64, vec![1, 2, 3, 4])];
    s.reference_save_blocks(&mut jobs, &mut pool);
    assert_eq!(ref_cnts(&pool, &[1, 2, 3, 4]), vec![0, 0, 1, 1]);
    let pinned: Vec<_> = s.pinned_saves().map(|(_, p)| p.block_ids.clone()).collect();
    assert_eq!(pinned, vec![vec![3, 4]]);
}
