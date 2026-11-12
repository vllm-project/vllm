// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Tests ported from `tests/v1/core/test_prefix_caching.py`,
//! `test_single_type_kv_cache_manager.py` and `test_kv_cache_utils.py`.

use expect_test::expect;
use sha2::{Digest, Sha256};

use super::*;
use crate::hash::BlockHash;

mod differential;

/// A minimal request for driving the KV cache manager in tests.
#[derive(Clone, Debug)]
pub(super) struct TestRequest {
    pub id: String,
    pub tokens: Vec<i32>,
    pub block_hashes: Vec<BlockHash>,
    pub num_prompt_tokens: usize,
    pub num_computed_tokens: usize,
    pub num_in_flight_tokens: usize,
    pub prefix_cache_hit_limit: Option<usize>,
    pub shared_prefix_boundary: usize,
    pub waiting: bool,
}

impl KvRequest for TestRequest {
    fn request_id(&self) -> &str {
        &self.id
    }
    fn block_hashes(&self) -> &[BlockHash] {
        &self.block_hashes
    }
    fn num_tokens(&self) -> usize {
        self.tokens.len()
    }
    fn num_prompt_tokens(&self) -> usize {
        self.num_prompt_tokens
    }
    fn num_computed_tokens(&self) -> usize {
        self.num_computed_tokens
    }
    fn num_in_flight_tokens(&self) -> usize {
        self.num_in_flight_tokens
    }
    fn num_preemptions(&self) -> usize {
        0
    }
    fn shared_prefix_boundary(&self) -> usize {
        self.shared_prefix_boundary
    }
    fn prefix_cache_hit_limit(&self) -> Option<usize> {
        self.prefix_cache_hit_limit
    }
    fn is_waiting_or_preempted(&self) -> bool {
        self.waiting
    }
    fn token_ids(&self, start: usize, end: usize) -> Vec<i32> {
        let end = end.min(self.tokens.len());
        self.tokens[start.min(end)..end].to_vec()
    }
}

/// Chained test hashes at `hash_block_size` granularity, each also covering
/// `lookahead` tokens past its block (not bitwise Python hashes).
pub(super) fn hash_tokens(
    tokens: &[i32],
    hash_block_size: usize,
    lookahead: usize,
) -> Vec<BlockHash> {
    let mut parent = [0u8; 32];
    let mut hashes = Vec::new();
    let num_tokens = tokens.len().saturating_sub(lookahead);
    let mut start = 0;
    while start + hash_block_size <= num_tokens {
        let mut hasher = Sha256::new();
        hasher.update(parent);
        for t in &tokens[start..start + hash_block_size + lookahead] {
            hasher.update(t.to_le_bytes());
        }
        parent = hasher.finalize().into();
        hashes.push(BlockHash::from_slice(&parent));
        start += hash_block_size;
    }
    hashes
}

pub(super) fn make_request(id: &str, tokens: Vec<i32>, hash_block_size: usize) -> TestRequest {
    make_request_with_lookahead(id, tokens, hash_block_size, 0)
}

fn make_request_with_lookahead(
    id: &str,
    tokens: Vec<i32>,
    hash_block_size: usize,
    lookahead: usize,
) -> TestRequest {
    TestRequest {
        id: id.to_owned(),
        block_hashes: hash_tokens(&tokens, hash_block_size, lookahead),
        num_prompt_tokens: tokens.len(),
        tokens,
        num_computed_tokens: 0,
        num_in_flight_tokens: 0,
        prefix_cache_hit_limit: None,
        shared_prefix_boundary: 0,
        waiting: false,
    }
}

fn full_spec(block_size: usize) -> KvCacheSpec {
    KvCacheSpec::new(KvCacheSpecType::FullAttention, block_size)
}

fn swa_spec(block_size: usize, sliding_window: usize) -> KvCacheSpec {
    KvCacheSpec::new(
        KvCacheSpecType::SlidingWindow {
            sliding_window,
            extra_retained_tokens: 0,
        },
        block_size,
    )
}

fn mamba_spec(block_size: usize) -> KvCacheSpec {
    KvCacheSpec::new(
        KvCacheSpecType::Mamba {
            mamba_cache_mode: MambaCacheMode::Align,
            num_speculative_blocks: 0,
            num_prefill_checkpoint_blocks: 0,
            prefill_checkpoint_alignment: None,
        },
        block_size,
    )
}

fn make_config(num_blocks: usize, specs: Vec<KvCacheSpec>) -> KvCacheConfig {
    KvCacheConfig::new(
        num_blocks,
        specs.into_iter().map(KvCacheGroupSpec::new).collect(),
    )
}

fn lcm(a: usize, b: usize) -> usize {
    fn gcd(a: usize, b: usize) -> usize {
        if b == 0 { a } else { gcd(b, a % b) }
    }
    a / gcd(a, b) * b
}

/// `make_kv_cache_manager`: derives `scheduler_block_size` from the groups.
fn make_manager(
    config: KvCacheConfig,
    hash_block_size: usize,
    f: impl FnOnce(&mut KvCacheManagerConfig),
) -> KvCacheManager {
    let scheduler_block_size =
        config.kv_cache_groups.iter().map(|g| g.kv_cache_spec.block_size).fold(1, lcm);
    let mut manager_config = KvCacheManagerConfig::new(8192, scheduler_block_size, hash_block_size);
    f(&mut manager_config);
    KvCacheManager::new(config, manager_config).unwrap()
}

fn alloc(
    manager: &mut KvCacheManager,
    req: &TestRequest,
    num_new_tokens: usize,
    num_new_computed_tokens: usize,
    computed: Option<&KvCacheBlocks>,
) -> Option<KvCacheBlocks> {
    manager
        .allocate_slots(
            req,
            num_new_tokens,
            AllocateSlotsArgs {
                num_new_computed_tokens,
                new_computed_blocks: computed,
                ..Default::default()
            },
        )
        .unwrap()
}

fn free_list(manager: &KvCacheManager) -> Vec<BlockId> {
    manager.block_pool().free_block_queue.get_all_free_blocks()
}

#[test]
fn free_kv_cache_block_queue_operations() {
    let mut queue = FreeKvCacheBlockQueue::new(5);
    assert_eq!(queue.num_free_blocks, 5);
    assert_eq!(queue.popleft(), 0);
    assert_eq!(queue.num_free_blocks, 4);
    queue.remove(2);
    assert_eq!(queue.get_all_free_blocks(), vec![1, 3, 4]);
    queue.append(0);
    queue.prepend_n(&[2]);
    assert_eq!(queue.get_all_free_blocks(), vec![2, 1, 3, 4, 0]);
    assert_eq!(queue.popleft_n(2), vec![2, 1]);
    queue.append_n(&[1, 2]);
    assert_eq!(queue.get_all_free_blocks(), vec![3, 4, 0, 1, 2]);
    assert_eq!(
        queue.iter_blocks_after(Some(0)).collect::<Vec<_>>(),
        vec![1, 2]
    );
    assert_eq!(queue.popleft_n(5), vec![3, 4, 0, 1, 2]);
    assert_eq!(queue.num_free_blocks, 0);
    assert!(queue.get_all_free_blocks().is_empty());
    // Drain and refill.
    queue.append_n(&[4, 3]);
    assert_eq!(queue.popleft_n(2), vec![4, 3]);
}

fn key(name: &str) -> BlockHashWithGroupId {
    BlockHashWithGroupId::new(BlockHash::from_slice(name.as_bytes()), 0)
}

#[test]
fn block_lookup_cache_single_block_per_key() {
    let mut cache = BlockHashToBlockMap::default();
    let (key0, key1) = (key("hash0"), key("hash1"));
    cache.insert(key0.clone(), 0);
    cache.insert(key1.clone(), 1);
    assert_eq!(cache.get_one_block(&key0), Some(0));
    assert_eq!(cache.get_one_block(&key1), Some(1));
    // No block popped due to block_id mismatch
    assert_eq!(cache.pop(&key0, 100), None);
    assert_eq!(cache.pop(&key0, 0), Some(0));
    assert_eq!(cache.get_one_block(&key0), None);
    assert_eq!(cache.pop(&key0, 1), None);
    assert_eq!(cache.pop(&key1, 1), Some(1));
    assert!(cache.is_empty());
}

#[test]
fn block_lookup_cache_multi_blocks_per_key() {
    let mut cache = BlockHashToBlockMap::default();
    let (key0, key1) = (key("hash0"), key("hash1"));
    cache.insert(key0.clone(), 0);
    cache.insert(key0.clone(), 1);
    cache.insert(key1.clone(), 10);
    cache.insert(key1.clone(), 11);
    assert!(cache.contain(&key0, 1));
    assert_eq!(cache.get_one_block(&key0), Some(0));
    assert_eq!(cache.pop(&key0, 0), Some(0));
    assert_eq!(cache.get_one_block(&key0), Some(1));
    assert_eq!(cache.pop(&key0, 1), Some(1));
    assert_eq!(cache.get_one_block(&key0), None);
    assert_eq!(cache.pop(&key0, 2), None);
    // Popping the first id promotes the next one in insertion order.
    cache.insert(key1.clone(), 12);
    assert_eq!(cache.pop(&key1, 10), Some(10));
    assert_eq!(cache.get_one_block(&key1), Some(11));
    assert_eq!(cache.pop(&key1, 11), Some(11));
    assert_eq!(cache.get_one_block(&key1), Some(12));
}

/// `test_prefill`.
#[test]
fn prefill() {
    let bs = 16;
    let mut manager = make_manager(make_config(11, vec![full_spec(bs)]), bs, |_| {});
    // Complete 3 blocks (48 tokens)
    let common: Vec<i32> = (0..3).flat_map(|i| [i; 16]).collect();

    // Fully cache miss
    let req0 = make_request("0", [common.clone(), vec![3; 7]].concat(), bs);
    let (computed, n, _) = manager.get_computed_blocks(&req0, None);
    assert_eq!(req0.block_hashes.len(), 3);
    assert!(computed.is_empty() && n == 0);
    let blocks = alloc(&mut manager, &req0, 55, 0, Some(&computed)).unwrap();
    assert_eq!(blocks.blocks, vec![vec![1, 2, 3, 4]]);
    for (i, block_id) in [1u32, 2, 3].into_iter().enumerate() {
        let blk = manager.block_pool().block(block_id);
        let hash = blk.block_hash.as_ref().unwrap();
        assert_eq!(hash.block_hash, req0.block_hashes[i]);
        assert_eq!(hash.group_id, 0);
        assert_eq!(blk.ref_cnt, 1);
    }
    assert!(manager.block_pool().block(4).block_hash.is_none());

    // Cache hit in the common prefix when the original block is still in use.
    let req1 = make_request("1", [common.clone(), vec![3; 5]].concat(), bs);
    let (computed, n, _) = manager.get_computed_blocks(&req1, None);
    assert_eq!(computed.blocks, vec![vec![1, 2, 3]]);
    assert_eq!(n, 48);
    let blocks = alloc(&mut manager, &req1, 53 - 48, 48, Some(&computed)).unwrap();
    assert_eq!(blocks.blocks, vec![vec![5]]);
    for &b in &computed.blocks[0] {
        assert_eq!(manager.block_pool().block(b).ref_cnt, 2);
    }
    assert_eq!(manager.block_pool().get_num_free_blocks(), 5);

    manager.free("0", false);
    manager.free("1", false);
    assert_eq!(manager.block_pool().get_num_free_blocks(), 10);
    // [partial without hashes from req1 and req0 (5, 4) - prepended for immediate reuse]
    // [unallocated (6, 7, 8, 9, 10)]
    // [common (3, 2, 1)]
    assert_eq!(free_list(&manager), vec![5, 4, 6, 7, 8, 9, 10, 3, 2, 1]);

    // Cache hit in the common prefix when the original block is already free.
    let req2 = make_request("2", [common.clone(), vec![3; 6]].concat(), bs);
    let (computed, n, _) = manager.get_computed_blocks(&req2, None);
    assert_eq!(computed.blocks, vec![vec![1, 2, 3]]);
    assert_eq!(n, 48);
    let blocks = alloc(&mut manager, &req2, 54 - 48, 48, Some(&computed)).unwrap();
    assert_eq!(blocks.blocks, vec![vec![5]]);
    assert_eq!(manager.block_pool().get_num_free_blocks(), 6);
    manager.free("2", false);

    // Cache miss and eviction.
    let req3 = make_request("3", vec![99; 160], bs);
    let (computed, n, _) = manager.get_computed_blocks(&req3, None);
    assert!(computed.is_empty() && n == 0);
    let blocks = alloc(&mut manager, &req3, 160, 0, Some(&computed)).unwrap();
    // This block ID order also checks the eviction order.
    assert_eq!(blocks.blocks, vec![vec![5, 4, 6, 7, 8, 9, 10, 3, 2, 1]]);
    assert_eq!(manager.block_pool().get_num_free_blocks(), 0);
    assert!(free_list(&manager).is_empty());
}

/// `test_evict`.
#[test]
fn evict() {
    let bs = 16;
    let mut manager = make_manager(make_config(11, vec![full_spec(bs)]), bs, |_| {});
    let last = 5 * 16 + 7;
    let req0 = make_request("0", (0..last).collect(), bs);
    let blocks = alloc(&mut manager, &req0, last as usize, 0, None).unwrap();
    assert_eq!(blocks.blocks[0].len(), 6);
    let req1 = make_request("1", (last..last + 48).collect(), bs);
    let blocks = alloc(&mut manager, &req1, 48, 0, None).unwrap();
    assert_eq!(blocks.blocks[0].len(), 3);
    assert_eq!(manager.block_pool().get_num_free_blocks(), 1);

    manager.free("0", false);
    // partial blocks (without hash) at head, other at tail (LRU policy):
    assert_eq!(free_list(&manager), vec![6, 10, 5, 4, 3, 2, 1]);
    manager.free("1", false);
    assert_eq!(free_list(&manager), vec![6, 10, 5, 4, 3, 2, 1, 9, 8, 7]);

    // Touch the first 2 blocks.
    let req2 = make_request("2", (0..2 * 16 + 3).collect(), bs);
    let (computed, n, _) = manager.get_computed_blocks(&req2, None);
    assert_eq!(computed.blocks, vec![vec![1, 2]]);
    assert_eq!(n, 32);
    let blocks = alloc(&mut manager, &req2, 3, 32, Some(&computed)).unwrap();
    assert_eq!(blocks.blocks, vec![vec![6]]);
    assert_eq!(manager.block_pool().get_num_free_blocks(), 7);
}

/// `_test_partial_request_hit`: evict `keys` from the lookup map, look up,
/// and restore them.
fn partial_request_hit(
    manager: &mut KvCacheManager,
    id: &str,
    tokens: &[i32],
    keys: &[BlockHashWithGroupId],
    expect_hit_blocks: usize,
) {
    let bs = 16;
    let pool = manager.block_pool_mut();
    let mut saved = Vec::new();
    for k in keys {
        let block = pool.cached_block_hash_to_block.get_one_block(k).unwrap();
        pool.cached_block_hash_to_block.pop(k, block).unwrap();
        saved.push((k.clone(), block));
    }
    let req = make_request(id, tokens.to_vec(), bs);
    let (computed, n, _) = manager.get_computed_blocks(&req, None);
    assert_eq!(n, expect_hit_blocks * bs, "request {id}");
    for group in &computed.blocks {
        assert_eq!(group.len(), expect_hit_blocks);
    }
    for (k, block) in saved {
        manager.block_pool_mut().cached_block_hash_to_block.insert(k, block);
    }
    manager.free(id, false);
}

/// `test_prefill_hybrid_model`.
#[test]
fn prefill_hybrid_model() {
    let bs = 16;
    let config = make_config(
        21,
        vec![full_spec(bs), swa_spec(bs, 2 * bs), swa_spec(bs, 2 * bs)],
    );
    let mut manager = make_manager(config, bs, |_| {});
    let common: Vec<i32> = (0..3).flat_map(|i| [i; 16]).collect();

    let req0 = make_request("0", [common.clone(), vec![3; 7]].concat(), bs);
    let blocks = alloc(&mut manager, &req0, 55, 0, None).unwrap();
    assert_eq!(
        blocks.blocks,
        vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8], vec![9, 10, 11, 12]]
    );
    for (i, ids) in [[1u32, 5, 9], [2, 6, 10], [3, 7, 11]].into_iter().enumerate() {
        for (group_id, block_id) in ids.into_iter().enumerate() {
            let hash = manager.block_pool().block(block_id).block_hash.clone().unwrap();
            assert_eq!(
                hash,
                BlockHashWithGroupId::new(req0.block_hashes[i].clone(), group_id)
            );
        }
    }

    let all_tokens = [common.clone(), vec![3; 5]].concat();
    let req1 = make_request("1", all_tokens.clone(), bs);
    let (computed, n, _) = manager.get_computed_blocks(&req1, None);
    assert_eq!(
        computed.blocks,
        vec![vec![1, 2, 3], vec![0, 6, 7], vec![0, 10, 11]]
    );
    assert_eq!(n, 48);
    let blocks = alloc(&mut manager, &req1, 5, 48, Some(&computed)).unwrap();
    assert_eq!(blocks.blocks, vec![vec![13], vec![14], vec![15]]);
    for &b in computed.blocks.iter().flatten().filter(|&&b| b != NULL_BLOCK_ID) {
        assert_eq!(manager.block_pool().block(b).ref_cnt, 2);
    }
    let hashes = req1.block_hashes.clone();
    manager.free("0", false);
    manager.free("1", false);

    let k = |i: usize, g: usize| BlockHashWithGroupId::new(hashes[i].clone(), g);
    // Evict the blocks outside sliding window, does not affect the hit length.
    partial_request_hit(&mut manager, "2", &all_tokens, &[k(0, 1), k(0, 2)], 3);
    // Evict the first block of full attention, makes total cache miss.
    partial_request_hit(&mut manager, "3", &all_tokens, &[k(0, 0)], 0);
    // Evict the last block of all layers, reduces the hit length to 2.
    partial_request_hit(
        &mut manager,
        "4",
        &all_tokens,
        &[k(2, 0), k(2, 1), k(2, 2)],
        2,
    );
    // Evict the last block of full attention, reduces the hit length to 2.
    partial_request_hit(&mut manager, "5", &all_tokens, &[k(2, 0)], 2);
    // Evict the last block of sliding window, reduces the hit length to 2.
    partial_request_hit(&mut manager, "6", &all_tokens, &[k(2, 1)], 2);
    partial_request_hit(&mut manager, "7", &all_tokens, &[k(2, 2)], 2);
    // Evict different set of blocks for full attention and sliding window makes
    // total cache miss.
    partial_request_hit(
        &mut manager,
        "8",
        &all_tokens,
        &[k(2, 0), k(0, 1), k(0, 2)],
        0,
    );
}

/// `test_sliding_window_possible_cached_prefix`.
#[test]
fn sliding_window_possible_cached_prefix() {
    let bs = 2;
    let spec = swa_spec(bs, 4);
    let mut pool = BlockPool::new(100, true, bs, false, None, 0, MEDIUM_GPU);
    let mut run_one_case = |cached: &[bool], expect_length: usize| {
        let hashes: Vec<BlockHash> = (0..cached.len())
            .map(|i| BlockHash::from_slice(i.to_string().as_bytes()))
            .collect();
        pool.cached_block_hash_to_block.clear();
        for (i, (h, &is_cached)) in hashes.iter().zip(cached).enumerate() {
            if is_cached {
                pool.cached_block_hash_to_block
                    .insert(BlockHashWithGroupId::new(h.clone(), 0), (i + 10) as BlockId);
            }
        }
        let (computed, _) = ManagerType::SlidingWindow.find_longest_cache_hit(
            &hashes,
            hashes.len() * bs,
            &[0],
            &pool,
            &spec,
            false,
            bs,
            1,
            1,
        );
        let computed = &computed[0];
        assert_eq!(computed.len(), expect_length, "{cached:?}");
        assert!(computed[..expect_length.saturating_sub(2)].iter().all(|&b| b == NULL_BLOCK_ID));
        for i in 0..2.min(expect_length) {
            let idx = expect_length - i - 1;
            assert_eq!(computed[idx], (idx + 10) as BlockId);
        }
    };
    let (t, f) = (true, false);
    run_one_case(&[f; 10], 0);
    run_one_case(&[t], 1);
    run_one_case(&[t, f], 1);
    run_one_case(&[t, t], 2);
    run_one_case(&[t, t, f], 2);
    run_one_case(&[t, t, t], 3);
    run_one_case(&[t, t, t, f], 3);
    run_one_case(&[t, t, f, t, f, f, t, t, f, t, t, t], 12);
    run_one_case(&[t, t, f, t, f, f, t, t, f, f, f], 8);
    run_one_case(&[t, t, f, t, f, f, t, t, f, f, f, t], 8);
}

/// `test_sliding_window_cache_hit_with_finer_hash_alignment`.
#[test]
fn sliding_window_cache_hit_with_finer_hash_alignment() {
    let spec = swa_spec(4, 8);
    let mut pool = BlockPool::new(100, true, 2, false, None, 0, MEDIUM_GPU);
    let hashes: Vec<BlockHash> =
        (0..4).map(|i: i32| BlockHash::from_slice(i.to_string().as_bytes())).collect();
    pool.cached_block_hash_to_block
        .insert(BlockHashWithGroupId::new(hashes[1].clone(), 0), 10);
    pool.cached_block_hash_to_block
        .insert(BlockHashWithGroupId::new(hashes[3].clone(), 0), 11);
    let (computed, hit_length) = ManagerType::SlidingWindow.find_longest_cache_hit(
        &hashes,
        8,
        &[0],
        &pool,
        &spec,
        false,
        2,
        1,
        1,
    );
    assert_eq!(hit_length, 8);
    assert_eq!(computed[0], vec![10, 11]);
}

fn manager_with_table(spec: KvCacheSpec, ids: &[BlockId]) -> (SingleTypeKvCacheManager, BlockPool) {
    let pool = BlockPool::new(2000, true, spec.block_size, false, None, 0, MEDIUM_GPU);
    let mut manager = SingleTypeKvCacheManager::new(
        spec,
        KvCacheGroupRole::Default,
        8192,
        8192,
        true,
        0,
        2,
        1,
        1,
        false,
    )
    .unwrap();
    manager.req_to_blocks.insert("test".to_owned(), ids.to_vec());
    (manager, pool)
}

/// `test_sliding_window_remove_skipped_blocks`.
#[test]
fn sliding_window_remove_skipped_blocks() {
    let original: Vec<BlockId> = (1000..1011).collect();
    let (mut manager, mut pool) = manager_with_table(swa_spec(2, 4), &original);
    let table = |m: &SingleTypeKvCacheManager| m.req_blocks("test").unwrap().to_vec();
    let nulls_then = |n: usize| -> Vec<BlockId> {
        std::iter::repeat_n(0, n).chain(original[n..].iter().copied()).collect()
    };
    manager.remove_skipped_blocks(&mut pool, "test", 0, None);
    assert_eq!(table(&manager), original);
    // As block 1000 also contains token 1 that is in the sliding window, block
    // 1000 cannot be removed.
    manager.remove_skipped_blocks(&mut pool, "test", 4, None);
    assert_eq!(table(&manager), original);
    manager.remove_skipped_blocks(&mut pool, "test", 5, None);
    assert_eq!(table(&manager), nulls_then(1));
    manager.remove_skipped_blocks(&mut pool, "test", 6, None);
    assert_eq!(table(&manager), nulls_then(1));
    manager.remove_skipped_blocks(&mut pool, "test", 7, None);
    assert_eq!(table(&manager), nulls_then(2));
    manager.remove_skipped_blocks(&mut pool, "test", 11, None);
    assert_eq!(table(&manager), nulls_then(4));
}

/// `test_chunked_local_attention_remove_skipped_blocks`.
#[test]
fn chunked_local_attention_remove_skipped_blocks() {
    let original: Vec<BlockId> = (1000..1011).collect();
    let spec = KvCacheSpec::new(
        KvCacheSpecType::ChunkedLocalAttention {
            attention_chunk_size: 4,
        },
        2,
    );
    let (mut manager, mut pool) = manager_with_table(spec, &original);
    let prefix =
        |m: &SingleTypeKvCacheManager, n: usize| m.req_blocks("test").unwrap()[..n].to_vec();
    manager.remove_skipped_blocks(&mut pool, "test", 0, None);
    assert_eq!(prefix(&manager, 11), original);
    manager.remove_skipped_blocks(&mut pool, "test", 4, None);
    assert_eq!(prefix(&manager, 2), vec![0, 0]);
    manager.remove_skipped_blocks(&mut pool, "test", 6, None);
    assert_eq!(prefix(&manager, 3), vec![0, 0, 1002]);
    manager.remove_skipped_blocks(&mut pool, "test", 12, None);
    assert_eq!(prefix(&manager, 7), vec![0, 0, 0, 0, 0, 0, 1006]);
}

/// `test_rswa_remove_skipped_blocks_gap_range` (gap blocks only).
#[test]
fn rswa_remove_skipped_blocks_gap_range() {
    let original: Vec<BlockId> = (1000..1012).collect();
    let spec = KvCacheSpec::new(KvCacheSpecType::Rswa { rswa_window: 4 }, 2);
    let (mut manager, mut pool) = manager_with_table(spec, &original);
    // Prompt of 5 tokens: blocks 0..3 are the prefix; window of 4 tokens.
    manager.remove_skipped_blocks(&mut pool, "test", 20, Some(5));
    expect![[r#"
        [
            1000,
            1001,
            1002,
            0,
            0,
            0,
            0,
            0,
            1008,
            1009,
            1010,
            1011,
        ]
    "#]]
    .assert_debug_eq(&manager.req_blocks("test").unwrap());
}

/// `test_kv_cache_events`.
#[test]
fn kv_cache_events() {
    let bs = 16;
    let blocks_to_cache = 3;
    let mut manager = make_manager(
        make_config(blocks_to_cache + 1, vec![full_spec(bs)]),
        bs,
        |c| {
            c.enable_kv_cache_events = true;
        },
    );
    let num_tokens = bs * blocks_to_cache;
    let req0 = make_request("0", (0..num_tokens as i32).collect(), bs);
    alloc(&mut manager, &req0, num_tokens, 0, None).unwrap();
    let events = manager.take_events();
    let KvCacheEvent::BlockStored(stored) = events.last().unwrap() else {
        panic!("expected BlockStored");
    };
    assert_eq!(stored.block_hashes.len(), blocks_to_cache);
    assert_eq!(
        manager.block_pool().cached_block_hash_to_block.len(),
        blocks_to_cache
    );
    assert_eq!(stored.token_ids.len(), num_tokens);
    assert_eq!(stored.kv_cache_spec_kind, Some("full_attention"));
    assert_eq!(stored.parent_block_hash, None);
    assert_eq!(stored.group_idx, Some(0));
    let stored_hashes = stored.block_hashes.clone();

    manager.free("0", false);
    let req1 = make_request("1", (0..num_tokens as i32).map(|t| t + 1000).collect(), bs);
    alloc(&mut manager, &req1, num_tokens, 0, None).unwrap();
    let events = manager.take_events();
    assert_eq!(events.len(), blocks_to_cache + 1);
    // Blocks freed in reverse order are evicted tail-first.
    let removed: Vec<_> = events[..blocks_to_cache]
        .iter()
        .map(|e| match e {
            KvCacheEvent::BlockRemoved(r) => r.block_hashes[0].clone(),
            _ => panic!("expected BlockRemoved"),
        })
        .collect();
    assert_eq!(removed, stored_hashes.into_iter().rev().collect::<Vec<_>>());

    manager.free("1", false);
    assert!(manager.reset_prefix_cache());
    let events = manager.take_events();
    assert_eq!(
        events.last(),
        Some(&KvCacheEvent::AllBlocksCleared(AllBlocksCleared))
    );
    assert!(manager.block_pool().cached_block_hash_to_block.is_empty());
}

/// `test_reset_prefix_cache`.
#[test]
fn reset_prefix_cache_requires_all_blocks_free() {
    let bs = 16;
    let mut manager = make_manager(make_config(11, vec![full_spec(bs)]), bs, |_| {});
    let req = make_request("0", (0..40).collect(), bs);
    alloc(&mut manager, &req, 40, 0, None).unwrap();
    assert!(!manager.reset_prefix_cache());
    manager.free("0", false);
    assert!(manager.reset_prefix_cache());
    assert!(manager.block_pool().blocks().iter().all(|b| b.block_hash.is_none()));
}

/// `test_hybrid_cache_mamba_align_shared_prefix_detection`.
#[test]
fn hybrid_cache_mamba_align_shared_prefix_detection() {
    let bs = 16;
    let mut manager = make_manager(
        make_config(30, vec![full_spec(bs), mamba_spec(bs)]),
        bs,
        |_| {},
    );
    let prefix: Vec<i32> = (0..3).flat_map(|i| [i; 16]).collect();

    let req0 = make_request("0", prefix.clone(), bs);
    let (computed, n, boundary) = manager.get_computed_blocks(&req0, None);
    assert_eq!((n, boundary), (0, 0));
    alloc(&mut manager, &req0, 3 * bs, 0, Some(&computed)).unwrap();

    let req1 = make_request("1", [prefix.clone(), vec![100; 7]].concat(), bs);
    let (computed, n, boundary) = manager.get_computed_blocks(&req1, None);
    assert_eq!((n, boundary), (3 * bs, 0));
    assert_eq!(computed.blocks, vec![vec![1, 2, 3], vec![0, 0, 4]]);
    // The mamba hit was cached this step, so the allocation is deferred.
    assert!(alloc(&mut manager, &req1, 7, 3 * bs, Some(&computed)).is_none());

    let mut tokens = prefix.clone();
    *tokens.last_mut().unwrap() = 101;
    let req2 = make_request("2", tokens, bs);
    let (computed, n, boundary) = manager.get_computed_blocks(&req2, None);
    // mamba_align doesn't cache intermediate blocks
    assert_eq!(n, 0);
    // boundary == junction: 2 shared blocks (num_computed 0 + 2 uncached blocks).
    assert_eq!(boundary, 2 * bs);
    let blocks = alloc(&mut manager, &req2, 3 * bs, 0, Some(&computed)).unwrap();
    assert_eq!(blocks.blocks, vec![vec![5, 6, 7], vec![0, 0, 8]]);
    assert_eq!(manager.block_pool().get_num_free_blocks(), 21);
}

/// `test_eagle_hybrid_mamba_hits_partial_prompt_boundary`: fine-grained
/// (sub-block) prefix-cache hits with `hash_block_size < block_size`.
#[test]
fn hybrid_mamba_hits_partial_prompt_boundary() {
    let hbs = 16;
    let config = make_config(200, vec![full_spec(hbs), mamba_spec(544)]);
    let mut manager = make_manager(config, hbs, |c| c.num_hash_lookahead_tokens = 1);
    assert!(manager.coordinator.enable_partial_hash_hits);
    let tokens: Vec<i32> = (0..1249).collect();

    let mut first = make_request_with_lookahead("first", tokens.clone(), hbs, 1);
    assert!(alloc(&mut manager, &first, 1248, 0, None).is_some());
    first.num_computed_tokens = 1248;
    manager.new_step_starts();
    assert!(alloc(&mut manager, &first, 1, 0, None).is_some());
    first.num_computed_tokens = 1249;
    manager.new_step_starts();
    manager.free("first", false);

    let second = make_request_with_lookahead("second", tokens, hbs, 1);
    let (_, n, _) = manager.get_computed_blocks(&second, None);
    assert_eq!(n, 1248);
    // Per-group lookup must not add the legacy EAGLE drop margin.
    let (_, per_group_hits) = manager
        .coordinator
        .find_longest_cache_hit_per_group(&second.block_hashes, 2 * hbs);
    assert_eq!(per_group_hits[0], 2 * hbs);
}

#[test]
fn hisparse_configs_without_host_source_are_rejected() {
    let spec = KvCacheSpec::new(
        KvCacheSpecType::HiSparseHot {
            blocks_per_request: 2,
        },
        16,
    );
    let err = KvCacheManager::new(
        make_config(10, vec![full_spec(16), spec]),
        KvCacheManagerConfig::new(64, 16, 16),
    )
    .unwrap_err();
    expect![[r#"
        InvalidConfig {
            message: "No HiSparse cache group is configured.",
        }
    "#]]
    .assert_debug_eq(&err);
    let mut config = make_config(10, vec![full_spec(16)]);
    config.kv_cache_groups[0].role = KvCacheGroupRole::HiSparseSource;
    let err = KvCacheManager::new(config, KvCacheManagerConfig::new(64, 16, 16)).unwrap_err();
    expect![[r#"
        InvalidConfig {
            message: "HiSparse host group needs host capacity.",
        }
    "#]]
    .assert_debug_eq(&err);
}

#[test]
fn allocate_slots_rejects_empty_step() {
    let bs = 16;
    let mut manager = make_manager(make_config(11, vec![full_spec(bs)]), bs, |_| {});
    let req = make_request("0", (0..10).collect(), bs);
    let err = manager.allocate_slots(&req, 0, AllocateSlotsArgs::default()).unwrap_err();
    assert!(matches!(err, KvCacheError::InvalidArgument { .. }));
}
