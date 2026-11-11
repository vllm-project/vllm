// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Per-block hot paths of the `MooncakeStoreConnector`
//! (`vllm/distributed/kv_transfer/kv_connector/v1/mooncake_store/`).
//!
//! - [`StoreCoordinator`] mirrors `coordinator.py`'s
//!   `MooncakeStoreCoordinator`: the external-store cache-hit search and the
//!   per-group lookup / load masks.
//! - [`StoreLookup`] mirrors the worker-side prefix lookup
//!   (`MooncakeStoreWorker.lookup`, `_longest_hit`, `_tail_key_boundaries`),
//!   split around the store's `batch_is_exist` RPC, which stays in Python.
//!
//! The Python implementation is the behavioral reference: for the same inputs
//! these return bitwise identical results.

use std::collections::BTreeSet;

use rustc_hash::FxHashSet;
use thiserror::Error;

use vllm_engine_core::hash::BlockHash;
use vllm_engine_core::kv_cache::{
    BlockId, CachedBlockLookup, CachedBlocks, KvCacheSpec, ManagerType, NULL_BLOCK_ID, SpecGroup,
    eagle_proof_margin,
};

/// Dummy id of a present block (`ExternalCachedBlockPool._present_block`).
const PRESENT_BLOCK_ID: BlockId = 1;

/// Errors of the Mooncake store hot paths. Python raises `AssertionError` for
/// each of them.
#[derive(Debug, Error)]
pub enum MooncakeStoreError {
    #[error("No tail key found for cache group {group_id} at hit length {hit_length}")]
    NoTailKey { group_id: usize, hit_length: usize },
    #[error("{message}")]
    InvalidArgument { message: String },
}

/// `ExternalCachedBlockPool`: a duck-typed block pool backed by a
/// `(group_id, hash)` exists set.
///
/// `exists = None` is used on the recv side where the hit length is already
/// determined and each spec's manager only applies its own mask.
#[derive(Debug)]
pub struct ExternalCachedBlockPool<'a> {
    hash_block_size: usize,
    exists: Option<FxHashSet<(usize, &'a [u8])>>,
}

impl<'a> ExternalCachedBlockPool<'a> {
    pub fn new(hash_block_size: usize, exists: Option<FxHashSet<(usize, &'a [u8])>>) -> Self {
        Self {
            hash_block_size,
            exists,
        }
    }

    /// Return whether a group has a loadable key for this hash.
    pub fn contains(&self, group_id: usize, block_hash: &BlockHash) -> bool {
        self.exists
            .as_ref()
            .is_some_and(|exists| exists.contains(&(group_id, block_hash.as_bytes())))
    }
}

impl CachedBlockLookup for ExternalCachedBlockPool<'_> {
    fn hash_block_size(&self) -> usize {
        self.hash_block_size
    }

    fn get_cached_block(
        &self,
        block_hash: &BlockHash,
        kv_cache_group_ids: &[usize],
    ) -> Option<CachedBlocks> {
        // Mirrors BlockPool.get_cached_block: hit only when every group_id
        // (groups sharing a spec) has the hash cached.
        if let Some(exists) = &self.exists
            && !kv_cache_group_ids
                .iter()
                .all(|&group_id| exists.contains(&(group_id, block_hash.as_bytes())))
        {
            return None;
        }
        Some(CachedBlocks::from_elem(
            PRESENT_BLOCK_ID,
            kv_cache_group_ids.len(),
        ))
    }
}

/// Per-group masks; `None` is the all-True sentinel.
pub type GroupMasks = Vec<Option<Vec<bool>>>;

/// Mirror of `HybridKVCacheCoordinator.find_longest_cache_hit` over an
/// [`ExternalCachedBlockPool`] (`MooncakeStoreCoordinator`).
///
/// Construction takes the Python coordinator's resolved state (the spec
/// groups it batched and their EAGLE flags) rather than re-deriving it, since
/// batching compares Python spec dataclasses on fields this port does not
/// model.
#[derive(Clone, Debug)]
pub struct StoreCoordinator {
    /// `_coverage_spec` of each KV cache group (DCP already resolved) and its
    /// registered manager.
    group_specs: Vec<(KvCacheSpec, ManagerType)>,
    /// Prefix-cacheable groups batched by spec, full attention first.
    attention_groups: Vec<SpecGroup>,
    eagle_group_ids: BTreeSet<usize>,
    hash_block_size: usize,
    lcm_block_size: usize,
    enable_partial_hash_hits: bool,
}

impl StoreCoordinator {
    pub fn new(
        group_specs: Vec<(KvCacheSpec, ManagerType)>,
        attention_groups: Vec<SpecGroup>,
        eagle_group_ids: BTreeSet<usize>,
        hash_block_size: usize,
        lcm_block_size: usize,
        enable_partial_hash_hits: bool,
    ) -> Self {
        Self {
            group_specs,
            attention_groups,
            eagle_group_ids,
            hash_block_size,
            lcm_block_size,
            enable_partial_hash_hits,
        }
    }

    pub fn num_groups(&self) -> usize {
        self.group_specs.len()
    }

    pub fn hash_block_size(&self) -> usize {
        self.hash_block_size
    }

    pub fn enable_partial_hash_hits(&self) -> bool {
        self.enable_partial_hash_hits
    }

    fn alignment_tokens(&self) -> usize {
        if self.enable_partial_hash_hits {
            self.hash_block_size
        } else {
            self.lcm_block_size
        }
    }

    pub fn align_lookup_length(&self, length: usize) -> usize {
        let alignment = self.alignment_tokens();
        length / alignment * alignment
    }

    /// Returns `(load_mask_per_group, hit_length)`. `mask[g][i]` is True iff
    /// group `g` populates physical block `i` (e.g. SWA and Mamba tail-only);
    /// recv-side callers skip False slots.
    ///
    /// `apply_eagle` controls whether the per-spec `use_eagle` last-block
    /// pop is applied. Lookup callers want it (the drafter requires
    /// recomputing the last block); load-mask callers must not, because
    /// `token_len` already reflects the eagle-pruned hit length and a second
    /// pop would leave the trailing block unloaded.
    pub fn find_longest_cache_hit(
        &self,
        block_hashes: &[BlockHash],
        max_length: usize,
        cached_block_pool: &ExternalCachedBlockPool,
        apply_eagle: bool,
    ) -> (Vec<Vec<bool>>, usize) {
        let (blocks_per_group, hit_length) =
            self.find_hit_blocks(block_hashes, max_length, cached_block_pool, apply_eagle);
        let masks = blocks_per_group
            .into_iter()
            .map(|blocks| blocks.into_iter().map(|b| b != NULL_BLOCK_ID).collect())
            .collect();
        (masks, hit_length)
    }

    /// Per-group load masks: `mask[g][i]` is True iff group `g`'s spec would
    /// populate physical block `i` at length `token_len` (e.g. SWA / Mamba
    /// tail-only).
    pub fn load_mask(&self, block_hashes: &[BlockHash], token_len: usize) -> Vec<Vec<bool>> {
        // Lookup has already applied Eagle's last-block drop.
        let pool = ExternalCachedBlockPool::new(self.hash_block_size, None);
        self.find_longest_cache_hit(block_hashes, token_len, &pool, false).0
    }

    /// Per-group lookup masks.
    ///
    /// `mask[g][i]` is True iff physical block `i` of group `g` should be
    /// looked up as an aligned hit boundary. `None` is the all-True sentinel.
    pub fn lookup_mask(&self, aligned_token_len: usize) -> GroupMasks {
        assert!(
            aligned_token_len.is_multiple_of(self.alignment_tokens()),
            "aligned_token_len ({aligned_token_len}) must be a multiple of {}",
            self.alignment_tokens()
        );
        self.group_specs
            .iter()
            .enumerate()
            .map(|(g_idx, (spec, manager_type))| {
                let end_block = aligned_token_len / spec.block_size;
                let mask = manager_type.reachable_block_mask(
                    0,
                    end_block,
                    Some(self.lcm_block_size),
                    spec,
                    self.eagle_group_ids.contains(&g_idx),
                    None,
                    &[],
                    // Worker initialization has already resolved DCP in this spec.
                    1,
                    None,
                );
                if let Some(mask) = &mask {
                    assert_eq!(mask.len(), end_block);
                }
                mask
            })
            .collect()
    }

    /// Converge registered managers' hits as in HybridKVCacheCoordinator.
    fn find_hit_blocks(
        &self,
        block_hashes: &[BlockHash],
        max_length: usize,
        cached_block_pool: &ExternalCachedBlockPool,
        apply_eagle: bool,
    ) -> (Vec<Vec<BlockId>>, usize) {
        let alignment_tokens = self.alignment_tokens();
        let num_groups = self.group_specs.len();
        if let [group] = self.attention_groups.as_slice() {
            let (hit_blocks, hit_length) = group.manager_type.find_longest_cache_hit(
                block_hashes,
                max_length,
                &group.group_ids,
                cached_block_pool,
                &group.spec,
                apply_eagle && group.use_eagle,
                alignment_tokens,
                1,
                1,
            );
            let mut blocks_by_group = vec![Vec::new(); num_groups];
            for (&gid, blks) in group.group_ids.iter().zip(hit_blocks) {
                blocks_by_group[gid] = blks;
            }
            return (blocks_by_group, hit_length);
        }

        let mut hit_length = max_length;
        let mut hit_blocks_by_group: Vec<Option<Vec<BlockId>>> = vec![None; num_groups];
        let mut hit_length_by_group = vec![0usize; num_groups];

        let is_simple_hybrid =
            self.attention_groups.len() == 2 && self.attention_groups[0].spec.is_full_attention();
        let mut eagle_verified: BTreeSet<usize> = BTreeSet::new();

        loop {
            let mut curr_hit_length = hit_length;

            for (idx, group) in self.attention_groups.iter().enumerate() {
                let first_group_id = group.group_ids[0];
                if group.spec.is_full_attention() && hit_blocks_by_group[first_group_id].is_some() {
                    curr_hit_length = curr_hit_length.min(hit_length_by_group[first_group_id]);
                    continue;
                }

                let drop_eagle_block =
                    apply_eagle && group.use_eagle && !eagle_verified.contains(&idx);
                let mut group_max_length = curr_hit_length;
                // No eagle peek margin for a recurrent (Mamba) group: its finder
                // never drops a block, so a widened bound would match past the
                // attention-verified hit and resume from speculative state (#43559).
                if drop_eagle_block && !group.spec.is_mamba() {
                    let eagle_margin = eagle_proof_margin(
                        group.spec.block_size,
                        self.hash_block_size,
                        self.enable_partial_hash_hits
                            && group.manager_type.supports_fine_grained_hash_lookup(),
                    );
                    group_max_length = (curr_hit_length + eagle_margin)
                        .min(block_hashes.len() * self.hash_block_size);
                }
                let (hit_blocks, new_hit_length) = group.manager_type.find_longest_cache_hit(
                    block_hashes,
                    group_max_length,
                    &group.group_ids,
                    cached_block_pool,
                    &group.spec,
                    drop_eagle_block,
                    alignment_tokens,
                    1,
                    1,
                );
                if drop_eagle_block {
                    eagle_verified.insert(idx);
                } else if new_hit_length < curr_hit_length {
                    eagle_verified.clear();
                }
                curr_hit_length = new_hit_length;
                for (&gid, blocks) in group.group_ids.iter().zip(hit_blocks) {
                    hit_blocks_by_group[gid] = Some(blocks);
                    hit_length_by_group[gid] = new_hit_length;
                }
            }

            if curr_hit_length >= hit_length {
                break;
            }
            hit_length = curr_hit_length;
            if is_simple_hybrid {
                break;
            }
        }

        // Truncate full-attention hit_blocks to final converged length;
        // other specs already trim themselves inside their hit logic. cdiv keeps
        // the partial tail block when hit_length is not block-aligned.
        for group in &self.attention_groups {
            if !group.spec.is_full_attention() {
                continue;
            }
            let num_blocks = hit_length.div_ceil(group.spec.block_size);
            for &gid in &group.group_ids {
                hit_blocks_by_group[gid]
                    .as_mut()
                    .expect("full attention group was looked up")
                    .truncate(num_blocks);
            }
        }

        (
            hit_blocks_by_group.into_iter().map(Option::unwrap_or_default).collect(),
            hit_length,
        )
    }
}

/// `TailKeyBoundary`: the hash boundary used to key a group's tail block in
/// the store.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TailKeyBoundary {
    pub group_id: usize,
    pub num_tokens: usize,
}

/// `MooncakeLookupResult`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LookupResult {
    pub hit_length: usize,
    pub tail_key_boundaries: Vec<TailKeyBoundary>,
}

/// The lookup-relevant state of one KV cache group on the lookup worker.
#[derive(Clone, Debug)]
pub struct LookupGroup {
    pub prefix_cacheable: bool,
    /// `kv_cache_spec.block_size` of the (DCP-resolved) group spec.
    pub spec_block_size: usize,
    /// `ChunkedTokenDatabase.chunk_size`.
    pub chunk_size: usize,
    /// Rank-namespace key prefixes that must all exist for a block to be
    /// reusable (`_lookup_key_prefixes`).
    pub key_prefixes: Vec<String>,
}

/// One `(group, hash, end_token)` candidate of a lookup; its keys are the
/// group's key prefixes, in order.
#[derive(Clone, Copy, Debug)]
struct Candidate {
    group: usize,
    hash_idx: usize,
    end_token: usize,
}

/// The store keys a lookup probes and what they stand for.
#[derive(Debug)]
pub struct LookupCandidates {
    num_tokens: usize,
    token_len: usize,
    meta: Vec<Candidate>,
    /// Candidate keys expanded across rank namespaces, in probe order.
    pub keys: Vec<String>,
}

/// The worker-side prefix lookup (`MooncakeStoreWorker.lookup`), split around
/// the existence RPC: [`Self::candidates`] builds the keys to probe and
/// [`Self::resolve`] turns the probe results into the hit.
#[derive(Clone, Debug)]
pub struct StoreLookup {
    coord: StoreCoordinator,
    groups: Vec<LookupGroup>,
}

impl StoreLookup {
    pub fn new(
        coord: StoreCoordinator,
        groups: Vec<LookupGroup>,
    ) -> Result<Self, MooncakeStoreError> {
        if groups.len() != coord.num_groups() {
            return Err(MooncakeStoreError::InvalidArgument {
                message: format!(
                    "{} lookup groups for {} coordinator groups",
                    groups.len(),
                    coord.num_groups()
                ),
            });
        }
        Ok(Self { coord, groups })
    }

    pub fn coord(&self) -> &StoreCoordinator {
        &self.coord
    }

    /// Build per-(group, hash) candidate keys expanded across rank
    /// namespaces. `None` is a miss without probing the store.
    pub fn candidates(
        &self,
        num_tokens: usize,
        block_hashes: &[BlockHash],
    ) -> Option<LookupCandidates> {
        let hash_block_size = self.coord.hash_block_size;
        let token_len = self.coord.align_lookup_length(num_tokens);
        if block_hashes.is_empty() || token_len == 0 {
            return None;
        }
        let fine_grained = self.coord.enable_partial_hash_hits;
        let lookup_masks = (!fine_grained).then(|| self.coord.lookup_mask(token_len));
        let mut keys = Vec::new();
        let mut meta = Vec::new();
        let mut hex = String::new();
        for (g_idx, group) in self.groups.iter().enumerate() {
            if !group.prefix_cacheable {
                continue;
            }
            let lookup_mask = lookup_masks.as_ref().and_then(|m| m[g_idx].as_ref());
            let unit_size = if fine_grained {
                hash_block_size
            } else {
                group.chunk_size
            };
            let max_units = token_len.min(block_hashes.len() * hash_block_size) / unit_size;
            for unit_id in 0..max_units {
                let block_id = unit_id * unit_size / group.spec_block_size;
                if let Some(mask) = lookup_mask
                    && !mask.get(block_id).copied().unwrap_or(false)
                {
                    continue;
                }
                let end_token = (unit_id + 1) * unit_size;
                let hash_idx = end_token / hash_block_size - 1;
                hex.clear();
                push_hex(&mut hex, block_hashes[hash_idx].as_bytes());
                for key_prefix in &group.key_prefixes {
                    keys.push(build_key_string(key_prefix, &hex));
                }
                meta.push(Candidate {
                    group: g_idx,
                    hash_idx,
                    end_token,
                });
            }
        }
        if keys.is_empty() {
            return None;
        }
        Some(LookupCandidates {
            num_tokens,
            token_len,
            meta,
            keys,
        })
    }

    /// Turn the existence results of `candidates.keys` (`1` = present) into
    /// the longest prefix hit and its tail-key boundaries.
    pub fn resolve(
        &self,
        block_hashes: &[BlockHash],
        candidates: &LookupCandidates,
        exists: &[i64],
    ) -> Result<LookupResult, MooncakeStoreError> {
        if exists.len() < candidates.keys.len() {
            return Err(MooncakeStoreError::InvalidArgument {
                message: format!(
                    "{} existence results for {} keys",
                    exists.len(),
                    candidates.keys.len()
                ),
            });
        }
        // A (group, hash) is "present" only when every namespace that will be
        // loaded has it (per-group count: sharded groups need every rank's
        // shard, replicated groups one namespace per unique KV head).
        let mut exists_set = FxHashSet::default();
        let mut pos = 0;
        let mut previous_page = None;
        let mut preceding_chunks_present = true;
        for c in &candidates.meta {
            let group = &self.groups[c.group];
            let page = (c.group, (c.end_token - 1) / group.spec_block_size);
            if previous_page != Some(page) {
                preceding_chunks_present = true;
                previous_page = Some(page);
            }
            let count = group.key_prefixes.len();
            let present = exists[pos..pos + count].iter().all(|&r| r == 1);
            // A page-prefix hit covers every Store chunk before its final chunk.
            if preceding_chunks_present && present {
                exists_set.insert((c.group, block_hashes[c.hash_idx].as_bytes()));
            }
            if c.end_token.is_multiple_of(group.chunk_size) {
                preceding_chunks_present &= present;
            }
            pos += count;
        }
        self.longest_hit(
            candidates.num_tokens,
            candidates.token_len,
            block_hashes,
            exists_set,
        )
    }

    /// Longest prefix hit over the stored group/hash pairs (`_longest_hit`).
    fn longest_hit(
        &self,
        num_tokens: usize,
        token_len: usize,
        block_hashes: &[BlockHash],
        exists_set: FxHashSet<(usize, &[u8])>,
    ) -> Result<LookupResult, MooncakeStoreError> {
        let pool = ExternalCachedBlockPool::new(self.coord.hash_block_size, Some(exists_set));
        let (mut load_masks, mut hit_length) =
            self.coord.find_longest_cache_hit(block_hashes, token_len, &pool, true);
        if hit_length >= num_tokens {
            let usable_length = self.coord.align_lookup_length(num_tokens - 1);
            if usable_length == 0 {
                return Ok(LookupResult::default());
            }
            (load_masks, hit_length) =
                self.coord.find_longest_cache_hit(block_hashes, usable_length, &pool, true);
        }
        let tail_key_boundaries =
            self.tail_key_boundaries(block_hashes, hit_length, &pool, &load_masks)?;
        Ok(LookupResult {
            hit_length,
            tail_key_boundaries,
        })
    }

    /// Return the hash boundary used to store each group's tail block.
    ///
    /// With fine-grained prefix matching, `hit_length` may fall within a
    /// physical cache block and may not align with the hash boundary used to
    /// store that block. For each KV-cache group, return the token boundary
    /// whose hash was used as the store key.
    fn tail_key_boundaries(
        &self,
        block_hashes: &[BlockHash],
        hit_length: usize,
        pool: &ExternalCachedBlockPool,
        load_masks: &[Vec<bool>],
    ) -> Result<Vec<TailKeyBoundary>, MooncakeStoreError> {
        if hit_length == 0 {
            return Ok(Vec::new());
        }
        let hash_block_size = self.coord.hash_block_size;
        let mut boundaries = Vec::new();
        let hit_boundary_hash_idx = hit_length / hash_block_size - 1;
        for (group_id, group) in self.groups.iter().enumerate() {
            if load_masks[group_id].last() != Some(&true) {
                continue;
            }
            let chunk_id = hit_length.div_ceil(group.chunk_size) - 1;
            let mut boundary_tokens = hit_length;
            let contains_hit_boundary =
                pool.contains(group_id, &block_hashes[hit_boundary_hash_idx]);
            if !self.coord.enable_partial_hash_hits && !contains_hit_boundary {
                return Err(MooncakeStoreError::NoTailKey {
                    group_id,
                    hit_length,
                });
            }
            if !contains_hit_boundary {
                let next_chunk_hash_idx =
                    ((chunk_id + 1) * group.chunk_size / hash_block_size).min(block_hashes.len());
                boundary_tokens = (hit_boundary_hash_idx + 1..next_chunk_hash_idx)
                    .find(|&hash_idx| pool.contains(group_id, &block_hashes[hash_idx]))
                    .map(|hash_idx| (hash_idx + 1) * hash_block_size)
                    .ok_or(MooncakeStoreError::NoTailKey {
                        group_id,
                        hit_length,
                    })?;
            }
            boundaries.push(TailKeyBoundary {
                group_id,
                num_tokens: boundary_tokens,
            });
        }
        Ok(boundaries)
    }
}

/// `PoolKey.build_key_string`.
fn build_key_string(key_prefix: &str, chunk_hash_hex: &str) -> String {
    let mut key = String::with_capacity(key_prefix.len() + 1 + chunk_hash_hex.len());
    key.push_str(key_prefix);
    key.push('@');
    key.push_str(chunk_hash_hex);
    key
}

/// Append the lowercase hex of `bytes` (`bytes.hex()`).
fn push_hex(out: &mut String, bytes: &[u8]) {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    out.reserve(bytes.len() * 2);
    for &b in bytes {
        out.push(DIGITS[usize::from(b >> 4)] as char);
        out.push(DIGITS[usize::from(b & 0xf)] as char);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vllm_engine_core::kv_cache::{KvCacheSpec, KvCacheSpecType};

    fn hashes(n: usize) -> Vec<BlockHash> {
        (0..n).map(|i| BlockHash::from_slice(&[i as u8; 4])).collect()
    }

    fn full_attention_lookup(block_size: usize) -> StoreLookup {
        let spec = KvCacheSpec::new(KvCacheSpecType::FullAttention, block_size);
        let coord = StoreCoordinator::new(
            vec![(spec.clone(), ManagerType::FullAttention)],
            vec![SpecGroup {
                spec,
                group_ids: vec![0],
                manager_type: ManagerType::FullAttention,
                use_eagle: false,
            }],
            BTreeSet::new(),
            block_size,
            block_size,
            false,
        );
        let group = LookupGroup {
            prefix_cacheable: true,
            spec_block_size: block_size,
            chunk_size: block_size,
            key_prefixes: vec!["m@tp_rank:0@group:0".to_owned()],
        };
        StoreLookup::new(coord, vec![group]).unwrap()
    }

    /// A full hit is re-derived below the request end so the last token is
    /// recomputed; a missing key truncates the hit at that block.
    #[test]
    fn lookup_stops_at_first_missing_block_and_recomputes_last_token() {
        let lookup = full_attention_lookup(4);
        let block_hashes = hashes(4);
        let cands = lookup.candidates(16, &block_hashes).unwrap();
        expect_test::expect![[r#"
            [
                "m@tp_rank:0@group:0@00000000",
                "m@tp_rank:0@group:0@01010101",
                "m@tp_rank:0@group:0@02020202",
                "m@tp_rank:0@group:0@03030303",
            ]
        "#]]
        .assert_debug_eq(&cands.keys);

        let all = lookup.resolve(&block_hashes, &cands, &[1, 1, 1, 1]).unwrap();
        let gap = lookup.resolve(&block_hashes, &cands, &[1, 1, 0, 1]).unwrap();
        expect_test::expect![[r#"
            (
                LookupResult {
                    hit_length: 12,
                    tail_key_boundaries: [
                        TailKeyBoundary {
                            group_id: 0,
                            num_tokens: 12,
                        },
                    ],
                },
                LookupResult {
                    hit_length: 8,
                    tail_key_boundaries: [
                        TailKeyBoundary {
                            group_id: 0,
                            num_tokens: 8,
                        },
                    ],
                },
            )
        "#]]
        .assert_debug_eq(&(all, gap));
    }
}
