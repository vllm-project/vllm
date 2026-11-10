// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! `MambaManager` overrides of `single_type_kv_cache_manager.py`.

use super::{ManagerType, SingleTypeKvCacheManager, py_index};
use crate::kv_cache::block::{BlockHashWithGroupId, BlockId, NULL_BLOCK_ID};
use crate::kv_cache::block_pool::BlockPool;
use crate::kv_cache::cdiv;
use crate::kv_cache::request::KvRequest;
use crate::kv_cache::spec::{
    KvCacheSpecType, get_mamba_prefill_checkpoint_position, is_mamba_prefill_checkpoint_valid,
};

impl SingleTypeKvCacheManager {
    fn prefill_checkpoint_alignment(&self) -> Option<usize> {
        match self.kv_cache_spec.spec_type {
            KvCacheSpecType::Mamba {
                prefill_checkpoint_alignment,
                ..
            } => prefill_checkpoint_alignment,
            _ => unreachable!("not a mamba spec"),
        }
    }

    pub(super) fn mamba_remove_blocks_in_range(
        &mut self,
        pool: &mut BlockPool,
        request_id: &str,
        first_block: usize,
        last_block: usize,
    ) {
        let num_retired = self.mamba.num_retired_blocks.get(request_id).copied().unwrap_or(0);
        let Some(blocks) = self.req_to_blocks.get_mut(request_id) else {
            return;
        };
        let first_block = first_block.max(num_retired);
        let last_block = last_block.min(blocks.len());
        if first_block >= last_block {
            return;
        }
        let mut freed = Vec::new();
        // Mamba prefill leaves null gaps between states awaiting retirement.
        for i in (first_block..last_block).rev() {
            if blocks[i] == NULL_BLOCK_ID {
                continue;
            }
            freed.push(blocks[i]);
            blocks[i] = NULL_BLOCK_ID;
        }
        if !freed.is_empty() {
            pool.free_blocks(freed, false);
        }
        self.mamba.num_retired_blocks.insert(request_id.to_owned(), last_block);
    }

    pub(super) fn mamba_remove_skipped_blocks(
        &mut self,
        pool: &mut BlockPool,
        request_id: &str,
        processed_computed_tokens: usize,
    ) {
        self.base_remove_skipped_blocks(pool, request_id, processed_computed_tokens);
        if !self.mamba.is_align() {
            return;
        }
        // `last_state_block_idx` refers to the block index allocated two steps ago.
        // The block allocated in the previous step is used to copy Mamba states
        // into the block allocated in the current step; the earlier block is
        // no longer needed and should be freed here.
        let Some(&last_state_block_idx) = self.mamba.last_state_block_idx.get(request_id) else {
            return;
        };
        // Blocks allocated during prefill may be non-contiguous. Use
        // `last_state_block_idx` to free the appropriate block and replace it
        // with a null block.
        if last_state_block_idx < cdiv(processed_computed_tokens, self.block_size) as i64 - 1 {
            let blocks = self.req_blocks_entry(request_id);
            let idx = py_index(blocks.len(), last_state_block_idx);
            let block = blocks[idx];
            if block != NULL_BLOCK_ID {
                blocks[idx] = NULL_BLOCK_ID;
                pool.free_blocks([block], false);
            }
        }
    }

    fn needs_internal_checkpoint(
        &mut self,
        request_id: &str,
        query_start: usize,
        query_end: usize,
        checkpoint_position: usize,
        prefill_end: usize,
        hash_block_size: usize,
    ) -> bool {
        let checkpoint_idx = cdiv(query_end, self.block_size) as i64 - 2;
        let num_speculative_blocks = self.mamba.num_speculative_blocks as i64;
        let hash_alignment = self.prefill_checkpoint_alignment();
        let has_prefill_checkpoint_blocks = self.mamba.has_prefill_checkpoint_blocks;
        let allocated = self.mamba.allocated_block_reqs.contains(request_id);
        let block_size = self.block_size;
        let blocks = self.req_blocks_entry(request_id);
        has_prefill_checkpoint_blocks
            && query_end >= prefill_end
            && is_mamba_prefill_checkpoint_valid(
                query_start,
                query_end,
                checkpoint_position,
                hash_block_size,
                block_size,
                hash_alignment,
            )
            && checkpoint_idx >= 0
            && (checkpoint_idx >= blocks.len() as i64
                || blocks[checkpoint_idx as usize] == NULL_BLOCK_ID
                || (allocated && checkpoint_idx >= blocks.len() as i64 - num_speculative_blocks))
    }

    pub(super) fn mamba_get_num_blocks_to_allocate(
        &mut self,
        pool: &BlockPool,
        request_id: &str,
        num_tokens: usize,
        new_computed_blocks: &[BlockId],
        total_computed_tokens: usize,
        num_local_computed_tokens: usize,
        num_tokens_main_model: usize,
        apply_admission_cap: bool,
        prefill_end: usize,
    ) -> i64 {
        if let Some(&last) = new_computed_blocks.last()
            && let Some(hash) = &pool.block(last).block_hash
            && self.mamba.cached_blocks_this_step.contains(hash)
        {
            // Mamba can't rely on blocks generated by other requests in the current step
            // To put it in the next step, we return num_gpu_blocks + 1 so
            // that kv_cache_manager will think there is no enough blocks to allocate now
            // and don't schedule it in the current step.
            return pool.num_gpu_blocks as i64 + 1;
        }
        if !self.mamba.is_align() {
            // Allocate extra `num_speculative_blocks` blocks for
            // speculative decoding (MTP/EAGLE) with linear attention.
            let num_tokens =
                num_tokens + self.kv_cache_spec.block_size * self.mamba.num_speculative_blocks;
            return self.base_get_num_blocks_to_allocate(
                pool,
                request_id,
                num_tokens,
                new_computed_blocks,
                total_computed_tokens,
                num_local_computed_tokens,
                apply_admission_cap,
            );
        }
        // We don't allocate blocks for lookahead tokens in align mode, because if
        // x * block_size tokens are scheduled, num_tokens is
        // x * block_size + num_lookahead_tokens and breaks the alignment.
        // We can ignore lookahead tokens because current draft models don't have
        // mamba layers.
        let num_tokens = num_tokens_main_model;
        let num_speculative_blocks = self.mamba.num_speculative_blocks;

        // NOTE(tdouble): this is an over-estimate of how many blocks we need because
        // num_tokens can include draft tokens that will later be rejected.
        let num_required_blocks = cdiv(num_tokens, self.block_size) + num_speculative_blocks;
        let num_req_blocks = self.req_blocks_entry(request_id).len();
        let mut num_new_blocks =
            num_required_blocks as i64 - new_computed_blocks.len() as i64 - num_req_blocks as i64;
        let has_partial_hit = self
            .has_partial_local_hit(new_computed_blocks, num_local_computed_tokens)
            || self.partial_hit_reqs.contains_key(request_id);
        if has_partial_hit {
            num_new_blocks = num_new_blocks.max(0) + 1;
        }
        // Keyed on the chunk end like the worker's
        // `compute_mamba_prefill_checkpoints`, which writes the state there.
        // The helper returns a boundary strictly below its input, so every
        // chunk has one; under sparse retention only the prefill-end chunk's
        // is kept, so `prefill_end` skips reserving blocks for the others.
        let mut checkpoint_position = get_mamba_prefill_checkpoint_position(
            num_tokens,
            self.cache_hit_alignment_tokens,
            self.mamba.drop_eagle_checkpoint_block,
            pool.num_hash_lookahead_tokens,
        );
        if !self.needs_internal_checkpoint(
            request_id,
            total_computed_tokens,
            num_tokens,
            checkpoint_position,
            prefill_end,
            pool.hash_block_size,
        ) {
            checkpoint_position = 0;
        }
        let checkpoint_block = i64::from(checkpoint_position > 0);
        if !apply_admission_cap {
            if checkpoint_position > 0 {
                let checkpoint_idx = cdiv(num_tokens, self.block_size) - 2;
                self.mamba
                    .checkpoints
                    .insert(request_id.to_owned(), (checkpoint_position, checkpoint_idx));
            } else {
                self.mamba.checkpoints.remove(request_id);
            }
        }
        if num_new_blocks > 0 {
            let blocks_allocated = self.mamba.allocated_block_reqs.contains(request_id);
            let mut physical_block_cap = 1 + i64::from(has_partial_hit) + checkpoint_block;
            if !blocks_allocated {
                physical_block_cap += num_speculative_blocks as i64;
            }
            num_new_blocks = num_new_blocks.min(physical_block_cap);
        }

        let num_evictable_computed_blocks =
            Self::get_num_evictable_blocks(pool, new_computed_blocks);
        num_new_blocks + num_evictable_computed_blocks
    }

    pub(super) fn mamba_allocate_new_blocks(
        &mut self,
        pool: &mut BlockPool,
        request_id: &str,
        num_tokens: usize,
        num_tokens_main_model: usize,
    ) -> Vec<BlockId> {
        let num_speculative_blocks = self.mamba.num_speculative_blocks;
        if !self.mamba.is_align() {
            // Allocate extra `num_speculative_blocks` blocks for
            // speculative decoding (MTP/EAGLE) with linear attention.
            let num_tokens = num_tokens + self.block_size * num_speculative_blocks;
            return self.base_allocate_new_blocks(pool, request_id, num_tokens);
        }
        // We don't allocate blocks for lookahead tokens in align mode, because if
        // x * block_size tokens are scheduled, num_tokens is
        // x * block_size + num_lookahead_tokens and breaks the alignment.
        // We can ignore lookahead tokens because current draft models don't have
        // mamba layers.
        let num_tokens = num_tokens_main_model;
        let block_size = self.block_size;
        let group_id = self.kv_cache_group_id;
        // NOTE(tdouble): this is an over-estimate of how many blocks we need because
        // num_tokens can include draft tokens that will later be rejected.
        let num_required_blocks = cdiv(num_tokens, block_size) + num_speculative_blocks;
        let checkpoint_block = usize::from(self.mamba.checkpoints.contains_key(request_id));
        let partial_hit = self.partial_hit_reqs.get(request_id).copied();
        let has_partial_hit = partial_hit.is_some();
        let prev_block_len = self.req_blocks_entry(request_id).len();
        // `num_required_blocks` might be less than `len(req_blocks)` if blocks are
        // over-allocated at last round.
        if num_required_blocks <= prev_block_len && !has_partial_hit && checkpoint_block == 0 {
            self.mamba.allocated_block_reqs.insert(request_id.to_owned());
            return Vec::new();
        }

        let blocks_allocated = self.mamba.allocated_block_reqs.contains(request_id);
        // Record the last state block
        if blocks_allocated {
            // We always save the running state at the last
            // (1 + num_speculative_blocks) block
            self.mamba.last_state_block_idx.insert(
                request_id.to_owned(),
                prev_block_len as i64 - 1 - num_speculative_blocks as i64,
            );
        } else if prev_block_len > 0 {
            // When a new request hits the prefix cache, the last block
            // saves the hit state.
            self.mamba
                .last_state_block_idx
                .insert(request_id.to_owned(), prev_block_len as i64 - 1);
        }

        let num_skipped_blocks = num_required_blocks as i64 - num_speculative_blocks as i64 - 1;
        let req_blocks = self.req_to_blocks.get_mut(request_id).unwrap();
        // null blocks
        if (prev_block_len as i64) < num_skipped_blocks {
            // minus the internal checkpoint block
            // so we don't set null for that block
            let null_end = num_skipped_blocks - checkpoint_block as i64;
            if null_end > prev_block_len as i64 {
                req_blocks.resize(null_end as usize, NULL_BLOCK_ID);
            }
        }

        if blocks_allocated {
            // Relocate exclusively owned scratch blocks this step leaves
            // behind. In a checkpoint step, one at the checkpoint index
            // stays: the worker exports the checkpoint into that column.
            let start = prev_block_len as i64 - num_speculative_blocks as i64;
            for block_idx in start..prev_block_len as i64 {
                if block_idx < num_skipped_blocks - checkpoint_block as i64 {
                    // `_relocate_speculative_block`.
                    let idx = py_index(req_blocks.len(), block_idx);
                    let block = req_blocks[idx];
                    assert!(
                        pool.is_block_writable(block),
                        "Speculative Mamba blocks must be exclusively owned and unhashed before relocation"
                    );
                    req_blocks.push(block);
                    let idx = py_index(req_blocks.len(), block_idx);
                    req_blocks[idx] = NULL_BLOCK_ID;
                } else {
                    break;
                }
            }
        }
        let mut num_new_blocks = num_required_blocks.saturating_sub(req_blocks.len());
        if has_partial_hit {
            num_new_blocks += 1;
        }
        let mut max_new_blocks = 1 + usize::from(has_partial_hit) + checkpoint_block;
        if !blocks_allocated {
            max_new_blocks += num_speculative_blocks;
        }
        assert!(num_new_blocks <= max_new_blocks);
        let mut new_blocks = pool.get_new_blocks(num_new_blocks);
        let mut returned_blocks = req_blocks[prev_block_len.min(req_blocks.len())..].to_vec();
        if let Some((block_idx, source_block)) = partial_hit {
            let cow_block = new_blocks.remove(0);
            if blocks_allocated {
                // The worker block table of a running request is
                // append-only, so the request must stay on
                // source_block. Move the cache entry to cow_block
                // instead; the queued copy fills it before forward
                // overwrites source_block.
                assert!(self.req_to_blocks[request_id][block_idx] == source_block);
                pool.move_block_hashes(source_block, cow_block);
                self.pending_cow_copies.push((source_block, cow_block));
                pool.block_mut(source_block).ref_cnt += 1;
                if let Some((marker_block, boundary_tokens)) =
                    self.mamba.producer_partial_tail_reqs.remove(request_id)
                {
                    assert!(marker_block == source_block);
                    // This CoW preserved a producer's own boundary
                    // state in cow_block; hand it to the connector for
                    // partial-tail offload once the copy has run.
                    self.pending_boundary_state_offloads.push((
                        request_id.to_owned(),
                        group_id,
                        cow_block,
                        boundary_tokens,
                    ));
                }
                if let Some(hash) = pool.block(cow_block).block_hash.clone() {
                    // The moved entry is only filled by this step's
                    // copy, so defer same-step hits on it.
                    self.mamba.cached_blocks_this_step.insert(hash);
                }
            } else {
                self.apply_cow(pool, request_id, block_idx, source_block, cow_block);
                returned_blocks.insert(0, cow_block);
            }
        }
        self.req_to_blocks.get_mut(request_id).unwrap().extend_from_slice(&new_blocks);
        self.mamba.allocated_block_reqs.insert(request_id.to_owned());
        self.partial_hit_reqs.remove(request_id);
        returned_blocks.extend(new_blocks);
        returned_blocks
    }

    pub(super) fn mamba_cache_blocks(
        &mut self,
        pool: &mut BlockPool,
        request: &impl KvRequest,
        num_tokens: usize,
        retention_interval: Option<usize>,
        replay_boundaries: &[usize],
    ) {
        let request_id = request.request_id();
        let num_cached_blocks_before = self.num_cached_block.get(request_id).copied().unwrap_or(0);
        self.base_cache_blocks(
            pool,
            request,
            num_tokens,
            retention_interval,
            replay_boundaries,
        );
        let num_cached_blocks_after = self.num_cached_block.get(request_id).copied().unwrap_or(0);
        let is_align = self.mamba.is_align();
        if is_align
            && let Some(partial_hash) =
                self.mamba_cache_partial_tail_block(pool, request, num_tokens, retention_interval)
        {
            self.mamba.cached_blocks_this_step.insert(partial_hash);
        }
        if num_cached_blocks_after > num_cached_blocks_before {
            let group_id = self.kv_cache_group_id;
            let blocks = &self.req_to_blocks[request_id];
            for &block in &blocks[num_cached_blocks_before..num_cached_blocks_after] {
                let blk = pool.block(block);
                // Skip null blocks (align-mode skipped states) and blocks that
                // were not cached this step — with sparse retention
                // (reachable_block_mask) the intermediate state snapshots carry
                // no hash and must not be recorded as cached-this-step.
                let Some(hash) = &blk.block_hash else {
                    continue;
                };
                if blk.is_null {
                    continue;
                }
                self.mamba.cached_blocks_this_step.insert(hash.clone());
                if is_align {
                    let num_tokens = blk
                        .block_hash_num_tokens
                        .expect("cached mamba block must record its boundary");
                    // Offer every retained boundary with its exact block.
                    // The connector filters against its save window, which may
                    // extend past the original prompt during resumed prefill.
                    self.pending_boundary_state_offloads.push((
                        request_id.to_owned(),
                        group_id,
                        block,
                        num_tokens,
                    ));
                }
            }
        }
    }

    fn mamba_cache_partial_tail_block(
        &mut self,
        pool: &mut BlockPool,
        request: &impl KvRequest,
        num_tokens: usize,
        retention_interval: Option<usize>,
    ) -> Option<BlockHashWithGroupId> {
        let hash_block_size = pool.hash_block_size;
        let request_id = request.request_id();
        let block_size = self.block_size;
        let group_id = self.kv_cache_group_id;
        let drop_eagle = self.mamba.drop_eagle_checkpoint_block;
        // Re-key the reserved block at its exported checkpoint boundary.
        if let Some(&(checkpoint_position, checkpoint_idx)) = self.mamba.checkpoints.get(request_id)
        {
            let blocks = self.req_blocks_entry(request_id);
            assert!(checkpoint_idx < blocks.len());
            let checkpoint_block = blocks[checkpoint_idx];
            // The prompt's final chunk caches only up to its last full block,
            // so ``num_tokens < num_prompt_tokens`` holds there too; its
            // checkpoint is the prompt's reusable boundary, not a transient one.
            let is_prompt_checkpoint = checkpoint_position
                == get_mamba_prefill_checkpoint_position(
                    request.num_prompt_tokens(),
                    self.cache_hit_alignment_tokens,
                    drop_eagle,
                    pool.num_hash_lookahead_tokens,
                );
            let mut is_retained_checkpoint = false;
            if checkpoint_position.is_multiple_of(block_size) {
                let block_idx = checkpoint_position / block_size - 1;
                let retained = ManagerType::Mamba.reachable_block_mask(
                    block_idx,
                    block_idx + 1,
                    Some(self.cache_hit_alignment_tokens),
                    &self.kv_cache_spec,
                    drop_eagle,
                    retention_interval,
                    &[],
                    1,
                    None,
                );
                is_retained_checkpoint = retained.is_none_or(|mask| mask[0]);
            }
            if !is_prompt_checkpoint
                && !is_retained_checkpoint
                && checkpoint_position != request.shared_prefix_boundary()
            {
                // The transient state overwrites this slot, so neither it nor
                // the full-block hash registered earlier this step is reusable.
                pool.maybe_evict_cached_block(checkpoint_block);
                return None;
            }
            if pool.block(checkpoint_block).block_hash_num_tokens == Some(checkpoint_position) {
                return None;
            }
            return pool.cache_partial_block(
                request,
                checkpoint_block,
                checkpoint_position,
                group_id,
                block_size,
                true,
            );
        }
        if num_tokens == 0 || !num_tokens.is_multiple_of(hash_block_size) {
            return None;
        }
        let latest_prompt_hash_boundary = get_mamba_prefill_checkpoint_position(
            request.num_prompt_tokens(),
            hash_block_size,
            drop_eagle,
            pool.num_hash_lookahead_tokens,
        );
        // The junction is the other position a sibling resumes at: where one was
        // observed to stop, and where the scheduler already ends a chunk. Bounded
        // to the prompt chunk being computed -- during decode the target is the
        // running state block, mutated in place, which equals what its key
        // promises only after that step's forward.
        if num_tokens != latest_prompt_hash_boundary
            && !(self.shared_prefix_checkpoint
                && num_tokens == request.shared_prefix_boundary()
                && request.num_computed_tokens() < num_tokens
                && num_tokens <= request.num_prompt_tokens())
        {
            return None;
        }

        let block_idx = cdiv(num_tokens, block_size) - 1;
        let blocks = self.req_blocks_entry(request_id);
        if block_idx >= blocks.len() {
            return None;
        }
        let source_block = blocks[block_idx];
        if source_block == NULL_BLOCK_ID {
            return None;
        }
        // Already keyed here by a hit or an earlier step; re-caching would
        // mark it cached this step and defer every sibling that hits it.
        if pool.block(source_block).block_hash_num_tokens == Some(num_tokens) {
            return None;
        }

        let partial_hash = pool.cache_partial_block(
            request,
            source_block,
            num_tokens,
            group_id,
            block_size,
            false,
        );
        if partial_hash.is_some() && !num_tokens.is_multiple_of(block_size) {
            self.partial_hit_reqs.insert(request_id.to_owned(), (block_idx, source_block));
            self.num_cached_block.insert(request_id.to_owned(), block_idx);
            // Producer of this partial tail: the boundary state currently lives
            // in ``source_block`` but the next step's forward overwrites it. The
            // upcoming CoW copies it into a durable cow_block; record the req so
            // allocate_new_blocks hands that block to the connector for offload.
            self.mamba
                .producer_partial_tail_reqs
                .insert(request_id.to_owned(), (source_block, num_tokens));
        }
        partial_hash
    }
}
