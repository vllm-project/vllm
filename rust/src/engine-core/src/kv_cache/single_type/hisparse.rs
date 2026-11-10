// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The per-manager state and the coordinator-free methods of
//! `HiSparseHotManager` and `HiSparseResidentManager`.

use rustc_hash::{FxHashMap, FxHashSet};

use super::super::block::{BlockId, NULL_BLOCK_ID, trim_null_head};
use super::super::block_pool::BlockPool;
use super::super::cdiv;
use super::SingleTypeKvCacheManager;

/// State of the HiSparse hot and resident managers.
#[derive(Debug, Default)]
pub(crate) struct HiSparseManagerState {
    /// `HiSparseHotSpec.blocks_per_request` (hot manager).
    pub(crate) blocks_per_request: usize,
    /// Requests that need a hot region (hot manager).
    pub(crate) hot_required: FxHashSet<String>,
    /// Per request, the pages whose GPU residency changed since the block
    /// table was built: a null block for a page now read from the host, or
    /// the pinned GPU copy a host-only page is now read from (resident
    /// manager).
    pub(crate) residency_changes: FxHashMap<String, FxHashMap<usize, BlockId>>,
}

impl SingleTypeKvCacheManager {
    /// `HiSparseHotManager.require_hot`.
    pub fn require_hot(&mut self, request_id: &str) {
        if !self.hisparse.hot_required.contains(request_id) {
            self.hisparse.hot_required.insert(request_id.to_owned());
        }
    }

    /// `HiSparseHotManager.has_hot`.
    pub fn has_hot(&self, request_id: &str) -> bool {
        self.num_req_blocks(request_id) == self.hisparse.blocks_per_request
    }

    /// `HiSparseHotManager.hot_required`: whether the request needs a hot
    /// region.
    pub fn hot_required(&self, request_id: &str) -> bool {
        self.hisparse.hot_required.contains(request_id)
    }

    /// The requests that need a hot region (`HiSparseHotManager.hot_required`).
    pub fn hot_required_ids(&self) -> impl Iterator<Item = &str> {
        self.hisparse.hot_required.iter().map(String::as_str)
    }

    /// `HiSparseHotSpec.blocks_per_request` (hot manager).
    pub fn blocks_per_request(&self) -> usize {
        self.hisparse.blocks_per_request
    }

    /// `HiSparseHotManager.get_num_required_blocks`.
    pub(crate) fn hot_num_required_blocks(&self, request_id: &str) -> usize {
        self.hisparse.blocks_per_request.saturating_sub(self.num_req_blocks(request_id))
    }

    /// `HiSparseResidentManager.adopt_resident_page`: point a null prefix page
    /// at a pinned GPU copy of its contents.
    pub(crate) fn adopt_resident_page(
        &mut self,
        request_id: &str,
        block_idx: usize,
        block: BlockId,
    ) -> bool {
        if block_idx >= self.num_req_blocks(request_id) {
            return false;
        }
        if self.get_resident_page(request_id, block_idx).is_some() {
            return false;
        }
        self.residency_changes_entry(request_id).insert(block_idx, block);
        true
    }

    /// `HiSparseResidentManager.drop_resident_page`: stop reading a page from
    /// the GPU; return the block it was read from.
    pub(crate) fn drop_resident_page(
        &mut self,
        request_id: &str,
        block_idx: usize,
    ) -> Option<BlockId> {
        let block = self.get_resident_page(request_id, block_idx);
        if block.is_some() {
            self.residency_changes_entry(request_id).insert(block_idx, NULL_BLOCK_ID);
        }
        block
    }

    fn residency_changes_entry(&mut self, request_id: &str) -> &mut FxHashMap<usize, BlockId> {
        let changes = &mut self.hisparse.residency_changes;
        if !changes.contains_key(request_id) {
            changes.insert(request_id.to_owned(), FxHashMap::default());
        }
        changes.get_mut(request_id).expect("just inserted")
    }

    /// `HiSparseResidentManager.get_resident_page`: the GPU block a page is
    /// read from, or None if it is read from the host.
    pub fn get_resident_page(&self, request_id: &str, block_idx: usize) -> Option<BlockId> {
        let blocks = self.req_to_blocks.get(request_id)?;
        let mut block = *blocks.get(block_idx)?;
        if let Some(changes) = self.hisparse.residency_changes.get(request_id) {
            block = changes.get(&block_idx).copied().unwrap_or(block);
        }
        (block != NULL_BLOCK_ID).then_some(block)
    }

    /// `HiSparseResidentManager.get_residency_row`: the request's row from
    /// `start` with residency changes applied. Unlike `get_resident_page`,
    /// host-only pages stay as the null block.
    pub fn get_residency_row(&self, request_id: &str, start: usize) -> Vec<BlockId> {
        let blocks = self.req_to_blocks.get(request_id).map_or(&[][..], Vec::as_slice);
        let blocks = &blocks[start.min(blocks.len())..];
        let Some(changes) = self.hisparse.residency_changes.get(request_id) else {
            return blocks.to_vec();
        };
        if changes.is_empty() {
            return blocks.to_vec();
        }
        blocks
            .iter()
            .enumerate()
            .map(|(i, &block)| changes.get(&(start + i)).copied().unwrap_or(block))
            .collect()
    }

    /// `HiSparseHotManager.get_num_blocks_to_allocate`.
    pub(crate) fn hot_get_num_blocks_to_allocate(
        &self,
        request_id: &str,
        new_computed_blocks: &[BlockId],
        total_computed_tokens: usize,
        num_local_computed_tokens: usize,
    ) -> i64 {
        assert!(new_computed_blocks.is_empty());
        // A hot region is needed to read host-backed history: an external
        // import, a new request resuming a host prefix, or one already asked
        // to transition. Running requests keep their earlier answer.
        let host_import = total_computed_tokens > num_local_computed_tokens;
        let resumes_host_prefix =
            num_local_computed_tokens > 0 && !self.num_cached_block.contains_key(request_id);
        if host_import || resumes_host_prefix || self.hot_required(request_id) {
            return self.hot_num_required_blocks(request_id) as i64;
        }
        0
    }

    /// `HiSparseHotManager.add_local_computed_blocks`.
    pub(crate) fn hot_add_local_computed_blocks(
        &mut self,
        request_id: &str,
        new_computed_blocks: &[BlockId],
        num_local_computed_tokens: usize,
        has_resident_managers: bool,
    ) {
        assert!(new_computed_blocks.is_empty());
        self.num_cached_block.insert(request_id.to_owned(), 0);
        if num_local_computed_tokens > 0 || !has_resident_managers {
            self.require_hot(request_id);
        }
    }

    /// `HiSparseHotManager.allocate_new_blocks`.
    pub(crate) fn hot_allocate_new_blocks(
        &mut self,
        pool: &mut BlockPool,
        request_id: &str,
    ) -> Vec<BlockId> {
        // Cold admissions can bypass add_local_computed_blocks.
        self.num_cached_block.insert(request_id.to_owned(), 0);
        if !self.hot_required(request_id) {
            return Vec::new();
        }
        let blocks_per_request = self.hisparse.blocks_per_request;
        let num_req_blocks = self.req_blocks_entry(request_id).len();
        if blocks_per_request <= num_req_blocks {
            return Vec::new();
        }
        let new_blocks = pool.get_new_blocks(blocks_per_request - num_req_blocks);
        self.req_blocks_entry(request_id).extend_from_slice(&new_blocks);
        new_blocks
    }

    /// `HiSparseHotManager.pop_blocks_for_free`.
    pub(crate) fn hot_pop_blocks_for_free(&mut self, request_id: &str) -> Vec<BlockId> {
        self.hisparse.hot_required.remove(request_id);
        self.pop_blocks_for_free(request_id)
    }

    /// `HiSparseResidentManager.get_num_blocks_to_allocate`.
    pub(crate) fn resident_get_num_blocks_to_allocate(
        &self,
        request_id: &str,
        num_tokens: usize,
        new_computed_blocks: &[BlockId],
        total_computed_tokens: usize,
        num_local_computed_tokens: usize,
        apply_admission_cap: bool,
    ) -> i64 {
        assert!(new_computed_blocks.is_empty());
        let block_size = self.block_size;
        let existing = self.num_req_blocks(request_id);
        if total_computed_tokens > num_local_computed_tokens {
            // Keep the last imported page writable, including the last-token
            // replay on a full prompt hit. Only earlier pages stay host-only.
            let imported_pages = cdiv(total_computed_tokens, block_size);
            let needs_tail = self.get_resident_page(request_id, imported_pages - 1).is_none();
            return i64::from(needs_tail)
                + cdiv(num_tokens, block_size).saturating_sub(imported_pages.max(existing)) as i64;
        }
        let host_pages = cdiv(num_local_computed_tokens, block_size);
        let mut required = cdiv(num_tokens, block_size);
        if apply_admission_cap {
            let cap = self
                .max_admission_blocks_per_request
                .expect("HiSparse resident managers cap admission");
            required = required.min(cap);
        }
        required.saturating_sub(existing.max(host_pages)) as i64
    }

    /// `HiSparseResidentManager.allocate_external_computed_blocks`: reserve a
    /// writable final page; earlier imported pages stay on host.
    pub(crate) fn resident_allocate_external_computed_blocks(
        &mut self,
        pool: &mut BlockPool,
        request_id: &str,
        num_local_computed_tokens: usize,
        num_external_computed_tokens: usize,
    ) {
        assert!(num_external_computed_tokens > 0);
        let num_tokens = num_local_computed_tokens + num_external_computed_tokens;
        let tail_page = (num_tokens - 1) / self.block_size;
        let blocks = self.req_blocks_entry(request_id);
        if blocks.len() <= tail_page {
            blocks.resize(tail_page + 1, NULL_BLOCK_ID);
        }
        if self.get_resident_page(request_id, tail_page).is_none() {
            let block = pool.get_new_blocks(1)[0];
            self.req_blocks_entry(request_id)[tail_page] = block;
        }
    }

    /// `HiSparseResidentManager.add_local_computed_blocks`, up to the
    /// coordinator's `commit_computed_blocks`. Returns the number of host
    /// pages the request now references.
    pub(crate) fn resident_add_local_computed_blocks(
        &mut self,
        request_id: &str,
        new_computed_blocks: &[BlockId],
        num_local_computed_tokens: usize,
    ) -> usize {
        assert!(new_computed_blocks.is_empty());
        let num_host_pages = cdiv(num_local_computed_tokens, self.block_size);
        let req_blocks = self.req_blocks_entry(request_id);
        assert!(req_blocks.is_empty());
        req_blocks.resize(num_host_pages, NULL_BLOCK_ID);
        self.num_cached_block.insert(request_id.to_owned(), 0);
        num_host_pages
    }

    /// `HiSparseResidentManager.allocate_new_blocks`.
    pub(crate) fn resident_allocate_new_blocks(
        &mut self,
        pool: &mut BlockPool,
        request_id: &str,
        num_tokens: usize,
    ) -> Vec<BlockId> {
        let num_required_blocks = cdiv(num_tokens, self.block_size);
        let num_req_blocks = self.req_blocks_entry(request_id).len();
        if num_required_blocks <= num_req_blocks {
            return Vec::new();
        }
        let new_blocks = pool.get_new_blocks(num_required_blocks - num_req_blocks);
        self.req_blocks_entry(request_id).extend_from_slice(&new_blocks);
        new_blocks
    }

    /// `HiSparseResidentManager.pop_blocks_for_free`, after the coordinator's
    /// `free`.
    pub(crate) fn resident_pop_blocks_for_free(&mut self, request_id: &str) -> Vec<BlockId> {
        let blocks = self.get_residency_row(request_id, 0);
        self.hisparse.residency_changes.remove(request_id);
        self.pop_blocks_for_free(request_id);
        trim_null_head(&blocks).to_vec()
    }
}
