// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Port of `vllm/v1/core/encoder_cache_manager.py`.

use std::collections::BTreeMap;

use rustc_hash::{FxHashMap, FxHashSet};

use super::py_int_set::PyIntSet;
use super::request::Request;

/// The parts of a request the encoder cache managers read.
pub trait EncoderInputs {
    fn request_id(&self) -> &str;
    /// `request.mm_features[input_id].identifier`.
    fn mm_identifier(&self, input_id: usize) -> &str;
    fn get_num_encoder_embeds(&self, input_id: usize) -> usize;
    /// `len(request.mm_features)`.
    fn num_encoder_inputs(&self) -> usize;
}

impl EncoderInputs for Request {
    fn request_id(&self) -> &str {
        &self.request_id
    }

    fn mm_identifier(&self, input_id: usize) -> &str {
        &self.mm_features[input_id].identifier
    }

    fn get_num_encoder_embeds(&self, input_id: usize) -> usize {
        Request::get_num_encoder_embeds(self, input_id)
    }

    fn num_encoder_inputs(&self) -> usize {
        Request::num_encoder_inputs(self)
    }
}

/// `OrderedDict[str, int]` restricted to the operations `freeable` uses.
#[derive(Clone, Debug, Default)]
struct OrderedCounts {
    entries: FxHashMap<String, (u64, usize)>,
    order: BTreeMap<u64, String>,
    next_seq: u64,
}

impl OrderedCounts {
    /// `d[key] = value`: an existing key keeps its position.
    fn insert(&mut self, key: &str, value: usize) {
        if let Some(entry) = self.entries.get_mut(key) {
            entry.1 = value;
            return;
        }
        let seq = self.next_seq;
        self.next_seq += 1;
        self.entries.insert(key.to_owned(), (seq, value));
        self.order.insert(seq, key.to_owned());
    }

    /// `d.pop(key)`.
    fn pop(&mut self, key: &str) -> Option<usize> {
        let (seq, value) = self.entries.remove(key)?;
        self.order.remove(&seq);
        Some(value)
    }

    /// `d.popitem(last=False)`.
    fn pop_first(&mut self) -> Option<(String, usize)> {
        let (_, key) = self.order.pop_first()?;
        let (_, value) = self.entries.remove(&key).unwrap();
        Some((key, value))
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
    }
}

#[cfg(test)]
impl OrderedCounts {
    fn contains(&self, key: &str) -> bool {
        self.entries.contains_key(key)
    }

    fn get(&self, key: &str) -> Option<usize> {
        self.entries.get(key).map(|&(_, value)| value)
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn iter(&self) -> impl Iterator<Item = (&str, usize)> {
        self.order.values().map(|key| (key.as_str(), self.entries[key].1))
    }
}

/// Manages caching of encoder outputs for multimodal models in vLLM V1.
///
/// The EncoderCacheManager handles the lifecycle of multimodal encoder outputs
/// (such as vision embeddings from images) during request processing. It
/// provides memory-aware caching to avoid recomputing encoder outputs when the
/// same multimodal inputs appear in different stages of request processing.
///
/// This manager is particularly important for:
/// - Vision-language models (e.g., LLaVA) where image encoder outputs are
///   cached
/// - Any multimodal model where encoder computation is expensive and
///   cacheable
///
/// The cache operates at the granularity of individual multimodal input items
/// within requests, allowing for fine-grained memory management and enabling
/// chunked processing of multimodal inputs.
///
/// Cache is enabled to share embeddings of same multimodal data
/// item (identified by their hash value) between different requests,
/// and eviction takes place at allocation time when there's no free
/// space for new embeddings.
/// Oldest cached embeddings with no request referenced will be first evicted.
///
/// NOTE: The EncoderCacheManager operates on the level of multimodal embeddings
/// instead of encoder tokens (i.e. all tokens that represent the multimodal data
/// in the input sequence). This means all break/text tokens in-between multimodal
/// embeddings are not considered with respect to the cache size and the number
/// of free slots.
#[derive(Clone, Debug)]
pub struct EncoderCacheManager {
    /// Total cache capacity in encoder embeddings.
    cache_size: usize,
    /// Current available cache capacity in encoder embeddings.
    num_free_slots: usize,
    /// Capacity that can be immediately reclaimed by evicting entries with
    /// zero references (in encoder embeddings).
    num_freeable_slots: usize,

    /// mm_hash of mm_data => ids of requests that reference the mm_data
    cached: FxHashMap<String, FxHashSet<String>>,
    /// mm_hash of mm_data => number of cached encoder embeddings
    num_embeds_by_hash: FxHashMap<String, usize>,
    /// request_id => set of input_ids cached for that request
    ///
    /// A [`PyIntSet`] so [`EncoderCacheManager::get_cached_input_ids`]
    /// yields Python's set iteration order.
    // TODO(rust-engine-core): a spec-defined order (see `py_int_set`).
    request_cached_ids: FxHashMap<String, PyIntSet>,
    /// request_id => mm_hash => number of that request's cached input_ids
    /// with the mm_hash
    request_hash_refs: FxHashMap<String, FxHashMap<String, usize>>,

    /// mm_hash of mm_data => num_encoder_embeds of the mm_data
    ///
    /// Entries whose no current running request is needed and that can be
    /// freed to make space when needed, oldest first.
    freeable: OrderedCounts,
    /// mm_hash strings that were actually evicted since the last call to
    /// `get_freed_mm_hashes()`.
    freed: Vec<String>,
}

impl EncoderCacheManager {
    /// Args:
    ///     cache_size: Limit the size of the cache, measured by the number of
    ///                 encoder embeddings from the input sequence.
    pub fn new(cache_size: usize) -> Self {
        Self {
            cache_size,
            num_free_slots: cache_size,
            num_freeable_slots: cache_size,
            cached: FxHashMap::default(),
            num_embeds_by_hash: FxHashMap::default(),
            request_cached_ids: FxHashMap::default(),
            request_hash_refs: FxHashMap::default(),
            freeable: OrderedCounts::default(),
            freed: Vec::new(),
        }
    }

    pub fn cache_size(&self) -> usize {
        self.cache_size
    }

    pub fn num_free_slots(&self) -> usize {
        self.num_free_slots
    }

    pub fn num_freeable_slots(&self) -> usize {
        self.num_freeable_slots
    }

    /// Reset the encoder cache to its initial state.
    ///
    /// This clears all cached encoder outputs and resets capacity tracking.
    /// Called when model weights are updated to invalidate stale embeddings.
    pub fn reset(&mut self) {
        self.cached.clear();
        self.num_embeds_by_hash.clear();
        self.request_cached_ids.clear();
        self.request_hash_refs.clear();
        self.freeable.clear();
        self.freed.clear();
        self.num_free_slots = self.cache_size;
        self.num_freeable_slots = self.cache_size;
    }

    /// Check if encoder output for a specific multimodal input is cached.
    ///
    /// If the encoder output is cached, update `cached` to add the request id
    /// to the set of request ids that reference the cached encoder output.
    /// If the encoder output was previously not referenced by any request,
    /// update `freeable` and `num_freeable_slots` accordingly.
    ///
    /// Args:
    ///     request: The request containing the multimodal input
    ///     input_id: Index of the multimodal input within the request
    ///
    /// Returns:
    ///     True if the encoder output for this input is already cached
    pub fn check_and_update_cache(
        &mut self,
        request: &impl EncoderInputs,
        input_id: usize,
    ) -> bool {
        let mm_hash = request.mm_identifier(input_id);
        // Not cached at all
        let Some(referrers) = self.cached.get_mut(mm_hash) else {
            return false;
        };

        // Cached but currently not referenced by any request
        if referrers.is_empty() {
            let num_encoder_embeds =
                self.freeable.pop(mm_hash).expect("unreferenced entry is not freeable");
            self.num_freeable_slots -= num_encoder_embeds;
        }

        referrers.insert(request.request_id().to_owned());
        self.add_cached_input_id(request.request_id(), input_id, mm_hash);
        true
    }

    /// Return the encoder-embedding count stored for a cached input.
    pub fn get_cached_num_encoder_embeds(
        &self,
        request: &impl EncoderInputs,
        input_id: usize,
    ) -> Option<usize> {
        let mm_hash = request.mm_identifier(input_id);
        self.num_embeds_by_hash.get(mm_hash).copied()
    }

    /// Check if there's sufficient cache space for a multimodal input.
    /// If there is, return True and update EncoderCacheManager state.
    ///
    /// If there is not enough free space in `num_free_slots` but there is
    /// enough reclaimable space in `num_freeable_slots`, entries will be
    /// evicted from `freeable` (their mm_hash appended to `freed`) until
    /// enough space is available, and then this method returns True.
    /// Older entries are evicted first.
    ///
    /// Returns False only if the requested number of tokens exceeds both
    /// the free and reclaimable capacities combined.
    ///
    /// Args:
    ///     request: The request containing the multimodal input.
    ///     input_id: Index of the multimodal input within the request.
    ///     encoder_compute_budget: Number of encoder embeddings allowed to be
    ///         computed when this method is invoked.
    ///     num_embeds_to_schedule: Number of encoder embeddings already scheduled to be
    ///         allocated with cache space when this method is invoked.
    ///
    /// Returns:
    ///     True if there's enough capacity to hold the encoder output for this
    ///     input (possibly after reclaiming `freeable` entries); otherwise
    ///     False.
    ///
    /// Note: This method does not allocate physical memory for the encoder
    /// output but only the state of EncoderCacheManager.
    pub fn can_allocate(
        &mut self,
        request: &impl EncoderInputs,
        input_id: usize,
        encoder_compute_budget: usize,
        num_embeds_to_schedule: usize,
    ) -> bool {
        let mut num_embeds = request.get_num_encoder_embeds(input_id);

        // Not enough compute budget
        if num_embeds > encoder_compute_budget {
            return false;
        }

        num_embeds += num_embeds_to_schedule;

        // Enough free slots
        if num_embeds <= self.num_free_slots {
            return true;
        }

        // Not enough reclaimable slots
        if num_embeds > self.num_freeable_slots {
            return false;
        }

        // Not enough free slots but enough reclaimable slots
        // NOTE: Eviction takes place here, but physical memory is not freed
        // until model runner is notified by the scheduler output.
        while num_embeds > self.num_free_slots {
            let (mm_hash, num_free_embeds) =
                self.freeable.pop_first().expect("popitem(): dictionary is empty");
            self.cached.remove(&mm_hash).expect("evicted entry not cached");
            self.num_embeds_by_hash
                .remove(&mm_hash)
                .expect("evicted entry has no embedding count");
            self.freed.push(mm_hash);
            self.num_free_slots += num_free_embeds;
        }
        true
    }

    /// Allocate cache space for a multimodal input's encoder output.
    ///
    /// This reserves cache space for storing the encoder output of the
    /// specified multimodal input. The actual encoder output storage happens in
    /// the model runner; this method updates the manager's bookkeeping.
    ///
    /// Note:
    ///     This method assumes can_allocate() returned True for the same input.
    pub fn allocate(&mut self, request: &impl EncoderInputs, input_id: usize) {
        let mm_hash = request.mm_identifier(input_id);
        let request_id = request.request_id();

        let num_encoder_embeds = request.get_num_encoder_embeds(input_id);
        let cached_num_encoder_embeds =
            *self.num_embeds_by_hash.entry(mm_hash.to_owned()).or_insert(num_encoder_embeds);
        // _try_schedule_encoder_inputs rejects local-hit and same-step
        // identifier/count mismatches before allocation.
        assert_eq!(cached_num_encoder_embeds, num_encoder_embeds);

        // NOTE: Encoder cache should always have enough space for encoder inputs
        // that are scheduled since eviction takes place at can_allocate().
        assert!(self.num_free_slots >= num_encoder_embeds);
        assert!(self.num_freeable_slots >= num_encoder_embeds);

        self.cached.entry(mm_hash.to_owned()).or_default().insert(request_id.to_owned());
        self.add_cached_input_id(request_id, input_id, mm_hash);
        self.num_free_slots -= num_encoder_embeds;
        self.num_freeable_slots -= num_encoder_embeds;
    }

    fn add_cached_input_id(&mut self, request_id: &str, input_id: usize, mm_hash: &str) {
        let cached_ids = self.request_cached_ids.entry(request_id.to_owned()).or_default();
        if cached_ids.contains(input_id) {
            return;
        }
        cached_ids.add(input_id);
        let hash_refs = self.request_hash_refs.entry(request_id.to_owned()).or_default();
        *hash_refs.entry(mm_hash.to_owned()).or_insert(0) += 1;
    }

    /// Get all cached multimodal input IDs for a request.
    ///
    /// A snapshot in Python's set iteration order; Python returns the live
    /// set, which every caller copies before freeing inputs.
    pub fn get_cached_input_ids(&self, request: &impl EncoderInputs) -> Vec<usize> {
        self.request_cached_ids
            .get(request.request_id())
            .map(|ids| ids.iter().collect())
            .unwrap_or_default()
    }

    /// Free the request's reference to the encoder input (`mm_data`).
    ///
    /// When the reference set for the corresponding `mm_hash` becomes empty,
    /// the entry is appended to `freeable` and `num_freeable_slots` is
    /// increased by the number of encoder embeddings for that input.
    ///
    /// The entry is NOT physically freed until capacity is needed (e.g., by
    /// `can_allocate`).
    pub fn free_encoder_input(&mut self, request: &impl EncoderInputs, input_id: usize) {
        let req_id = request.request_id();
        let mm_hash = request.mm_identifier(input_id);
        // Always clean up request_cached_ids, even if the mm_hash was
        // already evicted from cache (e.g. by can_allocate).
        if let Some(cached_ids) = self.request_cached_ids.get_mut(req_id)
            && cached_ids.discard(input_id)
        {
            let hash_refs =
                self.request_hash_refs.get_mut(req_id).expect("request_hash_refs out of sync");
            let refs = hash_refs.get_mut(mm_hash).expect("hash ref out of sync");
            *refs -= 1;
            if *refs == 0 {
                hash_refs.remove(mm_hash);
            }
            if cached_ids.is_empty() {
                self.request_cached_ids.remove(req_id);
                self.request_hash_refs.remove(req_id);
            }
        }
        // The mm_hash not in cache or the req_id set is empty
        let Some(referrers) = self.cached.get_mut(mm_hash) else {
            return;
        };
        if referrers.is_empty() {
            return;
        }
        // `cached` counts referencing requests, not positions, so one request
        // that repeats an item (an image carried across conversation turns) has
        // a single reference covering every occurrence. Hold it until the last
        // occurrence is freed: dropping it at the first makes the entry
        // evictable while the request still needs it, and the encoder then
        // recomputes an item it already has.
        if self
            .request_hash_refs
            .get(req_id)
            .is_some_and(|refs| refs.contains_key(mm_hash))
        {
            return;
        }
        referrers.remove(req_id);
        if referrers.is_empty() {
            let num_encoder_embeds = self.num_embeds_by_hash[mm_hash];
            self.freeable.insert(mm_hash, num_encoder_embeds);
            self.num_freeable_slots += num_encoder_embeds;
        }
    }

    /// Free all encoder input cache reference held by *request*.
    ///
    /// For each cached input ID, `free_encoder_input` is invoked.
    /// The data stays in memory until eviction is triggered by a future
    /// attempt allocation called by 'can_allocate'.
    ///
    /// Typically called when a request is finished, cancelled, or aborted.
    pub fn free(&mut self, request: &impl EncoderInputs) {
        for input_id in self.get_cached_input_ids(request) {
            self.free_encoder_input(request, input_id);
        }
    }

    /// Get and clear the list of recently freed encoder cache entries.
    ///
    /// Returns:
    ///     List of mm_hash strings that were actually evicted since the last
    ///     call to be used by the scheduler to notify workers about which
    ///     encoder outputs can be removed from their caches. The internal
    ///     list is cleared after this call.
    pub fn get_freed_mm_hashes(&mut self) -> Vec<String> {
        // An entry evicted early in the scheduling pass can be allocated again
        // later in the same pass. Keep its worker-side tensor in that case.
        let mut freed = std::mem::take(&mut self.freed);
        freed.retain(|mm_hash| !self.cached.contains_key(mm_hash));
        freed
    }
}

// NOTE (NickLucche): Temporary implementation for encoder-decoder models that only
// use the manager for scheduling purposes. Encoder-decoder models will eventually
// utilize the cache and this class will fold into EncoderCacheManager, as
// differences with MM models shrink.
#[derive(Clone, Debug)]
pub struct EncoderDecoderCacheManager {
    cache_size: usize,
    num_free_slots: usize,
    allocated: Vec<String>,
    to_free: Vec<String>,
}

impl EncoderDecoderCacheManager {
    pub fn new(cache_size: usize) -> Self {
        Self {
            cache_size,
            num_free_slots: cache_size,
            allocated: Vec::new(),
            to_free: Vec::new(),
        }
    }

    pub fn num_free_slots(&self) -> usize {
        self.num_free_slots
    }

    /// Reset the encoder cache to its initial state.
    pub fn reset(&mut self) {
        self.num_free_slots = self.cache_size;
        self.allocated.clear();
        self.to_free.clear();
    }

    pub fn check_and_update_cache(
        &mut self,
        _request: &impl EncoderInputs,
        _input_id: usize,
    ) -> bool {
        false
    }

    pub fn can_allocate(
        &mut self,
        request: &impl EncoderInputs,
        input_id: usize,
        encoder_compute_budget: usize,
        num_embeds_to_schedule: usize,
    ) -> bool {
        let mut num_encoder_embeds = request.get_num_encoder_embeds(input_id);
        // Not enough compute budget
        if num_encoder_embeds > encoder_compute_budget {
            return false;
        }

        num_encoder_embeds += num_embeds_to_schedule;
        // Enough free slots
        num_encoder_embeds <= self.num_free_slots
    }

    pub fn allocate(&mut self, request: &impl EncoderInputs, input_id: usize) {
        let num_encoder_embeds = request.get_num_encoder_embeds(input_id);
        self.num_free_slots -= num_encoder_embeds;

        let mm_hash = request.mm_identifier(input_id);
        self.allocated.push(mm_hash.to_owned());
    }

    pub fn free(&mut self, request: &impl EncoderInputs) {
        for input_id in 0..request.num_encoder_inputs() {
            self.free_encoder_input(request, input_id);
        }
    }

    pub fn get_cached_input_ids(&self, request: &impl EncoderInputs) -> Vec<usize> {
        (0..request.num_encoder_inputs()).collect()
    }

    pub fn get_freed_mm_hashes(&mut self) -> Vec<String> {
        // As encoder cache is not used for enc-dec models, we can free the entries here
        // The actual free happens in the runner, *before* the model is executed.
        // Therefore, `freeable` acts as a buffer to free the entries only after the
        // model is executed, mimicking the state transition of `EncoderCacheManager`.
        let to_free = std::mem::take(&mut self.to_free);
        self.to_free = std::mem::take(&mut self.allocated);
        to_free
    }

    pub fn free_encoder_input(&mut self, request: &impl EncoderInputs, input_id: usize) {
        let num_encoder_embeds = request.get_num_encoder_embeds(input_id);
        self.num_free_slots += num_encoder_embeds;
    }
}

/// The scheduler's encoder cache manager: Python picks the class at
/// construction and calls it polymorphically.
#[derive(Clone, Debug)]
pub enum AnyEncoderCacheManager {
    Mm(EncoderCacheManager),
    EncoderDecoder(EncoderDecoderCacheManager),
}

macro_rules! dispatch {
    ($self:ident, $m:ident => $e:expr) => {
        match $self {
            Self::Mm($m) => $e,
            Self::EncoderDecoder($m) => $e,
        }
    };
}

impl AnyEncoderCacheManager {
    pub fn reset(&mut self) {
        dispatch!(self, m => m.reset())
    }

    pub fn check_and_update_cache(
        &mut self,
        request: &impl EncoderInputs,
        input_id: usize,
    ) -> bool {
        dispatch!(self, m => m.check_and_update_cache(request, input_id))
    }

    /// Only meaningful for multimodal models: Python's encoder-decoder
    /// manager inherits a method that reads state it never initializes, and
    /// the scheduler does not call it for encoder-decoder models.
    pub fn get_cached_num_encoder_embeds(
        &self,
        request: &impl EncoderInputs,
        input_id: usize,
    ) -> Option<usize> {
        match self {
            Self::Mm(m) => m.get_cached_num_encoder_embeds(request, input_id),
            Self::EncoderDecoder(_) => {
                panic!("get_cached_num_encoder_embeds on an encoder-decoder cache manager")
            }
        }
    }

    pub fn can_allocate(
        &mut self,
        request: &impl EncoderInputs,
        input_id: usize,
        encoder_compute_budget: usize,
        num_embeds_to_schedule: usize,
    ) -> bool {
        dispatch!(self, m => m.can_allocate(
            request,
            input_id,
            encoder_compute_budget,
            num_embeds_to_schedule,
        ))
    }

    pub fn allocate(&mut self, request: &impl EncoderInputs, input_id: usize) {
        dispatch!(self, m => m.allocate(request, input_id))
    }

    pub fn get_cached_input_ids(&self, request: &impl EncoderInputs) -> Vec<usize> {
        dispatch!(self, m => m.get_cached_input_ids(request))
    }

    pub fn free_encoder_input(&mut self, request: &impl EncoderInputs, input_id: usize) {
        dispatch!(self, m => m.free_encoder_input(request, input_id))
    }

    pub fn free(&mut self, request: &impl EncoderInputs) {
        dispatch!(self, m => m.free(request))
    }

    pub fn get_freed_mm_hashes(&mut self) -> Vec<String> {
        dispatch!(self, m => m.get_freed_mm_hashes())
    }
}

/// Error from [`compute_mm_encoder_budget`].
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[error(
    "Chunked MM input disabled but max_tokens_per_mm_item \
     ({max_tokens_per_mm_item}) is larger than max_num_batched_tokens \
     ({max_num_batched_tokens}). Please increase max_num_batched_tokens."
)]
pub struct MmItemTooLarge {
    pub max_tokens_per_mm_item: usize,
    pub max_num_batched_tokens: usize,
}

/// The `SchedulerConfig` fields [`compute_mm_encoder_budget`] reads.
#[derive(Clone, Copy, Debug)]
pub struct EncoderBudgetConfig {
    pub disable_chunked_mm_input: bool,
    pub max_num_batched_tokens: usize,
    pub max_num_encoder_input_tokens: usize,
    pub encoder_cache_size: usize,
}

/// Compute the encoder cache budget based on the model and scheduler
/// configurations for a multimodal model.
///
/// Args:
///     scheduler_config: Scheduler configuration.
///     mm_max_toks_per_item: The maximum number of tokens per item for each
///         non-text modality.
///
/// Returns:
///     - Compute budget for encoder execution, measured in number of tokens
///         from the input sequence.
///     - Space budget for encoder cache size, measured in number of tokens
///         from the input sequence.
pub fn compute_mm_encoder_budget(
    scheduler_config: &EncoderBudgetConfig,
    mm_max_toks_per_item: impl IntoIterator<Item = usize>,
) -> Result<(usize, usize), MmItemTooLarge> {
    // TODO: log Python's warning when every modality is disabled.
    let Some(max_tokens_per_mm_item) = mm_max_toks_per_item.into_iter().max() else {
        return Ok((0, 0));
    };

    if scheduler_config.disable_chunked_mm_input
        && max_tokens_per_mm_item > scheduler_config.max_num_batched_tokens
    {
        return Err(MmItemTooLarge {
            max_tokens_per_mm_item,
            max_num_batched_tokens: scheduler_config.max_num_batched_tokens,
        });
    }

    let encoder_compute_budget =
        scheduler_config.max_num_encoder_input_tokens.max(max_tokens_per_mm_item);
    let encoder_cache_size = scheduler_config.encoder_cache_size.max(max_tokens_per_mm_item);

    Ok((encoder_compute_budget, encoder_cache_size))
}

#[cfg(test)]
mod tests;
