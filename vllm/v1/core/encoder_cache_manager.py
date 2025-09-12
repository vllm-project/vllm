# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from sortedcontainers import SortedList

from vllm.logger import init_logger
from vllm.multimodal import MultiModalRegistry
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.config import ModelConfig, SchedulerConfig

logger = init_logger(__name__)


class EncoderCacheManager:
    """Manages caching of encoder outputs for multimodal models in vLLM V1.

    The EncoderCacheManager handles the lifecycle of multimodal encoder outputs
    (such as vision embeddings from images) during request processing. It
    provides memory-aware caching to avoid recomputing encoder outputs when the
    same multimodal inputs appear in different stages of request processing.

    This manager is particularly important for:
    - Vision-language models (e.g., LLaVA) where image encoder outputs are
      cached
    - Any multimodal model where encoder computation is expensive and
      cacheable

    The cache operates at the granularity of individual multimodal input items
    within requests, allowing for fine-grained memory management and enabling
    chunked processing of multimodal inputs.

    Cache is enabled to share embeddings of same multimodal data 
    item (identified by their hash value) between different requests, 
    and eviction takes place at allocation time when there's no free 
    space for new embeddings.
    Oldest cached embeddings with no request referenced will be first evicted.
    
    Args:
        cache_size: Limit the size of the cache, measured by the number of
                    tokens from the input sequence.

    Attributes:
        cache_size: Total cache capacity in encoder tokens.
        num_free_slots: Current available cache capacity in encoder tokens.
        num_freeable_slots: Capacity that can be immediately reclaimed by
            evicting entries with zero references (in encoder tokens).
        cached: Mapping from mm_hash to a set of request IDs that currently
            reference the cached entry. If the set is empty, the entry exists
            but is not referenced by any request and is eligible for
            reclamation.
        freeable: List of tuples (mm_hash, num_tokens) representing entries
            whose no current running request is needed and that can be freed to
            make space when needed.
        freed: List of mm_hash strings that were actually evicted since the
            last call to get_freed_mm_hashes(). This list is cleared on return.
    """

    def __init__(self, cache_size: int):
        self.cache_size = cache_size
        self.num_free_slots = cache_size
        self.num_freeable_slots = cache_size

        # mm_hash of mm_data => ids of requests that reference the mm_data
        self.cached: dict[str, set[str]] = {}

        # mm_hash of mm_data => num_encoder_tokens of the mm_data
        self.freeable: OrderedDict[str, int] = OrderedDict()
        self.freed: list[str] = []

        # mm_hash of receiving mm_data by encoder disaggregation
        self.receiving: set[str] = set()

    def check_and_update_cache(self, request: Request, input_id: int) -> bool:
        """Check if encoder output for a specific multimodal input is cached.

        If the encoder output is cached, update `cached` to add the request id
        to the set of request ids that reference the cached encoder output.
        If the encoder output was previously not referenced by any request,
        update `freeable` and `num_freeable_slots` accordingly.

        Args:
            request: The request containing the multimodal input
            input_id: Index of the multimodal input within the request

        Returns:
            True if the encoder output for this input is already cached
        """
        mm_hash = request.mm_hashes[input_id]
        # Not cached at all
        if mm_hash not in self.cached:
            return False

        # Cached but currently not referenced by any request
        if not self.cached[mm_hash]:
            num_tokens = self.freeable.pop(mm_hash)
            self.num_freeable_slots -= num_tokens

        self.cached[mm_hash].add(request.request_id)
        return True

    def can_allocate(self, request: Request, input_id: int,
                     encoder_compute_budget: int,
                     num_tokens_to_schedule: int) -> bool:
        """Check if there's sufficient cache space for a multimodal input. 
        If there is, return True and update EncoderCacheManager state.

        If there is not enough free space in `num_free_slots` but there is
        enough reclaimable space in `num_freeable_slots`, entries will be
        evicted from `freeable` (their mm_hash appended to `freed`) until
        enough space is available, and then this method returns True. 
        Older entries are evicted first.
        
        Returns False only if the requested number of tokens exceeds both 
        the free and reclaimable capacities combined.

        Args:
            request: The request containing the multimodal input.
            input_id: Index of the multimodal input within the request.
            encoder_compute_budget: Number of encoder tokens allowed to be 
                computed when this method is invoked.
            num_tokens_to_schedule: Number of tokens already scheduled to be 
                allocated with cache space when this method is invoked.

        Returns:
            True if there's enough capacity to hold the encoder output for this
            input (possibly after reclaiming `freeable` entries); otherwise
            False.

        Note: This method does not allocate physical memory for the encoder 
        output but only the state of EncoderCacheManager.
        """
        num_tokens = request.get_num_encoder_tokens(input_id)

        # Not enough compute budget
        if num_tokens > encoder_compute_budget:
            return False

        num_tokens += num_tokens_to_schedule

        # Enough free slots
        if num_tokens <= self.num_free_slots:
            return True

        # Not enough reclaimable slots
        if num_tokens > self.num_freeable_slots:
            return False

        # Not enough free slots but enough reclaimable slots
        # NOTE: Eviction takes place here, but physical memory is not freed
        # until model runner is notified by the scheduler output.
        while num_tokens > self.num_free_slots:
            self._evict()
        return True

    def _evict(self) -> None:
        """Evict a freeable slot"""
        mm_hash, num_free_token = self.freeable.popitem(last=False)
        del self.cached[mm_hash]
        self.freed.append(mm_hash)
        self.num_free_slots += num_free_token

        return mm_hash

    def allocate(self, request: Request, input_id: int) -> None:
        """Allocate cache space for a multimodal input's encoder output.

        This reserves cache space for storing the encoder output of the
        specified multimodal input. The actual encoder output storage happens in
        the model runner; this method updates the manager's bookkeeping.

        Note:
            This method assumes can_allocate() returned True for the same input.
        """

        mm_hash = request.mm_hashes[input_id]
        request_id = request.request_id
        if mm_hash not in self.cached:
            self.cached[mm_hash] = set()

        num_encoder_tokens = request.get_num_encoder_tokens(input_id)

        # NOTE: Encoder cache should always have enough space for encoder inputs
        # that are scheduled since eviction takes place at can_allocate().
        assert self.num_free_slots >= num_encoder_tokens
        assert self.num_freeable_slots >= num_encoder_tokens

        self.cached[mm_hash].add(request_id)
        self.num_free_slots -= num_encoder_tokens
        self.num_freeable_slots -= num_encoder_tokens

    def allocate_remote(self, request: Request, input_id: int) -> None:
        """Allocate cache space for remote encoder input.
        """
        mm_hash = request.mm_hashes[input_id]
        self.allocate(request, input_id)
        self.receiving.add(mm_hash)

    def get_cached_input_ids(self, request: Request) -> set[int]:
        """Get all cached multimodal input IDs for a request.

        Returns the set of input IDs whose `mm_hash` exists in the cache map.
        This includes entries that are currently unreferenced (and thus present
        in `freeable`); for such entries, freeing for this request will be a
        no-op.
        """
        return {
            input_id
            for input_id in range(len(request.mm_hashes))
            if request.mm_hashes[input_id] in self.cached
        }

    def free_encoder_input(self, request: Request, input_id: int) -> None:
        """Free the request's reference to the encoder input (`mm_data`)

        When the reference set for the corresponding `mm_hash` becomes empty,
        the entry is appended to `freeable` and `num_freeable_slots` is
        increased by the number of encoder tokens for that input. 

        The entry is NOT physically freed until capacity is needed (e.g., by
        `can_allocate`).
        """
        req_id = request.request_id
        mm_hash = request.mm_hashes[input_id]
        # The mm_hash not in cache or the req_id set is empty
        if not self.cached.get(mm_hash, None):
            return
        self.cached[mm_hash].discard(req_id)
        if not self.cached[mm_hash]:
            num_tokens = request.get_num_encoder_tokens(input_id)
            self.freeable[mm_hash] = num_tokens
            self.num_freeable_slots += num_tokens

    def free(self, request: Request) -> None:
        """Free all encoder input cache reference held by *request*.

        For each cached input ID, `free_encoder_input` is invoked.  
        The data stays in memory until eviction is triggered by a future 
        attempt allocation called by 'can_allocate'.

        Typically called when a request is finished, cancelled, or aborted.
        """
        input_ids = self.get_cached_input_ids(request).copy()
        for input_id in input_ids:
            self.free_encoder_input(request, input_id)

    def get_freed_mm_hashes(self) -> list[str]:
        """Get and clear the list of recently freed encoder cache entries.

        Returns:
            List of mm_hash strings that were actually evicted since the last
            call to be used by the scheduler to notify workers about which 
            encoder outputs can be removed from their caches. The internal 
            list is cleared after this call. 
        """
        freed = self.freed
        self.freed = []
        return freed

    def get_segments(self, mm_hash: str):
        """Used for EncoderBlockCacheManager"""
        return []

    def get_required_encoder_inputs(self, request: Request):
        """Get the list of mm_hash that is neither cached nor in the encoder disaggregation
        receiving queue
        """
        required_inputs = list()
        receving_mm_hashes = list()
        for i, mm_hash in enumerate(request.mm_hashes):
            if mm_hash not in self.cached:
                required_inputs.append(i)
            if mm_hash in self.receiving:
                receving_mm_hashes.append(mm_hash)

        return required_inputs, receving_mm_hashes

    def recv_finished(self, request: Request):
        for mm_hash in request.mm_hashes:
            if mm_hash in self.receiving:
                return False
        return True

    def finish_recv(self, request: Request):
        for mm_hash in request.mm_hashes:
            if mm_hash in self.receiving:
                self.receiving.remove(mm_hash)


@dataclass
class MemorySegment:
    """Represents a contiguous memory segment"""
    address: int
    size: int

    @property
    def end_address(self):
        return self.address + self.size


class EncoderBlockCacheManager(EncoderCacheManager):

    def __init__(self, cache_size: int):
        super().__init__(cache_size)

        # Track free segments sorted by address for coalescing
        self.free_segments_by_addr = SortedList(key=lambda x: x.address)

        # Track free segments sorted by size for allocation
        self.free_segments_by_size = SortedList(
            key=lambda x: (-x.size, x.address))  # Negative for largest-first

        # Track allocations by hash key
        # hash -> list of segments that store this tensor
        self.allocations = dict[str, list[MemorySegment]]()

        init_segment = MemorySegment(address=0, size=cache_size)
        self.free_segments_by_addr.add(init_segment)
        self.free_segments_by_size.add(init_segment)

    def _find_segments_for_size(self, size: int) -> list[MemorySegment]:
        """
        Find a combination of free segments to fit the requested size.
        Uses a greedy approach: takes largest segments first.
        """
        needed_size = size
        selected_segments = []

        # First, try to find a single segment that fits
        for segment in self.free_segments_by_size:
            if segment.size >= size:
                # Found a single segment that's large enough
                return [segment]

        # Otherwise, collect multiple segments
        # Copy the list since we'll be modifying it
        available = list(self.free_segments_by_size)

        for segment in available:
            if needed_size <= 0:
                break

            if segment.size <= needed_size:
                # Take the entire segment
                selected_segments.append(segment)
                needed_size -= segment.size
            else:
                # Take only what we need (will split later)
                selected_segments.append(segment)
                needed_size = 0
                break

        if needed_size > 0:
            # Couldn't find enough space
            return []

        return selected_segments

    def _coalesce_free_segments(self):
        """Merge all adjacent free segments to reduce fragmentation"""
        if len(self.free_segments_by_addr) <= 1:
            return

        merged_segments = []
        current = None

        for segment in self.free_segments_by_addr:
            if current is None:
                current = MemorySegment(segment.address, segment.size)
            elif current.end_address == segment.address:
                # Merge with current
                current.size += segment.size
            else:
                # Can't merge, save current and start new
                merged_segments.append(current)
                current = MemorySegment(segment.address, segment.size)

        if current:
            merged_segments.append(current)

        # Replace the free segment lists
        self.free_segments_by_addr.clear()
        self.free_segments_by_size.clear()

        for segment in merged_segments:
            self.free_segments_by_addr.add(segment)
            self.free_segments_by_size.add(segment)

    def allocate(self, request: Request, input_id: int):
        """Allocate cache space for a multimodal input's encoder output.
        Returns:
            mm_hash ->  allocated segments in the multi-modal hash.
        """
        super().allocate(request, input_id)

        mm_hash = request.mm_hashes[input_id]
        num_encoder_tokens = request.get_num_encoder_tokens(input_id)
        # Find segments to use
        segments_to_use = self._find_segments_for_size(num_encoder_tokens)
        if not segments_to_use:
            logger.error(
                "Failed to allocate %s tokens to the encoder cache for hash %s",
                num_encoder_tokens, mm_hash)
            return []

        allocated_segments = []
        total_used = 0

        for segment in segments_to_use:
            # Remove from free lists
            self.free_segments_by_addr.remove(segment)
            self.free_segments_by_size.remove(segment)

            # Determine how much of this segment to use
            remaining_tokens = num_encoder_tokens - total_used
            use_size = min(segment.size, remaining_tokens)

            # If segment is larger than needed, split it
            if segment.size > use_size:
                # Create remainder free segment
                remainder = MemorySegment(address=segment.address + use_size,
                                          size=segment.size - use_size)
                self.free_segments_by_addr.add(remainder)
                self.free_segments_by_size.add(remainder)

                # Adjust current segment size
                segment.size = use_size

            # Track allocation
            allocated_segments.append(segment)

            total_used += use_size

            if total_used >= num_encoder_tokens:
                break

        # Store allocation info
        self.allocations[mm_hash] = allocated_segments

    def get_segments(self, mm_hash: str) -> list[MemorySegment]:
        return self.allocations.get(mm_hash, [])

    def _evict(self) -> None:
        """Evict a freeable slot

        From the block cache manager, it actually frees the memory in the
        segment tree.
        """
        mm_hash = super()._evict()
        segments = self.allocations.pop(mm_hash)

        for segment in segments:
            segment.freed = True
            self.free_segments_by_addr.add(segment)
            self.free_segments_by_size.add(segment)

        self._coalesce_free_segments()


def compute_encoder_budget(
    model_config: "ModelConfig",
    scheduler_config: "SchedulerConfig",
    mm_registry: MultiModalRegistry,
) -> tuple[int, int]:
    """Compute the encoder cache budget based on the model and scheduler 
    configurations.

    Returns:
        - Compute budget for encoder execution, measured in number of tokens
            from the input sequence.
        - Space budget for encoder cache size, measured in number of tokens
            from the input sequence.
    """
    if mm_registry.supports_multimodal_inputs(model_config):
        max_tokens_by_modality = mm_registry \
            .get_max_tokens_per_item_by_nonzero_modality(model_config)

        return compute_mm_encoder_budget(
            scheduler_config,
            max_tokens_by_modality,
        )

    return compute_text_encoder_budget(scheduler_config)


def compute_text_encoder_budget(
        scheduler_config: "SchedulerConfig") -> tuple[int, int]:
    """Compute the encoder cache budget based on the model and scheduler 
    configurations for a text-only model.

    Args:
        scheduler_config: Scheduler configuration.

    Returns:
        - Compute budget for encoder execution, in unit of number of tokens 
            in the input sequence.
        - Space budget for encoder cache size, in unit of number of tokens 
            in the input sequence.
    """
    # Currently text-only encoder-decoder models are not supported
    return 0, 0


def compute_mm_encoder_budget(
    scheduler_config: "SchedulerConfig",
    max_tokens_by_modality: Mapping[str, int],
) -> tuple[int, int]:
    """Compute the encoder cache budget based on the model and scheduler 
    configurations for a multimodal model.

    Args:
        scheduler_config: Scheduler configuration.
        max_tokens_by_modality: The maximum number of tokens for each
            non-text modality.

    Returns:
        - Compute budget for encoder execution, measured in number of tokens
            from the input sequence.
        - Space budget for encoder cache size, measured in number of tokens
            from the input sequence.
    """

    if not max_tokens_by_modality:
        logger.warning(
            "All non-text modalities supported by the model have been "
            "explicitly disabled via limit_mm_per_prompt. Encoder cache will "
            "not be initialized.")
        return 0, 0

    max_tokens_per_mm_item = max(max_tokens_by_modality.values())

    if (scheduler_config.disable_chunked_mm_input and max_tokens_per_mm_item
            > scheduler_config.max_num_batched_tokens):
        raise ValueError(
            "Chunked MM input disabled but max_tokens_per_mm_item "
            f"({max_tokens_per_mm_item}) is larger than max_num_batched_tokens"
            f" ({scheduler_config.max_num_batched_tokens}). Please increase "
            "max_num_batched_tokens.")

    encoder_compute_budget = max(scheduler_config.max_num_encoder_input_tokens,
                                 max_tokens_per_mm_item)
    encoder_cache_size = max(scheduler_config.encoder_cache_size,
                             max_tokens_per_mm_item)

    return encoder_compute_budget, encoder_cache_size
