# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import OrderedDict
from collections.abc import Mapping
from typing import TYPE_CHECKING

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
        
        # Blocking 
        self.preallocated: dict[str, dict[str, dict[int, int]]] = {}

        # mm_hash of mm_data => num_encoder_tokens of the mm_data
        self.freeable: OrderedDict[str, int] = OrderedDict()
        self.freed: list[str] = []

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
        # If mm_hash is in preallocated then it will not be in freeable 
        if not self.cached[mm_hash] and mm_hash not in self.preallocated:
            num_tokens = self.freeable.pop(mm_hash)
            self.num_freeable_slots -= num_tokens

        self.cached[mm_hash].add(request.request_id)
        return True

    def can_allocate_tokens(self, num_tokens):
        """Check if the specified number of tokens can be allocated in the cache.

        This method determines whether there is sufficient cache capacity to store
        the requested number of encoder tokens. If there isn't enough free space
        but there is enough reclaimable space, it will evict entries from the
        freeable list to make room.

        Args:
            num_tokens: The number of encoder tokens to allocate.

        Returns:
            True if the tokens can be allocated (either immediately or after
            eviction); False if there isn't enough total capacity even after
            reclaiming all freeable entries.
        """

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
            mm_hash, num_free_token = self.freeable.popitem(last=False)
            del self.cached[mm_hash]
            self.freed.append(mm_hash)
            self.num_free_slots += num_free_token
        return True


    def can_allocate(self, request: Request, input_id: int,
                     encoder_compute_budget: int,
                     num_tokens_to_schedule: int) -> bool:
        """Check if there's sufficient cache space for a multimodal input.

        This method verifies if the encoder output for the specified input
        can be allocated given the current compute budget and cache capacity
        constraints. It does not modify the cache state.

        The method checks two constraints:
        1. Compute budget: Whether the input's encoder tokens fit within the
        available encoder compute budget.
        2. Cache capacity: Whether there's enough space (free or reclaimable)
        to store the encoder output, accounting for tokens already scheduled.

        Args:
            request: The request containing the multimodal input.
            input_id: Index of the multimodal input within the request.
            encoder_compute_budget: Number of encoder tokens allowed to be 
                computed when this method is invoked.
            num_tokens_to_schedule: Number of tokens already scheduled to be 
                allocated with cache space when this method is invoked.

        Returns:
            True if both compute budget and cache capacity constraints are
            satisfied; False otherwise.

        Note: 
            - This method only checks feasibility without modifying cache state.
            - Actual eviction (if needed) happens in can_allocate_tokens().
            - The allocate() method should be called to reserve the space after
            this check passes.
        """
        num_tokens = request.get_num_encoder_tokens(input_id)
        
        # Not enough compute budget
        if num_tokens > encoder_compute_budget:
            return False
        
        return self.can_allocate_tokens(num_tokens + num_tokens_to_schedule)

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
        if not self.cached[mm_hash] and mm_hash not in self.preallocated:
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
    
    ########################################################################
    # Encode-Prefill-Decode Disaggregation Related Methods
    ########################################################################

    def free_encoder_input_after_finish(
        self, req_id: str, num_tokens: int, mm_hash: str
    ) -> None:
        """Free a request's reference in cached dictionary for mm input after
        request is finished long time ago.

        Removes the request ID from the cached mm input's reference set. When
        the  reference set becomes empty AND the entry is not preallocated by
        any  pending requests, the entry is marked as freeable.

        This method is used in disaggregated settings where the request object 
        may not be available, requiring explicit parameters for the multimodal 
        input metadata.

        Args:
            req_id: ID of the request releasing the reference.
            num_tokens: Number of encoder tokens associated with this input.
            mm_hash: Hash identifier of the multimodal input data.

        Note: 
            The entry is NOT physically freed until capacity is needed (e.g., by
            `can_allocate_tokens`). Entries that are preallocated remain 
            unfreeable even with zero references to prevent premature eviction.
        """
        # The mm_hash not in cache or the req_id set is empty
        if not self.cached.get(mm_hash, None):
            return
        self.cached[mm_hash].discard(req_id)
        if not self.cached[mm_hash] and mm_hash not in self.preallocated:
            self.freeable[mm_hash] = num_tokens
            self.num_freeable_slots += num_tokens

    def preallocate(self, req_id: str, input_id: int, 
                    num_tokens: int, mm_hash: str) -> bool:
        """Reserve cache space for an encoder input before actual allocation.
        
        Used in disaggregated settings to coordinate cache allocation across
        different processing stage instances. Helps in prevention of premature
        eviction of entries that will be needed and tracks which requests will
        use which inputs.
        
        Args:
            req_id: ID of the request making the preallocation.
            input_id: Index of the multimodal input within the request.
            num_tokens: Number of encoder tokens to preallocate.
            mm_hash: Hash identifier of the multimodal input data.
            
        Returns:
            True if encoder cache needs to be received (entry not cached),
            False if entry is already cached or will be provided by another
                request.
        """
        
        is_mm_hash_preallocated = (mm_hash in self.preallocated)
        is_cached = (mm_hash in self.cached)
        is_referenced = (bool(self.cached[mm_hash]) if is_cached else False)

        # Add mm_input preallocation fact to self.preallocated
        if not is_mm_hash_preallocated:
            self.preallocated[mm_hash] = {}

        preallocated_reqs = self.preallocated[mm_hash]
        if req_id not in preallocated_reqs:
            preallocated_reqs[req_id] = {}
        preallocated_reqs[req_id][input_id] = num_tokens
        
        if is_cached:
            # Block freeableness of the targeted mm_hash if it's freeable
            if not (is_referenced or is_mm_hash_preallocated):
                num_tokens = self.freeable.pop(mm_hash)
                self.num_freeable_slots -= num_tokens
            return False
        elif not is_mm_hash_preallocated:
            self.num_free_slots -= num_tokens
            self.num_freeable_slots -= num_tokens
            return True
        
        # Already preallocated in past, not cached, that means that encoder
        # cache will be injected by some other (req_id, input_id) pair
        return False

    def finalize_allocation(
        self, req_id: str, input_id: int, mm_hash: str, skipped: bool
    ) -> None:
        """Complete the allocation process for a preallocated encoder input.
        
        Converts a preallocation into an actual allocation or releases the
        preallocation if it was skipped. This method is called after the
        encoder cache is injected.
        
        Args:
            req_id: ID of the request finalizing allocation.
            input_id: Index of the multimodal input within the request.
            mm_hash: Hash identifier of the multimodal input data.
            skipped: True if this request skipped encoding (e.g., another
                    request provided the cached data), False otherwise.
        """
        preallocated_reqs = self.preallocated[mm_hash]
        num_tokens = preallocated_reqs[req_id].pop(input_id)        
        is_preallocated = True
        
        if not preallocated_reqs[req_id]:
            preallocated_reqs.pop(req_id)        
        if not self.preallocated[mm_hash]:
            self.preallocated.pop(mm_hash)
            is_preallocated = False
        
        if mm_hash not in self.cached:
            self.cached[mm_hash] = set()
        if not skipped:
            self.cached[mm_hash].add(req_id)
        elif not self.cached[mm_hash] and not is_preallocated:
            self.freeable[mm_hash] = num_tokens
            self.num_freeable_slots += num_tokens


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
