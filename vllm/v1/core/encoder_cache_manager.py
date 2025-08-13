# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.multimodal import MultiModalRegistry
from vllm.v1.request import Request
from collections import OrderedDict

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

    Note that no caching is shared between requests at this time. If the same
    input is used across multiple requests, it will be reprocessed for each
    request.
    
    Args:
        cache_size: Limit the size of the cache, measured by the number of
                    tokens from the input sequence.

    Attributes:
        cache_size: Total cache capacity in encoder tokens.
        num_free_slots: Current available cache capacity in encoder tokens.
        num_free_able_slots: Capacity that can be immediately reclaimed by
            evicting entries with zero references (in encoder tokens).
        cached: Mapping from mm_hash to a set of request IDs that currently
            reference the cached entry. If the set is empty, the entry exists
            but is not referenced by any request and is eligible for
            reclamation.
        freed_able: List of tuples (mm_hash, num_tokens) representing entries
            whose reference count has dropped to zero and that can be freed to
            make space when needed.
        freed: List of mm_hash strings that were actually evicted since the
            last call to get_freed_mm_hashes(). This list is cleared on return.
    """

    # ------------------------------------------------------------------ #
    def __init__(self, cache_size: int):
        self.cache_size = cache_size
        self.num_free_slots = cache_size
        self.num_free_able_slots = cache_size

        self.cached: dict[str, set[str]] = {}

        # List of mm_hash
        self.freed_able: OrderedDict[str, int] = OrderedDict()
        self.freed: list[str] = []

    def has_cache(self, request: Request, input_id: int) -> bool:
        """Check whether the encoder output for a specific multimodal input is
        cached.

        If the entry exists but is currently unreferenced (present with an
        empty request set and listed in `freed_able`), it is moved back to the
        active set: the item is removed from `freed_able`,
        `num_free_able_slots` is decreased accordingly, and the current
        request ID is added to its reference set.
        """
        mm_hash = request.mm_hashes[input_id]
        request_id = request.request_id
        # Not cached at all
        if mm_hash not in self.cached:
            return False

        # Cached but currently not referenced by any request
        if not self.cached[mm_hash]:
            num_tokens = self.freed_able.pop(mm_hash)
            self.num_free_able_slots -= num_tokens
        self.cached[mm_hash].add(request_id)
        return True

    def can_allocate(self, request: Request, input_id: int) -> bool:
        """Check if there's sufficient cache space for a multimodal input.

        If there is not enough free space in `num_free_slots` but there is
        enough reclaimable space in `num_free_able_slots`, entries will be
        evicted from `freed_able` (their mm_hash appended to `freed`) until
        enough space is available, and then this method returns True. Returns
        False only if the requested number of tokens exceeds both the free and
        reclaimable capacities combined.

        Args:
            request: The request containing the multimodal input.
            input_id: Index of the multimodal input within the request.

        Returns:
            True if there's enough capacity to hold the encoder output for this
            input (possibly after reclaiming `freed_able` entries); otherwise
            False.
        """
        num_tokens = request.get_num_encoder_tokens(input_id)
        if num_tokens <= self.num_free_slots:
            return True
        if num_tokens > self.num_free_able_slots:
            return False
        # Free some slot
        while num_tokens > self.num_free_slots:
            mm_hash, num_free_token = self.freed_able.popitem(last=False)
            del self.cached[mm_hash]
            self.freed.append(mm_hash)
            self.num_free_slots += num_free_token
        return True

    def allocate(self, request: Request, input_id: int) -> None:
        """Allocate cache space for a multimodal input's encoder output.

        This reserves cache space for storing the encoder output of the
        specified multimodal input. The actual encoder output storage happens in
        the model runner; this method updates the manager's bookkeeping.

        Note:
            This method assumes can_allocate() returned True for the same input
            and will decrease both `num_free_slots` and `num_free_able_slots`
            by the number of encoder tokens for the input.
        """
        mm_hash = request.mm_hashes[input_id]
        request_id = request.request_id
        if mm_hash not in self.cached:
            self.cached[mm_hash] = set()

        self.cached[mm_hash].add(request_id)
        self.num_free_slots -= request.get_num_encoder_tokens(input_id)
        self.num_free_able_slots -= request.get_num_encoder_tokens(input_id)

    def get_cached_input_ids(self, request: Request) -> set[int]:
        """Get all cached multimodal input IDs for a request.

        Returns the set of input IDs whose `mm_hash` exists in the cache map.
        This includes entries that are currently unreferenced (and thus present
        in `freed_able`); for such entries, freeing for this request will be a
        no-op.
        """
        return {
            input_id
            for input_id in range(len(request.mm_hashes))
            if request.mm_hashes[input_id] in self.cached
        }

    def free_encoder_input(self, request: Request, input_id: int) -> None:
        """Free cache space for a single multimodal input's encoder output.

        When the reference set for the corresponding `mm_hash` becomes empty,
        the entry is appended to `freed_able` and `num_free_able_slots` is
        increased by the number of encoder tokens for that input. The entry is
        not physically freed until capacity is needed (e.g., by
        `can_allocate`).
        """
        req_id = request.request_id
        mm_hash = request.mm_hashes[input_id]
        if mm_hash not in self.cached:
            return
        if not self.cached[mm_hash]:
            return
        self.cached[mm_hash].discard(req_id)
        if not self.cached[mm_hash]:
            num_tokens = request.get_num_encoder_tokens(input_id)
            self.freed_able[mm_hash] = num_tokens
            self.num_free_able_slots += num_tokens

    def free(self, request: Request) -> None:
        """Free all cached encoder outputs for a request.

        Typically called when a request is finished, cancelled, or aborted.
        """
        input_ids = self.get_cached_input_ids(request).copy()
        for input_id in input_ids:
            self.free_encoder_input(request, input_id)

    def get_freed_mm_hashes(self) -> list[str]:
        """Get and clear the list of recently freed encoder cache entries.

        Returns:
            List of mm_hash strings that were actually evicted since the last
            call. The internal list is cleared after this call.
        """
        freed = self.freed
        self.freed = []
        return freed


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

    if not mm_registry.supports_multimodal_inputs(model_config):
        return 0, 0

    # TODO: handle encoder-decoder models once we support them.
    (
        encoder_compute_budget,
        encoder_cache_size,
    ) = _compute_encoder_budget_multimodal(
        model_config,
        scheduler_config,
        mm_registry,
    )

    return encoder_compute_budget, encoder_cache_size


def _compute_encoder_budget_multimodal(
    model_config: "ModelConfig",
    scheduler_config: "SchedulerConfig",
    mm_registry: MultiModalRegistry,
) -> tuple[int, int]:
    """Compute the encoder cache budget for a multimodal model.

    Returns:
        - Compute budget for encoder execution, measured in number of tokens
          from the input sequence.
        - Space budget for encoder cache size, measured in number of tokens
          from the input sequence.
    """

    max_tokens_by_modality_dict = mm_registry \
        .get_max_tokens_per_item_by_nonzero_modality(model_config)

    if not max_tokens_by_modality_dict:
        logger.warning(
            "All non-text modalities supported by the model have been "
            "explicitly disabled via limit_mm_per_prompt. Encoder cache will "
            "not be initialized.")
        return 0, 0

    _, max_tokens_per_mm_item = max(max_tokens_by_modality_dict.items(),
                                    key=lambda item: item[1])

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
