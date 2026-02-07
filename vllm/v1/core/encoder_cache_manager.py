# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import OrderedDict
from collections.abc import Mapping
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.config import SchedulerConfig

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

    NOTE: The EncoderCacheManager operates on the level of multimodal embeddings
    instead of encoder tokens (i.e. all tokens that represent the multimodal data
    in the input sequence). This means all break/text tokens in-between multimodal
    embeddings are not considered with respect to the cache size and the number
    of free slots.

    Args:
        cache_size: Limit the size of the cache, measured by the number of
                    encoder embeddings from the input sequence.

    Attributes:
        cache_size: Total cache capacity in encoder embeddings.
        num_free_slots: Current available cache capacity in encoder embeddings.
        num_freeable_slots: Capacity that can be immediately reclaimed by
            evicting entries with zero references (in encoder embeddings).
        cached: Mapping from mm_hash to a set of request IDs that currently
            reference the cached entry. If the set is empty, the entry exists
            but is not referenced by any request and is eligible for
            reclamation.
        freeable: List of tuples (mm_hash, num_encoder_embeds) representing entries
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

        # mm_hash of mm_data => num_encoder_embeds of the mm_data
        self.freeable: OrderedDict[str, int] = OrderedDict()
        self.freed: list[str] = []

    def reset(self) -> None:
        """Reset the encoder cache to its initial state.

        This clears all cached encoder outputs and resets capacity tracking.
        Called when model weights are updated to invalidate stale embeddings.
        """
        self.cached.clear()
        self.freeable.clear()
        self.freed.clear()
        self.num_free_slots = self.cache_size
        self.num_freeable_slots = self.cache_size

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
        mm_hash = request.mm_features[input_id].identifier
        # Not cached at all
        if mm_hash not in self.cached:
            return False

        # Cached but currently not referenced by any request
        if not self.cached[mm_hash]:
            num_encoder_embeds = self.freeable.pop(mm_hash)
            self.num_freeable_slots -= num_encoder_embeds

        self.cached[mm_hash].add(request.request_id)
        return True

    def can_allocate(
        self,
        request: Request,
        input_id: int,
        encoder_compute_budget: int,
        num_embeds_to_schedule: int,
    ) -> bool:
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
            encoder_compute_budget: Number of encoder embeddings allowed to be
                computed when this method is invoked.
            num_embeds_to_schedule: Number of encoder embeddings already scheduled to be
                allocated with cache space when this method is invoked.

        Returns:
            True if there's enough capacity to hold the encoder output for this
            input (possibly after reclaiming `freeable` entries); otherwise
            False.

        Note: This method does not allocate physical memory for the encoder
        output but only the state of EncoderCacheManager.
        """
        num_embeds = request.get_num_encoder_embeds(input_id)

        # Not enough compute budget
        if num_embeds > encoder_compute_budget:
            return False

        num_embeds += num_embeds_to_schedule

        # Enough free slots
        if num_embeds <= self.num_free_slots:
            return True

        # Not enough reclaimable slots
        if num_embeds > self.num_freeable_slots:
            return False

        # Not enough free slots but enough reclaimable slots
        # NOTE: Eviction takes place here, but physical memory is not freed
        # until model runner is notified by the scheduler output.
        while num_embeds > self.num_free_slots:
            mm_hash, num_free_embeds = self.freeable.popitem(last=False)
            del self.cached[mm_hash]
            self.freed.append(mm_hash)
            self.num_free_slots += num_free_embeds
        return True

    def allocate(self, request: Request, input_id: int) -> None:
        """Allocate cache space for a multimodal input's encoder output.

        This reserves cache space for storing the encoder output of the
        specified multimodal input. The actual encoder output storage happens in
        the model runner; this method updates the manager's bookkeeping.

        Note:
            This method assumes can_allocate() returned True for the same input.
        """

        mm_hash = request.mm_features[input_id].identifier
        request_id = request.request_id
        if mm_hash not in self.cached:
            self.cached[mm_hash] = set()

        num_encoder_embeds = request.get_num_encoder_embeds(input_id)

        # NOTE: Encoder cache should always have enough space for encoder inputs
        # that are scheduled since eviction takes place at can_allocate().
        assert self.num_free_slots >= num_encoder_embeds
        assert self.num_freeable_slots >= num_encoder_embeds

        self.cached[mm_hash].add(request_id)
        self.num_free_slots -= num_encoder_embeds
        self.num_freeable_slots -= num_encoder_embeds

    def get_cached_input_ids(self, request: Request) -> set[int]:
        """Get all cached multimodal input IDs for a request.

        Returns the set of input IDs whose `mm_hash` exists in the cache map.
        This includes entries that are currently unreferenced (and thus present
        in `freeable`); for such entries, freeing for this request will be a
        no-op.
        """
        return {
            input_id
            for input_id in range(len(request.mm_features))
            if request.mm_features[input_id].identifier in self.cached
        }

    def free_encoder_input(self, request: Request, input_id: int) -> None:
        """Free the request's reference to the encoder input (`mm_data`)

        When the reference set for the corresponding `mm_hash` becomes empty,
        the entry is appended to `freeable` and `num_freeable_slots` is
        increased by the number of encoder embeddings for that input.

        The entry is NOT physically freed until capacity is needed (e.g., by
        `can_allocate`).
        """
        req_id = request.request_id
        mm_hash = request.mm_features[input_id].identifier
        # The mm_hash not in cache or the req_id set is empty
        if not self.cached.get(mm_hash, None):
            return
        self.cached[mm_hash].discard(req_id)
        if not self.cached[mm_hash]:
            num_encoder_embeds = request.get_num_encoder_embeds(input_id)
            self.freeable[mm_hash] = num_encoder_embeds
            self.num_freeable_slots += num_encoder_embeds

    def free(self, request: Request) -> None:
        """Free all encoder input cache reference held by *request*.

        For each cached input ID, `free_encoder_input` is invoked.
        The data stays in memory until eviction is triggered by a future
        attempt allocation called by 'can_allocate'.

        Typically called when a request is finished, cancelled, or aborted.
        """
        input_ids = self.get_cached_input_ids(request)
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


def _set_max_num_batched_encoder_input_tokens(
    scheduler_config: "SchedulerConfig",
    max_tokens_per_mm_item: int,
) -> int:
    max_num_batched_encoder_input_tokens = (
        scheduler_config.max_num_batched_encoder_input_tokens
    )
    max_num_batched_tokens = scheduler_config.max_num_batched_tokens

    if max_num_batched_encoder_input_tokens is None:
        max_num_batched_encoder_input_tokens = max_num_batched_tokens

    max_num_batched_encoder_input_tokens = max(
        max_num_batched_encoder_input_tokens, max_tokens_per_mm_item
    )

    if scheduler_config.max_num_batched_encoder_input_tokens is None:
        logger.info_once(
            "Defaulting max_num_batched_encoder_input_tokens to %d.",
            max_num_batched_encoder_input_tokens,
            scope="local",
        )
    elif (
        max_num_batched_encoder_input_tokens
        != scheduler_config.max_num_batched_encoder_input_tokens
    ):
        logger.warning_once(
            "Overriding max_num_batched_encoder_input_tokens with ",
            "max_tokens_per_mm_item=%d because it is larger than the "
            "user-specified value of %d. "
            "You can silence this warning by passing a larger value of "
            "max_num_batched_encoder_input_tokens.",
            max_num_batched_encoder_input_tokens,
            scheduler_config.max_num_batched_encoder_input_tokens,
            scope="local",
        )

    scheduler_config.max_num_batched_encoder_input_tokens = (
        max_num_batched_encoder_input_tokens
    )

    return max_num_batched_encoder_input_tokens


def _set_encoder_cache_size(
    scheduler_config: "SchedulerConfig",
    max_tokens_per_mm_item: int,
) -> int:
    encoder_cache_size = scheduler_config.encoder_cache_size
    max_num_batched_encoder_input_tokens = (
        scheduler_config.max_num_batched_encoder_input_tokens
    )

    assert max_num_batched_encoder_input_tokens is not None, (
        "Should be called after `_set_max_num_batched_encoder_input_tokens`"
    )

    if encoder_cache_size is None:
        encoder_cache_size = max_num_batched_encoder_input_tokens
    elif encoder_cache_size < max_num_batched_encoder_input_tokens:
        raise ValueError(
            "The encoder cache must be able to store all "
            f"encoder inputs in a single batch, so {encoder_cache_size=}"
            f"cannot be less than {max_num_batched_encoder_input_tokens=}. "
            "Please increase encoder_cache_size."
        )

    encoder_cache_size = max(encoder_cache_size, max_tokens_per_mm_item)

    if scheduler_config.encoder_cache_size is None:
        logger.info_once(
            "Defaulting encoder_cache_size to %d.",
            encoder_cache_size,
            scope="local",
        )
    elif encoder_cache_size != scheduler_config.encoder_cache_size:
        logger.warning_once(
            "Overriding encoder_cache_size with ",
            "max_tokens_per_mm_item=%d because it is larger than the "
            "user-specified value of %d. "
            "You can silence this warning by passing a larger value of "
            "encoder_cache_size.",
            encoder_cache_size,
            scheduler_config.encoder_cache_size,
            scope="local",
        )

    scheduler_config.encoder_cache_size = encoder_cache_size

    return encoder_cache_size


def compute_mm_encoder_budget(
    scheduler_config: "SchedulerConfig",
    mm_max_toks_per_item: Mapping[str, int],
) -> tuple[int, int]:
    """Compute the encoder cache budget based on the model and scheduler
    configurations for a multimodal model.

    Args:
        scheduler_config: Scheduler configuration.
        mm_max_toks_per_item: The maximum number of tokens per item for each
            non-text modality.

    Returns:
        - Compute budget for encoder execution, measured in number of tokens
            from the input sequence.
        - Space budget for encoder cache size, measured in number of tokens
            from the input sequence.

    Note:
        This may update `scheduler_config.max_num_batched_encoder_input_tokens` and
        `scheduler_config.encoder_cache_size` in place with new values.
    """

    if not mm_max_toks_per_item:
        logger.warning(
            "All non-text modalities supported by the model have been "
            "explicitly disabled via limit_mm_per_prompt. Encoder cache will "
            "not be initialized."
        )
        return 0, 0

    max_tokens_per_mm_item = max(mm_max_toks_per_item.values())
    max_num_batched_tokens = scheduler_config.max_num_batched_tokens

    if (
        scheduler_config.disable_chunked_mm_input
        and max_tokens_per_mm_item > max_num_batched_tokens
    ):
        raise ValueError(
            f"Chunked MM input disabled but {max_tokens_per_mm_item=} "
            f"is larger than {max_num_batched_tokens=}. "
            "Please increase max_num_batched_tokens."
        )

    max_num_batched_encoder_input_tokens = _set_max_num_batched_encoder_input_tokens(
        scheduler_config, max_tokens_per_mm_item
    )
    encoder_cache_size = _set_encoder_cache_size(
        scheduler_config, max_tokens_per_mm_item
    )

    if max_num_batched_encoder_input_tokens > encoder_cache_size:
        raise ValueError(
            f"{max_num_batched_encoder_input_tokens=} cannot be greater than "
            f"{encoder_cache_size=}. "
            "Please decrease max_num_batched_encoder_input_tokens or "
            "increase encoder_cache_size."
        )

    return max_num_batched_encoder_input_tokens, encoder_cache_size


# NOTE (NickLucche): Temporary implementation for encoder-decoder models that only
# use the manager for scheduling purposes. Encoder-decoder models will eventually
# utilize the cache and this class will fold into EncoderCacheManager, as
# differences with MM models shrink.
class EncoderDecoderCacheManager(EncoderCacheManager):
    def __init__(self, cache_size: int):
        self.cache_size = cache_size
        self.num_free_slots = cache_size
        self.allocated: list[str] = []
        self.to_free: list[str] = []

    def reset(self) -> None:
        """Reset the encoder cache to its initial state."""
        self.num_free_slots = self.cache_size
        self.allocated.clear()
        self.to_free.clear()

    def check_and_update_cache(self, request: Request, input_id: int) -> bool:
        return False

    def can_allocate(
        self,
        request: Request,
        input_id: int,
        encoder_compute_budget: int,
        num_embeds_to_schedule: int,
    ) -> bool:
        num_encoder_embeds = request.get_num_encoder_embeds(input_id)
        # Not enough compute budget
        if num_encoder_embeds > encoder_compute_budget:
            return False

        num_encoder_embeds += num_embeds_to_schedule
        # Enough free slots
        return num_encoder_embeds <= self.num_free_slots

    def allocate(self, request: Request, input_id: int) -> None:
        num_encoder_embeds = request.get_num_encoder_embeds(input_id)
        self.num_free_slots -= num_encoder_embeds

        mm_hash = request.mm_features[input_id].identifier
        self.allocated.append(mm_hash)

    def free(self, request: Request) -> None:
        for input_id in range(len(request.mm_features)):
            self.free_encoder_input(request, input_id)

    def get_cached_input_ids(self, request: Request) -> set[int]:
        return set(range(len(request.mm_features)))

    def get_freed_mm_hashes(self) -> list[str]:
        # As encoder cache is not used for enc-dec models, we can free the entries here
        # The actual free happens in the runner, *before* the model is executed.
        # Therefore, `freeable` acts as a buffer to free the entries only after the
        # model is executed, mimicking the state transition of `EncoderCacheManager`.
        to_free = self.to_free
        self.to_free = self.allocated
        self.allocated = []
        return to_free

    def free_encoder_input(self, request: Request, input_id: int) -> None:
        num_encoder_embeds = request.get_num_encoder_embeds(input_id)
        self.num_free_slots += num_encoder_embeds
