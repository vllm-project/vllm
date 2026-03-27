# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from vllm.config import VllmConfig
from typing import TYPE_CHECKING, Dict

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
    """

    if not mm_max_toks_per_item:
        logger.warning(
            "All non-text modalities supported by the model have been "
            "explicitly disabled via limit_mm_per_prompt. Encoder cache will "
            "not be initialized."
        )
        return 0, 0

    max_tokens_per_mm_item = max(mm_max_toks_per_item.values())

    if (
        scheduler_config.disable_chunked_mm_input
        and max_tokens_per_mm_item > scheduler_config.max_num_batched_tokens
    ):
        raise ValueError(
            "Chunked MM input disabled but max_tokens_per_mm_item "
            f"({max_tokens_per_mm_item}) is larger than max_num_batched_tokens"
            f" ({scheduler_config.max_num_batched_tokens}). Please increase "
            "max_num_batched_tokens."
        )

    encoder_compute_budget = max(
        scheduler_config.max_num_encoder_input_tokens, max_tokens_per_mm_item
    )
    encoder_cache_size = max(
        scheduler_config.encoder_cache_size, max_tokens_per_mm_item
    )

    return encoder_compute_budget, encoder_cache_size


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

@dataclass
class CacheEntry:
    mm_hash: str        # Unique identifier of the multimodal input
    freq: int           # Access frequency
    clock: int          # Clock value used for aging
    num_embeds: int     # Number of slots occupied by this embedding
    cal_cost: int       # Theoretical recomputation cost of this embedding (used for score calculation)


class ScoreEncoderCacheManager(EncoderCacheManager):
    """
    Score-based encoder cache manager.

    The overall structure is a two-level cache:
        NPU cache (fast / small capacity)
        CPU cache (slower / large capacity)

    Core strategy:
    1. Newly generated encoder embeddings are first placed into the CPU cache
    2. If an entry is accessed frequently enough and has a sufficiently high score,
       it can be promoted to the NPU cache
    3. When the NPU cache runs out of space, entries with the lowest scores are evicted
    4. A clock-based aging mechanism is used to prevent stale hot entries
       from occupying the cache for too long
    """
    def __init__(self, cache_size: int, vllm_config: VllmConfig):
        # ---------------- NPU cache ----------------
        self.npu_num_free_slots = cache_size    # Empty slots
        self.npu_num_freeable_slots = cache_size    # Reclaimable capacity: reclaimable slots + empty slots

        # ---------------- CPU cache ----------------
        self.cpu_cache_size = vllm_config.score_encoder_cache_config.cpu_cache_slots
        self.cpu_num_free_slots = self.cpu_cache_size
        self.cpu_num_freeable_slots = self.cpu_cache_size

        # mm_hash of mm_data => ids of requests that reference the mm_data
        self.cached: dict[str, set[str]] = {}

        # Actual cache contents
        self.npu_cache: Dict[str, CacheEntry] = {}
        self.cpu_cache: Dict[str, CacheEntry] = {}

        # mm_hash of mm_data => num_encoder_embeds of the mm_data
        # Evictable cache entries (entries not referenced by any request)
        self.npu_freeable: Dict[str, CacheEntry] = {}
        self.cpu_freeable: OrderedDict[str, CacheEntry] = OrderedDict()

        # mm_hashes evicted in the previous round; after NPU eviction they may be placed into CPU,
        # and after CPU eviction they may also be recorded here

        self.req_cnt = 0

        self.watermark = vllm_config.score_encoder_cache_config.watermark
        self.promote_percentile = vllm_config.score_encoder_cache_config.promote_percentile
        self.max_clock = vllm_config.score_encoder_cache_config.max_clock
        self.clock_decay_every = vllm_config.score_encoder_cache_config.clock_decay_every

        # Actions to execute in the current round
        self.promoting: list[str] = []             # mm_hashes to be promoted from CPU -> NPU
        self.cpu_get_encoder_mm_hashes: list[str] = []  # mm_hashes whose embeddings need to be prefetched from CPU

        # ---------------- Load model config (used to estimate theoretical compute cost) ----------------
        self.attn_heads = vllm_config.model_config.hf_config.vision_config.num_heads
        self.hidden_size = vllm_config.model_config.hf_config.vision_config.hidden_size
        self.feedforward = vllm_config.model_config.hf_config.vision_config.intermediate_size

        # Hardware throughput (FLOPs)
        self.hardware_flops = 4 * 1e14

        # TODO: there may be more kinds of compute ways
        # Coefficients used to estimate the compute cost of encoder embeddings
        self.alpha = 4 * self.hidden_size + 5 * self.attn_heads
        self.beta = self.hidden_size * (8 * self.hidden_size + 6 * self.feedforward + 14)

        self.stats = EmbCacheStats()

    def score(self, ent: CacheEntry) -> float:
        return (ent.freq + ent.clock) * ent.cal_cost

    def evict_from_npu(self, ent: CacheEntry):
        """
        Evict an entry from the NPU cache.
        """
        del self.npu_cache[ent.mm_hash]
        self.freed.append(ent.mm_hash)
        self.npu_num_free_slots += ent.num_embeds

        self.stats.evict_npu += 1
        self.stats.npu_freed_entries += 1

    def should_promote(self, mm_hash: str) -> bool:
        """
        Determine whether an entry in the CPU cache should be promoted to the NPU cache.

        Logic:
        1. If the NPU has enough free space, promote directly
        2. If space is insufficient, decide based on the score percentile
        3. If needed, evict lower-score entries from the NPU cache
        """
        ent = self.cpu_cache[mm_hash]

        # No reclaimable space on the NPU, promotion is impossible
        if ent.num_embeds > self.npu_num_freeable_slots:
            self.stats.promote_fail_no_space += 1
            return False

        if ent.num_embeds <= self.npu_num_free_slots:
            # The NPU has free space, place it directly
            return True

        ent_value = self.score(ent)
        scored = []
        for cur_hash, cur_ent in self.npu_freeable.items():
            value = self.score(cur_ent)
            scored.append((value, cur_hash, cur_ent))

        scored.sort(key=lambda x: x[0])
        idx = max(0, min(len(scored) - 1, int(len(scored) * self.promote_percentile)))

        threshold = scored[idx][0]
        if ent_value < threshold:
            self.stats.promote_fail_low_score += 1
            return False

        free_slots = max(self.cache_size * self.watermark - self.npu_num_free_slots,
                         ent.num_embeds - self.npu_num_free_slots)

        i = 0
        while free_slots > 0:
            min_hash = scored[i][1]
            ent = self.npu_freeable.pop(min_hash)
            self.evict_from_npu(ent)
            i += 1
            free_slots -= ent.num_embeds

        return True

    def check_and_update_cache(self, request: Request, input_id: int) -> bool:
        """
        Check whether the multimodal embedding corresponding to the current input
        is already cached. If so, update reference tracking, access statistics,
        and hotness information.

        Returns:
            bool:
                True  indicates a cache hit and no need to recompute the encoder output
                False indicates a cache miss and the encoder must be recomputed
        """
        mm_hash = request.mm_features[input_id].identifier

        # Not cached at all
        if mm_hash not in self.cached:
            self.stats.total_requests += 1
            self.on_request()
            self.stats.cache_misses += 1
            return False


        if not self.cached[mm_hash]:
            if mm_hash in self.cpu_freeable:
                ent = self.cpu_freeable.pop(mm_hash)
                self.cpu_num_freeable_slots -= ent.num_embeds
            if mm_hash in self.npu_freeable:
                ent = self.npu_freeable.pop(mm_hash)
                self.npu_num_freeable_slots -= ent.num_embeds

        if request.request_id not in self.cached[mm_hash]:
            self.cached[mm_hash].add(request.request_id)
            self.stats.total_requests += 1
            self.stats.cache_hits += 1
            ent = None
            if mm_hash in self.npu_cache:
                ent = self.npu_cache[mm_hash]
                self.stats.npu_hits += 1
            else:
                self.stats.cpu_hits += 1

                if self.should_promote(mm_hash):
                    # Promote
                    ent = self.cpu_cache[mm_hash]
                    self.npu_cache[mm_hash] = ent
                    self.npu_num_free_slots -= ent.num_embeds
                    self.npu_num_freeable_slots -= ent.num_embeds
                    self.promoting.append(mm_hash)

                    self.stats.promote_success += 1
                else:
                    self.cpu_get_encoder_mm_hashes.append(mm_hash)
                    ent = self.cpu_cache[mm_hash]

            self.on_request()
            ent.freq += 1
            ent.clock = self.max_clock

        return True

    def on_request(self):
        self.req_cnt += 1
        if self.req_cnt % self.clock_decay_every == 0:
            for ent in self.npu_cache.values():
                ent.clock = max(0, ent.clock - 1)

        if self.req_cnt % 1 == 0:
            self.emb_log_stats()

        if self.req_cnt % 1000 == 0:
            self._check_invariant()

    def can_allocate(
        self,
        request: Request,
        input_id: int,
        encoder_compute_budget: int,
        num_embeds_to_schedule: int,
    ) -> bool:
        """
        Determine whether CPU cache space can be allocated for the current input.

        Conditions:
        1. The encoder compute cost of the current input must not exceed the budget of this round
        2. The CPU cache must have enough available or reclaimable space
        3. If free space is insufficient, try evicting entries from CPU freeable

        Returns:
            bool: Whether allocation can be completed
        """

        num_embeds = request.get_num_encoder_embeds(input_id)

        # Not enough compute budget
        if num_embeds > encoder_compute_budget:
            return False

        num_embeds += num_embeds_to_schedule

        if num_embeds > self.cpu_num_freeable_slots:
            return False

        while num_embeds > self.cpu_num_free_slots:
            mm_hash, ent = self.cpu_freeable.popitem(last=False)
            del self.cached[mm_hash]
            del self.cpu_cache[mm_hash]
            self.freed.append(mm_hash)
            self.cpu_num_free_slots += ent.num_embeds
            self.stats.cpu_evict_due_to_alloc += 1

        return True

    def cal_theory_cost_storage_cost(self, seq_len: int) -> float:
        """
        Compute the theoretical recomputation cost of an encoder output.

        The return value represents:
            A rough estimate of the time required to recompute the embedding
            (derived from FLOPs / hardware_flops)

        Notes:
        - The input parameter uses seq_len as an approximation of embedding size
        - The current formula is a rough theoretical estimate based on the vision encoder
        - b*s[(4h+5a)s +(14h+8h**2 +6h*ffn)]
        """

        cost = 32 * (self.alpha * seq_len + self.beta)
        return cost / self.hardware_flops

    def allocate(self, request: Request, input_id: int) -> None:
        """
        Allocate a CPU cache entry for the current input.

        Notes:
        - Newly computed encoder embeddings are placed into the CPU cache by default
        - This only updates the manager's metadata and does not involve actual tensor storage
        """

        mm_hash = request.mm_features[input_id].identifier
        request_id = request.request_id
        if mm_hash not in self.cached:
            self.cached[mm_hash] = set()

        num_encoder_embeds = request.get_num_encoder_embeds(input_id)
        cache_entry = CacheEntry(
                mm_hash=mm_hash,
                freq=1,
                clock=self.max_clock,
                num_embeds=num_encoder_embeds,
                cal_cost=self.cal_theory_cost_storage_cost(num_encoder_embeds),
            )

        assert self.cpu_num_free_slots >= num_encoder_embeds
        assert self.cpu_num_freeable_slots >= num_encoder_embeds

        self.cpu_num_free_slots -= num_encoder_embeds
        self.cpu_num_freeable_slots -= num_encoder_embeds

        assert mm_hash not in self.cpu_cache, f"mm_hash={mm_hash}"
        self.cpu_cache[mm_hash] = cache_entry

        self.cached[mm_hash].add(request_id)

    def free_encoder_input(self, request: Request, input_id: int) -> None:
        req_id = request.request_id
        mm_hash = request.mm_features[input_id].identifier
        # The mm_hash not in cache or the req_id set is empty
        if not self.cached.get(mm_hash, None):
            return
        self.cached[mm_hash].discard(req_id)
        if self.cached[mm_hash]:
            return
        num_encoder_embeds = request.get_num_encoder_embeds(input_id)
        if mm_hash in self.cpu_cache:
            self.cpu_freeable[mm_hash] = self.cpu_cache[mm_hash]
            self.cpu_num_freeable_slots += num_encoder_embeds
        if mm_hash in self.npu_cache:
            self.npu_freeable[mm_hash] = self.npu_cache[mm_hash]
            self.npu_num_freeable_slots += num_encoder_embeds

    def get_promoting_mm_hashes(self) -> list[str]:
        promoting = self.promoting
        self.promoting = []
        return promoting

    def get_cpu_get_encoder_mm_hashes(self) -> list[str]:
        cpu_get_encoder_mm_hashes = self.cpu_get_encoder_mm_hashes
        self.cpu_get_encoder_mm_hashes = []
        return cpu_get_encoder_mm_hashes

    def _check_invariant(self):
        """
        Validate internal state.

        Main checks:
        1. Occupied cache slots + free slots = total capacity
        2. Free slots + slots occupied by freeable entries = freeable_slots
        3. Entries in freeable must not be referenced by any request
        """

        # ---------- CPU ----------
        cpu_sum = sum(ent.num_embeds for ent in self.cpu_cache.values())
        assert (cpu_sum + self.cpu_num_free_slots == self.cpu_cache_size), (
            f"cpu_sum + cpu_num_free_slots != cpu_cache_size, "
            f"cpu_sum={cpu_sum}, "
            f"cpu_num_free_slots={self.cpu_num_free_slots}, "
            f"cpu_cache_size={self.cpu_cache_size}"
        )

        cpu_freeable_sum = sum(ent.num_embeds for ent in self.cpu_freeable.values())
        assert (
            self.cpu_num_freeable_slots
            == self.cpu_num_free_slots + cpu_freeable_sum
        ), (
            f"CPU invariant broken: "
            f"freeable={self.cpu_num_freeable_slots}, "
            f"free={self.cpu_num_free_slots}, "
            f"freeable_sum={cpu_freeable_sum}"
        )

        for mm_hash in self.cpu_freeable:
            assert not self.cached.get(mm_hash), (
                f"CPU freeable entry {mm_hash} still referenced: "
                f"{self.cached.get(mm_hash)}"
            )

        # ---------- NPU ----------
        npu_sum = sum(ent.num_embeds for ent in self.npu_cache.values())
        assert (npu_sum + self.npu_num_free_slots == self.cache_size), (
            f"npu_sum + npu_num_free_slots != cache_size, "
            f"npu_sum={npu_sum}, "
            f"npu_num_free_slots={self.npu_num_free_slots}, "
            f"cache_size={self.cache_size}"
        )
        npu_freeable_sum = sum(ent.num_embeds for ent in self.npu_freeable.values())
        assert (
            self.npu_num_freeable_slots
            == self.npu_num_free_slots + npu_freeable_sum
        ), (
            f"NPU invariant broken: "
            f"freeable={self.npu_num_freeable_slots}, "
            f"free={self.npu_num_free_slots}, "
            f"freeable_sum={npu_freeable_sum}"
        )

        for mm_hash in self.npu_freeable:
            assert not self.cached.get(mm_hash), (
                f"NPU freeable entry {mm_hash} still referenced: "
                f"{self.cached.get(mm_hash)}"
            )

    def emb_log_stats(self):
        s = self.stats
        assert s.total_requests == self.req_cnt, f"total_requests={s.total_requests}, req_cnt={self.req_cnt}"

        hit_rate = s.cache_hits * 100 / max(1, s.total_requests)
        npu_hit_rate = s.npu_hits * 100 / max(1, s.total_requests)
        cpu_hit_rate = s.cpu_hits * 100 / max(1, s.total_requests)

        cpu_entries = len(self.cpu_cache)
        npu_entries = len(self.npu_cache)
        cpu_freeable_entries = len(self.cpu_freeable)
        npu_freeable_entries = len(self.npu_freeable)

        logger.info(
            "[EmbCacheStats] "
            "req=%d | hit=%d npu_hit=%d cpu_hit=%d | "
            "hit_rate=%.3f%% npu_hit_rate=%.3f%% cpu_hit_rate=%.3f%% | "
            "promote=%d/%d | "
            "evict(cpu=%d npu2cpu=%d due2alloc=%d freed=%d) | "
            "entries(cpu=%d freeable=%d | npu=%d freeable=%d) | "
            "slots(cpu=%d/%d npu=%d/%d)",
            s.total_requests,
            s.cache_hits,
            s.npu_hits,
            s.cpu_hits,
            hit_rate,
            npu_hit_rate,
            cpu_hit_rate,
            s.promote_success,
            s.promote_attempts,
            s.evict_cpu,
            s.evict_npu_to_cpu,
            s.cpu_evict_due_to_alloc,
            s.freed_entries,
            cpu_entries,
            cpu_freeable_entries,
            npu_entries,
            npu_freeable_entries,
            self.cpu_num_free_slots,
            self.cpu_num_freeable_slots,
            self.npu_num_free_slots,
            self.npu_num_freeable_slots,
        )


@dataclass
class EmbCacheStats:
    # ---- access ----
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cpu_hits: int = 0
    npu_hits: int = 0

    # ---- promote ----
    promote_attempts: int = 0
    promote_success: int = 0
    promote_fail_no_space: int = 0
    promote_fail_low_score: int = 0

    # ---- eviction ----
    evict_npu: int = 0
    evict_cpu: int = 0
    evict_npu_to_cpu: int = 0
    cpu_evict_due_to_alloc: int = 0
    freed_entries: int = 0
    npu_freed_entries: int = 0