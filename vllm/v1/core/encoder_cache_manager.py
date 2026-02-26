# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.multimodal import MultiModalRegistry
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.config import ModelConfig, SchedulerConfig

logger = init_logger(__name__)


@dataclass
class EncoderCacheStats:
    """Statistics for encoder cache computation time tracking.

    Tracks per-entry computation times and aggregate cache hit/miss
    statistics to inform cache replacement algorithm design.
    """
    # Total number of cache hits (across all check_and_update_cache calls)
    num_cache_hits: int = 0
    # Total number of cache misses
    num_cache_misses: int = 0
    # Total computation time saved by cache hits (seconds)
    total_time_saved: float = 0.0
    # Total computation time spent on cache misses (seconds)
    total_compute_time: float = 0.0
    # Total computation time lost due to evictions of entries that were
    # later requested again (seconds)
    total_time_lost_to_eviction: float = 0.0
    # Number of evictions that resulted in a later re-computation
    num_recomputed_after_eviction: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.num_cache_hits + self.num_cache_misses
        return self.num_cache_hits / total if total > 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"EncoderCacheStats("
            f"hits={self.num_cache_hits}, "
            f"misses={self.num_cache_misses}, "
            f"hit_rate={self.hit_rate:.2%}, "
            f"time_saved={self.total_time_saved:.4f}s, "
            f"compute_time={self.total_compute_time:.4f}s, "
            f"time_lost_to_eviction={self.total_time_lost_to_eviction:.4f}s, "
            f"recomputed_after_eviction={self.num_recomputed_after_eviction})"
        )


@dataclass
class EncoderCacheEntryMeta:
    """Metadata for a single encoder cache entry, used for replacement
    algorithm design.

    Attributes:
        compute_time: Wall-clock time (seconds) to compute the encoder output
            (c_i in the Lagrangian algorithm).
        num_tokens: Number of encoder tokens for this entry (m_i in the
            Lagrangian algorithm).
        num_hits: Number of times this entry has been used as a cache hit.
        last_access_time: Monotonic timestamp of the last access.
        first_compute_time_stamp: Monotonic timestamp of the first computation.
        advantage: Accumulated eviction advantage (A_i in the Lagrangian
            algorithm). Initialized to -c_i on arrival. When A_i > 0 the
            algorithm recommends evicting this entry.
        cost_density: Computed cost per token (compute_time / num_tokens),
            useful for cost-aware replacement algorithms.
    """
    compute_time: float = 0.0
    num_tokens: int = 0
    num_hits: int = 0
    last_access_time: float = field(default_factory=time.monotonic)
    first_compute_time_stamp: float = field(default_factory=time.monotonic)
    advantage: float = 0.0

    @property
    def cost_density(self) -> float:
        """Cost per token — higher means more expensive to recompute."""
        return self.compute_time / self.num_tokens if self.num_tokens > 0 \
            else 0.0

    @property
    def is_evictable(self) -> bool:
        """Whether the Lagrangian algorithm recommends evicting this entry
        (A_i > 0)."""
        return self.advantage > 0


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

    Eviction uses a **Lagrangian online algorithm** that balances
    recomputation cost against memory pressure:

    - Each entry *i* carries an advantage score ``A_i``, initialised to
      ``-c_i`` (negative compute cost) on arrival.
    - At every scheduling step, for non-arriving entries:
      ``A_i += m_i * lambda``, where ``m_i`` is the memory footprint
      (tokens) and ``lambda`` is a dual variable tracking capacity pressure.
    - ``lambda`` is updated via gradient ascent:
      ``lambda <- lambda + eta * (U_t - B)``, where ``U_t`` is current
      usage and ``B`` is the capacity target.
    - When eviction is needed, the entry with the **highest** ``A_i``
      among unreferenced entries is evicted first.  Expensive-to-recompute
      entries (large ``c_i``) naturally survive longer; large-memory
      entries (large ``m_i``) accumulate advantage faster and are evicted
      sooner.

    Args:
        cache_size: Limit the size of the cache, measured by the number of
                    tokens from the input sequence.
        eta: Learning rate for the dual variable lambda update.
        capacity_target_ratio: Fraction of ``cache_size`` used as the
            soft capacity target ``B``.  E.g. 0.9 means we target 90 %
            utilisation; the remaining 10 % acts as a buffer before
            lambda starts rising.

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

    def __init__(
        self,
        cache_size: int,
        eta: float = 0.001,
        capacity_target_ratio: float = 0.9,
    ):
        self.cache_size = cache_size
        self.num_free_slots = cache_size
        self.num_freeable_slots = cache_size

        # mm_hash of mm_data => ids of requests that reference the mm_data
        self.cached: dict[str, set[str]] = {}

        # mm_hash of mm_data => num_encoder_tokens of the mm_data
        self.freeable: OrderedDict[str, int] = OrderedDict()
        self.freed: list[str] = []

        # --- Computation time tracking ---
        # mm_hash => metadata for the cached encoder output
        self.entry_meta: dict[str, EncoderCacheEntryMeta] = {}
        # mm_hashes that were evicted but had compute time recorded,
        # kept to detect re-computation after eviction
        self.evicted_meta: dict[str, EncoderCacheEntryMeta] = {}
        # Aggregate statistics
        self.stats = EncoderCacheStats()

        # --- Lagrangian replacement algorithm state ---
        # Dual variable (lambda_t) tracking cache capacity pressure
        self.lambda_val: float = 0.0
        # Learning rate for lambda update
        self.eta: float = eta
        # Soft capacity target B (e.g. 90% of cache_size)
        self.capacity_target: float = capacity_target_ratio * cache_size
        # mm_hashes whose compute time was recorded this step (e_{t,i}=1),
        # so that step() skips their advantage update.
        self._arrived_this_step: set[str] = set()

    def check_and_update_cache(self, request: Request, input_id: int) -> bool:
        """Check if encoder output for a specific multimodal input is cached.

        If the encoder output is cached, update `cached` to add the request id
        to the set of request ids that reference the cached encoder output.
        If the encoder output was previously not referenced by any request,
        update `freeable` and `num_freeable_slots` accordingly.

        Also updates computation time statistics: on a hit, records the
        saved computation time; on a miss, increments the miss counter.

        Args:
            request: The request containing the multimodal input
            input_id: Index of the multimodal input within the request

        Returns:
            True if the encoder output for this input is already cached
        """
        mm_hash = request.mm_features[input_id].identifier
        # Not cached at all
        if mm_hash not in self.cached:
            self.stats.num_cache_misses += 1
            return False

        # Cache hit — update stats
        self.stats.num_cache_hits += 1
        meta = self.entry_meta.get(mm_hash)
        if meta is not None:
            meta.num_hits += 1
            meta.last_access_time = time.monotonic()
            self.stats.total_time_saved += meta.compute_time

        # Cached but currently not referenced by any request
        if not self.cached[mm_hash]:
            num_tokens = self.freeable.pop(mm_hash)
            self.num_freeable_slots -= num_tokens

        self.cached[mm_hash].add(request.request_id)
        return True

    def can_allocate(
        self,
        request: Request,
        input_id: int,
        encoder_compute_budget: int,
        num_tokens_to_schedule: int,
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

        # Not enough free slots but enough reclaimable slots.
        # Use the Lagrangian algorithm to select which entries to evict:
        # the entry with the highest advantage A_i (most recommended for
        # eviction) is evicted first.
        # NOTE: Eviction takes place here, but physical memory is not freed
        # until model runner is notified by the scheduler output.
        while num_tokens > self.num_free_slots:
            mm_hash = self._select_eviction_candidate()
            num_free_token = self.freeable.pop(mm_hash)
            del self.cached[mm_hash]
            self.freed.append(mm_hash)
            self.num_free_slots += num_free_token
            # Preserve metadata of evicted entries so we can detect and
            # measure the cost of re-computation after eviction.
            if mm_hash in self.entry_meta:
                self.evicted_meta[mm_hash] = self.entry_meta.pop(mm_hash)
        return True

    def allocate(self, request: Request, input_id: int) -> None:
        """Allocate cache space for a multimodal input's encoder output.

        This reserves cache space for storing the encoder output of the
        specified multimodal input. The actual encoder output storage happens in
        the model runner; this method updates the manager's bookkeeping.

        Also initializes an :class:`EncoderCacheEntryMeta` for the entry so
        that computation time can later be recorded via
        :meth:`record_compute_time`.

        Note:
            This method assumes can_allocate() returned True for the same input.
        """

        mm_hash = request.mm_features[input_id].identifier
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

        # Initialize entry metadata (compute_time will be set later via
        # record_compute_time once the worker reports it back).
        if mm_hash not in self.entry_meta:
            now = time.monotonic()
            self.entry_meta[mm_hash] = EncoderCacheEntryMeta(
                num_tokens=num_encoder_tokens,
                last_access_time=now,
                first_compute_time_stamp=now,
            )

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
        increased by the number of encoder tokens for that input.

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

    # ---------- Computation time tracking API ----------

    def record_compute_time(
        self,
        mm_hash: str,
        compute_time: float,
    ) -> None:
        """Record the computation time for an encoder output.

        Called by the scheduler after the model runner reports back the
        actual wall-clock time spent computing each encoder output.

        This corresponds to the **arrival event** (e_{t,i} = 1) in the
        Lagrangian algorithm: the advantage is reset to ``-c_i`` and the
        entry is marked as arrived so that :meth:`step` skips its
        advantage accumulation for this round.

        If the ``mm_hash`` was previously evicted and is now being
        re-computed, the eviction cost is recorded in
        :pyattr:`stats.total_time_lost_to_eviction`.

        Args:
            mm_hash: The multimodal hash identifying the encoder output.
            compute_time: Wall-clock time in seconds to compute the output.
        """
        self.stats.total_compute_time += compute_time

        # Check if this is a re-computation after eviction
        if mm_hash in self.evicted_meta:
            prev_meta = self.evicted_meta.pop(mm_hash)
            self.stats.total_time_lost_to_eviction += compute_time
            self.stats.num_recomputed_after_eviction += 1
            logger.debug(
                "Re-computing previously evicted encoder output %s "
                "(prev_time=%.4fs, new_time=%.4fs, prev_hits=%d)",
                mm_hash,
                prev_meta.compute_time,
                compute_time,
                prev_meta.num_hits,
            )

        meta = self.entry_meta.get(mm_hash)
        if meta is not None:
            meta.compute_time = compute_time
            # Arrival event: A_i = -c_i, x_{t,i} = 0
            meta.advantage = -compute_time
            self._arrived_this_step.add(mm_hash)

    def get_entry_meta(self, mm_hash: str) -> EncoderCacheEntryMeta | None:
        """Get the metadata for a cached encoder output entry.

        Args:
            mm_hash: The multimodal hash identifying the encoder output.

        Returns:
            The entry metadata, or None if the entry is not tracked.
        """
        return self.entry_meta.get(mm_hash)

    def get_stats(self) -> EncoderCacheStats:
        """Get aggregate cache statistics.

        Returns:
            A snapshot of the current cache statistics including hit rates,
            computation times saved, and eviction costs.
        """
        return self.stats

    def get_all_entry_meta(self) -> dict[str, EncoderCacheEntryMeta]:
        """Get metadata for all currently tracked entries.

        Useful for analyzing cache behavior and testing replacement
        algorithm candidates offline.

        Returns:
            Dict mapping mm_hash to its :class:`EncoderCacheEntryMeta`.
        """
        return self.entry_meta

    def log_stats(self) -> None:
        """Log current cache statistics at INFO level."""
        logger.info("Encoder cache stats: %s", self.stats)

    # ---------- Lagrangian replacement algorithm ----------

    def step(self) -> None:
        """Execute one step of the Lagrangian cache replacement algorithm.

        Must be called **once per scheduling round**, after all
        :meth:`record_compute_time` calls for this round have been made.

        For every cached entry whose encoder output was **not** newly
        computed this round (``e_{t,i} = 0``), the advantage is updated::

            A_i += m_i * lambda_t

        Then the dual variable is updated via gradient ascent on the
        capacity constraint::

            lambda_{t+1} = lambda_t + eta * (U_t - B)

        where ``U_t`` is the current cache usage in tokens and ``B`` is
        the soft capacity target.
        """
        # Update A_i for non-arriving entries
        lam = self.lambda_val
        for mm_hash, meta in self.entry_meta.items():
            if mm_hash in self._arrived_this_step:
                # e_{t,i} = 1: advantage already set to -c_i in
                # record_compute_time(); skip accumulation.
                continue
            # e_{t,i} = 0: A_i += m_i * lambda_t
            meta.advantage += meta.num_tokens * lam

        # Compute current usage U_t (tokens occupied in cache)
        usage = self.cache_size - self.num_free_slots

        # Update lambda: lambda_{t+1} = lambda_t + eta * (U_t - B)
        self.lambda_val = lam + self.eta * (usage - self.capacity_target)

        # Clear arrival set for the next round
        self._arrived_this_step.clear()

    def _select_eviction_candidate(self) -> str:
        """Select the best eviction candidate from *freeable* entries.

        Returns the ``mm_hash`` with the **highest** advantage score
        ``A_i``.  A higher ``A_i`` means the algorithm more strongly
        recommends eviction (the entry is cheap to recompute and/or
        occupies a lot of memory relative to its value).

        Falls back to the oldest entry (FIFO) if no metadata is
        available.

        Returns:
            The ``mm_hash`` of the entry to evict.
        """
        best_hash: str | None = None
        best_advantage = float('-inf')

        for mm_hash in self.freeable:
            meta = self.entry_meta.get(mm_hash)
            advantage = meta.advantage if meta is not None else 0.0
            if advantage > best_advantage:
                best_advantage = advantage
                best_hash = mm_hash

        # Fallback: should not happen because callers ensure freeable is
        # non-empty, but guard against it anyway.
        if best_hash is None:
            best_hash = next(iter(self.freeable))

        return best_hash


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
        max_tokens_by_modality = mm_registry.get_max_tokens_per_item_by_modality(
            model_config
        )

        return compute_mm_encoder_budget(
            scheduler_config,
            max_tokens_by_modality,
        )

    return compute_text_encoder_budget(scheduler_config)


def compute_text_encoder_budget(scheduler_config: "SchedulerConfig") -> tuple[int, int]:
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
            "not be initialized."
        )
        return 0, 0

    max_tokens_per_mm_item = max(max_tokens_by_modality.values())

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
