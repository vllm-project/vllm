# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from itertools import islice
from typing import Any, NamedTuple

from vllm.config import VllmConfig
from vllm.distributed.kv_events import KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.utils import yield_req_data
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
    ReqId,
    TransferJob,
    TransferJobStatus,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.events import (
    OffloadingEventGroupSpec,
    OffloadingEventsTracker,
    get_offloading_event_group_spec,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
    _ConnectorMetricName,
    _TransferMetricName,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.store_triggers.base import (  # noqa: E501
    StoreJobsOutput,
    StoreTriggerHandler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.store_triggers.on_compute import (  # noqa: E501
    OnComputeStoreHandler,
)
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv, round_down
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
)
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    Locality,
    LookupResult,
    Medium,
    OffloadingManager,
    OffloadingSpec,
    OffloadKey,
    ReqContext,
    RequestOffloadingContext,
    ScheduleEndContext,
    StoreMode,
    StoreTrigger,
    TierFilter,
    TierMatcher,
    make_offload_key,
)
from vllm.v1.metrics.cache_hit_source import CacheHitSource
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

logger = init_logger(__name__)

KV_LOAD_TIERS_KEY = "kv_load_tiers"
MATCHER_MEDIUM_KEY = "medium"
MATCHER_LOCALITY_KEY = "locality"


class GroupOffloadConfig(NamedTuple):
    group_idx: int
    tokens_per_block: int
    tokens_per_chunk: int
    hashes_per_chunk: int
    # KV cache spec metadata propagated onto emitted BlockStored events so
    # KV-aware consumers can classify and filter the group.
    kv_event_group_spec: OffloadingEventGroupSpec
    # None below means full attention
    sliding_window_size_in_chunks: int | None
    kv_cache_spec: KVCacheSpec
    # Cached manager class for this group's KV cache spec, resolved at init time
    manager_cls: type[SingleTypeKVCacheManager]
    # Partial-tail data for this group comes from the scheduler's CoW hand-off
    # rather than the request block table.
    requires_cow_source: bool = False
    # True for EAGLE/MTP draft-model attention groups. The trailing chunk
    # of these groups is volatile and lacks a stable hash, so it must
    # be excluded from store and load scheduling.
    is_eagle_group: bool = False

    def load_window_size_in_chunks(self, num_tokens: int) -> int | None:
        window = self.sliding_window_size_in_chunks
        if isinstance(self.kv_cache_spec, SlidingWindowSpec):
            assert window is not None
            # Lookup rounds the right edge up, but allocation retains the
            # window ending at the actual hit boundary, possibly one chunk left.
            right_padding = -num_tokens % self.tokens_per_chunk
            window = max(
                window,
                cdiv(
                    self.kv_cache_spec.sliding_window
                    - 1
                    + self.kv_cache_spec.extra_retained_tokens
                    + right_padding,
                    self.tokens_per_chunk,
                ),
            )
        return window


def get_sliding_window_size_in_chunks(
    kv_cache_spec: KVCacheSpec, tokens_per_chunk: int
) -> int | None:
    if isinstance(kv_cache_spec, SlidingWindowSpec):
        assert kv_cache_spec.sliding_window > 0
        return max(
            cdiv(kv_cache_spec.sliding_window, tokens_per_chunk),
            cdiv(
                kv_cache_spec.sliding_window - 1 + kv_cache_spec.extra_retained_tokens,
                tokens_per_chunk,
            ),
        )

    if isinstance(kv_cache_spec, ChunkedLocalAttentionSpec):
        # Attention never reaches back past one chunk
        assert kv_cache_spec.attention_chunk_size > 0
        return cdiv(kv_cache_spec.attention_chunk_size, tokens_per_chunk)

    if isinstance(kv_cache_spec, MambaSpec):
        # Mamba depends on a single state
        return 1

    assert isinstance(kv_cache_spec, FullAttentionSpec)
    return None


def resolve_mamba_align_size(
    spec: "OffloadingSpec", kv_cache_config: KVCacheConfig
) -> int | None:
    """Scan all KV cache groups in *spec* and return the single mamba alignment
    size, or None if no group requires mamba alignment.

    For MambaSpec groups in "align" cache mode the hit window must be
    rounded down to a multiple of the offloaded chunk size. Asserts that all
    such groups agree on the same value.
    """
    mamba_align_size: int | None = None
    for group in spec.config.groups:
        tokens_per_block = group.tokens_per_block
        kv_spec = kv_cache_config.kv_cache_groups[group.group_id].kv_cache_spec
        if isinstance(kv_spec, MambaSpec) and kv_spec.mamba_cache_mode == "align":
            tokens_per_chunk = tokens_per_block * spec.blocks_per_chunk
            assert mamba_align_size is None or mamba_align_size == tokens_per_chunk
            mamba_align_size = tokens_per_chunk
    return mamba_align_size


class SchedulerOffloadConfig(NamedTuple):
    kv_group_configs: tuple[GroupOffloadConfig, ...]
    blocks_per_chunk: int
    tokens_per_hash: int
    num_workers: int
    offload_prompt_only: bool
    supports_partial_tail: bool
    alignment_tokens: int | None = None
    retention_interval: int | None = None
    dcp_world_size: int = 1
    store_mode: StoreMode = StoreMode.EAGER

    @property
    def partial_tail_block_size(self) -> int:
        if not self.supports_partial_tail:
            return 0
        return self.kv_group_configs[0].tokens_per_block

    @classmethod
    def from_spec(
        cls,
        spec: OffloadingSpec,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
    ) -> "SchedulerOffloadConfig":
        # Determine the alignment token count from the full-attention group(s).
        # This is the tokens_per_chunk of the full-attention group; load
        # hits are always aligned to this boundary, so SWA blocks earlier in
        # each segment can never serve a load hit. Relevant for hybrid
        # architectures like DeepSeek V4 (MLA + SWA groups).
        full_attn_tokens_per_chunk: set[int] = set()
        for group in spec.config.groups:
            tokens_per_block = group.tokens_per_block
            kv_spec = kv_cache_config.kv_cache_groups[group.group_id].kv_cache_spec
            sw = get_sliding_window_size_in_chunks(
                kv_spec, tokens_per_block * spec.blocks_per_chunk
            )
            if sw is None:
                full_attn_tokens_per_chunk.add(tokens_per_block * spec.blocks_per_chunk)

        # Only apply the optimization if there's a single consistent
        # full-attention alignment size.  For pure SWA/Mamba models (no
        # full-attention groups), fall back to the largest SWA/Mamba
        # tokens_per_chunk so retention_interval has a boundary
        # granularity to work with.
        alignment_tokens: int | None = None
        if len(full_attn_tokens_per_chunk) == 1:
            alignment_tokens = full_attn_tokens_per_chunk.pop()
        elif not full_attn_tokens_per_chunk:
            all_tokens_per_chunk = [
                tpb * spec.blocks_per_chunk for tpb in spec.tokens_per_block
            ]
            if all_tokens_per_chunk:
                alignment_tokens = max(all_tokens_per_chunk)

        retention_interval = vllm_config.cache_config.prefix_cache_retention_interval

        eagle_groups = {
            group.group_id
            for group in spec.config.groups
            if kv_cache_config.kv_cache_groups[group.group_id].is_eagle_group
        }

        use_eagle_block_drop = (
            vllm_config.speculative_config is not None
            and vllm_config.speculative_config.use_eagle_block_drop()
        )
        if use_eagle_block_drop and not eagle_groups:
            # No group is annotated as holding drafter layers. This happens
            # when the drafter owns no KV cache (e.g. KV-sharing drafters) or
            # the model runner does not tag draft layers' KV cache specs.
            # Treating every group as a drafter group here makes
            # the volatile-tail pop apply to ALL groups, which can zero the
            # whole request's offload hit whenever any group's servable
            # window is a single chunk (issue #52735). Drafter KV served one
            # chunk stale can only lower speculative acceptance rates — the
            # target model verifies every draft — so fail toward serving.
            logger.info_once(
                "KV offloading: speculative decoding is enabled but no "
                "KV-cache group is annotated as a drafter group; treating "
                "all groups as non-draft for offloading."
            )

        if eagle_groups:
            logger.info(
                "KV offloading: EAGLE/MTP draft attention groups %s "
                "detected. The trailing chunk of these groups will be "
                "excluded from offloading due to volatility.",
                sorted(eagle_groups),
            )

        kv_group_configs_list: list[GroupOffloadConfig] = []
        for group in spec.config.groups:
            group_id = group.group_id
            tokens_per_block = group.tokens_per_block
            kv_cache_group = kv_cache_config.kv_cache_groups[group_id]
            kv_spec = kv_cache_group.kv_cache_spec
            sw = get_sliding_window_size_in_chunks(
                kv_spec, tokens_per_block * spec.blocks_per_chunk
            )
            manager_cls = KVCacheSpecRegistry.get_manager_class(kv_spec)
            assert manager_cls is not None, (
                f"No manager found for KV cache spec {type(kv_spec).__name__}"
            )
            kv_group_configs_list.append(
                GroupOffloadConfig(
                    group_idx=group_id,
                    tokens_per_block=tokens_per_block,
                    tokens_per_chunk=tokens_per_block * spec.blocks_per_chunk,
                    hashes_per_chunk=(
                        (tokens_per_block * spec.blocks_per_chunk)
                        // spec.tokens_per_hash
                    ),
                    sliding_window_size_in_chunks=sw,
                    kv_cache_spec=kv_spec,
                    manager_cls=manager_cls,
                    kv_event_group_spec=get_offloading_event_group_spec(kv_cache_group),
                    is_eagle_group=group_id in eagle_groups,
                    requires_cow_source=(
                        isinstance(kv_spec, MambaSpec)
                        and kv_spec.mamba_cache_mode == "align"
                    ),
                )
            )
        kv_group_configs = tuple(kv_group_configs_list)
        group_block_sizes = {config.tokens_per_block for config in kv_group_configs}
        has_partial_recurrent_group = any(
            config.requires_cow_source
            and config.tokens_per_block > spec.tokens_per_hash
            for config in kv_group_configs
        )
        # Partial tails currently require one physical block per offload chunk
        # and uniform, non-windowed groups so one boundary identifies every
        # group's source. EAGLE and DCP need additional hand-off semantics.
        supports_partial_tail = (
            spec.blocks_per_chunk == 1
            and len(group_block_sizes) == 1
            and has_partial_recurrent_group
            and all(
                config.sliding_window_size_in_chunks is None
                or config.requires_cow_source
                for config in kv_group_configs
            )
            and not any(config.is_eagle_group for config in kv_group_configs)
            and vllm_config.parallel_config.decode_context_parallel_size == 1
        )

        if retention_interval is not None:
            if retention_interval < 0:
                raise ValueError(
                    f"prefix_cache_retention_interval "
                    f"({retention_interval}) must be non-negative."
                )
            for config in kv_group_configs:
                if (
                    config.sliding_window_size_in_chunks is not None
                    and retention_interval % config.tokens_per_chunk != 0
                ):
                    raise ValueError(
                        f"prefix_cache_retention_interval "
                        f"({retention_interval}) must be a multiple of "
                        f"tokens_per_chunk ({config.tokens_per_chunk})."
                    )

        return cls(
            num_workers=vllm_config.parallel_config.world_size,
            kv_group_configs=kv_group_configs,
            blocks_per_chunk=spec.blocks_per_chunk,
            tokens_per_hash=spec.tokens_per_hash,
            offload_prompt_only=spec.offload_prompt_only,
            supports_partial_tail=supports_partial_tail,
            alignment_tokens=alignment_tokens,
            retention_interval=retention_interval,
            dcp_world_size=vllm_config.parallel_config.decode_context_parallel_size,
            store_mode=spec.store_mode,
        )


@dataclass
class RequestGroupState:
    offload_keys: list[OffloadKey] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)
    # Index of the next chunk to offload.
    next_stored_chunk_idx: int = 0
    # Number of offloaded chunks hit (including GPU prefix cache)
    # when the request first started
    num_hit_chunks: int = 0


@dataclass(slots=True)
class RequestOffloadState:
    config: SchedulerOffloadConfig
    req: Request
    req_context: ReqContext
    offloading_context: RequestOffloadingContext
    group_states: tuple[RequestGroupState, ...] = field(init=False)
    # upper bound on tokens to load for this request; None means no cap
    max_load_tokens: int | None = None
    # upper bound on tokens to offload for this request; None means no cap
    max_offload_tokens: int | None = None
    # number of hits in the GPU cache
    num_locally_computed_tokens: int = 0
    # In-flight job IDs. Per the connector's invariant, at any given time
    # this contains either a single load job, or one or more store jobs.
    transfer_jobs: set[int] = field(default_factory=set)
    # time.monotonic() of this request's first deferred offload lookup;
    # None once consumed (observed) or while no lookup is pending.
    deferred_lookup_start_time: float | None = None
    # Fine-grained token boundary selected beyond the last complete offload
    # chunk. It is consumed when the corresponding load is scheduled.
    partial_tail_boundary: int | None = None
    # Loaded attention keys cover their token positions. Recurrent state
    # supports the whole reused external prefix.
    load_key_ranges: list[tuple[int, int, OffloadKey]] = field(default_factory=list)
    # True once on_request_finished has been signaled to the manager.
    finished_signaled: bool = False

    def __post_init__(self) -> None:
        self.group_states = tuple(
            RequestGroupState() for _ in self.config.kv_group_configs
        )
        params = self.req.kv_transfer_params

        # NOTE: This field is experimental and subject to change in the future.
        raw = params.get("max_load_tokens") if params else None
        if type(raw) is int and raw >= 0:
            self.max_load_tokens = raw
            logger.debug(
                "Request %s: max_load_tokens set to %d",
                self.req.request_id,
                raw,
            )
        elif raw is not None:
            logger.warning(
                "max_load_tokens must be a non-negative int, got %r; ignoring", raw
            )

        # NOTE: This field is experimental and subject to change in the future.
        raw = params.get("max_offload_tokens") if params else None
        if type(raw) is int and raw >= 0:
            self.max_offload_tokens = raw
            logger.debug(
                "Request %s: max_offload_tokens set to %d",
                self.req.request_id,
                raw,
            )
        elif raw is not None:
            logger.warning(
                "max_offload_tokens must be a non-negative int, got %r; ignoring", raw
            )

    def update_offload_keys(self) -> None:
        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            for req_block_hash in islice(
                self.req.block_hashes,
                group_config.hashes_per_chunk * len(group_state.offload_keys)
                + group_config.hashes_per_chunk
                - 1,
                None,
                group_config.hashes_per_chunk,
            ):
                key = make_offload_key(req_block_hash, group_config.group_idx)
                group_state.offload_keys.append(key)
                self.req_context.set_offload_key_position(
                    key,
                    len(group_state.offload_keys) * group_config.tokens_per_chunk,
                )

    def update_block_id_groups(
        self, new_block_id_groups: tuple[list[int], ...] | None
    ) -> None:
        if new_block_id_groups is None:
            return

        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            group_state.block_ids.extend(new_block_id_groups[group_config.group_idx])

    def storable_chunks(
        self,
        group_config: "GroupOffloadConfig",
        group_state: RequestGroupState,
        num_offloadable_tokens: int,
    ) -> int:
        """Number of allocated and keyed leading chunks eligible for store.

        For eagle/MTP groups the volatile trailing chunk of the offloadable
        range is excluded while decoding: the draft-layer KV of the last
        accepted position may be rewritten after spec-token rejection. During
        prefill the trailing chunk is stable (the draft input for a chunk's
        last position is the next prompt token), so it is stored immediately.
        Once the request has finished, no further spec-token rejection can
        rewrite the tail, so the exclusion is lifted and the final chunk
        becomes storable (issue #52735). The exclusion must be applied
        consistently everywhere ``next_stored_chunk_idx`` is derived:
        otherwise the trailing chunk of each step is skipped on collection but
        jumped over by ``next_stored_chunk_idx``, so it is never re-considered
        and a permanent hole breaks prefix-reuse lookup. ``is_finished`` is
        monotonic, so the finish-time calls all see the lifted exclusion.
        """
        num_chunks = num_offloadable_tokens // group_config.tokens_per_chunk
        is_decoding = num_offloadable_tokens > self.req.num_prompt_tokens
        # Finished requests have no pending speculation.
        if group_config.is_eagle_group and is_decoding and not self.req.is_finished():
            num_chunks = max(0, num_chunks - 1)
        num_allocated_chunks = (
            len(group_state.block_ids) // self.config.blocks_per_chunk
        )
        num_keyed_chunks = len(group_state.offload_keys)
        return min(num_chunks, num_allocated_chunks, num_keyed_chunks)

    def num_offloadable_tokens(self, num_computed_tokens: int) -> int:
        num = min(num_computed_tokens, self.req.num_tokens)
        if self.max_offload_tokens is not None:
            num = min(num, self.max_offload_tokens)
        if self.config.offload_prompt_only:
            num = min(num, self.req.num_prompt_tokens)
        return num

    def make_boundary_key(self, group_idx: int, boundary_tokens: int) -> OffloadKey:
        hash_idx = boundary_tokens // self.config.tokens_per_hash - 1
        key = make_offload_key(self.req.block_hashes[hash_idx], group_idx)
        self.req_context.set_offload_key_position(key, boundary_tokens)
        return key

    def update_num_hit_chunks(self, num_cached_tokens: int) -> None:
        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            group_state.num_hit_chunks = (
                num_cached_tokens // group_config.tokens_per_chunk
            )


def _parse_tier_filter(raw: Any) -> TierFilter:
    """Parse raw kv_transfer_params tier matchers into a TierFilter."""
    if not isinstance(raw, list):
        logger.warning(
            "_parse_tier_filter: expected list, got %s; ignoring",
            type(raw).__name__,
        )
        return TierFilter.ALL
    matchers: list[TierMatcher] = []
    for entry in raw:
        if not isinstance(entry, dict):
            logger.warning("_parse_tier_filter: entry is not a dict; skipping")
            continue
        medium: Medium | None = None
        locality: Locality | None = None
        raw_medium = entry.get(MATCHER_MEDIUM_KEY)
        if raw_medium is not None:
            try:
                medium = Medium(raw_medium.upper())
            except (ValueError, AttributeError):
                logger.warning(
                    "_parse_tier_filter: unknown medium %r; skipping entry",
                    raw_medium,
                )
                continue
        raw_locality = entry.get(MATCHER_LOCALITY_KEY)
        if raw_locality is not None:
            try:
                locality = Locality(raw_locality.upper())
            except (ValueError, AttributeError):
                logger.warning(
                    "_parse_tier_filter: unknown locality %r; skipping entry",
                    raw_locality,
                )
                continue
        matchers.append(TierMatcher(medium=medium, locality=locality))
    if not matchers:
        if not raw:  # input was [] — user explicitly wants nothing
            return TierFilter(matchers=())
        # all entries were invalid — fall back to ALL
        return TierFilter.ALL
    return TierFilter(matchers=tuple(matchers))


def _create_req_context(req: Request) -> ReqContext:
    params = req.kv_transfer_params
    load_filter = TierFilter.ALL
    if params:
        raw = params.get(KV_LOAD_TIERS_KEY)
        if raw is not None:
            load_filter = _parse_tier_filter(raw)
    return ReqContext(
        req_id=req.request_id,
        kv_transfer_params=params,
        kv_hints=req.kv_hints,
        load_tier_filter=load_filter,
    )


class OffloadingConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(
        self,
        spec: OffloadingSpec,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
    ):
        self.config = SchedulerOffloadConfig.from_spec(
            spec, vllm_config, kv_cache_config
        )
        self.manager: OffloadingManager = spec.get_manager()
        self._connector_stats = OffloadingConnectorStats()
        # The static config facts ride on the first stats payload of this
        # process. A Prometheus child holds the values it received, so one
        # payload is enough.
        self._info_sent = False

        full_attention_groups: list[int] = []
        sliding_window_groups: list[int] = []
        for config_idx, group_config in enumerate(self.config.kv_group_configs):
            if group_config.sliding_window_size_in_chunks is None:
                full_attention_groups.append(config_idx)
            else:
                sliding_window_groups.append(config_idx)

        # sort sliding window groups by window size in decreasing order
        def _sliding_window_sort_key(i: int) -> int:
            val = self.config.kv_group_configs[i].sliding_window_size_in_chunks
            assert val is not None
            return val

        sliding_window_groups.sort(key=_sliding_window_sort_key, reverse=True)

        # used by _lookup
        self._full_attention_groups: tuple[int, ...] = tuple(full_attention_groups)
        self._sliding_window_groups: tuple[int, ...] = tuple(sliding_window_groups)
        self._lookup_groups = self._full_attention_groups + self._sliding_window_groups
        self._mamba_align_size: int | None = resolve_mamba_align_size(
            spec, kv_cache_config
        )

        self._req_status: dict[ReqId, RequestOffloadState] = {}
        self._current_batch_load_jobs: dict[int, TransferJob] = {}
        self._current_batch_jobs_to_flush: set[int] = set()
        # GPU block IDs allocated in the current engine step
        self._current_batch_allocated_block_ids: set[int] = set()
        # if GPU prefix caching is enabled,
        # Track loaded chunks to avoid redundant loads.
        self._chunks_being_loaded: set[OffloadKey] | None = (
            set() if vllm_config.cache_config.enable_prefix_caching else None
        )

        # Job ID counter shared by loads and stores.
        self._job_counter: int = 0
        # Threshold value for stale jobs. All job ids >= _stale_job_threshold are
        # active jobs.
        self._stale_job_threshold: int = 0
        self._job_statuses: dict[int, TransferJobStatus] = {}

        # block_id -> pending store job_ids. Used to track jobs that needs
        # flushing in case a block is re-allocated by the KV cache manager.
        # Populated only for finished requests (running-request blocks are
        # protected by their ref_cnt) and for sliding window blocks (which can
        # be freed before a request finishes).
        self._block_id_to_pending_jobs: dict[int, set[int]] = {}

        self._events_tracker = OffloadingEventsTracker(spec.kv_events_config)

        self._store_handlers: dict[StoreTrigger, StoreTriggerHandler] = {
            StoreTrigger.ON_COMPUTE: OnComputeStoreHandler(
                self.config, self.manager, self._events_tracker
            )
        }

    # Holds the manager lock for the current schedule(), or None between steps.
    # Read and written only by the scheduler thread, so it needs no
    # synchronization of its own. Declared on the class so an instance built
    # without __init__ still reads None rather than raising.
    _step_region: ExitStack | None = None

    # ------------------------------------------------------------------
    # Manager exclusion
    #
    # The manager is entered from the scheduler thread and, for managers that
    # run one, from a tier-poll thread. Manager methods never take their
    # own lock, so the exclusion boundary lives here. manager.lock may be a
    # no-op for a manager that needs no exclusion; it is entered regardless.
    #
    # schedule() holds one region for the whole step: opened by the first
    # manager-touching hook and closed at the end of build_connector_meta.
    # That span is required, not tidiness -- get_num_new_matched_tokens
    # establishes lookup() HITs that update_state_after_alloc later pins with
    # prepare_load(), and prepare_store() evicts after both. Releasing between
    # them would let another thread evict a block the step already treated as
    # present.
    #
    # Hooks outside schedule() use a short region instead, so the manager is
    # free while the engine runs the model.
    #
    # The step region outlives any single call, so it cannot be a with-block;
    # an ExitStack carries the entered lock from open to close instead.
    # ------------------------------------------------------------------

    def _acquire_step(self) -> None:
        """Open the step-scoped region. Idempotent within a step."""
        if self._step_region is not None:
            return
        region = ExitStack()
        region.enter_context(self.manager.lock)
        self._step_region = region

    def _release_step(self) -> None:
        """Close the step-scoped region, letting other threads in."""
        region = self._step_region
        if region is None:
            return
        self._step_region = None
        region.close()

    @contextmanager
    def _manager_locked(self) -> Iterator[None]:
        """Hold the manager for the duration of one hook outside schedule().

        None of these hooks can legitimately run inside the step-scoped region:
        on_new_request comes from add_request, has_pending_push_work from the
        engine's liveness check before schedule(), the rest from output
        processing or a client RPC. So finding the region open here means a
        previous schedule() raised between _acquire_step() and the release in
        build_connector_meta. Recover instead of deadlocking.
        """
        if self._step_region is not None:
            logger.error(
                "Offloading manager lock was left held by a previous "
                "schedule(); releasing it before continuing."
            )
            self._release_step()
        with self.manager.lock:
            yield

    def _maybe_observe_lookup_async_delay(
        self, req_status: RequestOffloadState
    ) -> None:
        start_time = req_status.deferred_lookup_start_time
        if start_time is None:
            return
        req_status.deferred_lookup_start_time = None
        self._connector_stats.observe_histogram(
            _ConnectorMetricName.LOOKUP_ASYNC_DELAY,
            time.monotonic() - start_time,
        )

    def _generate_job_id(self) -> int:
        job_id = self._job_counter
        self._job_counter += 1
        return job_id

    def _remove_pending_job(self, job_id: int, block_ids: list[int] | None) -> None:
        for bid in block_ids or ():
            pending = self._block_id_to_pending_jobs[bid]
            pending.remove(job_id)
            if not pending:
                del self._block_id_to_pending_jobs[bid]

    def _maximal_prefix_lookup(
        self,
        keys: Iterable[OffloadKey],
        req_context: ReqContext,
        req: Request,
        group_config: GroupOffloadConfig,
        start_chunk_idx: int,
    ) -> int | None:
        """Return the number of consecutive offloaded chunks from the start,
        or None if the backend deferred a lookup."""
        hit_count = 0
        defer_lookup = False
        for local_idx, key in enumerate(keys):
            result = self.manager.lookup(key, req_context)
            match result:
                case LookupResult.HIT:
                    self._events_tracker.record_lookup(
                        req,
                        group_config,
                        start_chunk_idx + local_idx,
                        key,
                    )
                    hit_count += 1
                case LookupResult.HIT_PENDING:
                    defer_lookup = True
                    hit_count += 1
                case LookupResult.RETRY:
                    # Don't break: keep scanning to let manager kick off
                    # async lookups (until a miss is detected).
                    defer_lookup = True
                case LookupResult.MISS:
                    break
        return hit_count if not defer_lookup else None

    def _sliding_window_lookup(
        self,
        keys: Sequence[OffloadKey],
        sliding_window_size: int,
        req_context: ReqContext,
        initial_window_size: int | None = None,
    ) -> int | None:
        """Return the end index (in `keys`) of the last run of
        `sliding_window_size` consecutive hits, scanning from the end.
        The first run may need a larger window for a partial rightmost chunk.
        Returns 0 on miss, None if the backend deferred a lookup."""
        defer_lookup = False
        pending_in_window = False
        consecutive_hits = 0
        required_window = initial_window_size or sliding_window_size
        for idx in range(len(keys) - 1, -1, -1):
            match self.manager.lookup(keys[idx], req_context):
                case LookupResult.HIT:
                    consecutive_hits += 1
                case LookupResult.HIT_PENDING:
                    # Block is in cache, just not readable yet — counts
                    # as hit for the consecutive streak. Don't break:
                    # keep scanning to let manager kick off async lookups.
                    pending_in_window = True
                    consecutive_hits += 1
                case LookupResult.RETRY:
                    # Block location uncertain — does not count as hit.
                    # Don't break: keep scanning to let manager kick off
                    # async lookups.
                    defer_lookup = True
                    consecutive_hits = 0
                    pending_in_window = False
                    required_window = sliding_window_size
                case LookupResult.MISS:
                    consecutive_hits = 0
                    # This gap rules out the incomplete window to its right.
                    pending_in_window = False
                    required_window = sliding_window_size
            if consecutive_hits == required_window:
                return (
                    None if defer_lookup or pending_in_window else idx + required_window
                )
        return None if defer_lookup or pending_in_window else consecutive_hits

    def _lookup_complete_chunks(
        self,
        req_status: RequestOffloadState,
        max_num_new_tokens: int | None = None,
    ) -> int | None:
        """Find how many tokens beyond num_locally_computed_tokens can be loaded.

        Iterates full-attention groups first (prefix lookup), then sliding-window
        groups (suffix lookup). Each group may tighten max_hit_size_tokens, which
        can invalidate an earlier group's result, so the loop re-runs when that
        happens until num_hit_tokens converges.
        """
        num_computed_tokens = req_status.num_locally_computed_tokens
        max_hit_size_tokens: int = req_status.req.num_tokens
        if max_num_new_tokens is not None:
            max_hit_size_tokens = min(
                max_hit_size_tokens, num_computed_tokens + max_num_new_tokens
            )
        if req_status.max_load_tokens is not None:
            max_hit_size_tokens = min(
                max_hit_size_tokens,
                num_computed_tokens + req_status.max_load_tokens,
            )
        if self._sliding_window_groups:
            # the last prompt token has to be recomputed to get the logprobs
            # for sliding window attention, we must reduce by 1 to make sure
            # we still have a hit after reduction
            max_hit_size_tokens -= 1
            if self._mamba_align_size is not None:
                # Constrain hit-window to the mamba block size.
                max_hit_size_tokens = round_down(
                    max_hit_size_tokens, self._mamba_align_size
                )

        num_hit_tokens: int = 0
        defer_lookup = False
        lookup_groups = self._lookup_groups

        # Tracks which eagle groups have already popped their volatile trailing chunk
        # in the current convergence iteration. Reset when a non-eagle group
        # tightens the hit boundary, requiring a fresh pop.
        eagle_verified: set[int] = set()
        while lookup_groups:
            looked_up_sliding_window: bool = False
            groups_iter = iter(lookup_groups)
            lookup_groups = ()
            for group_idx in groups_iter:
                group_config: GroupOffloadConfig = self.config.kv_group_configs[
                    group_idx
                ]
                group_state: RequestGroupState = req_status.group_states[group_idx]
                tokens_per_chunk = group_config.tokens_per_chunk
                offload_keys = group_state.offload_keys

                assert (
                    len(offload_keys) >= req_status.req.num_tokens // tokens_per_chunk
                )

                is_eagle_unverified = (
                    group_config.is_eagle_group and group_idx not in eagle_verified
                )

                # Constrain to a chunk-aligned boundary for this group.
                max_hit_size_tokens = min(
                    max_hit_size_tokens, len(offload_keys) * tokens_per_chunk
                )
                if req_status.max_load_tokens is not None:
                    max_hit_size_tokens = round_down(
                        max_hit_size_tokens, tokens_per_chunk
                    )
                if max_hit_size_tokens - num_computed_tokens < tokens_per_chunk:
                    # We can only load less than a chunk, so skip.
                    return 0

                sliding_window_size_in_chunks = (
                    group_config.sliding_window_size_in_chunks
                )

                # For eagle groups, query one extra chunk that will be popped.
                # Widening applies to every group type: without it, the pop
                # below shrinks max_hit_size_tokens past what was queried,
                # which can push the confirmed boundary under a coarser
                # sibling group's chunk granularity and zero the whole
                # request's hit (issue #52735).
                query_max = max_hit_size_tokens
                if is_eagle_unverified:
                    query_max = min(
                        max_hit_size_tokens + tokens_per_chunk,
                        len(offload_keys) * tokens_per_chunk,
                    )

                num_chunks = min(cdiv(query_max, tokens_per_chunk), len(offload_keys))
                start_chunk_idx = num_computed_tokens // tokens_per_chunk
                offload_keys = offload_keys[start_chunk_idx:num_chunks]

                # end index (in the sliced offload_keys) up to which we
                # have backend-confirmed hits
                num_hit_chunks: int | None
                if sliding_window_size_in_chunks is None:
                    num_hit_chunks = self._maximal_prefix_lookup(
                        offload_keys,
                        req_status.req_context,
                        req_status.req,
                        group_config,
                        start_chunk_idx,
                    )
                else:
                    required_window = sliding_window_size_in_chunks
                    if is_eagle_unverified:
                        required_window += 1
                    candidate_end = min(
                        max_hit_size_tokens,
                        (num_chunks - int(is_eagle_unverified)) * tokens_per_chunk,
                    )
                    initial_window = group_config.load_window_size_in_chunks(
                        candidate_end
                    )
                    assert initial_window is not None
                    num_hit_chunks = self._sliding_window_lookup(
                        offload_keys,
                        required_window,
                        req_status.req_context,
                        initial_window + int(is_eagle_unverified),
                    )
                if num_hit_chunks == 0:
                    return 0

                if num_hit_chunks is None:
                    defer_lookup = True
                else:
                    if is_eagle_unverified:
                        num_hit_chunks -= 1
                        eagle_verified.add(group_idx)

                    max_hit_size_tokens = min(
                        max_hit_size_tokens,
                        tokens_per_chunk * (start_chunk_idx + num_hit_chunks),
                    )

                new_num_hit_tokens = max_hit_size_tokens - num_computed_tokens
                if new_num_hit_tokens < tokens_per_chunk:
                    # We can only load less than a chunk, so skip.
                    return 0

                if new_num_hit_tokens < num_hit_tokens:
                    if not group_config.is_eagle_group:
                        eagle_verified.clear()
                    if defer_lookup:
                        # make another iteration on all groups to check
                        # if we still need to defer lookup
                        defer_lookup = False
                        lookup_groups = self._lookup_groups
                    elif looked_up_sliding_window and not lookup_groups:
                        # we need another iteration to confirm previously looked up
                        # sliding window works with the new_num_hit_tokens
                        lookup_groups = self._sliding_window_groups

                looked_up_sliding_window |= sliding_window_size_in_chunks is not None
                num_hit_tokens = new_num_hit_tokens

        if defer_lookup:
            logger.debug(
                "Offloading manager delayed request %s as backend requested",
                req_status.req.request_id,
            )
            return None

        # Possibly delay the request if any hit chunk is already being loaded.
        if self._chunks_being_loaded:
            for group_config, group_state in zip(
                self.config.kv_group_configs, req_status.group_states
            ):
                tokens_per_chunk = group_config.tokens_per_chunk
                sliding_window_size_in_chunks = group_config.load_window_size_in_chunks(
                    num_computed_tokens + num_hit_tokens
                )
                offload_keys = group_state.offload_keys
                num_chunks = cdiv(
                    num_computed_tokens + num_hit_tokens, tokens_per_chunk
                )
                start_chunk_idx = num_computed_tokens // tokens_per_chunk
                offload_keys = offload_keys[start_chunk_idx:num_chunks]
                if sliding_window_size_in_chunks is not None:
                    offload_keys = offload_keys[-sliding_window_size_in_chunks:]
                if any(key in self._chunks_being_loaded for key in offload_keys):
                    # Hit chunks are being loaded, so delay the request.
                    logger.debug(
                        "Delaying request %s since some of its"
                        " chunks are already being loaded",
                        req_status.req.request_id,
                    )
                    return None

        logger.debug(
            "Request %s hit %s offloaded tokens after %s GPU hit tokens",
            req_status.req.request_id,
            num_hit_tokens,
            num_computed_tokens,
        )

        return num_hit_tokens

    def _lookup(
        self,
        req_status: RequestOffloadState,
        max_num_new_tokens: int | None = None,
    ) -> int | None:
        complete_hit = self._lookup_complete_chunks(req_status, max_num_new_tokens)
        req_status.partial_tail_boundary = None
        if complete_hit is None or not self.config.supports_partial_tail:
            return complete_hit

        local_tokens = req_status.num_locally_computed_tokens
        complete_boundary = local_tokens + complete_hit
        tokens_per_hash = self.config.tokens_per_hash
        block_end = complete_boundary + self.config.partial_tail_block_size
        max_boundary = min(req_status.req.num_prompt_tokens - 1, block_end - 1)
        if max_num_new_tokens is not None:
            max_boundary = min(max_boundary, local_tokens + max_num_new_tokens)
        if req_status.max_load_tokens is not None:
            max_boundary = min(
                max_boundary,
                local_tokens + req_status.max_load_tokens,
            )
        max_boundary = round_down(max_boundary, tokens_per_hash)
        if max_boundary <= complete_boundary:
            return complete_hit

        pending = False
        for boundary in range(max_boundary, complete_boundary, -tokens_per_hash):
            boundary_pending = False
            boundary_missed = False
            boundary_keys = []
            for group_config in self.config.kv_group_configs:
                key = req_status.make_boundary_key(group_config.group_idx, boundary)
                boundary_keys.append(key)
                result = self.manager.lookup(key, req_status.req_context)
                if result is LookupResult.MISS:
                    boundary_missed = True
                    break
                if result in (LookupResult.HIT_PENDING, LookupResult.RETRY):
                    boundary_pending = True

            pending |= boundary_pending
            if not boundary_missed and not boundary_pending:
                for group_config, key in zip(
                    self.config.kv_group_configs, boundary_keys
                ):
                    self._events_tracker.record_partial_lookup(
                        req_status.req, group_config, boundary, key
                    )
                req_status.partial_tail_boundary = boundary
                return boundary - local_tokens

        if pending and complete_hit == 0:
            return None
        return complete_hit

    def on_new_request(self, request: Request) -> None:
        """Called when a new request is added to the scheduler."""
        req_context = _create_req_context(request)
        with self._manager_locked():
            offloading_context = self.manager.on_new_request(req_context)
        req_status = RequestOffloadState(
            config=self.config,
            req=request,
            req_context=req_context,
            offloading_context=offloading_context,
        )
        self._req_status[request.request_id] = req_status

    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
        max_num_new_tokens: int | None = None,
    ) -> tuple[int | None, bool]:
        """Get number of new tokens that can be loaded beyond the
        num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request
            max_num_new_tokens (int | None): cap on the number of tokens that
                may be loaded beyond `num_computed_tokens`, if any.

        Returns:
            A tuple with the following elements:
                - The number of tokens that can be loaded beyond what is
                  already computed.
                  If None, it means that the connector needs more time to
                  determine the number of matched tokens, and the scheduler
                  should query for this request again later.
                - `True` if tokens will be loaded asynchronously
                  (between scheduler steps).

        """
        # First manager-touching hook of a step: opens the step-scoped region,
        # closed at the end of build_connector_meta.
        self._acquire_step()

        req_status = self._req_status[request.request_id]
        for group_state in req_status.group_states:
            group_state.block_ids.clear()

        if req_status.transfer_jobs:
            logger.debug(
                "Delaying request %s since it still has in-flight transfers",
                request.request_id,
            )
            return None, False

        req_status.update_offload_keys()
        req_status.num_locally_computed_tokens = num_computed_tokens

        num_hit_tokens: int | None
        if request.skip_reading_prefix_cache:
            num_hit_tokens = 0
        else:
            lookup_start = time.monotonic()
            num_hit_tokens = self._lookup(req_status, max_num_new_tokens)
            self._connector_stats.observe_histogram(
                _ConnectorMetricName.LOOKUP_SYNC_DELAY,
                time.monotonic() - lookup_start,
            )
            if num_hit_tokens is None:
                if req_status.deferred_lookup_start_time is None:
                    req_status.deferred_lookup_start_time = lookup_start
            else:
                self._maybe_observe_lookup_async_delay(req_status)
        req_status.update_num_hit_chunks(num_computed_tokens + (num_hit_tokens or 0))

        return num_hit_tokens, bool(num_hit_tokens)

    def get_external_cache_hit_sources(
        self,
        request: Request,
        num_external_tokens: int,
    ) -> dict[CacheHitSource, int]:
        """Split the accepted external hit by the tier each loaded key came from.

        Attention keys cover only their loaded token ranges; recurrent state
        covers the whole reused prefix. Overlaps report the outermost tier.
        Uncovered tokens, including omitted sliding-window history, are
        ``external_unspecified``.
        """
        sources: dict[CacheHitSource, int] = {}
        if num_external_tokens == 0:
            return sources
        req_status = self._req_status[request.request_id]
        start = req_status.num_locally_computed_tokens
        end = start + num_external_tokens

        events: list[tuple[int, int, CacheHitSource]] = [
            (end, 0, CacheHitSource.EXTERNAL_UNSPECIFIED)
        ]
        for lo, hi, key in req_status.load_key_ranges:
            lo, hi = max(start, lo), min(end, hi)
            if lo >= hi:
                continue
            source = CacheHitSource(
                self.manager.get_load_source(key, req_status.req_context)
            )
            events.extend(((lo, 1, source), (hi, -1, source)))

        # Sweep endpoints instead of rescanning every range for each interval.
        active: Counter[CacheHitSource] = Counter()
        position = start
        for boundary, delta, source in sorted(events):
            if boundary > position:
                tiers = [tier for tier, count in active.items() if count > 0]
                tier = (
                    CacheHitSource.outermost(tiers)
                    if tiers
                    else CacheHitSource.EXTERNAL_UNSPECIFIED
                )
                sources[tier] = sources.get(tier, 0) + boundary - position
            active[source] += delta
            position = boundary
        return sources

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
        # Track allocated block IDs regardless of num_external_tokens so
        # re-allocation flush detection sees blocks allocated by other
        # connectors in a MultiConnector setup.
        for group_blocks in blocks.blocks:
            self._current_batch_allocated_block_ids.update(
                block.block_id for block in group_blocks if block.block_id != 0
            )

        if num_external_tokens == 0:
            return

        # prepare_load() below pins keys that get_num_new_matched_tokens()
        # already saw as HIT, so both must sit inside the same region.
        self._acquire_step()

        req_status = self._req_status[request.request_id]

        num_locally_computed_tokens = req_status.num_locally_computed_tokens
        num_cached_tokens = num_locally_computed_tokens + num_external_tokens
        partial_tail_boundary = req_status.partial_tail_boundary
        if partial_tail_boundary is not None:
            assert partial_tail_boundary == num_cached_tokens

        keys_to_load: list[OffloadKey] = []
        req_status.load_key_ranges.clear()
        dst_block_ids: list[int] = []
        # per group
        group_sizes: list[int] = []
        block_indices: list[int] = []
        for group_config, group_state in zip(
            self.config.kv_group_configs,
            req_status.group_states,
        ):
            group_blocks = blocks.blocks[group_config.group_idx]
            tokens_per_block = group_config.tokens_per_block
            tokens_per_chunk = group_config.tokens_per_chunk
            offload_keys = group_state.offload_keys
            num_gpu_blocks = cdiv(num_cached_tokens, tokens_per_block)

            assert len(group_blocks) >= num_gpu_blocks
            # ``load_start_gpu_block_idx``: the index in ``group_blocks`` where the
            # load region begins -- a slice bound, not a count of computed blocks.
            # Scan from the computed boundary, not 0: sparse groups (Mamba,
            # SWA) legitimately hold non-null unhashed blocks below it.
            # Skip nulls (sentinel / out-of-retention padding).
            first_fresh_gpu_block_idx = cdiv(
                num_locally_computed_tokens, tokens_per_block
            )
            load_start_gpu_block_idx = num_gpu_blocks
            for i in range(first_fresh_gpu_block_idx, num_gpu_blocks):
                block = group_blocks[i]
                if not block.is_null and block.block_hash is None:
                    load_start_gpu_block_idx = i
                    break

            assert num_locally_computed_tokens % tokens_per_block == 0
            num_pending_gpu_blocks = num_gpu_blocks - load_start_gpu_block_idx

            if group_config.sliding_window_size_in_chunks is not None:
                assert (
                    num_pending_gpu_blocks
                    <= group_config.sliding_window_size_in_chunks
                    * self.config.blocks_per_chunk
                    + 1
                )

            num_chunks = cdiv(num_cached_tokens, tokens_per_chunk)
            if num_pending_gpu_blocks:
                start_chunk_idx = (
                    load_start_gpu_block_idx // self.config.blocks_per_chunk
                )
                end_chunk_idx = num_chunks - (partial_tail_boundary is not None)
                assert len(offload_keys) >= end_chunk_idx
                group_keys_to_load = offload_keys[start_chunk_idx:end_chunk_idx]
                if partial_tail_boundary is not None:
                    group_keys_to_load.append(
                        req_status.make_boundary_key(
                            group_config.group_idx, partial_tail_boundary
                        )
                    )
                keys_to_load.extend(group_keys_to_load)
                if isinstance(group_config.kv_cache_spec, MambaSpec):
                    req_status.load_key_ranges.extend(
                        (num_locally_computed_tokens, num_cached_tokens, key)
                        for key in group_keys_to_load
                    )
                else:
                    for chunk_idx, key in enumerate(
                        group_keys_to_load, start_chunk_idx
                    ):
                        range_start = max(
                            load_start_gpu_block_idx * tokens_per_block,
                            chunk_idx * tokens_per_chunk,
                        )
                        range_end = min(
                            num_cached_tokens, (chunk_idx + 1) * tokens_per_chunk
                        )
                        req_status.load_key_ranges.append((range_start, range_end, key))

            dst_block_ids.extend(
                block.block_id
                for block in group_blocks[load_start_gpu_block_idx:num_gpu_blocks]
            )
            group_sizes.append(num_pending_gpu_blocks)
            block_indices.append(load_start_gpu_block_idx)

        for handler in self._store_handlers.values():
            handler.on_load_scheduled(req_status, num_cached_tokens)

        src_spec = self.manager.prepare_load(keys_to_load, req_status.req_context)
        dst_spec = GPULoadStoreSpec(
            dst_block_ids, group_sizes=group_sizes, block_indices=block_indices
        )

        load_job_id = self._generate_job_id()
        self._current_batch_load_jobs[load_job_id] = TransferJob(
            req_id=request.request_id,
            src_spec=src_spec,
            dst_spec=dst_spec,
        )
        # a load can only be issued when no other jobs are pending.
        assert not req_status.transfer_jobs
        req_status.transfer_jobs.add(load_job_id)
        self._job_statuses[load_job_id] = TransferJobStatus(
            req_id=request.request_id,
            pending_count=self.config.num_workers,
            keys=set(keys_to_load),
            is_store=False,
        )

        if self._chunks_being_loaded is not None:
            self._chunks_being_loaded.update(keys_to_load)
        req_status.partial_tail_boundary = None

    def _update_req_states(self, scheduler_output: SchedulerOutput) -> None:
        """Update request states from the Scheduler's output."""
        # new_block_ids_end[req_id][i] = end of pre-existing block_ids for
        # the i-th sliding window group (before this step's extend).
        # Used to detect sliding window blocks that got re-allocated.
        new_block_ids_end: dict[str, tuple[int, ...]] = {}

        for req_id, new_block_id_groups, preempted in yield_req_data(scheduler_output):
            req_status = self._req_status[req_id]
            req_status.update_offload_keys()

            if preempted:
                for group_state in req_status.group_states:
                    group_state.block_ids.clear()

            if new_block_id_groups:
                if self._sliding_window_groups:
                    new_block_ids_end[req_id] = tuple(
                        len(req_status.group_states[grp_idx].block_ids)
                        for grp_idx in self._sliding_window_groups
                    )
                req_status.update_block_id_groups(new_block_id_groups)
                for group_config in self.config.kv_group_configs:
                    new_blocks = new_block_id_groups[group_config.group_idx]
                    for bid in new_blocks:
                        if bid != 0:
                            self._current_batch_allocated_block_ids.add(bid)

        for copy in scheduler_output.kv_cache_block_copies or ():
            self._current_batch_allocated_block_ids.add(copy.dst_block_id)

        # Zero out stale block_ids in sliding window groups' pending-store
        # positions. Only sliding window groups can have stale entries (blocks
        # freed by remove_skipped_blocks then reallocated). Only positions in
        # [next_stored_chunk_idx * bsf, end) need checking where end is the
        # pre-extend length: earlier positions were already offloaded, later
        # ones are fresh allocations from this step.
        if self._sliding_window_groups and self._current_batch_allocated_block_ids:
            blocks_per_chunk = self.config.blocks_per_chunk
            for req_id, req_status in self._req_status.items():
                ends = new_block_ids_end.get(req_id)
                for i, grp_idx in enumerate(self._sliding_window_groups):
                    group_state = req_status.group_states[grp_idx]
                    start = group_state.next_stored_chunk_idx * blocks_per_chunk
                    end = ends[i] if ends is not None else len(group_state.block_ids)
                    for j in range(start, end):
                        if (
                            group_state.block_ids[j]
                            in self._current_batch_allocated_block_ids
                        ):
                            group_state.block_ids[j] = 0

    def _add_store_jobs(self, output: StoreJobsOutput) -> dict[int, TransferJob]:
        """Give each store job of a handler a job ID, and track it."""
        if output.num_allocation_failures:
            self._connector_stats.increase_counter(
                _ConnectorMetricName.ALLOCATION_FAILURE,
                output.num_allocation_failures,
            )
        store_jobs: dict[int, TransferJob] = {}
        for job, job_status in output.jobs:
            req_status = self._req_status[job_status.req_id]
            job_id = self._generate_job_id()
            # a store can only be issued when no load is pending.
            if req_status.transfer_jobs:
                any_jid = next(iter(req_status.transfer_jobs))
                assert self._job_statuses[any_jid].is_store
            req_status.transfer_jobs.add(job_id)

            # Watch sliding window blocks as they may get evicted
            # before the request finishes
            for bid in job_status.fenced_block_ids or ():
                self._block_id_to_pending_jobs.setdefault(bid, set()).add(job_id)

            self._job_statuses[job_id] = job_status
            store_jobs[job_id] = job

            if req_status.req.is_finished():
                # Register non-sliding-window blocks for flush detection.
                for bid in job_status.deferred_fence_block_ids or ():
                    self._block_id_to_pending_jobs.setdefault(bid, set()).add(job_id)
                    if bid in self._current_batch_allocated_block_ids:
                        self._current_batch_jobs_to_flush.add(job_id)
        return store_jobs

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        # A step that scheduled nothing reaches build_connector_meta without
        # any earlier manager touch, so open the region here too.
        self._acquire_step()
        try:
            return self._build_connector_meta(scheduler_output)
        finally:
            # The step is over: hand the manager back.
            self._release_step()

    def _build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        self._update_req_states(scheduler_output)
        schedule_end_context = ScheduleEndContext(
            new_req_ids=[req.req_id for req in scheduler_output.scheduled_new_reqs],
            preempted_req_ids=scheduler_output.preempted_req_ids or (),
        )

        # Flush jobs for preempted requests.
        for req_id in scheduler_output.preempted_req_ids or ():
            req_status = self._req_status.get(req_id)
            if req_status is None or not req_status.transfer_jobs:
                continue
            any_jid = next(iter(req_status.transfer_jobs))
            assert self._job_statuses[any_jid].is_store
            self._current_batch_jobs_to_flush.update(req_status.transfer_jobs)

        # Flush jobs that contain re-allocated blocks.
        if (
            self._block_id_to_pending_jobs
            and not self._block_id_to_pending_jobs.keys().isdisjoint(
                self._current_batch_allocated_block_ids
            )
        ):
            self._current_batch_jobs_to_flush.update(
                jid
                for bid in self._current_batch_allocated_block_ids
                if bid in self._block_id_to_pending_jobs
                for jid in self._block_id_to_pending_jobs[bid]
            )

        store_jobs: dict[int, TransferJob] = {}
        for handler in self._store_handlers.values():
            store_jobs |= self._add_store_jobs(
                handler.build_store_jobs(scheduler_output, self._req_status)
            )
        meta = OffloadingConnectorMetadata(
            load_jobs=self._current_batch_load_jobs,
            store_jobs=store_jobs,
            jobs_to_flush=self._current_batch_jobs_to_flush,
        )

        # All prepare_store calls for finished requests have been issued.
        # Signal on_request_finished and clean up state where possible.
        for req_id in scheduler_output.finished_req_ids or ():
            req_status = self._req_status.get(req_id)
            if req_status is None:
                continue
            req_status.finished_signaled = True
            self.manager.on_request_finished(req_status.req_context)
            if not req_status.transfer_jobs:
                del self._req_status[req_id]
        self._current_batch_load_jobs = {}
        self._current_batch_jobs_to_flush = set()
        self._current_batch_allocated_block_ids = set()

        # Last manager call of the step, after every prepare_store and
        # on_request_finished above. on_schedule_end is documented as running at
        # the end of the step, and callers rely on that to bound the step's
        # exclusive access to the manager.
        self.manager.on_schedule_end(schedule_end_context)
        return meta

    def has_pending_push_work(self) -> bool:
        """Whether the engine must keep stepping.

        While True, build_connector_meta() and update_connector_output()
        continue to be called even when no requests are scheduled.

        Must use a short region, never the step-scoped one. The engine asks this
        before schedule() and returns early when the answer is False, so a
        step-scoped acquire here would never reach build_connector_meta's
        release and would hold the manager for good -- which is precisely the
        state an idle engine settles into.
        """
        if self._job_statuses:
            return True
        with self._manager_locked():
            return self.manager.has_pending_work()

    def update_connector_output(self, connector_output: KVConnectorOutput):
        """Update KVConnector state from worker-side connectors output.

        Args:
            connector_output (KVConnectorOutput): the worker-side
                connectors output.

        """
        with self._manager_locked():
            self._update_connector_output(connector_output)

    def _update_connector_output(self, connector_output: KVConnectorOutput):
        meta = connector_output.kv_connector_worker_meta
        if not isinstance(meta, OffloadingWorkerMetadata):
            assert meta is None
            meta = OffloadingWorkerMetadata()
        if not meta.transfer_stats.is_empty():
            transfer_stats = OffloadingConnectorStats()
            if not meta.transfer_stats.load.is_empty():
                transfer_stats.increase_counter(
                    _TransferMetricName.LOAD_BYTES,
                    meta.transfer_stats.load.bytes,
                )
                transfer_stats.increase_counter(
                    _TransferMetricName.LOAD_TIME,
                    meta.transfer_stats.load.time,
                )
                for size in meta.transfer_stats.load.sizes:
                    transfer_stats.observe_histogram(
                        _TransferMetricName.LOAD_SIZE, size
                    )
            if not meta.transfer_stats.store.is_empty():
                transfer_stats.increase_counter(
                    _TransferMetricName.STORE_BYTES,
                    meta.transfer_stats.store.bytes,
                )
                transfer_stats.increase_counter(
                    _TransferMetricName.STORE_TIME,
                    meta.transfer_stats.store.time,
                )
                for size in meta.transfer_stats.store.sizes:
                    transfer_stats.observe_histogram(
                        _TransferMetricName.STORE_SIZE, size
                    )
            self._connector_stats.aggregate(transfer_stats)

        for job_id, count in meta.completed_jobs.items():
            assert count > 0
            if job_id < self._stale_job_threshold:
                logger.debug(
                    "Skipping stale completed job %d (pre-reset counter: %d)",
                    job_id,
                    self._stale_job_threshold,
                )
                continue
            job_status = self._job_statuses[job_id]
            job_status.pending_count -= count
            if job_status.pending_count > 0:
                continue
            assert job_status.pending_count == 0

            req_status = self._req_status[job_status.req_id]
            if job_status.is_store:
                self.manager.complete_store(job_status.keys, req_status.req_context)
            else:
                self.manager.complete_load(job_status.keys, req_status.req_context)
                if self._chunks_being_loaded:
                    self._chunks_being_loaded.difference_update(job_status.keys)
            if self._block_id_to_pending_jobs:
                # Sliding window blocks are tracked from store creation
                # and must be cleaned up unconditionally.
                self._remove_pending_job(job_id, job_status.fenced_block_ids)
                # Non-sliding-window blocks are only tracked after
                # request_finished, so only clean up for finished requests.
                if req_status.req.is_finished():
                    self._remove_pending_job(
                        job_id, job_status.deferred_fence_block_ids
                    )

            del self._job_statuses[job_id]
            req_status.transfer_jobs.remove(job_id)
            if req_status.finished_signaled and not req_status.transfer_jobs:
                del self._req_status[job_status.req_id]

    def get_stats(self) -> OffloadingConnectorStats | None:
        stats: OffloadingConnectorStats | None = None
        if not self._connector_stats.is_empty():
            stats = self._connector_stats
            self._connector_stats = OffloadingConnectorStats()

        with self._manager_locked():
            manager_stats = self.manager.get_stats()
        if manager_stats is not None:
            if stats is None:
                stats = manager_stats
            else:
                stats.aggregate(manager_stats)

        if not self._info_sent:
            if stats is None:
                stats = OffloadingConnectorStats()
            # Sent even when the manager reports no facts, so that the metric
            # exists whenever offloading runs.
            stats.set_info(self.manager.config_info())
            self._info_sent = True

        return stats

    def request_finished(
        self,
        request: Request,
    ) -> tuple[bool, dict[str, Any] | None]:
        """Called when a request has finished, before its blocks are freed.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.

        """
        req_status = self._req_status.get(request.request_id)

        if req_status is None:
            # Untracked request (offloading never started): no in-flight jobs,
            # nothing was deferred, so finalize immediately.
            req_context = _create_req_context(request)
            with self._manager_locked():
                self.manager.on_new_request(req_context)
                self.manager.on_request_finished(req_context)
            return False, None

        self._maybe_observe_lookup_async_delay(req_status)

        # Update offload keys with final block hash so the ON_COMPUTE handler can
        # create store jobs for the last block(s) on the next schedule step.
        req_status.update_offload_keys()

        # Keep req_status alive: the ON_COMPUTE handler will process finished_req_ids
        # on the next step and handle cleanup after creating store jobs.
        # Register deferred fences so future block reuse triggers a flush via
        # _block_id_to_pending_jobs.
        for job_id in req_status.transfer_jobs:
            job_status = self._job_statuses[job_id]
            for bid in job_status.deferred_fence_block_ids or ():
                self._block_id_to_pending_jobs.setdefault(bid, set()).add(job_id)

        return False, None

    def take_events(self) -> Iterable[KVCacheEvent]:
        """Drain pending KV cache events.

        Complete metadata is available only when self-describing KV events
        are enabled, and only for full-attention groups. Other shapes retain
        the previous placeholder payload so consumers can ignore them.

        Yields:
            ``BlockStored`` or ``BlockRemoved`` events corresponding to
            the underlying :class:`OffloadingEvent` stream.

        """
        # Drain the manager eagerly, inside the region. manager.take_events() is
        # itself a generator whose backing list is cleared once exhausted, so
        # iterating it lazily would both hold the manager for as long as the
        # consumer takes and let a concurrent writer append events that the
        # clear then discards unpublished.
        with self._manager_locked():
            manager_events = list(self.manager.take_events())
        # Still a generator, so callers testing the result for truthiness keep
        # seeing a non-empty object the way they do today.
        yield from self._events_tracker.take_events(manager_events)

    def reset_cache(self) -> None:
        """Reset the offloading manager cache, evicting all stored chunks."""
        # reset_cache cannot be called in the middle of a schedule step
        assert not self._current_batch_load_jobs
        assert not self._current_batch_jobs_to_flush
        assert not self._current_batch_allocated_block_ids

        # Held for the whole reset. manager.reset_cache() drains tiers by
        # polling them itself, so a tier-poll thread locked out here loses
        # nothing; it simply skips rounds until the reset completes.
        with self._manager_locked():
            self._reset_cache()

    def _reset_cache(self) -> None:
        # Flush all in-flight jobs
        self._current_batch_jobs_to_flush.update(self._job_statuses.keys())

        for req_id, status in list(self._req_status.items()):
            if status.req.is_finished():
                if not status.finished_signaled:
                    self.manager.on_request_finished(status.req_context)
                del self._req_status[req_id]

        # Reset offloading manager cache
        self.manager.reset_cache()

        for status in self._req_status.values():
            status.transfer_jobs.clear()
            status.partial_tail_boundary = None

        # Discard jobs and save job_counter to be able to discard worker responses
        self._stale_job_threshold = self._job_counter
        self._job_statuses.clear()
        self._block_id_to_pending_jobs.clear()
        for handler in self._store_handlers.values():
            handler.reset(self._req_status)

        # The manager pool is empty; pending event payloads and announced
        # reference counts are stale.
        self._events_tracker.reset()

        # Note: _current_batch_jobs_to_flush is intentionally NOT cleared.
        # The load flush IDs collected above must be delivered to workers.
        if self._chunks_being_loaded is not None:
            self._chunks_being_loaded.clear()

    def shutdown(self) -> None:
        self.manager.shutdown()
