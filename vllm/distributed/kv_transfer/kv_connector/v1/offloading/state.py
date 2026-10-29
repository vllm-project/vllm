# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field
from itertools import islice
from typing import NamedTuple

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
    _group_kv_bytes_per_block,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.events import (
    OffloadingEventGroupSpec,
    get_offloading_event_group_spec,
)
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import eagle_proof_margin
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
)
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
from vllm.v1.kv_offload.base import (
    OffloadingSpec,
    OffloadKey,
    ReqContext,
    RequestOffloadingContext,
    make_offload_key,
)
from vllm.v1.request import Request

logger = init_logger(__name__)


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
    sync_load_budget_bytes: int = 0
    # KV bytes per token on one worker, summed over the offloaded groups.
    kv_bytes_per_token: int = 0

    # Per group, the tokens past a partial-tail boundary whose hash keys the
    # group's partial-tail block. Non-zero only for EAGLE groups.
    eagle_proof_margins: tuple[int, ...] = ()

    @property
    def partial_tail_block_size(self) -> int:
        if not self.supports_partial_tail:
            return 0
        return self.kv_group_configs[0].tokens_per_block

    @property
    def max_eagle_proof_margin(self) -> int:
        return max(self.eagle_proof_margins, default=0)

    def partial_tail_key_tokens(self, config_idx: int, boundary: int) -> int:
        """Token position whose hash keys a group's partial-tail block."""
        if not self.eagle_proof_margins:
            return boundary
        return boundary + self.eagle_proof_margins[config_idx]

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
            if group.sliding_window_size_in_chunks is None:
                full_attn_tokens_per_chunk.add(
                    group.tokens_per_block * spec.blocks_per_chunk
                )

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
                    sliding_window_size_in_chunks=group.sliding_window_size_in_chunks,
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
        # group's source. EAGLE groups key their partial tail at a proof
        # position past the boundary. DCP needs additional hand-off semantics.
        supports_partial_tail = (
            spec.blocks_per_chunk == 1
            and len(group_block_sizes) == 1
            and has_partial_recurrent_group
            and all(
                config.sliding_window_size_in_chunks is None
                or config.requires_cow_source
                for config in kv_group_configs
            )
            and vllm_config.parallel_config.decode_context_parallel_size == 1
        )

        # Core's prefix cache matches an EAGLE group this many tokens past a
        # hit and then drops them. A partial tail stores and looks up EAGLE
        # groups at that proof position, with the margin rule of the core
        # coordinator.
        eagle_proof_margins: tuple[int, ...] = ()
        if supports_partial_tail:
            partial_tail_block_size = kv_group_configs[0].tokens_per_block
            alignment = kv_cache_config.cache_hit_alignment_tokens
            partial_hash_hits = (
                alignment is not None and alignment < partial_tail_block_size
            )
            eagle_proof_margins = tuple(
                eagle_proof_margin(
                    partial_tail_block_size,
                    spec.tokens_per_hash,
                    partial_hash_hits
                    and config.manager_cls.supports_fine_grained_hash_lookup,
                )
                if config.is_eagle_group and not config.requires_cow_source
                else 0
                for config in kv_group_configs
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
            # Per-step KV bytes (per worker) loaded within the step (0 disables).
            sync_load_budget_bytes=int(
                spec.extra_config.get("sync_load_budget_bytes", 0)
            ),
            kv_bytes_per_token=sum(
                _group_kv_bytes_per_block(kv_cache_config.kv_cache_groups[g.group_id])
                // g.tokens_per_block
                for g in spec.config.groups
            ),
            eagle_proof_margins=eagle_proof_margins,
        )


@dataclass
class RequestGroupState:
    offload_keys: list[OffloadKey] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)
    # Number of offloaded chunks hit (including GPU prefix cache)
    # when the request first started
    num_hit_chunks: int = 0
    # End (exclusive) of the chunks that the current complete-chunk lookup
    # confirmed as ready hits, including a popped EAGLE chunk.
    lookup_hit_end_chunk_idx: int = 0


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
    # True once a load from the offload tier was scheduled for this request.
    loaded_from_offload: bool = False
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
        consistently everywhere the ON_COMPUTE progress is derived:
        otherwise the trailing chunk of each step is skipped on collection but
        jumped over by the progress, so it is never re-considered
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

    def register_boundary_key(self, group_idx: int, boundary_tokens: int) -> OffloadKey:
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
