# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SimpleReplicationConnector: deliberate multi-storage for shared-prefix
KV blocks. Structural fix for the cascade-collapse pathology described
in vllm-project/vllm#42948.

THE BUG
-------
Hybrid KV cache models (e.g. DeepSeek-V4-Flash, with one MLA + three
SWA-MLA + one Hidden-State group) cache each block under a key of the
form (block_hash, kv_cache_group_id) — one entry per (position, group).
`find_longest_cache_hit` walks positions and intersects the per-group
hit sets. If even one group loses its first-block entry (e.g. because
a concurrent request's get_new_blocks reassigned it under pool
pressure), the intersection at position 0 collapses to empty and the
request takes a full re-prefill despite the other 4 groups still
having their entries.

Independent reproduction (stecasta, GB300, vLLM v0.20.2, 32K shared
prefix + 2K ISL + 1K OSL, sequential sampling, N=2*BS): vanilla hit
rate drops from 94.3% at BS=2 to 1.0% at BS=4 and 0.3% at BS=8.
Protection-based PRs (#42985 recent-hit, #43191 v1+popular-insert)
mitigate to 46% / 21% but cannot eliminate the cliff because they only
delay eviction — once allocator fallback engages, popular blocks land
in the reassignment pipeline anyway.

THE FIX
-------
`BlockHashToBlockMap._cache[key]` already supports
`KVCacheBlock | dict[int, KVCacheBlock]` (see the merge-to-dict path
in `insert`), but it only triggers accidentally — when two requests
independently cache the same hash. This connector makes the
multi-storage transition deliberate:

1. At connector init (`bind_gpu_block_pool`), pre-reserve a fixed
   pool of pristine blocks via `get_new_blocks(replication_cap)`.
   These never return to the free queue (ref_cnt stays at 1
   permanently), so the model's allocator can never reassign them.
2. On every `get_one_block` cache hit, the BlockHashToBlockMap
   bumps a per-key lookup counter; once it crosses
   `REPLICATION_LOOKUP_THRESHOLD`, it invokes the registered
   replication callback (see `_on_trigger` below).
3. The callback pops one block from the reserve, queues a
   memcpy job in pending metadata. On the next scheduler tick (after
   the worker has executed the forward pass and the memcpy is
   complete), the dst is inserted into the cache map — merging it
   into a multi-storage entry alongside the original source.
4. The merged dict entry now has two block_ids per shared-prefix
   key. When one is evicted by the allocator under pool pressure
   (typically the original source, since the reserved dst is held
   ref_cnt=1 and out of the queue), the entry stays alive via the
   other.

WHY RESERVED BLOCKS
-------------------
An earlier iteration of this connector allocated dst blocks
on-demand via `get_new_blocks(1)` inside the trigger callback. Under
a saturated pool, that allocator path evicted *another* cached block
to satisfy the allocation — and crucially, the block at the queue
head when the trigger fired for H_pos0_g0 could itself be the cached
entry for H_pos0_g1. The "replication" then triggered the very
cascade collapse it was meant to prevent. Pre-reserving pristine
blocks at init time, before any caching has happened, sidesteps this
entirely.

The cost is a fixed slice of the KV pool held in reserve. With the
default `replication_cap=4096` and ~1 MB per block (fp8 MLA on
H200, block_size=256), this is roughly 4 GB of GPU memory
permanently reserved per worker. Reduce via
`kv_connector_extra_config.replication_cap` if pool pressure
matters more than coverage.

SCOPE
-----
The lookup-based trigger fires on the second-or-later get_one_block
hit for a key. It covers the canonical bug shapes:

  * Concurrent shared-prefix workloads (BS>=2): the second session's
    lookup triggers replication of the shared prefix.
  * Four-step `A.1 → A.2 → B → A.3`: A.2's sanity-check lookup of
    A.1's hashes triggers replication ahead of B's allocation.

It does NOT cover the 3-step `A → B → A` pattern (a single session
re-asking after another session has run in between) because A's
hashes get no lookup hit before B evicts them. That gap belongs to
a separate insert-count-based mechanism (see #43191 v3 for prior
art) and can be layered on top in a follow-up PR.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_events import KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import (
    BlockHashWithGroupId,
    KVCacheBlock,
    get_group_id,
)
from vllm.v1.outputs import KVConnectorOutput

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)

# Default size of the pre-reserved pristine block pool. Each reserved
# block is held with ref_cnt=1 for the lifetime of the engine, so this
# is a hard upper bound on the number of shared-prefix hashes that can
# be protected by multi-storage at any time. The default of 4096
# trades ~4 GB of permanently-reserved KV memory (fp8 MLA / H200 /
# block_size=256) for coverage of ~819 block positions across 5
# KV-cache groups — sufficient for stecasta's 32K shared-prefix
# workload and the canonical 4-step #42948 reproducer.
#
# Tune via kv_connector_extra_config.replication_cap when pool
# pressure matters more than coverage (lower) or when running
# substantially longer shared prefixes (higher).
DEFAULT_REPLICATION_CAP = 4096

# Minimum free-block headroom enforced when reducing the reserve from
# the configured cap (only relevant if the pool is unexpectedly small
# at init time — we leave at least this many blocks free for the
# model's own allocations).
DEFAULT_FREE_BLOCK_HEADROOM = 256


@dataclass
class ReplicationOp:
    """A single src→dst block memcpy job, scoped to one KV cache group."""

    src_block_id: int
    dst_block_id: int
    group_id: int


@dataclass
class ReplicationMetadata(KVConnectorMetadata):
    ops: list[ReplicationOp] = field(default_factory=list)


@dataclass
class _PendingReplica:
    """Scheduler-side bookkeeping for a replica not yet committed into the
    prefix cache map. The dst_block carries ref_cnt=1 (allocated via
    get_new_blocks) until commit, which prevents it from being reassigned
    before its content is valid."""

    key: BlockHashWithGroupId
    dst_block: KVCacheBlock
    src_block_id: int


class ReplicationScheduler:
    """Scheduler-side: detects replication triggers via the block_pool
    callback, allocates dst blocks from a pre-reserved pristine pool
    (set aside at bind_gpu_block_pool time so they never collide with
    model allocations), hands memcpy jobs to the worker via connector
    metadata, and commits the replicas into the cache map once the
    worker's forward (and therefore the memcpy) has completed.

    The reserve pool is critical: if we instead called
    `block_pool.get_new_blocks(1)` at trigger time on a saturated pool,
    the allocator would evict a currently-cached block to give us the
    dst — exactly the cascade-collapse pathology we're trying to
    prevent (e.g. a trigger for H_pos0_g0 ends up evicting H_pos0_g1,
    breaking the multi-group intersection at position 0). Pre-reserved
    blocks side-step this entirely by holding ref_cnt=1 permanently —
    they are never in the free queue, so neither the model's
    get_new_blocks nor competing requests can ever steal them. The
    cost is a fixed slice of the KV pool reserved up front."""

    def __init__(self, vllm_config: VllmConfig, replication_cap: int):
        self._vllm_config = vllm_config
        self._replication_cap = replication_cap
        self._block_pool: BlockPool | None = None
        # Pre-reserved pristine blocks, ref_cnt=1 held permanently.
        # Popped as dst for each accepted replication; never returned
        # to the free queue (their ref_cnt stays at 1 forever, which
        # means _maybe_evict_cached_block can never reach them via
        # get_new_blocks).
        self._reserve: list[KVCacheBlock] = []
        # Newly triggered, awaiting emit in next build_connector_meta.
        self._pending: list[_PendingReplica] = []
        # Emitted to worker last tick, awaiting commit in
        # update_connector_output (= after worker's forward completes).
        self._in_flight: list[_PendingReplica] = []
        # Committed dst blocks, ref_cnt=1, present in cache_dict.
        # Tracked so reset_cache can release them ahead of a pool
        # reset_prefix_cache (without this list, committed dsts stay
        # pinned indefinitely and the pool's num_used_blocks check
        # would never pass even after we clear reserve+pending).
        self._committed: list[KVCacheBlock] = []

    def bind_gpu_block_pool(self, gpu_block_pool: "BlockPool") -> None:
        self._block_pool = gpu_block_pool
        self._acquire_reserve()
        gpu_block_pool.register_replication_callback(self._on_trigger)

    def _acquire_reserve(self) -> None:
        """Acquire `replication_cap` pristine reserve blocks from the
        pool. Called at bind time (pool is fresh, guaranteed to succeed
        in full) and lazily at trigger time after `reset_cache` released
        the previous reserve."""
        if self._block_pool is None or self._reserve:
            return
        try:
            self._reserve = self._block_pool.get_new_blocks(self._replication_cap)
        except ValueError:
            reserve_size = max(
                0,
                self._block_pool.get_num_free_blocks() - DEFAULT_FREE_BLOCK_HEADROOM,
            )
            if reserve_size <= 0:
                logger.warning(
                    "SimpleReplicationConnector: pool too tight to "
                    "acquire any reserve blocks; replication disabled "
                    "until pool frees up.",
                )
                return
            logger.warning(
                "SimpleReplicationConnector: requested %d reserve blocks "
                "but pool has fewer free; acquiring %d instead.",
                self._replication_cap,
                reserve_size,
            )
            self._reserve = self._block_pool.get_new_blocks(reserve_size)
        logger.info(
            "SimpleReplicationConnector reserved %d blocks for replication "
            "(pool now has %d free).",
            len(self._reserve),
            self._block_pool.get_num_free_blocks(),
        )

    def reset_cache(self) -> bool:
        """Release reserve + any uncommitted dst blocks and re-run the
        pool's reset_prefix_cache.

        The engine's `scheduler.reset_prefix_cache` calls
        `kv_cache_manager.reset_prefix_cache()` BEFORE
        `connector.reset_cache()`, so the first pool reset likely
        failed its num_used_blocks check on our pinned reserve. We
        retry it here once the reserve is released; the next
        replication trigger then lazily re-acquires the reserve from
        the freshly-pristine pool."""
        if self._block_pool is None:
            return True
        to_release = (
            self._reserve
            + [p.dst_block for p in self._pending]
            + [p.dst_block for p in self._in_flight]
            + self._committed
        )
        self._reserve = []
        self._pending.clear()
        self._in_flight.clear()
        self._committed = []
        if to_release:
            self._block_pool.free_blocks(to_release)
            logger.info(
                "SimpleReplicationConnector reset_cache: released %d "
                "reserve+pending blocks back to the pool.",
                len(to_release),
            )
        # Retry the pool reset now that num_used_blocks is back to 1.
        # If the user invoked reset_prefix_cache with
        # reset_running_requests=True, running requests were already
        # preempted in scheduler.reset_prefix_cache before this hook
        # ran, so the only remaining used block is null_block.
        ok = self._block_pool.reset_prefix_cache()
        if not ok:
            logger.warning(
                "SimpleReplicationConnector reset_cache: pool reset "
                "still failed after releasing reserve. Some non-reserve "
                "blocks are likely still held by running requests."
            )
        return ok

    def _on_trigger(
        self,
        key: BlockHashWithGroupId,
        source_block: KVCacheBlock,
    ) -> bool:
        """Returns True iff a replica was accepted for this hash. False
        return signals BlockHashToBlockMap to keep the hash retry-eligible
        on the next lookup."""
        if self._block_pool is None:
            return False
        if not self._reserve:
            # Lazy refresh after reset_cache (or pool-too-tight at bind).
            self._acquire_reserve()
            if not self._reserve:
                return False
        if len(self._pending) + len(self._in_flight) >= self._replication_cap:
            return False
        # Pop from the pre-reserved pristine pool. dst already has
        # ref_cnt=1 from when we initially get_new_blocks'd it; it has
        # NEVER been in the free queue since, so no eviction risk.
        dst = self._reserve.pop()
        self._pending.append(
            _PendingReplica(
                key=key,
                dst_block=dst,
                src_block_id=source_block.block_id,
            )
        )
        return True

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> ReplicationMetadata:
        ops = [
            ReplicationOp(
                src_block_id=p.src_block_id,
                dst_block_id=p.dst_block.block_id,
                group_id=get_group_id(p.key),
            )
            for p in self._pending
        ]
        self._in_flight.extend(self._pending)
        self._pending.clear()
        return ReplicationMetadata(ops=ops)

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        # Worker has executed the forward pass with the metadata we
        # emitted last tick; memcpys are done. Publish each dst block
        # into the cache map (triggering the merge-to-dict path).
        # Crucially: do NOT free the dst block. We keep ref_cnt=1
        # permanently so the dst is never in the free queue, which
        # means it can never be reassigned by get_new_blocks and thus
        # never evicted from cache_dict by _maybe_evict_cached_block.
        # Multi-storage protection becomes durable across the rest of
        # the engine's lifetime (or until reset_prefix_cache).
        if self._block_pool is None or not self._in_flight:
            return
        committed = self._in_flight
        self._in_flight = []
        for p in committed:
            self._block_pool.insert_cached_block(p.key, p.dst_block)
            self._committed.append(p.dst_block)

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        return 0, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        return

    def request_finished(
        self, request: "Request", block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        return False, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        return False, None


class ReplicationWorker:
    """Worker-side: executes the per-layer GPU memcpy jobs handed down
    in ReplicationMetadata. Synchronous (cuda.synchronize at the end of
    start_load_kv) so the scheduler-side commit on the next tick can
    safely insert the dst blocks into the prefix cache map."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: "KVCacheConfig",
    ):
        self._vllm_config = vllm_config
        self._kv_cache_config = kv_cache_config
        # group_id → layer names that share the group's block table.
        self._group_to_layers: dict[int, list[str]] = {
            gid: list(spec.layer_names)
            for gid, spec in enumerate(kv_cache_config.kv_cache_groups)
        }
        self._kv_caches: dict[str, torch.Tensor] = {}
        self._pending_meta: ReplicationMetadata | None = None

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self._kv_caches = kv_caches

    def bind_connector_metadata(self, meta: ReplicationMetadata) -> None:
        self._pending_meta = meta

    def clear_connector_metadata(self) -> None:
        self._pending_meta = None

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        meta = self._pending_meta
        if meta is None or not meta.ops:
            return
        for op in meta.ops:
            for layer_name in self._group_to_layers.get(op.group_id, ()):
                tensor = self._kv_caches.get(layer_name)
                if tensor is None:
                    continue
                tensor[op.dst_block_id].copy_(tensor[op.src_block_id])
        # Memcpys complete on the default stream as part of the worker
        # step; vLLM's end-of-step output sync (after attention layers
        # run) ensures they have landed before the scheduler's
        # next-tick update_connector_output reads them. No explicit
        # device-wide sync here — would stall the CPU thread and block
        # kernel overlap on the worker hot path.


class SimpleReplicationConnector(KVConnectorBase_V1, SupportsHMA):
    """Multi-storage replication connector for shared-prefix KV blocks.
    Structural fix for #42948 cascade collapse. See module docstring
    for the full design rationale.

    Opt-in via:
        --kv-transfer-config '{"kv_connector": "SimpleReplicationConnector",
                               "kv_role": "kv_both",
                               "kv_connector_extra_config":
                                   {"replication_cap": 4096}}'

    Hybrid KV cache models additionally need
    --no-disable-hybrid-kv-cache-manager (the connector subclasses
    SupportsHMA so this flag is safe to enable).
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        extra = self._kv_transfer_config.kv_connector_extra_config or {}
        replication_cap = int(extra.get("replication_cap", DEFAULT_REPLICATION_CAP))

        self.scheduler: ReplicationScheduler | None = None
        self.worker: ReplicationWorker | None = None

        if not vllm_config.cache_config.enable_prefix_caching:
            logger.warning(
                "Prefix caching is disabled; SimpleReplicationConnector "
                "has nothing to replicate. Connector will be inert."
            )
            return

        if role == KVConnectorRole.SCHEDULER:
            self.scheduler = ReplicationScheduler(vllm_config, replication_cap)
            logger.info(
                "SimpleReplicationConnector scheduler ready (cap=%d).",
                replication_cap,
            )
        elif role == KVConnectorRole.WORKER:
            self.worker = ReplicationWorker(vllm_config, kv_cache_config)
            logger.info(
                "SimpleReplicationConnector worker ready (groups=%d).",
                len(kv_cache_config.kv_cache_groups),
            )

    # ------------------------------------------------------------------
    # Worker-side hooks
    # ------------------------------------------------------------------

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        if self.worker is not None:
            self.worker.register_kv_caches(kv_caches)

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        super().bind_connector_metadata(connector_metadata)
        if self.worker is not None and isinstance(
            connector_metadata, ReplicationMetadata
        ):
            self.worker.bind_connector_metadata(connector_metadata)

    def clear_connector_metadata(self) -> None:
        super().clear_connector_metadata()
        if self.worker is not None:
            self.worker.clear_connector_metadata()

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        if self.worker is not None:
            self.worker.start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        return

    def wait_for_save(self) -> None:
        return

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        return None, None

    # ------------------------------------------------------------------
    # Scheduler-side hooks
    # ------------------------------------------------------------------

    def bind_gpu_block_pool(self, gpu_block_pool: "BlockPool") -> None:
        if self.scheduler is not None:
            self.scheduler.bind_gpu_block_pool(gpu_block_pool)

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        if self.scheduler is not None:
            return self.scheduler.get_num_new_matched_tokens(
                request, num_computed_tokens
            )
        return 0, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        if self.scheduler is not None:
            self.scheduler.update_state_after_alloc(
                request, blocks, num_external_tokens
            )

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> KVConnectorMetadata:
        if self.scheduler is not None:
            return self.scheduler.build_connector_meta(scheduler_output)
        return ReplicationMetadata()

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        if self.scheduler is not None:
            self.scheduler.update_connector_output(connector_output)

    def request_finished(
        self, request: "Request", block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        if self.scheduler is not None:
            return self.scheduler.request_finished(request, block_ids)
        return False, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        if self.scheduler is not None:
            return self.scheduler.request_finished_all_groups(request, block_ids)
        return False, None

    def take_events(self) -> Iterable[KVCacheEvent]:
        return ()

    def reset_cache(self) -> bool | None:
        if self.scheduler is not None:
            return self.scheduler.reset_cache()
        return None
