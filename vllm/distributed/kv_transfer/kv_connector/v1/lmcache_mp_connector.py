# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import enum
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import torch
import zmq
from lmcache.integration.vllm.utils import mla_enabled
from lmcache.utils import init_logger as lmcache_init_logger

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus
from vllm.v1.utils import ConstantList

try:
    from lmcache.integration.vllm.vllm_multi_process_adapter import (
        LMCacheMPSchedulerAdapter,
        LMCacheMPWorkerAdapter,
        LoadStoreOp,
        ParallelStrategy,
    )

    try:
        from lmcache.v1.multiprocess.custom_types import RequestAllocationRecord
    except ImportError:
        from lmcache.v1.multiprocess.custom_types import (
            BlockAllocationRecord as RequestAllocationRecord,
        )
except ImportError:
    from lmcache.v1.multiprocess.custom_types import (
        BlockAllocationRecord as RequestAllocationRecord,
    )

    from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration import (
        LMCacheMPSchedulerAdapter,
        LMCacheMPWorkerAdapter,
        LoadStoreOp,
        ParallelStrategy,
    )

if TYPE_CHECKING:
    from vllm.distributed.kv_events import KVCacheEvent
    from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
        KVConnectorPromMetrics,
        KVConnectorStats,
        PromMetric,
        PromMetricT,
    )
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.kv_cache_utils import BlockHash
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = lmcache_init_logger(__name__)


# Helper functions
def extract_world_size_and_kv_rank(
    world_size: int,
    rank: int,
    vllm_config: VllmConfig,
) -> tuple[int, int]:
    """
    Convert the rank for the MLA.
    """
    use_mla = mla_enabled(vllm_config.model_config)
    if not use_mla:
        return world_size, rank
    else:
        # Tensor parallel does not change the KV caches for MLA models.
        # So we need to "exclude" the effect of TP on rank and world size
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        # vLLM constructs TP groups first, and then construct other
        # parallel groups on top of TP groups.
        # for example, TP=4, PP=2,
        # PP group: [0, 1, 2, 3], [4, 5, 6, 7]
        # TP group: [0, 4], [1, 5], [2, 6], [3, 7]
        # So we can "exclude" the effect of TP by rank // tp_size.
        return world_size // tp_size, rank // tp_size


def create_scheduler_adapter(
    server_url: str,
    zmq_context: zmq.Context,
    vllm_config: VllmConfig,
    mq_timeout: float,
    heartbeat_interval: float,
) -> LMCacheMPSchedulerAdapter:
    world_size, kv_rank = extract_world_size_and_kv_rank(
        vllm_config.parallel_config.world_size,
        vllm_config.parallel_config.rank,
        vllm_config,
    )
    parallel_strategy = ParallelStrategy(
        mla_enabled(vllm_config.model_config),
        world_size,
        kv_rank,
        vllm_config.parallel_config.world_size,
        vllm_config.parallel_config.rank,
        vllm_config.parallel_config.tensor_parallel_size,
        vllm_config.parallel_config.pipeline_parallel_size,
    )

    return LMCacheMPSchedulerAdapter(
        server_url=server_url,
        context=zmq_context,
        model_name=vllm_config.model_config.model,
        vllm_block_size=vllm_config.cache_config.block_size,
        parallel_strategy=parallel_strategy,
        mq_timeout=mq_timeout,
        heartbeat_interval=heartbeat_interval,
    )


def create_worker_adapter(
    server_url: str,
    zmq_context: zmq.Context,
    vllm_config: VllmConfig,
    mq_timeout: float,
    heartbeat_interval: float,
) -> LMCacheMPWorkerAdapter:
    world_size, kv_rank = extract_world_size_and_kv_rank(
        vllm_config.parallel_config.world_size,
        vllm_config.parallel_config.rank,
        vllm_config,
    )
    parallel_strategy = ParallelStrategy(
        mla_enabled(vllm_config.model_config),
        world_size,
        kv_rank,
        vllm_config.parallel_config.world_size,
        vllm_config.parallel_config.rank,
        vllm_config.parallel_config.tensor_parallel_size,
        vllm_config.parallel_config.pipeline_parallel_size,
    )

    return LMCacheMPWorkerAdapter(
        server_url=server_url,
        context=zmq_context,
        model_name=vllm_config.model_config.model,
        vllm_block_size=vllm_config.cache_config.block_size,
        parallel_strategy=parallel_strategy,
        mq_timeout=mq_timeout,
        heartbeat_interval=heartbeat_interval,
    )


class LMCacheMPRequestState(enum.Enum):
    """
    State machine:
    PREFETCHING -- update_state_after_alloc --> WAITING_FOR_LOAD
    WAITING_FOR_LOAD -- process_loading_requests --> READY
    """

    PREFETCHING = enum.auto()
    WAITING_FOR_LOAD = enum.auto()
    READY = enum.auto()


@dataclass
class LMCacheMPRequestTracker:
    # NOTE: this class used vLLM data structures, should be part of
    # vLLM integration code

    request_id: str

    # Read-only lists to track the token ids and block hashes
    all_token_ids: ConstantList[int]
    block_hashes: ConstantList["BlockHash"]

    # Per-vLLM-kv-cache-group block IDs allocated to this request.
    # The dict is keyed by the engine-side ``kv_cache_group_id``, and
    # each value is the (in-order) list of block IDs allocated for that
    # group. For non-hybrid models, only key ``0`` is ever populated —
    # the dict has a single entry that mirrors the prior flat-list
    # semantics. For DeepSeek-V4 with the hybrid manager active, vLLM
    # emits 5 disjoint namespaces (one per ``KVCacheGroupSpec``) and
    # each gid has its own list here.
    #
    # Per-gid list lengths are NOT equal: each gid has its own
    # ``KVCacheSpec.block_size`` and vLLM extends each gid's list by
    # ``cdiv(scheduled_tokens, gid.block_size)`` per scheduling event.
    # All gids advance in lockstep at the *coarse* (scheduler) block
    # boundary — i.e. after vLLM allocates a coarse chunk of
    # ``scheduler_block_size`` tokens, every gid's list grows by the
    # number of gid-blocks that span that coarse chunk. Coarse-block
    # indexing in :class:`LMCacheMPRequestMetadata` relies on this
    # alignment.
    allocated_block_ids: dict[int, list[int]] = field(default_factory=dict)

    # Number of scheduled tokens in this request. We keep tracking this to
    # avoid saving half-full blocks.
    num_scheduled_tokens: int = 0

    # Number of blocks stored will be initialized when lookup the external
    # hit tokens and will be updated when processing new requests and cached
    # requests.
    num_stored_blocks: int = 0

    # Staging load operation -- save vllm and lmcache hit tokens during lookup
    num_vllm_hit_blocks: int = 0
    num_lmcache_hit_blocks: int = 0

    # Main state
    state: LMCacheMPRequestState = LMCacheMPRequestState.PREFETCHING

    cache_salt: str = ""

    def __init__(self, request: "Request"):
        self.request_id = request.request_id
        self.cache_salt: str = request.cache_salt or ""
        self.all_token_ids = request.all_token_ids
        self.block_hashes = ConstantList(request.block_hashes)
        self.allocated_block_ids = {}
        self.num_stored_blocks = 0
        self.num_vllm_hit_blocks = 0
        self.num_lmcache_hit_blocks = 0
        self.state = LMCacheMPRequestState.PREFETCHING

    ####
    # Check the state of the request
    ####
    def needs_retrieve(self) -> bool:
        """Check whether the current request needs retrieve, will be used
        update_stage_after_alloc"""
        return (
            self.num_lmcache_hit_blocks > self.num_vllm_hit_blocks
            and self.state != LMCacheMPRequestState.READY
        )

    def is_ready_for_retrieving(self) -> bool:
        """Check whether the current request is ready for retrieving,
        will be used in process_loading_requests"""
        return (
            self.state == LMCacheMPRequestState.WAITING_FOR_LOAD
            and self.needs_retrieve()
        )

    ####
    # Update internal states
    ####
    def increase_num_scheduled_tokens(self, num_new_tokens: int):
        self.num_scheduled_tokens += num_new_tokens

    def increase_num_stored_blocks(self, num_new_blocks: int):
        """Increase the number of stored blocks for the current request
        This function will be called when processing the cached requests.
        """
        self.num_stored_blocks += num_new_blocks

    def append_block_ids_per_group(
        self,
        new_block_ids_per_group: tuple[list[int], ...],
    ) -> None:
        """Append per-gid block IDs allocated by vLLM since the last call.

        ``new_block_ids_per_group`` is the structure vLLM hands us via
        :meth:`KVCacheBlocks.get_block_ids` (or the matching
        ``cached_reqs.new_block_ids[idx]`` slice): one list per
        ``KVCacheGroupSpec``, in scheduler-fixed order. Each per-gid
        list is appended to the matching slot in
        :attr:`allocated_block_ids`, creating new gid entries on
        first sight.

        For non-hybrid models the tuple has length 1 and only key 0 is
        ever populated, so this collapses to the prior single-list
        ``extend`` semantics.

        Args:
            new_block_ids_per_group: Per-gid lists of newly allocated
                block IDs. Empty inner lists are allowed (gid had no
                new blocks this step) and are silently no-oped.
        """
        for gid, group_block_ids in enumerate(new_block_ids_per_group):
            if not group_block_ids:
                continue
            self.allocated_block_ids.setdefault(gid, []).extend(group_block_ids)

    def num_allocated_blocks_per_group(self) -> dict[int, int]:
        """Return ``{gid: len(allocated_block_ids[gid])}`` for every
        gid that has been touched. Used by callers that need to know
        per-gid block counts (e.g. to compute slice boundaries or
        avoid double-appending when ``update_state_after_alloc`` is
        called twice for an async-load request)."""
        return {gid: len(blocks) for gid, blocks in self.allocated_block_ids.items()}

    def total_allocated_blocks(self) -> int:
        """Return the total block count across all gids. Useful for
        legacy log lines that report a single number; do not use as a
        chunking-grid quantity (each gid has its own scheduler block
        size).

        For non-hybrid models this equals
        ``len(allocated_block_ids[0])``; for hybrid models it is the
        sum of per-gid counts and reflects how many bytes vLLM has
        materialised in total, *not* the number of token-aligned
        chunks LMCache should consider.
        """
        return sum(len(blocks) for blocks in self.allocated_block_ids.values())

    ####
    # For debugging
    ####
    def __repr__(self) -> str:
        return (
            f"LMCacheMPRequestTracker(request_id={self.request_id}, "
            f"num_tokens={len(self.all_token_ids)}, "
            f"num_block_hashes={len(self.block_hashes)}, "
            f"num_allocated_blocks_per_group={self.num_allocated_blocks_per_group()}, "
            f"num_stored_blocks={self.num_stored_blocks}, "
            f"vllm_hit_blocks={self.num_vllm_hit_blocks}, "
            f"lmcache_hit_blocks={self.num_lmcache_hit_blocks}, "
            f"state={self.state})"
        )

    def __str__(self) -> str:
        return self.__repr__()


@dataclass
class LMCacheMPRequestMetadata:
    request_id: str
    direction: Literal["STORE", "RETRIEVE"]
    op: LoadStoreOp
    cache_salt: str = ""

    @staticmethod
    def _per_gid_slice(
        tracker: "LMCacheMPRequestTracker",
        gid_to_block_size: dict[int, int],
        vllm_block_size: int,
        start: int,
        end: int,
    ) -> list[list[int]]:
        """Slice ``tracker.allocated_block_ids`` per gid for the
        coarse range ``[start, end)`` (units: ``vllm_block_size``).

        The fundamental relation: gid ``g``'s block ID grid has
        granularity ``gid_block_size_g``; vLLM's ``cache_config
        .block_size`` is the GCD of all gid block sizes (= 4 on V4
        HMA, 256 on non-hybrid). One coarse-block range of
        ``[start, end)`` covers ``(end - start) * vllm_block_size``
        tokens, which translates to a per-gid range of
        ``(end - start) * vllm_block_size / gid_block_size_g``
        gid-block IDs.

        When ``gid_block_size_g >= vllm_block_size`` (e.g. gid 0 on
        V4: 256 vs 4) we *divide* by ``gid_block_size_g //
        vllm_block_size``; the legacy ``multiplier = vllm_block_size
        // gid_block_size_g`` evaluates to 0 here and silently
        emits an empty slice — which trips the LMCache server's
        per-namespace kernel constraint check.

        When ``gid_block_size_g <= vllm_block_size`` (e.g. gid 3 on
        V4: 4 vs 4) we *multiply*. When the two are equal, both
        branches produce the same answer.

        Args:
            tracker: The request tracker holding per-gid block-ID
                lists.
            gid_to_block_size: Mapping from gid to that gid's
                ``KVCacheSpec.block_size``.
            vllm_block_size: ``cache_config.block_size``, the GCD
                grain at which ``start`` and ``end`` are expressed.
            start: First coarse block (inclusive), in
                ``vllm_block_size`` units.
            end: Last coarse block (exclusive), same units.

        Returns:
            One list per gid (in ``sorted(gid_to_block_size.keys())``
            order) holding that gid's block-ID slice for
            ``[start, end)``.
        """
        block_ids_per_group: list[list[int]] = []
        for gid in sorted(gid_to_block_size.keys()):
            gid_bs = gid_to_block_size[gid]
            if gid_bs >= vllm_block_size:
                ratio = gid_bs // vllm_block_size
                gid_start = start // ratio
                gid_end = end // ratio
            else:
                ratio = vllm_block_size // gid_bs
                gid_start = start * ratio
                gid_end = end * ratio
            gid_blocks = tracker.allocated_block_ids.get(gid, [])
            block_ids_per_group.append(gid_blocks[gid_start:gid_end])
        return block_ids_per_group

    @staticmethod
    def GetStoreMetadata(
        tracker: LMCacheMPRequestTracker,
        blocks_in_chunk: int,
        vllm_block_size: int,
        gid_to_block_size: dict[int, int],
    ) -> "LMCacheMPRequestMetadata | None":
        """
        Generate the store metadata for the current request tracker.

        Args:
            tracker: The request tracker to generate the metadata from.
            blocks_in_chunk: the number of blocks in a LMCache data chunk
                (``lmcache_chunk_size // vllm_block_size`` — coarse-block
                units).
            vllm_block_size: the scheduler block size used in vLLM (the
                ``--block-size`` CLI flag), equal to
                ``lcm(group.block_size for group in kv_cache_groups)``
                under HMA.
            gid_to_block_size: ``{gid: KVCacheSpec.block_size}`` for
                every kv_cache_group exposed to the connector. Used to
                translate coarse-block ranges into per-gid slice ranges.
                For non-hybrid models this is ``{0: vllm_block_size}``.
        """
        # Store the blocks that has block hashes
        # NOTE: the invariant here is that `num_stored_blocks` should
        # always be a multiple of `blocks_in_chunk`
        # TODO: This should be checked everytime we update the num_stored_blocks
        #
        # Why computed_blocks uses max(num_vllm_hit_blocks, num_lmcache_hit_blocks):
        #
        # Both values represent a prefix of blocks whose KV data is already
        # available (either from vLLM APC or from LMCache), so they must NOT
        # be summed (that would double-count the overlapping prefix).
        #
        # * num_lmcache_hit_blocks: LMCache-hit blocks are already counted in
        #   num_stored_blocks (set during lookup), so they must be included
        #   here to keep the upper bound consistent.  They are NOT re-stored.
        # * num_vllm_hit_blocks: LMCache stores in units of chunks (N blocks),
        #   so num_lmcache_hit_blocks is rounded DOWN to the nearest chunk
        #   boundary.  When vLLM APC hits more blocks than that rounded value
        #   (e.g. APC=44 blocks, LMCache=32 blocks after chunk alignment),
        #   using only num_lmcache_hit_blocks would set the upper bound too
        #   low and silently skip the APC-hit blocks that fall between the
        #   two values, causing under-storing.  Taking the max ensures we
        #   always use the tighter (larger) of the two hit counts.
        computed_blocks = tracker.num_scheduled_tokens // vllm_block_size + max(
            tracker.num_vllm_hit_blocks, tracker.num_lmcache_hit_blocks
        )
        # Coarse-block count of the request: smallest gid count after
        # normalising each per-gid list length back to coarse units. Any
        # gid whose ``KVCacheSpec.block_size`` divides
        # ``vllm_block_size`` evenly contributes a coarse count of
        # ``len(allocated_block_ids[gid]) * gid_block_size //
        # vllm_block_size``. Take the min as a conservative bound — if
        # vLLM ever returns lists of inconsistent coarse counts (a bug),
        # we'd silently store half-aligned data otherwise.
        per_gid_lengths = tracker.num_allocated_blocks_per_group()
        if per_gid_lengths:
            coarse_block_count = min(
                length * gid_to_block_size[gid] // vllm_block_size
                for gid, length in per_gid_lengths.items()
            )
        else:
            coarse_block_count = 0
        min_available_blocks = min(
            len(tracker.block_hashes),
            coarse_block_count,
            computed_blocks,
        )
        num_staging_blocks = min_available_blocks - tracker.num_stored_blocks
        num_chunks = num_staging_blocks // blocks_in_chunk

        if num_chunks >= 1:
            start = tracker.num_stored_blocks
            end = start + num_chunks * blocks_in_chunk
            # Build per-gid block-ID slice. ``_per_gid_slice`` handles
            # both ``gid_bs >= vllm_block_size`` (V4 hybrid main /
            # indexer-k / SWA-64 against vllm_bs=4) and
            # ``gid_bs <= vllm_block_size`` (non-hybrid: vllm_bs=256
            # equals gid 0's bs=256, multiplier=1).
            block_ids_per_group = LMCacheMPRequestMetadata._per_gid_slice(
                tracker, gid_to_block_size, vllm_block_size, start, end
            )
            start_token_idx = start * vllm_block_size
            end_token_idx = end * vllm_block_size
            token_ids = list(tracker.all_token_ids)
            op = LoadStoreOp(
                token_ids=token_ids,
                block_ids=block_ids_per_group,
                start=start_token_idx,
                end=end_token_idx,
            )

            ret = LMCacheMPRequestMetadata(
                request_id=tracker.request_id,
                direction="STORE",
                op=op,
                cache_salt=tracker.cache_salt,
            )

            # Update the request tracker
            tracker.increase_num_stored_blocks(end - start)
            return ret

        return None

    @staticmethod
    def GetRetrieveMetadata(
        tracker: LMCacheMPRequestTracker,
        blocks_in_chunk: int,
        vllm_block_size: int,
        gid_to_block_size: dict[int, int],
    ) -> "LMCacheMPRequestMetadata | None":
        """
        Generate the retrieve metadata for the current request tracker.

        Args:
            tracker: The request tracker to generate the metadata from.
            blocks_in_chunk: the number of blocks in a LMCache data chunk
                (coarse-block units).
            vllm_block_size: the scheduler block size used in vLLM.
            gid_to_block_size: ``{gid: KVCacheSpec.block_size}`` for
                every kv_cache_group exposed to the connector. See
                :meth:`GetStoreMetadata` for usage.
        """
        if not tracker.is_ready_for_retrieving():
            return None

        # |---------------------|-----------------|----------------|
        # | num_vllm_hit_blocks |
        # | lmcache chunk 1   | lmcache chunk 2   |
        #                     |  need to retrieve |

        start = tracker.num_vllm_hit_blocks // blocks_in_chunk * blocks_in_chunk
        end = tracker.num_lmcache_hit_blocks
        assert end % blocks_in_chunk == 0, (
            "The number of LMCache hit blocks should be a multiple of the "
            "number of blocks in a lmcache chunk. "
        )
        assert len(tracker.block_hashes) >= end, (
            "The number of block hashes should be greater than or equal to the "
            "number of LMCache hit blocks. "
        )
        if end > start:
            block_ids_per_group = LMCacheMPRequestMetadata._per_gid_slice(
                tracker, gid_to_block_size, vllm_block_size, start, end
            )
            start_token_idx = start * vllm_block_size
            end_token_idx = end * vllm_block_size
            token_ids = list(tracker.all_token_ids)

            # Compute how many tokens at the start of the retrieve range
            # overlap with APC-shared blocks. The server must skip writing
            # to these positions to avoid a cross-stream data race: the
            # retrieve writes on the LMCache CUDA stream while concurrent
            # requests may read these APC-shared blocks on the vLLM stream.
            apc_overlap_blocks = tracker.num_vllm_hit_blocks - start
            skip_first_n_tokens = apc_overlap_blocks * vllm_block_size

            op = LoadStoreOp(
                token_ids=token_ids,
                block_ids=block_ids_per_group,
                start=start_token_idx,
                end=end_token_idx,
                skip_first_n_tokens=skip_first_n_tokens,
            )

            ret = LMCacheMPRequestMetadata(
                request_id=tracker.request_id,
                direction="RETRIEVE",
                op=op,
                cache_salt=tracker.cache_salt,
            )
            return ret

        return None


class LMCacheMPConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        super().__init__()
        self.requests: list[LMCacheMPRequestMetadata] = []

    def add_request_metadata(self, request_metadata: LMCacheMPRequestMetadata):
        self.requests.append(request_metadata)

    def __len__(self):
        return len(self.requests)

    # For debugging
    def __str__(self):
        request_strs = []
        for req_meta in self.requests:
            request_strs.append(
                f"RequestMetadata(request_id={req_meta.request_id}, "
                f"direction={req_meta.direction}, "
                f"num_blocks={len(req_meta.op)}, "
                f"block_ids={req_meta.op.block_ids})"
            )
        return "[" + "\n".join(request_strs) + "]"

    def __repr__(self):
        return self.__str__()


class LMCacheMPConnector(KVConnectorBase_V1, SupportsHMA):
    """
    The connector for LMCache multi-process mode.

    Inherits :class:`SupportsHMA` so the scheduler will route per-group
    block-id tuples through :meth:`request_finished_all_groups`. Note that
    declaring ``SupportsHMA`` does NOT by itself flip vLLM's hybrid-mode
    default — :func:`vllm.config.vllm.VllmConfig._verify_kv_cache_config`
    still auto-disables HMA whenever ``kv_transfer_config`` is set unless
    the user explicitly passes ``--no-disable-hybrid-kv-cache-manager``.
    What inheriting buys us is: with that flag set, the scheduler stops
    asserting ``len(kv_cache_groups) == 1`` at the request-finished site
    and we get the full per-group block-id tuple instead.

    Per-group threading through STORE/RETRIEVE is NOT yet wired up
    (deferred to a follow-up). With HMA on, store/retrieve will currently
    crash or produce wrong data; this commit only turns on observability
    so we can see what arrives at registration.

    Extra configs (kv_transfer_config.extra_config):
    - lmcache.mp.host: the host of the LMCache server.
    - lmcache.mp.port: the port of the LMCache server.
    - lmcache.mp.mq_timeout: timeout (seconds) for message queue requests.
    - lmcache.mp.heartbeat_interval: interval (seconds) between server
      heartbeat pings.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        assert vllm_config.kv_transfer_config is not None
        server_host = vllm_config.kv_transfer_config.get_from_extra_config(
            "lmcache.mp.host", "tcp://localhost"
        )
        server_port = vllm_config.kv_transfer_config.get_from_extra_config(
            "lmcache.mp.port", 5555
        )
        mq_timeout = float(
            vllm_config.kv_transfer_config.get_from_extra_config(
                "lmcache.mp.mq_timeout", 300.0
            )
        )
        heartbeat_interval = float(
            vllm_config.kv_transfer_config.get_from_extra_config(
                "lmcache.mp.heartbeat_interval", 10.0
            )
        )

        server_url = f"{server_host}:{server_port}"
        zmq_context = zmq.Context.instance()
        if self.role == KVConnectorRole.SCHEDULER:
            self.scheduler_adapter = create_scheduler_adapter(
                server_url,
                zmq_context,
                vllm_config,
                mq_timeout,
                heartbeat_interval,
            )
            self.request_trackers: dict[str, LMCacheMPRequestTracker] = {}
        elif self.role == KVConnectorRole.WORKER:
            self.worker_adapter = create_worker_adapter(
                server_url,
                zmq_context,
                vllm_config,
                mq_timeout,
                heartbeat_interval,
            )
        else:
            raise ValueError(f"Unknown KVConnectorRole: {self.role}")

        self.vllm_block_size = vllm_config.cache_config.block_size

        # Per-gid block_size lookup: ``self._kv_cache_config.kv_cache_groups``
        # exposes one ``KVCacheSpec.block_size`` per group. For non-hybrid
        # models there is exactly one group with ``block_size ==
        # vllm_block_size``. For DeepSeek-V4 with the hybrid manager
        # active there are 5 groups with mixed block sizes (256 / 64 /
        # 64 / 4 / 8 in our verified probe). The metadata generators
        # use this to translate coarse-block indices (in
        # ``vllm_block_size`` units) into per-gid slice ranges.
        self._gid_to_block_size: dict[int, int] = {}
        kv_cache_config = getattr(self, "_kv_cache_config", None)
        if kv_cache_config is not None and getattr(
            kv_cache_config, "kv_cache_groups", None
        ):
            for gid, group in enumerate(kv_cache_config.kv_cache_groups):
                spec = getattr(group, "kv_cache_spec", None)
                spec_block_size = getattr(spec, "block_size", None)
                if spec_block_size is not None:
                    self._gid_to_block_size[gid] = int(spec_block_size)
        if not self._gid_to_block_size:
            # Single-group fallback: pretend gid 0 exists at the
            # scheduler block size. Keeps non-hybrid models on the
            # same code path (one gid, multiplier 1).
            self._gid_to_block_size = {0: self.vllm_block_size}

    @property
    def role(self) -> KVConnectorRole:
        return self._role

    # ==============================
    # Worker-side methods
    # ==============================

    def _get_connector_metadata(self) -> KVConnectorMetadata:
        """Get the connector metadata.

        This function should only be called inside the connector.

        Returns:
            ConnectorMetadata: the connector metadata.
        """

        # Should only be called while set to valid metadata.
        assert self._connector_metadata is not None
        return self._connector_metadata

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """
        Initialize with the KV caches. Useful for pre-registering the
        KV Caches in the KVConnector (e.g. for NIXL).

        Args:
            kv_caches: dictionary of layer names, kv cache
        """
        logger.info("Registering kv caches!")

        # Build per-positional-layer logical (scheduler) block size and
        # block-ID namespace from ``self._kv_cache_config.kv_cache_groups``.
        # Under hybrid KV cache management, each ``KVCacheGroupSpec``
        # reports its own ``KVCacheSpec.block_size`` and contributes its
        # own ``BlockPool`` slice; layers from different groups can
        # share the same physical tensor shape (via layer-tuple
        # aliasing) but were allocated against different scheduler
        # block sizes, and even when both block sizes match they may
        # still pull block IDs from disjoint pools (e.g. DeepSeek-V4's
        # vLLM gids 1 and 2 — the even+MTP and odd SWA layers, which
        # share every spec field but have separate ``req_to_blocks``
        # dicts).
        #
        # LMCache needs both signals to keep one transfer-kernel
        # dispatch unit per ``(physical_bs, logical_bs, namespace)``
        # triple. Without ``logical_bs`` it would merge the C4A-main
        # and per-layer-SWA layers (both physical ``shape[1] = 64``);
        # without ``namespace`` it would merge gids 1 and 2 even after
        # the logical_bs split. See
        # :class:`~lmcache.v1.gpu_connector.utils.LayoutHints`'s
        # ``per_layer_logical_block_size`` and
        # ``per_layer_kv_cache_group_id`` fields for the consumer side.
        #
        # When the engine produced only one ``KVCacheGroupSpec`` (the
        # non-hybrid case), every layer ends up with the same block
        # size and the same namespace, which collapses LMCache's
        # grouping back to the prior 5-tuple identity behavior.
        #
        # If a layer name in any group does not appear in the
        # registered ``kv_caches`` dict (i.e. that layer's KV cache is
        # not exposed to the connector), it's silently skipped — its
        # entry in the per-layer lists stays at the sentinel value
        # (logical_bs=0, namespace=-1). The LMCache-side consumer
        # rejects any non-positive logical_bs entry or negative
        # namespace entry with ``ValueError`` at registration, so a
        # partial cover surfaces as a loud registration failure rather
        # than silent miscount. If neither hint is populated we drop
        # both entirely so single-group engines fall back to the prior
        # 5-tuple identity behavior.
        per_layer_logical_block_size: list[int] | None = None
        per_layer_kv_cache_group_id: list[int] | None = None
        kv_cache_config = getattr(self, "_kv_cache_config", None)
        if kv_cache_config is not None and getattr(
            kv_cache_config, "kv_cache_groups", None
        ):
            kv_cache_layer_names = list(kv_caches.keys())
            name_to_idx = {name: idx for idx, name in enumerate(kv_cache_layer_names)}
            num_positional = len(kv_cache_layer_names)
            per_layer_logical_block_size = [0] * num_positional
            per_layer_kv_cache_group_id = [-1] * num_positional
            for gid, group in enumerate(kv_cache_config.kv_cache_groups):
                spec = getattr(group, "kv_cache_spec", None)
                spec_block_size = getattr(spec, "block_size", None)
                if spec_block_size is None:
                    continue
                for name in group.layer_names:
                    pos = name_to_idx.get(name)
                    if pos is not None:
                        per_layer_logical_block_size[pos] = int(spec_block_size)
                        per_layer_kv_cache_group_id[pos] = gid
            # If neither list got populated (no usable specs), drop
            # both hints so the consumer falls back to the prior
            # behavior cleanly. Layers that ended up with sentinel
            # values (some specs missing block_size, some layer names
            # absent from kv_caches) will trigger a loud ValueError
            # on the LMCache side.
            if not any(per_layer_logical_block_size) and all(
                ns == -1 for ns in per_layer_kv_cache_group_id
            ):
                per_layer_logical_block_size = None
                per_layer_kv_cache_group_id = None

        extra_layout_hints: dict[str, object] | None = None
        if (
            per_layer_logical_block_size is not None
            or per_layer_kv_cache_group_id is not None
        ):
            extra_layout_hints = {}
            if per_layer_logical_block_size is not None:
                extra_layout_hints["per_layer_logical_block_size"] = (
                    per_layer_logical_block_size
                )
            if per_layer_kv_cache_group_id is not None:
                extra_layout_hints["per_layer_kv_cache_group_id"] = (
                    per_layer_kv_cache_group_id
                )
        self.worker_adapter.register_kv_caches(
            kv_caches, extra_layout_hints=extra_layout_hints
        )
        return

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """
        Start loading the KV cache from the connector to vLLM's paged
        KV buffer. This is called from the forward context before the
        forward pass to enable async loading during model execution.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.

        """
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, LMCacheMPConnectorMetadata)

        request_ids = []
        ops = []
        cache_salts = []

        for meta in metadata.requests:
            if meta.direction != "RETRIEVE":
                continue
            request_ids.append(meta.request_id)
            ops.append(meta.op)
            cache_salts.append(meta.cache_salt)

        if len(request_ids) == 0:
            return

        with torch.cuda.stream(torch.cuda.current_stream()):
            event = torch.cuda.Event(interprocess=True)
            event.record()

        self.worker_adapter.batched_submit_retrieve_requests(
            request_ids, ops, event, cache_salts=cache_salts
        )

    def wait_for_layer_load(self, layer_name: str) -> None:
        """
        Block until the KV for a specific layer is loaded into vLLM's
        paged buffer. This is called from within attention layer to ensure
        async copying from start_load_kv is complete.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        """
        Start saving a layer of KV cache from vLLM's paged buffer
        to the connector. This is called from within attention layer to
        enable async copying during execution.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        return

    def wait_for_save(self):
        """
        Block until all the save operations is done. This is called
        as the forward context exits to ensure that the async saving
        from save_kv_layer is complete before finishing the forward.

        This prevents overwrites of paged KV buffer before saving done.
        """
        # In MLA scenario, only the first rank of the pipeline group
        # needs to save the KV cache.
        if (
            self.worker_adapter.use_mla
            and not self.worker_adapter.is_first_rank_of_pp_group
        ):
            return

        metadata = self._get_connector_metadata()
        assert isinstance(metadata, LMCacheMPConnectorMetadata)

        request_ids = []
        ops = []
        cache_salts = []
        for meta in metadata.requests:
            if meta.direction != "STORE":
                continue
            request_ids.append(meta.request_id)
            ops.append(meta.op)
            cache_salts.append(meta.cache_salt)

        if len(request_ids) == 0:
            return

        with torch.cuda.stream(torch.cuda.current_stream()):
            event = torch.cuda.Event(interprocess=True)
            event.record()

        self.worker_adapter.batched_submit_store_requests(
            request_ids, ops, event, cache_salts=cache_salts
        )

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens on the worker.
        The scheduler process (via the Executors) will use this output
        to track which workers are done.

        Returns:
            ids of requests that have finished asynchronous transfer
            (requests that previously returned True from request_finished()),
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """
        val = self.worker_adapter.get_finished(finished_req_ids)
        # logger.error("Finished req ids: %s, %s", val[0], val[1])
        return val

    def get_block_ids_with_load_errors(self) -> set[int]:
        """
        Get the set of block IDs that failed to load.

        Returns:
            Set of block IDs that encountered load errors.
            Empty set if no load errors occurred.

        Notes:
            - Applies to both sync- and async-loading requests.
            - Async loading: failed blocks may be reported in any forward pass
              up to and including the pass where the request ID is returned by
              `get_finished()`. Even if failures occur, the request must still
              be reported via `get_finished()`, and the failed block IDs must
              appear here no later than that same pass.
            - Sync loading: failed blocks should be reported in the forward
              pass in which they are detected.
        """
        return self.worker_adapter.get_block_ids_with_load_errors()

    def shutdown(self):
        """
        Shutdown the connector. This is called when the worker process
        is shutting down to ensure that all the async operations are
        completed and the connector is cleaned up properly.
        """
        if hasattr(self, "worker_adapter"):
            self.worker_adapter.shutdown()
        return None

    def get_kv_connector_stats(self) -> "KVConnectorStats | None":
        """
        Get the KV connector stats collected during the last interval.
        """
        return None

    # ==============================
    # Scheduler-side methods
    # ==============================

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            A tuple with the following elements:
                - An optional number of tokens that can be loaded from the
                  external KV cache beyond what is already computed.
                  If None, it means that the connector needs more time to
                  determine the number of matched tokens, and the scheduler
                  should query for this request again later.
                - `True` if external KV cache tokens will be loaded
                  asynchronously (between scheduler steps). Must be
                  'False' if the first element is 0.

        Notes:
            The connector should only consider the largest prefix of prompt-
            tokens for which KV cache is actually available at the time of the
            call. If the cache cannot be loaded for some tokens (e.g., due to
            connectivity issues or eviction), those tokens must not be taken
            into account.
        """
        tracker = self._get_or_create_request_tracker(request)
        # TODO: support loading KV for preempted requests in the future
        if request.status == RequestStatus.PREEMPTED:
            return 0, False

        self.scheduler_adapter.maybe_submit_lookup_request(
            request.request_id,
            token_ids=list(request.all_token_ids),
            cache_salt=tracker.cache_salt,
        )

        ret = self.scheduler_adapter.check_lookup_result(request.request_id)
        if ret is None:
            return None, True

        if ret == 0:
            return 0, False

        assert (
            ret % (self.scheduler_adapter.num_blocks_per_chunk() * self.vllm_block_size)
            == 0
        )

        # Update num stored blocks for the tracker
        num_vllm_blocks = num_computed_tokens // self.vllm_block_size
        num_lmcache_blocks = ret // self.vllm_block_size
        tracker.increase_num_stored_blocks(num_lmcache_blocks)

        # Save the vllm and lmcache hit tokens
        tracker.num_vllm_hit_blocks = num_vllm_blocks
        tracker.num_lmcache_hit_blocks = num_lmcache_blocks

        need_to_load = max(0, ret - num_computed_tokens)
        logger.debug(
            "vLLM hit is: %d, Need to load is %d", num_computed_tokens, need_to_load
        )
        return need_to_load, need_to_load > 0

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """
        Update KVConnector state after block allocation.

        If get_num_new_matched_tokens previously returned True for a
        request, this function may be called twice for that same request -
        first when blocks are allocated for the connector tokens to be
        asynchronously loaded into, and second when any additional blocks
        are allocated, after the load/transfer is complete.

        Args:
            request (Request): the request object.
            blocks (KVCacheBlocks): the blocks allocated for the request.
            num_external_tokens (int): the number of tokens that will be
                loaded from the external KV cache.
        """
        # NOTE: `blocks` comes from kv_cache_manager.get_blocks(request_id),
        # which returns ALL blocks for the request (not just newly allocated).
        # This function may be called twice for async-load requests:
        #   1st call: blocks = initial allocation (APC + fresh)
        #   2nd call: blocks = all blocks
        #  (initial + newly allocated for remaining tokens)
        # We must only append the NEW blocks beyond what's already tracked
        # to avoid duplication, which would corrupt the store path's block indexing.
        tracker = self._get_request_tracker(request.request_id)
        per_gid_block_ids = blocks.get_block_ids()  # tuple[list[int], ...]

        # Only append blocks beyond what's already tracked, per-gid.
        # ``KVCacheBlocks.get_block_ids()`` returns ALL blocks vLLM has
        # allocated for the request so far; on the second call (async
        # load completion) some prefix has already been recorded on
        # this tracker and must not be re-appended. Per-gid existing
        # counts let us advance each gid independently.
        existing_per_gid = tracker.num_allocated_blocks_per_group()
        new_per_gid: list[list[int]] = []
        for gid, gid_blocks in enumerate(per_gid_block_ids):
            existing = existing_per_gid.get(gid, 0)
            new_per_gid.append(list(gid_blocks[existing:]))
        if any(new_per_gid):
            tracker.append_block_ids_per_group(tuple(new_per_gid))

        # Update the state of the tracker
        condition = tracker.needs_retrieve()
        if tracker.state == LMCacheMPRequestState.PREFETCHING:
            # If need to retrieve, change to WAITING_FOR_LOAD
            # Otherwise, change to READY
            tracker.state = (
                LMCacheMPRequestState.WAITING_FOR_LOAD
                if condition
                else LMCacheMPRequestState.READY
            )
            # Clean up lookup future in scheduler adapter
            self.scheduler_adapter.cleanup_lookup_result(request.request_id)

            # Free locks on chunks that vLLM already computed and won't
            # retrieve from LMCache.
            if tracker.num_lmcache_hit_blocks > 0:
                if not condition:
                    # No retrieve needed — free ALL locked chunks
                    free_end = tracker.num_lmcache_hit_blocks * self.vllm_block_size
                else:
                    # Note(Roy): Boundary misalignment between vLLM blocks and LMCache
                    # blocks is handled in free_lookup_locks. It makes sure that if
                    # the last vLLM computed block ends in the middle of a LMCache
                    # block, the end LMCache block is not freed (i.e., floor division)
                    # since it will still be needed by vLLM and such block's lock will
                    # be freed by vLLM's retrieve.
                    free_end = tracker.num_vllm_hit_blocks * self.vllm_block_size

                if free_end > 0:
                    self.scheduler_adapter.free_lookup_locks(
                        token_ids=list(tracker.all_token_ids),
                        start=0,
                        end=free_end,
                        request_id=request.request_id,
                    )
                    logger.debug(
                        "Free locks of tokens %d-%d since it is cached by vLLM.",
                        0,
                        free_end,
                    )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """
        Build the connector metadata for this step.

        This function should NOT modify fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        metadata = LMCacheMPConnectorMetadata()

        self._process_retrieve_requests(metadata)
        self._process_new_requests(scheduler_output, metadata)
        self._process_cached_requests(scheduler_output, metadata)

        if len(metadata) > 0:
            logger.debug("Final connector metadata: %s", metadata)

        # Report block allocation deltas to LMCache for observability
        self._report_block_allocation_deltas(scheduler_output)

        return metadata

    def update_connector_output(self, connector_output: KVConnectorOutput):
        """
        Update KVConnector state from worker-side connectors output.

        Args:
            connector_output (KVConnectorOutput): the worker-side
                connectors output.
        """
        return

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Called exactly once when a request has finished, before its blocks are
        freed.

        The connector may assumes responsibility for freeing the blocks
        asynchronously by returning True.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """

        params: dict[str, Any] | None = getattr(request, "kv_transfer_params", None)
        return_params: dict[str, Any] | None = {} if params is not None else None

        if (
            params is not None
            and return_params is not None
            and "num_lmcache_extra_cached_tokens" in params
        ):
            request_tracker = self._get_request_tracker(request.request_id)
            num_extra_cached_blocks = max(
                0,
                request_tracker.num_lmcache_hit_blocks
                - request_tracker.num_vllm_hit_blocks,
            )
            return_params["num_lmcache_extra_cached_tokens"] = (
                num_extra_cached_blocks * self.vllm_block_size
            )

        # Clean up request tracker to prevent memory leak
        self._cleanup_request_tracker(request.request_id)
        # Notify LMCache to end the session for this request
        self.scheduler_adapter.end_session(request.request_id)

        return True, return_params

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        :class:`SupportsHMA` variant of :meth:`request_finished`. Receives
        per-group block IDs as ``tuple[list[int], ...]``; the scheduler
        dispatches to this method instead of ``request_finished`` whenever
        the connector inherits :class:`SupportsHMA`.

        Cleanup is by ``request_id`` only — neither the per-request
        tracker nor the LMCache session manager needs the block IDs at
        this point, so this implementation simply forwards to the same
        cleanup path as the single-group variant. Once per-group STORE/
        RETRIEVE threading lands, this method will become the natural
        site to flush any per-group offload state that hasn't been
        committed yet.

        The transfer-params extraction is identical to
        ``request_finished``; the per-group ``block_ids`` argument is
        intentionally unused here (the per-group counts that go into
        ``num_lmcache_extra_cached_tokens`` come from the tracker, not
        from the freshly-passed block IDs).
        """
        params: dict[str, Any] | None = getattr(request, "kv_transfer_params", None)
        return_params: dict[str, Any] | None = {} if params is not None else None

        if (
            params is not None
            and return_params is not None
            and "num_lmcache_extra_cached_tokens" in params
        ):
            request_tracker = self._get_request_tracker(request.request_id)
            num_extra_cached_blocks = max(
                0,
                request_tracker.num_lmcache_hit_blocks
                - request_tracker.num_vllm_hit_blocks,
            )
            return_params["num_lmcache_extra_cached_tokens"] = (
                num_extra_cached_blocks * self.vllm_block_size
            )

        # Clean up request tracker to prevent memory leak
        self._cleanup_request_tracker(request.request_id)
        # Notify LMCache to end the session for this request
        self.scheduler_adapter.end_session(request.request_id)

        return True, return_params

    def take_events(self) -> Iterable["KVCacheEvent"]:
        """
        Take the KV cache events from the connector.

        Yields:
            New KV cache events since the last call.
        """
        return ()

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: "VllmConfig") -> str | None:
        """
        Get the required KV cache layout for this connector.
        Args:
            vllm_config (VllmConfig): the vllm config.

        Returns:
            str: the required KV cache layout. e.g. HND, or NHD.
            None if the connector does not require a specific layout.
        """

        if cls is KVConnectorBase_V1:
            raise TypeError(
                "get_required_kvcache_layout should not be called "
                "on the abstract base class"
            )
        return None

    def get_finished_count(self) -> int | None:
        """
        Get the count of requests expected to complete send/receive operations
        via this connector. This method is used to initialize the
        KVOutputAggregator, overwriting the default world_size.

        Returns:
            int: expected sending or receiving completion count.
        """
        return None

    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> "KVConnectorStats | None":
        """
        KVConnectorStats resolution method. This method allows dynamically
        registered connectors to return their own KVConnectorStats object,
        which can implement custom aggregation logic on the data dict.
        """
        return None

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: "VllmConfig",
        metric_types: dict[type["PromMetric"], type["PromMetricT"]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> "KVConnectorPromMetrics | None":
        """
        Create a KVConnectorPromMetrics subclass which should register
        per-connector Prometheus metrics and implement observe() to
        expose connector transfer stats via Prometheus.
        """
        return None

    ##############################
    # Helper functions
    ##############################
    def _process_retrieve_requests(
        self,
        metadata: LMCacheMPConnectorMetadata,
    ) -> None:
        blocks_per_chunk = self.scheduler_adapter.num_blocks_per_chunk()

        for request_tracker in self.request_trackers.values():
            if request_tracker.state != LMCacheMPRequestState.WAITING_FOR_LOAD:
                continue
            r_metadata = LMCacheMPRequestMetadata.GetRetrieveMetadata(
                request_tracker,
                blocks_per_chunk,
                vllm_block_size=self.vllm_block_size,
                gid_to_block_size=self._gid_to_block_size,
            )
            if r_metadata is not None:
                metadata.add_request_metadata(r_metadata)
            request_tracker.state = LMCacheMPRequestState.READY

    def _process_new_requests(
        self,
        scheduler_output: SchedulerOutput,
        metadata: LMCacheMPConnectorMetadata,
    ) -> None:
        blocks_per_chunk = self.scheduler_adapter.num_blocks_per_chunk()

        for new_request in scheduler_output.scheduled_new_reqs:
            request_tracker = self._get_request_tracker(new_request.req_id)

            num_new_tokens = scheduler_output.num_scheduled_tokens[new_request.req_id]
            request_tracker.increase_num_scheduled_tokens(num_new_tokens)

            r_meta = LMCacheMPRequestMetadata.GetStoreMetadata(
                request_tracker,
                blocks_per_chunk,
                self.vllm_block_size,
                self._gid_to_block_size,
            )
            if r_meta is not None:
                metadata.add_request_metadata(r_meta)

    def _process_cached_requests(
        self,
        scheduler_output: SchedulerOutput,
        metadata: LMCacheMPConnectorMetadata,
    ) -> None:
        blocks_per_chunk = self.scheduler_adapter.num_blocks_per_chunk()

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for idx, request_id in enumerate(cached_reqs.req_ids):
            request_tracker = self._get_request_tracker(request_id)

            # Update block ids: cached_reqs.new_block_ids[idx] is
            # ``tuple[list[int], ...] | None``, one list per
            # ``KVCacheGroupSpec``. For non-hybrid models the tuple has
            # length 1; for hybrid it has the per-gid breakdown.
            new_block_ids_per_group = cached_reqs.new_block_ids[idx]
            if new_block_ids_per_group is None:
                new_block_ids_per_group = ()
            if request_id not in cached_reqs.resumed_req_ids:
                request_tracker.append_block_ids_per_group(new_block_ids_per_group)

            # Use the incremental num_scheduled_tokens to
            # stay consistent with _process_new_requests.
            num_new_tokens = scheduler_output.num_scheduled_tokens[request_id]
            request_tracker.increase_num_scheduled_tokens(num_new_tokens)

            r_meta = LMCacheMPRequestMetadata.GetStoreMetadata(
                request_tracker,
                blocks_per_chunk,
                self.vllm_block_size,
                self._gid_to_block_size,
            )

            if r_meta is not None:
                metadata.add_request_metadata(r_meta)

    def _report_block_allocation_deltas(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        """Gather per-request block allocation deltas and report to LMCache.

        For new requests: all allocated_block_ids and token_ids are new.
        For cached requests: only newly appended block_ids and token_ids.

        L0 telemetry currently consumes a flat list of block IDs at
        ``vllm_block_size`` granularity, so we report the gid 0
        ("primary") namespace's blocks here. For non-hybrid models this
        is the only namespace and equals the prior flat list. For
        hybrid models (e.g. DeepSeek-V4), gid 0 corresponds to the
        scheduler-block-size group (block_size == vllm_block_size); the
        other gids' block IDs live at a finer grid and are not
        currently exposed via the L0 channel.
        """
        records: list[RequestAllocationRecord] = []

        # New requests: send all tokens covering all allocated blocks so
        # the L0 metrics subscriber can correctly map each block to its
        # actual token content (not just the newly-scheduled slice).
        for new_request in scheduler_output.scheduled_new_reqs:
            tracker = self.request_trackers.get(new_request.req_id)
            if tracker is None:
                continue
            primary_block_ids = tracker.allocated_block_ids.get(0, [])
            num_blocks = len(primary_block_ids)
            total_tokens = num_blocks * self.vllm_block_size
            records.append(
                RequestAllocationRecord(
                    req_id=new_request.req_id,
                    new_block_ids=list(primary_block_ids),
                    new_token_ids=list(tracker.all_token_ids[:total_tokens]),
                )
            )

        # Cached requests: only the newly added blocks and their full
        # token content.  We send all tokens covered by the new blocks
        # (not just the tokens scheduled this step) so the L0 subscriber
        # can correctly identify block content.
        cached_reqs = scheduler_output.scheduled_cached_reqs
        for idx, request_id in enumerate(cached_reqs.req_ids):
            new_block_ids_per_group = cached_reqs.new_block_ids[idx]
            # gid 0 = primary (vllm_block_size) namespace.
            new_primary_block_ids = (
                list(new_block_ids_per_group[0])
                if new_block_ids_per_group
                else []
            )
            if not new_primary_block_ids:
                continue
            tracker = self.request_trackers.get(request_id)
            if tracker is None:
                continue
            # The new blocks sit at the end of the request's gid 0 list.
            # Compute the token range they cover.
            primary_block_ids = tracker.allocated_block_ids.get(0, [])
            total_blocks = len(primary_block_ids)
            num_new_blocks = len(new_primary_block_ids)
            start_token = (total_blocks - num_new_blocks) * self.vllm_block_size
            end_token = total_blocks * self.vllm_block_size
            new_token_ids = list(tracker.all_token_ids[start_token:end_token])
            records.append(
                RequestAllocationRecord(
                    req_id=request_id,
                    new_block_ids=new_primary_block_ids,
                    new_token_ids=new_token_ids,
                )
            )

        if records:
            self.scheduler_adapter.report_block_allocations(records)

    def _get_request_tracker(self, request_id: str) -> LMCacheMPRequestTracker:
        assert request_id in self.request_trackers, (
            f"Request tracker for request_id {request_id} not found. "
        )
        return self.request_trackers[request_id]

    def _get_or_create_request_tracker(
        self, request: "Request"
    ) -> LMCacheMPRequestTracker:
        request_id = request.request_id
        # Remove the old trackers that is created before the preemption
        if (
            request.status == RequestStatus.PREEMPTED
            and request_id in self.request_trackers
        ):
            tracker = self.request_trackers[request_id]

            # NOTE: since this function may be called multiple times
            # for a single request (because get_num_new_matched_tokens
            # may be called multiple times) for the same request, we
            # will only do the remove if the tracker is not in the "fresh"
            # state, i.e., PREFETCHING
            if tracker.state != LMCacheMPRequestState.PREFETCHING:
                self.request_trackers.pop(request_id)

        if request_id not in self.request_trackers:
            new_tracker = LMCacheMPRequestTracker(request)
            self.request_trackers[request_id] = new_tracker
        return self.request_trackers[request_id]

    def _cleanup_request_tracker(self, request_id: str) -> None:
        """
        Clean up request tracker and associated lookup future for a request.
        This should be called when a request is finished to prevent memory leak.
        """
        # Clean up request tracker
        if self.request_trackers.pop(request_id, None):
            logger.debug(
                "[KVConnector] Cleaned up request_tracker for request %s",
                request_id,
            )
