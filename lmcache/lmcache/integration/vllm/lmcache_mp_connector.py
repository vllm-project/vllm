# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Standard
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal
import enum
import math
import sys

# Third Party
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)

try:
    # Third Party
    from vllm.distributed.kv_transfer.kv_connector.v1.base import SupportsHMA
except ImportError:
    # Older vLLM builds do not expose HMA. They cannot route per-group
    # request-finished calls, but keeping the class importable preserves
    # legacy single-group behavior.
    class SupportsHMA:  # type: ignore[no-redef]
        pass


# Third Party
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus
from vllm.v1.utils import ConstantList
import torch
import zmq

# First Party
from lmcache import torch_dev
from lmcache.banner import print_banner_once
from lmcache.integration.vllm.experimental import dispatch
from lmcache.integration.vllm.kv_cache_group_edits import (
    apply_kv_cache_group_edits,
    validate_kv_cache_groups,
)
from lmcache.integration.vllm.kv_cache_groups import (
    create_engine_group_infos_from_vllm,
)
from lmcache.integration.vllm.utils import (
    apply_mm_hashes_to_token_ids,
    extract_mm_features,
    mla_only,
    vllm_layout_hints,
)
from lmcache.utils import init_logger as lmcache_init_logger
from lmcache.v1.multiprocess.group_view import slice_block_ids_per_group

try:
    # First Party
    from lmcache.integration.vllm.vllm_multi_process_adapter import (
        LMCacheMPSchedulerAdapter,
        LMCacheMPWorkerAdapter,
        LoadStoreOp,
        ParallelStrategy,
        send_lmcache_request,
    )

    try:
        # First Party
        from lmcache.v1.multiprocess.custom_types import (  # type: ignore[attr-defined]
            RequestAllocationRecord,
        )
    except ImportError:
        # First Party
        from lmcache.v1.multiprocess.custom_types import (
            BlockAllocationRecord as RequestAllocationRecord,
        )
except ImportError:
    # Third Party
    from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration import (  # type: ignore[no-redef]
        LMCacheMPSchedulerAdapter,
        LMCacheMPWorkerAdapter,
        LoadStoreOp,
        ParallelStrategy,
    )

    # First Party
    from lmcache.v1.multiprocess.custom_types import (
        BlockAllocationRecord as RequestAllocationRecord,
    )

if TYPE_CHECKING:
    # Third Party
    from vllm.distributed.kv_events import KVCacheEvent
    from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
        KVConnectorPromMetrics,
        KVConnectorStats,
        PromMetric,
        PromMetricT,
    )
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = lmcache_init_logger(__name__)


# Helper functions
def _has_preemption_reqs(scheduler_output: SchedulerOutput) -> bool:
    """Return whether the scheduler output contains preemption-related requests.

    Checks for the presence of resumed or preempted requests in the
    scheduler output.

    A preemption is detected if:
    - ``scheduled_cached_reqs.resumed_req_ids``: Requests resumed from
      preemption this step.
    - ``scheduler_output.preempted_req_ids``: Requests preempted this step.

    Args:
        scheduler_output: The vLLM scheduler output for this step.

    Returns:
        True if preemption-related requests exist, False otherwise.
    """
    cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)

    # Primary signal: requests resumed from preemption this step.
    resumed_ids = getattr(cached_reqs, "resumed_req_ids", None)
    if resumed_ids:
        logger.warning("<preempted> by resumed requests: %s", resumed_ids)
        return True

    # Primary signal: requests preempted this step.
    preempted_ids = getattr(scheduler_output, "preempted_req_ids", None)
    if preempted_ids:
        logger.warning("<preempted> by preempted requests: %s", preempted_ids)
        return True

    return False


def validate_mamba_step_alignment(vllm_config: VllmConfig) -> None:
    """Reject scheduler configs that can skip Mamba state snapshots.

    In ``mamba_cache_mode="align"`` vLLM snapshots the recurrent state only at
    the end of each scheduler step, and a step that advances more than one
    block fills the skipped block-table positions with the null block
    (``MambaManager.allocate_new_blocks``). LMCache keys chunks by token hash,
    so a skipped boundary would be stored as null-block garbage under a valid
    key and silently corrupt any request that later resumes from that prefix.
    Requiring ``block_size <= max_num_batched_tokens < 2 * block_size`` makes
    vLLM's block-aligned splitting (``Scheduler._mamba_block_aligned_split``)
    advance every mid-prefill step by exactly one block, so every chunk
    boundary holds a real snapshot.

    Args:
        vllm_config: The vLLM config; only Mamba-hybrid models in ``align``
            cache mode are constrained, others pass.

    Raises:
        ValueError: If ``max_num_batched_tokens`` is not in
            ``[block_size, 2 * block_size)``.
    """
    if getattr(vllm_config.cache_config, "mamba_cache_mode", "none") != "align":
        return
    block_size = vllm_config.cache_config.block_size
    max_batched = vllm_config.scheduler_config.max_num_batched_tokens
    if not (block_size <= max_batched < 2 * block_size):
        raise ValueError(
            f"Mamba-hybrid models with LMCache require "
            f"block_size <= max_num_batched_tokens < 2 * block_size so every "
            f"prefill step advances exactly one block and every block boundary "
            f"gets a state snapshot; got max_num_batched_tokens={max_batched}, "
            f"block_size={block_size}. Set --max-num-batched-tokens "
            f"{block_size}."
        )


def build_parallel_strategy_from_vllm_config(
    vllm_config: "VllmConfig",
    n_servers: int,
) -> ParallelStrategy:
    """Build a ParallelStrategy from a vLLM config.

    Centralises the (vllm_config -> KV parallel geometry) mapping.

    Args:
        vllm_config: The vLLM configuration object.
        n_servers: Number of LMCache servers backing this deployment.

    Returns:
        The constructed ParallelStrategy.
    """
    pc = vllm_config.parallel_config
    return ParallelStrategy(
        mla_only=mla_only(vllm_config.model_config),
        vllm_world_size=pc.world_size,
        vllm_worker_id=pc.rank,
        tp_size=pc.tensor_parallel_size,
        pp_size=pc.pipeline_parallel_size,
        n_servers=n_servers,
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

    # Read-only list to track the token ids
    all_token_ids: ConstantList[int]

    # Block ids will be updated at update_states_after_alloc and
    # during generation. Keyed by engine_group_idx; non-HMA models use 0.
    allocated_block_ids: dict[int, list[int]] = field(default_factory=dict)

    # Number of scheduled tokens in this request. We keep tracking this to
    # avoid saving tokens whose KV has not been computed yet.
    num_scheduled_tokens: int = 0

    # Number of tokens stored will be initialized when lookup the external
    # hit tokens and will be updated when processing new requests and cached
    # requests.
    num_stored_tokens: int = 0

    # Staging load operation -- save vllm and lmcache hit tokens during lookup
    num_vllm_hit_tokens: int = 0
    num_lmcache_hit_tokens: int = 0

    # Main state
    state: LMCacheMPRequestState = LMCacheMPRequestState.PREFETCHING

    cache_salt: str = ""

    mm_adjusted_prompt_ids: list[int] = field(default_factory=list)

    def __init__(self, request: "Request"):
        self.request_id = request.request_id
        self.cache_salt: str = request.cache_salt or ""
        self.all_token_ids = request.all_token_ids
        self.allocated_block_ids = {}
        self.num_stored_tokens = 0
        self.num_vllm_hit_tokens = 0
        self.num_lmcache_hit_tokens = 0
        self.state = LMCacheMPRequestState.PREFETCHING
        self.mm_adjusted_prompt_ids = []
        mm_hashes, mm_positions = extract_mm_features(request)
        if mm_hashes and mm_positions:
            prompt_ids = torch.tensor(request.prompt_token_ids)
            apply_mm_hashes_to_token_ids(prompt_ids, mm_hashes, mm_positions)
            self.mm_adjusted_prompt_ids = prompt_ids.tolist()

    ####
    # Check the state of the request
    ####
    def needs_retrieve(self) -> bool:
        """Check whether the current request needs retrieve, will be used
        update_stage_after_alloc"""
        return (
            self.num_lmcache_hit_tokens > self.num_vllm_hit_tokens
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

    def increase_num_stored_tokens(self, num_new_tokens: int):
        """Increase the number of stored tokens for the current request
        This function will be called when processing the cached requests.
        """
        self.num_stored_tokens += num_new_tokens

    def append_block_ids(
        self,
        new_block_ids: tuple[list[int], ...],
    ):
        """Update the block ids for the current request
        This function will be called when processing the cached requests.
        """
        for engine_group_idx, group_block_ids in enumerate(new_block_ids):
            if group_block_ids:
                self.allocated_block_ids.setdefault(engine_group_idx, []).extend(
                    group_block_ids
                )

    def num_allocated_blocks(self) -> dict[int, int]:
        return {
            engine_group_idx: len(blocks)
            for engine_group_idx, blocks in self.allocated_block_ids.items()
        }

    def get_token_ids(self) -> list[int]:
        """Return the token ids to use for LMCache key derivation."""
        if not self.mm_adjusted_prompt_ids:
            return list(self.all_token_ids)
        num_prompt_tokens = len(self.mm_adjusted_prompt_ids)
        return self.mm_adjusted_prompt_ids + list(
            self.all_token_ids[num_prompt_tokens:]
        )

    ####
    # For debugging
    ####
    def __repr__(self) -> str:
        return (
            f"LMCacheMPRequestTracker(request_id={self.request_id}, "
            f"num_tokens={len(self.all_token_ids)}, "
            f"num_allocated_blocks="
            f"{self.num_allocated_blocks()}, "
            f"num_stored_tokens={self.num_stored_tokens}, "
            f"vllm_hit_tokens={self.num_vllm_hit_tokens}, "
            f"lmcache_hit_tokens={self.num_lmcache_hit_tokens}, "
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
    def GetStoreMetadata(
        tracker: LMCacheMPRequestTracker,
        lmcache_tokens_per_chunk: int,
        group_tokens_per_block: list[int],
    ) -> "LMCacheMPRequestMetadata | None":
        """
        Generate the store metadata for the current request tracker.

        Args:
            tracker: The request tracker to generate the metadata from.
            lmcache_tokens_per_chunk: the number of tokens in a LMCache data chunk
            group_tokens_per_block: per-engine-group tokens covered by one
                paged chunk (one block ID) of that group, i.e. the group's
                KV cache spec ``block_size``. Must each divide
                ``lmcache_tokens_per_chunk`` (hybrid models can mix different values).
        """
        num_engine_groups = len(group_tokens_per_block)
        # NOTE: the invariant here is that `num_stored_tokens` should
        # always be a multiple of `lmcache_tokens_per_chunk`
        # TODO: This should be checked every time we update the num_stored_tokens
        #
        # Why computed_tokens uses max(num_vllm_hit_tokens, num_lmcache_hit_tokens):
        #
        # Both values represent a prefix of tokens whose KV data is already
        # available (either from vLLM APC or from LMCache), so they must NOT
        # be summed (that would double-count the overlapping prefix).
        #
        # * num_lmcache_hit_tokens: LMCache-hit tokens are already counted in
        #   num_stored_tokens (set during lookup), so they must be included
        #   here to keep the upper bound consistent.  They are NOT re-stored.
        # * num_vllm_hit_tokens: LMCache stores in units of chunks, so
        #   num_lmcache_hit_tokens is rounded DOWN to the nearest chunk
        #   boundary.  When vLLM APC hits more tokens than that rounded value
        #   (e.g. APC=704 tokens, LMCache=512 tokens after chunk alignment),
        #   using only num_lmcache_hit_tokens would set the upper bound too
        #   low and silently skip the APC-hit tokens that fall between the
        #   two values, causing under-storing.  Taking the max ensures we
        #   always use the tighter (larger) of the two hit counts.
        computed_tokens = tracker.num_scheduled_tokens + max(
            tracker.num_vllm_hit_tokens, tracker.num_lmcache_hit_tokens
        )
        # Each group covers ``len(block_ids) * tokens_per_block`` tokens; the
        # storable prefix is bounded by the least-covered group (e.g.
        # gemma-4 sliding: one 32-token ID covers 2x the tokens of a
        # 16-token full-attention ID).
        allocated_lengths = tracker.num_allocated_blocks()
        allocated_tokens = (
            min(
                allocated_lengths.get(engine_group_idx, 0)
                * group_tokens_per_block[engine_group_idx]
                for engine_group_idx in range(num_engine_groups)
            )
            if num_engine_groups > 0
            else 0
        )
        min_available_tokens = min(
            len(tracker.all_token_ids),
            allocated_tokens,
            computed_tokens,
        )
        num_staging_tokens = min_available_tokens - tracker.num_stored_tokens
        num_chunks = num_staging_tokens // lmcache_tokens_per_chunk

        if num_chunks >= 1:
            start_token_idx = tracker.num_stored_tokens
            end_token_idx = start_token_idx + num_chunks * lmcache_tokens_per_chunk
            block_ids = slice_block_ids_per_group(
                tracker.allocated_block_ids,
                group_tokens_per_block,
                start_token_idx,
                end_token_idx,
            )
            token_ids = tracker.get_token_ids()
            op = LoadStoreOp(
                token_ids=token_ids,
                block_ids=block_ids,
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
            tracker.increase_num_stored_tokens(end_token_idx - start_token_idx)
            return ret

        return None

    @staticmethod
    def GetRetrieveMetadata(
        tracker: LMCacheMPRequestTracker,
        lmcache_tokens_per_chunk: int,
        group_tokens_per_block: list[int],
    ) -> "LMCacheMPRequestMetadata | None":
        """
        Generate the retrieve metadata for the current request tracker.

        Args:
            tracker: The request tracker to generate the metadata from.
            lmcache_tokens_per_chunk: the number of tokens in a LMCache data chunk
            group_tokens_per_block: per-engine-group tokens covered by one
                paged chunk (one block ID) of that group, i.e. the group's
                KV cache spec ``block_size``. Must each divide
                ``lmcache_tokens_per_chunk`` (hybrid models can mix different values).
        """
        if not tracker.is_ready_for_retrieving():
            return None

        # |---------------------|-----------------|----------------|
        # | num_vllm_hit_tokens |
        # | lmcache chunk 1   | lmcache chunk 2   |
        #                     |  need to retrieve |

        start_token_idx = (
            tracker.num_vllm_hit_tokens
            // lmcache_tokens_per_chunk
            * lmcache_tokens_per_chunk
        )
        end_token_idx = tracker.num_lmcache_hit_tokens
        assert end_token_idx % lmcache_tokens_per_chunk == 0, (
            "The number of LMCache hit tokens should be a multiple of the "
            "LMCache chunk size. "
        )
        assert len(tracker.all_token_ids) >= end_token_idx, (
            "The number of tokens should be greater than or equal to the "
            "number of LMCache hit tokens. "
        )
        if end_token_idx > start_token_idx:
            block_ids = slice_block_ids_per_group(
                tracker.allocated_block_ids,
                group_tokens_per_block,
                start_token_idx,
                end_token_idx,
            )
            token_ids = tracker.get_token_ids()

            # Compute how many tokens at the start of the retrieve range
            # overlap with APC-shared blocks. The server must skip writing
            # to these positions to avoid a cross-stream data race: the
            # retrieve writes on the LMCache CUDA stream while concurrent
            # requests may read these APC-shared blocks on the vLLM stream.
            skip_first_n_tokens = tracker.num_vllm_hit_tokens - start_token_idx

            op = LoadStoreOp(
                token_ids=token_ids,
                block_ids=block_ids,
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
        self.need_flush_before_forward: bool = False

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
                f"num_blocks={len(req_meta.op.flat_block_ids)}, "
                f"block_ids={req_meta.op.block_ids})"
            )
        return (
            f"need_flush_before_forward={self.need_flush_before_forward}; ["
            + "\n".join(request_strs)
            + "]"
        )

    def __repr__(self):
        return self.__str__()


def _ensure_zmq_scheme(server_url: str) -> str:
    """Ensure a ZMQ server URL carries a transport scheme.

    ZeroMQ requires an explicit transport (e.g. ``tcp://``) in the address;
    a bare ``host:port`` such as ``127.0.0.1:5557`` is rejected with
    ``ZMQError: Invalid argument``. Users naturally configure
    ``lmcache.mp.host`` as a plain IP/hostname, so prepend ``tcp://`` when no
    scheme is present.

    Args:
        server_url: A server URL, with or without a ``<scheme>://`` prefix.

    Returns:
        The URL with a transport scheme, defaulting to ``tcp://``.
    """
    if "://" in server_url:
        return server_url
    return f"tcp://{server_url}"


class LMCacheMPConnector(KVConnectorBase_V1, SupportsHMA):
    """
    The connector for LMCache multi-process mode.

    Extra configs (kv_transfer_config.extra_config):

    Multi-server deployment:
    - lmcache.mp.server_urls: server URL list or comma-separated string,
      e.g. "tcp://host1:6667,tcp://host2:6667".

    Single-server deployment:
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
        kv_cache_config: "KVCacheConfig | None" = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        # Fail fast, before the server handshake below.
        validate_mamba_step_alignment(vllm_config)
        validate_kv_cache_groups(getattr(self, "_kv_cache_config", None))

        assert vllm_config.kv_transfer_config is not None

        # Multi-server: prefer lmcache.mp.server_urls (list or comma-separated
        # string) over the single-server lmcache.mp.host / lmcache.mp.port.
        server_urls_cfg = vllm_config.kv_transfer_config.get_from_extra_config(
            "lmcache.mp.server_urls", None
        )
        if server_urls_cfg:
            if isinstance(server_urls_cfg, list):
                server_urls = [u.strip() for u in server_urls_cfg if u.strip()]
            else:
                server_urls = [
                    u.strip() for u in server_urls_cfg.split(",") if u.strip()
                ]
        else:
            # Legacy single-server fallback.
            server_host = vllm_config.kv_transfer_config.get_from_extra_config(
                "lmcache.mp.host", "tcp://localhost"
            )
            server_port = vllm_config.kv_transfer_config.get_from_extra_config(
                "lmcache.mp.port", 5555
            )
            server_urls = [f"{server_host}:{server_port}"]

        # Normalize so a bare host:port (no transport scheme) is accepted;
        # ZMQ requires an explicit transport such as ``tcp://``.
        server_urls = [_ensure_zmq_scheme(u) for u in server_urls]

        # The server count is derived from lmcache.mp.server_urls.
        n_servers = len(server_urls)

        assert vllm_config.parallel_config.world_size % n_servers == 0, (
            f"world_size ({vllm_config.parallel_config.world_size}) must be "
            f"divisible by n_servers ({n_servers})"
        )

        # Multi-server + DP is not supported yet.
        dp_size = getattr(vllm_config.parallel_config, "data_parallel_size", 1)
        if n_servers > 1 and dp_size > 1:
            raise ValueError(
                "LMCacheMPConnector multi-server mode (n_servers > 1) does not "
                f"support data parallelism yet; got dp_size={dp_size}. "
                "DP across multiple LMCache servers will be "
                "supported in a follow-up PR."
            )

        # Multi-server + MLA: only TP is supported (no PP).
        # PP splits layers across nodes, which would cause per-piece
        # reader counts to vary per (server, pp_stage) pair and break
        # the single-``tp_size`` LOOKUP / FREE_LOOKUP_LOCKS protocol.
        # Non-MLA mode is not affected by this restriction.
        if n_servers > 1:
            pp_size = vllm_config.parallel_config.pipeline_parallel_size
            if pp_size > 1:
                raise ValueError(
                    "LMCacheMPConnector multi-server mode only supports "
                    "tensor parallelism (TP), not pipeline parallelism (PP). "
                    f"Got pp_size={pp_size}."
                )

        zmq_context = zmq.Context.instance()
        parallel_strategy = build_parallel_strategy_from_vllm_config(
            vllm_config, n_servers
        )

        self.dispatcher = None

        if self.role == KVConnectorRole.SCHEDULER:
            # Banner from the scheduler role only, so tensor-parallel
            # deployments print it once rather than once per worker.
            print_banner_once(sys.stderr)
            self.scheduler_adapter = LMCacheMPSchedulerAdapter(
                server_urls=server_urls,
                context=zmq_context,
                model_name=vllm_config.model_config.model,
                vllm_block_size=vllm_config.cache_config.block_size,
                parallel_strategy=parallel_strategy,
                extra_config=vllm_config.kv_transfer_config.kv_connector_extra_config,
            )
            self.request_trackers: dict[str, LMCacheMPRequestTracker] = {}
        elif self.role == KVConnectorRole.WORKER:
            # Node routing: a worker connects only to its local LMCache server.
            # Global ranks are assigned to nodes in contiguous blocks:
            #   node 0 → ranks [0, ranks_per_node),
            #   node 1 → [ranks_per_node, 2 * ranks_per_node), ...
            ranks_per_node = parallel_strategy.vllm_world_size // n_servers
            local_server_url = server_urls[
                parallel_strategy.vllm_worker_id // ranks_per_node
            ]
            self.worker_adapter = LMCacheMPWorkerAdapter(
                server_url=local_server_url,
                context=zmq_context,
                model_name=vllm_config.model_config.model,
                vllm_block_size=vllm_config.cache_config.block_size,
                parallel_strategy=parallel_strategy,
                extra_config=vllm_config.kv_transfer_config.kv_connector_extra_config,
            )
            if self.transfer_intermediate_tensors:
                # First Party
                from lmcache.integration.vllm.experimental import (
                    FeatureContext,
                    init_dispatcher,
                )
                from lmcache.v1.multiprocess.modules.experimental import TRANSFER_QUERY

                ctx = FeatureContext(
                    worker_adapter=self.worker_adapter,
                    send_lmcache_request=send_lmcache_request,
                )
                requested = (
                    {TRANSFER_QUERY} if self.transfer_intermediate_tensors else set()
                )
                self.dispatcher = init_dispatcher(ctx, requested)
                self.worker_adapter.dispatcher = self.dispatcher
        else:
            raise ValueError(f"Unknown KVConnectorRole: {self.role}")

        kv_cache_config = getattr(self, "_kv_cache_config", None)
        vllm_groups = (
            getattr(kv_cache_config, "kv_cache_groups", ()) or ()
            if kv_cache_config is not None
            else ()
        )
        # Tokens covered by one paged chunk (one block ID) of each engine
        # group, from the group's KV cache spec. Hybrid models can mix
        # different values (e.g. gemma-4: sliding-window groups 32,
        # full-attention groups 16; DeepSeek V4: 256/64/8/4). Falls back to
        # the engine's base block size when no group metadata is available
        # (single non-hybrid group).
        self._group_tokens_per_block: list[int] = [
            group.kv_cache_spec.block_size for group in vllm_groups
        ] or [vllm_config.cache_config.block_size]
        for engine_group_idx, tokens_per_block in enumerate(
            self._group_tokens_per_block
        ):
            if tokens_per_block <= 0:
                raise ValueError(
                    f"group {engine_group_idx} tokens_per_block "
                    f"{tokens_per_block} must be positive"
                )
        # Smallest token count aligned to every group's paged-chunk
        # boundary; used to round down vLLM APC hit counts.
        self._hit_alignment_tokens = math.lcm(*self._group_tokens_per_block)
        if self.role == KVConnectorRole.SCHEDULER:
            # Chunk boundaries must land on every group's paged-chunk
            # boundary so per-group block-id slicing stays aligned.
            lmcache_tokens_per_chunk = self.scheduler_adapter.lmcache_tokens_per_chunk
            for engine_group_idx, tokens_per_block in enumerate(
                self._group_tokens_per_block
            ):
                if lmcache_tokens_per_chunk % tokens_per_block != 0:
                    raise ValueError(
                        f"LMCache chunk size {lmcache_tokens_per_chunk} must be "
                        f"a multiple of group {engine_group_idx} "
                        f"tokens_per_block {tokens_per_block}"
                    )

    @property
    def role(self) -> KVConnectorRole:
        return self._role

    @property
    def transfer_intermediate_tensors(self) -> bool:
        cfg = self._vllm_config.kv_transfer_config
        if cfg is None:
            return False
        vllm_transfer = cfg.get_from_extra_config(
            "lmcache.mp.transfer_intermediate_tensors", False
        )
        return vllm_transfer

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
        kv_cache_config = getattr(self, "_kv_cache_config", None)
        # Must precede both group-info creation and transfer registration so
        # they see the same edited views.
        layout_hints = vllm_layout_hints()
        kv_caches = apply_kv_cache_group_edits(
            kv_cache_config, kv_caches, layout_hints=layout_hints
        )
        engine_group_infos = create_engine_group_infos_from_vllm(
            kv_cache_config,
            kv_caches,
            layout_hints=layout_hints,
        )
        self.worker_adapter.register_kv_caches(
            kv_caches, engine_group_infos=engine_group_infos
        )
        if self.dispatcher is not None:
            dispatch(
                self.dispatcher,
                "register",
                kv_caches=kv_caches,
                kv_cache_config=kv_cache_config,
                vllm_config=self._vllm_config,
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

        event = torch_dev.Event(interprocess=True)
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
        if self.dispatcher is not None:
            dispatch(
                self.dispatcher,
                "save_kv_layer",
                layer_name=layer_name,
                metadata=self._get_connector_metadata(),
                attn_metadata=attn_metadata,
                **kwargs,
            )
        return

    def wait_for_save(self):
        """
        Block until all the save operations is done. This is called
        as the forward context exits to ensure that the async saving
        from save_kv_layer is complete before finishing the forward.

        This prevents overwrites of paged KV buffer before saving done.
        """
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
            if self.dispatcher is not None:
                dispatch(self.dispatcher, "wait_for_save", event=None)
            return

        event = torch_dev.Event(interprocess=True)
        event.record()

        self.worker_adapter.batched_submit_store_requests(
            request_ids, ops, event, cache_salts=cache_salts
        )
        if self.dispatcher is not None:
            dispatch(self.dispatcher, "wait_for_save", event=event)

    # TODO: How does lmcache driven path handle preemption?
    # NOTE1: handle_preemptions is called by vllm each step regardless
    #        preemption really happens or not.
    # NOTE2: preemption hint is managed by KVConnectorRole.SCHEDULER,
    #        that's why here we have to judge preemption by
    #        need_flush_before_forward flag which is set by SCHEDULER.
    def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata) -> None:
        """Flush async engine-driven stores only when scheduler metadata requests it.

        Args:
            kv_connector_metadata: Connector metadata produced by the scheduler;
                only acts when it is a :class:`LMCacheMPConnectorMetadata` with
                ``need_flush_before_forward=True``.
        """
        worker_adapter = getattr(self, "worker_adapter", None)
        if self.role != KVConnectorRole.WORKER or worker_adapter is None:
            return
        need_flush_before_forward = (
            isinstance(kv_connector_metadata, LMCacheMPConnectorMetadata)
            and kv_connector_metadata.need_flush_before_forward
        )
        worker_adapter.handle_preemptions(need_flush_before_forward)

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
        if hasattr(self, "scheduler_adapter"):
            self.scheduler_adapter.shutdown()
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
            token_ids=tracker.get_token_ids(),
            cache_salt=tracker.cache_salt,
        )

        ret = self.scheduler_adapter.check_lookup_result(request.request_id)
        if ret is None:
            return None, True

        if ret == 0:
            return 0, False

        assert ret % self.scheduler_adapter.lmcache_tokens_per_chunk == 0

        # Update num stored tokens for the tracker
        tracker.increase_num_stored_tokens(ret)

        # Save the vllm and lmcache hit tokens. The vLLM hit count is
        # rounded down to a boundary aligned for every engine group (e.g.
        # a full-prompt APC hit reports ``num_prompt_tokens - 1``), so the
        # retrieve-skip range stays paged-chunk-aligned in all groups.
        tracker.num_vllm_hit_tokens = (
            num_computed_tokens
            // self._hit_alignment_tokens
            * self._hit_alignment_tokens
        )
        tracker.num_lmcache_hit_tokens = ret

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
        block_ids = blocks.get_block_ids() or ()

        # Only append blocks beyond what's already tracked, per engine group.
        existing_counts = tracker.num_allocated_blocks()
        new_block_ids: list[list[int]] = []
        for engine_group_idx, group_blocks in enumerate(block_ids):
            existing = existing_counts.get(engine_group_idx, 0)
            new_block_ids.append(list(group_blocks[existing:]))
        if any(new_block_ids):
            tracker.append_block_ids(tuple(new_block_ids))

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
            if tracker.num_lmcache_hit_tokens > 0:
                if not condition:
                    # No retrieve needed — free ALL locked chunks
                    free_end = tracker.num_lmcache_hit_tokens
                else:
                    # Note(Roy): Boundary misalignment between vLLM blocks and LMCache
                    # blocks is handled in free_lookup_locks. It makes sure that if
                    # the last vLLM computed block ends in the middle of a LMCache
                    # block, the end LMCache block is not freed (i.e., floor division)
                    # since it will still be needed by vLLM and such block's lock will
                    # be freed by vLLM's retrieve.
                    free_end = tracker.num_vllm_hit_tokens

                if free_end > 0:
                    self.scheduler_adapter.free_lookup_locks(
                        token_ids=tracker.get_token_ids(),
                        start=0,
                        end=free_end,
                        request_id=request.request_id,
                        cache_salt=tracker.cache_salt,
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
        metadata.need_flush_before_forward = _has_preemption_reqs(scheduler_output)

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
            and "cached_token_stats" in params
        ):
            request_tracker = self._get_request_tracker(request.request_id)
            num_vllm = request_tracker.num_vllm_hit_tokens
            num_lmcache = request_tracker.num_lmcache_hit_tokens
            return_params["cached_token_stats"] = {
                "num_vllm_cached_tokens": num_vllm,
                "num_lmcache_cached_tokens": num_lmcache,
                "num_lmcache_extra_cached_tokens": max(0, num_lmcache - num_vllm),
            }

        # Clean up request tracker to prevent memory leak
        self._cleanup_request_tracker(request.request_id)
        # Notify LMCache to end the session for this request
        self.scheduler_adapter.end_session(request.request_id)

        return True, (return_params or None)

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        """HMA request-finished entry point; cleanup is request-id based."""
        return self.request_finished(request, block_ids[0] if block_ids else [])

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
        lmcache_tokens_per_chunk = self.scheduler_adapter.lmcache_tokens_per_chunk

        for request_tracker in self.request_trackers.values():
            if request_tracker.state != LMCacheMPRequestState.WAITING_FOR_LOAD:
                continue
            r_metadata = LMCacheMPRequestMetadata.GetRetrieveMetadata(
                request_tracker,
                lmcache_tokens_per_chunk,
                group_tokens_per_block=self._group_tokens_per_block,
            )
            if r_metadata is not None:
                metadata.add_request_metadata(r_metadata)
            request_tracker.state = LMCacheMPRequestState.READY

    def _process_new_requests(
        self,
        scheduler_output: SchedulerOutput,
        metadata: LMCacheMPConnectorMetadata,
    ) -> None:
        lmcache_tokens_per_chunk = self.scheduler_adapter.lmcache_tokens_per_chunk

        for new_request in scheduler_output.scheduled_new_reqs:
            request_tracker = self._get_request_tracker(new_request.req_id)

            num_new_tokens = scheduler_output.num_scheduled_tokens[new_request.req_id]
            request_tracker.increase_num_scheduled_tokens(num_new_tokens)

            r_meta = LMCacheMPRequestMetadata.GetStoreMetadata(
                request_tracker,
                lmcache_tokens_per_chunk,
                self._group_tokens_per_block,
            )
            if r_meta is not None:
                metadata.add_request_metadata(r_meta)

    def _process_cached_requests(
        self,
        scheduler_output: SchedulerOutput,
        metadata: LMCacheMPConnectorMetadata,
    ) -> None:
        lmcache_tokens_per_chunk = self.scheduler_adapter.lmcache_tokens_per_chunk

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for idx, request_id in enumerate(cached_reqs.req_ids):
            request_tracker = self._get_request_tracker(request_id)

            # Update block ids
            new_block_ids = cached_reqs.new_block_ids[idx] or ()
            if request_id not in cached_reqs.resumed_req_ids:
                request_tracker.append_block_ids(new_block_ids)

            # Use the incremental num_scheduled_tokens to
            # stay consistent with _process_new_requests.
            num_new_tokens = scheduler_output.num_scheduled_tokens[request_id]
            request_tracker.increase_num_scheduled_tokens(num_new_tokens)

            r_meta = LMCacheMPRequestMetadata.GetStoreMetadata(
                request_tracker,
                lmcache_tokens_per_chunk,
                self._group_tokens_per_block,
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
        The L0 allocation telemetry is flat today, so HMA reports engine group 0.
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
            total_tokens = num_blocks * self._group_tokens_per_block[0]
            records.append(
                RequestAllocationRecord(
                    req_id=new_request.req_id,
                    new_block_ids=list(primary_block_ids),
                    new_token_ids=tracker.get_token_ids()[:total_tokens],
                )
            )

        # Cached requests: only the newly added blocks and their full
        # token content.  We send all tokens covered by the new blocks
        # (not just the tokens scheduled this step) so the L0 subscriber
        # can correctly identify block content.
        cached_reqs = scheduler_output.scheduled_cached_reqs
        for idx, request_id in enumerate(cached_reqs.req_ids):
            # The L0 subscriber works on the primary (group 0) block-id list.
            new_group_block_ids = cached_reqs.new_block_ids[idx]
            new_block_ids = new_group_block_ids[0] if new_group_block_ids else []
            if not new_block_ids:
                continue
            tracker = self.request_trackers.get(request_id)
            if tracker is None:
                continue
            # The new blocks sit at the end of the request's block list.
            # Compute the token range they cover.
            total_blocks = len(tracker.allocated_block_ids.get(0, []))
            num_new_blocks = len(new_block_ids)
            tokens_per_block = self._group_tokens_per_block[0]
            start_token = (total_blocks - num_new_blocks) * tokens_per_block
            end_token = total_blocks * tokens_per_block
            new_token_ids = tracker.get_token_ids()[start_token:end_token]
            records.append(
                RequestAllocationRecord(
                    req_id=request_id,
                    new_block_ids=new_block_ids,
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
