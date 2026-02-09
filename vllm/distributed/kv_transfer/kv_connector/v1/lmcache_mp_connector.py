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
    )
except ImportError:
    from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration import (
        LMCacheMPSchedulerAdapter,
        LMCacheMPWorkerAdapter,
        LoadStoreOp,
    )

if TYPE_CHECKING:
    from vllm.config import VllmConfig
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
def reformat_block_ids(block_ids: tuple[list[int], ...] | None) -> list[int]:
    if block_ids is None:
        return []
    assert isinstance(block_ids, tuple), (
        f"Expected block_ids to be a tuple of lists, but got {type(block_ids)}"
    )

    if len(block_ids) > 1:
        raise RuntimeError(
            "LMCacheMPConnector only works without hybrid kv cache manager. "
            "Please pass --disable-hybrid-kv-cache-manager when starting vllm"
        )

    return block_ids[0]


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
        # TP group: [0, 1, 2, 3], [4, 5, 6, 7]
        # PP group: [0, 4], [1, 5], [2, 6], [3, 7]
        # So we can "exclude" the effect of TP by rank // tp_size.
        return world_size // tp_size, rank // tp_size


def create_scheduler_adapter(
    server_url: str, zmq_context: zmq.Context, vllm_config: VllmConfig
) -> LMCacheMPSchedulerAdapter:
    world_size, kv_rank = extract_world_size_and_kv_rank(
        vllm_config.parallel_config.world_size,
        vllm_config.parallel_config.rank,
        vllm_config,
    )
    return LMCacheMPSchedulerAdapter(
        server_url,
        zmq_context,
        vllm_config.model_config.model,
        world_size,
        kv_rank,
        vllm_config.cache_config.block_size,
    )


def create_worker_adapter(
    server_url: str, zmq_context: zmq.Context, vllm_config: VllmConfig
) -> LMCacheMPWorkerAdapter:
    world_size, kv_rank = extract_world_size_and_kv_rank(
        vllm_config.parallel_config.world_size,
        vllm_config.parallel_config.rank,
        vllm_config,
    )
    return LMCacheMPWorkerAdapter(
        server_url,
        zmq_context,
        vllm_config.model_config.model,
        world_size,
        kv_rank,
        vllm_config.cache_config.block_size,
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

    # Block ids and hashes will be updated at update_states_after_alloc and
    # during the generation
    allocated_block_ids: list[int] = field(default_factory=list)

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

    def __init__(self, request: "Request"):
        self.request_id = request.request_id
        self.all_token_ids = request.all_token_ids
        self.block_hashes = ConstantList(request.block_hashes)
        self.allocated_block_ids = []
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

    def append_block_ids(
        self,
        new_block_ids: list[int],
    ):
        """Update the block ids for the current request
        This function will be called when processing the cached requests.
        """
        self.allocated_block_ids.extend(new_block_ids)

    ####
    # For debugging
    ####
    def __repr__(self) -> str:
        return (
            f"LMCacheMPRequestTracker(request_id={self.request_id}, "
            f"num_tokens={len(self.all_token_ids)}, "
            f"num_block_hashes={len(self.block_hashes)}, "
            f"num_allocated_blocks={len(self.allocated_block_ids)}, "
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

    @staticmethod
    def GetStoreMetadata(
        tracker: LMCacheMPRequestTracker,
        blocks_in_chunk: int,
        vllm_block_size: int,
    ) -> "LMCacheMPRequestMetadata | None":
        """
        Generate the store metadata for the current request tracker.

        Args:
            tracker: The request tracker to generate the metadata from.
            blocks_in_chunk: the number of blocks in a LMCache data chunk
            vllm_block_size: the block size used in vLLM
        """
        # Store the blocks that has block hashes
        # NOTE: the invariant here is that `num_stored_blocks` should
        # always be a multiple of `blocks_in_chunk`
        # TODO: This should be checked everytime we update the num_stored_blocks
        min_available_blocks = min(
            len(tracker.block_hashes),
            len(tracker.allocated_block_ids),
            tracker.num_scheduled_tokens // vllm_block_size,
        )
        num_staging_blocks = min_available_blocks - tracker.num_stored_blocks
        num_chunks = num_staging_blocks // blocks_in_chunk

        if num_chunks >= 1:
            start = tracker.num_stored_blocks
            end = start + num_chunks * blocks_in_chunk
            block_ids = tracker.allocated_block_ids[start:end]

            # Token mode: pass full token_ids with start/end range
            start_token_idx = start * vllm_block_size
            end_token_idx = end * vllm_block_size
            token_ids = list(tracker.all_token_ids)
            op = LoadStoreOp(
                token_ids=token_ids, block_ids=block_ids,
                start=start_token_idx, end=end_token_idx,
            )

            ret = LMCacheMPRequestMetadata(
                request_id=tracker.request_id,
                direction="STORE",
                op=op,
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
    ) -> "LMCacheMPRequestMetadata | None":
        """
        Generate the retrieve metadata for the current request tracker.

        Args:
            tracker: The request tracker to generate the metadata from.
            blocks_in_chunk: the number of blocks in a LMCache data chunk
            vllm_block_size: the block size used in vLLM
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
            block_ids = tracker.allocated_block_ids[start:end]

            # Token mode: pass full token_ids with start/end range
            start_token_idx = start * vllm_block_size
            end_token_idx = end * vllm_block_size
            token_ids = list(tracker.all_token_ids)
            op = LoadStoreOp(
                token_ids=token_ids, block_ids=block_ids,
                start=start_token_idx, end=end_token_idx,
            )

            ret = LMCacheMPRequestMetadata(
                request_id=tracker.request_id,
                direction="RETRIEVE",
                op=op,
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


class LMCacheMPConnector(KVConnectorBase_V1):
    """
    The connector for LMCache multi-process mode.

    Extra configs (kv_transfer_config.extra_config):
    - lmcache.mp.host: the host of the LMCache server.
    - lmcache.mp.port: the port of the LMCache server.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        assert vllm_config.kv_transfer_config is not None
        server_host = vllm_config.kv_transfer_config.get_from_extra_config(
            "lmcache.mp.host", "tcp://localhost"
        )
        server_port = vllm_config.kv_transfer_config.get_from_extra_config(
            "lmcache.mp.port", 9006
        )

        server_url = f"{server_host}:{server_port}"
        zmq_context = zmq.Context.instance()
        if self.role == KVConnectorRole.SCHEDULER:
            self.scheduler_adapter = create_scheduler_adapter(
                server_url, zmq_context, vllm_config
            )
            self.request_trackers: dict[str, LMCacheMPRequestTracker] = {}
        elif self.role == KVConnectorRole.WORKER:
            self.worker_adapter = create_worker_adapter(
                server_url, zmq_context, vllm_config
            )
        else:
            raise ValueError(f"Unknown KVConnectorRole: {self.role}")

        self.vllm_block_size = vllm_config.cache_config.block_size

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
        self.worker_adapter.register_kv_caches(kv_caches)
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

        for meta in metadata.requests:
            if meta.direction != "RETRIEVE":
                continue
            request_ids.append(meta.request_id)
            ops.append(meta.op)

        if len(request_ids) == 0:
            return

        with torch.cuda.stream(torch.cuda.current_stream()):
            event = torch.cuda.Event(interprocess=True)
            event.record()

        self.worker_adapter.batched_submit_retrieve_requests(request_ids, ops, event)

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
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, LMCacheMPConnectorMetadata)

        request_ids = []
        ops = []
        for meta in metadata.requests:
            if meta.direction != "STORE":
                continue
            request_ids.append(meta.request_id)
            ops.append(meta.op)

        if len(request_ids) == 0:
            return

        with torch.cuda.stream(torch.cuda.current_stream()):
            event = torch.cuda.Event(interprocess=True)
            event.record()

        self.worker_adapter.batched_submit_store_requests(request_ids, ops, event)

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
        # TODO: add error tracking
        return set()

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
        # NOTE: the `blocks` are NEW BLOCKS allocated for this request.
        tracker = self._get_request_tracker(request.request_id)
        block_ids = reformat_block_ids(blocks.get_block_ids())

        # No matter we need to retrieve or not, we need to update
        # the block ids into the tracker
        tracker.append_block_ids(block_ids)

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
        # Clean up request tracker to prevent memory leak
        self._cleanup_request_tracker(request.request_id)
        # Notify LMCache to end the session for this request
        self.scheduler_adapter.end_session(request.request_id)

        return True, None

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
                request_tracker, blocks_per_chunk, self.vllm_block_size
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

            # Update block ids
            new_block_ids = reformat_block_ids(cached_reqs.new_block_ids[idx])
            if request_id not in cached_reqs.resumed_req_ids:
                request_tracker.append_block_ids(new_block_ids)

            # Update new scheduled tokens
            num_new_tokens = cached_reqs.num_computed_tokens[idx]
            request_tracker.increase_num_scheduled_tokens(num_new_tokens)

            r_meta = LMCacheMPRequestMetadata.GetStoreMetadata(
                request_tracker, blocks_per_chunk, self.vllm_block_size
            )

            if r_meta is not None:
                metadata.add_request_metadata(r_meta)

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
