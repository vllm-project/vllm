# SPDX-License-Identifier: Apache-2.0
import contextlib
import math
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator

import msgspec
import torch
import zmq
from typing_extensions import Optional

from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole, KVTransferParams)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size,
    get_tp_group)
from vllm.logger import init_logger
from vllm.utils import round_down
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

GET_META_MSG = b"get_meta_msg"

logger = init_logger(__name__)

# Lazy import nixl_wrapper to avoid loading nixl_bindings if nixl is not used
try:
    from nixl._api import nixl_agent as NixlWrapper
    logger.info("NIXL is available")
except ImportError:
    logger.warning("NIXL is not available")
    NixlWrapper = None


@dataclass
class NixlKVTransferParams(KVTransferParams):

    def __init__(
        self,
        do_remote_prefill: bool,
        do_remote_decode: bool,
        remote_block_ids: Optional[list[int]] = None,
        remote_host: Optional[str] = None,
        remote_port: Optional[int] = None,
        remote_engine_id: Optional[str] = None,
    ):
        self.do_remote_prefill = do_remote_prefill
        self.do_remote_decode = do_remote_decode
        self.remote_block_ids = remote_block_ids
        self.remote_host = remote_host
        self.remote_port = remote_port
        self.remote_engine_id = remote_engine_id

    @staticmethod
    def from_raw_dict(
        raw_dict: Optional[dict[str,
                                Any]]) -> Optional["NixlKVTransferParams"]:

        # If no raw transfer params passed, return None.
        if raw_dict is None:
            return None

        # Validate the request is formatted properly.
        if (("do_remote_prefill" not in raw_dict)
                or ("do_remote_decode" not in raw_dict)
                or ("remote_block_ids" not in raw_dict)
                or ("remote_host" not in raw_dict)
                or ("remote_port" not in raw_dict)
                or ("remote_engine_id" not in raw_dict)):
            logger.warning(
                "Got invalid KVTransferParams: %s. This "
                "request will not utilize KVTransfer", raw_dict)
            return None

        return NixlKVTransferParams(
            do_remote_prefill=raw_dict["do_remote_prefill"],
            do_remote_decode=raw_dict["do_remote_decode"],
            remote_block_ids=raw_dict["remote_block_ids"],
            remote_host=raw_dict["remote_host"],
            remote_port=raw_dict["remote_port"],
            remote_engine_id=raw_dict["remote_engine_id"],
        )


class NixlAgentMetadata(
        msgspec.Struct,
        omit_defaults=True,  # type: ignore[call-arg]
        # required for @cached_property.
        dict=True):
    engine_id: str
    agent_metadata: bytes
    kv_caches_base_addr: list[int]
    num_blocks: int


@dataclass
class ReqMeta:
    local_block_ids: list[int]
    remote_block_ids: list[int]
    remote_host: str
    remote_port: int
    remote_engine_id: str


class NixlConnectorMetadata(KVConnectorMetadata):

    def __init__(self):
        self.requests: dict[str, ReqMeta] = {}

    def add_new_req(
        self,
        request_id: str,
        local_block_ids: list[int],
        kv_transfer_params: NixlKVTransferParams,
    ):
        assert request_id not in self.requests
        assert kv_transfer_params.remote_block_ids is not None
        assert kv_transfer_params.remote_engine_id is not None
        assert kv_transfer_params.remote_host is not None
        assert kv_transfer_params.remote_port is not None

        self.requests[request_id] = ReqMeta(
            local_block_ids=local_block_ids,
            remote_block_ids=kv_transfer_params.remote_block_ids,
            remote_engine_id=kv_transfer_params.remote_engine_id,
            remote_host=kv_transfer_params.remote_host,
            remote_port=kv_transfer_params.remote_port,
        )


class NixlConnector(KVConnectorBase_V1):
    _KVTransferParams: type[NixlKVTransferParams] = NixlKVTransferParams

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole):
        assert vllm_config.kv_transfer_config is not None
        self.engine_id = vllm_config.kv_transfer_config.engine_id

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler : Optional[NixlConnectorScheduler] = \
                NixlConnectorScheduler(vllm_config, str(self.engine_id))
            self.connector_worker: Optional[NixlConnectorWorker] = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = NixlConnectorWorker(str(self.engine_id))

    ############################################################
    # Scheduler Side Methods
    ############################################################

    def get_num_new_matched_tokens(
            self, request: "Request",
            num_computed_tokens: int) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens)

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    ############################################################
    # Worker Side Methods
    ############################################################
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def get_finished(self,
                     finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """Get the finished recving and sending requests."""
        assert self.connector_worker is not None
        return self.connector_worker.get_finished()

    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, NixlConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """NixlConnector does not do layerwise saving."""
        pass

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        """NixlConnector does not save explicitly."""
        pass

    def wait_for_save(self):
        """NixlConnector does not save explicitly."""
        pass


class NixlConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self.engine_id = engine_id
        logger.info("Initializing NIXL Scheduler %s", engine_id)

        # Requests that need to start recv.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        self._reqs_need_recv: dict[str, tuple[Request, list[int]]] = {}

    def get_num_new_matched_tokens(
            self, request: "Request",
            num_computed_tokens: int) -> tuple[int, bool]:
        """
        For remote prefill, pull all prompt blocks from remote
        asynchronously relative to engine execution.
        
        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request
        Returns:
            * the number of tokens that can be loaded from the 
              external KV cache beyond what is already computed.
            * true if the external KV cache tokens will be loaded
              asynchronously (between scheduler steps).
        """

        logger.debug(
            "NIXLConnector get_num_new_matched_tokens: "
            "num_computed_tokens=%s, kv_transfer_params=%s",
            num_computed_tokens, request.kv_transfer_params)

        # No KVTransfer for this request.
        if request.kv_transfer_params is None:
            return 0, False
        assert isinstance(request.kv_transfer_params, NixlKVTransferParams)

        # Remote prefill: get all prompt blocks from remote.
        if request.kv_transfer_params.do_remote_prefill:
            assert num_computed_tokens % self.block_size == 0
            rounded_num_prompt_tokens = round_down(
                len(request.prompt_token_ids), self.block_size)
            count = max(rounded_num_prompt_tokens - num_computed_tokens, 0)
            return count, count > 0

        return 0, False

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):

        logger.debug(
            "NIXLConnector update_state_after_alloc: "
            "num_external_tokens=%s, kv_transfer_params=%s",
            num_external_tokens, request.kv_transfer_params)

        if request.kv_transfer_params is None:
            return

        assert isinstance(request.kv_transfer_params, NixlKVTransferParams)
        if request.kv_transfer_params.do_remote_prefill:
            # NOTE(rob): if prompt < block_size, no remote blocks
            # since the remote only sends fully computed blocks, so
            # skip recving for this request. num_external_tokens
            # should be 0 if there are no remote blocks.
            if request.kv_transfer_params.remote_block_ids:
                # Get unhashed blocks to pull from remote.
                self._reqs_need_recv[request.request_id] = (
                    request, blocks.get_unhashed_block_ids())
            else:
                assert num_external_tokens == 0
            # Only trigger 1 KV transfer per request.
            request.kv_transfer_params.do_remote_prefill = False

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = NixlConnectorMetadata()

        # Loop through scheduled reqs and convert to ReqMeta.
        for req_id, (req, block_ids) in self._reqs_need_recv.items():
            assert isinstance(req.kv_transfer_params, NixlKVTransferParams)
            meta.add_new_req(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=req.kv_transfer_params,
            )

        # Clear the list once workers start the transfers
        self._reqs_need_recv.clear()

        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """

        logger.debug(
            "NIXLConnector request_finished, "
            "request_status=%s, kv_transfer_params=%s", request.status,
            request.kv_transfer_params)

        if request.kv_transfer_params is None:
            return False, None
        assert isinstance(request.kv_transfer_params, NixlKVTransferParams)

        if ((not request.kv_transfer_params.do_remote_decode)
                or (request.status != RequestStatus.FINISHED_LENGTH_CAPPED)):
            return False, None

        # Get computed blocks.
        all_full = request.num_computed_tokens % self.block_size == 0
        computed_block_ids = (block_ids if all_full else block_ids[:-1])

        # If prompt < block_size, no xfer so free blocks immediately.
        delay_free_blocks = len(computed_block_ids) > 0

        return delay_free_blocks, NixlKVTransferParams(
            do_remote_prefill=True,
            do_remote_decode=False,
            remote_block_ids=computed_block_ids,
            remote_engine_id=self.engine_id,
            remote_host=envs.VLLM_NIXL_SIDE_CHANNEL_HOST,
            remote_port=envs.VLLM_NIXL_SIDE_CHANNEL_PORT,
        ).__dict__


class NixlConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(self, engine_id: str):
        if NixlWrapper is None:
            logger.error("NIXL is not available")
            raise RuntimeError("NIXL is not available")
        logger.info("Initializing NIXL wrapper")
        logger.info("Initializing NIXL worker %s", engine_id)

        # Agent.
        self.nixl_wrapper = NixlWrapper(str(uuid.uuid4()), None)
        # Map of engine_id -> agent_name.
        self._remote_agents: dict[str, str] = {}

        # Metadata.
        self.engine_id = engine_id
        self.rank = get_tensor_model_parallel_rank()
        self.world_size = get_tensor_model_parallel_world_size()
        self.tp_group = get_tp_group()

        # KV Caches and nixl tracking data.
        self.kv_caches: dict[str, torch.Tensor] = {}

        # Map of engine_id -> kv_caches_base_addr
        self.kv_caches_base_addr: dict[str, list[int]] = {}

        # Number of NIXL regions. Currently one region per cache
        # (so 1 per layer for MLA, otherwise 2 per layer)
        self.num_regions = 0

        # nixl_prepped_dlist_handle (int).
        self.src_xfer_side_handle: int = 0
        # Map of engine_id -> nixl_prepped_dlist_handle (int)].
        self.dst_xfer_side_handles: dict[str, int] = {}

        # Map of engine_id -> num_blocks.
        self.dst_num_blocks: dict[str, int] = {}
        self._registered_descs: list[Any] = []

        # In progress transfers.
        # [req_id -> list[handle]]
        self._recving_transfers: defaultdict[str, list[Any]] = defaultdict(
            list[Any])

        # Complete transfer tracker. Used by the rank 0 to track finished
        # transactions on ranks 1 to N-1.
        # [req_id -> count]
        self._done_recving_count: defaultdict[str,
                                              int] = defaultdict(lambda: 0)
        self._done_sending_count: defaultdict[str,
                                              int] = defaultdict(lambda: 0)

        # Background thread for establishing new connections.
        self._nixl_handshake_listener_t: Optional[threading.Thread] = None

    @staticmethod
    def _nixl_handshake_listener(metadata: NixlAgentMetadata,
                                 ready_event: threading.Event, rank: int):
        """Background thread for getting new NIXL handshakes."""
        # NOTE(rob): this is a simple implementation. We will move
        # to a better approach like an ETCD server in the future.

        # NOTE(rob): to support heterogeneous TP, we will have to
        # move this into the scheduler rather than worker, since
        # each rank needs the metadata of all other ranks (whereas
        # in this setup, each rank only gets one other rank's meta.

        encoder = msgspec.msgpack.Encoder()
        encoded_data = encoder.encode(metadata)
        size_in_bytes = len(encoded_data)
        logger.debug("Size of encoded NixlAgentMetadata: %s bytes",
                     str(size_in_bytes))

        # Listen for new requests for metadata.
        host = envs.VLLM_NIXL_SIDE_CHANNEL_HOST
        # NOTE(rob): we need each rank to have a unique port. This
        # hack to keeps us moving. We will switch when moving to etcd
        # or where we have a single ZMQ socket in the scheduler.
        port = envs.VLLM_NIXL_SIDE_CHANNEL_PORT + rank
        path = f"tcp://{host}:{port}"
        logger.debug("Starting listening on path: %s", path)
        with zmq_ctx(zmq.ROUTER, path) as sock:
            ready_event.set()
            while True:
                identity, _, msg = sock.recv_multipart()
                if msg != GET_META_MSG:
                    logger.warning(
                        "Connection listener got unexpected message %s", msg)
                sock.send_multipart((identity, b"", encoded_data))

    def _nixl_handshake(self, host: str, port: int):
        """Do a NIXL handshake with a remote instance."""

        start_time = time.perf_counter()
        # NOTE(rob): we need each rank to have a unique port. This is
        # a hack to keep us moving. We will switch when moving to etcd
        # or where we have a single ZMQ socket in the scheduler.
        path = f"tcp://{host}:{port + self.rank}"
        logger.debug("Querying metadata on path: %s", path)
        with zmq_ctx(zmq.REQ, path) as sock:
            # Send query for the request.
            sock.send(GET_META_MSG)
            metadata_bytes = sock.recv()
            decoder = msgspec.msgpack.Decoder(NixlAgentMetadata)
            metadata = decoder.decode(metadata_bytes)
            got_metadata_time = time.perf_counter()

            # Register Remote agent.
            self.add_remote_agent(metadata)
            setup_agent_time = time.perf_counter()

            logger.debug("NIXL handshake: get metadata took: %s",
                         got_metadata_time - start_time)
            logger.debug("NIXL handshake: add agent took: %s",
                         setup_agent_time - got_metadata_time)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register the KV Cache data in nixl."""

        _, first_kv_cache = next(iter(kv_caches.items()))
        kv_elem_size = first_kv_cache.element_size()

        # TODO(tms): Find a more robust way to detect and handle MLA
        use_mla = len(first_kv_cache.shape) == 3
        if use_mla:
            # MLA case.
            self.num_blocks = first_kv_cache.shape[0]
            block_rank = 2  # [block_size, latent_dim]
            block_shape = first_kv_cache.shape[-block_rank:]
        else:
            # [2 (k and v), num_blocks, ...]
            self.num_blocks = first_kv_cache.shape[1]
            block_rank = 3  # [block_size, kv_heads, head_dim]
            block_shape = first_kv_cache.shape[-block_rank:]

        # TODO(tms): self.block_len needs to be per-layer for sliding window,
        # hybrid attn, etc
        self.block_len = kv_elem_size * math.prod(block_shape)

        logger.debug("Registering KV_Caches. use_mla: %s, shape %s", use_mla,
                     first_kv_cache.shape)
        logger.debug("num_blocks: %s, block_shape: %s", self.num_blocks,
                     block_shape)
        logger.debug("Per layer kv cache size: %s", first_kv_cache.shape)
        self.dst_num_blocks[self.engine_id] = self.num_blocks
        self.kv_caches = kv_caches
        kv_caches_base_addr = []
        caches_data = []

        # Note(tms): I modified this from the original region setup code.
        # K and V are now in different regions. Advantage is that we can
        # elegantly support MLA and any cases where the K and V tensors
        # are non-contiguous (it's not locally guaranteed that they will be)
        # Disadvantage is that the encoded NixlAgentMetadata is now larger
        # (roughly 8KB vs 5KB).
        for cache_or_caches in kv_caches.values():
            # Normalize to always be a list of caches
            cache_list = [cache_or_caches] if use_mla else cache_or_caches
            for cache in cache_list:
                base_addr = cache.data_ptr()
                region_len = self.num_blocks * self.block_len
                caches_data.append((base_addr, region_len, self.rank, ""))
                kv_caches_base_addr.append(base_addr)
        self.kv_caches_base_addr[self.engine_id] = kv_caches_base_addr
        self.num_regions = len(caches_data)

        descs = self.nixl_wrapper.get_reg_descs(caches_data, "VRAM")
        logger.debug("Registering descs: %s", caches_data)
        self.nixl_wrapper.register_memory(descs)
        logger.debug("Done registering descs")

        self._registered_descs.append(descs)

        # After KV Caches registered, listen for new connections.
        metadata = NixlAgentMetadata(
            engine_id=self.engine_id,
            agent_metadata=self.nixl_wrapper.get_agent_metadata(),
            kv_caches_base_addr=self.kv_caches_base_addr[self.engine_id],
            num_blocks=self.num_blocks,
        )
        ready_event = threading.Event()
        self._nixl_handshake_listener_t = threading.Thread(
            target=self._nixl_handshake_listener,
            args=(metadata, ready_event, self.rank),
            daemon=True,
            name="nixl_handshake_listener")
        self._nixl_handshake_listener_t.start()
        ready_event.wait()

    def add_remote_agent(self, nixl_agent_meta: NixlAgentMetadata):
        engine_id = nixl_agent_meta.engine_id
        if engine_id in self._remote_agents:
            return

        self._remote_agents[engine_id] = self.nixl_wrapper.add_remote_agent(
            nixl_agent_meta.agent_metadata)
        self.kv_caches_base_addr[
            engine_id] = nixl_agent_meta.kv_caches_base_addr

        # Create src descs and xfer side handles.
        blocks_data = []
        for base_addr in self.kv_caches_base_addr[self.engine_id]:
            for block_id in range(self.num_blocks):
                block_offset = block_id * self.block_len
                # (addr, len, device id)
                blocks_data.append(
                    (base_addr + block_offset, self.block_len, self.rank))
        logger.debug("Created %s blocks for src engine %s and rank %s",
                     len(blocks_data), self.engine_id, self.rank)

        # Register with NIXL.
        descs = self.nixl_wrapper.get_xfer_descs(blocks_data, "VRAM")
        self.src_xfer_side_handle = self.nixl_wrapper.prep_xfer_dlist(
            "NIXL_INIT_AGENT", descs)

        # Create dst descs and xfer side handles.
        self.dst_num_blocks[engine_id] = nixl_agent_meta.num_blocks
        blocks_data = []
        for base_addr in self.kv_caches_base_addr[engine_id]:
            for block_id in range(nixl_agent_meta.num_blocks):
                block_offset = block_id * self.block_len
                # (addr, len, device id)
                blocks_data.append(
                    (base_addr + block_offset, self.block_len, self.rank))
        logger.debug("Created %s blocks for dst engine %s and rank %s",
                     len(blocks_data), engine_id, self.rank)

        # Register with NIXL.
        descs = self.nixl_wrapper.get_xfer_descs(blocks_data, "VRAM")
        self.dst_xfer_side_handles[
            engine_id] = self.nixl_wrapper.prep_xfer_dlist(
                self._remote_agents[engine_id], descs)

    def get_finished(self) -> tuple[set[str], set[str]]:
        """
        Get requests that are done sending or recving.

        In TP>1 setup, each rank exchanges KVs with its counterpart
        ranks independently. get_finished() runs in a worker creates
        the done_sending and done_recving sets that are sent to the
        scheduler via ModelRunnerOutput by Rank 0. To ensure trnxs
        are done before adding to finished, Ranks 1 to N-1 communicate
        to Rank 0 once their transaction is done + Rank 0 returns
        finished sets to Scheduler only once all ranks are done.
        """
        done_sending = self._get_new_notifs()
        done_recving = self._pop_done_transfers(self._recving_transfers)
        if len(done_sending) > 0 or len(done_recving) > 0:
            logger.debug(
                "Rank %s, get_finished: %s requests done sending "
                "and %s requests done recving", self.rank, len(done_sending),
                len(done_recving))

        if self.world_size == 1:
            return done_sending, done_recving

        # Rank 0: get finished from all other ranks.
        if self.rank == 0:
            for req_id in done_sending:
                self._done_sending_count[req_id] += 1
            for req_id in done_recving:
                self._done_recving_count[req_id] += 1

            # Keep track of how many other ranks have finished.
            other_ranks_finished_ids: list[str] = []
            for i in range(1, self.world_size):
                other_ranks_finished_ids.extend(
                    self.tp_group.recv_object(src=i))
            for req_id in other_ranks_finished_ids:
                if (req_id in self._done_recving_count
                        or req_id in self._recving_transfers):
                    self._done_recving_count[req_id] += 1
                else:
                    self._done_sending_count[req_id] += 1

            # Return ids that finished on all ranks to the scheduler.
            all_done_recving: set[str] = set()
            for req_id in list(self._done_recving_count.keys()):
                if self._done_recving_count[req_id] == self.world_size:
                    del self._done_recving_count[req_id]
                    all_done_recving.add(req_id)

            all_done_sending: set[str] = set()
            for req_id in list(self._done_sending_count.keys()):
                if self._done_sending_count[req_id] == self.world_size:
                    del self._done_sending_count[req_id]
                    all_done_sending.add(req_id)

            return all_done_sending, all_done_recving

        # Ranks 1 to N-1: send finished ids to Rank 0.
        else:
            finished_req_ids = list(done_recving.union(done_sending))
            self.tp_group.send_object(finished_req_ids, dst=0)

            # Unused as only Rank 0 results are sent to scheduler.
            return done_sending, done_recving

    def _get_new_notifs(self) -> set[str]:
        """Get req_ids which got a remote xfer message."""

        notified_req_ids: set[str] = set()
        for req_ids in self.nixl_wrapper.get_new_notifs().values():
            for req_id in req_ids:
                assert req_id not in notified_req_ids
                notified_req_ids.add(req_id.decode("utf-8"))
        return notified_req_ids

    def _pop_done_transfers(self, transfers: dict[str, list[int]]) -> set[str]:
        """
        Pop completed xfers by checking for DONE state.
        Args:
            transfers: dict of req_id -> list[running_xfer]
        Returns:
            set of req_ids that have all done xfers
        """
        done_req_ids: set[str] = set()
        for req_id, handles in list(transfers.items()):
            running_reqs = []
            for handle in handles:
                xfer_state = self.nixl_wrapper.check_xfer_state(handle)
                if xfer_state == "DONE":
                    # TODO ptarasiewicz: why abort is throwing errors?
                    # self.nixl_wrapper.release_xfer_handle(handle)
                    continue
                if xfer_state == "PROC":
                    running_reqs.append(handle)
                else:
                    raise RuntimeError("Transfer failed with state %s",
                                       xfer_state)
            if len(running_reqs) == 0:
                done_req_ids.add(req_id)
                del transfers[req_id]
            else:
                transfers[req_id] = running_reqs
        return done_req_ids

    def start_load_kv(self, metadata: NixlConnectorMetadata):
        """
        Start loading by triggering non-blocking nixl_xfer.
        We check for these trnxs to complete in each step().
        """
        for req_id, meta in metadata.requests.items():
            logger.debug(
                "start_load_kv for request %s from remote engine %s. "
                "Num local_block_ids: %s. Num remote_block_ids: %s. ", req_id,
                meta.remote_engine_id, len(meta.local_block_ids),
                len(meta.remote_block_ids))
            self._read_blocks(
                request_id=req_id,
                dst_engine_id=meta.remote_engine_id,
                local_block_ids=meta.local_block_ids,
                remote_block_ids=meta.remote_block_ids,
                remote_host=meta.remote_host,
                remote_port=meta.remote_port,
            )

    def _read_blocks(
        self,
        local_block_ids: list[int],
        remote_block_ids: list[int],
        remote_host: str,
        remote_port: int,
        dst_engine_id: str,
        request_id: str,
    ):
        # NOTE(rob): this takes ~2s. We need to get this off the hotpath.
        if dst_engine_id not in self._remote_agents:
            self._nixl_handshake(remote_host, remote_port)

        # NOTE(rob): having the staging blocks be on the READER side is
        # not going to work well (since we will have to call rearrange tensors).
        # after we detect the txn is complete (which means we cannot make the
        # read trxn async easily). If we want to make "READ" happen cleanly,
        # then we will need to have the staging blocks on the remote side.

        # NOTE(rob): according to nvidia the staging blocks are used to
        # saturate IB with heterogeneous TP sizes. We should remove the staging
        # blocks until we are ready.

        # Full prefix cache hit: do not need to read remote blocks,
        # just notify P worker that we have the blocks we need.
        num_local_blocks = len(local_block_ids)
        if num_local_blocks == 0:
            self.nixl_wrapper.send_notif(dst_engine_id,
                                         notif_msg=request_id.encode("utf-8"))
            return

        # Partial prefix cache hit: just read uncomputed blocks.
        num_remote_blocks = len(remote_block_ids)
        assert num_local_blocks <= num_remote_blocks
        if num_local_blocks < num_remote_blocks:
            remote_block_ids = remote_block_ids[-num_local_blocks:]

        # Get side handles.
        local_xfer_side_handle = self.src_xfer_side_handle
        remote_xfer_side_handle = self.dst_xfer_side_handles[dst_engine_id]

        # Get descs ids.
        remote_block_descs_ids = self._get_block_descs_ids(
            dst_engine_id, remote_block_ids)
        local_block_descs_ids = self._get_block_descs_ids(
            self.engine_id, local_block_ids)
        assert len(local_block_descs_ids) == len(remote_block_descs_ids)

        # Prepare transfer with Nixl.
        handle = self.nixl_wrapper.make_prepped_xfer(
            "READ",
            local_xfer_side_handle,
            local_block_descs_ids,
            remote_xfer_side_handle,
            remote_block_descs_ids,
            notif_msg=request_id.encode("utf-8"),
        )

        # Begin async xfer.
        self.nixl_wrapper.transfer(handle)

        # Use handle to check completion in future step().
        self._recving_transfers[request_id].append(handle)

    def _get_block_descs_ids(self, engine_id: str,
                             block_ids: list[int]) -> list[int]:
        """Get the descs ids for a set of block ids."""

        # range(1) for MLA, range(2) otherwise.
        region_ids = range(self.num_regions)
        num_blocks = self.dst_num_blocks[engine_id]

        # Compute the desc ids for each block.
        descs_ids: list[int] = []
        for reg_id in region_ids:
            for block_id in block_ids:
                descs_ids.append(reg_id * num_blocks + block_id)
        return descs_ids


@contextlib.contextmanager
def zmq_ctx(socket_type: Any, addr: str) -> Iterator[zmq.Socket]:
    """Context manager for a ZMQ socket"""

    ctx: Optional[zmq.Context] = None
    try:
        ctx = zmq.Context()  # type: ignore[attr-defined]

        if socket_type == zmq.ROUTER:
            socket = ctx.socket(zmq.ROUTER)
            socket.bind(addr)
        elif socket_type == zmq.REQ:
            socket = ctx.socket(zmq.REQ)
            socket.connect(addr)
        else:
            raise ValueError(f"Unexpected socket type: {socket_type}")

        yield socket
    finally:
        if ctx is not None:
            ctx.destroy(linger=0)
