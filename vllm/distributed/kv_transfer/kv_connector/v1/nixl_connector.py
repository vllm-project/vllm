# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import logging
import math
import queue
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import msgspec
import torch
import zmq

from vllm import envs
from vllm.attention.selector import backend_name_to_enum, get_attn_backend
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    CopyBlocksOp, KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size,
    get_tp_group)
from vllm.distributed.utils import divide
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.platforms import _Backend, current_platform
from vllm.utils import make_zmq_path, make_zmq_socket
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

Transfer = tuple[int, float]  # (xfer_handle, start_time)
EngineId = str
ReqId = str

GET_META_MSG = b"get_meta_msg"

logger = init_logger(__name__)

# Lazy import nixl_wrapper to avoid loading nixl_bindings if nixl is not used
try:
    from nixl._api import nixl_agent as NixlWrapper
    logger.info("NIXL is available")
except ImportError:
    logger.warning("NIXL is not available")
    NixlWrapper = None

# Supported xPUs and types of kv transfer buffer.
# {xPU: tuple of supported kv buffer types}
_NIXL_SUPPORTED_XPUS = {
    "cuda": ("cuda", ),
    "tpu": ("cpu", ),
}


class NixlAgentMetadata(
        msgspec.Struct,
        omit_defaults=True,  # type: ignore[call-arg]
        # required for @cached_property.
        dict=True):
    engine_id: str
    agent_metadata: bytes
    kv_caches_base_addr: list[int]
    num_blocks: int
    block_len: int
    attn_backend_name: str


@dataclass
class ReqMeta:
    local_block_ids: list[int]
    remote_block_ids: list[int]
    remote_host: str
    remote_port: int
    remote_engine_id: str
    tp_size: int


class NixlConnectorMetadata(KVConnectorMetadata):

    def __init__(self):
        self.reqs_to_recv: dict[ReqId, ReqMeta] = {}
        self.reqs_to_save: dict[ReqId, ReqMeta] = {}
        self.reqs_to_send: dict[ReqId, float] = {}

    def add_new_req(
        self,
        request_id: ReqId,
        local_block_ids: list[int],
        kv_transfer_params: dict[str, Any],
        load_remote_cache: bool = True,
        save_to_host: bool = False,
    ):
        # save and load are mutually exclusive
        assert load_remote_cache ^ save_to_host
        _req = ReqMeta(
            local_block_ids=local_block_ids,
            remote_block_ids=kv_transfer_params["remote_block_ids"],
            remote_engine_id=kv_transfer_params["remote_engine_id"],
            remote_host=kv_transfer_params["remote_host"],
            remote_port=kv_transfer_params["remote_port"],
            # P workers don't need to receive tp_size from proxy here.
            tp_size=kv_transfer_params.get("tp_size", 1),
        )
        if save_to_host:
            self.reqs_to_save[request_id] = _req
        if load_remote_cache:
            self.reqs_to_recv[request_id] = _req


class NixlConnector(KVConnectorBase_V1):

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole):
        assert vllm_config.kv_transfer_config is not None
        assert vllm_config.kv_transfer_config.engine_id is not None
        self.engine_id: EngineId = vllm_config.kv_transfer_config.engine_id

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler: Optional[NixlConnectorScheduler] = \
                NixlConnectorScheduler(vllm_config, self.engine_id)
            self.connector_worker: Optional[NixlConnectorWorker] = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = NixlConnectorWorker(
                vllm_config, self.engine_id)

    ############################################################
    # Class Methods
    ############################################################
    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig):
        if vllm_config.model_config is None:
            logger.warning_once("Unable to detect current VLLM config. "
                                "Fallback to default kv cache layout.")
            return None
        use_mla = vllm_config.model_config.use_mla
        if use_mla:
            # return None when we have mla
            # as the layout should not matter in that case,
            # which fallback to the default behavior.
            return None
        logger.info_once("NixlConnector setting KV cache "
                         "layout to HND for better xfer performance.")
        return "HND"

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

    def set_host_xfer_buffer_ops(self, copy_operation: CopyBlocksOp):
        assert self.connector_worker is not None
        self.connector_worker.set_host_xfer_buffer_ops(copy_operation)

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
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, NixlConnectorMetadata)
        if self.connector_worker.use_host_buffer and \
           self.connector_worker.copy_blocks:
            self.connector_worker.save_kv_to_host(self._connector_metadata)


class NixlConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self.engine_id: EngineId = engine_id
        self.side_channel_host = envs.VLLM_NIXL_SIDE_CHANNEL_HOST
        self.side_channel_port = (
            envs.VLLM_NIXL_SIDE_CHANNEL_PORT +
            vllm_config.parallel_config.data_parallel_rank *
            vllm_config.parallel_config.tensor_parallel_size)
        self.use_host_buffer = \
            vllm_config.kv_transfer_config.kv_buffer_device == "cpu"
        logger.info("Initializing NIXL Scheduler %s", engine_id)

        # Requests that need to start recv/send.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        self._reqs_need_recv: dict[ReqId, tuple[Request, list[int]]] = {}
        self._reqs_need_save: dict[ReqId, tuple[Request, list[int]]] = {}
        # Reqs to send and their expiration time
        self._reqs_need_send: dict[ReqId, float] = {}

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

        params = request.kv_transfer_params
        logger.debug(
            "NIXLConnector get_num_new_matched_tokens: "
            "num_computed_tokens=%s, kv_transfer_params=%s",
            num_computed_tokens, params)

        if params is not None and params.get("do_remote_prefill"):
            # Remote prefill: get all prompt blocks from remote.
            count = len(request.prompt_token_ids) - num_computed_tokens
            if count > 0:
                return count, True

        # No remote prefill for this request.
        return 0, False

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):

        params = request.kv_transfer_params
        logger.debug(
            "NIXLConnector update_state_after_alloc: "
            "num_external_tokens=%s, kv_transfer_params=%s",
            num_external_tokens, params)

        if not params:
            return
        if self.use_host_buffer and params.get("do_remote_decode"):
            # NOTE: when accelerator is not directly supported by Nixl,
            # prefilled blocks need to be saved to host memory before transfer.

            # save all blocks
            block_ids = blocks.get_block_ids()[0]
            # TODO: skip the blocks that are already in the host xfer buffer.
            # Currently, the host xfer buffer block is 1-to-1 mapped to device
            # kv blocks, so host blocks won't be flushed as long as its device
            # block is not overwritten; and it will be safe to skip saving them
            # to host xfer buffer.
            if block_ids:
                self._reqs_need_save[request.request_id] = \
                    (request, block_ids)
        elif params.get("do_remote_prefill"):
            if params.get("remote_block_ids"):
                if all(p in params for p in ("remote_engine_id", "remote_host",
                                             "remote_port")):
                    # If remote_blocks and num_external_tokens = 0, we have
                    # a full prefix cache hit on the D worker. We need to call
                    # send_notif in _read_blocks to free the memory on the P.
                    local_block_ids = (blocks.get_unhashed_block_ids()
                                       if num_external_tokens > 0 else [])
                    # Get unhashed blocks to pull from remote.
                    self._reqs_need_recv[request.request_id] = (
                        request, local_block_ids)

                else:
                    logger.warning(
                        "Got invalid KVTransferParams: %s. This "
                        "request will not utilize KVTransfer", params)
            else:
                assert num_external_tokens == 0
            # Only trigger 1 KV transfer per request.
            params["do_remote_prefill"] = False

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = NixlConnectorMetadata()

        # Loop through scheduled reqs and convert to ReqMeta.
        for req_id, (req, block_ids) in self._reqs_need_recv.items():
            assert req.kv_transfer_params is not None
            meta.add_new_req(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=req.kv_transfer_params,
            )

        for req_id, (req, block_ids) in self._reqs_need_save.items():
            assert req.kv_transfer_params is not None
            meta.add_new_req(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=req.kv_transfer_params,
                load_remote_cache=False,
                save_to_host=True,
            )

        meta.reqs_to_send = self._reqs_need_send

        # Clear the list once workers start the transfers
        self._reqs_need_recv.clear()
        self._reqs_need_save.clear()
        self._reqs_need_send = {}

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

        params = request.kv_transfer_params
        logger.debug(
            "NIXLConnector request_finished, request_status=%s, "
            "kv_transfer_params=%s", request.status, params)
        if not params:
            return False, None

        if params.get("do_remote_prefill"):
            # If do_remote_prefill is still True when the request is finished,
            # update_state_after_alloc must not have been called (the request
            # must have been aborted before it was scheduled).
            # To avoid stranding the prefill blocks in the prefill instance,
            # we must add empty block_ids to _reqs_need_recv so that our
            # worker side will notify and free blocks in the prefill instance.
            self._reqs_need_recv[request.request_id] = (request, [])
            params["do_remote_prefill"] = False
            return False, None

        if (not params.get("do_remote_decode")
                or request.status != RequestStatus.FINISHED_LENGTH_CAPPED):
            return False, None

        # TODO: check whether block_ids actually ever be 0. If not we could
        # remove the conditional below
        delay_free_blocks = len(block_ids) > 0

        if delay_free_blocks:
            # Prefill request on remote. It will be read from D upon completion
            self._reqs_need_send[request.request_id] = time.perf_counter(
            ) + envs.VLLM_NIXL_ABORT_REQUEST_TIMEOUT

        return delay_free_blocks, dict(
            do_remote_prefill=True,
            do_remote_decode=False,
            remote_block_ids=block_ids,
            remote_engine_id=self.engine_id,
            remote_host=self.side_channel_host,
            remote_port=self.side_channel_port,
            tp_size=self.vllm_config.parallel_config.tensor_parallel_size)


class NixlConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        if NixlWrapper is None:
            logger.error("NIXL is not available")
            raise RuntimeError("NIXL is not available")
        logger.info("Initializing NIXL wrapper")
        logger.info("Initializing NIXL worker %s", engine_id)

        # Config.
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size

        # Agent.
        self.nixl_wrapper = NixlWrapper(str(uuid.uuid4()), None)
        # Map of engine_id -> {rank0: agent_name0, rank1: agent_name1..}.
        self._remote_agents: dict[EngineId, dict[int, str]] = defaultdict(dict)

        # NIXL handshake port.
        # NOTE(rob): Within a DP group, each DP rank gets its own
        # base port (which is sent in the KVTransferParams).
        # Each TP rank listens/queries on the base_port + tp_rank.
        self.side_channel_port: int = (
            envs.VLLM_NIXL_SIDE_CHANNEL_PORT +
            vllm_config.parallel_config.data_parallel_rank *
            vllm_config.parallel_config.tensor_parallel_size)

        # Metadata.
        self.engine_id: EngineId = engine_id
        self.tp_rank = get_tensor_model_parallel_rank()
        self.world_size = get_tensor_model_parallel_world_size()
        self.tp_group = get_tp_group()
        self.num_blocks = 0

        # KV Caches and nixl tracking data.
        self.device_type = current_platform.device_type
        self.kv_buffer_device: str = \
            vllm_config.kv_transfer_config.kv_buffer_device
        if self.device_type not in _NIXL_SUPPORTED_XPUS:
            raise RuntimeError(f"{self.device_type} is not supported.")
        elif self.kv_buffer_device not in _NIXL_SUPPORTED_XPUS[
                self.device_type]:
            raise RuntimeError(
                f"{self.device_type} with {self.kv_buffer_device} kv_buffer "
                "is not supported.")
        self.device_kv_caches: dict[str, torch.Tensor] = {}

        # cpu kv buffer for xfer
        # used when xPU memory can not be registered under nixl
        self.host_xfer_buffers: dict[str, torch.Tensor] = {}
        self.use_host_buffer = self.kv_buffer_device == "cpu"
        if self.kv_buffer_device == "cuda":
            self.nixl_memory_type = "VRAM"
        elif self.kv_buffer_device == "cpu":
            self.nixl_memory_type = "DRAM"
        else:
            raise RuntimeError(
                f"{self.device_type} with {self.kv_buffer_device} kv_buffer "
                "is not supported.")

        # Note: host xfer buffer ops when use_host_buffer is True
        self.copy_blocks: Optional[CopyBlocksOp] = None

        # Map of engine_id -> kv_caches_base_addr. For TP case, each local
        # rank will still only pull from a single remote TP worker.
        self.kv_caches_base_addr: dict[EngineId, list[int]] = {}

        # Number of NIXL regions. Currently one region per cache
        # (so 1 per layer for MLA, otherwise 2 per layer)
        self.num_regions = 0
        self.num_layers = 0

        # nixl_prepped_dlist_handle.
        self.src_xfer_side_handle: int = 0
        # Map of engine_id -> nixl_prepped_dlist_handle (int)].
        self.dst_xfer_side_handles: dict[EngineId, int] = {}

        # Map of engine_id -> num_blocks. All ranks in the same deployment will
        # have the same number of blocks.
        self.dst_num_blocks: dict[EngineId, int] = {}
        self._registered_descs: list[Any] = []

        # In progress transfers.
        # [req_id -> list[handle]]
        self._recving_metadata: dict[ReqId, ReqMeta] = {}
        self._recving_transfers = defaultdict[ReqId, list[Transfer]](list)
        # Track the expiration time of requests that are waiting to be sent.
        self._reqs_to_send: dict[ReqId, float] = {}

        # Background thread for handling new handshake requests.
        self._nixl_handshake_listener_t: Optional[threading.Thread] = None
        # Background thread for initializing new NIXL handshakes.
        self._handshake_initiation_executor = ThreadPoolExecutor(
            # NIXL is not guaranteed to be thread-safe, limit 1 worker.
            max_workers=1,
            thread_name_prefix="vllm-nixl-handshake-initiator")
        self._ready_requests = queue.Queue[tuple[ReqId, ReqMeta]]()
        self._handshake_futures: dict[EngineId, Future[dict[int, str]]] = {}
        # Protects _handshake_futures and _remote_agents.
        self._handshake_lock = threading.RLock()

        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config

        # TODO(mgoin): remove this once we have hybrid memory allocator
        # Optimization for models with local attention (Llama 4)
        # List of block window sizes for each layer for local attention
        self.block_window_per_layer: list[Optional[int]] = []
        self.use_mla = self.model_config.use_mla

        backend = get_attn_backend(self.model_config.get_head_size(),
                                   self.model_config.dtype,
                                   self.cache_config.cache_dtype,
                                   self.block_size,
                                   self.model_config.is_attention_free,
                                   use_mla=self.use_mla)
        self.backend_name = backend.get_name()
        attn_backend = backend_name_to_enum(self.backend_name)
        self._use_flashinfer = attn_backend == _Backend.FLASHINFER_VLLM_V1
        self._use_pallas_v1 = attn_backend == _Backend.PALLAS_VLLM_V1
        logger.debug("Detected attention backend %s", self.backend_name)

        self._tp_size: dict[EngineId, int] = {self.engine_id: self.world_size}
        # With heterogeneous TP, P must wait for all assigned D TP workers to
        # finish reading before safely freeing the blocks.
        self.consumer_notification_counts_by_req = defaultdict[ReqId, int](int)

    def __del__(self):
        """Cleanup background threads on destruction."""
        self._handshake_initiation_executor.shutdown(wait=False)
        if self._nixl_handshake_listener_t:
            self._nixl_handshake_listener_t.join(timeout=0)

    @staticmethod
    def _nixl_handshake_listener(metadata: NixlAgentMetadata,
                                 ready_event: threading.Event, base_port: int,
                                 tp_rank: int):
        """Background thread for getting new NIXL handshakes."""
        # NOTE(rob): this is a simple implementation. We will move
        # to a better approach via HTTP endpoint soon.

        encoder = msgspec.msgpack.Encoder()
        encoded_data = encoder.encode(metadata)
        size_in_bytes = len(encoded_data)
        logger.debug("Size of encoded NixlAgentMetadata: %s bytes",
                     str(size_in_bytes))

        # Listen for new requests for metadata.
        host = envs.VLLM_NIXL_SIDE_CHANNEL_HOST
        path = make_zmq_path("tcp", host, base_port + tp_rank)
        logger.debug("Starting listening on path: %s", path)
        with zmq_ctx(zmq.ROUTER, path) as sock:
            ready_event.set()
            while True:
                identity, _, msg = sock.recv_multipart()
                if msg != GET_META_MSG:
                    logger.warning(
                        "Connection listener got unexpected message %s", msg)
                sock.send_multipart((identity, b"", encoded_data))

    def _nixl_handshake(
        self,
        host: str,
        port: int,
        remote_tp_size: int,
        expected_engine_id: str,
    ) -> dict[int, str]:
        """Do a NIXL handshake with a remote instance."""

        start_time = time.perf_counter()

        # NOTE(rob): we need each rank to have a unique port. This is
        # a hack to keep us moving. We will switch when moving to etcd
        # or where we have a single ZMQ socket in the scheduler.

        # Handshake only with the remote TP rank that current local rank will
        # pull from. With homogeneous TP it happens to be the same rank_i.
        tp_ratio = self._tp_size[self.engine_id] // remote_tp_size
        p_remote_rank = self.tp_rank // tp_ratio
        path = make_zmq_path("tcp", host, port + p_remote_rank)
        logger.debug("Querying metadata on path: %s at remote rank %s", path,
                     p_remote_rank)

        # Send query for the request.
        with zmq_ctx(zmq.REQ, path) as sock:
            sock.send(GET_META_MSG)
            metadata_bytes = sock.recv()
            decoder = msgspec.msgpack.Decoder(NixlAgentMetadata)
            metadata = decoder.decode(metadata_bytes)
            got_metadata_time = time.perf_counter()
            logger.debug("NIXL handshake: get metadata took: %s",
                         got_metadata_time - start_time)

            # Ensure engine id matches.
            if metadata.engine_id != expected_engine_id:
                raise RuntimeError(f"Remote NIXL agent engine ID mismatch. "
                                   f"Expected {expected_engine_id},"
                                   f"received {metadata.engine_id}.")

            # Register Remote agent.
            remote_agent_name = self.add_remote_agent(metadata, p_remote_rank,
                                                      remote_tp_size)
            setup_agent_time = time.perf_counter()
            logger.debug("NIXL handshake: add agent took: %s",
                         setup_agent_time - got_metadata_time)

        # Remote rank -> agent name.
        return {p_remote_rank: remote_agent_name}

    def initialize_host_xfer_buffer(
            self, kv_caches: dict[str, torch.Tensor]) -> None:
        """
        Initialize transfer buffer in CPU mem for accelerators
        NOT directly supported by NIXL (e.g., tpu)
        """
        xfer_buffers: dict[str, torch.Tensor] = {}
        try:
            for layer_name, kv_cache in kv_caches.items():
                kv_shape = kv_cache.shape
                kv_dtype = kv_cache.dtype
                xfer_buffers[layer_name] = torch.empty(kv_shape,
                                                       dtype=kv_dtype,
                                                       device="cpu")
        except MemoryError as e:
            logger.error("NIXLConnectorWorker gets %s.", e)
            raise

        self.host_xfer_buffers = xfer_buffers

    def set_host_xfer_buffer_ops(self, copy_operation: CopyBlocksOp):
        """Assign copy (d2h, h2d) operations when host buffer is used."""
        assert self.use_host_buffer
        self.copy_blocks = copy_operation

    def _background_nixl_handshake(self, req_id: str,
                                   remote_engine_id: EngineId, meta: ReqMeta):
        # Do NIXL handshake in background and add to _ready_requests when done.
        fut = self._handshake_futures.get(remote_engine_id)
        if fut is None:
            fut = self._handshake_initiation_executor.submit(
                self._nixl_handshake, meta.remote_host, meta.remote_port,
                meta.tp_size, remote_engine_id)
            self._handshake_futures[remote_engine_id] = fut

            def done_callback(f: Future[dict[int, str]], eid=remote_engine_id):
                with self._handshake_lock:
                    del self._handshake_futures[eid]
                    try:
                        self._remote_agents[eid] = f.result()
                    except Exception:
                        logger.exception("Handshake with %s failed", eid)

            fut.add_done_callback(done_callback)

        # TODO: handle failure state of future in the
        # callback, we want to fail the request in this case.
        def request_ready(_f: Future[Any], entry=(req_id, meta)):
            self._ready_requests.put(entry)

        fut.add_done_callback(request_ready)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register the KV Cache data in nixl."""

        _, first_kv_cache = next(iter(kv_caches.items()))
        kv_elem_size = first_kv_cache.element_size()

        if self.use_host_buffer:
            self.initialize_host_xfer_buffer(kv_caches=kv_caches)
            assert len(self.host_xfer_buffers) == len(kv_caches), (
                f"host_buffer: {len(self.host_xfer_buffers)}, "
                f"kv_caches: {len(kv_caches)}")
            xfer_buffers = self.host_xfer_buffers
        else:
            xfer_buffers = kv_caches
            assert not self.host_xfer_buffers, (
                "host_xfer_buffer should not be initialized when "
                f"kv_buffer_device is {self.kv_buffer_device}")

        # TODO(tms): Find a more robust way to detect and handle MLA
        # NOTE (NickLucche) To move blocks efficiently with NIXL, the expected
        # KV memory layout is HND, as opposed to the default NHD. Note that it
        # will only affects the strides. For MLA instead, we make require no
        # such thing and resort to the standard layout.
        use_mla = len(first_kv_cache.shape) == 3
        if self.device_type == "tpu":
            assert not use_mla, f"{self.kv_buffer_device} does not support MLA."
            assert self._use_pallas_v1, f"attn backend: {self.backend_name}"
            # tpu (v1) kv shape per layer:
            # (num_blocks, block_size, num_kv_heads * 2, head_size)
            self.num_blocks = first_kv_cache.shape[0]
            block_rank = 3  # [block_size, kv_heads, head_dim]
            block_shape = first_kv_cache.shape[-block_rank:]
            block_size, n_kv_heads_x_2, head_dim = block_shape
            self.slot_size_bytes = kv_elem_size * n_kv_heads_x_2 * head_dim
        elif self.device_type == "cuda":
            assert use_mla == self.use_mla
            # TODO (NickLucche) not compatible with hybrid allocator.
            # Enforce check once it goes live, as a single kv layout
            # is expected for xfers.
            if use_mla:
                # MLA case.
                self.num_blocks = first_kv_cache.shape[0]
                block_rank = 2  # [block_size, latent_dim]
                block_shape = first_kv_cache.shape[-block_rank:]
                block_size, kv_latent_dim = block_shape
                self.slot_size_bytes = kv_elem_size * kv_latent_dim
            else:
                # [2 (k and v), num_blocks, ...]
                if self._use_flashinfer:
                    # FlashInfer swaps 2<->num_blocks dimensions.
                    self.num_blocks = first_kv_cache.shape[0]
                    block_rank = 4  # [2, block_size, kv_heads, head_dim]
                else:
                    self.num_blocks = first_kv_cache.shape[1]
                    block_rank = 3  # [block_size, kv_heads, head_dim]
                block_shape = first_kv_cache.shape[-block_rank:]
                block_size, n_kv_heads, head_dim = block_shape[-3:]
                # head size in bytes.
                self.slot_size_bytes = kv_elem_size * n_kv_heads * head_dim
            assert block_size == self.block_size
        else:
            raise RuntimeError(
                f"{self.device_type} ({self.backend_name}) is not supported.")

        # TODO(tms): self.block_len needs to be per-layer for sliding window,
        # hybrid attn, etc
        # block size in bytes
        self.block_len = kv_elem_size * math.prod(block_shape)
        logger.info(
            "Registering KV_Caches. use_mla: %s, kv_buffer_device: %s, "
            "use_host_buffer: %s, num_blocks: %s, block_shape: %s, "
            "per_layer_kv_cache_shape: %s", use_mla, self.kv_buffer_device,
            self.use_host_buffer, self.num_blocks, block_shape,
            first_kv_cache.shape)
        self.dst_num_blocks[self.engine_id] = self.num_blocks
        self.device_kv_caches = kv_caches
        kv_caches_base_addr = []
        caches_data = []

        # Note(tms): I modified this from the original region setup code.
        # K and V are now in different regions. Advantage is that we can
        # elegantly support MLA and any cases where the K and V tensors
        # are non-contiguous (it's not locally guaranteed that they will be)
        # Disadvantage is that the encoded NixlAgentMetadata is now larger
        # (roughly 8KB vs 5KB).
        # Conversely for FlashInfer, K and V are transferred in the same tensor
        # to better exploit the memory layout (ie num_blocks is the first dim).
        for cache_or_caches in xfer_buffers.values():
            # Normalize to always be a list of caches
            cache_list = [cache_or_caches] if use_mla \
                         or self._use_pallas_v1 or self._use_flashinfer \
                         else cache_or_caches
            for cache in cache_list:
                base_addr = cache.data_ptr()
                region_len = self.num_blocks * self.block_len
                # NOTE: use tp_rank for device_id since multi-node TP
                # is rarely used.
                caches_data.append((base_addr, region_len, self.tp_rank, ""))
                kv_caches_base_addr.append(base_addr)
        self.kv_caches_base_addr[self.engine_id] = kv_caches_base_addr
        self.num_regions = len(caches_data)
        self.num_layers = len(xfer_buffers.keys())

        # TODO(mgoin): remove this once we have hybrid memory allocator
        # Optimization for models with local attention (Llama 4)
        if self.vllm_config.model_config.hf_config.model_type == "llama4":
            from transformers import Llama4TextConfig
            assert isinstance(self.vllm_config.model_config.hf_text_config,
                              Llama4TextConfig)
            llama4_config = self.vllm_config.model_config.hf_text_config
            no_rope_layers = llama4_config.no_rope_layers
            chunk_size = llama4_config.attention_chunk_size
            chunk_block_size = math.ceil(chunk_size / self.block_size)
            for layer_idx in range(self.num_layers):
                # no_rope_layers[layer_idx] == 0 means NoPE (global)
                # Any other value means RoPE (local chunked)
                is_local_attention = no_rope_layers[layer_idx] != 0
                block_window = chunk_block_size if is_local_attention else None
                self.block_window_per_layer.append(block_window)
            logger.debug("Llama 4 block window per layer mapping: %s",
                         self.block_window_per_layer)
            assert len(self.block_window_per_layer) == self.num_layers

        descs = self.nixl_wrapper.get_reg_descs(caches_data,
                                                self.nixl_memory_type)
        logger.debug("Registering descs: %s", caches_data)
        self.nixl_wrapper.register_memory(descs)
        logger.debug("Done registering descs")
        self._registered_descs.append(descs)

        # Register local/src descr for NIXL xfer.
        blocks_data = []
        for base_addr in self.kv_caches_base_addr[self.engine_id]:
            # NOTE With heter-TP, more blocks are prepared than what are
            # needed as self.num_blocks >= nixl_agent_meta.num_blocks. We
            # could create fewer, but then _get_block_descs_ids needs to
            # select agent_meta.num_blocks instead of self.num_blocks for
            # local descr, and that makes handling regular flow less clean.
            for block_id in range(self.num_blocks):
                block_offset = block_id * self.block_len
                addr = base_addr + block_offset
                # (addr, len, device id)
                # TODO: does device_id matter to DRAM?
                blocks_data.append((addr, self.block_len, self.tp_rank))
        logger.debug("Created %s blocks for src engine %s and rank %s",
                     len(blocks_data), self.engine_id, self.tp_rank)

        descs = self.nixl_wrapper.get_xfer_descs(blocks_data,
                                                 self.nixl_memory_type)
        # NIXL_INIT_AGENT to be used for preparations of local descs.
        self.src_xfer_side_handle = self.nixl_wrapper.prep_xfer_dlist(
            "NIXL_INIT_AGENT", descs)

        # After KV Caches registered, listen for new connections.
        metadata = NixlAgentMetadata(
            engine_id=self.engine_id,
            agent_metadata=self.nixl_wrapper.get_agent_metadata(),
            kv_caches_base_addr=self.kv_caches_base_addr[self.engine_id],
            num_blocks=self.num_blocks,
            block_len=self.block_len,
            attn_backend_name=self.backend_name)
        ready_event = threading.Event()
        self._nixl_handshake_listener_t = threading.Thread(
            target=self._nixl_handshake_listener,
            args=(metadata, ready_event, self.side_channel_port, self.tp_rank),
            daemon=True,
            name="nixl_handshake_listener")
        self._nixl_handshake_listener_t.start()
        ready_event.wait()  # Wait for listener ZMQ socket to be ready.

    def add_remote_agent(self,
                         nixl_agent_meta: NixlAgentMetadata,
                         remote_tp_rank: int = 0,
                         remote_tp_size: int = 1) -> str:
        """
        Add the remote NIXL agent and prepare the descriptors for reading cache
        blocks from remote.

        In particular, handle both homogeneous and heterogeneous TP. The former
        requires local rank_i to read from remote rank_i. 
        The latter, assuming D.world_size > P.world_size, requires that two or 
        more local TP worker share the xfer from a single TP worker.

        Here's an example:

        rank_offset     p_remote_tp_rank
        (kv split no)    
        --------------------------------
            0                 0      Worker0  ---- 1st half of KV ----> Worker0  [ KV Cache ]
                                                                        /
            1                 0      Worker1  ---- 2nd half of KV -----/

            0                 1      Worker2  ---- 1st half of KV ----> Worker1  [ KV Cache ]
                                                                        /
            1                 1      Worker3  ---- 2nd half of KV -----/


                                Decoder TP workers                     Prefix TP workers
                                  (world_size=4)                         (world_size=2)
                                                 tp_ratio = 4 // 2 = 2                  
                                
        Considering the KV Caches, if P-Worker_i has cache size [2, num_blocksP, kv_heads, block_size, head_dim]  
        then D-Worker_j has [2, num_blocksD, kv_heads//tp_ratio, block_size, head_dim]. Mind the "HND" layout format.
        Assuming num_blocksD >= num_blocksP, D-Worker0 reads from P-Worker0 by preparing the kv_heads//tp_ratio 
        first heads from all the slots of all the blocks. D-Worker1 will do the same, but reading the second split
        along the kv_heads dimension, and so forth until "tp_ratio" D TP workers have pulled from P-Worker0.   
        
        Note that the above will also hold true for the homogeneous TP case, where tp_ratio evaluates to 1.

        Regarding MLA case, the cache is replicated across TP workers so the rank_offset will just always be 0
        so that the whole cache is shared by "tp_ratio" D TP workers.
        """ # noqa: E501
        engine_id = nixl_agent_meta.engine_id
        # TODO re-evaluate refreshing for scaling/recovery
        if remote_tp_rank in self._remote_agents.get(engine_id, {}):
            return self._remote_agents[engine_id][remote_tp_rank]

        if engine_id not in self._tp_size:
            self._tp_size[engine_id] = remote_tp_size
        else:
            assert self._tp_size[engine_id] == remote_tp_size
        # We may eventually enable this after asserting equality in cache
        # layout and close outputs.
        assert nixl_agent_meta.attn_backend_name == self.backend_name

        remote_agent_name = self.nixl_wrapper.add_remote_agent(
            nixl_agent_meta.agent_metadata)

        # Number of D TP workers reading from a single P TP worker. This is
        # 1 when P and D `--tensor-parallel-size` match.
        tp_ratio = divide(self._tp_size[self.engine_id],
                          self._tp_size[engine_id])
        assert tp_ratio > 0, "Decode TP cannot be smaller than prefill TP"
        assert not self._use_pallas_v1 or tp_ratio == 1, \
               "TPU (pallas_v1) DOES NOT support heterogeneous TP yet."

        # Handle tp_size>num_kv_heads: replicate KV cache.
        total_num_kv_heads = self.model_config.get_total_num_kv_heads()
        is_kv_replicated = self._tp_size[engine_id] // total_num_kv_heads >= 1

        if self.use_mla or is_kv_replicated:
            # With MLA the only difference is in the number of blocks.
            remote_block_size = nixl_agent_meta.block_len // (
                self.slot_size_bytes)
            assert self.block_len == nixl_agent_meta.block_len
        else:
            remote_block_size = nixl_agent_meta.block_len // (
                self.slot_size_bytes * tp_ratio)
            if self._use_flashinfer:
                # Account for joint KV in FlashInfer.
                remote_block_size //= 2

            assert nixl_agent_meta.block_len == self.block_len * tp_ratio, (
                "Remote P worker KV layer cache must be of shape [2, N, "
                "local_kv_heads*tp_ratio, block_size, head_dim] and same dtype."
            )

        assert self.block_size == remote_block_size, (
            "Remote P worker with different block size is not supported "
            f"{self.block_size=} {remote_block_size=}")

        # Create dst descs and xfer side handles. TP workers have same #blocks.
        if engine_id in self.dst_num_blocks:
            assert self.dst_num_blocks[engine_id] == nixl_agent_meta.num_blocks
        else:
            self.dst_num_blocks[engine_id] = nixl_agent_meta.num_blocks

        blocks_data = []
        # With homogeneous TP, D pulls the whole kv cache from corresponding
        # rank. With heterogeneous TP, prepare the descriptors by splitting the
        # P KV cache along kv_head dim, of D worker's kv_head size (D>P).
        # Eg. PTP1 DTP2 => P0 KV:[block0-KV_0 | block0-KV_1..].
        # Only register the remote's descriptors if current rank pulls from it.
        self.kv_caches_base_addr[
            engine_id] = nixl_agent_meta.kv_caches_base_addr
        rank_offset = self.tp_rank % tp_ratio * self.block_len \
            if not (self.use_mla or is_kv_replicated) else 0
        # Register all remote blocks, but only the corresponding kv heads.
        for base_addr in nixl_agent_meta.kv_caches_base_addr:
            for block_id in range(nixl_agent_meta.num_blocks):
                block_offset = block_id * nixl_agent_meta.block_len
                # For each block, grab the heads chunk belonging to rank_i
                # of size remote_nheads // tp_ratio, which correspond to
                # self.block_len == remote_block_len//tp_ratio bytes.
                addr = base_addr + block_offset + rank_offset
                # (addr, len, device id)
                blocks_data.append((addr, self.block_len, remote_tp_rank))
        logger.debug(
            "Created %s blocks for dst engine %s with remote rank %s and "
            "local rank %s", len(blocks_data), engine_id, remote_tp_rank,
            self.tp_rank)

        # Register with NIXL.
        descs = self.nixl_wrapper.get_xfer_descs(blocks_data,
                                                 self.nixl_memory_type)
        self.dst_xfer_side_handles[
            engine_id] = self.nixl_wrapper.prep_xfer_dlist(
                remote_agent_name, descs)

        return remote_agent_name

    def sync_recved_kv_to_device(self, req_id: str, meta: ReqMeta):
        """copy recved kv from host buffer to device."""
        assert self.use_host_buffer
        assert self.copy_blocks is not None

        local_block_ids = meta.local_block_ids
        self.copy_blocks(self.host_xfer_buffers, self.device_kv_caches,
                         local_block_ids, local_block_ids, "h2d")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "synced recved kv of request[%s] to device kv buffer,"
                "local_block_ids: %s. ", req_id,
                ",".join(map(str, meta.local_block_ids)))

    def save_kv_to_host(self, metadata: NixlConnectorMetadata):
        """copy kv from device to host buffer."""
        assert self.use_host_buffer
        assert self.copy_blocks is not None

        for req_id, meta in metadata.reqs_to_save.items():
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "save_load_kv for request[%s] to host xfer buffer."
                    "local_block_ids: %s. ", req_id,
                    ",".join(map(str, meta.local_block_ids)))
            # blocking
            self.copy_blocks(self.device_kv_caches, self.host_xfer_buffers,
                             meta.local_block_ids, meta.local_block_ids, "d2h")

    def get_finished(self) -> tuple[set[str], set[str]]:
        """
        Get requests that are done sending or recving on this specific worker.
        The scheduler process (via the MultiprocExecutor) will use this output
        to track which workers are done.
        """
        done_sending = self._get_new_notifs()
        done_recving = self._pop_done_transfers(self._recving_transfers)
        if len(done_sending) > 0 or len(done_recving) > 0:
            logger.debug(
                "Rank %s, get_finished: %s requests done sending "
                "and %s requests done recving", self.tp_rank,
                len(done_sending), len(done_recving))

        if self.use_host_buffer:
            for req_id in done_recving:
                meta = self._recving_metadata.pop(req_id)
                assert meta, f"{req_id} not found in recving_metadata list"
                self.sync_recved_kv_to_device(req_id, meta)

        # Handle timeout to avoid stranding blocks on remote.
        now = time.perf_counter()
        while self._reqs_to_send:
            req_id, expires = next(iter(self._reqs_to_send.items()))
            # Sorted dict, oldest requests are put first so we can exit early.
            if now < expires:
                break
            count = self.consumer_notification_counts_by_req.pop(req_id, 0)
            logger.warning(
                "Releasing expired KV blocks for request %s which were "
                "retrieved by %d decode worker(s) within %d seconds.", req_id,
                count, envs.VLLM_NIXL_ABORT_REQUEST_TIMEOUT)
            del self._reqs_to_send[req_id]
            done_sending.add(req_id)

        return done_sending, done_recving

    def _get_new_notifs(self) -> set[str]:
        """
        Get req_ids which got a remote xfer message. When multiple consumers
        are reading from the same producer (heterogeneous TP scenario), wait
        for all consumers to be done pulling.
        """
        notified_req_ids: set[str] = set()
        for notifs in self.nixl_wrapper.get_new_notifs().values():
            for notif in notifs:
                req_id, tp_ratio = notif.decode("utf-8").rsplit(":", 1)
                if req_id not in self._reqs_to_send:
                    logger.error(
                        "Potentially invalid KV blocks for "
                        "unrecognized request %s were retrieved by "
                        "a decode worker. They may have expired.", req_id)
                    continue

                self.consumer_notification_counts_by_req[req_id] += 1
                # Wait all consumers (D) to be done reading before freeing.
                if self.consumer_notification_counts_by_req[req_id] == int(
                        tp_ratio):
                    notified_req_ids.add(req_id)
                    del self.consumer_notification_counts_by_req[req_id]
                    del self._reqs_to_send[req_id]
        return notified_req_ids

    def _pop_done_transfers(
            self, transfers: dict[str, list[tuple[int, float]]]) -> set[str]:
        """
        Pop completed xfers by checking for DONE state.
        Args:
            transfers: dict of req_id -> list[running_xfer]
        Returns:
            set of req_ids that have all done xfers
        """
        done_req_ids: set[str] = set()
        for req_id, handles in list(transfers.items()):
            in_progress = False
            for handle, _xfer_stime in handles:
                xfer_state = self.nixl_wrapper.check_xfer_state(handle)
                if xfer_state == "DONE":
                    self.nixl_wrapper.release_xfer_handle(handle)
                elif xfer_state == "PROC":
                    in_progress = True
                    continue
                else:
                    raise RuntimeError("Transfer failed with state %s",
                                       xfer_state)
            if not in_progress:
                done_req_ids.add(req_id)
                del transfers[req_id]
        return done_req_ids

    def start_load_kv(self, metadata: NixlConnectorMetadata):
        """
        Start loading by triggering non-blocking nixl_xfer.
        We check for these trnxs to complete in each step().
        """
        for req_id, meta in metadata.reqs_to_recv.items():
            remote_engine_id = meta.remote_engine_id
            logger.debug(
                "start_load_kv for request %s from remote engine %s. "
                "Num local_block_ids: %s. Num remote_block_ids: %s. ", req_id,
                remote_engine_id, len(meta.local_block_ids),
                len(meta.remote_block_ids))
            if self.use_host_buffer:
                self._recving_metadata[req_id] = meta
            if remote_engine_id not in self._remote_agents:
                # Initiate handshake with remote engine to exchange metadata.
                with self._handshake_lock:
                    if remote_engine_id not in self._remote_agents:
                        self._background_nixl_handshake(
                            req_id, remote_engine_id, meta)
                        continue

            # Handshake already completed, start async read xfer.
            self._read_blocks_for_req(req_id, meta)

        # Start transfers for requests whose handshakes have now finished.
        while not self._ready_requests.empty():
            self._read_blocks_for_req(*self._ready_requests.get_nowait())

        # Add to requests that are waiting to be read and track expiration.
        self._reqs_to_send.update(metadata.reqs_to_send)

    def _read_blocks_for_req(self, req_id: str, meta: ReqMeta):
        logger.debug(
            "Remote agent %s available, calling _read_blocks for req %s",
            meta.remote_engine_id, req_id)
        self._read_blocks(
            request_id=req_id,
            dst_engine_id=meta.remote_engine_id,
            local_block_ids=meta.local_block_ids,
            remote_block_ids=meta.remote_block_ids,
        )

    def _read_blocks(self, local_block_ids: list[int],
                     remote_block_ids: list[int], dst_engine_id: str,
                     request_id: str):
        # NOTE(rob): having the staging blocks be on the READER side is
        # not going to work well (since we will have to call rearrange tensors).
        # after we detect the txn is complete (which means we cannot make the
        # read trxn async easily). If we want to make "READ" happen cleanly,
        # then we will need to have the staging blocks on the remote side.

        # NOTE(rob): according to nvidia the staging blocks are used to
        # saturate IB with heterogeneous TP sizes. We should remove the staging
        # blocks until we are ready.

        # Number of D TP workers that will read from dst P. Propagate tp_ratio
        # on notification so that dst worker can wait before freeing blocks.
        tp_ratio = self._tp_size[
            self.engine_id] // self._tp_size[dst_engine_id]
        notif_id = f"{request_id}:{tp_ratio}".encode()

        # Full prefix cache hit: do not need to read remote blocks,
        # just notify P worker that we have the blocks we need.
        num_local_blocks = len(local_block_ids)
        if num_local_blocks == 0:
            remote_rank = self.tp_rank // tp_ratio
            agent_name = self._remote_agents[dst_engine_id][remote_rank]
            self.nixl_wrapper.send_notif(agent_name, notif_msg=notif_id)
            return

        # Partial prefix cache hit: just read uncomputed blocks.
        num_remote_blocks = len(remote_block_ids)
        assert num_local_blocks <= num_remote_blocks
        if num_local_blocks < num_remote_blocks:
            remote_block_ids = remote_block_ids[-num_local_blocks:]

        # Get side handles.
        local_xfer_side_handle = self.src_xfer_side_handle
        remote_xfer_side_handle = self.dst_xfer_side_handles[dst_engine_id]

        # NOTE (nicolo) With homogeneous TP, each TP worker loads KV from
        # corresponding rank. With heterogeneous TP, fixing D>P, the D tp
        # workers will issue xfers to parts of the P worker remote kv caches.

        # Get descs ids.
        local_block_descs_ids: list[int] = []
        remote_block_descs_ids: list[int] = []
        if not self.block_window_per_layer:
            # Default case: assume global attention
            remote_block_descs_ids = self._get_block_descs_ids(
                dst_engine_id, remote_block_ids)
            local_block_descs_ids = self._get_block_descs_ids(
                self.engine_id, local_block_ids)
        else:
            # TODO(mgoin): remove this once we have hybrid memory allocator
            # Optimization for models with local attention (Llama 4)
            for layer_idx, block_window in enumerate(
                    self.block_window_per_layer):
                # For each layer:
                if block_window is None:
                    # If not chunked, we just use the
                    # full block lists (global attention)
                    layer_local_block_ids = local_block_ids
                    layer_remote_block_ids = remote_block_ids
                else:
                    # If chunked, get the last block_window blocks
                    layer_local_block_ids = local_block_ids[-block_window:]
                    layer_remote_block_ids = remote_block_ids[-block_window:]

                # Get descs ids for the layer.
                layer_local_desc_ids = self._get_block_descs_ids(
                    self.engine_id, layer_local_block_ids, layer_idx)
                layer_remote_desc_ids = self._get_block_descs_ids(
                    dst_engine_id, layer_remote_block_ids, layer_idx)

                local_block_descs_ids.extend(layer_local_desc_ids)
                remote_block_descs_ids.extend(layer_remote_desc_ids)

        assert len(local_block_descs_ids) == len(remote_block_descs_ids)

        # Prepare transfer with Nixl.
        handle = self.nixl_wrapper.make_prepped_xfer(
            "READ",
            local_xfer_side_handle,
            local_block_descs_ids,
            remote_xfer_side_handle,
            remote_block_descs_ids,
            notif_msg=notif_id,
        )

        # Begin async xfer.
        self.nixl_wrapper.transfer(handle)

        # Use handle to check completion in future step().
        # TODO (NickLucche) surface xfer elapsed time
        self._recving_transfers[request_id].append(
            (handle, time.perf_counter()))

    def _get_block_descs_ids(self,
                             engine_id: str,
                             block_ids: list[int],
                             layer_idx: Optional[int] = None) -> list[int]:
        """
        Get the descs ids for a set of block ids.
        If layer_idx is provided, we use the region_ids for the given layer.
        Otherwise, we use all regions.
        """
        if layer_idx is None:
            region_ids = range(self.num_regions)
        else:
            assert layer_idx < self.num_layers
            if self.num_layers < self.num_regions:
                # If we have more regions than layers, we assume that
                # the regions are organized as [K0, V0, K1, V1, ...]
                # and we select K_i and V_i
                assert 2 * self.num_layers == self.num_regions
                region_ids = range(2 * layer_idx, 2 * layer_idx + 2)
            else:
                # Otherwise, we assume we have MLA and select i-th layer
                assert self.num_layers == self.num_regions
                region_ids = range(layer_idx, layer_idx + 1)

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

    if socket_type not in (zmq.ROUTER, zmq.REQ):
        raise ValueError(f"Unexpected socket type: {socket_type}")

    ctx: Optional[zmq.Context] = None
    try:
        ctx = zmq.Context()  # type: ignore[attr-defined]
        yield make_zmq_socket(ctx=ctx,
                              path=addr,
                              socket_type=socket_type,
                              bind=socket_type == zmq.ROUTER)
    finally:
        if ctx is not None:
            ctx.destroy(linger=0)
