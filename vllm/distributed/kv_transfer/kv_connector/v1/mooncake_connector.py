# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import msgspec
import torch
import zmq

from vllm import envs
from vllm.attention.selector import backend_name_to_enum, get_attn_backend
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.distributed.parallel_state import (get_tensor_model_parallel_rank,
                                             get_tp_group)
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.platforms import _Backend
from vllm.utils import get_ip, make_zmq_path, make_zmq_socket
from vllm.v1.attention.backends.utils import get_kv_cache_layout
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

EngineId = str
ReqId = str

TRANS_DONE = b"trans_done"

logger = init_logger(__name__)


class MooncakeAgentMetadata(
        msgspec.Struct,
        omit_defaults=True,  # type: ignore[call-arg]
        # required for @cached_property.
        dict=True):
    remote_hostname: str
    remote_port: int
    request_id: ReqId
    kv_caches_base_addr: list[int]
    block_ids: list[int]


@dataclass
class RecvReqMeta:
    local_block_ids: list[int]
    remote_host: str
    remote_port: int
    remote_engine_id: str


@dataclass
class SendReqMeta:
    local_block_ids: dict[ReqId, list[int]]
    lock: threading.Lock


@dataclass
class FinishedReqSet:
    set: set[ReqId]
    lock: threading.Lock


class MooncakeConnectorMetadata(KVConnectorMetadata):

    def __init__(self):
        self.reqs_to_recv: dict[ReqId, RecvReqMeta] = {}
        self.reqs_to_send: dict[ReqId, list[int]] = {}

    def add_new_req(self,
                    request_id: ReqId,
                    local_block_ids: list[int],
                    kv_transfer_params: dict[str, Any],
                    load_remote_cache: bool = True):
        if load_remote_cache:
            self.reqs_to_recv[request_id] = RecvReqMeta(
                local_block_ids=local_block_ids,
                remote_engine_id=kv_transfer_params["remote_engine_id"],
                remote_host=kv_transfer_params["remote_host"],
                remote_port=kv_transfer_params["remote_port"])
        else:
            self.reqs_to_send[request_id] = local_block_ids


class MooncakeConnector(KVConnectorBase_V1):

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole):
        assert vllm_config.kv_transfer_config is not None
        assert vllm_config.kv_transfer_config.engine_id is not None
        self.engine_id: EngineId = vllm_config.kv_transfer_config.engine_id

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler: Optional[MooncakeConnectorScheduler] = \
                MooncakeConnectorScheduler(vllm_config, self.engine_id)
            self.connector_worker: Optional[MooncakeConnectorWorker] = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = MooncakeConnectorWorker(
                vllm_config, self.engine_id)

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

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        """Get the finished recving and sending requests."""
        assert self.connector_worker is not None
        return self.connector_worker.get_finished()

    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, MooncakeConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """MooncakeConnector does not do layerwise saving."""
        pass

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        """MooncakeConnector does not save explicitly."""
        pass

    def wait_for_save(self):
        pass


class MooncakeConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self.engine_id: EngineId = engine_id
        self.side_channel_host = get_ip()
        self.side_channel_port = (
            envs.VLLM_MOONCAKE_SIDE_CHANNEL_PORT +
            vllm_config.parallel_config.data_parallel_rank *
            vllm_config.parallel_config.tensor_parallel_size)

        assert vllm_config.kv_transfer_config
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        logger.info("Initializing Mooncake Transfer Engine Scheduler %s",
                    engine_id)

        # Requests that need to start recv/send.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        self._reqs_need_recv: dict[ReqId, tuple[Request, list[int]]] = {}
        self._reqs_need_send: dict[ReqId, list[int]] = {}

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
            "MooncakeConnector get_num_new_matched_tokens: "
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
            "MooncakeConnector update_state_after_alloc: "
            "num_external_tokens=%s, kv_transfer_params=%s",
            num_external_tokens, params)

        if not params:
            return

        if params.get("do_remote_prefill"):
            if all(p in params for p in ("remote_engine_id", "remote_host",
                                         "remote_port")):
                # If remote_blocks and num_external_tokens = 0, we have
                # a full prefix cache hit on the D worker. We need to call
                # send_notif in _read_blocks to free the memory on the P.
                local_block_ids = (blocks.get_unhashed_block_ids()
                                   if num_external_tokens > 0 else [])
                # Get unhashed blocks to pull from remote.
                self._reqs_need_recv[request.request_id] = (request,
                                                            local_block_ids)
            else:
                logger.warning(
                    "Got invalid KVTransferParams: %s. This "
                    "request will not utilize KVTransfer", params)
            # Only trigger 1 KV transfer per request.
            params["do_remote_prefill"] = False

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = MooncakeConnectorMetadata()

        # Loop through scheduled reqs and convert to RecvReqMeta.
        for req_id, (req, block_ids) in self._reqs_need_recv.items():
            assert req.kv_transfer_params is not None
            meta.add_new_req(request_id=req_id,
                             local_block_ids=block_ids,
                             kv_transfer_params=req.kv_transfer_params)

        for req_id, block_ids in self._reqs_need_send.items():
            meta.add_new_req(request_id=req_id,
                             local_block_ids=block_ids,
                             kv_transfer_params={},
                             load_remote_cache=False)

        # Clear the list once workers start the transfers
        self._reqs_need_recv.clear()
        self._reqs_need_send.clear()

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
            "MooncakeConnector request_finished, request_status=%s, "
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
            self._reqs_need_send[request.request_id] = block_ids

        return delay_free_blocks, dict(do_remote_prefill=True,
                                       do_remote_decode=False,
                                       remote_engine_id=self.engine_id,
                                       remote_host=self.side_channel_host,
                                       remote_port=self.side_channel_port)


class MooncakeConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        try:
            from mooncake.engine import TransferEngine
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                "to run VLLM with MooncakeTransferEngine.") from e
        logger.info("Initializing Mooncake Transfer Engine worker %s",
                    engine_id)

        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size

        self.engine = TransferEngine()
        self.hostname = get_ip()
        ret_value = self.engine.initialize(self.hostname, "P2PHANDSHAKE",
                                           "rdma", "")
        if ret_value != 0:
            raise RuntimeError(
                "Mooncake Transfer Engine initialization failed.")

        self.rpc_port = self.engine.get_rpc_port()

        logger.debug("Mooncake Transfer Engine initialized at %s:%d",
                     self.hostname, self.rpc_port)

        # Mooncake handshake port.
        self.side_channel_port: int = (
            envs.VLLM_MOONCAKE_SIDE_CHANNEL_PORT +
            vllm_config.parallel_config.data_parallel_rank *
            vllm_config.parallel_config.tensor_parallel_size)

        self.engine_id: EngineId = engine_id
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_group = get_tp_group()
        self.num_blocks = 0

        assert vllm_config.kv_transfer_config
        self.kv_role = vllm_config.kv_transfer_config.kv_role

        # Map of engine_id -> kv_caches_base_addr. For TP case, each local
        # rank will still only pull from a single remote TP worker.
        self.kv_caches_base_addr: dict[EngineId, list[int]] = {}
        self.device_kv_caches: dict[str, torch.Tensor] = {}
        self.reqs_need_send: SendReqMeta = SendReqMeta(local_block_ids={},
                                                       lock=threading.Lock())

        # Background thread for handling new handshake requests.
        self._mooncake_handshake_listener_t: Optional[threading.Thread] = None
        self.finished_sending_reqs: FinishedReqSet = FinishedReqSet(
            set(), threading.Lock())
        self.finished_recving_reqs: FinishedReqSet = FinishedReqSet(
            set(), threading.Lock())

        self.block_size = vllm_config.cache_config.block_size
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
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
        self.kv_cache_layout = get_kv_cache_layout()
        logger.debug("Detected attention backend %s", self.backend_name)
        logger.debug("Detected kv cache layout %s", self.kv_cache_layout)

    def _mooncake_handshake_listener(self, ready_event: threading.Event,
                                     base_port: int, tp_rank: int):
        """Background thread for getting new Mooncake handshakes."""

        # Listen for new requests for metadata.
        path = make_zmq_path("tcp", self.hostname, base_port + tp_rank)
        logger.debug(
            "Mooncake handshake & sender starting listening on path: %s", path)
        with zmq_ctx(zmq.ROUTER, path) as sock:
            ready_event.set()
            while True:
                identity, _, metadata_bytes = sock.recv_multipart()
                decoder = msgspec.msgpack.Decoder(MooncakeAgentMetadata)
                metadata = decoder.decode(metadata_bytes)
                self.send_kv_to_decode(metadata)
                sock.send_multipart((identity, b"", TRANS_DONE))

    def send_kv_to_decode(self, meta: MooncakeAgentMetadata):
        local_block_ids = None
        while local_block_ids is None:
            with self.reqs_need_send.lock:
                local_block_ids = self.reqs_need_send.local_block_ids.get(
                    meta.request_id)
            if local_block_ids is None:
                # The next step of start_load_kv() is not completed yet.
                # Just wait for a bit
                time.sleep(0.1)

        self._send_blocks(local_block_ids, meta)

        with self.reqs_need_send.lock:
            del self.reqs_need_send.local_block_ids[meta.request_id]

        with self.finished_sending_reqs.lock:
            self.finished_sending_reqs.set.add(meta.request_id)

    def _send_blocks(self, local_block_ids: list[int],
                     agentmeta: MooncakeAgentMetadata):

        remote_block_ids = agentmeta.block_ids
        num_remote_blocks = len(remote_block_ids)

        if num_remote_blocks == 0:
            return

        # Partial prefix cache hit: just read uncomputed blocks.
        num_local_blocks = len(local_block_ids)
        assert num_local_blocks >= num_remote_blocks
        if num_local_blocks > num_remote_blocks:
            local_block_ids = local_block_ids[-num_remote_blocks:]

        local_base_addr = self.kv_caches_base_addr[self.engine_id]
        remote_base_addr = agentmeta.kv_caches_base_addr
        src_ptrs = []
        dst_ptrs = []
        lengths = []
        block_len = self.block_len

        for local_layer_addr, remote_layer_addr in zip(local_base_addr,
                                                       remote_base_addr):
            for local_block_id, remote_block_id in zip(local_block_ids,
                                                       remote_block_ids):
                src_ptrs.append(local_layer_addr + local_block_id * block_len)
                dst_ptrs.append(remote_layer_addr +
                                remote_block_id * block_len)
                lengths.append(block_len)

        remote_session = f"{agentmeta.remote_hostname}:{agentmeta.remote_port}"
        logger.debug("Sending kv_caches for request %s (%d blocks) to %s",
                     agentmeta.request_id, num_remote_blocks, remote_session)

        ret_value = self.engine.batch_transfer_sync_write(
            remote_session, src_ptrs, dst_ptrs, lengths)
        if ret_value != 0:
            raise RuntimeError(
                f"Error in batch_transfer_sync_write: {ret_value}")

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register the KV Cache data in mooncake."""

        logger.info("Registering KV_Caches. use_mla: %s", self.use_mla)

        kv_data_ptrs = []
        kv_data_lens = []
        # With hybrid allocator, layers can share a kv cache tensor
        seen_base_addresses = []

        self.split_k_and_v = not (self.use_mla or self._use_pallas_v1
                                  or self._use_flashinfer)
        tensor_size_bytes = None
        for layer_name, cache_or_caches in kv_caches.items():
            cache_list = cache_or_caches if self.split_k_and_v else [
                cache_or_caches
            ]

            for cache in cache_list:
                base_addr = cache.data_ptr()
                if base_addr in seen_base_addresses:
                    continue

                seen_base_addresses.append(base_addr)
                curr_tensor_size_bytes = cache.nbytes

                if tensor_size_bytes is None:
                    tensor_size_bytes = curr_tensor_size_bytes
                    self.num_blocks = cache.shape[0]

                assert tensor_size_bytes == curr_tensor_size_bytes, \
                    "All kv cache tensors must have the same size"
                kv_data_ptrs.append(base_addr)
                kv_data_lens.append(tensor_size_bytes)

        self.kv_caches_base_addr[self.engine_id] = seen_base_addresses

        ret_value = self.engine.batch_register_memory(kv_data_ptrs,
                                                      kv_data_lens)
        if ret_value != 0:
            raise RuntimeError("Mooncake batch memory registration failed.")

        assert tensor_size_bytes is not None
        assert self.num_blocks != 0
        assert tensor_size_bytes % self.num_blocks == 0
        self.block_len = tensor_size_bytes // self.num_blocks
        self.slot_size_bytes = self.block_len // self.block_size
        if self._use_flashinfer:
            assert self.slot_size_bytes % 2 == 0
            self.slot_size_bytes /= 2
        self.device_kv_caches = kv_caches

        # No need to launch server for D node.
        if self.kv_role == "kv_consumer":
            return

        ready_event = threading.Event()
        self._mooncake_handshake_listener_t = threading.Thread(
            target=self._mooncake_handshake_listener,
            args=(ready_event, self.side_channel_port, self.tp_rank),
            daemon=True,
            name="mooncake_handshake_listener")
        self._mooncake_handshake_listener_t.start()
        ready_event.wait()  # Wait for listener ZMQ socket to be ready.

    def get_finished(self) -> tuple[Optional[set[str]], Optional[set[str]]]:
        """
        Get requests that are done sending or recving on this specific worker.
        The scheduler process (via the MultiprocExecutor) will use this output
        to track which workers are done.
        """
        with self.finished_recving_reqs.lock:
            finished_recving_reqs = self.finished_recving_reqs.set
            self.finished_recving_reqs.set = set()

        with self.finished_sending_reqs.lock:
            finished_sending_reqs = self.finished_sending_reqs.set
            self.finished_sending_reqs.set = set()

        if finished_sending_reqs or finished_recving_reqs:
            logger.debug(
                "Rank %s, get_finished: %s requests done sending "
                "and %s requests done recving", self.tp_rank,
                len(finished_sending_reqs), len(finished_recving_reqs))

        return finished_sending_reqs or None, finished_recving_reqs or None

    def receive_kv(self, req_id: ReqId, meta: RecvReqMeta):
        metadata = MooncakeAgentMetadata(
            remote_hostname=self.hostname,
            remote_port=self.rpc_port,
            request_id=req_id,
            kv_caches_base_addr=self.kv_caches_base_addr[self.engine_id],
            block_ids=meta.local_block_ids,
        )

        encoder = msgspec.msgpack.Encoder()
        encoded_data = encoder.encode(metadata)
        size_in_bytes = len(encoded_data)
        logger.debug("Size of encoded MooncakeAgentMetadata: %s bytes",
                     str(size_in_bytes))

        path = make_zmq_path("tcp", meta.remote_host,
                             meta.remote_port + self.tp_rank)
        logger.debug("Sending pull request for %s on path: %s", req_id, path)
        # Send query for the request.
        with zmq_ctx(zmq.REQ, path) as sock:
            sock.send(encoded_data)
            ret_msg = sock.recv()
            assert ret_msg == TRANS_DONE

        with self.finished_recving_reqs.lock:
            self.finished_recving_reqs.set.add(req_id)

        logger.debug("pulling kv_caches for %s finished", req_id)

    def start_load_kv(self, metadata: MooncakeConnectorMetadata):
        if self.kv_role != "kv_producer":
            for req_id, recv_meta in metadata.reqs_to_recv.items():
                remote_engine_id = recv_meta.remote_engine_id
                logger.debug(
                    "start_load_kv for request %s from remote engine %s. "
                    "Num local_block_ids: %s.", req_id, remote_engine_id,
                    len(recv_meta.local_block_ids))

                receive_kv_thread = threading.Thread(target=self.receive_kv,
                                                     args=(req_id, recv_meta),
                                                     daemon=True,
                                                     name="receive_kv")
                receive_kv_thread.start()

        if self.kv_role != "kv_consumer":
            with self.reqs_need_send.lock:
                for req_id, send_meta in metadata.reqs_to_send.items():
                    self.reqs_need_send.local_block_ids[req_id] = send_meta


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
