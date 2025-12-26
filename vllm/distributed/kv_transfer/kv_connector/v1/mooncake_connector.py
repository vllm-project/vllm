# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import threading
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import msgspec
import numpy as np
import torch
import zmq
import zmq.asyncio

from vllm import envs
from vllm.attention.backends.abstract import AttentionMetadata
from vllm.attention.selector import get_attn_backend
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import TpKVTopology
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.utils.network_utils import get_ip, make_zmq_path, make_zmq_socket
from vllm.v1.attention.backends.utils import get_kv_cache_layout
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus

try:
    from mooncake.engine import TransferEngine
except ImportError as e:
    raise ImportError(
        "Please install mooncake by following the instructions at "
        "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
        "to run VLLM with MooncakeTransferEngine."
    ) from e

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

EngineId = str
ReqId = str

TRANS_DONE = b"trans_done"
TRANS_ERROR = b"trans_error"

logger = init_logger(__name__)


class MooncakeAgentMetadata(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    # required for @cached_property.
    dict=True,
):
    remote_hostname: str
    remote_port: int
    request_ids: list[ReqId]
    kv_caches_base_addr: list[int]
    block_ids: list[list[int]]


@dataclass
class RecvReqMeta:
    local_block_ids: list[int]
    remote_host: str
    remote_port: int


@dataclass
class SendBlockMeta:
    local_block_ids: list[int]
    ready: threading.Event
    expire_time: float = float("inf")


@dataclass
class SendReqMeta:
    reqs: dict[ReqId, SendBlockMeta]
    lock: threading.Lock


@dataclass
class FinishedSendReqSet:
    set: set[ReqId]
    lock: threading.Lock


@dataclass
class FinishedReceiveReqSet:
    set: set[ReqId]
    lock: asyncio.Lock


class MooncakeConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        self.reqs_to_recv: dict[ReqId, RecvReqMeta] = {}
        self.reqs_to_send: dict[ReqId, list[int]] = {}

    def add_new_req(
        self,
        request_id: ReqId,
        local_block_ids: list[int],
        kv_transfer_params: dict[str, Any],
        load_remote_cache: bool = True,
    ):
        if load_remote_cache:
            self.reqs_to_recv[request_id] = RecvReqMeta(
                local_block_ids=local_block_ids,
                remote_host=kv_transfer_params["remote_host"],
                remote_port=kv_transfer_params["remote_port"],
            )
        else:
            self.reqs_to_send[request_id] = local_block_ids


class MooncakeConnector(KVConnectorBase_V1):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        assert vllm_config.kv_transfer_config is not None
        assert vllm_config.kv_transfer_config.engine_id is not None
        self.engine_id: EngineId = vllm_config.kv_transfer_config.engine_id

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler: MooncakeConnectorScheduler | None = (
                MooncakeConnectorScheduler(vllm_config, self.engine_id)
            )
            self.connector_worker: MooncakeConnectorWorker | None = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = MooncakeConnectorWorker(vllm_config, self.engine_id)

    ############################################################
    # Scheduler Side Methods
    ############################################################

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens
        )

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens
        )

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
    ) -> tuple[bool, dict[str, Any] | None]:
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
    ) -> tuple[set[str] | None, set[str] | None]:
        """Get the finished recving and sending requests."""
        assert self.connector_worker is not None
        return self.connector_worker.get_finished()

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, MooncakeConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """MooncakeConnector does not do layerwise saving."""
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> None:
        """MooncakeConnector does not save explicitly."""
        pass

    def wait_for_save(self):
        pass


class MooncakeConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        self.vllm_config = vllm_config
        self.engine_id: EngineId = engine_id
        self.side_channel_host = get_ip()
        self.side_channel_port = get_mooncake_side_channel_port(vllm_config)

        assert vllm_config.kv_transfer_config
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        logger.info("Initializing Mooncake Transfer Engine Scheduler %s", engine_id)

        # Requests that need to start recv/send.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        self._reqs_need_recv: dict[ReqId, tuple[Request, list[int]]] = {}
        self._reqs_need_send: dict[ReqId, list[int]] = {}

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
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
            num_computed_tokens,
            params,
        )

        if params is not None and params.get("do_remote_prefill"):
            # Remote prefill: get all prompt blocks from remote.
            token_ids = request.prompt_token_ids or []
            count = len(token_ids) - num_computed_tokens
            if count > 0:
                return count, True

        # No remote prefill for this request.
        return 0, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector update_state_after_alloc: "
            "num_external_tokens=%s, kv_transfer_params=%s",
            num_external_tokens,
            params,
        )

        if not params:
            return

        if params.get("do_remote_prefill"):
            assert self.kv_role != "kv_producer"
            if all(p in params for p in ("remote_host", "remote_port")):
                # If remote_blocks and num_external_tokens = 0, we have
                # a full prefix cache hit on the D worker. We need to call
                # send_notif in _read_blocks to free the memory on the P.
                local_block_ids = (
                    blocks.get_unhashed_block_ids() if num_external_tokens > 0 else []
                )
                # Get unhashed blocks to pull from remote.
                self._reqs_need_recv[request.request_id] = (request, local_block_ids)
            else:
                logger.warning(
                    "Got invalid KVTransferParams: %s. This "
                    "request will not utilize KVTransfer",
                    params,
                )
            # Only trigger 1 KV transfer per request.
            params["do_remote_prefill"] = False

        elif params.get("do_remote_decode"):
            # Add an empty list to worker to create event.
            self._reqs_need_send[request.request_id] = []

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = MooncakeConnectorMetadata()

        # Loop through scheduled reqs and convert to RecvReqMeta.
        if self.kv_role != "kv_producer":
            for req_id, (req, block_ids) in self._reqs_need_recv.items():
                assert req.kv_transfer_params is not None
                meta.add_new_req(
                    request_id=req_id,
                    local_block_ids=block_ids,
                    kv_transfer_params=req.kv_transfer_params,
                )
            self._reqs_need_recv.clear()

        if self.kv_role != "kv_consumer":
            for req_id, block_ids in self._reqs_need_send.items():
                meta.add_new_req(
                    request_id=req_id,
                    local_block_ids=block_ids,
                    kv_transfer_params={},
                    load_remote_cache=False,
                )
            self._reqs_need_send.clear()

        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """

        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector request_finished, request_status=%s, "
            "kv_transfer_params=%s",
            request.status,
            params,
        )
        if not params:
            return False, None

        if params.get("do_remote_prefill"):
            # If do_remote_prefill is still True when the request is finished,
            # update_state_after_alloc must not have been called (the request
            # must have been aborted before it was scheduled).
            # To avoid stranding the prefill blocks in the prefill instance,
            # we must add empty block_ids to _reqs_need_recv so that our
            # worker side will notify and free blocks in the prefill instance.
            assert self.kv_role != "kv_producer"
            self._reqs_need_recv[request.request_id] = (request, [])
            params["do_remote_prefill"] = False
            return False, None

        if (
            not params.get("do_remote_decode")
            or request.status != RequestStatus.FINISHED_LENGTH_CAPPED
        ):
            return False, None

        assert self.kv_role != "kv_consumer"

        # TODO: check whether block_ids actually ever be 0. If not we could
        # remove the conditional below
        delay_free_blocks = len(block_ids) > 0

        if delay_free_blocks:
            self._reqs_need_send[request.request_id] = block_ids

        return delay_free_blocks, dict(
            do_remote_prefill=True,
            do_remote_decode=False,
            remote_host=self.side_channel_host,
            remote_port=self.side_channel_port,
        )


class MooncakeConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        logger.info("Initializing Mooncake Transfer Engine worker %s", engine_id)

        self.vllm_config = vllm_config

        self.engine = TransferEngine()
        self.hostname = get_ip()
        protocol = self.vllm_config.kv_transfer_config.kv_connector_extra_config.get(  # type: ignore[union-attr]
            "mooncake_protocol", "rdma"
        )
        logger.info(
            "The Mooncake Transfer Engine is using %s as its protocol.", protocol
        )
        ret_value = self.engine.initialize(self.hostname, "P2PHANDSHAKE", protocol, "")
        if ret_value != 0:
            raise RuntimeError("Mooncake Transfer Engine initialization failed.")

        self.rpc_port = self.engine.get_rpc_port()

        logger.debug(
            "Mooncake Transfer Engine initialized at %s:%d",
            self.hostname,
            self.rpc_port,
        )

        # Mooncake handshake port.
        self.side_channel_port: int = get_mooncake_side_channel_port(vllm_config)

        self.engine_id: EngineId = engine_id
        self.tp_rank = get_tensor_model_parallel_rank()
        self.world_size = get_tensor_model_parallel_world_size()
        self.tp_group = get_tp_group()
        self.num_blocks = 0

        assert vllm_config.kv_transfer_config
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.num_workers = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "num_workers", 10
        )

        self.kv_caches_base_addr: list[int] = []
        self.device_kv_caches: dict[str, torch.Tensor] = {}
        self.reqs_need_send: SendReqMeta = SendReqMeta(reqs={}, lock=threading.Lock())

        # For kv_both, we will act both prefiller and decoder.
        if self.kv_role != "kv_consumer":
            # Background thread for sending kvcaches to D.
            self._mooncake_sender_t: threading.Thread | None = None
            # Background thread for processing new sending requests.
            self._sender_executor = ThreadPoolExecutor(
                max_workers=self.num_workers, thread_name_prefix="vllm-mooncake-sender"
            )
            logger.debug(
                "Mooncake Prefiller: use %d workers to send kvcaches", self.num_workers
            )
        if self.kv_role != "kv_producer":
            self.receiver_loop = asyncio.new_event_loop()
            self._mooncake_receiver_t = threading.Thread(
                target=self._receiver_loop, args=(self.receiver_loop,), daemon=True
            )
            self._mooncake_receiver_t.start()
            logger.debug("Mooncake Decoder: start receiver thread")

        self.finished_sending_reqs: FinishedSendReqSet = FinishedSendReqSet(
            set(), threading.Lock()
        )
        self.finished_recving_reqs: FinishedReceiveReqSet = FinishedReceiveReqSet(
            set(), asyncio.Lock()
        )

        self.block_size = vllm_config.cache_config.block_size
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.use_mla = self.model_config.use_mla

        backend = get_attn_backend(
            self.model_config.get_head_size(),
            self.model_config.dtype,
            self.cache_config.cache_dtype,
            self.block_size,
            use_mla=self.use_mla,
        )
        self.backend_name = backend.get_name()
        self.kv_cache_layout = get_kv_cache_layout()
        logger.debug("Detected attention backend %s", self.backend_name)
        logger.debug("Detected kv cache layout %s", self.kv_cache_layout)

        self._tp_size: dict[EngineId, int] = {self.engine_id: self.world_size}
        self._block_size: dict[EngineId, int] = {self.engine_id: self.block_size}
        self.kv_topo = TpKVTopology(
            tp_rank=self.tp_rank,
            engine_id=self.engine_id,
            remote_tp_size=self._tp_size,  # shared state
            remote_block_size=self._block_size,  # shared state
            is_mla=self.use_mla,
            total_num_kv_heads=self.model_config.get_total_num_kv_heads(),
            attn_backend=backend,
        )
        self._use_pallas = self.kv_topo._use_pallas

        self.zmq_ctx = zmq.Context()
        self.async_zmq_ctx = zmq.asyncio.Context()
        self._encoder = msgspec.msgpack.Encoder()
        self._decoder = msgspec.msgpack.Decoder(MooncakeAgentMetadata)

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        """Cleanup background threads on destruction."""
        self.zmq_ctx.term()
        self.async_zmq_ctx.term()
        if self.kv_role != "kv_consumer":
            self._sender_executor.shutdown(wait=False)
            if self._mooncake_sender_t:
                self._mooncake_sender_t.join()
        if self.kv_role != "kv_producer" and self.receiver_loop.is_running():
            self.receiver_loop.call_soon_threadsafe(self.receiver_loop.stop)
            self._mooncake_receiver_t.join()

    def _receiver_loop(self, loop: asyncio.AbstractEventLoop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def _mooncake_sender(
        self, ready_event: threading.Event, base_port: int, tp_rank: int
    ):
        """
        Background thread that listens for Mooncake requests, dispatches them
        to a thread pool, and sends acknowledgments upon completion.
        """

        frontend_path = make_zmq_path("tcp", self.hostname, base_port + tp_rank)
        frontend = make_zmq_socket(self.zmq_ctx, frontend_path, zmq.ROUTER)
        logger.debug("Mooncake sender starting listening on path: %s", frontend_path)

        backend_path = make_zmq_path("inproc", str(uuid.uuid4()))
        backend = make_zmq_socket(self.zmq_ctx, backend_path, zmq.PULL)

        poller = zmq.Poller()
        poller.register(frontend, zmq.POLLIN)
        poller.register(backend, zmq.POLLIN)

        ready_event.set()

        try:
            while True:
                sockets = dict(poller.poll())

                if frontend in sockets:
                    identity, _, metadata_bytes = frontend.recv_multipart()
                    self._sender_executor.submit(
                        self._sender_worker,
                        identity,
                        metadata_bytes,
                        backend_path,
                    )

                if backend in sockets:
                    identity, status = backend.recv_multipart()
                    frontend.send_multipart((identity, b"", status))

        except zmq.ContextTerminated:
            logger.debug("ZMQ context terminated, exiting Mooncake sender thread.")
        except Exception as e:
            logger.error("Error in Mooncake sender thread: %s. Exiting thread.", str(e))
        finally:
            frontend.close()
            backend.close()

    def _sender_worker(
        self, identity: bytes, metadata_bytes: bytes, worker_channel_path: str
    ):
        status = TRANS_ERROR

        try:
            metadata = self._decoder.decode(metadata_bytes)
            self.send_kv_to_decode(metadata)
            status = TRANS_DONE
        except Exception as e:
            logger.error("Error processing Mooncake handshake: %s", e)
        finally:
            pusher = make_zmq_socket(self.zmq_ctx, worker_channel_path, zmq.PUSH)
            try:
                pusher.send_multipart((identity, status))
            except zmq.ZMQError as e:
                logger.warning(
                    "Internal error, maybe the server is shutting down. Error: %s",
                    e,
                )
            finally:
                pusher.close()

    def send_kv_to_decode(self, meta: MooncakeAgentMetadata):
        send_reqs: list[tuple[ReqId, SendBlockMeta]] = []
        with self.reqs_need_send.lock:
            for req_id in meta.request_ids:
                send_meta = self.reqs_need_send.reqs.get(req_id)
                if send_meta is None:
                    logger.warning("Request %s not found in reqs_need_send", req_id)
                    return
                # Mark it as not expired. We will send it now.
                send_meta.expire_time = float("inf")
                send_reqs.append((req_id, send_meta))

        self._send_blocks(send_reqs, meta)

        with self.reqs_need_send.lock:
            for req_id in meta.request_ids:
                del self.reqs_need_send.reqs[req_id]

        with self.finished_sending_reqs.lock:
            self.finished_sending_reqs.set.update(meta.request_ids)

    def _send_blocks(
        self,
        send_reqs: list[tuple[ReqId, SendBlockMeta]],
        agent_meta: MooncakeAgentMetadata,
    ):
        src_ptrs = []
        dst_ptrs = []
        lengths = []
        local_base_addr = self.kv_caches_base_addr
        remote_base_addr = agent_meta.kv_caches_base_addr
        block_len = self.block_len
        remote_session = f"{agent_meta.remote_hostname}:{agent_meta.remote_port}"

        assert len(send_reqs) == len(agent_meta.block_ids)
        for (req_id, send_meta), remote_block_ids in zip(
            send_reqs, agent_meta.block_ids
        ):
            send_meta.ready.wait()

            num_remote_blocks = len(remote_block_ids)
            if num_remote_blocks == 0:
                continue

            local_block_ids = send_meta.local_block_ids
            # Partial prefix cache hit: just read uncomputed blocks.
            num_local_blocks = len(local_block_ids)
            assert num_local_blocks >= num_remote_blocks
            if num_local_blocks > num_remote_blocks:
                local_block_ids = local_block_ids[-num_remote_blocks:]

            # Group by indices
            group_local_block_ids, group_remote_block_ids = group_concurrent_contiguous(
                local_block_ids, remote_block_ids
            )

            for local_layer_addr, remote_layer_addr in zip(
                local_base_addr, remote_base_addr
            ):
                for group_local_block_id, group_remote_block_id in zip(
                    group_local_block_ids, group_remote_block_ids
                ):
                    src_ptrs.append(
                        local_layer_addr + group_local_block_id[0] * block_len
                    )
                    dst_ptrs.append(
                        remote_layer_addr + group_remote_block_id[0] * block_len
                    )
                    lengths.append(block_len * len(group_local_block_id))

            logger.debug(
                "Sending kv_caches for request %s (%d blocks) to %s",
                req_id,
                num_remote_blocks,
                remote_session,
            )

        start_time = time.perf_counter()
        ret_value = self.engine.batch_transfer_sync_write(
            remote_session, src_ptrs, dst_ptrs, lengths
        )
        if ret_value != 0:
            raise RuntimeError(f"Error in batch_transfer_sync_write: {ret_value}")

        logger.debug(
            "Sending to %s done, took %s",
            remote_session,
            time.perf_counter() - start_time,
        )

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register the KV Cache data in mooncake."""

        logger.info("Registering KV_Caches. use_mla: %s", self.use_mla)

        kv_data_ptrs = []
        kv_data_lens = []
        seen_base_addresses = []

        split_k_and_v = self.kv_topo.split_k_and_v
        tensor_size_bytes = None
        for layer_name, cache_or_caches in kv_caches.items():
            logger.debug(
                "registering layer %s with shape %s", layer_name, cache_or_caches.shape
            )
            cache_list = cache_or_caches if split_k_and_v else [cache_or_caches]

            for cache in cache_list:
                base_addr = cache.data_ptr()
                if base_addr in seen_base_addresses:
                    continue

                seen_base_addresses.append(base_addr)
                curr_tensor_size_bytes = cache.nbytes

                if tensor_size_bytes is None:
                    tensor_size_bytes = curr_tensor_size_bytes
                    self.num_blocks = cache.shape[0]

                assert tensor_size_bytes == curr_tensor_size_bytes, (
                    "All kv cache tensors must have the same size"
                )
                kernel_block_size = cache.shape[-2 if self.use_mla else -3]
                assert self.block_size == kernel_block_size
                kv_data_ptrs.append(base_addr)
                kv_data_lens.append(tensor_size_bytes)

        self.kv_caches_base_addr = seen_base_addresses

        ret_value = self.engine.batch_register_memory(kv_data_ptrs, kv_data_lens)
        if ret_value != 0:
            raise RuntimeError("Mooncake batch memory registration failed.")

        assert tensor_size_bytes is not None
        assert self.num_blocks != 0
        assert tensor_size_bytes % self.num_blocks == 0
        self.block_len = tensor_size_bytes // self.num_blocks
        self.device_kv_caches = kv_caches
        logger.debug(
            "registered num_blocks=%d block_len=%d", self.num_blocks, self.block_len
        )

        # No need to launch server for D node.
        if self.kv_role == "kv_consumer":
            return

        ready_event = threading.Event()
        self._mooncake_sender_t = threading.Thread(
            target=self._mooncake_sender,
            args=(ready_event, self.side_channel_port, self.tp_rank),
            daemon=True,
            name="mooncake_sender",
        )
        self._mooncake_sender_t.start()
        ready_event.wait()  # Wait for listener ZMQ socket to be ready.

    async def fetch_finished_recving_reqs(self) -> set[ReqId]:
        async with self.finished_recving_reqs.lock:
            finished_recving_reqs = self.finished_recving_reqs.set
            self.finished_recving_reqs.set = set()
        return finished_recving_reqs

    def get_finished(self) -> tuple[set[str] | None, set[str] | None]:
        """
        Get requests that are done sending or recving on this specific worker.
        The scheduler process (via the MultiprocExecutor) will use this output
        to track which workers are done.
        """
        fut = None
        if self.kv_role != "kv_producer":
            fut = asyncio.run_coroutine_threadsafe(
                self.fetch_finished_recving_reqs(), self.receiver_loop
            )

        if self.kv_role != "kv_consumer":
            with self.finished_sending_reqs.lock:
                finished_sending_reqs = self.finished_sending_reqs.set
                self.finished_sending_reqs.set = set()
        else:
            finished_sending_reqs = set()

        finished_recving_reqs = fut.result() if fut else set()

        if finished_sending_reqs or finished_recving_reqs:
            logger.debug(
                "Rank %s, get_finished: %s requests done sending "
                "and %s requests done recving",
                self.tp_rank,
                len(finished_sending_reqs),
                len(finished_recving_reqs),
            )

        # Handle timeout to avoid stranding blocks on remote.
        now = time.perf_counter()
        with self.reqs_need_send.lock:
            expired_reqs = [
                req_id
                for req_id, send_meta in self.reqs_need_send.reqs.items()
                if send_meta.expire_time < now
            ]
            for req_id in expired_reqs:
                logger.warning(
                    "Request %s timed out after %d seconds without "
                    "being sent. Freeing its blocks on the producer side.",
                    req_id,
                    envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT,
                )
                del self.reqs_need_send.reqs[req_id]
            if expired_reqs:
                finished_sending_reqs.update(expired_reqs)

        return finished_sending_reqs or None, finished_recving_reqs or None

    async def receive_kv(self, path: str, req_blocks: list[tuple[str, list[int]]]):
        req_ids, block_ids = map(list, zip(*req_blocks))
        metadata = MooncakeAgentMetadata(
            remote_hostname=self.hostname,
            remote_port=self.rpc_port,
            request_ids=req_ids,
            kv_caches_base_addr=self.kv_caches_base_addr,
            block_ids=block_ids,
        )

        encoded_data = self._encoder.encode(metadata)
        logger.debug(
            "Size of encoded MooncakeAgentMetadata: %d bytes", len(encoded_data)
        )
        logger.debug("Sending kv transfer request for %s on path: %s", req_ids, path)

        # Send query for the request.
        sock: zmq.asyncio.Socket = make_zmq_socket(
            self.async_zmq_ctx, path, zmq.REQ, bind=False, linger=0
        )
        sock.setsockopt(zmq.RCVTIMEO, 60000)
        try:
            await sock.send(encoded_data)
            ret_msg = await sock.recv()
            if ret_msg != TRANS_DONE:
                logger.error(
                    "Error happens during tranfering kvcache for %s, see logs in prefiller.",  # noqa: E501
                    req_ids,
                )
                return
        except zmq.ContextTerminated:
            logger.debug("ZMQ context terminated, exiting Mooncake receiver thread.")
        except Exception as e:
            logger.error("MooncakeAgentMetadata transfer failed for %s: %s", req_ids, e)
            return
        finally:
            sock.close()

        async with self.finished_recving_reqs.lock:
            self.finished_recving_reqs.set.update(req_ids)

        logger.debug("pulling kv_caches for %s finished", req_ids)

    def group_kv_pull(self, metadata: MooncakeConnectorMetadata):
        kv_pulls = defaultdict(list)
        for req_id, meta in metadata.reqs_to_recv.items():
            logger.debug(
                "start_load_kv for request %s from remote engine. "
                "Num local_block_ids: %s.",
                req_id,
                len(meta.local_block_ids),
            )
            path = make_zmq_path(
                "tcp", meta.remote_host, meta.remote_port + self.tp_rank
            )
            kv_pulls[path].append((req_id, meta.local_block_ids))

        return kv_pulls

    def start_load_kv(self, metadata: MooncakeConnectorMetadata):
        if self.kv_role != "kv_producer":
            kv_pulls = self.group_kv_pull(metadata)
            for path, req_blocks in kv_pulls.items():
                asyncio.run_coroutine_threadsafe(
                    self.receive_kv(path, req_blocks), self.receiver_loop
                )

        if self.kv_role != "kv_consumer":
            with self.reqs_need_send.lock:
                for req_id, block_ids in metadata.reqs_to_send.items():
                    if block_ids:
                        # Already gone through request_finished()
                        send_meta = self.reqs_need_send.reqs[req_id]
                        send_meta.local_block_ids = block_ids
                        send_meta.ready.set()
                        send_meta.expire_time = (
                            time.perf_counter()
                            + envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT
                        )
                    else:
                        # From update_state_after_alloc(),
                        # but not reach request_finished() yet
                        self.reqs_need_send.reqs[req_id] = SendBlockMeta(
                            local_block_ids=[], ready=threading.Event()
                        )


def group_concurrent_contiguous(
    src_indices: list[int], dst_indices: list[int]
) -> tuple[list[list[int]], list[list[int]]]:
    """Vectorised NumPy implementation."""
    if len(src_indices) == 0:
        return [], []

    brk = np.where((np.diff(src_indices) != 1) | (np.diff(dst_indices) != 1))[0] + 1
    src_groups = np.split(src_indices, brk)
    dst_groups = np.split(dst_indices, brk)

    src_groups = [g.tolist() for g in src_groups]
    dst_groups = [g.tolist() for g in dst_groups]

    return src_groups, dst_groups


def get_mooncake_side_channel_port(vllm_config: VllmConfig) -> int:
    # This logic is now centralized
    return (
        envs.VLLM_MOONCAKE_BOOTSTRAP_PORT
        + vllm_config.parallel_config.data_parallel_rank
        * vllm_config.parallel_config.tensor_parallel_size
    )
