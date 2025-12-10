# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import logging
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import httpx
import msgspec
import numpy as np
import torch
import uvicorn
import zmq
import zmq.asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

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
    is_global_first_rank,
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

http_log_level = logger.getEffectiveLevel()
# INFO logs of http are too noisy. Silence them.
# Setting vllm log level to DEBUG if we really want to see.
if http_log_level == logging.INFO:
    http_log_level = logging.WARNING
logging.getLogger("httpx").setLevel(http_log_level)


class MooncakeXferMetadata(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    # required for @cached_property.
    dict=True,
):
    remote_hostname: str
    remote_port: int
    req_blocks: dict[ReqId, list[int]]
    kv_caches_base_addr: list[int]


class MooncakeXferResponse(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    # required for @cached_property.
    dict=True,
):
    status: str
    ok_reqs: list[ReqId] | None = None
    err_reqs: list[ReqId] | None = None


@dataclass
class PullReqMeta:
    req_id: ReqId
    local_block_ids: list[int]
    bootstrap_server_host: str
    bootstrap_server_port: int
    # Set expire time to avoid infinitely sending requests.
    expire_time: float = float("inf")
    # Designed for one D pairing to multiple P
    pull_tasks_count: int = 0


@dataclass
class SendBlockMeta:
    local_block_ids: list[int]
    ready: asyncio.Event
    expire_time: float = float("inf")
    need_send: int = 0
    sended: int = 0
    sending: int = 0


@dataclass
class XferTask:
    worker_addr: str
    pull_req_meta: PullReqMeta


@dataclass
class PullKvTasks:
    query: list[PullReqMeta] = field(default_factory=list)
    # {worker_addr: {req_id: XferTask}}
    xfer: dict[str, dict[ReqId, XferTask]] = field(
        default_factory=lambda: defaultdict(dict)
    )


class RegisterWorkerPayload(BaseModel):
    dp_rank: int
    tp_rank: int
    worker_addr: str


class RegisterRequestPayload(BaseModel):
    req_id: ReqId
    dp_rank: int


class QueryRequestsPayload(BaseModel):
    req_ids: list[ReqId]
    tp_rank: int
    tp_size: int


class QueryRequestsResponse(BaseModel):
    results: dict[ReqId, tuple[str, list[str]]]


class MooncakeConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        self.reqs_to_recv: list[PullReqMeta] = []
        self.reqs_to_send: dict[ReqId, list[int]] = {}
        self.reqs_not_processed: set[ReqId] = set()

    def add_new_req(
        self,
        request_id: ReqId,
        local_block_ids: list[int],
        kv_transfer_params: dict[str, Any],
        load_remote_cache: bool = True,
    ):
        if load_remote_cache:
            self.reqs_to_recv.append(
                PullReqMeta(
                    req_id=request_id,
                    local_block_ids=local_block_ids,
                    bootstrap_server_host=kv_transfer_params["bootstrap_server_host"],
                    bootstrap_server_port=kv_transfer_params["bootstrap_server_port"],
                )
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

        assert vllm_config.parallel_config
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank

        assert vllm_config.kv_transfer_config
        self.kv_role = vllm_config.kv_transfer_config.kv_role

        if self.kv_role != "kv_consumer":
            host, port = get_mooncake_bootstrap_addr(vllm_config)
            self.bootstrap_addr = make_zmq_path("http", host, port)
            self.http_client: HTTPClientManager = HTTPClientManager()

            self.sender_loop = asyncio.new_event_loop()
            self._sender_loop_t = threading.Thread(
                target=_async_loop, args=(self.sender_loop,), daemon=True
            )
            self._sender_loop_t.start()

        logger.info("Initializing Mooncake Transfer Engine Scheduler %s", engine_id)

        # Requests that need to start recv/send.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        self._reqs_need_recv: dict[ReqId, tuple[Request, list[int]]] = {}
        self._reqs_need_send: dict[ReqId, list[int]] = {}
        # Reqs to remove from processed set because they're not to send after
        # remote prefill or aborted.
        self._reqs_not_processed: set[ReqId] = set()

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        if self.kv_role != "kv_consumer" and self.sender_loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self.http_client.shutdown(), self.sender_loop
            ).result()
            self.sender_loop.call_soon_threadsafe(self.sender_loop.stop)
            self._sender_loop_t.join()

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

        if not params:
            return 0, False

        if params.get("do_remote_prefill"):
            # Remote prefill: get all prompt blocks from remote.
            assert self.kv_role != "kv_producer"
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
            if all(
                p in params for p in ("bootstrap_server_host", "bootstrap_server_port")
            ):
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
            assert self.kv_role != "kv_consumer"
            # Add an empty list to worker to create event.
            self._reqs_need_send[request.request_id] = []
            # Register request with bootstrap server.
            asyncio.run_coroutine_threadsafe(
                self.register_req_with_bootstrap(request.request_id), self.sender_loop
            )

    async def register_req_with_bootstrap(self, req_id: ReqId):
        client = await self.http_client.get_client()
        url = self.bootstrap_addr + "/register_request"
        payload = RegisterRequestPayload(req_id=req_id, dp_rank=self.dp_rank)

        response = None
        try:
            response = await client.post(url, json=payload.model_dump())
            response.raise_for_status()
            logger.debug("Registered request %s with bootstrap server", req_id)
        except Exception as e:
            err_msg = (
                e.response.text if isinstance(e, httpx.HTTPStatusError) else str(e)
            )
            logger.error(
                "Failed to register request %s with bootstrap server: %s",
                req_id,
                err_msg,
            )
        finally:
            if response:
                await response.aclose()

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

            meta.reqs_not_processed = self._reqs_not_processed
            self._reqs_not_processed = set()

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

        if not params.get("do_remote_decode"):
            return False, None

        assert self.kv_role != "kv_consumer"

        if request.status != RequestStatus.FINISHED_LENGTH_CAPPED:
            # Also include the case of a P/D Prefill request with immediate
            # block free (eg abort). Stop tracking this request.
            self._reqs_not_processed.add(request.request_id)
            return False, None

        # TODO: check whether block_ids actually ever be 0. If not we could
        # remove the conditional below
        delay_free_blocks = len(block_ids) > 0

        if delay_free_blocks:
            self._reqs_need_send[request.request_id] = block_ids

        return delay_free_blocks, None


class MooncakeConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        logger.info("Initializing Mooncake Transfer Engine worker %s", engine_id)

        self.vllm_config = vllm_config

        self.engine = TransferEngine()
        self.hostname = get_ip()
        ret_value = self.engine.initialize(self.hostname, "P2PHANDSHAKE", "rdma", "")
        if ret_value != 0:
            raise RuntimeError("Mooncake Transfer Engine initialization failed.")

        self.rpc_port = self.engine.get_rpc_port()

        logger.debug(
            "Mooncake Transfer Engine initialized at %s:%d",
            self.hostname,
            self.rpc_port,
        )

        self.side_channel_port: int = 0  # we will bind it in register_kv_caches()
        self.engine_id: EngineId = engine_id
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_blocks = 0

        assert vllm_config.parallel_config
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank
        self.dp_size = vllm_config.parallel_config.data_parallel_size
        if vllm_config.parallel_config.pipeline_parallel_size > 1:
            raise ValueError(
                "Mooncake Transfer Engine does not support pipeline parallelism yet."
            )

        assert vllm_config.kv_transfer_config
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.num_workers = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "num_workers", 10
        )

        self.kv_caches_base_addr: list[int] = []
        self.device_kv_caches: dict[str, torch.Tensor] = {}
        self.reqs_need_send: dict[ReqId, SendBlockMeta] = {}

        # Only used by prefillers.
        host, port = get_mooncake_bootstrap_addr(vllm_config)
        self.bootstrap_addr = make_zmq_path("http", host, port)

        if self.kv_role != "kv_consumer":
            # Background threads for sending kvcaches to D.
            self._sender_executor = ThreadPoolExecutor(
                max_workers=self.num_workers, thread_name_prefix="vllm-mooncake-sender"
            )
            logger.debug(
                "Mooncake Prefiller: use %d workers to send kvcaches", self.num_workers
            )
            # An asyncio queue to buffer incoming requests for the sender
            self.sender_worker_queue: asyncio.Queue[tuple[bytes, bytes]] = (
                asyncio.Queue()
            )
            self.sender_loop = asyncio.new_event_loop()
            # Background thread for processing new sending requests.
            self._mooncake_sender_t: threading.Thread = threading.Thread(
                target=_async_loop, args=(self.sender_loop,), daemon=True
            )
            self._mooncake_sender_t.start()

            # Start bootstrap server on global rank 0.
            if is_global_first_rank():
                self.bootstrap_server = MooncakeBootstrapServer("0.0.0.0", port)
                self.bootstrap_server.start()

        if self.kv_role != "kv_producer":
            self.receiver_loop = asyncio.new_event_loop()
            self._mooncake_receiver_t = threading.Thread(
                target=_async_loop, args=(self.receiver_loop,), daemon=True
            )
            self._mooncake_receiver_t.start()
            self._pull_kv_list: dict[str, PullKvTasks] = defaultdict(PullKvTasks)

        self.finished_sending_reqs: set[ReqId] = set()
        self.finished_recving_reqs: set = set()

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

        self._tp_size: dict[EngineId, int] = {self.engine_id: self.tp_size}
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

        self.async_zmq_ctx = zmq.asyncio.Context()
        self._encoder = msgspec.msgpack.Encoder()
        self._xfer_meta_decoder = msgspec.msgpack.Decoder(MooncakeXferMetadata)
        self._xfer_resp_decoder = msgspec.msgpack.Decoder(MooncakeXferResponse)
        self.http_client: HTTPClientManager = HTTPClientManager()

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        """Cleanup background threads on destruction."""
        self.async_zmq_ctx.term()
        if self.kv_role != "kv_consumer":
            self._sender_executor.shutdown(wait=False)
            if self.sender_loop.is_running():
                self.sender_loop.call_soon_threadsafe(self.sender_loop.stop)
                self._mooncake_sender_t.join()
            if is_global_first_rank():
                self.bootstrap_server.shutdown()
        if self.kv_role != "kv_producer" and self.receiver_loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self.http_client.shutdown(), self.receiver_loop
            ).result()
            self.receiver_loop.call_soon_threadsafe(self.receiver_loop.stop)
            self._mooncake_receiver_t.join()

    async def register_worker_with_bootstrap(self):
        url = self.bootstrap_addr + "/register_worker"
        worker_addr = make_zmq_path("tcp", self.hostname, self.side_channel_port)
        payload = RegisterWorkerPayload(
            dp_rank=self.dp_rank, tp_rank=self.tp_rank, worker_addr=worker_addr
        )

        while True:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(url, json=payload.model_dump())
                    response.raise_for_status()
                logger.debug("Successfully registered with bootstrap server at %s", url)
                break
            except httpx.ConnectError:
                # Bootstrap server not ready, wait for a while and retry.
                time.sleep(0.1)
            except Exception as e:
                raise RuntimeError("Could not connect to bootstrap server") from e

    async def _mooncake_sender(self, ready_event: threading.Event):
        """
        Background thread that listens for Mooncake requests, dispatches them
        to a thread pool, and sends acknowledgments upon completion.
        """

        sock = self.async_zmq_ctx.socket(zmq.ROUTER)
        self.side_channel_port = sock.bind_to_random_port(f"tcp://{self.hostname}")
        logger.debug(
            "Mooncake sender starting listening on path: tcp://%s:%d",
            self.hostname,
            self.side_channel_port,
        )

        await self.register_worker_with_bootstrap()

        # Create async worker tasks that process items from the queue
        worker_tasks = [
            asyncio.create_task(self._sender_worker(sock))
            for _ in range(self.num_workers * 2)
        ]

        ready_event.set()

        # Main loop to receive ZMQ requests and put them on the queue
        try:
            while True:
                identity, metadata_bytes = await sock.recv_multipart()
                await self.sender_worker_queue.put((identity, metadata_bytes))
        except zmq.ContextTerminated:
            logger.debug("ZMQ context terminated, exiting Mooncake sender thread.")
        except Exception as e:
            logger.error("Error in Mooncake sender thread: %s. Exiting thread.", str(e))
        finally:
            # Clean up worker tasks
            for task in worker_tasks:
                task.cancel()
            await asyncio.gather(*worker_tasks, return_exceptions=True)
            sock.close()

    async def _sender_worker(self, sock: zmq.asyncio.Socket):
        while True:
            try:
                identity, metadata_bytes = await self.sender_worker_queue.get()
                try:
                    metadata = self._xfer_meta_decoder.decode(metadata_bytes)
                    await self.send_kv_to_decode(identity, sock, metadata)
                except Exception as e:
                    logger.error("Error processing Mooncake xfer request: %s", e)
                    error_response = MooncakeXferResponse(status=f"error: {e}")
                    await sock.send_multipart(
                        (identity, self._encoder.encode(error_response))
                    )
                finally:
                    self.sender_worker_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in _sender_worker: %s", e)

    async def send_kv_to_decode(
        self, identity: bytes, sock: zmq.asyncio.Socket, meta: MooncakeXferMetadata
    ):
        pending_reqs: dict[ReqId, SendBlockMeta] = {}
        err_reqs: list[ReqId] = []
        for req_id in meta.req_blocks:
            send_meta = self.reqs_need_send.get(req_id)
            if send_meta is None:
                err_reqs.append(req_id)
            else:
                pending_reqs[req_id] = send_meta

        if not pending_reqs:
            response = MooncakeXferResponse(status="ok", err_reqs=err_reqs)
            await sock.send_multipart((identity, self._encoder.encode(response)))
            return

        async def wait_and_ret(
            req_id: ReqId, send_meta: SendBlockMeta
        ) -> tuple[ReqId, SendBlockMeta]:
            await send_meta.ready.wait()
            return req_id, send_meta

        wait_tasks = [
            asyncio.create_task(wait_and_ret(req_id, send_meta))
            for req_id, send_meta in pending_reqs.items()
        ]

        while wait_tasks:
            done, pending = await asyncio.wait(
                wait_tasks, return_when=asyncio.FIRST_COMPLETED
            )
            wait_tasks = list(pending)
            ready_reqs: list[tuple[ReqId, SendBlockMeta]] = []
            for task in done:
                req_id, send_meta = task.result()
                # Do we still in reqs_need_send (not expired)?
                if req_id in self.reqs_need_send:
                    # Mark it sending to avoid expiration.
                    send_meta.sending += 1
                    if not send_meta.need_send:
                        self.resolve_need_send(send_meta)
                    ready_reqs.append((req_id, send_meta))
                # Otherwise (expired, very unlikely), forget it. Do not let D retry.

            src_ptrs, dst_ptrs, lengths = self._build_transfer_params(ready_reqs, meta)
            if src_ptrs:
                remote_session = f"{meta.remote_hostname}:{meta.remote_port}"

                ret_value = await self.sender_loop.run_in_executor(
                    self._sender_executor,
                    self._send_blocks,
                    remote_session,
                    src_ptrs,
                    dst_ptrs,
                    lengths,
                )

                if ret_value != 0:
                    for req_id, send_meta in ready_reqs:
                        send_meta.sending -= 1
                        err_reqs.append(req_id)
                    # Do best effort to transfer the remaining reqs.
                    response = MooncakeXferResponse(
                        status="continue", err_reqs=err_reqs
                    )
                    err_reqs = []
                    await sock.send_multipart(
                        (identity, self._encoder.encode(response))
                    )
                    continue

            for req_id, send_meta in ready_reqs:
                # Todo: for heterogeneous TP (one P pairs to multiple D),
                # we need to check whether all headers are sent.
                # If not, we should set expire_time to normal and skip the below.
                send_meta.sending -= 1
                send_meta.sended += 1
                if send_meta.sended == send_meta.need_send:
                    del self.reqs_need_send[req_id]
                    self.finished_sending_reqs.add(req_id)

            response = MooncakeXferResponse(
                status="continue" if wait_tasks else "ok",
                ok_reqs=[req_id for req_id, _ in ready_reqs],
                err_reqs=err_reqs,
            )
            await sock.send_multipart((identity, self._encoder.encode(response)))

    def resolve_need_send(self, send_meta: SendBlockMeta):
        # Prepare for heterogeneous TP (one P pairs to multiple D)
        send_meta.need_send = 1

    def _build_transfer_params(
        self,
        send_reqs: list[tuple[ReqId, SendBlockMeta]],
        agent_meta: MooncakeXferMetadata,
    ) -> tuple[list[int], list[int], list[int]]:
        src_ptrs = []
        dst_ptrs = []
        lengths = []
        local_base_addr = self.kv_caches_base_addr
        remote_base_addr = agent_meta.kv_caches_base_addr
        block_len = self.block_len
        remote_session = f"{agent_meta.remote_hostname}:{agent_meta.remote_port}"

        for req_id, send_meta in send_reqs:
            remote_block_ids = agent_meta.req_blocks[req_id]
            num_remote_blocks = len(remote_block_ids)
            if num_remote_blocks == 0:
                continue

            local_block_ids = send_meta.local_block_ids
            # Partial prefix cache hit: just read uncomputed blocks.
            num_local_blocks = len(local_block_ids)
            if num_local_blocks < num_remote_blocks:
                logger.error(
                    "req %s: local blocks(%d) less than remote blocks(%d)!",
                    req_id,
                    num_local_blocks,
                    num_remote_blocks,
                )
                continue
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

        return src_ptrs, dst_ptrs, lengths

    def _send_blocks(
        self,
        remote_session: str,
        src_ptrs: list[int],
        dst_ptrs: list[int],
        lengths: list[int],
    ) -> int:
        start_time = time.perf_counter()
        ret_value = self.engine.batch_transfer_sync_write(
            remote_session, src_ptrs, dst_ptrs, lengths
        )
        if ret_value == 0:
            logger.debug(
                "Sending to %s done, took %s",
                remote_session,
                time.perf_counter() - start_time,
            )
        return ret_value

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
        asyncio.run_coroutine_threadsafe(
            self._mooncake_sender(ready_event), self.sender_loop
        )
        ready_event.wait()  # Wait for listener ZMQ socket to be ready.

    async def fetch_finished_recving_reqs(self) -> set[ReqId]:
        finished_recving_reqs = self.finished_recving_reqs
        self.finished_recving_reqs = set()
        return finished_recving_reqs

    async def fetch_finished_sending_reqs(self) -> set[ReqId]:
        finished_sending_reqs = self.finished_sending_reqs
        self.finished_sending_reqs = set()

        # Handle timeout to avoid stranding blocks on remote.
        now = time.perf_counter()
        expired_reqs = [
            req_id
            for req_id, send_meta in self.reqs_need_send.items()
            if send_meta.expire_time < now and send_meta.sending == 0
        ]
        for req_id in expired_reqs:
            logger.warning(
                "Request %s timed out after %d seconds without "
                "being sent. Freeing its blocks on the producer side.",
                req_id,
                envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT,
            )
            del self.reqs_need_send[req_id]
        if expired_reqs:
            finished_sending_reqs.update(expired_reqs)

        return finished_sending_reqs

    def get_finished(self) -> tuple[set[str] | None, set[str] | None]:
        """
        Get requests that are done sending or recving on this specific worker.
        The scheduler process (via the MultiprocExecutor) will use this output
        to track which workers are done.
        """
        recv_fut = None
        send_fut = None
        if self.kv_role != "kv_producer":
            recv_fut = asyncio.run_coroutine_threadsafe(
                self.fetch_finished_recving_reqs(), self.receiver_loop
            )

        if self.kv_role != "kv_consumer":
            send_fut = asyncio.run_coroutine_threadsafe(
                self.fetch_finished_sending_reqs(), self.sender_loop
            )

        finished_recving_reqs = recv_fut.result() if recv_fut else set()
        finished_sending_reqs = send_fut.result() if send_fut else set()

        if finished_sending_reqs or finished_recving_reqs:
            logger.debug(
                "Rank %s, get_finished: %s requests done sending "
                "and %s requests done recving",
                self.tp_rank,
                len(finished_sending_reqs),
                len(finished_recving_reqs),
            )

        return finished_sending_reqs or None, finished_recving_reqs or None

    async def receive_kv(
        self, path: str, xfer_task_list: dict[ReqId, XferTask], bootstrap_addr: str
    ):
        req_ids = set(xfer_task_list)
        metadata = MooncakeXferMetadata(
            remote_hostname=self.hostname,
            remote_port=self.rpc_port,
            req_blocks={
                req_id: task.pull_req_meta.local_block_ids
                for req_id, task in xfer_task_list.items()
            },
            kv_caches_base_addr=self.kv_caches_base_addr,
        )

        encoded_data = self._encoder.encode(metadata)
        logger.debug(
            "Size of encoded MooncakeXferMetadata: %d bytes", len(encoded_data)
        )
        logger.debug("Sending kv transfer request for %s on path: %s", req_ids, path)

        # Send query for the request.
        try:
            with make_zmq_socket(
                self.async_zmq_ctx, path, zmq.DEALER, bind=False, linger=0
            ) as sock:
                await sock.send(encoded_data)
                while True:
                    ret_msg = await sock.recv()
                    response = self._xfer_resp_decoder.decode(ret_msg)
                    if response.status not in ("ok", "continue"):
                        logger.error(
                            "Error happens during tranfering kvcache for %s: %s",  # noqa: E501
                            req_ids,
                            response.status,
                        )
                        return
                    self.process_pulling_result(
                        response, path, xfer_task_list, bootstrap_addr
                    )
                    if response.status == "ok":
                        break
        except zmq.ContextTerminated:
            logger.debug("ZMQ context terminated, exiting Mooncake receiver thread.")
        except Exception as e:
            logger.error("MooncakeXferMetadata transfer failed for %s: %s", req_ids, e)
            return

    def process_pulling_result(
        self,
        response: MooncakeXferResponse,
        path: str,
        xfer_task_list: dict[ReqId, XferTask],
        bootstrap_addr: str,
    ):
        ok_reqs: list[ReqId] = response.ok_reqs or []

        for req_id in ok_reqs:
            xfer_task = xfer_task_list[req_id]
            # No race because we are in async loop.
            xfer_task.pull_req_meta.pull_tasks_count -= 1
            if xfer_task.pull_req_meta.pull_tasks_count == 0:
                self.finished_recving_reqs.add(xfer_task.pull_req_meta.req_id)

        retry_reqs: list[ReqId] = response.err_reqs or []
        # Add retry_reqs to list.
        # These reqs will be received again in next start_load_kv().
        now = time.perf_counter()
        for req_id in retry_reqs:
            xfer_task = xfer_task_list[req_id]
            if xfer_task.pull_req_meta.expire_time > now:
                self._pull_kv_list[bootstrap_addr].xfer[path][req_id] = xfer_task

        logger.debug("pulling kv_caches for %s finished", ok_reqs)

    def group_by_worker(
        self,
        pull_kv_tasks: PullKvTasks,
        bootstrap_addr: str,
        bootstrap_resp: dict[ReqId, tuple[str, list[str]]],
    ):
        """
        Second-level grouping:
        group requests by their final target prefiller worker's ZMQ path.
        """
        now = time.perf_counter()

        for task in pull_kv_tasks.query:
            req_id = task.req_id
            response = bootstrap_resp.get(req_id)
            if not response:
                logger.warning(
                    "Bootstrap server internal error! "
                    "No address found for req %s in bootstrap response.",
                    req_id,
                )
                continue

            status, worker_addrs = response
            if status != "ok":
                if status == "retry":
                    # Add task to list.
                    # These reqs will be handled again in next start_load_kv().
                    if task.expire_time > now:
                        self._pull_kv_list[bootstrap_addr].query.append(task)
                else:
                    logger.error(
                        "Bootstrap server internal error! "
                        "Error happens during querying req %s: %s",
                        req_id,
                        status,
                    )
                continue

            # for heterogeneous TP, one D may pair to multiple P.
            task.pull_tasks_count = len(worker_addrs)
            # Update expire_time for XferTask again,
            # since the req may not be ready on P when query task is created.
            task.expire_time = now + envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT
            for addr in worker_addrs:
                pull_kv_tasks.xfer[addr][task.req_id] = XferTask(
                    worker_addr=addr, pull_req_meta=task
                )

    async def batch_query_requests(
        self, bootstrap_addr: str, query_task_list: list[PullReqMeta]
    ) -> dict[ReqId, tuple[str, list[str]]]:
        client = await self.http_client.get_client()
        url = bootstrap_addr + "/query_requests"
        req_ids = [task.req_id for task in query_task_list]
        payload = QueryRequestsPayload(
            req_ids=req_ids, tp_rank=self.tp_rank, tp_size=self.tp_size
        )

        try:
            response = await client.post(url, json=payload.model_dump())
            response.raise_for_status()
            data = response.json()
            logger.debug("Received responses from bootstrap server: %s", data)
            response_data = QueryRequestsResponse.model_validate(data)
            return response_data.results
        except Exception as e:
            err_msg = (
                e.response.text if isinstance(e, httpx.HTTPStatusError) else str(e)
            )
            logger.error(
                "Failed to query bootstrap server for %d requests: %s",
                len(req_ids),
                err_msg,
            )
            return {}

    async def handle_bootstrap_group(
        self, bootstrap_addr: str, pull_kv_tasks: PullKvTasks
    ):
        if pull_kv_tasks.query:
            bootstrap_resp = await self.batch_query_requests(
                bootstrap_addr, pull_kv_tasks.query
            )
            self.group_by_worker(pull_kv_tasks, bootstrap_addr, bootstrap_resp)

        for worker_addr, xfer_task_list in pull_kv_tasks.xfer.items():
            if xfer_task_list:
                asyncio.create_task(
                    self.receive_kv(worker_addr, xfer_task_list, bootstrap_addr)
                )

    async def _start_load_kv(self, reqs_to_recv: list[PullReqMeta]):
        """
        Main part for decoders to receive kv caches.
        We handle pulling kv caches in two steps:
        1. Group the requests by their bootstrap server address
           (they belong to the same prefiller),
           and query their matched prefiller-worker's zmq addresses.
        2. Group the requests by their matched prefiller-worker's zmq addresses.
           (They belong to the same dp rank in the same prefiller)
           If one request matched multiple prefiller-workers, we create XferTask
           for each worker's zmq address. Finally we send zmq requests to
           pull kv caches.
        """

        if not reqs_to_recv and not self._pull_kv_list:
            # Nothing to do.
            return

        pull_kv_list = self._pull_kv_list
        self._pull_kv_list = defaultdict(PullKvTasks)

        now = time.perf_counter()
        """
        First-level grouping:
        group requests by their bootstrap server address.
        """
        for meta in reqs_to_recv:
            bootstrap_addr = make_zmq_path(
                "http", meta.bootstrap_server_host, meta.bootstrap_server_port
            )
            # Update expire_time for query task.
            meta.expire_time = now + envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT
            pull_kv_list[bootstrap_addr].query.append(meta)

        for bootstrap_addr, pull_kv_tasks in pull_kv_list.items():
            asyncio.create_task(
                self.handle_bootstrap_group(bootstrap_addr, pull_kv_tasks)
            )

    async def record_send_reqs(self, metadata: MooncakeConnectorMetadata):
        for req_id, block_ids in metadata.reqs_to_send.items():
            if block_ids:
                # Already gone through request_finished()
                send_meta = self.reqs_need_send[req_id]
                send_meta.local_block_ids = block_ids
                send_meta.expire_time = (
                    time.perf_counter() + envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT
                )
                send_meta.ready.set()
            else:
                # From update_state_after_alloc(),
                # but not reach request_finished() yet
                self.reqs_need_send[req_id] = SendBlockMeta(
                    local_block_ids=[],
                    ready=asyncio.Event(),
                )
        for req_id in metadata.reqs_not_processed:
            send_meta = self.reqs_need_send.pop(req_id)
            if send_meta:
                assert not send_meta.ready.is_set()

    def start_load_kv(self, metadata: MooncakeConnectorMetadata):
        if self.kv_role != "kv_producer":
            asyncio.run_coroutine_threadsafe(
                self._start_load_kv(metadata.reqs_to_recv), self.receiver_loop
            )

        if self.kv_role != "kv_consumer":
            asyncio.run_coroutine_threadsafe(
                self.record_send_reqs(metadata), self.sender_loop
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


def _async_loop(loop: asyncio.AbstractEventLoop):
    loop.set_debug(True)
    asyncio.set_event_loop(loop)
    loop.run_forever()


class HTTPClientManager:
    """A self-contained manager for a shared httpx.AsyncClient instance."""

    def __init__(self):
        self._client: httpx.AsyncClient | None = None

    async def get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=None,
                limits=httpx.Limits(
                    max_connections=None, max_keepalive_connections=None
                ),
            )
        return self._client

    async def shutdown(self):
        if self._client:
            await self._client.aclose()
            self._client = None


# ####################################################################
# ## Mooncake Bootstrap Server
# ####################################################################


class MooncakeBootstrapServer:
    """
    A centralized server running on the global rank 0 prefiller worker.
    Its main purpose is to act as a service discovery mechanism.

    1. Prefiller workers register their connection info (IP, port, ranks) here.
    2. Prefiller workers register which requests they will be serving.
    3. Decoder workers query this server to find out which prefiller worker to
       contact for a specific request's KV cache.
    """

    def __init__(self, host: str, port: int):
        # store workers info: {dp_rank: {tp_rank: worker_addr}}
        self.workers: defaultdict[int, dict[int, str]] = defaultdict(dict)
        # store reqs info: {req_id: dp_rank}
        self.req_to_dp_rank: dict[ReqId, int] = {}
        self.tp_size = get_tensor_model_parallel_world_size()

        self.host = host
        self.port = port
        self.app = FastAPI(
            on_startup=[self.prune_startup], on_shutdown=[self.prune_shutdown]
        )
        self._register_routes()
        self.server_thread: threading.Thread | None = None
        self.server: uvicorn.Server | None = None

        self._pruning_task: asyncio.Task | None = None
        # A set of request IDs to be pruned in the next cycle.
        self.reqs_to_prune: set[ReqId] = set()
        self.prune_interval: int = envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT * 2

    def __del__(self):
        self.shutdown()

    def _register_routes(self):
        # All methods are async. No need to use lock to protect data.
        self.app.post("/register_worker")(self.register_worker)
        self.app.post("/register_request")(self.register_request)
        self.app.post("/query_requests", response_model=QueryRequestsResponse)(
            self.query_requests
        )

    def start(self):
        if self.server_thread:
            return

        config = uvicorn.Config(
            app=self.app, host=self.host, port=self.port, log_level=http_log_level
        )
        self.server = uvicorn.Server(config=config)
        self.server_thread = threading.Thread(
            target=self.server.run, name="mooncake_bootstrap_server", daemon=True
        )
        self.server_thread.start()
        while not self.server.started:
            time.sleep(0.1)  # Wait for the server to start
        logger.info("Mooncake Bootstrap Server started at %s:%d", self.host, self.port)

    def shutdown(self):
        if self.server_thread is None or self.server is None or not self.server.started:
            return

        self.server.should_exit = True
        self.server_thread.join()
        logger.info("Mooncake Bootstrap Server stopped.")

    async def register_worker(self, payload: RegisterWorkerPayload):
        """Handles registration of a prefiller worker."""
        self.workers[payload.dp_rank][payload.tp_rank] = payload.worker_addr
        logger.debug(
            "Registered worker: dp_rank=%d, tp_rank=%d at %s",
            payload.dp_rank,
            payload.tp_rank,
            payload.worker_addr,
        )
        return {"status": "ok"}

    async def register_request(self, payload: RegisterRequestPayload):
        """Handles associating a request ID with a DP rank."""
        if (reg := self.req_to_dp_rank.get(payload.req_id)) is not None:
            raise HTTPException(
                status_code=400,
                detail=f"Request '{payload.req_id}' already registered with rank {reg} "
                f"but still want to register with rank {payload.dp_rank}",
            )
        self.req_to_dp_rank[payload.req_id] = payload.dp_rank
        logger.debug(
            "Registered request '%s' with dp_rank=%d", payload.req_id, payload.dp_rank
        )
        return {"status": "ok"}

    async def query_requests(
        self, payload: QueryRequestsPayload
    ) -> QueryRequestsResponse:
        """Handles a query (batch req_ids) from a decoder worker."""

        # We only support homogeneous TP now.
        if self.tp_size != payload.tp_size:
            raise HTTPException(
                status_code=400,
                detail=(
                    "heterogeneous TP is not supported yet. TP size mismatch: "
                    f"expected {self.tp_size}, got {payload.tp_size}"
                ),
            )

        results: dict[ReqId, tuple[str, list[str]]] = {}
        for req_id in payload.req_ids:
            prefiller_dp_rank = self.req_to_dp_rank.get(req_id)
            if prefiller_dp_rank is None:
                results[req_id] = ("retry", [])
                continue

            prefiller_dp_group = self.workers.get(prefiller_dp_rank)
            if not prefiller_dp_group:
                results[req_id] = (
                    f"Prefiller DP group {prefiller_dp_rank} not found",
                    [],
                )
                continue

            worker_addrs = self.match_rank(prefiller_dp_group, payload)
            if not worker_addrs:
                results[req_id] = (
                    (
                        f"Prefiller TP rank {payload.tp_rank} not matched "
                        f"in DP group {prefiller_dp_rank}"
                    ),
                    [],
                )
                continue

            results[req_id] = ("ok", worker_addrs)

        return QueryRequestsResponse(results=results)

    def match_rank(
        self, prefiller_dp_group: dict[int, str], payload: QueryRequestsPayload
    ):
        worker_addrs = []
        # Todo: Support heterogeneous TP.
        # One payload.tp_rank may map to multiple prefiller_tp_ranks.
        prefiller_tp_rank = payload.tp_rank
        work_addr = prefiller_dp_group.get(prefiller_tp_rank)
        if work_addr is not None:
            worker_addrs.append(work_addr)
        return worker_addrs

    async def _pruning_loop(self):
        """The dedicated background task that periodically prunes stale requests."""
        try:
            while True:
                # Perform the pruning logic
                for req_id in self.reqs_to_prune:
                    del self.req_to_dp_rank[req_id]
                logger.debug("pruned %d requests", len(self.reqs_to_prune))
                self.reqs_to_prune = set(self.req_to_dp_rank)

                # Wait for the specified interval
                await asyncio.sleep(self.prune_interval)
        except asyncio.CancelledError:
            # This is the expected way to exit the loop
            pass
        except Exception as e:
            # Log any other unexpected errors in the background task
            logger.error("Error in background pruning loop: %s", e)

    async def prune_startup(self):
        self._pruning_task = asyncio.create_task(self._pruning_loop())

    async def prune_shutdown(self):
        if self._pruning_task:
            self._pruning_task.cancel()
            await self._pruning_task


def get_mooncake_bootstrap_addr(vllm_config: VllmConfig) -> tuple[str, int]:
    """
    Returns the address of the Mooncake bootstrap server.
    This is only used by prefillers to register workers and requests.
    Decoders should get addr from kv_transfer_params.
    """
    assert vllm_config.parallel_config
    host = vllm_config.parallel_config.data_parallel_master_ip
    port = envs.VLLM_MOONCAKE_BOOTSTRAP_PORT
    return (host, port)
