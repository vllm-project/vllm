# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Optional

import httpx
import msgspec
import numpy as np
import torch
import zmq
import zmq.asyncio

from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import (
    EngineId,
    TpKVTopology,
    get_current_attn_backend,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_utils import (
    MooncakeBootstrapServer,
    RegisterWorkerPayload,
)
from vllm.distributed.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    is_local_first_rank,
)
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.utils.network_utils import get_ip, make_zmq_path, make_zmq_socket
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.backends.utils import get_kv_cache_layout
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus

try:
    from mooncake.engine import TransferEngine
except ImportError as e:
    raise ImportError(
        "Please install mooncake by following the instructions at "
        "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "
        "to run VLLM with MooncakeTransferEngine."
    ) from e

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

ReqId = str  # Internal scheduler request ID
TransferId = str  # KV transfer coordination ID (shared by P/D)

logger = init_logger(__name__)


class MooncakeXferMetadata(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
):
    remote_hostname: str
    remote_port: int
    remote_tp_size: int
    remote_tp_rank: int
    req_blocks: dict[ReqId, tuple[TransferId, list[int]]]
    kv_caches_base_addr: list[int]


class MooncakeXferResponseStatus(IntEnum):
    # Transfer finished
    FINISH = 0
    # Continue to receive
    CONTINUE = 1
    # Something wrong, see err_msg
    ERROR = 2


class MooncakeXferResponse(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
):
    status: MooncakeXferResponseStatus
    ok_reqs: list[ReqId] | None = None
    err_reqs: list[ReqId] | None = None
    err_msg: str | None = None


@dataclass
class PullReqMeta:
    d_req_id: ReqId
    transfer_id: TransferId
    local_block_ids: list[int]
    remote_engine_id: EngineId
    remote_bootstrap_addr: str
    # Set expire time to avoid infinitely sending requests.
    expire_time: float = float("inf")
    # Designed for one D pairing to multiple P
    pull_tasks_count: int = 0


@dataclass
class SendBlockMeta:
    p_req_id: ReqId
    transfer_id: TransferId
    local_block_ids: list[int]
    ready: asyncio.Event
    expire_time: float = float("inf")
    need_send: int = 0
    sended: int = 0
    sending: int = 0


class MooncakeConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        # Use (engine_id, dp_rank) to group reqs with same dp.
        # See comments in MooncakeBootstrapServer.
        self.reqs_to_recv: dict[EngineId, dict[ReqId, PullReqMeta]] = defaultdict(dict)
        self.reqs_to_send: dict[ReqId, tuple[TransferId, list[int]]] = {}
        self.reqs_not_processed: set[TransferId] = set()

    def add_new_req(
        self,
        request_id: ReqId,
        local_block_ids: list[int],
        kv_transfer_params: dict[str, Any],
        load_remote_cache: bool = True,
    ):
        transfer_id = kv_transfer_params["transfer_id"]
        if load_remote_cache:
            remote_engine_id = kv_transfer_params["remote_engine_id"]
            self.reqs_to_recv[remote_engine_id][request_id] = PullReqMeta(
                d_req_id=request_id,
                local_block_ids=local_block_ids,
                remote_engine_id=remote_engine_id,
                remote_bootstrap_addr=kv_transfer_params["remote_bootstrap_addr"],
                transfer_id=transfer_id,
            )
        else:
            self.reqs_to_send[request_id] = (transfer_id, local_block_ids)


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

        assert vllm_config.kv_transfer_config
        self.is_kv_producer: bool = (
            vllm_config.kv_transfer_config.kv_role == "kv_producer"
        )
        self.is_kv_consumer: bool = (
            vllm_config.kv_transfer_config.kv_role == "kv_consumer"
        )
        logger.info("Initializing Mooncake Transfer Engine Scheduler %s", engine_id)

        # Requests that need to start recv/send.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        self._reqs_need_recv: dict[ReqId, tuple[Request, list[int]]] = {}
        self._reqs_need_send: dict[ReqId, tuple[Request, list[int]]] = {}
        # Reqs to remove from processed set because they're not to send after
        # remote prefill or aborted.
        self._reqs_not_processed: set[TransferId] = set()

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
            assert not self.is_kv_producer
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
            "req_id=%s num_external_tokens=%s, kv_transfer_params=%s",
            request.request_id,
            num_external_tokens,
            params,
        )

        if not params:
            return

        if params.get("do_remote_prefill"):
            assert not self.is_kv_producer
            if all(
                p in params
                for p in ("remote_engine_id", "remote_bootstrap_addr", "transfer_id")
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
            assert not self.is_kv_consumer
            if not params.get("transfer_id"):
                logger.warning(
                    "Got invalid KVTransferParams: %s. This "
                    "request will not utilize KVTransfer",
                    params,
                )
            else:
                # Add an empty list to worker to create event.
                self._reqs_need_send[request.request_id] = (request, [])

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = MooncakeConnectorMetadata()

        # Loop through scheduled reqs and convert to PullReqMeta.
        if not self.is_kv_producer:
            for req_id, (req, block_ids) in self._reqs_need_recv.items():
                assert req.kv_transfer_params is not None
                meta.add_new_req(
                    request_id=req_id,
                    local_block_ids=block_ids,
                    kv_transfer_params=req.kv_transfer_params,
                )
            self._reqs_need_recv.clear()

        if not self.is_kv_consumer:
            for req_id, (req, block_ids) in self._reqs_need_send.items():
                assert req.kv_transfer_params is not None
                meta.add_new_req(
                    request_id=req_id,
                    local_block_ids=block_ids,
                    kv_transfer_params=req.kv_transfer_params,
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
            "MooncakeConnector request_finished, req_id=%s, request_status=%s, "
            "kv_transfer_params=%s",
            request.request_id,
            request.status,
            params,
        )
        if not params or not params.get("transfer_id"):
            return False, None

        if params.get("do_remote_prefill"):
            # If do_remote_prefill is still True when the request is finished,
            # update_state_after_alloc must not have been called (the request
            # must have been aborted before it was scheduled).
            # To avoid stranding the prefill blocks in the prefill instance,
            # we must add empty block_ids to _reqs_need_recv so that our
            # worker side will notify and free blocks in the prefill instance.
            assert not self.is_kv_producer
            self._reqs_need_recv[request.request_id] = (request, [])
            params["do_remote_prefill"] = False
            return False, None

        if not params.get("do_remote_decode"):
            return False, None

        assert not self.is_kv_consumer

        if request.status != RequestStatus.FINISHED_LENGTH_CAPPED:
            # Also include the case of a P/D Prefill request with immediate
            # block free (eg abort). Stop tracking this request.
            self._reqs_not_processed.add(params["transfer_id"])
            return False, None

        # TODO: check whether block_ids actually ever be 0. If not we could
        # remove the conditional below
        delay_free_blocks = len(block_ids) > 0

        if delay_free_blocks:
            self._reqs_need_send[request.request_id] = (request, block_ids)

        return delay_free_blocks, None


class MooncakeConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        logger.info("Initializing Mooncake Transfer Engine worker %s", engine_id)

        self.vllm_config = vllm_config

        self.engine = TransferEngine()
        self.hostname = get_ip()

        assert (kv_transfer_config := vllm_config.kv_transfer_config)
        self.is_kv_producer: bool = kv_transfer_config.kv_role == "kv_producer"
        self.is_kv_consumer: bool = kv_transfer_config.kv_role == "kv_consumer"
        self.num_sender_workers = kv_transfer_config.kv_connector_extra_config.get(
            "num_workers", 10
        )
        # Create more tasks than workers to keep the thread pool saturated.
        # Tasks can await async events, so a surplus (2x is a robust heuristic)
        # prevents workers from idling.
        self.num_sender_tasks = self.num_sender_workers * 2
        protocol = kv_transfer_config.kv_connector_extra_config.get(  # type: ignore[union-attr]
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

        self._remote_agents: dict[EngineId, dict[int, dict[int, str]]] = {}
        self._pending_bootstrap_querys: dict[str, asyncio.Event] = {}
        self.side_channel_port: int = 0  # we will bind it in register_kv_caches()
        self.engine_id: EngineId = engine_id
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_blocks = 0

        assert (parallel_config := vllm_config.parallel_config)
        dp_rank = parallel_config.data_parallel_index
        dp_local_rank = parallel_config.data_parallel_rank_local
        self.dp_rank = dp_local_rank if parallel_config.local_engines_only else dp_rank
        pp_size = vllm_config.parallel_config.pipeline_parallel_size
        if pp_size > 1:
            raise ValueError(
                "Mooncake Transfer Engine does not support pipeline parallelism yet."
            )
        self.pp_rank = get_pp_group().rank_in_group

        self.kv_caches_base_addr: list[int] = []
        self.device_kv_caches: dict[str, torch.Tensor] = {}
        self.reqs_need_send: dict[TransferId, SendBlockMeta] = {}

        # Only used by prefillers.
        host, port = get_mooncake_bootstrap_addr(vllm_config)
        self.bootstrap_addr = make_zmq_path("http", host, port)

        # For kv_both, we will act both prefiller and decoder.
        if not self.is_kv_consumer:
            # Background threads for sending kvcaches to D.
            self._sender_executor = ThreadPoolExecutor(
                max_workers=self.num_sender_workers,
                thread_name_prefix="vllm-mooncake-sender",
            )
            logger.debug(
                "Mooncake Prefiller: use %d workers to send kvcaches",
                self.num_sender_workers,
            )
            # An asyncio queue to buffer incoming requests for the sender
            self.sender_worker_queue = asyncio.Queue[tuple[bytes, bytes]]()
            self.sender_loop = asyncio.new_event_loop()
            # Background thread for processing new sending requests.
            self._sender_listener_t = threading.Thread(
                target=_async_loop, args=(self.sender_loop,), daemon=True
            )
            self._sender_listener_t.start()

            # Start bootstrap server on global rank 0.
            if should_launch_bootstrap_server(vllm_config):
                self.bootstrap_server = MooncakeBootstrapServer(
                    vllm_config, "0.0.0.0", port
                )
                self.bootstrap_server.start()

        if not self.is_kv_producer:
            self.receiver_loop = asyncio.new_event_loop()
            self._mooncake_receiver_t = threading.Thread(
                target=_async_loop, args=(self.receiver_loop,), daemon=True
            )
            self._mooncake_receiver_t.start()
            logger.debug("Mooncake Decoder: start receiver thread")

        self.finished_sending_reqs: set[ReqId] = set()
        self.finished_recving_reqs: set[ReqId] = set()

        self.block_size = vllm_config.cache_config.block_size
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.use_mla = self.model_config.use_mla

        # Get the attention backend from the first layer
        # NOTE (NickLucche) models with multiple backends are not supported yet
        backend = get_current_attn_backend(vllm_config)
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

        self.async_zmq_ctx = zmq.asyncio.Context()
        self._encoder = msgspec.msgpack.Encoder()
        self._xfer_meta_decoder = msgspec.msgpack.Decoder(MooncakeXferMetadata)
        self._xfer_resp_decoder = msgspec.msgpack.Decoder(MooncakeXferResponse)

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        """Cleanup background threads on destruction."""
        self.async_zmq_ctx.term()
        if not self.is_kv_consumer:
            self._sender_executor.shutdown(wait=False)
            if self.sender_loop.is_running():
                self.sender_loop.call_soon_threadsafe(self.sender_loop.stop)
                self._sender_listener_t.join()
            if should_launch_bootstrap_server(self.vllm_config):
                self.bootstrap_server.shutdown()
        if not self.is_kv_producer and self.receiver_loop.is_running():
            self.receiver_loop.call_soon_threadsafe(self.receiver_loop.stop)
            self._mooncake_receiver_t.join()

    async def register_worker_with_bootstrap(self):
        url = self.bootstrap_addr + "/register"
        worker_addr = make_zmq_path("tcp", self.hostname, self.side_channel_port)
        payload = RegisterWorkerPayload(
            engine_id=self.engine_id,
            dp_rank=self.dp_rank,
            tp_rank=self.tp_rank,
            pp_rank=self.pp_rank,
            addr=worker_addr,
        )
        while True:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(url, json=payload.model_dump())
                    response.raise_for_status()
                logger.debug("Successfully registered with bootstrap server at %s", url)
                break
            except httpx.ConnectError:
                # Bootstrap server not ready, wait for a while and retry.
                await asyncio.sleep(1)
            except Exception as e:
                err_msg = (
                    e.response.text if isinstance(e, httpx.HTTPStatusError) else str(e)
                )
                logger.error(
                    "Error registering %s with bootstrap server: %s", payload, err_msg
                )
                raise e

    async def _mooncake_sender_listener(self, ready_event: threading.Event):
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
        sender_tasks = [
            asyncio.create_task(self._sender_worker(sock))
            for _ in range(self.num_sender_tasks)
        ]

        ready_event.set()

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
            for task in sender_tasks:
                task.cancel()
            await asyncio.gather(*sender_tasks, return_exceptions=True)
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
                    error_response = MooncakeXferResponse(
                        status=MooncakeXferResponseStatus.ERROR, err_msg=str(e)
                    )
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
        remote_tp_ranks = self.kv_topo.get_target_remote_ranks(meta.remote_tp_size)
        if self.tp_rank not in remote_tp_ranks:
            # This D worker does not pair with the P worker.
            msg = f"This P tp_rank {self.tp_rank} not in remote D target ranks {remote_tp_ranks}"  # noqa: E501
            logger.error(msg)
            response = MooncakeXferResponse(
                status=MooncakeXferResponseStatus.ERROR,
                err_msg=msg,
            )
            await sock.send_multipart((identity, self._encoder.encode(response)))
            return
        for d_req_id, (transfer_id, _) in meta.req_blocks.items():
            if transfer_id not in self.reqs_need_send:
                # This req is not enqueued in P side yet, create it here.
                self.reqs_need_send[transfer_id] = SendBlockMeta(
                    p_req_id="",
                    transfer_id=transfer_id,
                    local_block_ids=[],
                    ready=asyncio.Event(),
                )
            send_meta = self.reqs_need_send[transfer_id]
            pending_reqs[d_req_id] = send_meta

        async def wait_and_ret(
            d_req_id: ReqId, send_meta: SendBlockMeta
        ) -> tuple[ReqId, SendBlockMeta]:
            await send_meta.ready.wait()
            return d_req_id, send_meta

        wait_tasks = [
            asyncio.create_task(wait_and_ret(d_req_id, send_meta))
            for d_req_id, send_meta in pending_reqs.items()
        ]

        while wait_tasks:
            done, pending = await asyncio.wait(
                wait_tasks,
                timeout=envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if not done:
                # Timeout, abort all pending requests.
                for task in wait_tasks:
                    task.cancel()
                response = MooncakeXferResponse(
                    status=MooncakeXferResponseStatus.FINISH,
                    err_reqs=list(pending_reqs),
                    err_msg="Timeout waiting for P side ready.",
                )
                await sock.send_multipart((identity, self._encoder.encode(response)))
                break

            wait_tasks = list(pending)
            response_status = (
                MooncakeXferResponseStatus.CONTINUE
                if wait_tasks
                else MooncakeXferResponseStatus.FINISH
            )
            ready_reqs: list[tuple[ReqId, SendBlockMeta]] = []
            for task in done:
                d_req_id, send_meta = task.result()
                del pending_reqs[d_req_id]
                # Do we still in reqs_need_send (not expired)?
                if send_meta.transfer_id in self.reqs_need_send:
                    # Mark it sending to avoid expiration.
                    send_meta.sending += 1
                    if not send_meta.need_send:
                        self.resolve_need_send(send_meta, remote_tp_ranks)
                    ready_reqs.append((d_req_id, send_meta))
                # Otherwise (expired, very unlikely), forget it. Do not let D retry.

            src_ptrs, dst_ptrs, lengths, err_reqs = await self._build_transfer_params(
                ready_reqs, meta
            )

            if err_reqs:
                response = MooncakeXferResponse(
                    status=response_status,
                    err_reqs=err_reqs,
                    err_msg="P num blocks less than D",
                )
                await sock.send_multipart((identity, self._encoder.encode(response)))

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
                    err_reqs = []
                    for d_req_id, send_meta in ready_reqs:
                        send_meta.sending -= 1
                        err_reqs.append(d_req_id)
                    # Do best effort to transfer the remaining reqs.
                    response = MooncakeXferResponse(
                        status=response_status,
                        err_reqs=err_reqs,
                        err_msg=f"Mooncake transfer engine returned {ret_value}",
                    )
                    await sock.send_multipart(
                        (identity, self._encoder.encode(response))
                    )
                    continue

            for d_req_id, send_meta in ready_reqs:
                # Todo: for heterogeneous TP (one P pairs to multiple D),
                # we need to check whether all headers are sent.
                # If not, we should set expire_time to normal and skip the below.
                send_meta.sending -= 1
                send_meta.sended += 1
                if send_meta.sended == send_meta.need_send:
                    del self.reqs_need_send[send_meta.transfer_id]
                    self.finished_sending_reqs.add(send_meta.p_req_id)

            response = MooncakeXferResponse(
                status=response_status,
                ok_reqs=[d_req_id for d_req_id, _ in ready_reqs],
            )
            await sock.send_multipart((identity, self._encoder.encode(response)))

    def resolve_need_send(self, send_meta: SendBlockMeta, remote_tp_ranks: list[int]):
        # Prepare for heterogeneous TP (one P pairs to multiple D)
        send_meta.need_send = len(remote_tp_ranks)
        if send_meta.need_send != 1:
            logger.error("Mooncake: Heterogeneous TP is not supported yet.")

    async def _build_transfer_params(
        self,
        ready_reqs: list[tuple[ReqId, SendBlockMeta]],
        agent_meta: MooncakeXferMetadata,
    ) -> tuple[list[int], list[int], list[int], list[ReqId]]:
        src_ptrs = []
        dst_ptrs = []
        lengths = []
        err_reqs: list[ReqId] = []
        local_base_addr = self.kv_caches_base_addr
        remote_base_addr = agent_meta.kv_caches_base_addr
        block_len = self.block_len
        remote_session = f"{agent_meta.remote_hostname}:{agent_meta.remote_port}"

        for d_req_id, send_meta in ready_reqs:
            _, remote_block_ids = agent_meta.req_blocks[d_req_id]
            num_remote_blocks = len(remote_block_ids)
            if num_remote_blocks == 0:
                continue

            local_block_ids = send_meta.local_block_ids
            # Partial prefix cache hit: just read uncomputed blocks.
            num_local_blocks = len(local_block_ids)
            if num_local_blocks < num_remote_blocks:
                logger.error(
                    "req %s: local blocks(%d) less than remote blocks(%d)!",
                    d_req_id,
                    num_local_blocks,
                    num_remote_blocks,
                )
                err_reqs.append(d_req_id)
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
                d_req_id,
                num_remote_blocks,
                remote_session,
            )

        return src_ptrs, dst_ptrs, lengths, err_reqs

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
        if self.is_kv_consumer:
            return

        ready_event = threading.Event()
        asyncio.run_coroutine_threadsafe(
            self._mooncake_sender_listener(ready_event), self.sender_loop
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

        expired_transfer_id = []
        for transfer_id, send_meta in self.reqs_need_send.items():
            if (
                send_meta.p_req_id
                and send_meta.expire_time < now
                and send_meta.sending == 0
            ):
                logger.warning(
                    "Request %s timed out after %d seconds without "
                    "being sent. Freeing its blocks on the producer side.",
                    send_meta.p_req_id,
                    envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT,
                )
                finished_sending_reqs.add(send_meta.p_req_id)
                expired_transfer_id.append(transfer_id)

        for transfer_id in expired_transfer_id:
            del self.reqs_need_send[transfer_id]

        return finished_sending_reqs

    def get_finished(self) -> tuple[set[str] | None, set[str] | None]:
        """
        Get requests that are done sending or recving on this specific worker.
        The scheduler process (via the MultiprocExecutor) will use this output
        to track which workers are done.
        """
        recv_fut = None
        send_fut = None
        if not self.is_kv_producer:
            recv_fut = asyncio.run_coroutine_threadsafe(
                self.fetch_finished_recving_reqs(), self.receiver_loop
            )

        if not self.is_kv_consumer:
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

    async def receive_kv_from_single_worker(
        self,
        worker_addr: str,
        pull_metas: dict[ReqId, PullReqMeta],
    ):
        req_ids = set(pull_metas)
        metadata = MooncakeXferMetadata(
            remote_hostname=self.hostname,
            remote_port=self.rpc_port,
            remote_tp_size=self.tp_size,
            remote_tp_rank=self.tp_rank,
            req_blocks={
                req_id: (pull_meta.transfer_id, pull_meta.local_block_ids)
                for req_id, pull_meta in pull_metas.items()
            },
            kv_caches_base_addr=self.kv_caches_base_addr,
        )

        encoded_data = self._encoder.encode(metadata)
        logger.debug(
            "Size of encoded MooncakeXferMetadata: %d bytes", len(encoded_data)
        )
        logger.debug(
            "Sending kv transfer request for %s on path: %s", req_ids, worker_addr
        )

        # Send query for the request.
        try:
            with make_zmq_socket(
                self.async_zmq_ctx, worker_addr, zmq.DEALER, bind=False, linger=0
            ) as sock:
                await sock.send(encoded_data)
                while True:
                    ret_msg = await sock.recv()
                    response = self._xfer_resp_decoder.decode(ret_msg)
                    if response.status == MooncakeXferResponseStatus.ERROR:
                        logger.error(
                            "Error happens during tranfering kvcache for %s: %s",
                            req_ids,
                            response.err_msg,
                        )
                        return
                    self.process_pulling_result(response, pull_metas)
                    if response.status == MooncakeXferResponseStatus.FINISH:
                        break
        except zmq.ContextTerminated:
            logger.debug("ZMQ context terminated, exiting Mooncake receiver thread.")
        except Exception as e:
            logger.error("MooncakeXferMetadata transfer failed for %s: %s", req_ids, e)
            return

    def process_pulling_result(
        self,
        response: MooncakeXferResponse,
        pull_metas: dict[ReqId, PullReqMeta],
    ):
        ok_reqs: list[ReqId] = response.ok_reqs or []

        for req_id in ok_reqs:
            pull_meta = pull_metas[req_id]
            # No race because we are in async loop.
            pull_meta.pull_tasks_count -= 1
            if pull_meta.pull_tasks_count == 0:
                self.finished_recving_reqs.add(pull_meta.d_req_id)

        if ok_reqs:
            logger.debug("pulling kv_caches for %s finished", ok_reqs)

        if response.err_reqs:
            logger.error(
                "pulling kv_caches for %s failed: %s",
                response.err_reqs,
                response.err_msg,
            )

    async def _connect_to_prefiller_bootstrap(self, remote_bootstrap_addr: str):
        url = remote_bootstrap_addr + "/query"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()
                data: dict = response.json()
                for _, dp_entry in data.items():
                    remote_engine_id = dp_entry["engine_id"]
                    self._remote_agents[remote_engine_id] = {
                        int(tp_rank): {
                            int(pp_rank): worker_addr
                            for pp_rank, worker_addr in tp_entry.items()
                        }
                        for tp_rank, tp_entry in dp_entry["worker_addr"].items()
                    }
                    self._tp_size[remote_engine_id] = len(dp_entry["worker_addr"])
        except Exception as e:
            logger.error(
                "Failed to connect to bootstrap server %s: %s",
                remote_bootstrap_addr,
                e,
            )

        # Always notify others regardless of connection success or failure.
        self._pending_bootstrap_querys[remote_bootstrap_addr].set()
        del self._pending_bootstrap_querys[remote_bootstrap_addr]

    def receive_kv(
        self,
        remote_engine_id: EngineId,
        pull_metas: dict[ReqId, PullReqMeta],
    ):
        remote_tp_ranks = self.kv_topo.get_target_remote_ranks_from_engine_id(
            remote_engine_id
        )
        count = len(remote_tp_ranks)
        if count != 1:
            logger.error("Mooncake: Heterogeneous TP is not supported yet.")
            return
        for pull_meta in pull_metas.values():
            pull_meta.pull_tasks_count = count
        for remote_tp_rank in remote_tp_ranks:
            worker_addr = self._remote_agents[remote_engine_id][remote_tp_rank][0]
            asyncio.create_task(
                self.receive_kv_from_single_worker(worker_addr, pull_metas)
            )

    async def handle_new_engine_id(
        self,
        remote_engine_id: EngineId,
        pull_metas: dict[ReqId, PullReqMeta],
    ):
        remote_bootstrap_addr = next(iter(pull_metas.values())).remote_bootstrap_addr
        if remote_bootstrap_addr not in self._pending_bootstrap_querys:
            self._pending_bootstrap_querys[remote_bootstrap_addr] = asyncio.Event()
            await self._connect_to_prefiller_bootstrap(remote_bootstrap_addr)
        else:
            await self._pending_bootstrap_querys[remote_bootstrap_addr].wait()

        if remote_engine_id not in self._remote_agents:
            logger.error(
                "Failed to find remote engine_id %s from bootstrap server %s",
                remote_engine_id,
                remote_bootstrap_addr,
            )
            return

        self.receive_kv(remote_engine_id, pull_metas)

    async def _start_load_kv(
        self, reqs_to_recv: dict[EngineId, dict[ReqId, PullReqMeta]]
    ):
        for remote_engine_id, pull_metas in reqs_to_recv.items():
            if remote_engine_id not in self._remote_agents:
                asyncio.create_task(
                    self.handle_new_engine_id(remote_engine_id, pull_metas)
                )
                continue
            self.receive_kv(remote_engine_id, pull_metas)

    async def record_send_reqs(self, metadata: MooncakeConnectorMetadata):
        for p_req_id, (transfer_id, block_ids) in metadata.reqs_to_send.items():
            if block_ids:
                # Already gone through request_finished()
                send_meta = self.reqs_need_send[transfer_id]
                send_meta.p_req_id = p_req_id
                send_meta.local_block_ids = block_ids
                send_meta.expire_time = (
                    time.perf_counter() + envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT
                )
                send_meta.ready.set()
            else:
                # From update_state_after_alloc(),
                # but not reach request_finished() yet
                # This may be already created by send_kv_to_decode()
                # when D is sending MooncakeXferMetadata.
                if transfer_id not in self.reqs_need_send:
                    self.reqs_need_send[transfer_id] = SendBlockMeta(
                        p_req_id=p_req_id,
                        transfer_id=transfer_id,
                        local_block_ids=[],
                        ready=asyncio.Event(),
                    )
        for transfer_id in metadata.reqs_not_processed:
            send_meta = self.reqs_need_send.pop(transfer_id)
            if send_meta:
                assert not send_meta.ready.is_set()

    def start_load_kv(self, metadata: MooncakeConnectorMetadata):
        if not self.is_kv_producer and metadata.reqs_to_recv:
            asyncio.run_coroutine_threadsafe(
                self._start_load_kv(metadata.reqs_to_recv), self.receiver_loop
            )

        if not self.is_kv_consumer and (
            metadata.reqs_to_send or metadata.reqs_not_processed
        ):
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


def get_mooncake_side_channel_port(vllm_config: VllmConfig) -> int:
    # This logic is now centralized
    return (
        envs.VLLM_MOONCAKE_BOOTSTRAP_PORT
        + vllm_config.parallel_config.data_parallel_index
        * vllm_config.parallel_config.tensor_parallel_size
    )


def _async_loop(loop: asyncio.AbstractEventLoop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


def should_launch_bootstrap_server(vllm_config: VllmConfig) -> bool:
    assert (parallel_config := vllm_config.parallel_config)
    # In hybrid or external LB mode,
    # each instance should have its own bootstrap server.
    #
    # In internal LB mode,
    # only the real global first rank need to launch the bootstrap server.
    return is_local_first_rank() and (
        parallel_config.local_engines_only or parallel_config.data_parallel_index == 0
    )


def get_mooncake_bootstrap_addr(vllm_config: VllmConfig) -> tuple[str, int]:
    """
    Returns the address of the Mooncake bootstrap server.
    This is only used by prefillers to register workers.
    Decoders should get addr from kv_transfer_params.
    """
    assert (parallel_config := vllm_config.parallel_config)
    if parallel_config.local_engines_only:
        # In hybrid or external LB mode, connect to local server.
        host = "127.0.0.1"
    else:
        host = parallel_config.data_parallel_master_ip
    port = envs.VLLM_MOONCAKE_BOOTSTRAP_PORT
    return (host, port)
