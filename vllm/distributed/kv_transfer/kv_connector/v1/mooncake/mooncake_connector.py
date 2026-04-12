# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import aiohttp
import asyncio
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any

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
from vllm.distributed import get_pcp_group
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_utils import (
    MooncakeBootstrapServer,
    RegisterWorkerPayload,
    RegisterEngineCorePayload
)
from vllm.distributed.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    is_local_first_rank,
    get_decode_context_model_parallel_rank
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
DONE_SENDING_MSG = b"done_sending_msg"

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
class MooncakeAgentMetadata:
    engine_id: str
    worker_addr: str
    kv_caches_base_addr: list[int]
    rpc_addr: str
    dp_size: int
    tp_size: int
    pcp_size: int
    dcp_size: int
    dp_rank: int
    tp_rank: int
    pcp_rank: int
    dcp_rank: int

class MooncakeLayerwiseXferMetadata(msgspec.Struct, omit_defaults=True, dict=True):
    # agent_info: {engine_id: {tp: (worker_addr, rpc_addr, kv_cache_base_addr)}}
    # req_blocks: dict[ReqId, tuple[TransferId, list[int]]]
    agent_info: dict[EngineId, dict[int, tuple[str, str, list[int]]]] | None
    req_blocks: dict[ReqId, tuple[EngineId, TransferId, list[int]]]


@dataclass
class MooncakeHandshakePayload(KVConnectorHandshakeMetadata):
    agent_metadata_bytes: bytes  # MooncakeAgentMetadata encoded

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
class PushReqMeta:
    d_req_id: ReqId
    transfer_id: TransferId
    local_block_ids: list[int]
    remote_block_ids: list[int]
    remote_engine_id: EngineId
    # Designed for one D pairing to multiple P
    push_tasks_count: int = 0

@dataclass
class TransferMeta:
    src: list[int]
    dst: list[int]
    length: list[int]
    req_ids: list[str]
    remote_worker_addr: str


@dataclass
class SendTask:
    send_request: dict[str, PushReqMeta] = field(default_factory=dict)
    # pd_head_ratio == 1 use
    wait_event: torch.npu.Event | None = None
    # pd_head_ratio > 1 use
    k_cache: torch.Tensor | None = None
    v_cache: torch.Tensor | None = None
    layer_idx: int = 0

@dataclass
class SendBlockMeta:
    p_req_id: ReqId
    transfer_id: TransferId
    local_block_ids: list[int]
    ready: asyncio.Event
    expire_time: float = float("inf")
    need_send: int = 0
    sent: int = 0
    sending: int = 0


class MooncakeConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        # Use (engine_id, dp_rank) to group reqs with same dp.
        # See comments in MooncakeBootstrapServer.
        self.reqs_to_recv: dict[EngineId, dict[ReqId, PullReqMeta]] = defaultdict(dict)
        self.reqs_to_send: dict[ReqId, tuple[TransferId, list[int]]] = {}
        self.reqs_not_processed: set[TransferId] = set()
        self.agent_info = dict()  # {engine_id: {tp: (worker_addr, rpc_addr, kv_cache_base_addr)}}
        self.send_task: SendTask = SendTask()
    
    def __str__(self):
        """User-friendly string representation"""
        return (f"MooncakeConnectorMetadata("
                f"{self.agent_info=},"
                f"{self.reqs_to_recv=}, "
                f"{self.reqs_to_send=}, "
                f"{self.reqs_not_processed=},"
                f"{self.send_task=},"
                f"{self.send_task.send_request=},"
                )
    
    def add_new_req(
        self,
        request_id: ReqId,
        local_block_ids: list[int],
        kv_transfer_params: dict[str, Any],
        load_remote_cache: bool = True,
        use_layerwise=False,
        agent_info=None,
        remote_block_ids=None,
        remote_engine_id=None,
        d_req_id=None
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
            if not use_layerwise:
                self.reqs_to_send[request_id] = (transfer_id, local_block_ids)
            else:
                if agent_info is not None:
                    self.agent_info.update(agent_info)
                self.send_task.send_request[request_id] = PushReqMeta(
                    d_req_id=d_req_id,
                    transfer_id=transfer_id,
                    local_block_ids=local_block_ids,
                    remote_block_ids=remote_block_ids,
                    remote_engine_id=remote_engine_id,
                    push_tasks_count=1
                )


class MooncakeConnector(KVConnectorBase_V1):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
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
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, MooncakeConnectorMetadata)
        self.connector_worker.save_kv_layer(layer_name, kv_layer, attn_metadata, self._connector_metadata)

    def wait_for_save(self):
        pass

    def get_handshake_metadata(self) -> KVConnectorHandshakeMetadata | None:
        assert self.connector_worker is not None
        return self.connector_worker.xfer_handshake_metadata
    
    def set_xfer_handshake_metadata(
        self, metadata: dict[int, KVConnectorHandshakeMetadata]
    ) -> None:
        """
        Set the KV connector handshake metadata for this connector.

        Args:
            metadata (dict): the handshake metadata to set.
        """
        assert self.connector_scheduler is not None
        self.connector_scheduler.set_xfer_handshake_metadata(metadata)

class MooncakeConnectorScheduler:
    """Implementation of Scheduler side methods"""

    _DEFAULT_HANDLE_REQUEST_MAX_WORKERS = max(4, min(32, (os.cpu_count() or 1) * 4))

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        self.vllm_config = vllm_config

        assert vllm_config.kv_transfer_config
        self.is_kv_producer: bool = (
            vllm_config.kv_transfer_config.kv_role == "kv_producer"
        )
        self.is_kv_consumer: bool = (
            vllm_config.kv_transfer_config.kv_role == "kv_consumer"
        )
        self.use_layerwise = vllm_config.kv_transfer_config.kv_connector_extra_config.get("use_layerwise", False)
        logger.info("Initializing Mooncake Transfer Engine Scheduler %s", engine_id)
        self.engine_id = engine_id
        assert (parallel_config := vllm_config.parallel_config)
        dp_rank = parallel_config.data_parallel_index
        dp_local_rank = parallel_config.data_parallel_rank_local
        self.dp_rank = dp_local_rank if parallel_config.local_engines_only else dp_rank
        self._encoded_xfer_handshake_metadata: dict[int, Any] = {}
        self.hostname = get_ip()
        self.async_zmq_ctx = zmq.asyncio.Context()

        # Requests that need to start recv/send.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        self._reqs_need_recv: dict[ReqId, tuple[Request, list[int]]] = {}
        self._reqs_need_send: dict[ReqId, tuple[Request, list[int]]] = {}
        # Reqs to remove from processed set because they're not to send after
        # remote prefill or aborted.
        self._reqs_not_processed: set[TransferId] = set()
        self._handle_request_executor: ThreadPoolExecutor | None = None
        self._handle_request_futures: set[Future[None]] = set()
        if self.is_kv_producer and self.use_layerwise:
            self.remote_ready_req = dict() # {transfer_id: (block_ids)}
            self.remote_new_agent = dict() # {engine_id : agent_meta}
            self.xfer_metadata_queue = asyncio.Queue[bytes]()
            self.sender_loop = asyncio.new_event_loop()
            # Background thread for processing new sending requests.
            self._sender_listener_t = threading.Thread(
                target=_async_loop, args=(self.sender_loop,), daemon=True
            )
            self._sender_listener_t.start()
            ready_event = threading.Event()
            asyncio.run_coroutine_threadsafe(
                self._mooncake_enginecore_listener(ready_event), self.sender_loop
            )
            ready_event.wait()  # Wait for listener ZMQ socket to be ready.
            self.remote_engine_ids = set()
            self.xfermetadata_decoder = msgspec.msgpack.Decoder(MooncakeLayerwiseXferMetadata)
        if self.is_kv_consumer and self.use_layerwise:
            max_workers = (
                self.vllm_config.kv_transfer_config.kv_connector_extra_config.get(
                    "handle_request_max_workers",
                    self._DEFAULT_HANDLE_REQUEST_MAX_WORKERS,
                )
            )
            self._handle_request_executor = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="vllm-mooncake-handle-request",
            )
        self.remote_engine_addr = dict()
    
    def __del__(self):
        self.shutdown()

    def shutdown(self):
        executor = getattr(self, "_handle_request_executor", None)
        if executor is not None:
            executor.shutdown(wait=False)
            self._handle_request_executor = None
        pending_futures = getattr(self, "_handle_request_futures", None)
        if pending_futures is not None:
            pending_futures.clear()

    def _run_handle_request_task(
        self,
        request_id: str,
        remote_bootstrap_addr: str,
        params: dict[str, Any],
        block_ids: list[int],
    ) -> None:
        asyncio.run(
            self.handle_request(request_id, remote_bootstrap_addr, params, block_ids)
        )

    def _log_handle_request_result(self, future: Future[None]) -> None:
        try:
            future.result()
        except Exception:
            logger.exception("Mooncake handle_request task failed.")

    def _submit_handle_request(
        self,
        request_id: str,
        remote_bootstrap_addr: str,
        params: dict[str, Any],
        block_ids: list[int],
    ) -> None:
        if self._handle_request_executor is None:
            raise RuntimeError(
                "Mooncake handle_request executor is not initialized for "
                "layerwise kv_consumer scheduler."
            )

        future = self._handle_request_executor.submit(
            self._run_handle_request_task,
            request_id,
            remote_bootstrap_addr,
            dict(params),
            list(block_ids),
        )
        self._handle_request_futures.add(future)
        future.add_done_callback(self._handle_request_futures.discard)
        future.add_done_callback(self._log_handle_request_result)

    async def _mooncake_enginecore_listener(self, ready_event: threading.Event):
        """
        Background thread that listens for Mooncake requests, dispatches them
        to a thread pool, and sends acknowledgments upon completion.
        """
        sock = self.async_zmq_ctx.socket(zmq.ROUTER)
        self.side_channel_port = sock.bind_to_random_port(f"tcp://{self.hostname}")
        
        logger.info(
            f"Mooncake engine sender starting listening on path: tcp://{self.hostname}:{self.side_channel_port}")
        await self.register_engine_with_bootstrap()
        ready_event.set()

        try:
            while True:
                identity, metadata_bytes = await sock.recv_multipart()
                sock.send_multipart((identity, b"", b"ACK"), flags=zmq.NOBLOCK)
                logger.debug(f"MooncakeConnector _mooncake_enginecore_listener p node get {self.xfermetadata_decoder.decode(metadata_bytes)}")
                await self.xfer_metadata_queue.put(metadata_bytes)

        except zmq.ContextTerminated:
            logger.error("ZMQ context terminated, exiting Mooncake sender thread.")
        except Exception as e:
            logger.error("Error in Mooncake sender thread: %s. Exiting thread.", str(e))
        finally:
            sock.close()

    async def register_engine_with_bootstrap(self):
        host, port = get_mooncake_bootstrap_addr(self.vllm_config)
        url = make_zmq_path("http", host, port) + "/register_engine"
        worker_addr = make_zmq_path("tcp", self.hostname, self.side_channel_port)
        payload = RegisterEngineCorePayload(
            engine_id=self.engine_id,
            dp_rank=self.dp_rank,
            addr=worker_addr,
        )

        while True:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(url, json=payload.model_dump())
                    response.raise_for_status()
                break
            except httpx.ConnectError:
                # Bootstrap server not ready, wait for a while and retry.
                await asyncio.sleep(1)
            except Exception as e:
                err_msg = (
                    e.response.text if isinstance(e, httpx.HTTPStatusError) else str(e)
                )
                logger.error(
                    "Error registering %s with engine bootstrap server: %s", payload, err_msg
                )
                raise e
        logger.info(f"Successfully registered with engine bootstrap server at {url}")

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
        
        if params.get("do_remote_decode") and self.use_layerwise:
            while not self.xfer_metadata_queue.empty():
                xfer_metadata = self.xfer_metadata_queue.get_nowait()
                xfer_metadata = self.xfermetadata_decoder.decode(xfer_metadata)
                for req_id, (engine_id, transfer_id, block_ids) in xfer_metadata.req_blocks.items():
                    logger.debug(f"MooncakeConnector get_num_new_matched_tokens p node xfer_metadata_queue get req_id={req_id} transfer_id={transfer_id}")
                    self.remote_ready_req[transfer_id] = (req_id, engine_id, block_ids)

                if xfer_metadata.agent_info is not None:
                    for engine_id, agent in xfer_metadata.agent_info.items():
                        logger.debug(f"MooncakeConnector get_num_new_matched_tokens p node get remote agent info: {engine_id}: {agent}")
                        self.remote_new_agent[engine_id] = agent

        if params.get("do_remote_prefill"):
            # Remote prefill: get all prompt blocks from remote.
            assert not self.is_kv_producer
            token_ids = request.prompt_token_ids or []
            count = len(token_ids) - num_computed_tokens
            if count > 0:
                return count, True
        if self.use_layerwise:
            transfer_id = params.get("transfer_id")
            if transfer_id in self.remote_ready_req:
                logger.info(f"MooncakeConnector get_num_new_matched_tokens p node return (0, False) for layerwise (remote is ready), req_id={request.request_id} transfer_id={transfer_id}")
                return 0, False
            return None, False
        else:
            return 0, False

    async def handle_request(
        self, request_id: str, remote_bootstrap_addr: str, params: dict, block_ids: list[int]
    ):
        """协程：负责与远程 P 节点进行控制面交互"""
        remote_engine_id = params.get("remote_engine_id")
        transfer_id = params.get("transfer_id")

        if not remote_bootstrap_addr:
            logger.error("remote_bootstrap_addr is None, cannot handle remote prefill.")
            return

        remote_engine_addr = self.remote_engine_addr.get(remote_engine_id)

        # ---------- 步骤1：查询 remote P 的 bootstrap server ----------
        if remote_engine_addr is None:
            agent_info = self.agent_info
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{remote_bootstrap_addr}/query", timeout=aiohttp.ClientTimeout(total=5)
                    ) as resp:
                        data = await resp.json()
            except Exception as e:
                logger.error(f"Bootstrap query failed: {e}")
                return

            # 查找匹配的 engine_id
            for dp_entry in data.values():
                if dp_entry.get("engine_id") == remote_engine_id:
                    remote_engine_addr = dp_entry.get("engine_addr")
                    break
        else:
            agent_info = None

        if not remote_engine_addr:
            logger.error(f"No worker address found for engine_id {remote_engine_id}")
            return

        # ---------- 步骤2：推送 metadata 到远程 P 节点 ----------
        push_data = MooncakeLayerwiseXferMetadata(
            agent_info=agent_info,
            req_blocks={request_id: (self.engine_id, transfer_id, block_ids)}
        )
        encoder = msgspec.msgpack.Encoder()
        encoded_data = encoder.encode(push_data)

        # 使用 zmq.asyncio 异步发送接收
        ctx = zmq.asyncio.Context.instance()
        sock = ctx.socket(zmq.DEALER)
        sock.connect(remote_engine_addr)
        try:
            await sock.send_multipart([encoded_data])
            await sock.recv()
        except Exception as e:
            logger.error(f"MooncakeXferMetadata transfer failed: {e}")
        finally:
            sock.close()

        # 缓存地址（只有第一次查询后才需要缓存）
        if remote_engine_id not in self.remote_engine_addr:
            self.remote_engine_addr[remote_engine_id] = remote_engine_addr
 
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
            if self.use_layerwise:
                params["do_remote_prefill"] = False
                # layerwise TODO 4. decoder engine core send [req, block_ids, kv_addr, worker_addr] to prefiller engine core
                # 1.查询 remote P的boost server，获取P节点的worker addr
                # 2.推送自己的agent metadata (worker_addr、kv_addr、engine_id)和 请求信息给（request_id、remote block ids)给P节点的worker addr
                remote_bootstrap_addr = params.get("remote_bootstrap_addr", None)
                # 把任务加入事件循环
                # asyncio.create_task(
                #     self.handle_request(request.request_id, remote_bootstrap_addr, params)  # 如果remote_bootstrap_addr对应的engine_id是第一次遇到，就query；并发送transfer_info
                # )
                self._submit_handle_request(
                    request.request_id,
                    remote_bootstrap_addr,
                    params,
                    local_block_ids,
                )

        elif params.get("do_remote_decode"):
            assert not self.is_kv_consumer
            if not params.get("transfer_id"):
                logger.warning("Missing transfer_id in kv_transfer_params from router!")
            else:
                # Add an empty list to worker to create event.
                block_ids = blocks.get_block_ids()[0] if self.use_layerwise else []
                self._reqs_need_send[request.request_id] = (request, block_ids)

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
            cached_reqs = scheduler_output.scheduled_cached_reqs
            for req_id, new_blocks in zip(cached_reqs.req_ids, cached_reqs.new_block_ids):
                if req_id in self._reqs_need_send and new_blocks is not None:
                    self._reqs_need_send[req_id][1].extend(new_blocks[0])
            
            to_remove = []
            for req_id, (req, block_ids) in self._reqs_need_send.items():
                if (scheduler_output.num_scheduled_tokens[req_id] + req.num_computed_tokens) < len(req.all_token_ids):
                    logger.debug(f"MooncakeConnector build_connector_meta skip {req_id}: (num_scheduled_tokens+num_computed_tokens)<prompt_len {scheduler_output.num_scheduled_tokens[req_id]} + {req.num_computed_tokens} < {len(req.all_token_ids)}")
                    continue
                assert req.kv_transfer_params is not None
                transfer_id = req.kv_transfer_params["transfer_id"]
                d_req_id, remote_engine_id, remote_block_ids = self.remote_ready_req.pop(transfer_id)

                agent_info = self.remote_new_agent.pop(remote_engine_id, None)
                agent_info = None if agent_info is None else {remote_engine_id : agent_info}
                meta.add_new_req(
                    request_id=req_id,
                    local_block_ids=block_ids,
                    kv_transfer_params=req.kv_transfer_params,
                    load_remote_cache=False,
                    agent_info=agent_info,
                    use_layerwise=self.use_layerwise,
                    remote_block_ids=remote_block_ids,
                    remote_engine_id=remote_engine_id,
                    d_req_id=d_req_id
                )
                to_remove.append(req_id)
            if len(meta.send_task.send_request.keys()) > 0:
                logger.debug(f"MooncakeConnector build_connector_meta p node build_connector_meta {meta}")
            for req_id in to_remove:
                self._reqs_need_send.pop(req_id, None)
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
        if self.use_layerwise:
            delay_free_blocks = False

        if delay_free_blocks:
            self._reqs_need_send[request.request_id] = (request, block_ids)

        return delay_free_blocks, None

    def set_xfer_handshake_metadata(
        self, metadata: dict[int, KVConnectorHandshakeMetadata]
    ) -> None:
        """
        Set the KV connector handshake metadata for this connector.

        Args:
            metadata (dict): the handshake metadata to set.
        """
        encoded_data: dict[int, bytes] = {}
        encoder = msgspec.msgpack.Encoder()
        for tp_rank, rank_metadata in metadata.items():
            if not isinstance(rank_metadata, MooncakeHandshakePayload):
                raise ValueError(
                    "MooncakeConnectorScheduler expects MooncakeHandshakePayload for "
                    "handshake metadata."
                )
            encoded_data[tp_rank] = encoder.encode(rank_metadata)
            logger.debug(
                "Tp rank %d: encoded MooncakeHandshakePayload size: %s bytes",
                tp_rank,
                str(len(encoded_data[tp_rank])),
            )
        self._encoded_xfer_handshake_metadata = encoded_data

        # generate self.agent_info
        import copy
        encoded_data_copy = copy.deepcopy(encoded_data)
        for i in range(len(encoded_data_copy)):
            handshake_decoder = msgspec.msgpack.Decoder(MooncakeHandshakePayload)
            handshake_payload = handshake_decoder.decode(encoded_data[i])
            metadata_decoder = msgspec.msgpack.Decoder(MooncakeAgentMetadata)
            metadata = metadata_decoder.decode(handshake_payload.agent_metadata_bytes)

            agent_mata = (metadata.worker_addr, metadata.rpc_addr, metadata.kv_caches_base_addr)
            encoded_data_copy[i] = agent_mata
        self.agent_info = {self.engine_id: encoded_data_copy}


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
        self.use_layerwise = vllm_config.kv_transfer_config.kv_connector_extra_config.get("use_layerwise", False)
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
                _, port = get_mooncake_bootstrap_addr(vllm_config)
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
        self.layerwise_send_queue = asyncio.Queue[SendTask]()
        self.remote_agent_meta = dict() # {engine_id : agent_meta}
        self.total_layers = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        connector_metadata: MooncakeConnectorMetadata,
        **kwargs,
    ) -> None:
        """MooncakeConnector does not save explicitly."""
        if self.vllm_config.kv_transfer_config.is_kv_producer and connector_metadata.send_task.send_request.keys():
            reshape_cache_event = attn_metadata.reshape_cache_event

            keys = None
            values = None

            assert self.layerwise_send_queue is not None
            assert reshape_cache_event is not None
            import copy
            layer_send_task = copy.deepcopy(connector_metadata.send_task)

            layer_send_task.wait_event = reshape_cache_event
            layer_send_task.k_cache = keys
            layer_send_task.v_cache = values
            layer_send_task.layer_idx = self.current_layer

            logger.debug(f"MooncakeConnector save_kv_layer {self.current_layer} put : {layer_send_task.send_request.keys()} to layerwise_send_queue")
            self.sender_loop.call_soon_threadsafe(
                self.layerwise_send_queue.put_nowait, layer_send_task
            )
            self.current_layer += 1
    
    async def _mooncake_recv_listener(self, ready_event: threading.Event):
        """
        Background thread that listens for Mooncake requests, dispatches them
        to a thread pool, and sends acknowledgments upon completion.
        """
        sock = self.async_zmq_ctx.socket(zmq.ROUTER)
        self.side_channel_port = sock.bind_to_random_port(f"tcp://{self.hostname}")
        logger.info(
            f"Mooncake worker receiver starting listening on path: tcp://{self.hostname}:{self.side_channel_port}")
        ready_event.set()
        decoder = msgspec.msgpack.Decoder(type=tuple)

        while True:
            try:
                frames = await sock.recv_multipart()
                if len(frames) < 2:
                    logger.error("Invalid message format: %s", frames)
                    continue

                identity = frames[0]
                payload = [f for f in frames[1:] if f != b""]
                if len(payload) != 1:
                    logger.error("Invalid message format: %s", frames)
                    continue

                msg = decoder.decode(payload[0])
                if msg[0] == DONE_SENDING_MSG:
                    req_id = msg[1]
                    logger.debug(f"MooncakeConnector _mooncake_recv_listener get req_id={req_id}")
                    self.finished_recving_reqs.add(req_id)
                    sock.send_multipart((identity, b"", b"ACK"))
                else:
                    logger.error("Connection listener got unexpected message %s", msg)

            except Exception as e:
                logger.error("Failed to decode message: %s", e)

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
        host, port = get_mooncake_bootstrap_addr(self.vllm_config)
        url = make_zmq_path("http", host, port) + "/register_worker"
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

    async def _mooncake_layerwise_sender(self):
        """
        Background thread that listens for Mooncake requests, dispatches them
        to a thread pool, and sends acknowledgments upon completion.
        """
        while True:
            send_task = await self.layerwise_send_queue.get()
            logger.debug(f"MooncakeConnector _mooncake_layerwise_sender get {send_task.send_request.keys()}")
            self._handle_request(send_task)

    def _handle_request(self, send_task: SendTask):
        try:
            logger.debug(f"Starting to transfer KV cache for request {send_task.send_request.keys()}.")
            self._transfer_kv_cache(send_task)
            logger.debug(f"Starting to transfer KV cache for request {send_task.send_request.keys()}.")
        except Exception as e:
            logger.error(f"Error in _handle_request: {e}", exc_info=True)

    def get_transfer_meta(self, send_task: SendTask, req_id: str, req_meta: PushReqMeta, remote_kv_base_addrs: list[int]):
        src_list: list[str] = []
        dst_list: list[str] = []
        length_list: list[int] = []

        layer_idx = send_task.layer_idx
        remote_block_ids = req_meta.remote_block_ids
        local_kv_base_addr = self.kv_caches_base_addr
        local_block_ids = req_meta.local_block_ids


        layer_local_kv_base_addr = [local_kv_base_addr[i] for i in [2 * layer_idx, 2 * layer_idx + 1]]
        layer_remote_kv_base_addr = [
            remote_kv_base_addrs[i]  # type:ignore
            for i in [2 * layer_idx, 2 * layer_idx + 1]
        ]
        grouped_remote_block_ids, grouped_local_block_ids = group_concurrent_contiguous(
            remote_block_ids, local_block_ids
        )

        for k, (src_layer_base_addr, dst_layer_base_addr) in enumerate(
            zip(layer_local_kv_base_addr, layer_remote_kv_base_addr)
        ):
            block_len = self.block_len
            for group_remote_block_id, group_local_block_id in zip(
                grouped_remote_block_ids, grouped_local_block_ids
            ):
                src = src_layer_base_addr + group_local_block_id[0] * block_len
                dst = dst_layer_base_addr + group_remote_block_id[0] * block_len
                length = len(group_local_block_id) * block_len
                src_list.append(src)
                dst_list.append(dst)
                length_list.append(length)
    
        return (src_list, dst_list, length_list)

    def _transfer_kv_cache(self, send_task: SendTask):
        # Merge transmission tasks of the same session
        session_meta: dict[str, TransferMeta] = {}
        for req_id, req_meta in send_task.send_request.items():
            # assert req_meta.remote_engine_id in self.remote_agent_meta , f"[===] _transfer_kv_cache {req_meta.remote_engine_id} not in {self.remote_agent_meta.keys()}"
            # assert self.tp_rank in self.remote_agent_meta[req_meta.remote_engine_id] , f"[===] _transfer_kv_cache {self.tp_rank} not in {self.remote_agent_meta[req_meta.remote_engine_id]}"
            remote_worker_addr, session_id, remote_kv_base_addrs = self.remote_agent_meta[req_meta.remote_engine_id][self.tp_rank]
            if session_id not in session_meta:
                session_meta[session_id] = TransferMeta(src=[], dst=[], length=[], req_ids=[], remote_worker_addr=None)

            (src_list, dst_list, length_list) = self.get_transfer_meta(send_task, req_id, req_meta, remote_kv_base_addrs)

            session_meta[session_id].src.extend(src_list)
            session_meta[session_id].dst.extend(dst_list)
            session_meta[session_id].length.extend(length_list)
            session_meta[session_id].req_ids.append(req_id)
            session_meta[session_id].remote_worker_addr = remote_worker_addr
        """
        Note: Due to a bug in ADXL, calling current_event.synchronize() may occasionally hang.
        This issue will be fixed in CANN version 8.5.rc1.
        You can manually build the master branch of the project at https://gitcode.com/cann/hixl
        to resolve this issue before the 8.5.RC1 release.
        """
        send_task.wait_event.synchronize()  # type:ignore
        for session_id, transfer_meta in session_meta.items():
            
            if len(transfer_meta.src) > 0:
                ret = self.engine.batch_transfer_sync_write(
                    session_id, transfer_meta.src, transfer_meta.dst, transfer_meta.length
                )
                if ret < 0:
                    logger.error(
                        f"Mooncake transfer failed for send requests {transfer_meta.req_ids} kv cache to {session_id}"
                    )
                    if send_task.layer_idx == (self.total_layers - 1):
                        for req_id in transfer_meta.req_ids:
                            d_req_id = send_task.send_request[req_id].d_req_id
                            self.send_done_send_signal(
                                d_req_id, transfer_meta.remote_worker_addr
                            )  # TODO Send a signal indicating transmission failure
                else:
                    if send_task.layer_idx == (self.total_layers - 1):
                        for req_id in transfer_meta.req_ids:
                            d_req_id = send_task.send_request[req_id].d_req_id
                            self.send_done_send_signal(d_req_id, transfer_meta.remote_worker_addr, layer_idx=send_task.layer_idx, total_layers=self.total_layers)

    def send_done_send_signal(self, req_id, remote_worker_addr, layer_idx=None, total_layers=None):
        logger.info(
            f"Sending done sending signal for request {req_id} to {remote_worker_addr}, {layer_idx=} {total_layers=}"
        )
        try:
            path = remote_worker_addr
            msg_encoder = msgspec.msgpack.Encoder()
            encoded_data = msg_encoder.encode((DONE_SENDING_MSG, req_id))
            with zmq_ctx(zmq.REQ, path) as sock:  # type: ignore
                ensure_zmq_send(sock, encoded_data, path)
                ack = sock.recv()
                if ack != b"ACK":
                    raise ValueError(f"Unexpected ACK response: {ack}")
        except Exception as e:
            logger.error(
                f"Sending done sending signal for request {req_id} to "
                f"{remote_worker_addr} fail with error: {e}"
            )

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
                logger.warning(
                    "Timeout waiting for P side ready: %s", list(pending_reqs)
                )
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
                else:
                    # Otherwise (expired, very unlikely), just forget it.
                    logger.warning(
                        "Request %s expired before sending on P side.", d_req_id
                    )

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
                # TODO: for heterogeneous TP (one P pairs to multiple D),
                # we need to check whether all headers are sent.
                # If not, we should set expire_time to normal and skip the below.
                send_meta.sending -= 1
                send_meta.sent += 1
                if send_meta.sent == send_meta.need_send:
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
            raise NotImplementedError(
                "Mooncake: Heterogeneous TP is not supported yet."
            )

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
            logger.info(
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
            try:
                shape_info = cache_or_caches.shape
            except AttributeError:
                if isinstance(cache_or_caches, (tuple, list)):
                    shape_info = [getattr(c, 'shape', 'unknown') for c in cache_or_caches]
                else:
                    shape_info = str(type(cache_or_caches))

            logger.debug("registering layer %s with shape %s", layer_name, shape_info)
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
        if self.use_layerwise:
            if self.is_kv_producer:
                # TODO add push thread
                asyncio.run_coroutine_threadsafe(
                    self._mooncake_layerwise_sender(), self.sender_loop
                )
            elif self.is_kv_consumer:
                # TODO add recv done single thread
                ready_event = threading.Event()
                asyncio.run_coroutine_threadsafe(
                    self._mooncake_recv_listener(ready_event), self.receiver_loop
                )
                ready_event.wait()  # Wait for listener ZMQ socket to be ready.
        else:
            if self.is_kv_producer:
                ready_event = threading.Event()
                asyncio.run_coroutine_threadsafe(
                    self._mooncake_sender_listener(ready_event), self.sender_loop
                )
                ready_event.wait()  # Wait for listener ZMQ socket to be ready.
        # After KV Caches registered, listen for new connections.
        agent_metadata = MooncakeAgentMetadata(
            engine_id=self.engine_id,
            worker_addr=make_zmq_path("tcp", self.hostname, self.side_channel_port),
            kv_caches_base_addr=self.kv_caches_base_addr,
            rpc_addr= str(self.hostname) + ":" + str(self.rpc_port),
            dp_size=self.vllm_config.parallel_config.data_parallel_size,
            tp_size=self.vllm_config.parallel_config.tensor_parallel_size,
            pcp_size=self.vllm_config.parallel_config.prefill_context_parallel_size,
            dcp_size=self.vllm_config.parallel_config.decode_context_parallel_size,
            dp_rank=self.vllm_config.parallel_config.data_parallel_index,
            tp_rank=get_tensor_model_parallel_rank(),
            pcp_rank=get_pcp_group().rank_in_group if get_pcp_group().world_size > 1 else 0,
            dcp_rank=get_decode_context_model_parallel_rank() if self.vllm_config.parallel_config.decode_context_parallel_size > 1 else 0,
        )
        # Wrap metadata in payload with hash for defensive decoding
        encoder = msgspec.msgpack.Encoder()
        self.xfer_handshake_metadata = MooncakeHandshakePayload(
            agent_metadata_bytes=encoder.encode(agent_metadata),
        )

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
                # If something goes wrong, let P wait timeout first (in asyncio.wait()).
                sock.setsockopt(
                    zmq.RCVTIMEO, (envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT + 60) * 1000
                )
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
            raise NotImplementedError(
                "Mooncake: Heterogeneous TP is not supported yet."
            )
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
            else:
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
        # layerwise TODO 7.2 if not use layerwise prefiller worker add send task
        if not self.use_layerwise:
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
        else:
            if self.is_kv_producer:
                if len(metadata.agent_info.keys()) > 0:
                    logger.info(f"MooncakeConnector start_load_kv update {self.remote_agent_meta}")
                    self.remote_agent_meta.update(metadata.agent_info)
                self.current_layer = 0


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

import  contextlib
@contextlib.contextmanager
def zmq_ctx(socket_type: Any, addr: str):
    """Context manager for a ZMQ socket"""

    if socket_type not in (zmq.ROUTER, zmq.REQ, zmq.DEALER):  # type: ignore
        raise ValueError(f"Unexpected socket type: {socket_type}")

    ctx: zmq.Context | None = None  # type: ignore
    try:
        ctx = zmq.Context()  # type: ignore
        yield make_zmq_socket(ctx=ctx, path=addr, socket_type=socket_type, bind=socket_type == zmq.ROUTER)  # type: ignore
    finally:
        if ctx is not None:
            ctx.destroy(linger=0)

def ensure_zmq_send(
    socket: zmq.Socket,  # type: ignore
    data: bytes,
    path: str,
    max_retries: int = 3,
):
    retries_left = max_retries
    while True:
        try:
            socket.send(data)
            return
        except zmq.ZMQError as e:  # type: ignore
            retries_left -= 1
            if retries_left > 0:
                logger.warning(f"Send failed: {e}, retrying... ({retries_left} attempts left)")
                time.sleep(0.1)
            else:
                logger.error(f"Send failed after all retries: {e}")
                raise RuntimeError(f"Failed to send data to {path} after {max_retries} retries: {e}")


def ensure_zmq_recv(
    socket: zmq.Socket,  # type: ignore
    poller: zmq.Poller,  # type: ignore
    path: str,
    timeout: float = 1.0,
    max_retries: int = 3,
) -> bytes:
    retries_left = max_retries
    while True:
        try:
            if dict(poller.poll(int(timeout * 1000))):  # milliseconds
                data = socket.recv()
                return data
            else:
                raise zmq.ZMQError("Receive timeout")  # type: ignore
        except zmq.ZMQError as e:  # type: ignore
            retries_left -= 1
            if retries_left > 0:
                logger.warning(f"Receive failed: {e}, retrying... ({retries_left} attempts left)")
                time.sleep(0.1)
            else:
                logger.error(f"Receive failed after all retries: {e}")
                raise RuntimeError(f"Failed to receive data from {path} after {max_retries} retries: {e}")
