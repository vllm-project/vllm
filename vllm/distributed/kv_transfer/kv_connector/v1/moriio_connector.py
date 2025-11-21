# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import logging
import math
import os
import queue
import threading
import time
from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional
from weakref import ref as weakref_ref

import msgpack
import msgspec
import numpy as np
import torch
import zmq

from vllm import envs
from vllm.attention.backends.registry import AttentionBackendEnum
from vllm.attention.selector import get_attn_backend
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    get_world_group,
)
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.utils.network_utils import get_ip, make_zmq_path, make_zmq_socket, get_open_port
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

from dataclasses import field
from enum import Enum
from queue import Empty, Queue

logger = init_logger(__name__)

Transfer = tuple[int, float]
EngineId = str
ReqId = str


class MoRIIOConstants:
    """Constants for MoRIIO connector."""

    # ZMQ message types
    GET_META_MSG = b"get_meta_msg"
    POP_DONE_RECV = b"pop_done_recv"
    OVER = b"OVER"
    COMPLETION_PREFIX = "cmpl"

    PING_INTERVAL = 5
    MAX_PING_RETRIES = 100000
    DEFAULT_HANDSHAKE_PORT = "6301"
    DEFAULT_NOTIFY_PORT="61005"

try:
    from mori.io import (
        BackendType,
        EngineDesc,
        IOEngine,
        IOEngineConfig,
        MemoryDesc,
        PollCqMode,
        RdmaBackendConfig
    )

    logger.info("MoRIIO is available")
    MoRIIO_enabled = True
except ImportError:
    logger.error("MoRIIO is not available")
    MoRIIO_enabled = False


@dataclass
class WriteTask:
    request_id: str
    dst_engine_id: str
    local_block_ids: list[int]
    remote_block_ids_hint: list[int] | None
    layer_name: str
    event: torch.cuda.Event
    remote_notify_port: int
    remote_ip: str
    enqueue_time: float = field(default_factory=time.perf_counter)
    retried: int = 0


@dataclass
class LayerTransferPlan:
    """Plan for transferring a single layer."""

    request_id: str
    layer_name: str
    sess_idx: int
    transfer_local_offsets: list[int]
    transfer_remote_offsets: list[int]
    transfer_sizes: list[int]
    use_batch: bool = True


@dataclass
class RemoteAllocInfo:
    """Information about remote block allocation."""

    block_ids: list[int]
    writes_done: int = 0
    decode_dp_rank: int = 0
    transfer_offset: tuple[list[int], list[int], list[int]] | None = None


class ROLE(Enum):
    PRODUCER = "producer"
    CONSUMER = "consumer"
    NOTINIT = "notinit"


class RoleManager:
    """Manages role state across the connector."""

    _instance: Optional["RoleManager"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._role: ROLE = ROLE.NOTINIT

    @classmethod
    def get_instance(cls) -> "RoleManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def set_role(self, role: ROLE) -> None:
        """Set the current role."""
        with self._lock:
            self._role = role

    def get_role(self) -> ROLE:
        """Get the current role."""
        return self._role


def set_role(role: ROLE):
    """Set the global role."""
    RoleManager.get_instance().set_role(role)


def get_role() -> ROLE:
    """Get the global role."""
    return RoleManager.get_instance().get_role()


class MoRIIOMode(Enum):
    READ = "read"
    WRITE = "write"


class MoRIIOError(Exception):
    """Base exception for MoRIIO operations."""

    pass


class HandshakeError(MoRIIOError):
    """Exception raised when handshake fails."""

    pass


class TransferError(MoRIIOError):
    """Exception raised when transfer fails."""

    pass


def get_moriio_mode() -> MoRIIOMode:
    read_mode = os.environ.get("MORIIO_CONNECTOR_READ_MODE", "false").lower()
    logger.debug("MoRIIO Connector read_mode: %s", read_mode)
    if read_mode in ("true", "1", "yes", "on"):
        return MoRIIOMode.READ
    else:
        return MoRIIOMode.WRITE
    


def get_port_offset(dp_rank: int, tp_rank: int, tp_size: int = 1) -> int:
    return (dp_rank) * tp_size + tp_rank


@dataclass
class MoRIIOConfig:
    local_ip: str
    local_kv_port: int
    proxy_ip: str
    local_ping_port: int
    proxy_ping_port: int
    http_port: int
    handshake_port: int
    notify_port: int
    tp_rank: int
    dp_rank: int
    dp_size: int
    tp_size: int

    @classmethod
    def from_vllm_config(cls, vllm_config: VllmConfig) -> "MoRIIOConfig":
        
        # Port Configuration:
        # local_ping_port   -> Outgoing heartbeat to proxy
        # proxy_ping_port   -> Remote proxy's heartbeat ingress port
        # http_port         -> Instance's HTTP service endpoint
        # local_kv_port     -> service port for mori engine
        # notify_port       -> For synchronizing stages between prefill and decode
        # handshake_port    -> For initial handshake between mori engine

        #TODO : merge notify_port and handshake_port to simplify port management, supports non-contiguous ports 
        
        kv_transfer_config = vllm_config.kv_transfer_config
        extra_config = kv_transfer_config.kv_connector_extra_config
        tp_rank = get_tensor_model_parallel_rank()
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        base_notify_port = int(extra_config["notify_port"])
        dp_size = vllm_config.parallel_config.data_parallel_size
        tp_size = get_tensor_model_parallel_world_size()
        port_offset = get_port_offset(dp_rank, tp_rank)

        return cls(
            local_ip=get_ip(),
            local_kv_port=get_open_port(),
            proxy_ip=extra_config["proxy_ip"],
            local_ping_port=get_open_port(),
            proxy_ping_port=int(extra_config["proxy_ping_port"]),
            http_port=int(extra_config["http_port"]),
            handshake_port=int(extra_config["handshake_port"]),
            notify_port=base_notify_port + port_offset,
            tp_rank=tp_rank,
            dp_rank=dp_rank,
            dp_size=dp_size,
            tp_size=tp_size,
        )


"""Write task execution logic for MoRIIO connector."""


class MoRIIOWriter:
    """Handles write operations for KV cache transfers.
    Implements distributed KV cache transfer using the MoRIIO library
    for RDMA-based communication between prefill and decode instances."""

    def __init__(self, worker: "MoRIIOConnectorWorker"):
        """Initialize the writer.

        Args:
            worker: Reference to the parent worker
        """
        # self.worker = worker
        self._worker_ref: weakref_ref[MoRIIOConnectorWorker] = weakref_ref(worker)
        self._write_task_q: Queue[WriteTask] = Queue()
        self._write_worker_started = False
        self._write_worker_lock = threading.Lock()
        self._deferred_tasks: list[WriteTask] = []

    @property
    def worker(self) -> "MoRIIOConnectorWorker":
        """Get the worker instance.

        Returns:
            The parent worker instance

        Raises:
            RuntimeError: If worker has been garbage collected
        """
        worker = self._worker_ref()
        if worker is None:
            raise RuntimeError("Parent worker has been garbage collected")
        return worker

    def ensure_worker_started(self) -> None:
        """Ensure the background write worker is running."""
        if self._write_worker_started:
            return
        self._write_worker_started = True
        with self._write_worker_lock:
            thread = threading.Thread(
                target=self._write_worker_loop, daemon=True, name="moriio-write-worker"
            )
            thread.start()
            logger.info("Started MoRIIO write worker thread")

    def schedule_write(self, task: WriteTask) -> None:
        """Schedule a write task.

        Args:
            task: The write task to schedule
        """
        self.ensure_worker_started()
        self._write_task_q.put(task)

    def _write_worker_loop(self) -> None:
        """Main loop for the write worker thread."""

        while True:
            # Process deferred tasks first
            self._process_deferred_tasks()

            # Get new task
            try:
                task = self._write_task_q.get(timeout=0.01)
            except Empty:
                continue

            # Check if remote blocks are ready
            if not self._is_remote_ready(task):
                # task.retry_count += 1
                self._deferred_tasks.append(task)
                # logger.debug(
                #     "Deferred task for request %s (retry %d)",
                #     task.request_id, task.retry_count
                # )
                continue

            # Execute the task

            self._execute_write_task(task)

    def _process_deferred_tasks(self) -> None:
        """Process tasks that were previously deferred."""
        if not self._deferred_tasks:
            return

        still_deferred: list[WriteTask] = []
        for task in self._deferred_tasks:
            if self._is_remote_ready(task):
                self._execute_write_task(task)
            else:
                still_deferred.append(task)

        self._deferred_tasks = still_deferred

    def _is_remote_ready(self, task: WriteTask) -> bool:
        """Check if remote blocks are allocated for this task.

        Args:
            task: The write task

        Returns:
            True if remote blocks are ready
        """
        return (
            task.request_id in self.worker.moriio_wrapper.done_remote_allocate_req_dict
        )

    def _get_remote_alloc_info(self, request_id: str) -> RemoteAllocInfo:
        """Get remote allocation info for a request.

        Args:
            request_id: The request ID

        Returns:
            Remote allocation information

        Raises:
            KeyError: If allocation info is missing
        """
        try:
            return self.worker.moriio_wrapper.done_remote_allocate_req_dict[request_id]
        except KeyError as e:
            raise KeyError(
                f"Remote allocation info missing for request {request_id}"
            ) from e

    def _execute_write_task(self, task: WriteTask) -> None:
        """Execute a single write task.

        Args:
            task: The write task to execute

        """
        # Get remote allocation info
        request_info = self._get_remote_alloc_info(task.request_id)

        if request_info.block_ids is None:
            logger.debug("Request %s remote block IDs not ready", task.request_id)
            return

        # Wait for CUDA event
        # The attention computation of the current layer cannot
        # overlap with the kv transfer task,
        # otherwise it will cause precision issues.
        # This event is used to synchronize the kv transfer and computation tasks.
        task.event.synchronize()

        # Update engine ID with DP rank
        task.dst_engine_id = self.worker.get_engine_name_with_dp(
            task.dst_engine_id, request_info.decode_dp_rank
        )

        # Get or create sessions
        sessions = self.worker._get_built_session(task.dst_engine_id)

        # Prepare transfer plan
        plan = self._prepare_transfer_plan(task, request_info)

        # Execute transfer
        self._do_layer_write(plan, sessions)

        # Finalize if all layers complete
        self._finalize_if_complete(task, request_info)

    def _prepare_transfer_plan(
        self, task: WriteTask, request_info: RemoteAllocInfo
    ) -> LayerTransferPlan:
        """Prepare the transfer plan for a layer.

        Args:
            task: The write task
            request_info: Remote allocation information

        Returns:
            The transfer plan
        """
        # Compute offsets if not cached
        if request_info.transfer_offset is None:
            offsets = self.worker._compute_block_transfer_offsets(
                task.layer_name, task.local_block_ids, request_info.block_ids
            )
            request_info.transfer_offset = offsets

        # Get session index
        layer_names = list(self.worker.layer_name_to_local_kv_cache_metadata.keys())
        sess_idx = layer_names.index(task.layer_name)

        local_off, remote_off, sizes = request_info.transfer_offset

        return LayerTransferPlan(
            request_id=task.request_id,
            layer_name=task.layer_name,
            sess_idx=sess_idx,
            transfer_local_offsets=local_off,
            transfer_remote_offsets=remote_off,
            transfer_sizes=sizes,
            use_batch=True,
        )

    def _do_layer_write(self, plan: LayerTransferPlan, sessions: list) -> None:
        """Perform the actual layer write.

        Args:
            plan: The transfer plan
            sessions: List of transfer sessions
        """
        if plan.use_batch:
            self.worker.moriio_wrapper.write_remote_data(
                plan.transfer_sizes,
                plan.transfer_local_offsets,
                plan.transfer_remote_offsets,
                sessions[plan.sess_idx],
            )
        else:
            for i in range(len(plan.transfer_local_offsets)):
                self.worker.moriio_wrapper.write_remote_data_single(
                    plan.transfer_sizes[i],
                    plan.transfer_local_offsets[i],
                    plan.transfer_remote_offsets[i],
                    plan.sess_idx,
                )

    def _finalize_if_complete(
        self, task: WriteTask, request_info: RemoteAllocInfo
    ) -> None:
        """Finalize transfer if all layers are complete.

        Args:
            task: The write task
            request_info: Remote allocation information
        """
        request_info.writes_done += 1

        if request_info.writes_done >= self.worker.num_layers:
            # Wait for transfer to complete
            self.worker.moriio_wrapper.waiting_for_transfer_complete()

            remote_port = task.remote_notify_port + get_port_offset(
                request_info.decode_dp_rank, self.worker.tp_rank
            )
            # Consider using RDMA immediate data in decode side
            # to eliminate the need for this notification.
            # Consider including the first gen token from prefill in the notification

            # Send completion notification
            self.worker.moriio_wrapper.send_notify(
                task.request_id, task.remote_ip, remote_port
            )
            del self.worker.moriio_wrapper.done_remote_allocate_req_dict[
                task.request_id
            ]
            logger.debug(
                "Completed transfer for request %s, notified port %d",
                task.request_id,
                remote_port,
            )


class MoRIIOWrapper:
    """Wrapper for MoRIIO engine operations.

    Handles both producer and consumer roles for KV cache transfers.

    Args:
        moriio_engine:  MoRIIO engine instance
        tp_rank: Tensor parallel rank
        dp_rank: Data parallel rank
    """

    def __init__(self, moriio_engine=None, tp_rank=0, dp_rank=0):
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.moriio_engine = moriio_engine
        self.remote_memory_metadata = None
        self.local_memory_registered = False
        self.local_memory_metadata = None
        self.transfer_status = []
        self.remote_engine_ip = None
        self.notify_port = None
        self.notify_sock = None
        self.lock = threading.Lock()
        self.done_req_ids = []
        self.done_remote_allocate_req_dict: dict[str, RemoteAllocInfo] = {}
        self.done_write_cache_req_ids = []
        self.notify_thread = None
        self.sock = None
        self.sessions: list[IOEngine.Session] = []
        self.paths: dict[str, zmq.Socket] = {}

    def set_moriio_engine(self, moriio_engine):
        assert moriio_engine is not None, (
            "You Cannot pass None engine to MoRIIOWrapper!"
        )
        self.moriio_engine = moriio_engine

    def set_backend_type(self, backend_type):
        qp_per_transfer = int(os.getenv("VLLM_MORI_QP_PER_TRANSFER", "1"))
        post_batch_size = int(os.getenv("VLLM_MORI_POST_BATCH_SIZE", "-1"))
        num_worker_threads = int(os.getenv("VLLM_MORI_NUM_WORKERS", "1"))
        poll_mode = PollCqMode.POLLING
        rdma_cfg = RdmaBackendConfig(
            qp_per_transfer,
            post_batch_size,
            num_worker_threads,
            poll_mode,
        )
        self.moriio_engine.create_backend(backend_type, rdma_cfg)

    def get_agent_metadata(self):
        engine_metadata = self.moriio_engine.get_engine_desc()
        engine_metadata_packed = engine_metadata.pack()
        return engine_metadata_packed

    def register_remote_engine(self, remote_packed_engine_metadata):
        consumer_engine_metadata = EngineDesc.unpack(remote_packed_engine_metadata)
        self.moriio_engine.register_remote_engine(consumer_engine_metadata)
        return consumer_engine_metadata.key

    def register_local_tensor(self, tensor: torch.Tensor):
        try:
            self.local_memory_metadata = self.moriio_engine.register_torch_tensor(
                tensor
            )
            local_memory_metadata_packed = self.local_memory_metadata.pack()
        except Exception as e:
            raise MoRIIOError(f"Failed to register local memory: {e}") from e
        self.local_memory_registered = True
        return local_memory_metadata_packed

    def get_unpack_memory_metadata(self, packed_memory_metadata):
        return MemoryDesc.unpack(packed_memory_metadata)

    def build_session(self, local_memory_metadata, remote_memory_metadata):
        return self.moriio_engine.create_session(
            local_memory_metadata, remote_memory_metadata
        )

    def read_remote_data(
        self, transfer_size_byte, local_offset=0, remote_offset=0, session=None
    ):
        assert self.local_memory_registered, "You have not register local memory data!"

        transfer_status = session.batch_read(
            local_offset,
            remote_offset,
            transfer_size_byte,
            self.moriio_engine.allocate_transfer_uid(),
        )

        return transfer_status

    def write_remote_data(
        self, transfer_size_byte, local_offset=0, remote_offset=0, session=None
    ):
        assert self.local_memory_registered, "You have not register local memory data!"
        write_uid = self.moriio_engine.allocate_transfer_uid()

        transfer_status = session.batch_write(
            local_offset, remote_offset, transfer_size_byte, write_uid
        )
        with self.lock:
            self.transfer_status.append(transfer_status)

    def write_remote_data_single(
        self, transfer_size_byte, local_offset=0, remote_offset=0, sess_idx=0
    ):
        assert self.local_memory_registered, "You have not register local memory data!"

        transfer_status = self.sessions[sess_idx].write(
            local_offset,
            remote_offset,
            transfer_size_byte,
            self.moriio_engine.allocate_transfer_uid(),
        )
        with self.lock:
            self.transfer_status.append(transfer_status)

    def waiting_for_transfer_complete(self):
        if not self.transfer_status:
            return

        transfers_to_wait = []
        with self.lock:
            transfers_to_wait = self.transfer_status[:]
            self.transfer_status.clear()

        for status in transfers_to_wait:
            try:
                status.Wait()
                if not status.Succeeded():
                    logger.error(
                        "Transfer failed: %s, Code: %s", status.Message(), status.Code()
                    )
                    raise TransferError("MoRIIO transfer failed!")
            except Exception as e:
                logger.error("Transfer %s failed: %s", status, e)
                raise

    def async_wait_reqid(self):
        assert self.notify_port is not None, "Notify port cannot be None"

        if self.notify_thread is not None:
            return

        def _async_wait():
            host = "*"
            path = make_zmq_path("tcp", host, self.notify_port)
            logger.info("Node starting to listen notify from path = %s", path)

            with zmq_ctx(zmq.ROUTER, path) as sock:
                while True:
                    try:
                        identity, msg = sock.recv_multipart()
                        self._handle_message(msg)
                    except Exception as e:
                        logger.error("Error processing message: %s", e)
                        raise HandshakeError(f"Error processing message: {e}") from e

        self.notify_thread = threading.Thread(
            target=_async_wait, daemon=True, name="moriio-notify-listener"
        )
        self.notify_thread.start()

    def _handle_message(self, msg: bytes):
        """Handles incoming messages from remote nodes."""
        # Handles incoming remote messages:
        # Prefill Role:
        #   [write] mode: receives block information (allocation)
        #   [read]  mode: receives block release messages from decode side
        # Decode Role:
        #   [write] mode: receives KV cache write completion notifications
        handled = False
        try:
            data = msgpack.loads(msg)
            if isinstance(data, dict) and "req_id" in data:
                self._handle_structured_message(data)

                return
        except (msgpack.exceptions.ExtraData, msgpack.exceptions.UnpackException):
            pass

        try:
            msg_str = msg.decode("UTF-8")
            if msg_str.startswith(MoRIIOConstants.COMPLETION_PREFIX):
                self._handle_completion_message(msg_str)
                handled = True
        except UnicodeDecodeError:
            logger.warning("Received non-UTF8 message: %s", msg_str)
        if not handled:
            raise MoRIIOError(f"Unhandled message format: {msg_str}")

    def _handle_structured_message(self, data: dict):
        
        assert get_role()==ROLE.PRODUCER, "Only prefill can get block messages"
        req_id = data["req_id"]
        block_notify_list = data.get("block_notify_list", [])
        decode_dp_rank = data.get("decode_rank", 0)
        assert len(block_notify_list) > 0, (
            "block_notify_list cannot be empty in remote allocate message"
        )

        with self.lock:
            self.done_remote_allocate_req_dict[req_id] = RemoteAllocInfo(
                block_ids=block_notify_list, decode_dp_rank=decode_dp_rank
            )

    def _handle_completion_message(self, msg: str):
        with self.lock:
            if get_role() == ROLE.PRODUCER:
                self.done_req_ids.append(msg)
            else:
                self.done_write_cache_req_ids.append(msg)

    def send_notify(self, req_ids, remote_ip=None, remote_port=None):
        if not remote_ip or not remote_port:
            logger.warning("Missing remote_ip or remote_port for notification")
            return

        path = make_zmq_path("tcp", remote_ip, str(remote_port))

        if path not in self.paths:
            ctx = zmq.Context.instance()
            sock = make_zmq_socket(
                ctx=ctx, path=path, socket_type=zmq.DEALER, bind=False
            )
            self.paths[path] = sock

        req_list = req_ids if isinstance(req_ids, list) else [req_ids]

        sock = self.paths[path]
        try:
            for req_id in req_list:
                if not isinstance(req_id, str):
                    logger.warning(
                        "Invalid req_id type: %s, expected str", type(req_id)
                    )
                    continue
                sock.send(req_id.encode("utf-8"))
        except Exception as e:
            logger.error("Failed to send notification to %s: %s", path, e)
            self.paths.pop(path, None)
            raise

    def pop_finished_req_ids(self):
        # producer invocation: get the set of completed requests at the decode
        with self.lock:
            done_send = set(self.done_req_ids)
            self.done_req_ids = []
        return done_send

    def pop_finished_write_req_ids(self):
        # Call the consumer in write mode to get the collection after write completion
        with self.lock:
            done_write_cache = set(self.done_write_cache_req_ids)
            self.done_write_cache_req_ids = []
        return done_write_cache


class MoRIIOAgentMetadata(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    # required for @cached_property.d
    dict=True,
):
    engine_id: str
    agent_metadata: bytes
    kv_caches_base_addr: list[int]
    num_blocks: int
    block_len: int
    attn_backend_name: str


@dataclass
class ReqMeta:
    """Metadata for a single request."""

    local_block_ids: list[int]
    remote_block_ids: list[int]
    remote_host: str
    remote_port: int
    remote_handshake_port: int
    remote_notify_port: int
    remote_engine_id: str
    tp_size: int
    remote_dp_size: int


class MoRIIOConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        self.reqs_to_recv: dict[ReqId, ReqMeta] = {}
        self.reqs_to_save: dict[ReqId, ReqMeta] = {}
        self.reqs_to_send: dict[ReqId, float] = {}

    def __repr__(self):
        return_str = ""
        for req_id, req_meta in self.reqs_to_recv.items():
            return_str += (
                f"{req_id = },{req_meta.local_block_ids = },"
                f"{req_meta.remote_host = },{req_meta.remote_port = }"
                f"{req_meta.remote_engine_id = },{req_meta.tp_size = }"
            )
        return_str = f"MoRIIOConnectorMetadata:reqs_to_recv:{return_str},"

        for req_id, expiry in self.reqs_to_send.items():
            return_str += f"{req_id = },{expiry = }"
        return_str = f"MoRIIOConnectorMetadata:reqs_to_send:{return_str},"
        return return_str

    def add_new_req(
        self,
        request_id: ReqId,
        local_block_ids: list[int],
        kv_transfer_params: dict[str, Any],
        write_mode=False,
    ):
        _req = ReqMeta(
            local_block_ids=local_block_ids,
            remote_block_ids=kv_transfer_params["remote_block_ids"],
            remote_engine_id=kv_transfer_params["remote_engine_id"],
            remote_host=kv_transfer_params["remote_host"],
            remote_port=kv_transfer_params["remote_port"],
            remote_handshake_port=kv_transfer_params["remote_handshake_port"],
            remote_notify_port=kv_transfer_params["remote_notify_port"],
            tp_size=kv_transfer_params.get("tp_size", 1),
            remote_dp_size=kv_transfer_params.get("remote_dp_size", 1),
        )
        if write_mode:
            self.reqs_to_save[request_id] = _req
        else:
            self.reqs_to_recv[request_id] = _req


class MoRIIOConnector(KVConnectorBase_V1):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        super().__init__(vllm_config, role)
        assert vllm_config.kv_transfer_config is not None
        # assert vllm_config.kv_transfer_config.engine_id is not None
        self._set_port_defaults(vllm_config)

        self.engine_id = (
            str(get_ip())
            + ":"
            + str(
                vllm_config.kv_transfer_config.kv_connector_extra_config[
                    "handshake_port"
                ]
            )
        )
        self.mode = get_moriio_mode()
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler: MoRIIOConnectorScheduler | None = (
                MoRIIOConnectorScheduler(vllm_config, self.engine_id)
            )
            self.connector_worker: MoRIIOConnectorWorker | None = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = MoRIIOConnectorWorker(vllm_config, self.engine_id)
        logger.info(
            "Initialized MoRIIO Connector,engine_id:%s,role: %s", 
            self.engine_id, role.value
        )

    ############################################################
    # Scheduler Side Methods
    ############################################################

    def _set_port_defaults(self, vllm_config: VllmConfig):
        kv_transfer_config = vllm_config.kv_transfer_config
        extra_config = kv_transfer_config.kv_connector_extra_config

        if "handshake_port" not in extra_config or not extra_config["handshake_port"]:
            extra_config["handshake_port"] = MoRIIOConstants.DEFAULT_HANDSHAKE_PORT

        if "notify_port" not in extra_config or not extra_config["notify_port"]:
            extra_config["notify_port"] = MoRIIOConstants.DEFAULT_NOTIFY_PORT



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
            request, blocks, num_external_tokens, self.connector_worker
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

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """Get the finished recving and sending requests."""
        assert self.connector_worker is not None
        return self.connector_worker.get_finished()

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        if self.mode == MoRIIOMode.WRITE and get_role() == ROLE.CONSUMER:
            self.connector_worker.moriio_wrapper.async_wait_reqid()

        assert isinstance(self._connector_metadata, MoRIIOConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        # Only producer/prefill saves KV Cache
        if get_role() == ROLE.CONSUMER:
            return
        assert self.connector_worker is not None, (
            "save_kv_layer called on scheduler role"
        )

        assert isinstance(self._connector_metadata, MoRIIOConnectorMetadata), (
            "Connector metadata not initialized yet"
        )
        self.connector_worker.save_kv_layer(
            self._connector_metadata, layer_name, kv_layer, attn_metadata, **kwargs
        )

        return None

    def wait_for_save(self):
        pass

    def has_connector_metadata(self) -> bool:
        """Check whether the connector metadata is currently set.

        Returns:
            bool: True if connector metadata exists, False otherwise.
        """
        try:
            return self._connector_metadata is not None
        except AttributeError:
            return False


class MoRIIOConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self.engine_id: EngineId = engine_id
        self.side_channel_host = envs.VLLM_NIXL_SIDE_CHANNEL_HOST
        self.mode = get_moriio_mode()

        self.handshake_port = (
            self.vllm_config.kv_transfer_config.kv_connector_extra_config[
                "handshake_port"
            ]
        )
        logger.info("Initializing MoRIIO Scheduler engine_id = %s", engine_id)

        self.side_notify_port = (
            self.vllm_config.kv_transfer_config.kv_connector_extra_config["notify_port"]
        )
        self.tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        self.dp_rank = self.vllm_config.parallel_config.data_parallel_rank
        self.is_producer = vllm_config.kv_transfer_config.kv_role == "kv_producer"
        # Requests that need to start recv/send.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        self._reqs_need_recv: dict[ReqId, tuple[Request, list[int]]] = {}
        self._reqs_need_save: dict[ReqId, tuple[Request, list[int]]] = {}

        # For chunked prefill, we perform layer-wise access within the final chunk.
        # TODO: Perform transfer at end chunk.
        self._reqs_need_pending_save: dict[ReqId, tuple[Request, list[int]]] = {}

        if self.is_producer:
            set_role(ROLE.PRODUCER)
        else:
            set_role(ROLE.CONSUMER)
        # Reqs to send and their expiration time
        self._reqs_need_send: dict[ReqId, float] = {}
        self.sock = None
        self.is_producer = vllm_config.kv_transfer_config.kv_role == "kv_producer"
        self.paths: dict[str, zmq.Socket] = {}

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
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
        if self.is_producer:
            return 0, False

        if self.mode == MoRIIOMode.WRITE:
            # MoriiO in write mode, no remote prefill

            return len(request.prompt_token_ids) - num_computed_tokens, True

        return len(request.prompt_token_ids) - 1 - num_computed_tokens, False

    def send_notify_block(
        self, req_id: str, block_notify_list: list[int], host=None, port=None
    ):
        path = make_zmq_path("tcp", host, port)
        if path not in self.paths:
            ctx = zmq.Context.instance()
            sock = make_zmq_socket(
                ctx=ctx, path=path, socket_type=zmq.DEALER, bind=False
            )
            self.paths[path] = sock

        data = {
            "req_id": req_id,
            "block_notify_list": block_notify_list or [],
            "decode_rank": self.dp_rank,
            "type": "remote_blocks",
        }
        serialized_data = msgpack.dumps(data)
        self.paths[path].send(serialized_data)

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
        connector_worker: Optional["MoRIIOConnectorWorker"] = None,
    ):
        params = request.kv_transfer_params
        if params.get("do_remote_decode"):
            local_block_ids = blocks.get_block_ids()[0]
            self._reqs_need_save[request.request_id] = (request, local_block_ids)

        if params is not None and params.get("do_remote_prefill"):
            if self.mode == MoRIIOMode.READ:
                if remote_block_ids := params.get("remote_block_ids"):
                    if all(
                        p in params
                        for p in ("remote_engine_id", "remote_host", "remote_port")
                    ):
                        # If remote_blocks and num_external_tokens = 0, we
                        # a full prefix cache hit on the D worker. We need to call
                        # send_notif in _read_blocks to free the memory on the P.

                        # Get unhashed blocks to pull from remote.
                        local_block_ids = blocks.get_block_ids()[0]
                        assert len(local_block_ids) <= len(remote_block_ids)
                        if len(local_block_ids) == len(remote_block_ids):
                            pass
                        else:
                            local_block_ids = remote_block_ids[-len(local_block_ids) :]

                        self._reqs_need_recv[request.request_id] = (
                            request,
                            local_block_ids,
                        )
                    else:
                        logger.warning(
                            "Got invalid KVTransferParams: %s. This "
                            "request will not utilize KVTransfer",
                            params,
                        )

            else:
                remote_dp_rank = request.kv_transfer_params.get("remote_dp_rank", 0)

                for tp_index in range(self.tp_size):
                    target_port = request.kv_transfer_params[
                        "remote_notify_port"
                    ] + get_port_offset(remote_dp_rank, tp_index)

                    self.send_notify_block(
                        req_id=request.request_id,
                        block_notify_list=blocks.get_block_ids()[0],
                        host=params.get("remote_host"),
                        port=target_port,
                    )

            # Only trigger 1 KV transfer per request.

            params["do_remote_prefill"] = False

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = MoRIIOConnectorMetadata()

        if self.mode == MoRIIOMode.WRITE:
            # when async_load_kv finished,
            # new reqs will be added to scheduler_output.scheduled_new_reqs

            if get_role() == ROLE.CONSUMER:
                for new_req in scheduler_output.scheduled_new_reqs:
                    red_id = new_req.req_id
                    local_block_ids = list(new_req.block_ids)[0]
                    assert new_req.sampling_params is not None, (
                        f"sampling_params is None for req {new_req.req_id}"
                    )
                    assert hasattr(new_req.sampling_params, "extra_args"), (
                        f"sampling_params missing extra_args for req {new_req.req_id}"
                    )
                    kv_transfer_params = new_req.sampling_params.extra_args[
                        "kv_transfer_params"
                    ]
                    meta.add_new_req(
                        red_id,
                        local_block_ids,
                        kv_transfer_params,
                    )
            if get_role() == ROLE.PRODUCER:
                # This is the logic for checking against chunked prefill.
                # When the last chunk is identified,
                # It places the request metadata into the saving queue.

                for i, req_id in enumerate(
                    scheduler_output.scheduled_cached_reqs.req_ids
                ):
                    new_block_ids = (
                        scheduler_output.scheduled_cached_reqs.new_block_ids[i]
                    )

                    if new_block_ids is not None:
                        block_ids = new_block_ids[0]
                        #TODO : hybrid attn, etc
                        req, existing_blocks = self._reqs_need_pending_save[req_id]
                        updated_blocks = list(existing_blocks) + (block_ids)
                        self._reqs_need_pending_save[req_id] = (req, updated_blocks)
                        if (
                            len(self._reqs_need_pending_save[req_id][1])
                            * self.block_size
                            >= req.num_prompt_tokens
                        ):
                            meta.add_new_req(
                                request_id=req_id,
                                local_block_ids=self._reqs_need_pending_save[req_id][1],
                                kv_transfer_params=req.kv_transfer_params,
                                write_mode=True,
                            )
                            del self._reqs_need_pending_save[req_id]

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
            if req.num_prompt_tokens > len(block_ids) * self.block_size:
                # not last chunk prefill
                self._reqs_need_pending_save[req_id] = (req, block_ids)
                continue
            meta.add_new_req(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=req.kv_transfer_params,
                write_mode=True,
            )
        # Clear the list once workers start the transfers

        meta.reqs_to_send = self._reqs_need_send

        self._reqs_need_recv.clear()
        self._reqs_need_save.clear()
        self._reqs_need_send = {}

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
            "MoriioConnector request_finished, request_status=%s, "
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
            self._reqs_need_recv[request.request_id] = (request, [])
            params["do_remote_prefill"] = False
            return False, None

        if (
            not params.get("do_remote_decode")
            or request.status != RequestStatus.FINISHED_LENGTH_CAPPED
        ):
            return False, None

        # computed_block_ids = block_ids if all_full else block_ids[:-1]
        computed_block_ids = block_ids
        # If prompt < block_size, no xfer so free blocks immediately.
        delay_free_blocks = len(computed_block_ids) > 0

        if delay_free_blocks:
            # Prefill request on remote. It will be read from D upon completion
            self._reqs_need_send[request.request_id] = (
                time.perf_counter() + envs.VLLM_NIXL_ABORT_REQUEST_TIMEOUT
            )

        # If we execute in P-D serial mode, no notification port is needed.
        return delay_free_blocks, dict(
            do_remote_prefill=True,
            do_remote_decode=False,
            remote_block_ids=computed_block_ids,
            remote_engine_id=self.engine_id,
            remote_host=self.side_channel_host,
            remote_port=self.handshake_port,
            tp_size=self.vllm_config.parallel_config.tensor_parallel_size,
        )


class MoRIIOConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        if not MoRIIO_enabled:
            raise RuntimeError(
                "MoRIIO is not available. Please ensure the 'mori' package "
                "is installed and properly configured."
            )

        self.moriio_config = MoRIIOConfig.from_vllm_config(vllm_config)
        self.mode = get_moriio_mode()

        logger.info("Initializing MoRIIO worker %s", engine_id)

        logging.getLogger("aiter").disabled = True

        # Config.
        self.vllm_config = vllm_config
        self.kv_transfer_config = vllm_config.kv_transfer_config
        self.is_producer = self.kv_transfer_config.is_kv_producer

        if self.is_producer:
            set_role(ROLE.PRODUCER)
        else:
            set_role(ROLE.CONSUMER)
        # mori engine
        self._rank = get_world_group().rank
        self._local_rank = get_world_group().local_rank
        self.tp_rank = self.moriio_config.tp_rank
        self.dp_rank = self.moriio_config.dp_rank

        self.local_ip = self.moriio_config.local_ip
        self.local_kv_port = self.moriio_config.local_kv_port
        self.proxy_ip = self.moriio_config.proxy_ip
        self.local_ping_port = self.moriio_config.local_ping_port
        self.proxy_ping_port = self.moriio_config.proxy_ping_port
        self.http_port = self.moriio_config.http_port
        self.handshake_port = self.moriio_config.handshake_port
        self.notify_port = self.moriio_config.notify_port

        self.zmq_context = zmq.Context()
        self.metadata_address = (
            f"{self.moriio_config.local_ip}:{self.moriio_config.local_ping_port}"
        )
        self.request_address = (
            f"{self.moriio_config.local_ip}:{self.moriio_config.http_port}"
        )

        self.moriio_engine = None
        self._handle_request_thread = None
        self._ping_thread = None
        self._writer = MoRIIOWriter(self)

        engine_suffix = (
            f"{self.moriio_config.local_ip}:{self.moriio_config.handshake_port}"
            f":tp {self.tp_rank}:dp {self.dp_rank}"
        )
        if not self.is_producer:
            self.moriio_engine = IOEngine(
                "consumer:" + engine_suffix,
                IOEngineConfig(
                    self.moriio_config.local_ip, self.moriio_config.local_kv_port
                ),
            )
        else:
            self.moriio_engine = IOEngine(
                "producer:" + engine_suffix,
                IOEngineConfig(
                    self.moriio_config.local_ip, self.moriio_config.local_kv_port
                ),
            )
        logger.debug(
            "build MORI IOEngine %s:%s",
            self.moriio_config.local_ip,
            self.moriio_config.local_kv_port,
        )

        if self._rank == 0 and self.moriio_config.proxy_ip:
            self._ping_thread = threading.Thread(
                target=self._ping, args=(self.zmq_context,), daemon=True
            )
            self._ping_thread.start()

        logger.info(
            "Initializing MoRIIO Engine, engine = %s, role = %s",
            self.moriio_engine,
            "producer" if self.is_producer else "consumer",
        )

        # Agent.
        self.moriio_wrapper = MoRIIOWrapper(tp_rank=self.tp_rank, dp_rank=self.dp_rank)
        self.moriio_wrapper.set_moriio_engine(self.moriio_engine)
        self.moriio_wrapper.set_backend_type(BackendType.RDMA)
        self.moriio_wrapper.notify_port = self.moriio_config.notify_port
        self.local_kv_cache_metadata: list[bytes] = []
        self.local_kv_cache_size: list[int] = []
        self.layer_name_to_local_kv_cache_metadata: dict[str, list[bytes]] = {}

        self.remote_kv_cache_metadata: list[bytes] = []
        self.remote_kv_cache_size: list[int] = []
        self.layer_name_to_remote_kv_cache_metadata: dict[str, dict[str, list[Any]]] = (
            dict()
        )
        self.slot_size_bytes = 0

        self.load_ready_flag: dict[str, bool] = {}
        self.write_ready_flags: dict[str, bool] = {}
        self.kv_cache_shape = None
        self.block_shape = None
        self.kv_element_size = 0

        # Map of engine_id -> {agent_name0, agent_name1..}.
        self._remote_agents: dict[EngineId, set[str]] = {}

        self.side_channel_port: int = (
            self.moriio_config.handshake_port
            + get_port_offset(self.dp_rank, self.tp_rank)
        )
        self.engine_id: EngineId = engine_id

        self.world_size = get_tensor_model_parallel_world_size()
        self.tp_group = get_tp_group()

        # KV Caches and moriio tracking data.
        self.kv_caches: dict[str, torch.Tensor] = {}

        # Map of engine_id -> kv_caches_base_addr. For TP case, each local
        # rank will still only pull from a single remote TP worker.
        self.kv_caches_base_addr: dict[EngineId, list[int]] = {}

        # Number of MoRIIO regions. Currently one region per cache
        # (so 1 per layer for MLA, otherwise 2 per layer)
        self.num_regions = 0
        self.num_layers = 0

        # Map of engine_id -> num_blocks. All ranks in the same deployment will
        # have the same number of blocks.
        self.dst_num_blocks: dict[EngineId, int] = {}
        # In progress transfers.
        self._recving_transfers: defaultdict[ReqId, list] = defaultdict(list)
        self._recving_transfers_callback_addr: dict[ReqId, tuple[str, str]] = {}

        # Track the expiration time of requests that are waiting to be sent.
        self._reqs_to_send: dict[ReqId, float] = {}

        # Background thread for handling new handshake requests.
        self._moriio_handshake_listener_t: threading.Thread | None = None
        # Background thread for initializing new MoRIIO handshakes.
        self._handshake_initiation_executor = ThreadPoolExecutor(
            # MoRIIO is not guaranteed to be thread-safe, limit 1 worker.
            max_workers=1,
            thread_name_prefix="vllm-moriio-handshake-initiator",
        )
        self._ready_requests = queue.Queue[tuple[ReqId, ReqMeta]]()
        self._handshake_futures: dict[EngineId, Future[set[str]]] = {}
        # Protects _handshake_futures and _remote_agents.
        self._handshake_lock = threading.RLock()

        self.block_size = vllm_config.cache_config.block_size
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config

        self.block_window_per_layer: list[int | None] = []
        self.use_mla = self.model_config.use_mla
        self.built_session = False
        self.built_write_session: defaultdict[str, list] = defaultdict(list)
        backend = get_attn_backend(
            self.model_config.get_head_size(),
            self.model_config.dtype,
            self.cache_config.cache_dtype,
            self.block_size,
            use_mla=self.use_mla,
        )
        self.backend_name = backend.get_name()
        attn_backend = AttentionBackendEnum[self.backend_name]
        self._use_flashinfer = attn_backend == AttentionBackendEnum.FLASHINFER
        self._use_pallas = attn_backend == AttentionBackendEnum.PALLAS
        # attn_backend = backend_name_to_enum(self.backend_name)
        # self._use_flashinfer = attn_backend == _Backend.FLASHINFER
        logger.debug("Detected attention backend %s", self.backend_name)

    def schedule_write_blocks(
        self,
        request_id: str,
        dst_engine_id: str,
        local_block_ids: list[int],
        remote_block_ids: list[int] | None,
        layer_name: str,
        kv_layer: torch.Tensor,
        remote_notify_port: int,
        remote_ip: str,
    ) -> None:
        """Schedule a block write operation.

        Args:
            request_id: Unique identifier for the request
            dst_engine_id: Destination engine ID
            local_block_ids: Local block IDs to transfer
            remote_block_ids: Hint for remote block IDs
            layer_name: Name of the layer
            kv_layer: KV cache tensor
            remote_notify_port: Port for completion notification
            remote_ip: IP address of remote node
        """

        stream = torch.cuda.current_stream()
        event = torch.cuda.Event()
        event.record(stream)

        task = WriteTask(
            request_id=request_id,
            dst_engine_id=dst_engine_id,
            local_block_ids=local_block_ids,
            remote_block_ids_hint=remote_block_ids,
            layer_name=layer_name,
            event=event,
            remote_notify_port=remote_notify_port,
            remote_ip=remote_ip,
        )
        self._writer.schedule_write(task)

    def _get_built_session(self, remote_engine_id):
        if remote_engine_id not in self.built_write_session:
            cur_remote_engine_sessions = []
            for ln, local_meta in self.layer_name_to_local_kv_cache_metadata.items():
                unpcaked_local_memory_meta = (
                    self.moriio_wrapper.get_unpack_memory_metadata(local_meta[0])
                )
                unpcaked_remote_memory_meta = (
                    self.moriio_wrapper.get_unpack_memory_metadata(
                        self.layer_name_to_remote_kv_cache_metadata[remote_engine_id][
                            ln
                        ][0]
                    )
                )
                cur_remote_engine_sessions.append(
                    self.moriio_wrapper.build_session(
                        unpcaked_local_memory_meta, unpcaked_remote_memory_meta
                    )
                )
            self.built_write_session[remote_engine_id] = cur_remote_engine_sessions
        return self.built_write_session[remote_engine_id]

    def _ping(self, zmq_context):
        http_request_address = f"http://{self.request_address}/v1/completions"
        role = "P" if self.is_producer else "D"

        retry_count = 0
        index = 1
        should_break = True
        with zmq_context.socket(zmq.DEALER) as sock:
            sock.connect(f"tcp://{self.proxy_ip}:{self.proxy_ping_port}")

            while True:
                try:
                    data = {
                        "type": "register",
                        "role": role,
                        "index": str(index),
                        "request_address": http_request_address,
                        "handshake_port": self.handshake_port,
                        "notify_port": self.notify_port,
                        "dp_size": self.moriio_config.dp_size,
                        "tp_size": self.moriio_config.tp_size,
                        "transfer_mode": self.mode.name,
                    }

                    sock.send(msgpack.dumps(data))
                    # logger.debug(f"Successfully sent ping message #{index}")
                    retry_count = 0

                except ConnectionRefusedError:
                    logger.info(
                        "Connection refused: %s:%s -> %s:%s",
                        self.local_ip,
                        self.local_ping_port,
                        self.proxy_ip,
                        self.proxy_ping_port,
                    )
                    retry_count += 1

                except OSError as e:
                    logger.info("OS error when sending ping: %s", e)
                    retry_count += 1

                except Exception as e:
                    logger.info("Unexpected error when sending ping: %s", e)
                    retry_count += 1

                finally:
                    if retry_count >= MoRIIOConstants.MAX_PING_RETRIES:
                        logger.error(
                            "Max retries (%s) exceeded. Stopping ping loop.",
                            MoRIIOConstants.MAX_PING_RETRIES,
                        )
                        should_break = True
                    time.sleep(MoRIIOConstants.PING_INTERVAL)
                    index += 1
                if should_break:
                    break

    # def handle_proxy_request(self):
    #     if self.is_producer:
    #         raise NotImplementedError(
    #             "prefill instance doesn't need to send kv cache in pull mode"
    #         )
    #     while True:
    #         socks = dict(self.poller.poll())
    #         logger.debug("handle_proxy_request: socks = %s", socks)

    #         if self.metadata_socket not in socks:
    #             continue

    def close(self):
        if hasattr(self, "_handshake_initiation_executor"):
            self._handshake_initiation_executor.shutdown(wait=False)

        if (
            hasattr(self, "_moriio_handshake_listener_t")
            and self._moriio_handshake_listener_t
        ):
            self._moriio_handshake_listener_t.join(timeout=0)

        if hasattr(self, "zmq_context") and self.zmq_context:
            self.zmq_context.destroy(linger=0)
            self.zmq_context = None

    def __del__(self):
        self.close()

    @staticmethod
    def _moriio_handshake_listener(
        metadata: MoRIIOAgentMetadata,
        ready_event: threading.Event,
        base_port: int,
        tp_rank: int,
        dp_rank: int,
        layer_name_to_local_kv_cache_metadata: dict,
    ):
        """Background thread for getting new MoRIIO handshakes."""

        encoder = msgspec.msgpack.Encoder()
        encoded_data = encoder.encode(metadata)
        size_in_bytes = len(encoded_data)
        logger.debug(
            "Size of encoded MoRIIOAgentMetadata: %s bytes", str(size_in_bytes)
        )

        # Listen for new requests for metadata.
        host = "*"

        path = make_zmq_path("tcp", host, base_port)
        logger.debug("mori handshake starting listening on path: %s", path)

        with zmq_ctx(zmq.ROUTER, path) as sock:
            ready_event.set()
            while True:
                identity, msg = sock.recv_multipart()
                if (
                    msg != MoRIIOConstants.GET_META_MSG
                    and msg != MoRIIOConstants.POP_DONE_RECV
                ):
                    logger.error("Connection listener got unexpected message")
                    raise HandshakeError("handshake failed, unexpected msg type")
                elif msg == MoRIIOConstants.GET_META_MSG:
                    sock.send_multipart(
                        (identity, b"", encoded_data)
                    )  # send local mori io engine meta data
                    logger.debug("MoRIIO handshake listener sent metadata")
                    # now we send tensor meta data for each block
                    buf = msgpack.dumps(layer_name_to_local_kv_cache_metadata)
                    sock.send_multipart((identity, b"", buf))
                elif msg == MoRIIOConstants.POP_DONE_RECV:
                    _, req_id = sock.recv_multipart()
                    logger.debug(
                        "MoRIIO handshake listener received done recv for req",
                        req_id.decode(),
                    )

    def _moriio_handshake(
        self,
        host: str,
        port: int,
        remote_tp_size: int,
        expected_engine_id: str,
        remote_dp_rank: int = 0,
    ) -> set[str]:
        """Do a MoRIIO handshake with a remote instance."""

        start_time = time.perf_counter()

        # NOTE(rob): we need each rank to have a unique port. This is
        # a hack to keep us moving. We will switch when moving to etcd
        # or where we have a single ZMQ socket in the scheduler.

        port_offset = get_port_offset(remote_dp_rank, self.tp_rank)
        path = make_zmq_path("tcp", host, port + port_offset)
        logger.debug("handshake Querying metadata on path: %s", path)

        # Send query for the request.
        with zmq_ctx(zmq.DEALER, path) as sock:
            logger.debug("prepare send msg INSTAZNCE: %s", path)
            sock.send(MoRIIOConstants.GET_META_MSG)
            received_frame = sock.recv_multipart()
            if len(received_frame) != 2 or received_frame[0] != b"":
                assert 0, f"unexpected frame! {received_frame = }"

            metadata_bytes = received_frame[1]
            decoder = msgspec.msgpack.Decoder(MoRIIOAgentMetadata)
            metadata = decoder.decode(metadata_bytes)
            got_metadata_time = time.perf_counter()
            logger.info(
                "MoRIIO handshake: get metadata took: %s",
                got_metadata_time - start_time,
            )

            self.moriio_wrapper.remote_engine_ip = host
            remote_agent_name = self.moriio_wrapper.register_remote_engine(
                metadata.agent_metadata
            )

            logger.debug(
                "MoRIIO handshake: registered"
                "remote agent %s for engine ID %s, path = %s",
                remote_agent_name,
                expected_engine_id,
                path,
            )

            if len(self.local_kv_cache_metadata) > 0:
                logger.warning(
                    "len(self.local_kv_cache_metadata) = %s,"
                    "maybe you didnt clear this buffer correctly",
                    len(self.local_kv_cache_metadata),
                )
                self.local_kv_cache_metadata = []
            if len(self.remote_kv_cache_metadata) > 0:
                logger.warning(
                    "len(self.remote_kv_cache_metadata) = %s,"
                    "maybe you didnt clear this buffer correctly",
                    len(self.remote_kv_cache_metadata),
                )
                self.remote_kv_cache_metadata = []

            received_frame = sock.recv_multipart()
            if len(received_frame) != 2 or received_frame[0] != b"":
                assert 0, f"Unexpected frame! {received_frame = }"
            buf = received_frame[1]
            self.layer_name_to_remote_kv_cache_metadata[expected_engine_id] = (
                msgpack.loads(buf)
            )

            setup_agent_time = time.perf_counter()
            logger.debug(
                "MoRIIO handshake: add agent took: %s",
                setup_agent_time - got_metadata_time,
            )

        return {remote_agent_name}

    def _background_moriio_handshake(
        self, req_id: str, remote_engine_id: EngineId, meta: ReqMeta
    ):
        # Do MoRIIO handshake in background and add to _ready_requests when done.
        fut = None
        if remote_engine_id is not None:
            fut = self._handshake_futures.get(remote_engine_id)
        if fut is None:
            host = meta.remote_host
            port = int(meta.remote_handshake_port)
            tp_size = int(meta.tp_size)
            remote_dp_size = int(meta.remote_dp_size)

        def request_ready(_f: Future[Any], entry=(req_id, meta)):
            logger.info("MoRIIO handshake done for request %s", req_id)
            self._ready_requests.put(entry)
            self.load_ready_flag[remote_engine_id] = True
            self.write_ready_flags[remote_engine_id] = True

        fut_list = []

        # In dp(prefill)<->dp(decode) communication, we require an all-to-all handshake.

        for cur_dp_rank in range(remote_dp_size):
            dp_engine_id = self.get_engine_name_with_dp(remote_engine_id, cur_dp_rank)
            future = self._handshake_initiation_executor.submit(
                self._moriio_handshake, host, port, tp_size, dp_engine_id, cur_dp_rank
            )
            fut_list.append(future)

            def done_callback(f: Future[set[str]], eid=dp_engine_id):
                with self._handshake_lock:
                    self._handshake_futures.pop(eid, None)
                    try:
                        self._remote_agents[eid] = f.result()
                    except Exception:
                        logger.exception("Handshake with %s failed", eid)

            future.add_done_callback(done_callback)
            self._handshake_futures[dp_engine_id] = future

        # fut = fut_list
        def wait_all_dp():
            for future in fut_list:
                future.result()
            return True

        all_done_future = self._handshake_initiation_executor.submit(wait_all_dp)
        all_done_future.add_done_callback(request_ready)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register the KV Cache data in moriio."""

        _, first_kv_cache = next(iter(kv_caches.items()))
        kv_elem_size = first_kv_cache.element_size()

        use_mla = len(first_kv_cache.shape) == 3
        assert use_mla == self.use_mla

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
            self.slot_size_bytes = (
                kv_elem_size * n_kv_heads * head_dim
            )  # 1 token 1 layer size , slot size
        assert block_size == self.block_size
        # TODO(tms): self.block_len needs to be per-layer for sliding window,
        # hybrid attn, etc
        # block size in bytes
        self.block_len = kv_elem_size * math.prod(block_shape)
        self.kv_cache_shape = first_kv_cache.shape
        self.block_shape = block_shape
        self.kv_element_size = kv_elem_size

        self.dst_num_blocks[self.engine_id] = self.num_blocks
        self.kv_caches = kv_caches  # layer name to kv cache
        kv_caches_base_addr = []
        caches_data = []

        for cache_or_caches in kv_caches.values():
            cache_list = (
                [cache_or_caches]
                if use_mla or self._use_flashinfer
                else cache_or_caches
            )
            for cache in cache_list:
                base_addr = cache.data_ptr()
                region_len = self.num_blocks * self.block_len
                caches_data.append((base_addr, region_len, cache.device.index, ""))
                kv_caches_base_addr.append(base_addr)

        for layer_name, kv_cache in kv_caches.items():
            if layer_name not in self.layer_name_to_local_kv_cache_metadata:
                self.layer_name_to_local_kv_cache_metadata[layer_name] = []

            moriio_mem_metadata = self.moriio_wrapper.register_local_tensor(kv_cache)
            self.layer_name_to_local_kv_cache_metadata[layer_name].append(
                moriio_mem_metadata
            )

            self.local_kv_cache_size.append(cache.nelement() * cache.element_size())

        self.kv_caches_base_addr[self.engine_id] = kv_caches_base_addr
        self.num_regions = len(caches_data)
        self.num_layers = len(self.kv_caches.keys())

        # Optimization for models with local attention (Llama 4)
        if self.vllm_config.model_config.hf_config.model_type == "llama4":
            from transformers import Llama4TextConfig

            assert isinstance(
                self.vllm_config.model_config.hf_text_config, Llama4TextConfig
            )
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
            logger.debug(
                "Llama 4 block window per layer mapping: %s",
                self.block_window_per_layer,
            )
            assert len(self.block_window_per_layer) == self.num_layers

        metadata = MoRIIOAgentMetadata(
            engine_id=self.engine_id,
            agent_metadata=self.moriio_wrapper.get_agent_metadata(),
            kv_caches_base_addr=self.kv_caches_base_addr[self.engine_id],
            num_blocks=self.num_blocks,
            block_len=self.block_len,
            attn_backend_name=self.backend_name,
        )
        ready_event = threading.Event()
        self._moriio_handshake_listener_t = threading.Thread(
            target=self._moriio_handshake_listener,
            args=(
                metadata,
                ready_event,
                self.side_channel_port,
                self.tp_rank,
                self.dp_rank,
                self.layer_name_to_local_kv_cache_metadata,
            ),
            daemon=True,
            name="moriio_handshake_listener",
        )
        self._moriio_handshake_listener_t.start()
        ready_event.wait()  # Wait for listener ZMQ socket to be ready.
        self.moriio_wrapper.async_wait_reqid()

    def get_finished(self) -> tuple[set[str], set[str]]:
        """
        Get requests that are done sending or recving on this specific worker.
        The scheduler process (via the MultiprocExecutor) will use this output
        to track which workers are done.
        """

        done_sending = set()

        if self.is_producer:
            done_sending = self.moriio_wrapper.pop_finished_req_ids()
            if self.mode == MoRIIOMode.WRITE:
                done_recving = set()
            else:
                done_recving = self._pop_done_transfers()
        else:
            if self.mode == MoRIIOMode.WRITE:
                self.moriio_wrapper.async_wait_reqid()
            done_sending, done_recving = (
                set(),
                self.moriio_wrapper.pop_finished_write_req_ids(),
            )

        return done_sending, done_recving

    def _pop_done_transfers(self) -> set[str]:
        done_req_ids: set[str] = set()
        for req_id, status_list in self._recving_transfers.items():
            if status_list[-1].Succeeded():
                done_req_ids.add(req_id)

                self.moriio_wrapper.send_notify(
                    req_id,
                    self._recving_transfers_callback_addr[req_id][0],
                    self._recving_transfers_callback_addr[req_id][1],
                )
                del self._recving_transfers[req_id]
                del self._recving_transfers_callback_addr[req_id]

        return done_req_ids

    def save_kv_layer(
        self,
        metadata: MoRIIOConnectorMetadata,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ):
        if not self.is_producer:
            return
        if self.mode == MoRIIOMode.READ:
            return
        remote_engine_id = None

        for req_id, meta in metadata.reqs_to_save.items():
            # we only need to check if dp0 in rank
            remote_engine_id = (
                str(meta.remote_host) + ":" + str(meta.remote_handshake_port)
            )

            meta.remote_engine_id = remote_engine_id

            dp0_remote_engine_id = self.get_engine_name_with_dp(remote_engine_id, 0)
            if dp0_remote_engine_id not in self._remote_agents:
                # Initiate handshake with remote engine to exchange metadata.
                with self._handshake_lock:
                    if remote_engine_id not in self._remote_agents:
                        self._background_moriio_handshake(
                            req_id, remote_engine_id, meta
                        )

                        continue
            self._write_blocks_for_req(req_id, meta, layer_name, kv_layer)

        while True:
            if (
                self._ready_requests.empty()
                and remote_engine_id not in self.write_ready_flags
            ):
                continue
            elif not self._ready_requests.empty() and (
                remote_engine_id in self.write_ready_flags
            ):
                self._write_blocks_for_req(
                    *self._ready_requests.get_nowait(), layer_name, kv_layer
                )
                break
            else:
                break

    def get_engine_name_with_dp(self, engine_name, dp_rank):
        return f"{engine_name}_dp{dp_rank}"

    def start_load_kv(self, metadata: MoRIIOConnectorMetadata):
        """
        Start loading by triggering non-blocking moriio_xfer.
        We check for these trnxs to complete in each step().
        """
        if self.is_producer:
            self.moriio_wrapper.async_wait_reqid()
            return
        if self.mode == MoRIIOMode.WRITE:
            return

        wait_handshake_readd_req = False
        remote_engine_id = None

        for req_id, meta in metadata.reqs_to_recv.items():
            remote_engine_id = (
                str(meta.remote_host) + ":" + str(meta.remote_handshake_port)
            )
            meta.remote_engine_id = remote_engine_id
            dp0_remote_engine_id = self.get_engine_name_with_dp(remote_engine_id, 0)
            if dp0_remote_engine_id not in self._remote_agents:
                # Initiate handshake with remote engine to exchange metadata.
                with self._handshake_lock:
                    if remote_engine_id not in self._remote_agents:
                        self._background_moriio_handshake(
                            req_id, remote_engine_id, meta
                        )
                        wait_handshake_readd_req = True

                        continue

            # Handshake already completed, start async read xfer.
            self._read_blocks_for_req(req_id, meta)
        # Start transfers for requests whose handshakes have now finished.

        while True:
            if (
                self._ready_requests.empty()
                and remote_engine_id not in self.load_ready_flag
                and wait_handshake_readd_req
            ):
                continue
            elif (
                not self._ready_requests.empty()
                and remote_engine_id in self.load_ready_flag
            ):
                self._read_blocks_for_req(*self._ready_requests.get_nowait())
                break
            else:
                break

        self._reqs_to_send.update(metadata.reqs_to_send)

    def _read_blocks_for_req(self, req_id: str, meta: ReqMeta):
        logger.debug(
            "Remote agent %s available, calling _read_blocks for req %s",
            meta.remote_engine_id,
            req_id,
        )
        self._read_blocks(
            request_id=req_id,
            dst_engine_id=meta.remote_engine_id,
            local_block_ids=meta.local_block_ids,
            remote_block_ids=meta.remote_block_ids,
            remote_host=meta.remote_host,
            remote_notify_port=meta.remote_notify_port,
        )

    def _write_blocks_for_req(self, req_id: str, meta: ReqMeta, layer_name, kv_layer):
        # logger.debug(f"write block for req {req_id} to remote engine "
        #             f"{meta.remote_engine_id}")

        self.schedule_write_blocks(
            request_id=req_id,
            dst_engine_id=meta.remote_engine_id,
            local_block_ids=meta.local_block_ids,
            remote_block_ids=meta.remote_block_ids,
            layer_name=layer_name,
            kv_layer=kv_layer,
            remote_notify_port=meta.remote_notify_port,
            remote_ip=meta.remote_host,
        )

    def _is_last_layer(self, layer_name):
        return layer_name == list(self.kv_caches.keys())[-1]

    def merge_contiguous_blocks(
        self,
        offsets_local: list[int],
        offsets_remote: list[int],
        sizes: list[int],
        assume_sorted: bool = False,
    ) -> tuple[list[int], list[int], list[int]]:
        n = len(offsets_local)
        if n == 0:
            return [], [], []
        if not (n == len(offsets_remote) == len(sizes)):
            raise ValueError("Input list lengths mismatch")
        local_arr = np.fromiter(offsets_local, dtype=np.int64, count=n)
        remote_arr = np.fromiter(offsets_remote, dtype=np.int64, count=n)
        sizes_arr = np.fromiter(sizes, dtype=np.int64, count=n)

        if assume_sorted:
            local_sorted = local_arr
            remote_sorted = remote_arr
            sizes_sorted = sizes_arr
        else:
            if np.all(local_arr[:-1] <= local_arr[1:]):
                local_sorted = local_arr
                remote_sorted = remote_arr
                sizes_sorted = sizes_arr
            else:
                sort_idx = np.argsort(local_arr, kind="stable")
                local_sorted = local_arr[sort_idx]
                remote_sorted = remote_arr[sort_idx]
                sizes_sorted = sizes_arr[sort_idx]

        if n == 1:
            return (
                [int(local_sorted[0])],
                [int(remote_sorted[0])],
                [int(sizes_sorted[0])],
            )

        diff_local = local_sorted[1:] - local_sorted[:-1]
        diff_remote = remote_sorted[1:] - remote_sorted[:-1]
        prev_size = sizes_sorted[:-1]

        contiguous = (diff_local == prev_size) & (diff_remote == prev_size)

        if not contiguous.any():
            return local_sorted.tolist(), remote_sorted.tolist(), sizes_sorted.tolist()

        if contiguous.all():
            total_size = int(sizes_sorted.sum())
            return [int(local_sorted[0])], [int(remote_sorted[0])], [total_size]

        break_positions = np.flatnonzero(~contiguous) + 1
        segment_starts = np.concatenate(([0], break_positions))
        segment_ends = np.concatenate((break_positions, [n]))

        seg_count = len(segment_starts)
        merged_local = [0] * seg_count
        merged_remote = [0] * seg_count
        merged_sizes = [0] * seg_count

        for si in range(seg_count):
            s = segment_starts[si]
            e = segment_ends[si]
            merged_local[si] = int(local_sorted[s])
            merged_remote[si] = int(remote_sorted[s])

            merged_sizes[si] = int(
                local_sorted[e - 1] + sizes_sorted[e - 1] - local_sorted[s]
            )

        return merged_local, merged_remote, merged_sizes

    def _compute_block_transfer_offsets(
        self,
        layer_name: str,
        local_block_ids: list[int],
        remote_block_ids: list[int],
    ) -> tuple[list[int], list[int], list[int]]:
        """Compute transfer offsets for block data.

        Args:
            layer_name: Name of the layer to transfer
            local_block_ids: IDs of local blocks
            remote_block_ids: IDs of remote blocks

        Returns:
            Tuple of (local_offsets, remote_offsets, transfer_sizes)
        """
        assert self.kv_cache_shape is not None, "KV caches shape not initialized"
        is_mla = len(self.kv_cache_shape) == 3
        stride = self.kv_caches[layer_name].stride()
        sz = self.kv_caches[layer_name].element_size()
        if is_mla:
            blknum, blksize, hs = self.kv_cache_shape
            hn = 1
            block_stride = stride[0]
        else:
            _, blknum, blksize, hn, hs = self.kv_cache_shape
            ktov_stride = stride[0]
            block_stride = stride[1]

        transfer_size_byte = blksize * hn * hs * sz
        per_block = 1 if is_mla else 2
        total = len(local_block_ids) * per_block
        offset_local = [0] * total
        offset_remote = [0] * total
        sizes = [transfer_size_byte] * total

        w = 0
        for i, lb in enumerate(local_block_ids):
            rb = remote_block_ids[i]
            # K
            offset_local[w] = sz * (lb * block_stride)
            offset_remote[w] = sz * (rb * block_stride)
            w += 1
            if not is_mla:
                # V
                offset_local[w] = sz * (1 * ktov_stride + lb * block_stride)
                offset_remote[w] = sz * (1 * ktov_stride + rb * block_stride)
                w += 1

        merged_l, merged_r, merged_s = self.merge_contiguous_blocks(
            offset_local, offset_remote, sizes, assume_sorted=False
        )
        return merged_l, merged_r, merged_s

    def _read_blocks(
        self,
        local_block_ids: list[int],
        remote_block_ids: list[int],
        dst_engine_id: str,
        request_id: str,
        remote_host: str,
        remote_notify_port: int,
    ) -> None:
        if self.mode == MoRIIOMode.WRITE:
            return

        dp0_engine_id = self.get_engine_name_with_dp(dst_engine_id, 0)
        sessions = self._get_built_session(dp0_engine_id)

        first_layer = list(self.layer_name_to_local_kv_cache_metadata.keys())[0]
        offs = self._compute_block_transfer_offsets(
            first_layer, local_block_ids, remote_block_ids
        )

        for layer_name in self.layer_name_to_local_kv_cache_metadata:
            sess_idx = list(self.layer_name_to_local_kv_cache_metadata.keys()).index(
                layer_name
            )
            #TODO : apply multi-session batch-read when moriio support it
            transfer_status = self.moriio_wrapper.read_remote_data(
                offs[2], offs[0], offs[1], sessions[sess_idx]
            )

            self._recving_transfers[request_id].append(transfer_status)
            self._recving_transfers_callback_addr[request_id] = (
                remote_host,
                str(remote_notify_port + self.tp_rank),
            )


@contextlib.contextmanager
def zmq_ctx(socket_type: Any, addr: str) -> Iterator[zmq.Socket]:
    """Context manager for a ZMQ socket"""

    if socket_type not in (zmq.ROUTER, zmq.REQ, zmq.DEALER):
        raise ValueError(f"Unexpected socket type: {socket_type}")

    ctx: zmq.Context | None = None
    try:
        ctx = zmq.Context()  # type: ignore[attr-defined]
        yield make_zmq_socket(
            ctx=ctx, path=addr, socket_type=socket_type, bind=socket_type == zmq.ROUTER
        )
    finally:
        if ctx is not None:
            ctx.destroy(linger=0)
