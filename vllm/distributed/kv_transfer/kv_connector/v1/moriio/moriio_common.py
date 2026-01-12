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
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.utils.network_utils import (
    get_ip,
    get_open_port,
    make_zmq_socket,
)

if TYPE_CHECKING:
    pass

from dataclasses import field
from enum import Enum

logger = init_logger(__name__)


Transfer = tuple[int, float]
EngineId = str
ReqId = str


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
    read_mode = envs.VLLM_MORIIO_CONNECTOR_READ_MODE
    logger.debug("MoRIIO Connector read_mode: %s", read_mode)
    if read_mode:
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

        # TODO : merge notify_port and handshake_port to simplify port management
        #        supports non-contiguous ports
        assert vllm_config.kv_transfer_config is not None, (
            "kv_transfer_config must be set for MoRIIOConnector"
        )
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


class MoRIIOConstants:
    """Constants for MoRIIO connector."""

    # ZMQ message types
    GET_META_MSG = b"get_meta_msg"
    POP_DONE_RECV = b"pop_done_recv"
    OVER = b"OVER"
    COMPLETION_PREFIX = "cmpl"

    PING_INTERVAL = 5
    MAX_PING_RETRIES = 100
    DEFAULT_HANDSHAKE_PORT = "6301"
    DEFAULT_NOTIFY_PORT = "61005"

    VLLM_MORI_READ_ABORT_REQUEST_TIMEOUT = 3600


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
