# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import os
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import msgspec
import psutil
import regex as re
import torch
import zmq

from vllm.config import KVTransferConfig, VllmConfig
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
    is_valid_ipv6_address,
    split_zmq_path,
)

if TYPE_CHECKING:
    pass

from dataclasses import field
from enum import Enum

logger = init_logger(__name__)


Transfer = tuple[int, float]
EngineId = str
ReqId = str
TransferId = str


@dataclass
class WriteTask:
    request_id: ReqId
    transfer_id: TransferId
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

    request_id: ReqId
    transfer_id: TransferId
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

    _instance: "RoleManager | None" = None
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


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def get_moriio_mode(kv_transfer_config: KVTransferConfig) -> MoRIIOMode:
    read_mode = _as_bool(
        kv_transfer_config.kv_connector_extra_config.get("read_mode", False)
    )
    logger.debug("MoRIIO Connector read_mode: %s", read_mode)
    if read_mode:
        return MoRIIOMode.READ
    else:
        return MoRIIOMode.WRITE


def get_port_offset(dp_rank: int, tp_rank: int, tp_size: int = 1) -> int:
    return (dp_rank) * tp_size + tp_rank


def _normalize_node_hosts(value: Any, config_key: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [host.strip() for host in value.split(",") if host.strip()]
    try:
        return [str(host).strip() for host in value if str(host).strip()]
    except TypeError as exc:
        raise ValueError(
            f"{config_key} must be a comma-separated string or iterable of hosts"
        ) from exc


def get_moriio_node_hosts(
    kv_transfer_config: KVTransferConfig, default_host: str
) -> list[str]:
    extra_config = kv_transfer_config.kv_connector_extra_config
    node_hosts = _normalize_node_hosts(
        extra_config.get("node_hosts"),
        "kv_connector_extra_config['node_hosts']",
    )
    if node_hosts:
        return node_hosts

    env_node_hosts = os.environ.get("VLLM_MORIIO_NODE_HOSTS", "").strip()
    if env_node_hosts:
        logger.warning_once(
            "The environment variable %s is deprecated. Set %r inside "
            "kv_transfer_config.kv_connector_extra_config instead.",
            "VLLM_MORIIO_NODE_HOSTS",
            "node_hosts",
        )
        node_hosts = _normalize_node_hosts(env_node_hosts, "VLLM_MORIIO_NODE_HOSTS")
        if node_hosts:
            return node_hosts

    return [default_host]


_DEPRECATED_ENV_VARS: dict[str, str] = {
    "VLLM_MORIIO_CONNECTOR_READ_MODE": "read_mode",
    "VLLM_MORIIO_QP_PER_TRANSFER": "qp_per_transfer",
    "VLLM_MORIIO_POST_BATCH_SIZE": "post_batch_size",
    "VLLM_MORIIO_NUM_WORKERS": "num_workers",
}


def _warn_deprecated_env_vars() -> None:
    for env_var, new_key in _DEPRECATED_ENV_VARS.items():
        if env_var in os.environ:
            logger.warning_once(
                "The environment variable %s is deprecated and ignored. "
                "Set %r inside kv_transfer_config.kv_connector_extra_config "
                "instead.",
                env_var,
                new_key,
            )


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
    transfer_timeout: float
    defer_timeout: float
    read_mode: bool = False
    qp_per_transfer: int = 1
    post_batch_size: int = -1
    num_workers: int = 1
    backend: str = "rdma"
    node_hosts: list[str] = field(default_factory=list)

    @classmethod
    def from_vllm_config(cls, vllm_config: VllmConfig) -> "MoRIIOConfig":
        # Port Configuration:
        # local_ping_port   -> Outgoing heartbeat to proxy
        # proxy_ping_port   -> Remote proxy's heartbeat ingress port
        # http_port         -> Instance's HTTP service endpoint
        # local_kv_port     -> service port for mori engine
        # notify_port       -> For synchronizing stages between prefill and decode
        # handshake_port    -> For initial handshake between mori engine

        # Optional tuning knobs
        # read_mode        -> If true, run the connector in READ mode (consumer
        #                     pulls KV from producer) instead of the default
        #                     WRITE mode.
        # transfer_timeout -> Timeout for waiting_for_transfer_complete before
        #                     raising TransferError (sec).
        # defer_timeout    -> Timeout before a deferred send with no finished_sending
        #                     notification is reaped and its blocks force-freed (sec).

        # Knobs for RDMA transfers, ignored if on xgmi backend
        # qp_per_transfer  -> Number of RDMA Queue Pairs per KV transfer.
        # post_batch_size  -> Batch size for posting transfer work requests
        #                     (-1 lets the MoRI backend choose).
        # num_workers      -> Number of background worker threads the MoRI
        #                     engine uses for transfer processing.

        # TODO : merge notify_port and handshake_port to simplify port management
        #        supports non-contiguous ports
        assert vllm_config.kv_transfer_config is not None, (
            "kv_transfer_config must be set for MoRIIOConnector"
        )
        _warn_deprecated_env_vars()
        kv_transfer_config = vllm_config.kv_transfer_config
        extra_config = kv_transfer_config.kv_connector_extra_config
        tp_rank = get_tensor_model_parallel_rank()
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        base_notify_port = int(extra_config["notify_port"])
        dp_size = vllm_config.parallel_config.data_parallel_size
        tp_size = get_tensor_model_parallel_world_size()
        port_offset = get_port_offset(dp_rank, tp_rank, tp_size)
        backend = str(extra_config.get("backend", "rdma")).lower()
        if backend not in ("rdma", "xgmi"):
            raise ValueError(
                f"Invalid MoRIIO backend {backend!r} in kv_connector_extra_config; "
                "must be one of 'rdma' or 'xgmi'."
            )

        transfer_timeout = float(
            extra_config.get(
                "transfer_timeout", MoRIIOConstants.DEFAULT_TRANSFER_TIMEOUT
            )
        )
        defer_timeout = float(
            extra_config.get("defer_timeout", MoRIIOConstants.DEFAULT_DEFER_TIMEOUT)
        )

        local_ip = get_ip()

        return cls(
            local_ip=local_ip,
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
            read_mode=get_moriio_mode(kv_transfer_config) == MoRIIOMode.READ,
            qp_per_transfer=int(extra_config.get("qp_per_transfer", 1)),
            post_batch_size=int(extra_config.get("post_batch_size", -1)),
            num_workers=int(extra_config.get("num_workers", 1)),
            backend=backend,
            node_hosts=get_moriio_node_hosts(kv_transfer_config, local_ip),
            transfer_timeout=transfer_timeout,
            defer_timeout=defer_timeout,
        )


class MoRIIOConstants:
    """Constants for MoRIIO connector."""

    # ZMQ message types
    GET_META_MSG = b"get_meta_msg"
    POP_DONE_RECV = b"pop_done_recv"
    OVER = b"OVER"
    COMPLETION_PREFIX = "cmpl"
    TRANSFER_PREFIX = "tx"

    PING_INTERVAL = 3
    MAX_PING_RETRIES = 100
    DEFAULT_HANDSHAKE_PORT = "6301"
    DEFAULT_NOTIFY_PORT = "61005"

    VLLM_MORI_READ_ABORT_REQUEST_TIMEOUT = 3600

    # Timeout (seconds) for waiting_for_transfer_complete before raising TransferError.
    # Overridable via kv_connector_extra_config["transfer_timeout"].
    DEFAULT_TRANSFER_TIMEOUT = 30.0
    # Timeout (seconds) before a deferred send with no finished_sending
    # notification is reaped and its blocks force-freed.
    # Overridable via kv_connector_extra_config["defer_timeout"].
    DEFAULT_DEFER_TIMEOUT = 60.0
    # Grace period (seconds) during which has_pending_deferred_sends keeps
    # scheduler probes active after a deferred send is created.
    # Overridable via kv_connector_extra_config["defer_drain_grace"].
    DEFAULT_DEFER_DRAIN_GRACE = 2.0


# The router embeds both zmq_addresses in the request_id:
#   "___prefill_addr_{zmq}___decode_addr_{zmq}_{32-hex-uuid}"
# MoRIIO zmq_address format: "host:IP,handshake:PORT,notify:PORT"
#
# This lets each connector side parse the peer's connection info without
# requiring the router to pass it explicitly in kv_transfer_params.
_PREFILL_ZMQ_RE = re.compile(r"___prefill_addr_(.+?)___decode_addr_")
# vLLM wraps the router's X-Request-Id as "cmpl-<id>-<seq>-<hex>" so there may
# be a trailing "-<seq>-<hex>" suffix after the 32-char UUID.  Allow it.
_DECODE_ZMQ_RE = re.compile(r"___decode_addr_(.+)_[0-9a-f]{32}(?:-.*)?$")


def parse_moriio_zmq_address(
    zmq_address: str,
) -> tuple[str, int, int]:
    """Parse the MoRI-IO zmq address into its components.

    Parses ``"host:IP,handshake:PORT,notify:PORT"`` into
        (host, handshake_port, notify_port).

    Each key-value pair is split on the *first* colon so that IPv6 addresses
    (e.g. ``host:::1``) are handled correctly.  Raises ``ValueError`` if any
    of ``host``, ``handshake``, or ``notify`` keys are absent or if the port
    values are non-numeric.
    """
    parts: dict[str, str] = {}
    for segment in zmq_address.split(","):
        key, _, val = segment.partition(":")
        parts[key.strip()] = val.strip()
    try:
        host = parts["host"]
        handshake_port = int(parts["handshake"])
        notify_port = int(parts["notify"])
    except (KeyError, ValueError) as e:
        raise ValueError(
            f"Malformed zmq_address {zmq_address!r}: expected "
            f"'host:IP,handshake:PORT,notify:PORT' format"
        ) from e
    return host, handshake_port, notify_port


def get_peer_zmq_from_request_id(request_id: str, is_producer: bool) -> str:
    """Extract the *peer's* zmq_address from the vLLM router request_id.

    The producer (prefill) needs the decode's address; the consumer (decode)
    needs the prefill's address.
    """
    if is_producer:
        m = _DECODE_ZMQ_RE.search(request_id)
    else:
        m = _PREFILL_ZMQ_RE.search(request_id)
    if m is None:
        raise ValueError(
            f"Cannot parse peer zmq_address from request_id: {request_id!r}"
        )
    return m.group(1)


@dataclass
class ReqMeta:
    """Metadata for a single request."""

    transfer_id: TransferId
    local_block_ids: list[int]
    remote_block_ids: list[int]
    remote_host: str
    remote_port: int
    remote_handshake_port: int
    remote_notify_port: int
    remote_engine_id: str
    tp_size: int
    remote_dp_size: int
    # DP rank that handled the prefill on the remote side. Proxy sets this
    # from `selected_prefill_dp_rank` in kv_transfer_params. Used by
    # `_read_blocks` to pick the correct per-rank session/MR instead of
    # always reading from remote DP0 (which mismatches num_blocks across
    # ranks and overflows remote DP0's MR at high concurrency).
    remote_dp_rank: int = 0
    # Ordered list of all prefill-instance host IPs for multi-node TP.
    # Each decode worker picks remote_hosts[tp_rank // ranks_per_node] as its
    # actual peer host for handshake + post-transfer notify. None or len<=1
    # falls back to single-host behaviour (remote_host).
    remote_hosts: list[str] | None = None


class MoRIIOConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        self.reqs_to_recv: dict[ReqId, ReqMeta] = {}
        self.reqs_to_save: dict[ReqId, ReqMeta] = {}
        self.reqs_to_send: dict[ReqId, float] = {}
        self.transfer_id_to_request_id: dict[TransferId, ReqId] = {}
        self.freed_transfer_ids: set[TransferId] = set()

    def __repr__(self):
        return (
            f"MoRIIOConnectorMetadata: reqs_to_recv={self.reqs_to_recv}, "
            f"reqs_to_save={self.reqs_to_save}, "
            f"reqs_to_send={self.reqs_to_send}, "
            f"transfer_id_to_request_id={self.transfer_id_to_request_id}, "
            f"freed_transfer_ids={self.freed_transfer_ids}"
        )

    def add_new_req(
        self,
        request_id: ReqId,
        local_block_ids: list[int],
        kv_transfer_params: dict[str, Any],
        write_mode=False,
    ):
        transfer_id = kv_transfer_params["transfer_id"]

        # Parse host/ports from the request_id. The router embeds both zmq_addresses
        # in the request_id. Some cleanup-path requests (e.g. those that hit the
        # external prefix cache and take the "aborted before scheduling" branch
        # in `request_finished`) surface a short internal request_id without
        # the proxy's `___prefill_addr_...` prefix; `get_peer_zmq_from_request_id`
        # raises ValueError on those. Fall back to `kv_transfer_params.remote_hosts`
        # (forwarded for multi-node TP) and MoRIIO's default well-known ports.
        # Only the request_id parse path is affected; once we have a valid
        # host/port triple we proceed as before.
        remote_hosts = _normalize_node_hosts(
            kv_transfer_params.get("remote_hosts"),
            "kv_transfer_params['remote_hosts']",
        )
        try:
            peer_zmq = get_peer_zmq_from_request_id(request_id, is_producer=write_mode)
            remote_host, remote_handshake_port, remote_notify_port = (
                parse_moriio_zmq_address(peer_zmq)
            )
        except ValueError:
            # Normalize remote_hosts: callers may pass a list (per-rank host
            # vector from the proxy) or a single host string from older
            # proxy versions. A bare string would silently slice into "1."
            # for "172.30.0.1" if we just did remote_hosts[0].
            if not remote_hosts:
                raise ValueError(
                    f"MoRIIO add_new_req: could not resolve peer host/ports "
                    f"for {request_id!r}; neither request_id parse nor "
                    f"kv_transfer_params.remote_hosts provided them"
                ) from None
            remote_host = remote_hosts[0]
            remote_handshake_port = int(MoRIIOConstants.DEFAULT_HANDSHAKE_PORT)
            remote_notify_port = int(MoRIIOConstants.DEFAULT_NOTIFY_PORT)

        # Cleanup-path requests (the same "aborted before scheduling" branch
        # that surfaces a short request_id) can also omit remote_block_ids
        # and remote_engine_id. Use .get() defaults so we don't crash the
        # same EngineCore the request_id fallback above is meant to keep
        # alive — an empty block list is the correct no-op for an aborted
        # request.
        _req = ReqMeta(
            transfer_id=transfer_id,
            local_block_ids=local_block_ids,
            remote_block_ids=kv_transfer_params.get("remote_block_ids", []),
            remote_engine_id=kv_transfer_params.get("remote_engine_id", ""),
            remote_host=remote_host,
            remote_port=remote_handshake_port,
            remote_handshake_port=remote_handshake_port,
            remote_notify_port=remote_notify_port,
            # Defense-in-depth: callers (e.g. moriio_toy_proxy_server in
            # READ-mode multi-node TP) may forward `remote_tp_size` only.
            # Read `tp_size` first, fall back to `remote_tp_size`, default 1.
            tp_size=kv_transfer_params.get(
                "tp_size",
                kv_transfer_params.get("remote_tp_size", 1),
            ),
            remote_dp_size=kv_transfer_params.get("remote_dp_size", 1),
            remote_dp_rank=int(kv_transfer_params.get("remote_dp_rank", 0) or 0),
            remote_hosts=remote_hosts or None,
        )
        if write_mode:
            self.reqs_to_save[request_id] = _req
        else:
            self.reqs_to_recv[request_id] = _req


_MORIIO_ZMQ_KEEPALIVE_OPTS = (
    (zmq.TCP_KEEPALIVE, 1),
    (zmq.TCP_KEEPALIVE_IDLE, 30),
    (zmq.TCP_KEEPALIVE_INTVL, 10),
    (zmq.TCP_KEEPALIVE_CNT, 3),
)


def _make_moriio_zmq_socket(
    ctx: zmq.Context,
    path: str,
    socket_type: Any,
    *,
    identity: bytes | None = None,
    linger: int | None = None,
    router_handover: bool = False,
) -> zmq.Socket:
    mem = psutil.virtual_memory()
    total_mem = mem.total / 1024**3
    available_mem = mem.available / 1024**3
    buf_size = int(0.5 * 1024**3) if total_mem > 32 and available_mem > 16 else -1

    sock = ctx.socket(socket_type)

    if socket_type in (zmq.PULL, zmq.DEALER, zmq.ROUTER):
        sock.setsockopt(zmq.RCVHWM, 0)
        sock.setsockopt(zmq.RCVBUF, buf_size)

    if socket_type in (zmq.PUSH, zmq.DEALER, zmq.ROUTER):
        sock.setsockopt(zmq.SNDHWM, 0)
        sock.setsockopt(zmq.SNDBUF, buf_size)

    if socket_type == zmq.ROUTER and router_handover:
        sock.setsockopt(zmq.ROUTER_HANDOVER, 1)

    if identity is not None:
        sock.setsockopt(zmq.IDENTITY, identity)

    if linger is not None:
        sock.setsockopt(zmq.LINGER, linger)

    # Mgmt-rail notify/handshake streams are sparse; a silently dead TCP peer
    # can park reads until the retransmit ladder expires. Keepalive applies to
    # accepted connections too and must be set before bind/connect.
    for option, value in _MORIIO_ZMQ_KEEPALIVE_OPTS:
        sock.setsockopt(option, value)

    scheme, host, _ = split_zmq_path(path)
    if scheme == "tcp" and is_valid_ipv6_address(host):
        sock.setsockopt(zmq.IPV6, 1)

    if socket_type == zmq.ROUTER:
        sock.bind(path)
    else:
        sock.connect(path)

    return sock


@contextlib.contextmanager
def zmq_ctx(
    socket_type: Any,
    addr: str,
    *,
    identity: bytes | None = None,
    linger: int | None = None,
    router_handover: bool = False,
) -> Iterator[zmq.Socket]:
    """Context manager for a ZMQ socket"""

    if socket_type not in (zmq.ROUTER, zmq.REQ, zmq.DEALER):
        raise ValueError(f"Unexpected socket type: {socket_type}")

    ctx: zmq.Context | None = None
    try:
        ctx = zmq.Context()  # type: ignore[attr-defined]
        sock = _make_moriio_zmq_socket(
            ctx,
            addr,
            socket_type,
            identity=identity,
            linger=linger,
            router_handover=router_handover,
        )
        yield sock
    finally:
        if ctx is not None:
            ctx.destroy(linger=0)
