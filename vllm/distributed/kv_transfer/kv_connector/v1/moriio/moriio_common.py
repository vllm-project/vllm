# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import os
import threading
import time
from collections.abc import Collection, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

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


class MoRIIOTransferAck(NamedTuple):
    transfer_id: TransferId
    consumer_tp_size: int = 1


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
    session: Any | None = None
    use_batch: bool = True


@dataclass
class RemoteAllocInfo:
    """Information about remote block allocation."""

    block_ids: list[int]
    writes_done: int = 0
    writes_expected: int | None = None
    decode_dp_rank: int = 0
    completion_request_id: str | None = None
    completion_remote_notify_port: int | None = None
    completion_remote_ip: str | None = None
    completion_notified: bool = False
    transfer_statuses: list[Any] = field(default_factory=list)
    transfer_offsets: dict[
        tuple[tuple[int, ...], tuple[int, ...], torch.dtype],
        tuple[list[int], list[int], list[int]],
    ] = field(default_factory=dict)


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


def resolve_host_ip(extra_config: dict) -> str:
    """The IP this MoRIIO process advertises for KV transfer.

    Honors an explicit ``host_ip`` in ``kv_connector_extra_config`` before
    falling back to ``get_ip()``. An external router/orchestrator can set it to
    the node's routable address; this is required under frameworks (e.g. Ray)
    where ``get_ip()`` resolves to an unroutable public IP and ``VLLM_HOST_IP``
    cannot be propagated to the worker processes that bind the transfer engine.
    """
    return extra_config.get("host_ip") or get_ip()


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


def validate_moriio_trusted_host(
    remote_host: str,
    trusted_hosts: Collection[str] | None,
    source: str,
) -> None:
    trusted_host_list = _normalize_node_hosts(trusted_hosts, "trusted_remote_hosts")
    if not trusted_host_list:
        raise ValueError(
            f"MoRIIO {source} host {remote_host!r} is not trusted; configure "
            "kv_connector_extra_config['trusted_remote_hosts'] with trusted peer "
            "hosts"
        )
    if remote_host not in set(trusted_host_list):
        raise ValueError(
            f"MoRIIO {source} host {remote_host!r} is not in trusted peer hosts "
            f"{trusted_host_list!r}"
        )


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

    return [default_host]


def get_moriio_trusted_remote_hosts(
    kv_transfer_config: KVTransferConfig,
) -> list[str]:
    extra_config = kv_transfer_config.kv_connector_extra_config
    return _normalize_node_hosts(
        extra_config.get("trusted_remote_hosts"),
        "kv_connector_extra_config['trusted_remote_hosts']",
    )


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


def _get_non_negative_int_extra_config(
    extra_config: dict[str, Any],
    key: str,
    default: int,
) -> int:
    value = int(extra_config.get(key, default))
    if value < 0:
        raise ValueError(
            f"Invalid MoRIIO {key}={value}; expected a non-negative integer."
        )
    return value


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
    handshake_timeout: float = 10.0
    max_inflight_global: int = 0
    max_inflight_per_transfer: int = 0
    max_dispatch_layers: int = 0

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

        local_ip = resolve_host_ip(extra_config)

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
            handshake_timeout=float(
                extra_config.get(
                    "handshake_timeout", MoRIIOConstants.DEFAULT_HANDSHAKE_TIMEOUT
                )
            ),
            max_inflight_global=_get_non_negative_int_extra_config(
                extra_config,
                "max_inflight_global",
                0,
            ),
            max_inflight_per_transfer=_get_non_negative_int_extra_config(
                extra_config,
                "max_inflight_per_transfer",
                0,
            ),
            max_dispatch_layers=_get_non_negative_int_extra_config(
                extra_config,
                "max_dispatch_layers",
                0,
            ),
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
    # Timeout (seconds) for a single MoRIIO handshake metadata exchange.
    # Bounds an unresponsive remote listener so TP workers do not block in recv().
    # Overridable via kv_connector_extra_config["handshake_timeout"].
    DEFAULT_HANDSHAKE_TIMEOUT = 10.0


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


# vLLM appends a trailing "-<8hex>" engine-local suffix to the router
# request_id; it differs between the prefill and decode engines for the same
# logical request. The canonical (router) request_id is the stripped base,
# which both sides agree on — use it as a stable cross-node map key.
_VLLM_REQUEST_SUFFIX_RE = re.compile(r"(.+)-[0-9a-fA-F]{8}$")


def _strip_vllm_request_suffix(request_id: str) -> str:
    match = _VLLM_REQUEST_SUFFIX_RE.fullmatch(request_id)
    return match.group(1) if match is not None else request_id


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
        self.transfer_id_to_remote_tp_size: dict[TransferId, int] = {}
        self.freed_transfer_ids: set[TransferId] = set()

    def __repr__(self):
        return (
            f"MoRIIOConnectorMetadata: reqs_to_recv={self.reqs_to_recv}, "
            f"reqs_to_save={self.reqs_to_save}, "
            f"reqs_to_send={self.reqs_to_send}, "
            f"transfer_id_to_request_id={self.transfer_id_to_request_id}, "
            f"transfer_id_to_remote_tp_size={self.transfer_id_to_remote_tp_size}, "
            f"freed_transfer_ids={self.freed_transfer_ids}"
        )

    def add_new_req(
        self,
        request_id: ReqId,
        local_block_ids: list[int],
        kv_transfer_params: dict[str, Any],
        write_mode=False,
        trusted_remote_hosts: Collection[str] | None = None,
    ):
        transfer_id = kv_transfer_params["transfer_id"]

        remote_host = kv_transfer_params.get("remote_host")
        remote_handshake_port = kv_transfer_params.get("remote_handshake_port")
        remote_notify_port = kv_transfer_params.get("remote_notify_port")
        remote_host_source = "kv_transfer_params.remote_host"
        # Parse host/ports from request_id. The router normally embeds both
        # zmq_addresses there. If the embedded address is absent, fall back to
        # `kv_transfer_params.remote_zmq_address`, which carries the same
        # host/handshake/notify tuple explicitly. A host list alone is not
        # enough because deployments may use non-default ports.
        remote_hosts = _normalize_node_hosts(
            kv_transfer_params.get("remote_hosts"),
            "kv_transfer_params['remote_hosts']",
        )
        if (
            remote_host is None
            or remote_handshake_port is None
            or remote_notify_port is None
        ):
            try:
                peer_zmq = get_peer_zmq_from_request_id(
                    request_id, is_producer=write_mode
                )
                remote_host, remote_handshake_port, remote_notify_port = (
                    parse_moriio_zmq_address(peer_zmq)
                )
                remote_host_source = "request_id"
            except ValueError:
                peer_zmq = kv_transfer_params.get("remote_zmq_address")
                if peer_zmq:
                    remote_host, remote_handshake_port, remote_notify_port = (
                        parse_moriio_zmq_address(peer_zmq)
                    )
                    remote_host_source = "kv_transfer_params.remote_zmq_address"
                elif not remote_hosts:
                    raise ValueError(
                        f"MoRIIO add_new_req: could not resolve peer host/ports "
                        f"for {request_id!r}; neither request_id parse nor "
                        f"kv_transfer_params.remote_zmq_address provided them"
                    ) from None
                else:
                    raise ValueError(
                        f"MoRIIO add_new_req: kv_transfer_params.remote_hosts for "
                        f"{request_id!r} does not include handshake/notify ports; "
                        f"forward remote_zmq_address with the host list"
                    ) from None

        if trusted_remote_hosts is not None:
            validate_moriio_trusted_host(
                remote_host,
                trusted_remote_hosts,
                remote_host_source,
            )
            for remote_hosts_entry in remote_hosts:
                validate_moriio_trusted_host(
                    remote_hosts_entry,
                    trusted_remote_hosts,
                    "kv_transfer_params.remote_hosts",
                )

        # If remote block metadata is absent, use empty defaults so the request
        # remains a no-op.
        _req = ReqMeta(
            transfer_id=transfer_id,
            local_block_ids=local_block_ids,
            remote_block_ids=kv_transfer_params.get("remote_block_ids", []),
            remote_engine_id=kv_transfer_params.get("remote_engine_id", ""),
            remote_host=remote_host,
            remote_port=int(remote_handshake_port),
            remote_handshake_port=int(remote_handshake_port),
            remote_notify_port=int(remote_notify_port),
            # Callers may forward `remote_tp_size` without `tp_size`.
            # Prefer `tp_size`, then fall back to `remote_tp_size`, then 1.
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
