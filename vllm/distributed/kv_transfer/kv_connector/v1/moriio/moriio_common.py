# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import os
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

import msgspec
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


def get_moriio_mode(kv_transfer_config: KVTransferConfig) -> MoRIIOMode:
    read_mode = str(
        kv_transfer_config.kv_connector_extra_config.get("read_mode", "false")
    ).lower().strip() in ("true", "1")
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
        # For per-node port allocation we want the local dp_rank within
        # the node, not the global one. data_parallel_rank is global, so
        # fold it back to [0, data_parallel_size_local).
        dp_rank = (
            vllm_config.parallel_config.data_parallel_rank
            % vllm_config.parallel_config.data_parallel_size_local
        )
        base_notify_port = int(extra_config["notify_port"])
        dp_size = vllm_config.parallel_config.data_parallel_size_local
        tp_size = get_tensor_model_parallel_world_size()
        port_offset = get_port_offset(dp_rank, tp_rank)
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

        return cls(
            local_ip=resolve_host_ip(extra_config),
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


def get_peer_zmq_from_request_id(
    request_id: str, is_producer: bool
) -> str | None:
    """Extract the *peer's* zmq_address from the vLLM router request_id.

    The producer (prefill) needs the decode's address; the consumer (decode)
    needs the prefill's address.

    Returns ``None`` when the request_id does not encode peer info. The
    llm-d routing sidecar (``llm-d-inference-scheduler``) does not embed
    addresses in ``request_id``; instead it passes ``remote_host``,
    ``remote_handshake_port`` and ``remote_notify_port`` explicitly in
    ``kv_transfer_params``. Callers must handle the ``None`` return by
    falling back to those fields. See ``add_new_req`` for the canonical
    fallback path.
    """
    if is_producer:
        m = _DECODE_ZMQ_RE.search(request_id)
    else:
        m = _PREFILL_ZMQ_RE.search(request_id)
    if m is None:
        return None
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
    # Wide-EP multi-pod support: list of remote pod IPs to address when
    # the remote DP fan-out is split across more than one pod. Indexed by
    # ``remote_global_dp_rank // remote_dp_size_local`` -- i.e.
    # ranks 0..dp_local-1 live on remote_hosts[0], ranks dp_local..2*dp_local-1
    # live on remote_hosts[1], and so on. For single-pod deployments the
    # sidecar may either emit a 1-element list or leave the field empty, in
    # which case add_new_req falls back to ``[remote_host]`` so behaviour is
    # bit-identical to the single-pod path.
    remote_hosts: list[str] = field(default_factory=list)
    # The remote DP-size-local. When ``remote_dp_size_local == 0`` callers
    # must treat it as "fall back to remote_dp_size" (single-pod).
    remote_dp_size_local: int = 0


class MoRIIOConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        self.reqs_to_recv: dict[ReqId, ReqMeta] = {}
        self.reqs_to_save: dict[ReqId, ReqMeta] = {}
        self.reqs_to_send: dict[ReqId, float] = {}
        self.transfer_id_to_request_id: dict[TransferId, ReqId] = {}

    def __repr__(self):
        return (
            f"MoRIIOConnectorMetadata: reqs_to_recv={self.reqs_to_recv}, "
            f"reqs_to_save={self.reqs_to_save}, "
            f"reqs_to_send={self.reqs_to_send}, "
            f"transfer_id_to_request_id={self.transfer_id_to_request_id}"
        )

    def add_new_req(
        self,
        request_id: ReqId,
        local_block_ids: list[int],
        kv_transfer_params: dict[str, Any],
        write_mode=False,
    ):
        transfer_id = kv_transfer_params["transfer_id"]

        # Parse host/ports from the request_id. The vLLM router embeds both
        # zmq_addresses in the request_id; the llm-d routing sidecar does
        # not, and instead populates ``remote_host``, ``remote_handshake_port``
        # and ``remote_notify_port`` directly in ``kv_transfer_params``. Try
        # the request_id form first for backwards compatibility and fall back
        # to the explicit fields for sidecar-driven deployments.
        peer_zmq = get_peer_zmq_from_request_id(request_id, is_producer=write_mode)
        if peer_zmq is not None:
            remote_host, remote_handshake_port, remote_notify_port = (
                parse_moriio_zmq_address(peer_zmq)
            )
        else:
            try:
                remote_host = kv_transfer_params["remote_host"]
                remote_handshake_port = int(
                    kv_transfer_params["remote_handshake_port"]
                )
                remote_notify_port = int(kv_transfer_params["remote_notify_port"])
            except (KeyError, TypeError, ValueError) as e:
                raise ValueError(
                    f"request_id {request_id!r} does not embed a peer "
                    f"zmq_address and kv_transfer_params is missing one or "
                    f"more sidecar-fallback keys (need remote_host, "
                    f"remote_handshake_port, remote_notify_port): {e}"
                ) from e
            if not remote_host:
                raise ValueError(
                    f"request_id {request_id!r} does not embed a peer "
                    f"zmq_address and kv_transfer_params['remote_host'] is "
                    f"empty; cannot route MoRI-IO transfer"
                )

        # Wide-EP multi-pod support: derive the remote_hosts list from
        # ``kv_transfer_params["remote_hosts"]`` (a list of pod IPs the
        # router can address) with a fallback to ``[remote_host]`` so the
        # single-pod path is unchanged. ``remote_dp_size_local`` tells the
        # consumer how to map a global DP rank back to a pod index.
        _remote_hosts = kv_transfer_params.get("remote_hosts") or [remote_host]
        if not isinstance(_remote_hosts, list):
            _remote_hosts = [str(_remote_hosts)]
        _remote_dp_size_local = int(
            kv_transfer_params.get(
                "remote_dp_size_local",
                kv_transfer_params.get("remote_dp_size", 1),
            )
        )

        _req = ReqMeta(
            transfer_id=transfer_id,
            local_block_ids=local_block_ids,
            remote_block_ids=kv_transfer_params["remote_block_ids"],
            remote_engine_id=kv_transfer_params["remote_engine_id"],
            remote_host=remote_host,
            remote_port=int(remote_handshake_port),
            remote_handshake_port=int(remote_handshake_port),
            remote_notify_port=int(remote_notify_port),
            tp_size=kv_transfer_params.get("tp_size", 1),
            remote_dp_size=kv_transfer_params.get("remote_dp_size", 1),
            remote_hosts=[str(h) for h in _remote_hosts],
            remote_dp_size_local=_remote_dp_size_local,
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
