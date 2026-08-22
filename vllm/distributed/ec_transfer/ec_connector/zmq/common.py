# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared types and configuration for the ECZmqConnector."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorMetadata,
    ECConnectorWorkerMetadata,
)
from vllm.logger import init_logger
from vllm.utils.network_utils import make_zmq_path

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

DEFAULT_STAGING_BYTES = 4 * 1024**3
DEFAULT_STAGING_TTL_S = 300.0
DEFAULT_RECV_TIMEOUT_S = 60.0
DEFAULT_SEND_TIMEOUT_S = 30.0
DEFAULT_MAX_INFLIGHT_SENDS = 64


@dataclass(frozen=True)
class ZmqDst:
    """A consumer engine's receive endpoints.

    Every rank holding an encoder cache binds its own socket, so one engine is
    addressed as a contiguous port range starting at `port`.
    """

    host: str
    port: int
    num_ranks: int = 1

    def endpoints(self) -> list[str]:
        return [
            make_zmq_path("tcp", self.host, self.port + rank)
            for rank in range(self.num_ranks)
        ]

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "ZmqDst":
        """Build a destination from user-supplied config or request params.

        Raises:
            ValueError: `host` or `port` is missing, or `num_ranks` is not
                positive.
        """
        try:
            host = raw["host"]
            port = int(raw["port"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"An EC ZMQ destination needs 'host' and 'port', got {raw!r}"
            ) from exc
        num_ranks = int(raw.get("num_ranks", 1))
        if num_ranks < 1:
            raise ValueError(f"num_ranks must be >= 1, got {num_ranks}")
        return ZmqDst(host=host, port=port, num_ranks=num_ranks)


@dataclass
class ECZmqConnectorMetadata(ECConnectorMetadata):
    """Per-step scheduler -> worker payload for the ECZmqConnector."""

    # Producer role: mm_hash -> destinations to push the freshly computed
    # embedding to. Consumed by the worker's `save_caches`.
    sends: dict[str, list[ZmqDst]] = field(default_factory=dict)

    # Consumer role: mm_hashes whose embedding has landed in the worker's
    # staging area and must be moved into the GPU encoder cache this step.
    loads: list[str] = field(default_factory=list)


@dataclass
class ECZmqWorkerMetadata(ECConnectorWorkerMetadata):
    """Per-step worker -> scheduler report for the ECZmqConnector.

    `staged` counts how many ranks have an embedding in hand. The scheduler
    only treats an item as available once every rank has reported it, so a
    load can never be scheduled for a rank that is still waiting.
    """

    staged: dict[str, int] = field(default_factory=dict)

    def aggregate(self, other: "ECConnectorWorkerMetadata") -> "ECZmqWorkerMetadata":
        assert isinstance(other, ECZmqWorkerMetadata)
        merged = dict(self.staged)
        for mm_hash, count in other.staged.items():
            merged[mm_hash] = merged.get(mm_hash, 0) + count
        return ECZmqWorkerMetadata(staged=merged)


@dataclass(frozen=True)
class ECZmqOptions:
    """Resolved `ec_connector_extra_config` for one process."""

    # Consumer side.
    bind_host: str
    recv_port_base: int
    num_recv_ranks: int
    staging_bytes: int
    staging_ttl_s: float
    recv_timeout_s: float
    wait_for_all_remote: bool

    # Producer side.
    consumers: tuple[ZmqDst, ...]
    send_timeout_s: float
    max_inflight_sends: int


def parse_zmq_options(vllm_config: "VllmConfig") -> ECZmqOptions:
    """Resolve the ZMQ connector options from `vllm_config`.

    Raises:
        ValueError: `ec_transfer_config` is unset, or a configured destination
            is malformed.
    """
    ec_config = vllm_config.ec_transfer_config
    if ec_config is None:
        raise ValueError("ec_transfer_config must be set to use the ECZmqConnector")

    extra = ec_config.ec_connector_extra_config
    raw_consumers = extra.get("ec_zmq_consumers")
    if raw_consumers:
        consumers = tuple(ZmqDst.from_dict(raw) for raw in raw_consumers)
    else:
        # A single-consumer deployment needs no extra config: the base
        # ec_ip/ec_port pair already describes where to push.
        consumers = (ZmqDst(host=ec_config.ec_ip, port=ec_config.ec_port),)

    return ECZmqOptions(
        bind_host=extra.get("ec_zmq_bind_host", "0.0.0.0"),
        recv_port_base=recv_port_base(vllm_config),
        num_recv_ranks=num_recv_ranks(vllm_config),
        staging_bytes=int(extra.get("ec_zmq_staging_bytes", DEFAULT_STAGING_BYTES)),
        staging_ttl_s=float(extra.get("ec_zmq_staging_ttl_s", DEFAULT_STAGING_TTL_S)),
        recv_timeout_s=float(
            extra.get("ec_zmq_recv_timeout_s", DEFAULT_RECV_TIMEOUT_S)
        ),
        wait_for_all_remote=bool(extra.get("ec_zmq_wait_for_all_remote", False)),
        consumers=consumers,
        send_timeout_s=float(
            extra.get("ec_zmq_send_timeout_s", DEFAULT_SEND_TIMEOUT_S)
        ),
        max_inflight_sends=int(
            extra.get("ec_zmq_max_inflight_sends", DEFAULT_MAX_INFLIGHT_SENDS)
        ),
    )


def num_recv_ranks(vllm_config: "VllmConfig") -> int:
    """How many ranks of one engine hold an encoder cache.

    The encoder cache lives on the first pipeline stage only, replicated across
    its TP and prefill-CP ranks (DCP subdivides TP, so it adds no ranks).
    """
    parallel_config = vllm_config.parallel_config
    return (
        parallel_config.tensor_parallel_size
        * parallel_config.prefill_context_parallel_size
    )


def recv_port_base(vllm_config: "VllmConfig") -> int:
    """First receive port of this engine.

    Engines of a data-parallel deployment share one config, so each one claims
    its own `num_recv_ranks`-wide slice of the port space.
    """
    ec_config = vllm_config.ec_transfer_config
    assert ec_config is not None
    dp_rank = vllm_config.parallel_config.data_parallel_rank
    return ec_config.ec_port + dp_rank * num_recv_ranks(vllm_config)
