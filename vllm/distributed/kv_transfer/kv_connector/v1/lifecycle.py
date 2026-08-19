# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Privacy-safe lifecycle events for KV connectors.

The logging sink is intentionally opt-in and stateless. Request identifiers
are hashed before they leave the process, and deterministic sampling groups
request IDs that differ only by vLLM's trailing random suffix.
"""

import hashlib
import math
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Literal

import msgspec
import regex as re

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

KV_CONNECTOR_LIFECYCLE_TRACE_SAMPLE_RATE = "kv_connector_lifecycle_trace_sample_rate"
KV_CONNECTOR_LIFECYCLE_LOG_PREFIX = "KV_CONNECTOR_LIFECYCLE "
_SCHEMA = "vllm-kv-connector-v1"
_RANDOM_SUFFIX_RE = re.compile(r"-[0-9a-f]{8}$", re.IGNORECASE)

KVConnectorLifecycleComponent = Literal["scheduler", "worker"]
KVConnectorLifecycleRole = Literal["producer", "consumer"]


class KVConnectorLifecycleEvent(str, Enum):
    """Connector transitions shared by control-plane and data-plane adapters."""

    TRANSFER_STAGED = "transfer_staged"
    REGISTRATION_STAGED = "registration_staged"
    REGISTRATION_QUEUED = "registration_queued"
    REGISTRATION_SENT = "registration_sent"
    REGISTRATION_RECEIVED = "registration_received"
    REGISTRATION_FAILED = "registration_failed"
    BLOCKS_MATCHED = "blocks_matched"
    TRANSFER_STARTED = "transfer_started"
    TRANSFER_COMPLETED = "transfer_completed"
    TRANSFER_FAILED = "transfer_failed"


class _KVConnectorLifecycleLogRecord(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,  # type: ignore[call-arg]
):
    schema: str
    event: KVConnectorLifecycleEvent
    wall_ns: int
    monotonic_ns: int
    connector: str
    transfer_mode: str
    component: KVConnectorLifecycleComponent
    role: KVConnectorLifecycleRole
    request_id_hash: str
    request_group_id_hash: str
    remote_request_id_hash: str | None = None
    block_count: int | None = None


def _request_group_id(request_id: str) -> str:
    return _RANDOM_SUFFIX_RE.sub("", request_id)


def _hash_request_id(request_id: str) -> str:
    return hashlib.blake2b(
        request_id.encode("utf-8"),
        digest_size=8,
        person=b"vllm-kv-trace",
        usedforsecurity=False,
    ).hexdigest()


def _is_sampled(request_id: str, sample_rate: float) -> bool:
    if sample_rate <= 0.0:
        return False
    if sample_rate >= 1.0:
        return True
    digest = hashlib.blake2b(
        _request_group_id(request_id).encode("utf-8"),
        digest_size=8,
        person=b"vllm-kv-sample",
        usedforsecurity=False,
    ).digest()
    return int.from_bytes(digest, "big") / (1 << 64) < sample_rate


class KVConnectorLifecycleSink(ABC):
    """Typed sink implemented by connector lifecycle exporters."""

    @abstractmethod
    def emit(
        self,
        event: KVConnectorLifecycleEvent,
        request_id: str,
        *,
        role: KVConnectorLifecycleRole,
        remote_request_id: str | None = None,
        block_count: int | None = None,
    ) -> None:
        """Emit a single request transition."""


class NoOpKVConnectorLifecycleSink(KVConnectorLifecycleSink):
    """Disabled sink used to avoid conditionals in connector adapters."""

    def emit(
        self,
        event: KVConnectorLifecycleEvent,
        request_id: str,
        *,
        role: KVConnectorLifecycleRole,
        remote_request_id: str | None = None,
        block_count: int | None = None,
    ) -> None:
        return


class LoggingKVConnectorLifecycleSink(KVConnectorLifecycleSink):
    """Emit sampled, structured lifecycle events through the vLLM logger."""

    def __init__(
        self,
        *,
        connector: str,
        transfer_mode: str,
        component: KVConnectorLifecycleComponent,
        sample_rate: float,
    ) -> None:
        if not math.isfinite(sample_rate) or not 0.0 <= sample_rate <= 1.0:
            raise ValueError(
                f"{KV_CONNECTOR_LIFECYCLE_TRACE_SAMPLE_RATE} must be between "
                f"0.0 and 1.0, got {sample_rate}"
            )
        self._connector = connector
        self._transfer_mode = transfer_mode
        self._component = component
        self._sample_rate = sample_rate

    def emit(
        self,
        event: KVConnectorLifecycleEvent,
        request_id: str,
        *,
        role: KVConnectorLifecycleRole,
        remote_request_id: str | None = None,
        block_count: int | None = None,
    ) -> None:
        if not _is_sampled(request_id, self._sample_rate):
            return

        group_id = _request_group_id(request_id)
        record = _KVConnectorLifecycleLogRecord(
            schema=_SCHEMA,
            event=event,
            wall_ns=time.time_ns(),
            monotonic_ns=time.monotonic_ns(),
            connector=self._connector,
            transfer_mode=self._transfer_mode,
            component=self._component,
            role=role,
            request_id_hash=_hash_request_id(request_id),
            request_group_id_hash=_hash_request_id(group_id),
            remote_request_id_hash=(
                _hash_request_id(remote_request_id)
                if remote_request_id is not None
                else None
            ),
            block_count=block_count,
        )
        logger.info(
            "%s%s",
            KV_CONNECTOR_LIFECYCLE_LOG_PREFIX,
            msgspec.json.encode(record).decode("utf-8"),
        )


_NO_OP_SINK = NoOpKVConnectorLifecycleSink()


def create_kv_connector_lifecycle_sink(
    vllm_config: "VllmConfig",
    *,
    transfer_mode: str,
    component: KVConnectorLifecycleComponent,
) -> KVConnectorLifecycleSink:
    """Create the configured connector lifecycle sink.

    ``sample_rate=0`` is the default and returns a shared stateless no-op sink.
    """
    config = vllm_config.kv_transfer_config
    if config is None:
        return _NO_OP_SINK

    raw_sample_rate = config.get_from_extra_config(
        KV_CONNECTOR_LIFECYCLE_TRACE_SAMPLE_RATE, 0.0
    )
    try:
        sample_rate = float(raw_sample_rate)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{KV_CONNECTOR_LIFECYCLE_TRACE_SAMPLE_RATE} must be a number, "
            f"got {raw_sample_rate!r}"
        ) from exc
    if sample_rate == 0.0:
        return _NO_OP_SINK

    return LoggingKVConnectorLifecycleSink(
        connector=config.kv_connector or "unknown",
        transfer_mode=transfer_mode,
        component=component,
        sample_rate=sample_rate,
    )
