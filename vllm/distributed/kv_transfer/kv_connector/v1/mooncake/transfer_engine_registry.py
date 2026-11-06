# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Process-local Mooncake TransferEngine shared by PD and store connectors.

``MooncakeConnector`` publishes the engine it initializes; ``MooncakeStoreConnector``
borrows it when ``reuse_pd_transfer_engine`` is set. One worker process holds at
most one ``MooncakeConnectorWorker``, and every access runs from the worker
lifecycle calls (connector construction, KV registration, shutdown), which are
serialized, so no locking is needed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


def reuse_pd_transfer_engine_requested(extra_config: Mapping[str, Any]) -> bool:
    """Whether ``kv_connector_extra_config`` asks the store to borrow the PD engine."""
    value = extra_config.get("reuse_pd_transfer_engine", False)
    return str(value).strip().lower() == "true"


@dataclass(frozen=True)
class SharedMooncakeTransferEngine:
    """TransferEngine published by the ``MooncakeConnector`` worker."""

    engine: Any
    hostname: str
    rpc_port: int
    protocol: str


_SHARED: SharedMooncakeTransferEngine | None = None


def register_shared_transfer_engine(
    *, engine: Any, hostname: str, rpc_port: int, protocol: str
) -> None:
    """Publish ``engine`` for this worker process, replacing a previous entry."""
    global _SHARED
    _SHARED = SharedMooncakeTransferEngine(engine, hostname, rpc_port, protocol)
    logger.info(
        "Published Mooncake TransferEngine at %s:%s (%s).",
        hostname,
        rpc_port,
        protocol,
    )


def get_shared_transfer_engine() -> SharedMooncakeTransferEngine | None:
    """Return the engine published in this process, if any."""
    return _SHARED


def clear_shared_transfer_engine(engine: Any) -> None:
    """Drop the entry when it still points at ``engine``."""
    global _SHARED
    if _SHARED is not None and _SHARED.engine is engine:
        _SHARED = None
