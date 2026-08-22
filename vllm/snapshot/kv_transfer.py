# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Protocol
from uuid import uuid4

import regex as re

from vllm.logger import init_logger

logger = init_logger(__name__)


class SnapshotEngine(Protocol):
    vllm_config: Any
    scheduler: Any
    model_executor: Any


def rotate_engine_id(engine_id: str) -> str:
    """Replace the runtime UUID while preserving the instance and DP rank."""
    match = re.match(r"^(.+)-([0-9a-f]{32})(_dp\d+)?$", engine_id)
    if match is None:
        logger.warning(
            "[snapshot][rebuild] engine_id %s does not match expected format; "
            "appending a new UUID",
            engine_id,
        )
        return f"{engine_id}-{uuid4().hex}"
    prefix = match.group(1)
    dp_suffix = match.group(3) or ""
    return f"{prefix}-{uuid4().hex}{dp_suffix}"


def refresh_scheduler_after_resume(
    engine_core: SnapshotEngine,
    local_ip: str,
) -> None:
    """Refresh scheduler-side KV transport identity and address state."""
    kv_config = engine_core.vllm_config.kv_transfer_config
    if kv_config is None or not (kv_config.is_kv_producer or kv_config.is_kv_consumer):
        return

    connector = engine_core.scheduler.connector
    connector_scheduler = (
        getattr(connector, "connector_scheduler", None) if connector else None
    )
    if connector_scheduler is None:
        return

    if hasattr(connector_scheduler, "side_channel_host"):
        old_host = connector_scheduler.side_channel_host
        connector_scheduler.side_channel_host = local_ip
        logger.info(
            "[snapshot][rebuild] scheduler side_channel_host %s->%s",
            old_host,
            local_ip,
        )

    if hasattr(connector_scheduler, "engine_id"):
        old_engine_id = str(connector_scheduler.engine_id)
        new_engine_id = rotate_engine_id(old_engine_id)
        connector_scheduler.engine_id = new_engine_id
        if hasattr(connector, "engine_id"):
            connector.engine_id = new_engine_id
        kv_config.engine_id = new_engine_id
        logger.info(
            "[snapshot][rebuild] scheduler engine_id %s->%s",
            old_engine_id,
            new_engine_id,
        )


def refresh_scheduler_handshake_metadata_after_resume(
    engine_core: SnapshotEngine,
) -> None:
    """Refresh scheduler per-rank KV endpoint mappings after worker rebuild."""
    kv_config = engine_core.vllm_config.kv_transfer_config
    if kv_config is None or not kv_config.is_kv_producer:
        return
    if "Layerwise" in (kv_config.kv_connector or ""):
        return

    kv_connector = engine_core.scheduler.get_kv_connector()
    if kv_connector is None:
        return

    handshake_metadata = (
        engine_core.model_executor.get_kv_connector_handshake_metadata()
    )
    if not handshake_metadata:
        return

    content: dict[tuple[int, int], Any] = {}
    for worker_metadata in handshake_metadata:
        if worker_metadata is not None:
            content.update(worker_metadata)
    kv_connector.set_xfer_handshake_metadata_pp_aware(content)
