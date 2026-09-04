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
            "[snapshot][kv-transfer] engine ID format not recognized; "
            "appending a new UUID: engine_id=%s",
            engine_id,
        )
        return f"{engine_id}-{uuid4().hex}"
    prefix = match.group(1)
    dp_suffix = match.group(3) or ""
    return f"{prefix}-{uuid4().hex}{dp_suffix}"


def refresh_scheduler_handshake_metadata_after_snapshot_restore(
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
