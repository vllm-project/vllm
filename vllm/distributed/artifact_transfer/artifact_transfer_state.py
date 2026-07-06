# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.distributed.artifact_transfer.artifact_connector.base import (
    ArtifactConnectorBaseType,
)
from vllm.distributed.artifact_transfer.artifact_connector.factory import (
    ArtifactConnectorFactory,
)
from vllm.distributed.artifact_transfer.artifact_connector.v1 import (
    ArtifactConnectorRole,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

_ARTIFACT_CONNECTOR_AGENT: ArtifactConnectorBaseType | None = None


def get_artifact_transfer_group() -> ArtifactConnectorBaseType:
    assert _ARTIFACT_CONNECTOR_AGENT is not None, (
        "artifact transfer connector is not initialized"
    )
    return _ARTIFACT_CONNECTOR_AGENT


def has_artifact_transfer_group() -> bool:
    return _ARTIFACT_CONNECTOR_AGENT is not None


def _sync_engine_id_across_tp(vllm_config: "VllmConfig") -> None:
    """Broadcast engine_id from TP rank 0 so all workers share it."""
    from vllm.distributed.parallel_state import get_tp_group

    assert vllm_config.artifact_transfer_config is not None
    synced_id = get_tp_group().broadcast_object(
        vllm_config.artifact_transfer_config.engine_id, src=0
    )
    vllm_config.artifact_transfer_config.engine_id = synced_id


def ensure_artifact_transfer_initialized(vllm_config: "VllmConfig") -> None:
    """Initialize the worker-side artifact transfer connector."""

    global _ARTIFACT_CONNECTOR_AGENT

    if vllm_config.artifact_transfer_config is None:
        return

    if (
        vllm_config.artifact_transfer_config.is_artifact_transfer_instance
        and _ARTIFACT_CONNECTOR_AGENT is None
    ):
        _sync_engine_id_across_tp(vllm_config)
        _ARTIFACT_CONNECTOR_AGENT = ArtifactConnectorFactory.create_connector(
            config=vllm_config,
            role=ArtifactConnectorRole.WORKER,
        )


def ensure_artifact_transfer_shutdown() -> None:
    global _ARTIFACT_CONNECTOR_AGENT
    if _ARTIFACT_CONNECTOR_AGENT is not None:
        _ARTIFACT_CONNECTOR_AGENT.shutdown()
        _ARTIFACT_CONNECTOR_AGENT = None
