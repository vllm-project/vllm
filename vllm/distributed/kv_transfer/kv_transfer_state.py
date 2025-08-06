# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, Optional

from vllm import envs
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBaseType
from vllm.distributed.kv_transfer.kv_connector.factory import (
    KVConnectorFactory)
from vllm.distributed.kv_transfer.kv_connector.v1 import (KVConnectorBase_V1,
                                                          KVConnectorRole)
from vllm.distributed.parallel_state import get_world_group

if TYPE_CHECKING:
    from vllm.config import VllmConfig

_KV_CONNECTOR_AGENT: Optional[KVConnectorBaseType] = None


def get_kv_transfer_group() -> KVConnectorBaseType:
    assert _KV_CONNECTOR_AGENT is not None, (
        "disaggregated KV cache transfer parallel group is not initialized")
    return _KV_CONNECTOR_AGENT


def has_kv_transfer_group() -> bool:
    return _KV_CONNECTOR_AGENT is not None


def is_v1_kv_transfer_group(
        connector: Optional[KVConnectorBaseType] = None) -> bool:
    """Check if the KV connector is the v1 connector.
    If the argument is None, it will check the global KV connector

    Args:
        connector: The KV connector to check. If None, it will check the
            global KV connector.

    Note:
        This function will no-longer be needed after the v1 KV connector
        becomes the default.
    """
    if connector is None:
        connector = _KV_CONNECTOR_AGENT

    if connector is None:
        return False

    return isinstance(connector, KVConnectorBase_V1)


def ensure_kv_transfer_initialized(vllm_config: "VllmConfig") -> None:
    """
    Initialize KV cache transfer parallel group.
    """

    global _KV_CONNECTOR_AGENT

    if vllm_config.kv_transfer_config is None:
        return

    if (vllm_config.kv_transfer_config.is_kv_transfer_instance
            and _KV_CONNECTOR_AGENT is None):
        if envs.VLLM_USE_V1:
            _KV_CONNECTOR_AGENT = KVConnectorFactory.create_connector_v1(
                config=vllm_config, role=KVConnectorRole.WORKER)
        else:
            _KV_CONNECTOR_AGENT = KVConnectorFactory.create_connector_v0(
                rank=get_world_group().rank,
                local_rank=get_world_group().local_rank,
                config=vllm_config,
            )
