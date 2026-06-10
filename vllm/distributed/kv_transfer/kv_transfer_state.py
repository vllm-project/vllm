# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBaseType
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

_KV_CONNECTOR_AGENT: KVConnectorBaseType | None = None


def get_kv_transfer_group() -> KVConnectorBaseType:
    assert _KV_CONNECTOR_AGENT is not None, (
        "disaggregated KV cache transfer parallel group is not initialized"
    )
    return _KV_CONNECTOR_AGENT


def has_kv_transfer_group() -> bool:
    return _KV_CONNECTOR_AGENT is not None


def is_v1_kv_transfer_group(connector: KVConnectorBaseType | None = None) -> bool:
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


def _sync_engine_id_across_tp(vllm_config: "VllmConfig") -> None:
    """Broadcast engine_id from TP rank 0 so all workers in a
    multi-node TP group share the same value."""
    from vllm.distributed.parallel_state import (
        get_tp_group,
    )

    assert vllm_config.kv_transfer_config is not None
    synced_id = get_tp_group().broadcast_object(
        vllm_config.kv_transfer_config.engine_id, src=0
    )
    vllm_config.kv_transfer_config.engine_id = synced_id


def ensure_kv_transfer_initialized(
    vllm_config: "VllmConfig", kv_cache_config: "KVCacheConfig"
) -> None:
    """
    Initialize KV cache transfer parallel group.
    """

    global _KV_CONNECTOR_AGENT

    if vllm_config.kv_transfer_config is None:
        return

    if (
        vllm_config.kv_transfer_config.is_kv_transfer_instance
        and _KV_CONNECTOR_AGENT is None
    ):
        _sync_engine_id_across_tp(vllm_config)

        _KV_CONNECTOR_AGENT = KVConnectorFactory.create_connector(
            config=vllm_config,
            role=KVConnectorRole.WORKER,
            kv_cache_config=kv_cache_config,
        )


def ensure_kv_transfer_shutdown() -> None:
    global _KV_CONNECTOR_AGENT
    if _KV_CONNECTOR_AGENT is not None:
        _KV_CONNECTOR_AGENT.shutdown()
        _KV_CONNECTOR_AGENT = None
