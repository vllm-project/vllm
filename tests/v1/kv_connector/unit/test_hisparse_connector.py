# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import MagicMock

from tests.v1.kv_connector.unit.utils import create_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.hisparse_connector import (
    HiSparseConnector,
    attach_hisparse_connector,
    bind_hisparse_worker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
)
from vllm.v1.kv_cache_interface import KVCacheConfig


def _kv_cache_config() -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=0,
        kv_cache_tensors=[],
        kv_cache_groups=[],
        hisparse_host_num_blocks=1,
    )


def _vllm_config():
    return create_vllm_config(disable_hybrid_kv_cache_manager=True)


def test_hisparse_connector_composes_without_replacing_existing_connector():
    config = _vllm_config()
    kv_cache_config = _kv_cache_config()
    primary = MagicMock()
    primary._kv_transfer_config = config.kv_transfer_config

    connector = attach_hisparse_connector(
        primary, config, KVConnectorRole.WORKER, kv_cache_config
    )

    assert isinstance(connector, MultiConnector)
    assert primary in connector._connectors
    hisparse = next(
        child for child in connector._connectors if isinstance(child, HiSparseConnector)
    )

    worker = MagicMock()
    bind_hisparse_worker(connector, worker)
    assert hisparse._worker is worker


def test_hisparse_composition_preserves_existing_connector_stats_schema():
    config = _vllm_config()
    primary = MagicMock()
    primary._kv_transfer_config = config.kv_transfer_config
    primary_stats = MagicMock()
    primary.get_kv_connector_stats.return_value = primary_stats

    connector = attach_hisparse_connector(
        primary, config, KVConnectorRole.WORKER, _kv_cache_config()
    )

    assert connector.get_kv_connector_stats() is primary_stats


def test_hisparse_initializes_as_the_only_worker_connector(monkeypatch):
    from vllm.distributed.kv_transfer import kv_transfer_state

    config = _vllm_config()
    config.kv_transfer_config = None
    monkeypatch.setattr(kv_transfer_state, "_KV_CONNECTOR_AGENT", None)

    kv_transfer_state.ensure_kv_transfer_initialized(config, _kv_cache_config())

    assert isinstance(kv_transfer_state.get_kv_transfer_group(), HiSparseConnector)
    kv_transfer_state.ensure_kv_transfer_shutdown()
