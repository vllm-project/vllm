# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
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
from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector


def _kv_cache_config() -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=0,
        kv_cache_tensors=[],
        kv_cache_groups=[],
        hisparse_host_num_blocks=1,
    )


def test_hisparse_connector_composes_without_replacing_existing_connector():
    config = create_vllm_config(disable_hybrid_kv_cache_manager=True)
    kv_cache_config = _kv_cache_config()
    primary = MagicMock()
    primary._kv_transfer_config = config.kv_transfer_config
    primary_stats = primary.get_kv_connector_stats.return_value

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
    assert connector.get_kv_connector_stats() is primary_stats


def test_no_forward_enqueues_deferred_hisparse_transfers():
    """A zero-token step must still enqueue deferred post-forward transfers."""
    connector = object.__new__(ActiveKVConnector)
    connector._disabled = False
    connector.pre_forward = MagicMock()
    connector.finish_forward = MagicMock()
    connector.post_forward = MagicMock(return_value=None)

    scheduler_output = SimpleNamespace(finished_req_ids=set())
    connector.no_forward(scheduler_output)

    connector.pre_forward.assert_called_once_with(scheduler_output)
    connector.finish_forward.assert_called_once_with()
