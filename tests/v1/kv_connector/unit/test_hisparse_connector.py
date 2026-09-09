# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector import (
    HiSparseConnector,
    HiSparseConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector
from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector


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


def test_nested_connector_binding_preserves_pool_and_hisparse_state():
    """Nested composition must bind HiSparse while retaining legacy pool hooks."""

    class PoolConnector:
        bind_kv_cache_manager = KVConnectorBase_V1.bind_kv_cache_manager
        bind_gpu_block_pool = MagicMock()

    pool_connector = PoolConnector()
    hisparse = object.__new__(HiSparseConnector)
    hisparse._role = KVConnectorRole.SCHEDULER
    hisparse.connector_scheduler = HiSparseConnectorScheduler(async_speculative=False)
    inner = object.__new__(MultiConnector)
    inner._connectors = [hisparse]
    outer = object.__new__(MultiConnector)
    outer._connectors = [pool_connector, inner]
    manager = SimpleNamespace(block_pool=object(), hisparse_coordinator=object())

    outer.bind_kv_cache_manager(manager)

    pool_connector.bind_gpu_block_pool.assert_called_once_with(manager.block_pool)
    assert hisparse.connector_scheduler.coordinator is manager.hisparse_coordinator
