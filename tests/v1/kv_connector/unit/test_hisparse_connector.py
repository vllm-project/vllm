# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import MagicMock

from tests.v1.kv_connector.unit.utils import create_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.hisparse_connector import (
    HiSparseConnector,
    HiSparseConnectorMetadata,
    HiSparseConnectorWorkerMetadata,
    attach_hisparse_connector,
    bind_hisparse_worker,
    get_hisparse_connector_metadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
    MultiKVConnectorMetadata,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.sparse.base import SparseKVOffloadCommand
from vllm.v1.outputs import KVConnectorOutput


def _kv_cache_config() -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=0,
        kv_cache_tensors=[],
        kv_cache_groups=[],
        hisparse_host_num_blocks=1,
    )


def _vllm_config():
    return create_vllm_config(disable_hybrid_kv_cache_manager=True)


def test_hisparse_connector_routes_commands_and_completions():
    manager = MagicMock()
    command = SparseKVOffloadCommand({}, [], fully_resident=True)
    manager.build_offload_command.return_value = command
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.num_scheduled_tokens = {"request": 1}

    scheduler = HiSparseConnector(
        _vllm_config(), KVConnectorRole.SCHEDULER, _kv_cache_config(), manager
    )
    metadata = scheduler.build_connector_meta(scheduler_output)

    assert metadata == HiSparseConnectorMetadata(command)
    manager.build_offload_command.assert_called_once_with(["request"])

    worker_runtime = MagicMock()
    worker_runtime.take_completed_transfer_ids.return_value = [3, 7]
    worker = HiSparseConnector(
        _vllm_config(), KVConnectorRole.WORKER, _kv_cache_config()
    )
    worker.bind_worker(worker_runtime)
    worker.bind_connector_metadata(metadata)
    worker.prepare_step(scheduler_output)
    worker.finish_forward()
    worker_metadata = worker.build_connector_worker_meta()

    worker_runtime.prepare_step.assert_called_once_with(command, scheduler_output)
    worker_runtime.finish_forward.assert_called_once_with()
    assert worker_metadata == HiSparseConnectorWorkerMetadata([3, 7])

    scheduler.update_connector_output(
        KVConnectorOutput(kv_connector_worker_meta=worker_metadata)
    )
    manager.complete_spills.assert_called_once_with([3, 7])


def test_hisparse_connector_composes_without_replacing_existing_connector():
    config = _vllm_config()
    kv_cache_config = _kv_cache_config()
    primary = MagicMock()
    primary._kv_transfer_config = config.kv_transfer_config

    connector = attach_hisparse_connector(
        primary, config, KVConnectorRole.WORKER, kv_cache_config
    )

    assert isinstance(connector, MultiConnector)
    assert connector._connectors[0] is primary

    runtime = MagicMock()
    bind_hisparse_worker(connector, runtime)
    assert connector._connectors[1]._worker is runtime

    metadata = MultiKVConnectorMetadata(
        (KVConnectorMetadata(), HiSparseConnectorMetadata(None))
    )
    assert get_hisparse_connector_metadata(metadata) is metadata.metadata[1]


def test_hisparse_initializes_as_the_only_worker_connector(monkeypatch):
    from vllm.distributed.kv_transfer import kv_transfer_state

    config = _vllm_config()
    config.kv_transfer_config = None
    monkeypatch.setattr(kv_transfer_state, "_KV_CONNECTOR_AGENT", None)

    kv_transfer_state.ensure_kv_transfer_initialized(config, _kv_cache_config())

    assert isinstance(kv_transfer_state.get_kv_transfer_group(), HiSparseConnector)
    kv_transfer_state.ensure_kv_transfer_shutdown()
