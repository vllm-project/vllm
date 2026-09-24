# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.distributed.kv_events import MEDIUM_CPU
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.example_connector import (
    ExampleConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector import (
    HiSparseConnector,
    HiSparseConnectorMetadata,
    HiSparseConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker import (
    HiSparseConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.hisparse.coordinator import get_hisparse_coordinator
from vllm.v1.hisparse.runtime import HiSparseCacheHandle
from vllm.v1.metrics.stats import KVCacheEvictionEvent
from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector


@pytest.mark.parametrize("nested", [False, True])
def test_cache_manager_binding_preserves_hisparse_and_legacy_pool_hooks(nested):
    """Composite binding must reach HiSparse and existing pool-only connectors."""
    hisparse = object.__new__(HiSparseConnector)
    hisparse._role = KVConnectorRole.SCHEDULER
    hisparse.connector_scheduler = HiSparseConnectorScheduler(async_speculative=False)
    legacy = object.__new__(ExampleConnector)
    legacy.bind_gpu_block_pool = MagicMock()
    connector = object.__new__(MultiConnector)
    connector._connectors = [hisparse, legacy]
    if nested:
        parent = object.__new__(MultiConnector)
        parent._connectors = [connector]
        connector = parent
    from tests.v1.core.test_prefix_caching import make_hisparse_kv_cache_manager

    manager = make_hisparse_kv_cache_manager(16, 16)

    connector.bind_kv_cache_manager(manager)

    assert hisparse.connector_scheduler.coordinator is get_hisparse_coordinator(manager)
    legacy.bind_gpu_block_pool.assert_called_once_with(manager.block_pool)


def test_hisparse_requires_block_outermost_device_layout():
    assert HiSparseConnector.get_required_kvcache_layout(MagicMock()) == "BLHNC"


@pytest.mark.parametrize("metrics_enabled", [False, True])
def test_hisparse_host_pool_preserves_device_lifecycle(metrics_enabled):
    from tests.v1.core.test_prefix_caching import (
        make_hisparse_kv_cache_config,
        make_kv_cache_manager,
    )
    from tests.v1.core.utils import create_requests

    collector = KVCacheMetricsCollector(sample_rate=1.0) if metrics_enabled else None
    manager = make_kv_cache_manager(
        make_hisparse_kv_cache_config(2, 2),
        max_model_len=128,
        enable_caching=True,
        hash_block_size=16,
        metrics_collector=collector,
    )
    coordinator = get_hisparse_coordinator(manager)
    assert coordinator.host_manager is not None
    device = manager.block_pool
    host = coordinator.host_manager.block_pool
    assert host is not device and host.medium == MEDIUM_CPU
    assert host.metrics_collector is device.metrics_collector is collector
    request = create_requests(1, num_tokens=16)[0]
    for pool, birth in ((device, 1), (host, 2)):
        with patch("time.monotonic_ns", return_value=birth * 10**9):
            block = pool.get_new_blocks(1)[0]
            assert block.block_id == 1 and block.pool is pool
            pool.cache_full_blocks(request, [block], 0, 1, 16, 0)
    if collector is not None:
        assert collector.drain_events() == []
    with patch("time.monotonic_ns", return_value=3_000_000_000):
        host.touch([host.blocks[1]])
        host.free_blocks([host.blocks[1]])
    with patch("time.monotonic_ns", return_value=4_000_000_000):
        host.evict_blocks({1})
    assert device.blocks[1].ref_cnt == 1
    with patch("time.monotonic_ns", return_value=6_000_000_000):
        device.evict_blocks({1})
    if collector is not None:
        assert collector.drain_events() == [
            KVCacheEvictionEvent(2.0, 1.0, ()),
            KVCacheEvictionEvent(5.0, 5.0, ()),
        ]
        assert collector.drain_events() == []
    for pool in (host, device):
        pool.free_blocks([pool.blocks[1]])
    assert manager.reset_prefix_cache()


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


def test_full_graph_step_prepares_host_mirror_outside_model():
    """Graph replay must restore host-mirror state cleared at step start."""
    runtime = SimpleNamespace(
        is_group_leader=True,
        eager_host_mirror=True,
        begin_forward=MagicMock(),
        invalidate_written_slots=MagicMock(),
    )
    handle = HiSparseCacheHandle(runtime)
    handle.mirror_slot_mapping = torch.tensor([4, 5])
    worker = object.__new__(HiSparseConnectorWorker)
    worker.cache_layer_names = ["layer"]
    worker.cache_handles = [handle]
    worker._group_leaders = (("layer", handle),)
    worker._per_layer_mirrored = set()
    worker._submitted_mirror_layers = set()
    worker.is_host_writer = True
    worker._enqueue_row_dma = MagicMock()
    worker.start_step = MagicMock(
        side_effect=lambda *_args, **_kwargs: worker._clear_forward_mirror_state()
    )

    connector = object.__new__(HiSparseConnector)
    connector.connector_worker = worker
    connector._get_connector_metadata = MagicMock(
        return_value=HiSparseConnectorMetadata(None, (), (), {}, True)
    )
    req_id_per_token = torch.tensor([0, 1], dtype=torch.int32)
    attn_metadata = SimpleNamespace(
        num_actual_tokens=2,
        num_decode_tokens=2,
        num_reqs=2,
        max_query_len=1,
        req_id_per_token=req_id_per_token,
    )

    connector.start_load_kv(
        SimpleNamespace(),
        request_state_indices=None,
        request_ids=[],
        attn_metadata={"layer": attn_metadata},
    )

    worker._enqueue_host_mirror()

    worker._enqueue_row_dma.assert_called_once_with((0,), ready_event=None)
    runtime.invalidate_written_slots.assert_called_once()
