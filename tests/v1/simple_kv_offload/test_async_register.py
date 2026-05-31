# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for async cudaHostRegister gate in SimpleCPUOffloadConnector."""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from tests.v1.simple_kv_offload.test_scheduler import (
    _make_kv_cache_config,
    _make_vllm_config,
    make_scheduler,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.simple_kv_offload.manager import SimpleCPUOffloadScheduler
from vllm.v1.simple_kv_offload.metadata import SimpleCPUOffloadWorkerMetadata
from vllm.v1.simple_kv_offload.worker import SimpleCPUOffloadWorker

BLOCK_SIZE = 16
HEAD_SIZE = 16
NUM_KV_HEADS = 1
DTYPE = torch.float16
_BYTES_PER_BLOCK = BLOCK_SIZE * NUM_KV_HEADS * HEAD_SIZE * 2 * DTYPE.itemsize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_async_scheduler(
    num_cpu_blocks: int = 8,
    num_gpu_blocks: int = 16,
    world_size: int = 1,
) -> SimpleCPUOffloadScheduler:
    kv_cache_config = _make_kv_cache_config(num_gpu_blocks)
    vllm_config = _make_vllm_config()
    vllm_config.parallel_config.world_size = world_size
    return SimpleCPUOffloadScheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        cpu_capacity_bytes=_BYTES_PER_BLOCK * num_cpu_blocks,
        scheduler_block_size=BLOCK_SIZE,
        hash_block_size=BLOCK_SIZE,
        async_register_cache=True,
    )


class _FakeWorker(SimpleCPUOffloadWorker):
    """Worker with all CUDA calls replaced by no-ops for unit testing."""

    def _setup_device(self) -> None:
        pass

    def _create_streams_and_backend(self) -> None:
        pass


def _make_worker(
    async_register: bool = False,
    worker_rank: int = 0,
) -> _FakeWorker:
    return _FakeWorker(
        vllm_config=_make_vllm_config(),
        kv_cache_config=_make_kv_cache_config(num_blocks=4),
        cpu_capacity_bytes=_BYTES_PER_BLOCK * 4,
        worker_rank=worker_rank,
        async_register_cache=async_register,
    )


def _signal_ready(sched: SimpleCPUOffloadScheduler, rank: int) -> None:
    meta = SimpleCPUOffloadWorkerMetadata(
        completed_store_events={},
        ready_worker_ids=frozenset({rank}),
    )
    sched.update_connector_output(
        KVConnectorOutput(
            finished_recving=None,
            finished_sending=None,
            kv_connector_worker_meta=meta,
        )
    )


# ---------------------------------------------------------------------------
# SimpleCPUOffloadWorkerMetadata aggregation
# ---------------------------------------------------------------------------


def test_aggregate_unions_ready_ids_and_sums_store_events():
    # Given
    m0 = SimpleCPUOffloadWorkerMetadata(
        completed_store_events={1: 1},
        ready_worker_ids=frozenset({0}),
    )
    m1 = SimpleCPUOffloadWorkerMetadata(
        completed_store_events={1: 1, 2: 1},
        ready_worker_ids=frozenset({1}),
    )

    # When
    merged = m0.aggregate(m1)

    # Then
    assert isinstance(merged, SimpleCPUOffloadWorkerMetadata)
    assert merged.ready_worker_ids == frozenset({0, 1})
    assert merged.completed_store_events == {1: 2, 2: 1}


def test_aggregate_deduplicates_same_rank():
    # Given
    m0 = SimpleCPUOffloadWorkerMetadata(
        completed_store_events={},
        ready_worker_ids=frozenset({3}),
    )
    m1 = SimpleCPUOffloadWorkerMetadata(
        completed_store_events={},
        ready_worker_ids=frozenset({3}),
    )

    # When
    merged = m0.aggregate(m1)

    # Then
    assert merged.ready_worker_ids == frozenset({3})


# ---------------------------------------------------------------------------
# SimpleCPUOffloadScheduler gate behaviour
# ---------------------------------------------------------------------------


def test_gate_open_by_default_in_sync_mode():
    # Given
    fix = make_scheduler()

    # Then
    assert fix.scheduler._gate_open is True


def test_gate_closed_on_construction_in_async_mode():
    # Given
    sched = _make_async_scheduler()

    # Then
    assert sched._gate_open is False


def test_gate_blocks_get_num_new_matched_tokens():
    # Given
    sched = _make_async_scheduler()

    # When
    result = sched.get_num_new_matched_tokens(MagicMock(), num_computed_tokens=0)

    # Then
    assert result == (0, False)


def test_gate_blocks_build_connector_meta():
    # Given
    from vllm.v1.simple_kv_offload.metadata import (
        INVALID_JOB_ID,
        SimpleCPUOffloadMetadata,
    )

    sched = _make_async_scheduler()

    # When
    meta = sched.build_connector_meta(MagicMock())

    # Then
    assert isinstance(meta, SimpleCPUOffloadMetadata)
    assert meta.load_event == INVALID_JOB_ID
    assert meta.store_event == INVALID_JOB_ID
    assert not meta.load_gpu_blocks
    assert not meta.store_gpu_blocks


def test_gate_opens_when_single_rank_signals_ready():
    # Given
    sched = _make_async_scheduler(world_size=1)
    assert not sched._gate_open

    # When
    _signal_ready(sched, rank=0)

    # Then
    assert sched._gate_open


def test_gate_stays_closed_until_all_ranks_signal():
    # Given
    sched = _make_async_scheduler(world_size=2)

    # When
    _signal_ready(sched, rank=0)

    # Then
    assert not sched._gate_open

    # When
    _signal_ready(sched, rank=1)

    # Then
    assert sched._gate_open


def test_gate_ignores_repeated_signals_from_same_rank():
    # Given
    sched = _make_async_scheduler(world_size=2)
    for _ in range(3):
        _signal_ready(sched, rank=0)

    # When
    _signal_ready(sched, rank=1)

    # Then
    assert sched._gate_open


# ---------------------------------------------------------------------------
# SimpleCPUOffloadWorker readiness signal
# ---------------------------------------------------------------------------


def test_worker_pin_done_set_after_sync_init():
    # Given
    w = _make_worker(async_register=False)
    w.gpu_kv_caches = {}

    # When
    w._do_alloc_pin_init(pin_memory=False)

    # Then
    assert w._pin_done.is_set()


def test_worker_pin_done_unset_before_async_init():
    # Given
    w = _make_worker(async_register=True)

    # Then
    assert not w._pin_done.is_set()


def test_worker_silent_before_pin_done():
    # Given
    w = _make_worker(async_register=True)
    assert not w._pin_done.is_set()

    # When
    result = w.build_connector_worker_meta()

    # Then
    assert result is None


def test_worker_emits_ready_ids_exactly_once():
    # Given
    w = _make_worker(async_register=True, worker_rank=7)
    w._pin_done.set()

    # When
    meta = w.build_connector_worker_meta()

    # Then
    assert meta is not None
    assert meta.ready_worker_ids == frozenset({7})

    # When
    meta2 = w.build_connector_worker_meta()

    # Then
    assert meta2 is None


def test_worker_emits_stores_and_readiness_together():
    # Given
    w = _make_worker(async_register=True, worker_rank=2)
    w._pin_done.set()
    w._completed_store_events = {42: 1}

    # When
    meta = w.build_connector_worker_meta()

    # Then
    assert meta is not None
    assert meta.ready_worker_ids == frozenset({2})
    assert meta.completed_store_events == {42: 1}
    assert w._completed_store_events == {}


def test_async_pin_sets_accelerator_device_before_pinning():
    # Given
    class _RecordingWorker(_FakeWorker):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.device_calls: list = []

        def _setup_device(self) -> None:
            self.device_calls.append(self.device)

    w = _RecordingWorker(
        vllm_config=_make_vllm_config(),
        kv_cache_config=_make_kv_cache_config(num_blocks=4),
        cpu_capacity_bytes=_BYTES_PER_BLOCK * 4,
        async_register_cache=True,
    )
    w.device = torch.device("cpu")
    w.gpu_kv_caches = {}

    # When
    w._start_async_pin(pin_memory=False)
    w._pin_done.wait(timeout=5.0)

    # Then
    assert w.device_calls == [torch.device("cpu")]
    assert w._pin_done.is_set()


def test_async_pin_failure_leaves_pin_done_unset():
    # Given
    class _FailingWorker(_FakeWorker):
        def _setup_device(self) -> None:
            raise RuntimeError("accelerator boom")

    w = _FailingWorker(
        vllm_config=_make_vllm_config(),
        kv_cache_config=_make_kv_cache_config(num_blocks=4),
        cpu_capacity_bytes=_BYTES_PER_BLOCK * 4,
        async_register_cache=True,
    )
    w.device = torch.device("cpu")
    w.gpu_kv_caches = {}

    # When
    w._start_async_pin(pin_memory=False)
    # Give the thread a moment to raise and be caught.

    # Then
    assert not w._pin_done.is_set()


def test_async_thread_sets_pin_done_on_completion():
    # Given
    w = _make_worker(async_register=True)
    gpu_tensor = torch.zeros(4, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, 2, dtype=DTYPE)

    # When
    w.register_kv_caches({"layer_0": gpu_tensor})
    completed = w._pin_done.wait(timeout=5.0)

    # Then
    assert completed, "Background thread did not set _pin_done within timeout"
