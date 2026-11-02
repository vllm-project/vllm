# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SimpleCPUOffloadConnector-level behavior."""

from __future__ import annotations

import gc
import os
import threading
import time
import weakref
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tests.v1.kv_connector.unit.utils import create_vllm_config
from tests.v1.simple_kv_offload.test_scheduler import (
    _BYTES_PER_BLOCK,
    BLOCK_SIZE,
    DTYPE,
    HEAD_SIZE,
    NUM_KV_HEADS,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (
    SimpleCPUOffloadConnector,
)
from vllm.v1.engine import core as engine_core_module
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from vllm.v1.simple_kv_offload.disk_backend import DiskBackend


def _make_kv_cache_config(num_blocks: int = 16) -> KVCacheConfig:
    fa_layers = ["layer0"]
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=DTYPE,
    )
    groups = [KVCacheGroupSpec(fa_layers, spec)]
    size = _BYTES_PER_BLOCK * num_blocks
    tensors = [
        KVCacheTensor(
            size=size,
            layers=fa_layers,
            layer_stride=size,
            block_stride=_BYTES_PER_BLOCK,
        )
    ]
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
    )


def _make_connector(
    extra_config: dict | None = None,
    num_gpu_blocks: int = 16,
    cpu_bytes: int = _BYTES_PER_BLOCK * 8,
) -> SimpleCPUOffloadConnector:
    kv_cache_config = _make_kv_cache_config(num_gpu_blocks)
    config_dict = {
        "cpu_bytes_to_use": cpu_bytes,
    }
    if extra_config:
        config_dict.update(extra_config)
    vllm_config = create_vllm_config(
        kv_connector="SimpleCPUOffloadConnector",
        kv_role="kv_both",
        kv_connector_extra_config=config_dict,
    )
    connector = SimpleCPUOffloadConnector(
        vllm_config=vllm_config,
        role=KVConnectorRole.SCHEDULER,
        kv_cache_config=kv_cache_config,
    )
    return connector


def test_disk_mode_rejects_non_positive_capacity() -> None:
    """Disk mode with disk_capacity_bytes <= 0 raises ValueError.

    Without this guard, the worker operates in disk mode but the scheduler
    classifies events as MEDIUM_CPU instead of MEDIUM_STORAGE, silently
    mislabeling events.
    """
    with pytest.raises(ValueError, match="disk_capacity_bytes > 0"):
        _make_connector(
            extra_config={
                "kv_offload_backend": "disk",
                "disk_path": "/tmp/fake",
                "disk_capacity_bytes": 0,
            }
        )


_DISK_SLOTS = 8


class _FakeEvent:
    """Stand-in for torch.Event so the IO threads run without an accelerator."""

    def record(self, stream=None) -> None:
        pass

    def synchronize(self) -> None:
        pass


class _FakeStream:
    def __init__(self, priority: int = 0) -> None:
        pass

    @staticmethod
    def priority_range() -> tuple[int, int]:
        return 0, 0


class _FakeCudart:
    """Host registration for torch.cuda.cudart(), keyed by base address.

    Fails the way the CUDA runtime does: registering a registered range
    returns cudaErrorHostMemoryAlreadyRegistered (712), unregistering an
    unregistered one cudaErrorHostMemoryNotRegistered (713).
    """

    def __init__(self) -> None:
        self.registered: set[int] = set()
        self.registers: list[int] = []
        self.unregisters: list[int] = []

    def cudaHostRegister(self, ptr: int, nbytes: int, flags: int):
        self.registers.append(ptr)
        if ptr in self.registered:
            return SimpleNamespace(value=712)
        self.registered.add(ptr)
        return SimpleNamespace(value=0)

    def cudaHostUnregister(self, ptr: int):
        self.unregisters.append(ptr)
        if ptr not in self.registered:
            return SimpleNamespace(value=713)
        self.registered.remove(ptr)
        return SimpleNamespace(value=0)


def _make_disk_worker_connector(
    tmp_path, monkeypatch, cudart: _FakeCudart | None = None
) -> SimpleCPUOffloadConnector:
    """A worker-role connector whose real DiskBackend.init() created its file.

    Only the CUDA pieces (streams, events, host (un)registration, batch-copy
    params, device binding) are stubbed.
    """
    fake_cudart = cudart if cudart is not None else _FakeCudart()
    monkeypatch.setattr(torch.cuda, "Stream", _FakeStream)
    monkeypatch.setattr(torch.cuda, "cudart", lambda: fake_cudart)
    monkeypatch.setattr(torch, "Event", _FakeEvent)
    monkeypatch.setattr(
        "vllm.v1.simple_kv_offload.disk_backend.build_params", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "vllm.v1.simple_kv_offload.disk_backend.current_platform",
        SimpleNamespace(set_device=lambda device: None),
    )
    vllm_config = create_vllm_config(
        kv_connector="SimpleCPUOffloadConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "kv_offload_backend": "disk",
            "disk_path": str(tmp_path / "simple-disk"),
            "disk_capacity_bytes": _BYTES_PER_BLOCK * _DISK_SLOTS,
            "use_page_cache": True,
        },
    )
    num_gpu_blocks = 16
    connector = SimpleCPUOffloadConnector(
        vllm_config=vllm_config,
        role=KVConnectorRole.WORKER,
        kv_cache_config=_make_kv_cache_config(num_gpu_blocks),
    )
    kv_cache = torch.zeros(_BYTES_PER_BLOCK * num_gpu_blocks, dtype=torch.int8)
    connector.register_kv_caches({"layer0": kv_cache})
    return connector


def _disk_backend(connector: SimpleCPUOffloadConnector) -> DiskBackend:
    assert connector.worker_handler is not None
    backend = connector.worker_handler._backend
    assert isinstance(backend, DiskBackend)
    return backend


def test_shutdown_removes_disk_file_and_is_idempotent(tmp_path, monkeypatch):
    """Graceful shutdown unlinks the per-rank file, whose blocks can encode
    prompts; a second call is a no-op and leaves other files alone."""
    other_rank = tmp_path / "simple-disk.rank_1"
    other_rank.write_bytes(b"other rank")
    connector = _make_disk_worker_connector(tmp_path, monkeypatch)
    backend = _disk_backend(connector)
    disk_file = tmp_path / "simple-disk.rank_0"
    assert disk_file.stat().st_size == _BYTES_PER_BLOCK * _DISK_SLOTS

    connector.shutdown()

    assert not disk_file.exists()
    assert other_rank.read_bytes() == b"other rank"
    assert backend._fd == -1
    assert backend._store_thread is not None and not backend._store_thread.is_alive()
    assert backend._load_thread is not None and not backend._load_thread.is_alive()

    # A later file at the same path (e.g. the next start) survives a repeated
    # shutdown.
    disk_file.write_bytes(b"next run")
    connector.shutdown()
    assert disk_file.read_bytes() == b"next run"


def test_shutdown_keeps_a_file_another_process_put_at_the_same_path(
    tmp_path, monkeypatch
):
    """Only the inode init() created is unlinked, not whatever another
    process's init() has since recreated at the same path."""
    backend = _disk_backend(_make_disk_worker_connector(tmp_path, monkeypatch))
    disk_file = tmp_path / "simple-disk.rank_0"
    disk_file.unlink()
    disk_file.write_bytes(b"another process")

    backend.shutdown()

    assert disk_file.read_bytes() == b"another process"


def test_shutdown_drops_queued_transfers(tmp_path, monkeypatch):
    """Queued stores are dropped, not drained into the file just removed."""
    connector = _make_disk_worker_connector(tmp_path, monkeypatch)
    backend = _disk_backend(connector)
    started = threading.Event()
    stored: list[list[int]] = []

    def blocking_store(gpu_blocks, *_) -> None:
        stored.append(gpu_blocks)
        started.set()
        deadline = time.monotonic() + 5
        while not backend._shutdown and time.monotonic() < deadline:
            time.sleep(0.01)

    monkeypatch.setattr(backend, "_do_store", blocking_store)
    events: list = []
    for i in range(4):
        backend.launch_copy([i], [i], is_store=True, event_idx=i, events_list=events)
    assert started.wait(timeout=5)

    connector.shutdown()

    assert stored == [[0]]
    assert backend._store_thread is not None and not backend._store_thread.is_alive()


def test_shutdown_unregisters_staging_buffers_exactly_once(tmp_path, monkeypatch):
    """Every staging buffer is unregistered once, after the IO threads exit,
    and then released. A buffer freed while still registered makes the next
    cudaHostRegister of that range fail (e.g. for the next DiskBackend in the
    same process)."""
    cudart = _FakeCudart()
    connector = _make_disk_worker_connector(tmp_path, monkeypatch, cudart=cudart)
    backend = _disk_backend(connector)
    # One store and one load staging buffer for the single KV cache tensor.
    assert len(cudart.registered) == 2
    buffers = [weakref.ref(t) for t in backend._pinned_buffers]

    connector.shutdown()
    connector.shutdown()

    assert sorted(cudart.unregisters) == sorted(cudart.registers)
    assert cudart.registered == set()
    gc.collect()
    assert [ref() for ref in buffers] == [None, None]


def test_shutdown_keeps_staging_buffers_registered_while_an_io_thread_runs(
    tmp_path, monkeypatch, caplog_vllm
):
    """An IO thread that outlives the join timeout may still DMA into its
    staging buffers, so they stay registered and the fd stays open."""
    from vllm.v1.simple_kv_offload import disk_backend

    cudart = _FakeCudart()
    connector = _make_disk_worker_connector(tmp_path, monkeypatch, cudart=cudart)
    backend = _disk_backend(connector)
    monkeypatch.setattr(disk_backend, "_SHUTDOWN_JOIN_TIMEOUT_S", 0.05)
    started, release = threading.Event(), threading.Event()

    def stuck_store(*_) -> None:
        started.set()
        release.wait(timeout=10)

    monkeypatch.setattr(backend, "_do_store", stuck_store)
    backend.launch_copy([0], [0], is_store=True, event_idx=0, events_list=[])
    assert started.wait(timeout=5)
    try:
        connector.shutdown()

        assert cudart.unregisters == []
        assert len(cudart.registered) == 2
        assert backend._fd >= 0
        assert not (tmp_path / "simple-disk.rank_0").exists()
        assert any("still running" in r.getMessage() for r in caplog_vllm.records)
    finally:
        release.set()
        assert backend._store_thread is not None
        backend._store_thread.join(timeout=5)
        os.close(backend._fd)


def _make_ring_kv_cache_config(num_blocks: int = 16) -> KVCacheConfig:
    """A paged full-attention group beside a per-request ring group."""
    full = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=DTYPE,
    )
    ring = CircularBufferSpec(
        block_size=6, num_kv_heads=1, head_size=32, head_size_v=0, dtype=DTYPE
    )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=_BYTES_PER_BLOCK * num_blocks,
                layers=["attn"],
                layer_stride=_BYTES_PER_BLOCK * num_blocks,
                block_stride=_BYTES_PER_BLOCK,
            ),
            KVCacheTensor(
                size=ring.page_size_bytes * num_blocks,
                layers=["ring"],
                layer_stride=ring.page_size_bytes * num_blocks,
                block_stride=ring.page_size_bytes,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["attn"], full),
            KVCacheGroupSpec(["ring"], ring),
        ],
    )


def test_ring_scratch_group_is_accepted_and_never_offloaded() -> None:
    """The ring group's capacity is not a token granularity, it holds no
    hashed blocks, and it never participates in CPU hits."""
    vllm_config = create_vllm_config(
        kv_connector="SimpleCPUOffloadConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"cpu_bytes_to_use": _BYTES_PER_BLOCK * 8},
    )
    connector = SimpleCPUOffloadConnector(
        vllm_config=vllm_config,
        role=KVConnectorRole.SCHEDULER,
        kv_cache_config=_make_ring_kv_cache_config(),
    )
    scheduler = connector.scheduler_manager
    assert scheduler.fa_gidx == 0
    cpu_groups = scheduler.cpu_kv_cache_config.kv_cache_groups
    assert [g.kv_cache_spec.prefix_cacheable for g in cpu_groups] == [True, False]
    # Only the paged group takes part in CPU-side hit lookup.
    attention_groups = scheduler.cpu_coordinator.attention_groups
    assert [g.group_ids for g in attention_groups] == [[0]]


@pytest.mark.parametrize(
    "connector, prefix_caching, worker_blocks, expected_blocks",
    [
        (None, True, 37, None),
        ("NixlConnector", True, 37, None),
        ("SimpleCPUOffloadConnector", False, 37, None),
        ("SimpleCPUOffloadConnector", True, 37, 37),
        ("MultiConnector", True, 37, 37),
        ("SimpleCPUOffloadConnector", True, None, None),
    ],
)
def test_engine_queries_offload_blocks_only_when_enabled(
    monkeypatch, connector, prefix_caching, worker_blocks, expected_blocks
) -> None:
    """Only active offload queries workers, which must return a block count."""
    config = create_vllm_config(
        kv_connector=connector or "SimpleCPUOffloadConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "connectors": [
                {"kv_connector": "SimpleCPUOffloadConnector", "kv_role": "kv_both"}
            ]
        },
    )
    if connector is None:
        config.kv_transfer_config = None
    config.cache_config.enable_prefix_caching = prefix_caching
    kv_config = KVCacheConfig(num_blocks=16, kv_cache_tensors=[], kv_cache_groups=[])
    executor = MagicMock()
    executor.get_kv_cache_specs.return_value = [{}]
    executor.get_simple_cpu_offload_num_blocks.return_value = worker_blocks
    engine = object.__new__(engine_core_module.EngineCore)
    engine.model_executor = executor
    monkeypatch.setattr(
        engine_core_module, "get_kv_cache_configs", lambda *args: [kv_config]
    )

    if worker_blocks is None:
        with pytest.raises(AssertionError, match="workers must report"):
            engine._initialize_kv_caches(config)
        return
    engine._initialize_kv_caches(config)

    assert executor.get_simple_cpu_offload_num_blocks.call_count == (
        expected_blocks is not None
    )
    assert config.cache_config.simple_cpu_offload_num_blocks == expected_blocks
