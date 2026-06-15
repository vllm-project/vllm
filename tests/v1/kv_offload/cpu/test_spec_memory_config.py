# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for CPU offload memory backend selection in offloading specs."""

import mmap
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

import vllm.v1.kv_offload.tiering.spec as tiering_spec_module
from vllm.config import KVTransferConfig
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from vllm.v1.kv_offload.cpu.memory import (
    HUGEPAGE_2MB,
    CPUOffloadMemoryBackend,
)
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec

PAGE_SIZE = mmap.PAGESIZE
BLOCK_SIZE = 16


def _make_vllm_config(
    extra_config: dict[str, Any], *, world_size: int = 1
) -> SimpleNamespace:
    return SimpleNamespace(
        kv_transfer_config=KVTransferConfig(
            kv_connector="OffloadingConnector",
            kv_role="kv_both",
            kv_connector_extra_config=extra_config,
        ),
        parallel_config=SimpleNamespace(
            world_size=world_size,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        cache_config=SimpleNamespace(
            block_size=BLOCK_SIZE,
            enable_prefix_caching=True,
            hash_block_size=None,
        ),
        kv_events_config=None,
        instance_id="spec-memory-config-test",
    )


def _make_kv_cache_config(
    *, num_blocks: int = 1, gpu_bytes_per_block: int = PAGE_SIZE
) -> KVCacheConfig:
    layer_names = ["layer"]
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=num_blocks * gpu_bytes_per_block,
                shared_by=layer_names,
            )
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names,
                FullAttentionSpec(
                    block_size=BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )


def _recording_shared_region(created: list[Any]) -> type:
    class RecordingSharedOffloadRegion:
        def __init__(
            self,
            instance_id: str,
            num_blocks: int,
            rank: int | None,
            kv_bytes_per_block: int,
            cpu_page_size: int,
            memory_config: Any = None,
        ) -> None:
            self.instance_id = instance_id
            self.num_blocks = num_blocks
            self.rank = rank
            self.kv_bytes_per_block = kv_bytes_per_block
            self.cpu_page_size = cpu_page_size
            self.memory_config = memory_config
            self.mmap_path = memory_config.mmap_path(instance_id)
            self.kv_view = memoryview(bytearray(num_blocks * kv_bytes_per_block))
            created.append(self)

        def create_kv_memoryview(self) -> memoryview:
            return self.kv_view

        def cleanup(self) -> None:
            self.kv_view.release()

    return RecordingSharedOffloadRegion


def test_cpu_offloading_spec_default_keeps_torch_backend() -> None:
    extra_config = {"cpu_bytes_to_use": 4 * PAGE_SIZE}
    spec = CPUOffloadingSpec(
        _make_vllm_config(extra_config),
        _make_kv_cache_config(),
    )

    assert spec.cpu_memory_config.backend == CPUOffloadMemoryBackend.DEFAULT
    assert spec.cpu_memory_config.effective_backend == CPUOffloadMemoryBackend.SHM
    assert spec.num_blocks == 4


@pytest.mark.parametrize("backend", ["shm", "hugetlbfs"])
def test_cpu_offloading_spec_rejects_explicit_shared_backend(
    backend: str,
) -> None:
    extra_config: dict[str, Any] = {
        "cpu_bytes_to_use": 4 * PAGE_SIZE,
        "cpu_memory_backend": backend,
    }

    with pytest.raises(ValueError, match="only supported by TieringOffloadingSpec"):
        CPUOffloadingSpec(
            _make_vllm_config(extra_config),
            _make_kv_cache_config(),
        )


def test_tiering_offloading_spec_default_keeps_shared_shm_backend() -> None:
    extra_config = {
        "cpu_bytes_to_use": 4 * PAGE_SIZE,
        "spec_name": "TieringOffloadingSpec",
    }
    spec = TieringOffloadingSpec(
        _make_vllm_config(extra_config),
        _make_kv_cache_config(),
    )

    assert spec.cpu_memory_config.backend == CPUOffloadMemoryBackend.DEFAULT
    assert spec.cpu_memory_config.effective_backend == CPUOffloadMemoryBackend.SHM
    assert spec.num_blocks == 4


def test_tiering_spec_threads_memory_config_to_scheduler_and_worker(
    monkeypatch, tmp_path
) -> None:
    created_regions: list[Any] = []
    secondary_views: list[memoryview] = []

    class FakeWorker:
        def __init__(self, *args: Any, mmap_region: Any, **kwargs: Any) -> None:
            self.mmap_region = mmap_region

    class FakeSecondaryTier:
        tier_type = "fake"

    def create_secondary_tier(
        tier_config: dict[str, Any],
        primary_kv_view: memoryview,
        offloading_spec: TieringOffloadingSpec,
    ) -> FakeSecondaryTier:
        secondary_views.append(primary_kv_view)
        return FakeSecondaryTier()

    monkeypatch.setattr(
        tiering_spec_module,
        "SharedOffloadRegion",
        _recording_shared_region(created_regions),
    )
    monkeypatch.setattr(
        tiering_spec_module,
        "CPUOffloadingWorker",
        FakeWorker,
    )
    monkeypatch.setattr(
        tiering_spec_module.SecondaryTierFactory,
        "create_secondary_tier",
        staticmethod(create_secondary_tier),
    )
    monkeypatch.setattr(
        tiering_spec_module.torch.accelerator,
        "current_device_index",
        lambda: 1,
    )

    extra_config = {
        "cpu_bytes_to_use": 8 * PAGE_SIZE,
        "spec_name": "TieringOffloadingSpec",
        "cpu_memory_backend": "hugetlbfs",
        "cpu_memory_path": str(tmp_path),
        "secondary_tiers": [{"type": "fake"}],
    }
    spec = TieringOffloadingSpec(
        _make_vllm_config(extra_config, world_size=2),
        _make_kv_cache_config(),
    )

    spec.get_manager()
    worker = spec.create_worker(MagicMock())

    scheduler_region = created_regions[0]
    worker_region = created_regions[1]
    assert scheduler_region.rank is None
    assert worker_region.rank == 1
    assert scheduler_region.memory_config is spec.cpu_memory_config
    assert worker_region.memory_config is spec.cpu_memory_config
    assert scheduler_region.mmap_path.startswith(str(tmp_path))
    assert worker_region.mmap_path.startswith(str(tmp_path))
    assert worker.mmap_region is worker_region
    assert secondary_views == [scheduler_region.kv_view]


def test_tiering_num_blocks_uses_logical_cpu_bytes_not_hugepage_padding(
    tmp_path,
) -> None:
    extra_config = {
        "cpu_bytes_to_use": PAGE_SIZE + 1,
        "spec_name": "TieringOffloadingSpec",
        "cpu_memory_backend": "hugetlbfs",
        "cpu_memory_path": str(tmp_path),
    }
    spec = TieringOffloadingSpec(
        _make_vllm_config(extra_config),
        _make_kv_cache_config(),
    )

    logical_size = spec.num_blocks * spec.kv_bytes_per_offloaded_block
    assert spec.num_blocks == 1
    assert logical_size == PAGE_SIZE
    assert spec.cpu_memory_config.mapped_size(logical_size) == HUGEPAGE_2MB
