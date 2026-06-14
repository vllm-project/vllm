# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for cross-layer KV cache support in SimpleCPUOffloadConnector."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.config import (
    CacheConfig,
    DeviceConfig,
    KVTransferConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BLOCK_SIZE = 16
HEAD_SIZE = 16
NUM_KV_HEADS = 1
DTYPE = torch.float16
NUM_LAYERS = 4
NUM_GPU_BLOCKS = 8
# page_size_bytes per layer:
# block_size * num_kv_heads * head_size * 2 (K+V) * element_size
PAGE_SIZE_PER_LAYER = BLOCK_SIZE * NUM_KV_HEADS * HEAD_SIZE * 2 * DTYPE.itemsize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_kv_cache_config(
    num_blocks: int = NUM_GPU_BLOCKS,
    num_layers: int = NUM_LAYERS,
) -> KVCacheConfig:
    """Build a single-group KVCacheConfig with multiple layers."""
    layer_names = [f"layer_{i}" for i in range(num_layers)]
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=DTYPE,
    )
    groups = [KVCacheGroupSpec(layer_names, spec)]
    tensors = [
        KVCacheTensor(
            size=int(PAGE_SIZE_PER_LAYER * num_blocks),
            shared_by=[name],
        )
        for name in layer_names
    ]
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
    )


def _make_vllm_config() -> VllmConfig:
    """Minimal VllmConfig for unit tests (no GPU)."""
    model_config = ModelConfig(
        model="facebook/opt-125m",
        trust_remote_code=True,
        dtype="float16",
        seed=42,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=16,
        max_num_batched_tokens=64,
        max_model_len=10000,
        enable_chunked_prefill=True,
        is_encoder_decoder=False,
    )
    cache_config = CacheConfig(
        block_size=BLOCK_SIZE,
        gpu_memory_utilization=0.9,
        enable_prefix_caching=True,
    )
    kv_transfer_config = KVTransferConfig(
        kv_connector="SimpleCPUOffloadConnector",
        kv_role="kv_both",
    )
    return VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        kv_transfer_config=kv_transfer_config,
        device_config=DeviceConfig("cpu"),
    )


def _make_cross_layer_tensor(
    num_blocks: int = NUM_GPU_BLOCKS,
    num_layers: int = NUM_LAYERS,
) -> torch.Tensor:
    """Create a fake cross-layer KV cache tensor on CPU.

    Mimics what allocate_uniform_kv_caches produces: a single flat
    buffer of size page_size_per_layer * num_blocks * num_layers.
    """
    total_bytes = int(PAGE_SIZE_PER_LAYER * num_blocks * num_layers)
    return torch.zeros(total_bytes, dtype=torch.int8, device="cpu")


# ---------------------------------------------------------------------------
# Tests: Connector property
# ---------------------------------------------------------------------------


class TestConnectorProperty:
    """Test that SimpleCPUOffloadConnector exposes cross-layer preference."""

    def test_prefer_cross_layer_blocks_returns_true(self):
        """Connector should prefer cross-layer blocks."""
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorRole,
        )
        from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (  # noqa: E501
            SimpleCPUOffloadConnector,
        )

        vllm_config = _make_vllm_config()
        kv_cache_config = _make_kv_cache_config()
        connector = SimpleCPUOffloadConnector(
            vllm_config, KVConnectorRole.WORKER, kv_cache_config
        )
        assert connector.prefer_cross_layer_blocks is True

    def test_register_cross_layers_delegates_to_worker(self):
        """Connector should delegate registration to worker handler."""
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorRole,
        )
        from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (  # noqa: E501
            SimpleCPUOffloadConnector,
        )

        vllm_config = _make_vllm_config()
        kv_cache_config = _make_kv_cache_config()
        connector = SimpleCPUOffloadConnector(
            vllm_config, KVConnectorRole.WORKER, kv_cache_config
        )

        # Replace worker handler with a mock
        mock_worker = MagicMock()
        connector.worker_handler = mock_worker

        fake_tensor = torch.zeros(10, dtype=torch.int8)
        fake_backend = MagicMock()
        connector.register_cross_layers_kv_cache(fake_tensor, fake_backend)

        mock_worker.register_cross_layers_kv_cache.assert_called_once_with(
            fake_tensor, fake_backend
        )

    def test_scheduler_role_does_not_delegate(self):
        """Scheduler-role connector has no worker, should be a no-op."""
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorRole,
        )
        from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (  # noqa: E501
            SimpleCPUOffloadConnector,
        )

        vllm_config = _make_vllm_config()
        kv_cache_config = _make_kv_cache_config()
        connector = SimpleCPUOffloadConnector(
            vllm_config, KVConnectorRole.SCHEDULER, kv_cache_config
        )
        assert connector.worker_handler is None

        # Should not raise — just a no-op
        fake_tensor = torch.zeros(10, dtype=torch.int8)
        fake_backend = MagicMock()
        connector.register_cross_layers_kv_cache(fake_tensor, fake_backend)


# ---------------------------------------------------------------------------
# Tests: Worker registration
# ---------------------------------------------------------------------------


def _make_worker(kv_cache_config=None, cpu_capacity_bytes=None):
    """Create a SimpleCPUOffloadWorker with CUDA dependencies mocked out."""
    from vllm.v1.simple_kv_offload.worker import SimpleCPUOffloadWorker

    if kv_cache_config is None:
        kv_cache_config = _make_kv_cache_config()
    if cpu_capacity_bytes is None:
        cpu_capacity_bytes = int(PAGE_SIZE_PER_LAYER * NUM_LAYERS * 4)

    vllm_config = _make_vllm_config()
    worker = SimpleCPUOffloadWorker(vllm_config, kv_cache_config, cpu_capacity_bytes)
    worker._backend = MagicMock()
    return worker


def _register_cross_layer(worker, kv_cache=None):
    """Call register_cross_layers_kv_cache with CUDA mocked out."""
    if kv_cache is None:
        num_layers = len(worker.kv_cache_config.kv_cache_groups[0].layer_names)
        kv_cache = _make_cross_layer_tensor(num_layers=num_layers)

    mock_stream_cls = MagicMock()
    mock_stream_cls.priority_range.return_value = (-1, 0)

    with (
        patch(
            "vllm.v1.simple_kv_offload.worker.is_pin_memory_available",
            return_value=False,
        ),
        patch("torch.cuda.Stream", mock_stream_cls),
    ):
        worker.register_cross_layers_kv_cache(kv_cache, MagicMock())
    return kv_cache


class TestWorkerRegistration:
    """Test SimpleCPUOffloadWorker.register_cross_layers_kv_cache."""

    def test_tensor_shapes(self):
        """GPU and CPU tensors should have correct shapes."""
        worker = _make_worker()
        _register_cross_layer(worker)

        # Should produce a single "cross_layer" entry
        assert "cross_layer" in worker.gpu_kv_caches
        assert len(worker.gpu_kv_caches) == 1

        gpu_tensor = worker.gpu_kv_caches["cross_layer"]
        expected_page_size = int(PAGE_SIZE_PER_LAYER * NUM_LAYERS)

        # GPU tensor: (num_gpu_blocks, page_size * num_layers)
        assert gpu_tensor.shape == (NUM_GPU_BLOCKS, expected_page_size)
        assert gpu_tensor.dtype == torch.int8

        # CPU tensor: (num_cpu_blocks, page_size * num_layers)
        assert "cross_layer" in worker.cpu_kv_caches
        cpu_tensor = worker.cpu_kv_caches["cross_layer"]
        assert cpu_tensor.shape[1] == expected_page_size
        assert cpu_tensor.dtype == torch.int8
        assert cpu_tensor.device == torch.device("cpu")

    def test_num_cpu_blocks(self):
        """CPU block count should be capacity / cross_layer_page_size."""
        worker = _make_worker()
        _register_cross_layer(worker)

        cross_layer_page_size = int(PAGE_SIZE_PER_LAYER * NUM_LAYERS)
        expected_cpu_blocks = worker.cpu_capacity_bytes // cross_layer_page_size
        assert worker.num_cpu_blocks == expected_cpu_blocks

    def test_backend_initialized(self):
        """Copy backend should be initialized with matching GPU/CPU dicts."""
        worker = _make_worker()
        _register_cross_layer(worker)

        worker._backend.init.assert_called_once()
        args = worker._backend.init.call_args
        gpu_caches_arg = args[0][0]
        cpu_caches_arg = args[0][1]

        # Same keys
        assert list(gpu_caches_arg.keys()) == list(cpu_caches_arg.keys())
        assert list(gpu_caches_arg.keys()) == ["cross_layer"]

        # Same page width (dim 1)
        assert (
            gpu_caches_arg["cross_layer"].shape[1]
            == cpu_caches_arg["cross_layer"].shape[1]
        )

    def test_gpu_tensor_shares_storage(self):
        """GPU tensor should be a view into the original cross-layer storage."""
        worker = _make_worker()
        kv_cache = _register_cross_layer(worker)

        original_data_ptr = kv_cache.untyped_storage().data_ptr()
        gpu_tensor = worker.gpu_kv_caches["cross_layer"]
        # The view should share the same underlying storage
        assert gpu_tensor.untyped_storage().data_ptr() == original_data_ptr

    def test_min_one_cpu_block(self):
        """Even with tiny CPU capacity, should get at least 1 CPU block."""
        worker = _make_worker(cpu_capacity_bytes=1)
        _register_cross_layer(worker)

        assert worker.num_cpu_blocks == 1

    def test_single_layer_degenerates_to_per_layer(self):
        """With 1 layer, cross-layer page_size equals per-layer page_size."""
        kv_cache_config = _make_kv_cache_config(num_layers=1)
        worker = _make_worker(
            kv_cache_config=kv_cache_config,
            cpu_capacity_bytes=int(PAGE_SIZE_PER_LAYER * 4),
        )
        _register_cross_layer(worker)

        gpu_tensor = worker.gpu_kv_caches["cross_layer"]
        assert gpu_tensor.shape == (NUM_GPU_BLOCKS, int(PAGE_SIZE_PER_LAYER))


# ---------------------------------------------------------------------------
# Tests: Multi-group rejection
# ---------------------------------------------------------------------------


class TestMultiGroupFallback:
    """Cross-layer requires a single KV cache group."""

    def test_multi_group_asserts(self):
        """register_cross_layers_kv_cache should reject multi-group configs."""
        spec = FullAttentionSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=NUM_KV_HEADS,
            head_size=HEAD_SIZE,
            dtype=DTYPE,
        )
        # Two groups — should be rejected
        kv_cache_config = KVCacheConfig(
            num_blocks=NUM_GPU_BLOCKS,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=int(PAGE_SIZE_PER_LAYER * NUM_GPU_BLOCKS),
                    shared_by=["layer_0"],
                ),
                KVCacheTensor(
                    size=int(PAGE_SIZE_PER_LAYER * NUM_GPU_BLOCKS),
                    shared_by=["layer_1"],
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer_0"], spec),
                KVCacheGroupSpec(["layer_1"], spec),
            ],
        )
        worker = _make_worker(kv_cache_config=kv_cache_config, cpu_capacity_bytes=10000)

        kv_cache = _make_cross_layer_tensor(num_layers=2)
        with pytest.raises(AssertionError):
            _register_cross_layer(worker, kv_cache)


# ---------------------------------------------------------------------------
# Tests: Storage size validation
# ---------------------------------------------------------------------------


class TestStorageValidation:
    """Test that misaligned storage sizes are caught."""

    def test_misaligned_storage_asserts(self):
        """Storage not divisible by cross-layer page size should assert."""
        worker = _make_worker(cpu_capacity_bytes=10000)

        # Create a tensor with an extra byte — not aligned to page size
        total_bytes = int(PAGE_SIZE_PER_LAYER * NUM_GPU_BLOCKS * NUM_LAYERS)
        kv_cache = torch.zeros(total_bytes + 1, dtype=torch.int8, device="cpu")

        with pytest.raises(AssertionError):
            _register_cross_layer(worker, kv_cache)
