# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.v1.kv_offload.sparse.hisparse_runtime as hisparse_runtime_module
import vllm.v1.worker.gpu.attn_utils as attn_utils_module
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVQuantMode,
)
from vllm.v1.worker.gpu.attn_utils import _allocate_kv_cache, _reshape_kv_cache
from vllm.v1.worker.utils import AttentionGroup


class _FakeSharedHostRegion:
    def __init__(self) -> None:
        self.cleanup_calls = 0

    def cleanup(self) -> None:
        self.cleanup_calls += 1


@pytest.fixture
def registered_shared_region():
    region = _FakeSharedHostRegion()
    hisparse_runtime_module._SHARED_HOST_REGIONS.append(region)
    yield region
    hisparse_runtime_module._INDEXER_SOURCES.clear()
    if region in hisparse_runtime_module._SHARED_HOST_REGIONS:
        hisparse_runtime_module._SHARED_HOST_REGIONS.remove(region)


def test_allocate_kv_cache_rolls_back_shared_region_on_device_failure(
    monkeypatch, registered_shared_region
):
    region = registered_shared_region
    host_tensor = torch.empty(8, dtype=torch.int8)
    kv_cache_config = SimpleNamespace(
        kv_cache_tensors=[
            SimpleNamespace(
                size=host_tensor.numel(),
                shared_by=["host"],
                host_resident=True,
                block_pool_id=None,
                block_stride=0,
            ),
            SimpleNamespace(
                size=8,
                shared_by=["device"],
                host_resident=False,
                block_pool_id=0,
                block_stride=0,
            ),
        ],
        kv_cache_groups=[
            SimpleNamespace(layer_names=["host"]),
            SimpleNamespace(layer_names=["device"]),
        ],
        hisparse_host_num_blocks=1,
        hisparse_host_block_stride=4096,
        hisparse_shared_host_pool=True,
    )
    monkeypatch.setattr(
        attn_utils_module,
        "allocate_hisparse_host_pools",
        lambda *args, **kwargs: ([host_tensor], region),
    )

    def fail_device_allocation(*args, **kwargs):
        raise RuntimeError("device allocation failed")

    monkeypatch.setattr(attn_utils_module.torch, "zeros", fail_device_allocation)

    with pytest.raises(RuntimeError, match="device allocation failed"):
        _allocate_kv_cache(
            kv_cache_config,
            shared_layers={},
            device=torch.device("cpu"),
            vllm_config=SimpleNamespace(),
        )

    assert region.cleanup_calls == 1
    assert region not in hisparse_runtime_module._SHARED_HOST_REGIONS


def test_init_kv_cache_rolls_back_shared_region_on_worker_failure(
    monkeypatch, registered_shared_region
):
    region = registered_shared_region
    kv_cache_config = SimpleNamespace()
    vllm_config = SimpleNamespace(
        attention_config=SimpleNamespace(hisparse_config=SimpleNamespace()),
        scheduler_config=SimpleNamespace(max_num_seqs=1),
        model_config=SimpleNamespace(max_model_len=1),
        max_concurrent_batches=1,
    )
    monkeypatch.setattr(attn_utils_module, "get_shared_kv_cache_layers", lambda _: {})
    monkeypatch.setattr(
        attn_utils_module,
        "_allocate_kv_cache",
        lambda *args, **kwargs: ({}, region),
    )
    monkeypatch.setattr(
        attn_utils_module,
        "_reshape_kv_cache",
        lambda *args, **kwargs: {},
    )

    def fail_worker_initialization(**kwargs):
        tensor = torch.empty(1)
        hisparse_runtime_module._INDEXER_SOURCES["layer"] = (tensor, tensor)
        raise RuntimeError("worker initialization failed")

    monkeypatch.setattr(
        attn_utils_module, "init_hisparse_worker", fail_worker_initialization
    )

    with pytest.raises(RuntimeError, match="worker initialization failed"):
        attn_utils_module.init_kv_cache(
            runner_kv_caches=[],
            forward_context={},
            kv_cache_config=kv_cache_config,
            attn_groups=[],
            device=torch.device("cpu"),
            cache_dtype="auto",
            kernel_block_sizes=[],
            vllm_config=vllm_config,
            block_tables=SimpleNamespace(),
        )

    assert region.cleanup_calls == 1
    assert region not in hisparse_runtime_module._SHARED_HOST_REGIONS
    assert not hisparse_runtime_module._INDEXER_SOURCES


class FakeFlashAttentionBackend:
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        assert not include_num_layers_dimension
        return (0, 1, 2, 3, 4)


class FakeHNDFlashAttentionBackend(FakeFlashAttentionBackend):
    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        assert not include_num_layers_dimension
        return (0, 1, 3, 2, 4)


def test_reshape_kv_cache_preserves_shared_host_row_stride():
    num_blocks = 3
    spec = FullAttentionSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float32,
    )
    row_stride = spec.page_size_bytes + 32
    backing = torch.zeros(num_blocks * row_stride, dtype=torch.int8)
    raw_tensors = {
        "layer": torch.as_strided(
            backing,
            size=(num_blocks, spec.page_size_bytes),
            stride=(row_stride, 1),
        )
    }
    attn_groups = [
        AttentionGroup(
            backend=FakeFlashAttentionBackend,
            layer_names=["layer"],
            kv_cache_spec=spec,
            kv_cache_group_id=0,
        )
    ]

    kv_cache = _reshape_kv_cache(
        attn_groups,
        raw_tensors,
        "auto",
        [spec.block_size],
        {},
    )["layer"]

    assert kv_cache.shape == (num_blocks, 2, 2, 1, 2)
    assert kv_cache.stride(0) == row_stride // spec.dtype.itemsize
    assert kv_cache[1].storage_offset() == row_stride // spec.dtype.itemsize


def test_reshape_padded_flash_attention_kv_cache_strides_by_page():
    num_blocks = 3
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float32,
        page_size_padded=384,
    )
    assert spec.real_page_size_bytes == 256

    raw_tensors = {
        "layer": torch.zeros(spec.page_size_bytes * num_blocks, dtype=torch.int8)
    }
    attn_groups = [
        AttentionGroup(
            backend=FakeFlashAttentionBackend,
            layer_names=["layer"],
            kv_cache_spec=spec,
            kv_cache_group_id=0,
        )
    ]

    kv_cache = _reshape_kv_cache(
        attn_groups,
        raw_tensors,
        "auto",
        [spec.block_size],
        {},
    )["layer"]

    assert kv_cache.shape == (num_blocks, 2, 16, 1, 2)
    assert kv_cache.stride(0) == spec.page_size_bytes // 4
    assert kv_cache.stride(1) == spec.real_page_size_bytes // 2 // 4
    assert kv_cache[1, 0].storage_offset() == spec.page_size_bytes // 4
    assert (
        kv_cache[1, 1].storage_offset()
        == (spec.page_size_bytes + spec.real_page_size_bytes // 2) // 4
    )


def test_reshape_padded_hnd_flash_attention_kv_cache_strides_by_page():
    num_blocks = 3
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=3,
        head_size=2,
        dtype=torch.float32,
        page_size_padded=1024,
    )
    assert spec.real_page_size_bytes == 768

    raw_tensors = {
        "layer": torch.zeros(spec.page_size_bytes * num_blocks, dtype=torch.int8)
    }
    attn_groups = [
        AttentionGroup(
            backend=FakeHNDFlashAttentionBackend,
            layer_names=["layer"],
            kv_cache_spec=spec,
            kv_cache_group_id=0,
        )
    ]

    kv_cache = _reshape_kv_cache(
        attn_groups,
        raw_tensors,
        "auto",
        [spec.block_size],
        {},
    )["layer"]

    assert kv_cache.shape == (num_blocks, 2, 16, 3, 2)
    assert kv_cache.stride(0) == spec.page_size_bytes // 4
    assert kv_cache.stride(1) == spec.real_page_size_bytes // 2 // 4
    assert kv_cache.stride(2) == 2
    assert kv_cache.stride(3) == spec.block_size * spec.head_size
    assert kv_cache[1, 0].storage_offset() == spec.page_size_bytes // 4
    assert (
        kv_cache[1, 1].storage_offset()
        == (spec.page_size_bytes + spec.real_page_size_bytes // 2) // 4
    )
    assert (
        kv_cache[1, 1, 3, 2].storage_offset()
        == (
            spec.page_size_bytes
            + spec.real_page_size_bytes // 2
            + 3 * spec.head_size * 4
            + 2 * spec.block_size * spec.head_size * 4
        )
        // 4
    )


class FakeDiffKVBackend:
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, num_kv_heads, head_size * 2)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        assert not include_num_layers_dimension
        return (0, 1, 2, 3)


def test_reshape_padded_diff_kv_cache_does_not_infer_kv_dim():
    num_blocks = 3
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float32,
        page_size_padded=384,
    )

    raw_tensors = {
        "layer": torch.zeros(spec.page_size_bytes * num_blocks, dtype=torch.int8)
    }
    attn_groups = [
        AttentionGroup(
            backend=FakeDiffKVBackend,
            layer_names=["layer"],
            kv_cache_spec=spec,
            kv_cache_group_id=0,
        )
    ]

    kv_cache = _reshape_kv_cache(
        attn_groups,
        raw_tensors,
        "auto",
        [spec.block_size],
        {},
    )["layer"]

    assert kv_cache.shape == (num_blocks, 16, 1, 4)
    assert kv_cache.stride(0) == spec.page_size_bytes // 4
    assert kv_cache.stride(1) == 4


class FakePerTokenScaleBackend:
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, 2, block_size, num_kv_heads, head_size + 4)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        assert not include_num_layers_dimension
        return (0, 1, 2, 3, 4)


def test_reshape_padded_quantized_kv_cache_preserves_scale_stride():
    num_blocks = 3
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.int8,
        kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
        page_size_padded=384,
    )
    assert spec.real_page_size_bytes == 128
    assert spec.page_size_bytes == 384

    raw_tensors = {
        "layer": torch.zeros(spec.page_size_bytes * num_blocks, dtype=torch.int8)
    }
    attn_groups = [
        AttentionGroup(
            backend=FakePerTokenScaleBackend,
            layer_names=["layer"],
            kv_cache_spec=spec,
            kv_cache_group_id=0,
        )
    ]

    kv_cache = _reshape_kv_cache(
        attn_groups,
        raw_tensors,
        "int8_per_token_head",
        [spec.block_size],
        {},
    )["layer"]

    assert kv_cache.shape == (num_blocks, 2, 16, 1, 8)
    assert kv_cache.stride(0) == spec.page_size_bytes
    assert kv_cache.stride(1) == 16 * 1 * 8
    assert kv_cache[1, 1].storage_offset() == spec.page_size_bytes + 16 * 1 * 8
