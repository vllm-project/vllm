# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheTensor,
    KVQuantMode,
    MambaSpec,
)
from vllm.v1.worker.gpu.attn_utils import (
    _reshape_kv_cache,
)
from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2
from vllm.v1.worker.gpu_model_runner import GPUModelRunner as GPUModelRunnerV1
from vllm.v1.worker.utils import AttentionGroup


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


def test_reshape_packed_mamba_kv_cache_preserves_block_layout():
    spec = MambaSpec(
        block_size=1,
        shapes=((2,), (4,)),
        dtypes=(torch.float32, torch.float16),
    )
    num_blocks = 3
    page_size = spec.page_size_bytes
    offset = 8
    block_stride = 32
    raw = torch.full((num_blocks * block_stride,), -1, dtype=torch.int8)
    group = AttentionGroup(
        backend=FakeFlashAttentionBackend,
        layer_names=["mamba"],
        kv_cache_spec=spec,
        kv_cache_group_id=0,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=raw.numel(),
                shared_by=["mamba"],
                offset=offset,
                block_stride=block_stride,
            )
        ],
        kv_cache_groups=[],
    )

    cache = _reshape_kv_cache(
        [group],
        {"mamba": raw},
        "auto",
        [spec.block_size],
        {},
        kv_cache_config,
    )["mamba"]

    assert cache.shape == (num_blocks, 1, 1, page_size)
    assert cache.stride() == (block_stride, page_size, page_size, 1)
    assert cache.data_ptr() == raw.data_ptr() + offset

    cache.fill_(0)
    raw_blocks = raw.view(num_blocks, block_stride)
    assert torch.count_nonzero(raw_blocks[:, offset : offset + page_size]) == 0
    assert torch.all(raw_blocks[:, :offset] == -1)
    assert torch.all(raw_blocks[:, offset + page_size :] == -1)

    layer = SimpleNamespace(
        get_state_shape=lambda: spec.shapes,
        get_state_dtype=lambda: spec.dtypes,
    )
    MambaBase.bind_kv_cache(layer, cache)
    for state, shape in zip(layer.kv_cache, spec.shapes):
        assert state.shape == (num_blocks, *shape)
        assert state.stride(0) * state.element_size() == block_stride


def _make_hybrid_attn_groups(mamba_layer_names):
    mamba_spec = MambaSpec(
        block_size=1,
        shapes=((2,), (3,)),
        dtypes=(torch.float32, torch.float32),
    )
    attention_spec = FullAttentionSpec(
        block_size=1,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    return [
        [
            AttentionGroup(
                backend=FakeFlashAttentionBackend,
                layer_names=mamba_layer_names,
                kv_cache_spec=mamba_spec,
                kv_cache_group_id=0,
            ),
            AttentionGroup(
                backend=FakeFlashAttentionBackend,
                layer_names=["attention"],
                kv_cache_spec=attention_spec,
                kv_cache_group_id=0,
            ),
        ]
    ]


def _run_post_kv_cache_wake_up(runner_cls, groups, forward_context):
    runner = SimpleNamespace(
        attn_groups=groups,
        compilation_config=SimpleNamespace(static_forward_context=forward_context),
    )
    if runner_cls is GPUModelRunnerV1:
        runner.init_fp8_kv_scales = lambda: None
    else:
        runner.block_tables = SimpleNamespace(
            init_block_table_layout_tensors=lambda: None
        )
    runner_cls.post_kv_cache_wake_up(runner)


@pytest.mark.parametrize(
    "runner_cls", [GPUModelRunnerV1, GPUModelRunnerV2], ids=["mrv1", "mrv2"]
)
def test_post_kv_cache_wake_up_zeros_only_mamba_state(runner_cls):
    groups = _make_hybrid_attn_groups(["mamba"])
    mamba_backing = torch.ones(4, 6)
    mamba_states = (mamba_backing[:, ::2], mamba_backing[:, 1::2])
    attention_cache = torch.ones(4)
    forward_context = {
        "mamba": SimpleNamespace(kv_cache=mamba_states),
        "attention": SimpleNamespace(kv_cache=attention_cache),
    }

    _run_post_kv_cache_wake_up(runner_cls, groups, forward_context)

    assert all(torch.count_nonzero(state) == 0 for state in mamba_states)
    assert torch.count_nonzero(mamba_backing) == 0
    assert torch.all(attention_cache == 1)


def test_post_kv_cache_wake_up_deduplicates_shared_views():
    groups = _make_hybrid_attn_groups(["mamba", "mamba_alias"])
    state = torch.ones(4, 2)
    forward_context = {
        "mamba": SimpleNamespace(kv_cache=(state,)),
        "mamba_alias": SimpleNamespace(kv_cache=(state,)),
        "attention": SimpleNamespace(kv_cache=torch.ones(4)),
    }
    version = state._version

    _run_post_kv_cache_wake_up(GPUModelRunnerV1, groups, forward_context)

    assert state._version == version + 1
    assert torch.count_nonzero(state) == 0


class FakeHNDFlashAttentionBackend(FakeFlashAttentionBackend):
    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        assert not include_num_layers_dimension
        return (0, 1, 3, 2, 4)


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


class FakeKVFirstBackend:
    """ROCm-style backend that puts K and V ahead of the block dim."""

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_block_dim(
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> int:
        return 1


def _kv_first_setup(shared_by: list[str]):
    num_blocks = 3
    attn_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float32,
    )
    mamba_spec = MambaSpec(
        block_size=16,
        shapes=((64,),),
        dtypes=(torch.float32,),
    )
    assert attn_spec.page_size_bytes == mamba_spec.page_size_bytes == 256

    raw_tensor = torch.zeros(attn_spec.page_size_bytes * num_blocks, dtype=torch.int8)
    raw_tensors = {name: raw_tensor for name in shared_by}
    attn_groups = [
        AttentionGroup(
            backend=FakeKVFirstBackend,
            layer_names=["attn"],
            kv_cache_spec=attn_spec,
            kv_cache_group_id=0,
        )
    ]
    if "mamba" in shared_by:
        attn_groups.append(
            AttentionGroup(
                backend=FakeKVFirstBackend,
                layer_names=["mamba"],
                kv_cache_spec=mamba_spec,
                kv_cache_group_id=1,
            )
        )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(size=raw_tensor.numel(), shared_by=list(shared_by))
        ],
        kv_cache_groups=[],
    )

    kv_caches = _reshape_kv_cache(
        attn_groups,
        raw_tensors,
        "auto",
        [attn_spec.block_size] * len(attn_groups),
        {},
        kv_cache_config,
    )
    return num_blocks, kv_caches["attn"]


def test_reshape_kv_first_kv_cache_pages_blocks_when_shared_with_mamba():
    num_blocks, kv_cache = _kv_first_setup(["attn", "mamba"])

    assert kv_cache.shape == (2, num_blocks, 16, 1, 2)
    # Block b has to own page b, so its K and V sit side by side within it.
    page = 16 * 1 * 2 * 2
    for block in range(num_blocks):
        assert kv_cache[0, block].storage_offset() == block * page
        assert kv_cache[1, block].storage_offset() == block * page + page // 2


def test_reshape_kv_first_kv_cache_keeps_layout_without_mamba():
    num_blocks, kv_cache = _kv_first_setup(["attn"])

    assert kv_cache.shape == (2, num_blocks, 16, 1, 2)
    # Nothing else indexes this allocation by page, so K and V stay split into
    # one contiguous half each.
    assert kv_cache[1, 0].storage_offset() == num_blocks * 16 * 1 * 2
