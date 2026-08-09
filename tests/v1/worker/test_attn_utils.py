# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVQuantMode, MambaSpec
from vllm.v1.worker.gpu.attn_utils import (
    _reshape_kv_cache,
    _reshape_mamba_kv_cache,
    zero_mamba_kv_cache,
)
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


@pytest.mark.parametrize("packing", [None, (8, 32)])
def test_reshape_mamba_kv_cache_preserves_block_layout(packing):
    spec = MambaSpec(
        block_size=1,
        shapes=((2,), (4,)),
        dtypes=(torch.float32, torch.float16),
    )
    num_blocks = 3
    page_size = spec.page_size_bytes
    block_stride = page_size if packing is None else packing[1]
    raw = torch.full((num_blocks * block_stride,), -1, dtype=torch.int8)

    cache = _reshape_mamba_kv_cache(raw, spec, num_blocks, packing)

    expected_offset = 0 if packing is None else packing[0]
    assert cache.shape == (num_blocks, 1, 1, page_size)
    assert cache.stride() == (block_stride, page_size, page_size, 1)
    assert cache.data_ptr() == raw.data_ptr() + expected_offset

    cache.fill_(0)
    raw_blocks = raw.view(num_blocks, block_stride)
    assert (
        torch.count_nonzero(
            raw_blocks[:, expected_offset : expected_offset + page_size]
        )
        == 0
    )
    if packing is not None:
        assert torch.all(raw_blocks[:, :expected_offset] == -1)
        assert torch.all(raw_blocks[:, expected_offset + page_size :] == -1)

    layer = SimpleNamespace(
        get_state_shape=lambda: spec.shapes,
        get_state_dtype=lambda: spec.dtypes,
    )
    MambaBase.bind_kv_cache(layer, cache)
    for state, shape in zip(layer.kv_cache, spec.shapes):
        assert state.shape == (num_blocks, *shape)
        assert state.stride(0) * state.element_size() == block_stride


@pytest.mark.parametrize("state_container", [list, tuple])
def test_zero_mamba_kv_cache_only_zeros_state_cache(state_container):
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
    groups = [
        [
            AttentionGroup(
                backend=FakeFlashAttentionBackend,
                layer_names=["mamba"],
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
    mamba_backing = torch.ones(4, 6)
    mamba_states = state_container([mamba_backing[:, ::2], mamba_backing[:, 1::2]])
    attention_cache = torch.ones(4)
    forward_context = {
        "mamba": SimpleNamespace(kv_cache=mamba_states),
        "attention": SimpleNamespace(kv_cache=attention_cache),
    }

    zero_mamba_kv_cache(groups, forward_context)

    assert all(torch.count_nonzero(state) == 0 for state in mamba_states)
    assert torch.count_nonzero(mamba_backing) == 0
    assert torch.all(attention_cache == 1)


def test_zero_mamba_kv_cache_deduplicates_shared_views():
    mamba_spec = MambaSpec(
        block_size=1,
        shapes=((2,),),
        dtypes=(torch.float32,),
    )
    group = AttentionGroup(
        backend=FakeFlashAttentionBackend,
        layer_names=["mamba", "mamba_alias"],
        kv_cache_spec=mamba_spec,
        kv_cache_group_id=0,
    )
    state = torch.ones(4, 2)
    forward_context = {
        "mamba": SimpleNamespace(kv_cache=(state,)),
        "mamba_alias": SimpleNamespace(kv_cache=(state,)),
    }
    version = state._version

    zero_mamba_kv_cache([[group]], forward_context)

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
