# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from math import prod

import pytest
import torch

from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.cpu_attn import CPUAttentionBackend
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.v1.attention.backends.flash_attn_diffkv import FlashAttentionDiffKVBackend
from vllm.v1.attention.backends.flashinfer import FlashInferBackend
from vllm.v1.attention.backends.flex_attention import FlexAttentionBackend
from vllm.v1.attention.backends.tree_attn import TreeAttentionBackend
from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend
from vllm.v1.attention.backends.utils import (
    get_kv_cache_layout,
    set_kv_cache_layout,
)
from vllm.v1.kv_cache_interface import AttentionSpec, FullAttentionSpec
from vllm.v1.worker import mamba_utils


def _build_full_attn_spec(
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    dtype: torch.dtype,
    pack_size: int,
) -> FullAttentionSpec:
    # Minimal valid FullAttentionSpec for layout tests.
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        head_size_v=head_size,
        dtype=dtype,
        pack_size=pack_size,
    )


def _compute_layout_ref(
    backend: AttentionBackend,
    kv_cache_spec: AttentionSpec,
    layer_idx: int,
    kernel_block_size: int,
    num_blocks: int,
    enable_hybrid_attn_mamba_layout: bool,
):
    """
    Reference implementation that mirrors the pre-refactor logic:
    - grouping layout in `_reshape_kv_cache_tensors`
    - followed by `_update_hybrid_attention_mamba_layout` when enabled.
    """
    dtype = kv_cache_spec.dtype
    block_size = kv_cache_spec.block_size
    attn_pack_size = kv_cache_spec.pack_size

    num_blocks_per_kv_block = block_size // kernel_block_size
    kernel_num_blocks = num_blocks * num_blocks_per_kv_block

    # Match `_reshape_kv_cache_tensors`: use `kernel_num_blocks` and
    # `kernel_block_size` when querying backend shape.
    kv_cache_shape_logical = backend.get_kv_cache_shape(
        kernel_num_blocks,
        kernel_block_size,
        kv_cache_spec.num_kv_heads,
        kv_cache_spec.head_size,
        cache_dtype_str="auto",
    )

    try:
        kv_cache_stride_order = backend.get_kv_cache_stride_order()
        assert len(kv_cache_stride_order) == len(kv_cache_shape_logical)
    except (AttributeError, NotImplementedError):
        kv_cache_stride_order = tuple(range(len(kv_cache_shape_logical)))

    # Physical shape (the one used for allocation / as_strided).
    kv_cache_shape = tuple(kv_cache_shape_logical[i] for i in kv_cache_stride_order)

    inv_order = [
        kv_cache_stride_order.index(i) for i in range(len(kv_cache_stride_order))
    ]

    # Base contiguous strides in physical layout.
    base_stride = list(torch.empty(kv_cache_shape).stride())

    storage_offset = 0
    if attn_pack_size > 1:
        # Match the original `_reshape_kv_cache_tensors` logic.
        kernel_blocks_idx = kv_cache_shape.index(kernel_num_blocks)
        base_stride[kernel_blocks_idx] *= attn_pack_size
        dtype_size = get_dtype_size(dtype)
        num_element_per_page = kv_cache_spec.page_size_bytes // dtype_size
        num_element_per_attn_pack = (
            num_element_per_page // num_blocks_per_kv_block // attn_pack_size
        )
        attn_pack_idx = layer_idx % attn_pack_size
        storage_offset = attn_pack_idx * num_element_per_attn_pack

    # Logical KV tensor after the initial reshape.
    kv = torch.empty_strided(
        size=kv_cache_shape,
        stride=tuple(base_stride),
        dtype=dtype,
    ).permute(*inv_order)

    # Optional hybrid attention+mamba layout update.
    # We analytically update the stride to match `_update_hybrid_attention_mamba_layout`
    # without actually changing the underlying storage.
    if (
        enable_hybrid_attn_mamba_layout
        and isinstance(kv_cache_spec, AttentionSpec)
        and kv.shape[0] == 2
    ):
        hidden_size = prod(kv.shape[2:])
        attn_pack_size_for_layout = kv_cache_spec.pack_size
        kv_stride = kv.stride()
        kv_stride = (
            hidden_size,
            2 * hidden_size * attn_pack_size_for_layout,
            *kv_stride[2:],
        )
        return kv.shape, kv_stride, storage_offset

    return kv.shape, kv.stride(), storage_offset


def _compute_layout_new(
    backend: AttentionBackend,
    kv_cache_spec: AttentionSpec,
    layer_idx: int,
    kernel_block_size: int,
    num_blocks: int,
    enable_hybrid_attn_mamba_layout: bool,
):
    """
    Layout computed via the new helper `_get_hybrid_attention_mamba_layout`.
    """
    dtype = kv_cache_spec.dtype
    block_size = kv_cache_spec.block_size

    num_blocks_per_kv_block = block_size // kernel_block_size
    kernel_num_blocks = num_blocks * num_blocks_per_kv_block

    kv_cache_shape_logical = backend.get_kv_cache_shape(
        kernel_num_blocks,
        kernel_block_size,
        kv_cache_spec.num_kv_heads,
        kv_cache_spec.head_size,
        cache_dtype_str="auto",
    )

    try:
        kv_cache_stride_order = backend.get_kv_cache_stride_order()
        assert len(kv_cache_stride_order) == len(kv_cache_shape_logical)
    except (AttributeError, NotImplementedError):
        kv_cache_stride_order = tuple(range(len(kv_cache_shape_logical)))

    kv_cache_shape = tuple(kv_cache_shape_logical[i] for i in kv_cache_stride_order)
    inv_order = [
        kv_cache_stride_order.index(i) for i in range(len(kv_cache_stride_order))
    ]

    kv_cache_stride = tuple(torch.empty(kv_cache_shape).stride())
    storage_offset = 0

    if enable_hybrid_attn_mamba_layout:
        kv_cache_stride, storage_offset = mamba_utils.get_hybrid_attention_mamba_layout(
            kv_cache_shape=kv_cache_shape,
            kv_cache_stride=kv_cache_stride,
            kv_cache_spec=kv_cache_spec,
            layer_idx=layer_idx,
            kernel_num_blocks=kernel_num_blocks,
            kernel_block_size=kernel_block_size,
        )

    kv = torch.empty_strided(
        size=kv_cache_shape,
        stride=kv_cache_stride,
        dtype=dtype,
    ).permute(*inv_order)

    # Sanity: group_size should not affect logical shape.
    assert kv.shape == tuple(
        backend.get_kv_cache_shape(
            num_blocks,
            block_size,
            kv_cache_spec.num_kv_heads,
            kv_cache_spec.head_size,
            cache_dtype_str="auto",
        )
    )

    return kv.shape, kv.stride(), storage_offset


@pytest.mark.parametrize(
    "backend_cls",
    [
        CPUAttentionBackend,
        FlashAttentionBackend,
        FlashInferBackend,
        TritonAttentionBackend,
        FlashAttentionDiffKVBackend,
        FlexAttentionBackend,
        TreeAttentionBackend,
    ],
)
@pytest.mark.parametrize("pack_size", [1, 2, 4])
@pytest.mark.parametrize("enable_hybrid_attn_mamba_layout", [False, True])
@pytest.mark.parametrize("cache_layout", ["NHD", "HND"])
def test_hybrid_attention_mamba_layout_matches_reference(
    backend_cls: type[AttentionBackend],
    cache_layout: str,
    pack_size: int,
    enable_hybrid_attn_mamba_layout: bool,
):
    if (not enable_hybrid_attn_mamba_layout) and pack_size > 1:
        pytest.skip("pack_size > 1 only occurs when hybrid attention+mamba is enabled")
    # Explicitly test both cache layouts for backends that depend on it
    # (FlashAttentionBackend / FlashInferBackend). Other backends ignore
    # this setting, but it is harmless to apply globally.
    set_kv_cache_layout(cache_layout)  # sets the override
    # Invalidate the cached value in get_kv_cache_layout so the override
    # takes effect for this test case.
    get_kv_cache_layout.cache_clear()  # type: ignore[attr-defined]

    block_size = 16
    num_kv_heads = 2
    head_size = 32
    dtype = torch.float16

    num_blocks = 100
    kernel_block_size = 16
    layer_idx = 1

    kv_cache_spec = _build_full_attn_spec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        pack_size=pack_size,
    )

    backend = backend_cls

    ref_shape, ref_stride, ref_offset = _compute_layout_ref(
        backend=backend,
        kv_cache_spec=kv_cache_spec,
        layer_idx=layer_idx,
        kernel_block_size=kernel_block_size,
        num_blocks=num_blocks,
        enable_hybrid_attn_mamba_layout=enable_hybrid_attn_mamba_layout,
    )

    new_shape, new_stride, new_offset = _compute_layout_new(
        backend=backend,
        kv_cache_spec=kv_cache_spec,
        layer_idx=layer_idx,
        kernel_block_size=kernel_block_size,
        num_blocks=num_blocks,
        enable_hybrid_attn_mamba_layout=enable_hybrid_attn_mamba_layout,
    )

    assert ref_shape == new_shape
    assert ref_stride == new_stride
    # storage_offset only differs from zero when group_size > 1.
    if pack_size > 1:
        assert new_offset == ref_offset
    else:
        assert ref_offset == 0
        assert new_offset == 0
