# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the fused RoPE and flash-cache update CUDA op."""

import pytest
import torch

from tests.kernels.utils import DEFAULT_OPCHECK_TEST_UTILS, opcheck
from tests.v1.attention.utils import dense_kv_cache_views
from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheLayout

NUM_BLOCKS = 8
MAX_POS = 4096
SEED = 0

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")

CASES = [
    pytest.param(
        torch.float16,
        4,
        4,
        64,
        64,
        16,
        1,
        1,
        KVCacheLayout.LBNHC,
        0,
        True,
        id="mha-full-rope",
    ),
    pytest.param(
        torch.bfloat16,
        8,
        2,
        128,
        64,
        16,
        8,
        5,
        KVCacheLayout.LBHNC,
        0,
        False,
        id="gqa-partial-rope-short-mapping",
    ),
    pytest.param(
        torch.float16,
        8,
        1,
        80,
        80,
        32,
        32,
        32,
        KVCacheLayout.BLNHC,
        256,
        False,
        id="mqa-interleaved-cross-layer-padded-page",
    ),
    pytest.param(
        torch.float16,
        8,
        2,
        128,
        128,
        16,
        8,
        8,
        KVCacheLayout.LHBNC,
        0,
        True,
        id="gqa-fp16-head-block-layout",
    ),
    pytest.param(
        torch.bfloat16,
        8,
        2,
        128,
        128,
        16,
        8,
        8,
        KVCacheLayout.BLHNC,
        0,
        True,
        id="gqa-bf16-cross-layer",
    ),
    pytest.param(
        torch.bfloat16,
        32,
        8,
        256,
        128,
        32,
        17,
        13,
        KVCacheLayout.BHLNC,
        0,
        True,
        id="gqa-head256-multicycle-block-head-layer-layout",
    ),
    pytest.param(
        torch.float16,
        40,
        40,
        128,
        64,
        32,
        257,
        129,
        KVCacheLayout.BLNHC,
        256,
        False,
        id="mha-vector-value-large-padded-query",
    ),
    pytest.param(
        torch.bfloat16,
        32,
        8,
        82,
        64,
        32,
        17,
        13,
        KVCacheLayout.LBHNC,
        0,
        True,
        id="gqa-unaligned-head-stride-scalar-value",
    ),
]


def _make_cache_views(
    layout: KVCacheLayout,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    dtype: torch.dtype,
    device: torch.device,
    page_padding: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_layers = 1 if layout.is_layer_compact else 2
    dense_page_size = (
        block_size
        * num_kv_heads
        * 2
        * head_size
        * torch.empty((), dtype=dtype).element_size()
    )
    cache_spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        page_size_padded=(dense_page_size + page_padding if page_padding else None),
    )
    physical_cache = torch.zeros(
        num_layers * NUM_BLOCKS * cache_spec.page_size_bytes,
        dtype=torch.int8,
        device=device,
    )
    packed_cache = dense_kv_cache_views(
        physical_cache,
        cache_spec,
        num_blocks=NUM_BLOCKS,
        num_layers=num_layers,
        layout=layout,
    )[-1]
    key_cache, value_cache = packed_cache.transpose(1, 2).split(head_size, dim=-1)
    return key_cache, value_cache, physical_cache


def _assert_cache_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "dtype,offset,head_padding,cache_head_padding",
    [
        (torch.float16, 0, 0, 0),
        (torch.bfloat16, 0, 1, 0),
        (torch.bfloat16, 0, 0, 1),
        (torch.float16, 1, 0, 0),
    ],
)
@torch.inference_mode()
def test_fused_value_copy_preserves_strided_storage(
    dtype: torch.dtype,
    offset: int,
    head_padding: int,
    cache_head_padding: int,
) -> None:
    device = torch.device("cuda")
    set_random_seed(SEED)
    tokens, q_heads, kv_heads, head_size, block_size = 9, 16, 8, 128, 16

    def strided(
        shape: tuple[int, ...],
        strides: tuple[int, ...],
        storage_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        size = storage_offset + sum((n - 1) * s for n, s in zip(shape, strides)) + 3
        storage = torch.full((size,), -7, dtype=dtype, device=device)
        return storage.as_strided(shape, strides, storage_offset), storage

    query = torch.randn(tokens, q_heads, head_size, dtype=dtype, device=device)
    key = torch.randn(tokens, kv_heads, head_size, dtype=dtype, device=device)
    value_head_stride = head_size + head_padding
    value, value_storage = strided(
        (tokens, kv_heads, head_size),
        (kv_heads * value_head_stride + 1, value_head_stride, 1),
        offset,
    )
    value.copy_(torch.randn_like(value))
    original_value_storage = value_storage.clone()
    cache_shape = (2, block_size, kv_heads, head_size)
    key_head_stride = head_size + 4
    key_token_stride = kv_heads * key_head_stride + 1
    key_cache, key_storage = strided(
        cache_shape,
        (block_size * key_token_stride + 3, key_token_stride, key_head_stride, 1),
    )
    value_cache_head_stride = head_size + 8 + cache_head_padding
    value_token_stride = kv_heads * value_cache_head_stride + 3
    value_cache, cache_storage = strided(
        cache_shape,
        (
            block_size * value_token_stride + 5,
            value_token_stride,
            value_cache_head_stride,
            1,
        ),
        offset,
    )
    expected_key_storage = key_storage.clone()
    expected_value_storage = cache_storage.clone()
    expected_key_cache = expected_key_storage.as_strided(
        key_cache.shape,
        key_cache.stride(),
        key_cache.storage_offset(),
    )
    expected_value_cache = expected_value_storage.as_strided(
        value_cache.shape,
        value_cache.stride(),
        value_cache.storage_offset(),
    )
    slots = [0, 8, 16, -1, 3, 18, 2, 4, 1]
    slot_mapping = torch.tensor(slots, dtype=torch.long, device=device)
    positions = torch.arange(tokens, device=device)
    cos_sin_cache = torch.randn(tokens, 64, dtype=dtype, device=device)
    query_ref, key_ref = query.clone(), key.clone()
    ops.rotary_embedding(positions, query_ref, key_ref, head_size, cos_sin_cache, True)
    # The incumbent cache op cannot consume arbitrary source/cache strides.
    for token, slot in enumerate(slots):
        if slot >= 0:
            block, row = divmod(slot, block_size)
            expected_key_cache[block, row].copy_(key_ref[token])
            expected_value_cache[block, row].copy_(value[token])

    query_out = torch.empty_like(query)
    ops.fused_rope_and_reshape_cache_flash_q_out(
        query,
        key,
        value,
        query_out,
        positions,
        cos_sin_cache,
        True,
        key_cache,
        value_cache,
        slot_mapping,
    )
    torch.testing.assert_close(query_out, query_ref, rtol=0, atol=0)
    _assert_cache_equal(key_storage, expected_key_storage)
    _assert_cache_equal(cache_storage, expected_value_storage)
    _assert_cache_equal(value_storage, original_value_storage)


@pytest.mark.parametrize(
    (
        "dtype",
        "num_q_heads",
        "num_kv_heads",
        "head_size",
        "rotary_dim",
        "block_size",
        "num_rope_tokens",
        "num_cache_tokens",
        "layout",
        "page_padding",
        "is_neox",
    ),
    CASES,
)
@torch.inference_mode()
def test_fused_rope_and_reshape_cache_flash_q_out_matches_unfused(
    dtype: torch.dtype,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    rotary_dim: int,
    block_size: int,
    num_rope_tokens: int,
    num_cache_tokens: int,
    layout: KVCacheLayout,
    page_padding: int,
    is_neox: bool,
) -> None:
    device = torch.device("cuda")
    set_random_seed(SEED)

    packed_qkv = torch.randn(
        num_rope_tokens,
        (num_q_heads + 2 * num_kv_heads) * head_size,
        dtype=dtype,
        device=device,
    )
    query_flat, key_flat, value_flat = packed_qkv.split(
        [
            num_q_heads * head_size,
            num_kv_heads * head_size,
            num_kv_heads * head_size,
        ],
        dim=-1,
    )
    query = query_flat.view(num_rope_tokens, num_q_heads, head_size)
    key = key_flat.view(num_rope_tokens, num_kv_heads, head_size)
    value = value_flat.view(num_rope_tokens, num_kv_heads, head_size)

    cos_sin_cache = torch.randn(MAX_POS, rotary_dim, dtype=dtype, device=device)
    positions = torch.randperm(MAX_POS, device=device)[:num_rope_tokens]
    slot_mapping = torch.randperm(NUM_BLOCKS * block_size, device=device)[
        :num_cache_tokens
    ].to(torch.long)
    if num_cache_tokens >= 4:
        slot_mapping[1] = -1

    key_cache_q_out, value_cache_q_out, physical_cache_q_out = _make_cache_views(
        layout,
        num_kv_heads,
        head_size,
        block_size,
        dtype,
        device,
        page_padding,
    )
    key_cache_ref, value_cache_ref, physical_cache_ref = _make_cache_views(
        layout,
        num_kv_heads,
        head_size,
        block_size,
        dtype,
        device,
        page_padding,
    )
    scale = torch.ones(1, dtype=torch.float32, device=device)

    query_ref = query.clone()
    key_ref = key.clone()
    packed_qkv_q_out = packed_qkv.clone()
    query_q_out_flat, key_q_out_flat, value_q_out_flat = packed_qkv_q_out.split(
        [
            num_q_heads * head_size,
            num_kv_heads * head_size,
            num_kv_heads * head_size,
        ],
        dim=-1,
    )
    query_q_out_input = query_q_out_flat.view(num_rope_tokens, num_q_heads, head_size)
    key_q_out_input = key_q_out_flat.view(num_rope_tokens, num_kv_heads, head_size)
    value_q_out_input = value_q_out_flat.view(num_rope_tokens, num_kv_heads, head_size)
    packed_qkv_q_out_ref = packed_qkv_q_out.clone()
    query_out_buffer = torch.empty_like(
        query_q_out_input, memory_format=torch.contiguous_format
    )
    ops.fused_rope_and_reshape_cache_flash_q_out(
        query_q_out_input,
        key_q_out_input,
        value_q_out_input,
        query_out_buffer,
        positions,
        cos_sin_cache,
        is_neox,
        key_cache_q_out,
        value_cache_q_out,
        slot_mapping,
    )
    ops.rotary_embedding(
        positions,
        query_ref,
        key_ref,
        head_size,
        cos_sin_cache,
        is_neox,
    )
    ops.reshape_and_cache_flash(
        key_ref,
        value,
        key_cache_ref,
        value_cache_ref,
        slot_mapping,
        "auto",
        scale,
        scale,
    )

    torch.testing.assert_close(query_out_buffer, query_ref, rtol=0, atol=0)
    assert query_out_buffer.is_contiguous()
    torch.testing.assert_close(packed_qkv_q_out, packed_qkv_q_out_ref, rtol=0, atol=0)
    _assert_cache_equal(key_cache_q_out, key_cache_ref)
    _assert_cache_equal(value_cache_q_out, value_cache_ref)
    _assert_cache_equal(physical_cache_q_out, physical_cache_ref)

    if layout is KVCacheLayout.LBNHC:
        opcheck(
            torch.ops._C_cache_ops.fused_rope_and_reshape_cache_flash_q_out,
            (
                query_q_out_input,
                key_q_out_input,
                value_q_out_input,
                query_out_buffer,
                positions,
                cos_sin_cache,
                is_neox,
                key_cache_q_out,
                value_cache_q_out,
                slot_mapping,
            ),
            test_utils=DEFAULT_OPCHECK_TEST_UTILS,
        )


def test_fused_rope_and_reshape_cache_rejects_mismatched_dtypes() -> None:
    device = torch.device("cuda")
    query = torch.zeros(1, 1, 64, dtype=torch.float16, device=device)
    key = torch.zeros_like(query)
    value = torch.zeros_like(query)
    query_out = torch.empty_like(query)
    positions = torch.zeros(1, dtype=torch.long, device=device)
    cos_sin_cache = torch.zeros(1, 64, dtype=torch.float16, device=device)
    key_cache = torch.zeros(1, 16, 1, 64, dtype=torch.float32, device=device)
    value_cache = torch.zeros_like(key_cache)
    slot_mapping = torch.zeros(1, dtype=torch.long, device=device)

    with pytest.raises(RuntimeError, match="cache dtype must match query dtype"):
        ops.fused_rope_and_reshape_cache_flash_q_out(
            query,
            key,
            value,
            query_out,
            positions,
            cos_sin_cache,
            True,
            key_cache,
            value_cache,
            slot_mapping,
        )

    key_cache = torch.zeros(1, 16, 1, 64, dtype=torch.float16, device=device)
    value_cache = torch.zeros_like(key_cache)
    cos_sin_cache = cos_sin_cache.float()
    with pytest.raises(
        RuntimeError, match="cos_sin_cache dtype must match query dtype"
    ):
        ops.fused_rope_and_reshape_cache_flash_q_out(
            query,
            key,
            value,
            query_out,
            positions,
            cos_sin_cache,
            True,
            key_cache,
            value_cache,
            slot_mapping,
        )
