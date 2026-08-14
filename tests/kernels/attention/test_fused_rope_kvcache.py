# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the fused RoPE and flash-cache update CUDA op."""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

BLOCK_SIZE = 16
NUM_BLOCKS = 8
MAX_POS = 4096
SEED = 0

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")

CASES = [
    pytest.param(
        torch.float16,
        "auto",
        4,
        4,
        64,
        64,
        1,
        1,
        "NHD",
        True,
        id="mha-full-rope",
    ),
    pytest.param(
        torch.bfloat16,
        "auto",
        8,
        2,
        128,
        64,
        8,
        5,
        "HND",
        False,
        id="gqa-partial-rope-short-mapping",
    ),
    pytest.param(
        torch.float16,
        "auto",
        8,
        1,
        80,
        80,
        32,
        32,
        "NHD",
        False,
        id="mqa-interleaved-padding",
    ),
    pytest.param(
        torch.float16,
        "fp8_e4m3",
        8,
        2,
        128,
        128,
        8,
        8,
        "HND",
        True,
        id="gqa-fp8-cache",
    ),
]


def _make_cache_views(
    layout: str,
    num_kv_heads: int,
    head_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if layout == "NHD":
        storage = torch.zeros(
            NUM_BLOCKS,
            BLOCK_SIZE,
            num_kv_heads,
            2 * head_size,
            dtype=dtype,
            device=device,
        )
        packed_cache = storage.permute(0, 2, 1, 3)
    else:
        packed_cache = torch.zeros(
            NUM_BLOCKS,
            num_kv_heads,
            BLOCK_SIZE,
            2 * head_size,
            dtype=dtype,
            device=device,
        )
    return packed_cache.transpose(1, 2).split(head_size, dim=-1)


def _assert_cache_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    if actual.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        torch.testing.assert_close(actual.view(torch.uint8), expected.view(torch.uint8))
    else:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    (
        "dtype",
        "kv_cache_dtype",
        "num_q_heads",
        "num_kv_heads",
        "head_size",
        "rotary_dim",
        "num_rope_tokens",
        "num_cache_tokens",
        "layout",
        "is_neox",
    ),
    CASES,
)
@torch.inference_mode()
def test_fused_rope_and_reshape_cache_flash_q_out_matches_unfused(
    dtype: torch.dtype,
    kv_cache_dtype: str,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    rotary_dim: int,
    num_rope_tokens: int,
    num_cache_tokens: int,
    layout: str,
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
    slot_mapping = torch.randperm(NUM_BLOCKS * BLOCK_SIZE, device=device)[
        :num_cache_tokens
    ].to(torch.long)
    if num_cache_tokens >= 4:
        slot_mapping[1] = -1

    cache_dtype = current_platform.fp8_dtype() if kv_cache_dtype != "auto" else dtype
    key_cache_q_out, value_cache_q_out = _make_cache_views(
        layout, num_kv_heads, head_size, cache_dtype, device
    )
    key_cache_ref, value_cache_ref = _make_cache_views(
        layout, num_kv_heads, head_size, cache_dtype, device
    )
    k_scale = torch.tensor([0.7], dtype=torch.float32, device=device)
    v_scale = torch.tensor([1.3], dtype=torch.float32, device=device)

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
        k_scale,
        v_scale,
        kv_cache_dtype,
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
        kv_cache_dtype,
        k_scale,
        v_scale,
    )

    torch.testing.assert_close(query_out_buffer, query_ref, rtol=0, atol=0)
    assert query_out_buffer.is_contiguous()
    torch.testing.assert_close(packed_qkv_q_out, packed_qkv_q_out_ref, rtol=0, atol=0)
    _assert_cache_equal(key_cache_q_out, key_cache_ref)
    _assert_cache_equal(value_cache_q_out, value_cache_ref)
