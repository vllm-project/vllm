# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for miscellaneous utilities
"""

import pytest
import torch

from tests.kernels.utils import opcheck
from vllm import ir
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding


def _make_cos_sin_cache(
    max_positions: int, rotary_dim: int, device: str
) -> torch.Tensor:
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )
    t = torch.arange(max_positions, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # [max_pos, rotary_dim // 2]
    cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)  # [max_pos, rotary_dim]
    return cache.to(device)


def rotary_embedding_opcheck(
    rot,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None = None,
):
    cos_sin_cache = rot.cos_sin_cache.to(query.device, dtype=query.dtype)

    # ops.rotary_embedding() is a in-place operation
    # that updates the query and key tensors.
    opcheck(
        torch.ops._C.rotary_embedding,
        (positions, query, key, rot.head_size, cos_sin_cache, rot.is_neox_style),
    )


@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("max_position", [11, 4096, 32768])
@pytest.mark.parametrize("is_neox_style", [True, False])
@pytest.mark.parametrize("rotary_dim", [32])
@pytest.mark.parametrize("head_size", [32, 108])
@pytest.mark.parametrize("seq_len", [11, 1024])
@pytest.mark.parametrize("use_key", [True, False])
@pytest.mark.parametrize("head_stride_is_contiguous", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rotary_embedding_opcheck(
    default_vllm_config,
    dist_init,
    device,
    max_position,
    is_neox_style,
    rotary_dim,
    head_size,
    seq_len,
    use_key,
    head_stride_is_contiguous,
    dtype,
):
    batch_size = 1
    base = 10000
    num_heads = 7
    rot = RotaryEmbedding(
        head_size, rotary_dim, max_position, base, is_neox_style, dtype
    )

    positions = torch.randint(0, max_position, (batch_size, seq_len), device=device)
    head_stride = head_size + (64 if head_stride_is_contiguous else 0)

    query = torch.randn(
        batch_size, seq_len, num_heads, head_stride, dtype=dtype, device=device
    )
    key = torch.randn_like(query) if use_key else None
    query = query[..., :head_size]
    key = key[..., :head_size] if key is not None else None

    rotary_embedding_opcheck(rot, positions, query, key)

    # if we have a contiguous head stride, test the alternate
    # [..., num_heads * head_dim] shape/layout
    if head_stride_is_contiguous:
        rotary_embedding_opcheck(
            rot,
            positions,
            query.flatten(start_dim=-2),
            key.flatten(start_dim=-2) if key is not None else None,
        )


@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("is_neox_style", [True, False])
@pytest.mark.parametrize("cos_sin_format", ["standard", "deepseek"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rotary_embedding_ir_op_2d_vs_3d(
    default_vllm_config,
    dist_init,
    device,
    is_neox_style,
    cos_sin_format,
    dtype,
):
    """2D [T, H*head_size] and 3D [T, H, head_size] inputs must produce
    the same rotated output — exercises the reshape logic in
    _apply_deepseek_rope that was missing before the fix."""
    num_tokens, num_heads, head_size, rotary_dim = 16, 4, 64, 32
    max_positions = 128

    cos_sin_cache = _make_cos_sin_cache(max_positions, rotary_dim, device)
    positions = torch.randint(0, max_positions, (num_tokens,), device=device)

    query_3d = torch.randn(num_tokens, num_heads, head_size, dtype=dtype, device=device)
    key_3d = torch.randn_like(query_3d)
    query_2d = query_3d.reshape(num_tokens, num_heads * head_size).clone()
    key_2d = key_3d.reshape(num_tokens, num_heads * head_size).clone()

    q_3d_out, k_3d_out = ir.ops.rotary_embedding(
        positions,
        query_3d.clone(),
        key_3d.clone(),
        head_size,
        rotary_dim,
        cos_sin_cache,
        is_neox_style,
        cos_sin_format=cos_sin_format,
    )
    q_2d_out, k_2d_out = ir.ops.rotary_embedding(
        positions,
        query_2d,
        key_2d,
        head_size,
        rotary_dim,
        cos_sin_cache,
        is_neox_style,
        cos_sin_format=cos_sin_format,
    )

    atol = 1e-5 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(
        q_2d_out,
        q_3d_out.reshape(num_tokens, num_heads * head_size),
        atol=atol,
        rtol=0,
    )
    torch.testing.assert_close(
        k_2d_out,
        k_3d_out.reshape(num_tokens, num_heads * head_size),
        atol=atol,
        rtol=0,
    )


@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("is_neox_style", [True, False])
@pytest.mark.parametrize("cos_sin_format", ["standard", "deepseek"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rotary_embedding_query_only_matches_full(
    default_vllm_config,
    dist_init,
    device,
    is_neox_style,
    cos_sin_format,
    dtype,
):
    """rotary_embedding_query_only must return the same query tensor as the
    query component from rotary_embedding (verifies the query-only fast path)."""
    num_tokens, num_heads, head_size, rotary_dim = 16, 4, 64, 32
    max_positions = 128

    cos_sin_cache = _make_cos_sin_cache(max_positions, rotary_dim, device)
    positions = torch.randint(0, max_positions, (num_tokens,), device=device)
    query = torch.randn(num_tokens, num_heads * head_size, dtype=dtype, device=device)
    key = torch.randn_like(query)

    q_full, _ = ir.ops.rotary_embedding(
        positions,
        query.clone(),
        key.clone(),
        head_size,
        rotary_dim,
        cos_sin_cache,
        is_neox_style,
        cos_sin_format=cos_sin_format,
    )
    q_only = ir.ops.rotary_embedding_query_only(
        positions,
        query.clone(),
        head_size,
        rotary_dim,
        cos_sin_cache,
        is_neox_style,
        cos_sin_format=cos_sin_format,
    )

    atol = 1e-5 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(q_only, q_full, atol=atol, rtol=0)
