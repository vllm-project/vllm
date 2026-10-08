# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for miscellaneous utilities."""

import pytest
import torch

from tests.kernels.utils import opcheck
from vllm.model_executor.layers.rotary_embedding import (
    LinearScalingRotaryEmbedding,
    RotaryEmbedding,
)


@pytest.mark.parametrize("scaled", [False, True])
@pytest.mark.parametrize("is_neox", [False, True])
def test_rotation_description_matches_partial_rotary_forward(
    default_vllm_config, scaled: bool, is_neox: bool
):
    args = (64, 32, 128, 10000, is_neox)
    rot = (
        LinearScalingRotaryEmbedding(*args, scaling_factors=2.0, dtype=torch.float32)
        if scaled
        else RotaryEmbedding(*args, dtype=torch.float32)
    )
    positions = torch.tensor([5, 0, 9, 2])
    query = torch.randn(4, 4 * 64, dtype=torch.bfloat16)
    key = torch.randn(4, 2 * 64, dtype=torch.bfloat16)
    expected = rot.forward_native(positions, query, key)

    rotation = rot.get_rotation(positions, query.dtype)

    assert rotation is not None
    assert rotation.positions is positions
    assert rotation.cos_sin is rot.cos_sin_cache
    assert rotation.cos_sin.dtype == query.dtype
    actual = RotaryEmbedding.forward_static(
        rotation.positions,
        query,
        key,
        rot.head_size,
        rotation.cos_sin.shape[-1],
        rotation.cos_sin,
        rotation.is_neox,
    )
    torch.testing.assert_close(actual, expected)


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
