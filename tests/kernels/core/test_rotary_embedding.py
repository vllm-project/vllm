# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for miscellaneous utilities."""

import pytest
import torch

from tests.kernels.utils import opcheck
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding, get_rope


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rotary_dim", [32, 64])
@pytest.mark.parametrize("max_position", [512, 8192])
def test_dynamic_ntk_encoder_packed_sequences(
    default_vllm_config, monkeypatch, device, dtype, rotary_dim, max_position
):
    """Packed requests must match independently scaled, fresh RoPE instances."""
    torch.manual_seed(0)
    head_size = 64
    lengths = [512, 7, 1] if max_position == 512 else [7, 2048, 1, 2049, 4096]
    positions = torch.cat([torch.arange(n) for n in lengths]).to(device)
    # MRV2 may leave stale positions in graph padding after a longer batch.
    positions = torch.cat((positions, torch.arange(1, 6, device=device)))
    is_padding = torch.arange(positions.numel(), device=device) >= sum(lengths)
    q = torch.randn(positions.numel(), head_size * 3, dtype=dtype, device=device)
    k = torch.randn_like(q)
    rope = get_rope(
        head_size=head_size,
        max_position=max_position,
        dtype=dtype,
        rope_parameters={
            "rope_type": "dynamic",
            "rope_theta": 1000.0,
            "factor": 2.0,
            "max_trained_positions": 2048,
            "partial_rotary_factor": rotary_dim / head_size,
            "apply_per_sequence": True,
        },
    ).to(device)
    with set_forward_context(None, default_vllm_config, is_padding=is_padding):
        actual_q, actual_k = rope(positions, q.clone(), k.clone())

        # Exercise the XPU query-only fallback without requiring XPU hardware.
        def reject_xpu_kernel(*args, **kwargs):
            pytest.fail("The XPU rotary kernel requires a key tensor")

        with monkeypatch.context() as patch:
            patch.setattr("vllm._custom_ops.rotary_embedding", reject_xpu_kernel)
            query_only, no_key = rope.forward_xpu(positions, q.clone())
        assert no_key is None
    positions = positions.masked_fill(is_padding, 0)
    start = 0
    for length in lengths:
        base = 1000.0
        if length > 2048:
            base *= (2.0 * length / 2048 - 1.0) ** (rotary_dim / (rotary_dim - 2))
        with torch.device(device):
            reference = RotaryEmbedding(
                head_size, rotary_dim, length, base, True, dtype
            )
        end = start + length
        expected_q, expected_k = reference.forward_native(
            positions[start:end], q[start:end], k[start:end]
        )
        # Long-position trigonometry differs slightly between CPU/GPU and
        # scalar/vector frequency construction; BF16 can round a few ULPs apart.
        atol = 5e-4 if dtype == torch.float32 else 0.016
        rtol = 1e-5 if dtype == torch.float32 else 0.016
        torch.testing.assert_close(
            actual_q[start:end], expected_q, atol=atol, rtol=rtol
        )
        torch.testing.assert_close(
            actual_k[start:end], expected_k, atol=atol, rtol=rtol
        )
        torch.testing.assert_close(
            query_only[start:end], expected_q, atol=atol, rtol=rtol
        )
        start = end

    # The same module must not retain long-request scaling on a later call.
    short_length = min(1024, max_position)
    short_positions = torch.arange(short_length, device=device)
    short_q, _ = rope(short_positions, q[:short_length].clone())
    default_rope = RotaryEmbedding(
        head_size, rotary_dim, short_length, 1000.0, True, dtype
    ).to(device)
    expected_q, _ = default_rope(short_positions, q[:short_length].clone())
    torch.testing.assert_close(short_q, expected_q, atol=0, rtol=0)


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
