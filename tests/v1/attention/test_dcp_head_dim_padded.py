# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for HEAD_DIM_PADDED and NaN guard in DCP kernels.

Tests that:
1. correct_attn_out works with non-power-of-2 HEAD_DIM (e.g. MLA's 576)
2. NaN * 0 guard: when a DCP rank has zero local KV tokens (LSE = -inf),
   the output is zeroed instead of NaN
3. dcp_lse_combine_triton works with non-power-of-2 HEAD_DIM
"""

import pytest
import torch

from vllm.v1.attention.ops.common import CPTritonContext, correct_attn_out


@pytest.mark.skipif(torch.accelerator.device_count() < 1, reason="CUDA is required.")
@pytest.mark.parametrize("head_dim", [32, 64, 128, 192, 576])
@pytest.mark.parametrize("is_base_e", [True, False])
def test_correct_attn_out_non_power_of_2_head_dim(
    head_dim: int, is_base_e: bool
) -> None:
    """correct_attn_out should produce correct results for any HEAD_DIM,
    including non-power-of-2 values like 576 (MLA compressed KV)."""
    device = "cuda"
    B, H, N = 4, 8, 2
    torch.manual_seed(42)

    out = torch.randn(B, H, head_dim, device=device, dtype=torch.float32)
    lses = torch.randn(N, B, H, device=device, dtype=torch.float32)
    cp_rank = 0

    out_orig = out.clone()
    lses_clone = lses.clone()

    ctx = CPTritonContext()
    result_out, result_lse = correct_attn_out(
        out, lses, cp_rank, ctx, is_lse_base_on_e=is_base_e
    )

    assert result_out.shape == (B, H, head_dim)
    assert result_lse.shape == (B, H)
    assert not torch.isnan(result_out).any(), "Output contains NaN"
    assert not torch.isinf(result_out).any(), "Output contains Inf"
    assert not torch.isnan(result_lse).any(), "LSE contains NaN"

    # Verify against manual computation for a single element
    for b in range(min(B, 2)):
        for h in range(min(H, 2)):
            lse_vals = lses_clone[:, b, h]
            lse_max = lse_vals.max()
            if is_base_e:
                log_sum = torch.log(torch.exp(lse_vals - lse_max).sum()) + lse_max
                factor = torch.exp(lse_vals[cp_rank] - log_sum)
            else:
                log_sum = (
                    torch.log2(torch.pow(2.0, (lse_vals - lse_max).float()).sum())
                    + lse_max
                )
                factor = torch.pow(2.0, (lse_vals[cp_rank] - log_sum).float())
            expected = out_orig[b, h] * factor
            torch.testing.assert_close(
                result_out[b, h],
                expected,
                rtol=1e-3,
                atol=1e-3,
                msg=f"Mismatch at b={b}, h={h} with head_dim={head_dim}",
            )


@pytest.mark.skipif(torch.accelerator.device_count() < 1, reason="CUDA is required.")
@pytest.mark.parametrize("is_base_e", [True, False])
def test_correct_attn_out_nan_guard(is_base_e: bool) -> None:
    """When a DCP rank has zero local KV tokens, its LSE is -inf and its
    attention output is NaN. The kernel should produce 0, not NaN."""
    device = "cuda"
    B, H, D, N = 2, 4, 64, 2
    cp_rank = 0

    out = torch.full((B, H, D), float("nan"), device=device, dtype=torch.float32)
    lses = torch.zeros(N, B, H, device=device, dtype=torch.float32)
    # Rank 0 has -inf LSE (no local KV tokens)
    lses[cp_rank] = float("-inf")
    # Rank 1 has valid LSE
    lses[1] = 1.0

    ctx = CPTritonContext()
    result_out, result_lse = correct_attn_out(
        out, lses, cp_rank, ctx, is_lse_base_on_e=is_base_e
    )

    # The output for rank 0 should be zeroed out (not NaN)
    assert not torch.isnan(result_out).any(), (
        f"NaN guard failed: output still contains NaN. "
        f"Max abs value: {result_out.abs().max().item()}"
    )
    torch.testing.assert_close(
        result_out,
        torch.zeros_like(result_out),
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.skipif(torch.accelerator.device_count() < 1, reason="CUDA is required.")
@pytest.mark.parametrize("head_dim", [192, 576])
@pytest.mark.parametrize("is_base_e", [True, False])
def test_a2a_unpack_combine_non_power_of_2(head_dim: int, is_base_e: bool) -> None:
    """dcp_a2a_unpack_combine should work with non-power-of-2 HEAD_DIM.

    The unpack_combine kernel uses HEAD_DIM_PADDED for its accumulator
    and masked loads/stores. We test it directly by constructing a
    recv_buffer that mimics what all-to-all would produce.
    """
    from vllm.v1.attention.ops.dcp_alltoall import (
        _dcp_a2a_lse_pack_dim,
        _dcp_a2a_unpack_combine,
        _lse_weighted_combine,
    )

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.float32  # Use float32 to avoid pack_send's power-of-2 issue
    world_size, B, h_per_rank = 2, 4, 4

    lse_pack_dim = _dcp_a2a_lse_pack_dim(dtype)  # 1 for float32

    # Build recv_buffer directly: [N, B, H_per_rank, D + lse_pack_dim]
    # This is what the all-to-all would produce
    recv_buffer = torch.empty(
        (world_size, B, h_per_rank, head_dim + lse_pack_dim),
        device=device,
        dtype=dtype,
    )
    # Fill output portion and LSE portion
    outputs_ref = torch.randn(
        world_size, B, h_per_rank, head_dim, device=device, dtype=dtype
    )
    lses_ref = torch.randn(
        world_size, B, h_per_rank, device=device, dtype=torch.float32
    )
    recv_buffer[:, :, :, :head_dim] = outputs_ref
    # Pack LSE into the last element(s)
    for n in range(world_size):
        for b in range(B):
            for h in range(h_per_rank):
                recv_buffer[n, b, h, head_dim] = lses_ref[n, b, h]

    actual_out, actual_lse = _dcp_a2a_unpack_combine(
        recv_buffer,
        head_dim,
        lse_pack_dim,
        return_lse=True,
        is_lse_base_on_e=is_base_e,
    )

    # Reference
    expected_out, expected_lse = _lse_weighted_combine(
        outputs_ref.float(),
        lses_ref,
        return_lse=True,
        is_lse_base_on_e=is_base_e,
    )

    assert actual_out.shape == (B, h_per_rank, head_dim)
    assert not torch.isnan(actual_out).any(), "A2A output contains NaN"
    torch.testing.assert_close(
        actual_out.float(),
        expected_out.float(),
        rtol=1e-4,
        atol=1e-4,
    )
    torch.testing.assert_close(actual_lse, expected_lse, rtol=1e-4, atol=1e-4)
