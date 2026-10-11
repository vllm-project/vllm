# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for orphan-row padding corruption in the NVFP4 experts quantization
kernel (silu_and_mul_scaled_fp4_experts_quant / scaled_fp4_experts_quant).

When expert_offsets[-1] < m_topk, rows beyond the last expert ("orphan rows")
fall through the expert lookup (linear scan or binary search) and default to
expert_idx=0, rowIdx_in_expert=0.  Without the fix, these rows overwrite
expert 0's scale factor at row 0.

This can happen in production with expert parallelism, where tokens routed
to non-local experts get topk_ids=-1 and are excluded from expert_offsets.
"""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform

if not current_platform.has_device_capability(100):
    pytest.skip(
        reason="Nvfp4 requires compute capability of 10 or above.",
        allow_module_level=True,
    )


def _make_blockscale_offsets(offsets, num_experts, device):
    """Compute 128-row-aligned blockscale offsets matching the swizzled
    SF layout used by cvt_quant_to_fp4_get_sf_out_offset (M-tile = 128)."""
    bs = [0]
    for i in range(num_experts):
        rows = offsets[i + 1] - offsets[i]
        bs.append(bs[-1] + (rows + 127) // 128 * 128)
    return torch.tensor(bs, dtype=torch.int32, device=device)


@pytest.mark.parametrize(
    "num_experts,m_topk,last_offset",
    [
        (8, 32, 24),
        (16, 64, 48),
        (32, 128, 96),
        (64, 256, 192),
        (32, 1024, 768),
    ],
    ids=["E8_m32", "E16_m64", "E32_m128", "E64_m256", "E32_m1024"],
)
@torch.inference_mode()
def test_fp4_experts_quant_orphan_row_padding(
    num_experts,
    m_topk,
    last_offset,
):
    """Orphan rows (beyond expert_offsets[-1]) must not corrupt expert 0's
    scale factors.

    Creates a gap: expert_offsets[-1] = last_offset < m_topk.
    Rows in [last_offset, m_topk) are orphan padding rows.
    Runs the kernel twice — once with clean padding, once with NaN padding —
    and verifies that expert 0's scales are identical.
    """
    device = "cuda"
    k = 128

    base = last_offset // num_experts
    remainder = last_offset % num_experts
    offsets = [0]
    for i in range(num_experts):
        n = base + (1 if i < remainder else 0)
        offsets.append(offsets[-1] + n)
    assert offsets[-1] == last_offset

    expert_offsets = torch.tensor(offsets, dtype=torch.int32, device=device)
    blockscale_offsets = _make_blockscale_offsets(offsets, num_experts, device)
    input_global_scale = torch.ones(num_experts, dtype=torch.float32, device=device)

    c1 = torch.randn(m_topk, k * 2, dtype=torch.bfloat16, device=device) * 0.1

    # Clean run: padding rows are zero
    c1_clean = c1.clone()
    c1_clean[last_offset:] = 0.0
    _, scales_clean = ops.silu_and_mul_scaled_fp4_experts_quant(
        c1_clean, input_global_scale, expert_offsets, blockscale_offsets, 1
    )

    # Dirty run: padding rows are NaN
    c1_dirty = c1.clone()
    c1_dirty[last_offset:] = float("nan")
    _, scales_dirty = ops.silu_and_mul_scaled_fp4_experts_quant(
        c1_dirty, input_global_scale, expert_offsets, blockscale_offsets, 1
    )

    # Real rows' scales must be identical regardless of padding content
    for e in range(num_experts):
        s, end = offsets[e], offsets[e + 1]
        if end <= s:
            continue
        assert torch.equal(scales_clean[s:end].float(), scales_dirty[s:end].float()), (
            f"Expert {e} scales corrupted by orphan padding rows "
            f"(E={num_experts}, m_topk={m_topk}, last_offset={last_offset})"
        )


@pytest.mark.parametrize(
    "num_experts,m_topk,num_real",
    [
        (8, 32, 16),
        (16, 64, 32),
        (32, 128, 64),
        (64, 256, 128),
        (32, 1024, 512),
    ],
    ids=["E8_m32", "E16_m64", "E32_m128", "E64_m256", "E32_m1024"],
)
@torch.inference_mode()
def test_fp4_experts_quant_no_cross_row_scale_corruption(
    num_experts,
    m_topk,
    num_real,
):
    """When expert_offsets[-1] == m_topk (no orphan rows), changing data in
    later rows must not affect earlier rows' scale factors.

    The warp shuffle in cvt_warp_fp16_to_fp4 pairs threads within the same
    row, so there is no cross-row contamination mechanism.  This test guards
    against regressions that might introduce one.
    """
    device = "cuda"
    k = 128

    base = m_topk // num_experts
    remainder = m_topk % num_experts
    offsets = [0]
    for i in range(num_experts):
        n = base + (1 if i < remainder else 0)
        offsets.append(offsets[-1] + n)
    assert offsets[-1] == m_topk

    expert_offsets = torch.tensor(offsets, dtype=torch.int32, device=device)
    blockscale_offsets = _make_blockscale_offsets(offsets, num_experts, device)
    input_global_scale = torch.ones(num_experts, dtype=torch.float32, device=device)

    c1 = torch.randn(m_topk, k * 2, dtype=torch.bfloat16, device=device) * 0.1

    # Clean run
    _, scales_clean = ops.silu_and_mul_scaled_fp4_experts_quant(
        c1.clone(), input_global_scale, expert_offsets, blockscale_offsets, 1
    )

    # Dirty run: zero out rows beyond num_real
    c1_dirty = c1.clone()
    c1_dirty[num_real:] = 0
    _, scales_dirty = ops.silu_and_mul_scaled_fp4_experts_quant(
        c1_dirty, input_global_scale, expert_offsets, blockscale_offsets, 1
    )

    # Check only real rows within experts that contain real data
    for e in range(num_experts):
        s, end = offsets[e], offsets[e + 1]
        real_end = min(end, num_real)
        if real_end <= s:
            continue
        assert torch.equal(
            scales_clean[s:real_end].float(),
            scales_dirty[s:real_end].float(),
        ), (
            f"Expert {e} scales corrupted by data changes in later rows "
            f"(E={num_experts}, m_topk={m_topk}, real={num_real})"
        )
