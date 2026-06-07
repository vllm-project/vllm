#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the ROCm RDNA3 fused MoE W4A16 HIP kernel (gfx1100).

Tests ``moe_gptq_gemm_rdna3`` against the dense ``gptq_gemm_rdna3`` as
reference: builds RDNA3-format weights (shuffled int32, synthesized qzeros),
runs the fused MoE kernel, and compares per-expert results.

Model parameters taken from:
  - cyankiwi/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit
    (hidden=2048, inter=768, E=128, top_k=8, G=32)
  - Qwen3.6-35B-A3B-GPTQ-W4A16-G32
    (hidden=2048, inter=512, E=256, top_k=8, G=32)

Run `pytest tests/kernels/quantization/test_rdna3_moe_w4a16.py`.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("RDNA3 MoE W4A16 kernel is ROCm-only", allow_module_level=True)

from vllm import _custom_ops as ops  # noqa: E402
from vllm.model_executor.layers.fused_moe.activation import (  # noqa: E402
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (  # noqa: E402
    moe_align_block_size,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (  # noqa: E402
    pack_quantized_values_into_int32,
)
from vllm.platforms.rocm import on_gfx1100  # noqa: E402
from vllm.scalar_type import scalar_types  # noqa: E402

device = "cuda"

gfx1100_only = pytest.mark.skipif(
    not (
        on_gfx1100()
        and hasattr(torch.ops, "_rocm_C")
        and hasattr(torch.ops._rocm_C, "moe_gptq_gemm_rdna3")
    ),
    reason="Requires gfx1100 with moe_gptq_gemm_rdna3 op",
)

# Model configurations: real K/N/top_k/group_size dims, E capped at 16 to
# fit in test GPU memory (full E=128/256 would need >20GB for weights alone).
# Kernel behavior is E-independent (per-expert tiling), so E=16 is sufficient.
MODEL_CONFIGS = [
    # cyankiwi/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit dims (E capped)
    pytest.param(16, 2048, 768, 8, 32, id="Qwen3-30B-A3B"),
    # Qwen3.6-35B-A3B-GPTQ-W4A16-G32 dims (E capped)
    pytest.param(16, 2048, 512, 8, 32, id="Qwen3.6-35B-A3B"),
]

# Token counts: decode (1), small batch (4), medium (16), prefill (64)
NUM_TOKENS = [1, 4, 16, 64, 256, 512]


def _make_packed_weights(E, K, N):
    """Create random 4-bit packed weights [E, K/8, N] int32 + shuffle."""
    w = torch.randint(0, 16, (E, K, N), dtype=torch.int32, device=device)
    packed = torch.zeros(E, K // 8, N, dtype=torch.int32, device=device)
    for i in range(8):
        packed |= (w[:, i::8, :] & 0xF) << (i * 4)
    g_idx = torch.empty(0, dtype=torch.int32, device=device)
    for e in range(E):
        we = packed[e].contiguous()
        ops.gptq_shuffle(we, g_idx, 4)
        packed[e] = we
    return packed


def _make_scales(E, groups, N, dtype):
    return torch.rand(E, groups, N, dtype=dtype, device=device) * 0.1


def _make_qzeros(E, groups, N):
    zeros = torch.full(
        (groups, N),
        scalar_types.uint4b8.bias - 1,
        dtype=torch.int32,
        device=device,
    )
    qz = pack_quantized_values_into_int32(
        zeros,
        scalar_types.uint4b8,
        packed_dim=1,
    )
    return qz.unsqueeze(0).expand(E, -1, -1).contiguous()


@gfx1100_only
@pytest.mark.parametrize("E, K, N_inter, top_k, group_size", MODEL_CONFIGS)
@pytest.mark.parametrize("M", NUM_TOKENS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("block_size_m", [1, 4])
def test_fused_moe_w1_matches_dense(
    E, K, N_inter, top_k, group_size, M, dtype, block_size_m
):
    """w1 GEMM via fused kernel matches per-expert dense kernel."""
    N_gate_up = N_inter * 2
    groups = K // group_size

    torch.manual_seed(42)
    x = torch.randn(M, K, dtype=dtype, device=device)
    w13 = _make_packed_weights(E, K, N_gate_up)
    w13_s = _make_scales(E, groups, N_gate_up, dtype)
    w13_z = _make_qzeros(E, groups, N_gate_up)
    g_idx = torch.empty(0, dtype=torch.int32, device=device)

    topk_ids = torch.randint(0, E, (M, top_k), device=device, dtype=torch.int32)
    si, ei, ntp = moe_align_block_size(topk_ids, block_size_m, E)

    # Fused kernel
    fused_out = torch.zeros(M * top_k, N_gate_up, dtype=dtype, device=device)
    ops.moe_gptq_gemm_rdna3(
        x,
        fused_out,
        w13,
        w13_s,
        w13_z,
        torch.empty(0, device=device),
        si,
        ei,
        ntp,
        top_k,
        block_size_m,
        False,
        0,
    )

    # Per-expert dense reference
    ref_out = torch.zeros(M * top_k, N_gate_up, dtype=dtype, device=device)
    for m in range(M):
        for k in range(top_k):
            e = topk_ids[m, k].item()
            flat = m * top_k + k
            ref = ops.gptq_gemm_rdna3(
                x[m : m + 1],
                w13[e],
                w13_z[e],
                w13_s[e],
                g_idx,
                False,
            )
            ref_out[flat] = ref.squeeze()

    # Split-K atomics can cause minor fp16/bf16 rounding differences
    # at large K (e.g. K=2048 → 8 K-blocks). Use allclose, not equal.
    atol = 0.5 if dtype == torch.bfloat16 else 0.1
    assert torch.allclose(fused_out, ref_out, atol=atol, rtol=0.01), (
        f"max diff: {(fused_out - ref_out).abs().max().item()}"
    )


@gfx1100_only
@pytest.mark.parametrize("E, K, N_inter, top_k, group_size", MODEL_CONFIGS)
@pytest.mark.parametrize("M", NUM_TOKENS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_moe_output_topk_reduces(E, K, N_inter, top_k, group_size, M, dtype):
    """output_topk fuses moe_sum: multiple experts write to same output row."""
    groups = K // group_size

    torch.manual_seed(123)
    x = torch.randn(M * top_k, K, dtype=dtype, device=device)
    w = _make_packed_weights(E, K, N_inter)
    ws = _make_scales(E, groups, N_inter, dtype)
    wz = _make_qzeros(E, groups, N_inter)

    topk_ids = torch.randint(0, E, (M, top_k), device=device, dtype=torch.int32)
    topk_w = torch.softmax(
        torch.randn(M, top_k, device=device),
        dim=-1,
    ).float()

    si, ei, ntp = moe_align_block_size(topk_ids, 1, E)

    # Without output_topk: write to [M*top_k, N] then moe_sum
    flat_out = torch.zeros(M * top_k, N_inter, dtype=dtype, device=device)
    ops.moe_gptq_gemm_rdna3(
        x,
        flat_out,
        w,
        ws,
        wz,
        topk_w.view(-1),
        si,
        ei,
        ntp,
        1,
        1,
        True,
        0,
    )
    ref = torch.zeros(M, N_inter, dtype=dtype, device=device)
    ops.moe_sum(flat_out.view(M, top_k, N_inter), ref)

    # With output_topk: write directly to [M, N]
    fused = torch.zeros(M, N_inter, dtype=dtype, device=device)
    ops.moe_gptq_gemm_rdna3(
        x,
        fused,
        w,
        ws,
        wz,
        topk_w.view(-1),
        si,
        ei,
        ntp,
        1,
        1,
        True,
        top_k,
    )

    atol = 1.0 if dtype == torch.bfloat16 else 0.1
    assert torch.allclose(fused, ref, atol=atol, rtol=0.01), (
        f"max diff: {(fused - ref).abs().max().item()}"
    )


@gfx1100_only
@pytest.mark.parametrize("E, K, N_inter, top_k, group_size", MODEL_CONFIGS)
@pytest.mark.parametrize("M", NUM_TOKENS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_full_moe_e2e(E, K, N_inter, top_k, group_size, M, dtype):
    """Full MoE forward: w1 + silu_and_mul + w2 with output_topk reduce."""
    N_gate_up = N_inter * 2
    hidden = K

    torch.manual_seed(7)
    x = torch.randn(M, K, dtype=dtype, device=device)
    w13 = _make_packed_weights(E, K, N_gate_up)
    w13_s = _make_scales(E, K // group_size, N_gate_up, dtype)
    w13_z = _make_qzeros(E, K // group_size, N_gate_up)
    w2 = _make_packed_weights(E, N_inter, hidden)
    w2_s = _make_scales(E, N_inter // group_size, hidden, dtype)
    w2_z = _make_qzeros(E, N_inter // group_size, hidden)
    g_idx = torch.empty(0, dtype=torch.int32, device=device)

    topk_ids = torch.randint(0, E, (M, top_k), device=device, dtype=torch.int32)
    topk_w = torch.softmax(
        torch.randn(M, top_k, device=device),
        dim=-1,
    ).float()

    si, ei, ntp = moe_align_block_size(topk_ids, 1, E)

    # Fused path (what apply() does)
    w1_out = torch.zeros(M * top_k, N_gate_up, dtype=dtype, device=device)
    ops.moe_gptq_gemm_rdna3(
        x,
        w1_out,
        w13,
        w13_s,
        w13_z,
        torch.empty(0, device=device),
        si,
        ei,
        ntp,
        top_k,
        1,
        False,
        0,
    )
    act_out = torch.empty(M * top_k, N_inter, dtype=dtype, device=device)
    apply_moe_activation(MoEActivation.SILU, act_out, w1_out)
    fused = torch.zeros(M, hidden, dtype=dtype, device=device)
    ops.moe_gptq_gemm_rdna3(
        act_out,
        fused,
        w2,
        w2_s,
        w2_z,
        topk_w.view(-1),
        si,
        ei,
        ntp,
        1,
        1,
        True,
        top_k,
    )

    # Per-expert reference
    ref = torch.zeros(M, hidden, dtype=dtype, device=device)
    for m_idx in range(M):
        for k_idx in range(top_k):
            e = topk_ids[m_idx, k_idx].item()
            w = topk_w[m_idx, k_idx].item()
            r1 = ops.gptq_gemm_rdna3(
                x[m_idx : m_idx + 1],
                w13[e],
                w13_z[e],
                w13_s[e],
                g_idx,
                False,
            )
            a = torch.empty(1, N_inter, dtype=dtype, device=device)
            apply_moe_activation(MoEActivation.SILU, a, r1)
            r2 = ops.gptq_gemm_rdna3(
                a,
                w2[e],
                w2_z[e],
                w2_s[e],
                g_idx,
                False,
            )
            ref[m_idx] += r2.squeeze() * w

    # E2E chains w1 + activation + w2 + topk_w + output_topk reduce.
    # Each step accumulates rounding error (split-K atomics, topk_w
    # multiply order). Use relative L2 norm like the dense kernel test.
    diff_l2 = torch.norm(fused.float() - ref.float())
    ref_l2 = torch.norm(ref.float())
    rel_l2 = (diff_l2 / ref_l2).item() if ref_l2 > 0 else 0.0
    threshold = 0.05 if dtype == torch.float16 else 0.10
    assert rel_l2 < threshold, (
        f"rel L2 = {rel_l2:.4f} (threshold {threshold}), "
        f"max abs diff: {(fused - ref).abs().max().item()}"
    )


@gfx1100_only
def test_expert_id_minus_one():
    """Kernel handles expert_id == -1 (expert parallelism) without crash."""
    # Qwen3-30B-A3B dims (E capped for memory)
    E, K, N = 16, 2048, 768
    groups = K // 32

    w = _make_packed_weights(E, K, N)
    ws = _make_scales(E, groups, N, torch.bfloat16)
    wz = _make_qzeros(E, groups, N)
    x = torch.randn(1, K, dtype=torch.bfloat16, device=device)

    # Manually create sorted_token_ids/expert_ids with -1
    sorted_ids = torch.tensor([0], dtype=torch.int32, device=device)
    expert_ids = torch.tensor([-1], dtype=torch.int32, device=device)
    ntp = torch.tensor([1], dtype=torch.int32, device=device)

    out = torch.zeros(1, N, dtype=torch.bfloat16, device=device)
    ops.moe_gptq_gemm_rdna3(
        x,
        out,
        w,
        ws,
        wz,
        torch.empty(0, device=device),
        sorted_ids,
        expert_ids,
        ntp,
        1,
        1,
        False,
        0,
    )
    current_platform.synchronize()

    # Output should remain zero (expert skipped)
    assert torch.equal(out, torch.zeros_like(out))
