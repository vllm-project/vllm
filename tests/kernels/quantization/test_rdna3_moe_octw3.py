# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness of the RDNA3 Octave-W3 fused MoE kernel (gfx1100).

Both GEMMs of an expert layer (gate_up with a Hadamard-128 input rotation;
down with SiLU-and-mul on load, Hadamard-32 and the top-k reduction) are
compared against a dense fp32 reference built from the planar checkpoint
format, so the repack, the code lookup and the rotation are all covered.

Run `pytest tests/kernels/quantization/test_rdna3_moe_octw3.py`.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("RDNA3 Octave-W3 MoE kernel is ROCm-only", allow_module_level=True)

from vllm import _custom_ops as ops  # noqa: E402
from vllm.model_executor.layers.fused_moe.experts.rdna3_octw3_moe import (  # noqa: E402
    OCTW3_LEVELS,
    octw3_repack,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (  # noqa: E402
    moe_align_block_size,
)
from vllm.platforms.rocm import on_gfx1100  # noqa: E402

device = "cuda"

pytestmark = pytest.mark.skipif(
    not (on_gfx1100() and hasattr(torch.ops._rocm_C, "moe_octw3_gemm_rdna3")),
    reason="Requires gfx1100 with moe_octw3_gemm_rdna3",
)

# Qwen3.8-Flash-Next at TP4: hidden 2560, expert intermediate 640 / 4, top-10.
E, HIDDEN, INTER, TOP_K = 16, 2560, 160, 10


def _hadamard(n: int) -> torch.Tensor:
    h = torch.ones(1, 1, device=device)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)


def _planar(codes: torch.Tensor) -> torch.Tensor:
    g = codes.to(torch.int64).unflatten(-1, (-1, 32))
    weights = 1 << torch.arange(32, device=device, dtype=torch.int64)
    p = torch.stack([(((g >> b) & 1) * weights).sum(-1) for b in range(3)], -1)
    return torch.where(p >= 2**31, p - 2**32, p).to(torch.int32).flatten(-2)


def _make_expert_weights(n: int, k: int, had: int):
    """Planar weights, their kernel layout and the dense fp32 reference."""
    codes = torch.randint(0, 8, (E, n, k), device=device)
    scales = (torch.rand(E, n, k // 32, device=device) * 0.02 + 0.005).half()
    levels = torch.tensor(OCTW3_LEVELS, device=device)
    dense = levels[codes] * scales.float().repeat_interleave(32, -1)
    dense = (dense.unflatten(-1, (-1, had)) @ _hadamard(had)).flatten(-2)
    return octw3_repack(_planar(codes)), scales.transpose(1, 2).contiguous(), dense


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b).norm() / b.norm()).item()


@pytest.mark.parametrize(
    "num_tokens,block_m", [(1, 1), (5, 1), (24, 1), (7, 2), (64, 4), (300, 4), (300, 8)]
)
def test_moe_octw3_matches_dense_reference(num_tokens: int, block_m: int):
    torch.manual_seed(0)
    w1, s1, dense1 = _make_expert_weights(2 * INTER, HIDDEN, 128)
    w2, s2, dense2 = _make_expert_weights(HIDDEN, INTER, 32)

    x = torch.randn(num_tokens, HIDDEN, device=device).half()
    topk_w, topk_ids = (
        torch.randn(num_tokens, E, device=device).softmax(-1).topk(TOP_K, -1)
    )
    topk_w = (topk_w / topk_w.sum(-1, keepdim=True)).reshape(-1).float()
    rows = num_tokens * TOP_K
    if block_m == 1:
        sorted_ids = torch.arange(rows, dtype=torch.int32, device=device)
        expert_ids = topk_ids.reshape(-1).to(torch.int32)
        num_post_pad = torch.tensor([rows], dtype=torch.int32, device=device)
    else:
        sorted_ids, expert_ids, num_post_pad = moe_align_block_size(
            topk_ids, block_m, E, None
        )
    no_weights = torch.empty(0, device=device)

    gate_up = torch.zeros(rows, 2 * INTER, device=device, dtype=torch.half)
    ops.moe_octw3_gemm_rdna3(
        x,
        gate_up,
        w1,
        s1,
        no_weights,
        sorted_ids,
        expert_ids,
        num_post_pad,
        TOP_K,
        block_m,
        False,
        0,
        128,
    )
    out = torch.zeros(num_tokens, HIDDEN, device=device, dtype=torch.half)
    ops.moe_octw3_gemm_rdna3(
        gate_up,
        out,
        w2,
        s2,
        topk_w,
        sorted_ids,
        expert_ids,
        num_post_pad,
        1,
        block_m,
        True,
        TOP_K,
        32,
    )

    flat_ids = topk_ids.reshape(-1)
    x_rows = x.float().repeat_interleave(TOP_K, 0)
    act = (
        (F.silu(gate_up[:, :INTER].float()).half().float() * gate_up[:, INTER:].float())
        .half()
        .float()
    )
    ref1 = torch.empty(rows, 2 * INTER, device=device)
    ref2 = torch.empty(rows, HIDDEN, device=device)
    for e in flat_ids.unique().tolist():
        r = (flat_ids == e).nonzero().flatten()
        ref1[r] = x_rows[r] @ dense1[e].t()
        ref2[r] = act[r] @ dense2[e].t()
    ref2 = (ref2 * topk_w[:, None]).view(num_tokens, TOP_K, HIDDEN).sum(1)

    assert _rel(gate_up, ref1) < 2e-3
    assert _rel(out, ref2) < 2e-3
