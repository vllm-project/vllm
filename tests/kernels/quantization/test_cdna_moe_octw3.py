# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The CDNA3 (gfx942) Octave-W3 MoE prefill path against a dense fp32
reference built from the planar checkpoint format: repack, code lookup,
the rotation kernel and both MFMA GEMMs."""

import math

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fused_moe.experts.octw3_mfma_moe import (
    octw3_cdna_ops,
    octw3_dec_pack,
    on_gfx942,
)
from vllm.model_executor.layers.fused_moe.experts.rdna3_octw3_moe import (
    OCTW3_LEVELS,
    octw3_repack,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)

pytestmark = pytest.mark.skipif(not on_gfx942(), reason="Requires gfx942")

device = "cuda"
# Qwen3.8-Flash-Next at TP1: hidden 2560, expert intermediate 640, top-10.
E, HIDDEN, INTER, TOP_K = 16, 2560, 640, 10


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
    codes = torch.randint(0, 8, (E, n, k), device=device)
    scales = (torch.rand(E, n, k // 32, device=device) * 0.02 + 0.005).half()
    levels = torch.tensor(OCTW3_LEVELS, device=device)
    dense = levels[codes] * scales.float().repeat_interleave(32, -1)
    dense = (dense.unflatten(-1, (-1, had)) @ _hadamard(had)).flatten(-2)
    return octw3_repack(_planar(codes)), scales.transpose(1, 2).contiguous(), dense


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b).norm() / b.norm()).item()


@pytest.mark.parametrize("cfg", range(12))
@pytest.mark.parametrize("num_tokens", [40, 1500])
def test_moe_octw3_mfma_matches_dense_reference(num_tokens: int, cfg: int):
    torch.manual_seed(0)
    ops = octw3_cdna_ops()
    block_m = ops.moe_octw3_mfma_cfg_block_m(cfg)
    w1, s1, dense1 = _make_expert_weights(2 * INTER, HIDDEN, 128)
    w2, s2, dense2 = _make_expert_weights(HIDDEN, INTER, 32)

    x = torch.randn(num_tokens, HIDDEN, device=device).half()
    topk_w, topk_ids = (
        torch.randn(num_tokens, E, device=device).softmax(-1).topk(TOP_K, -1)
    )
    topk_w = (topk_w / topk_w.sum(-1, keepdim=True)).reshape(-1).float()
    rows = num_tokens * TOP_K
    sorted_ids, expert_ids, num_post_pad = moe_align_block_size(
        topk_ids, block_m, E, None
    )
    no_weights = torch.empty(0, device=device)

    x_rot = torch.empty_like(x)
    ops.octw3_rotate_cdna(x, x_rot, False, 128)
    gate_up = torch.empty(rows, 2 * INTER, device=device, dtype=torch.half)
    ops.moe_octw3_mfma_gemm_cdna_cfg(
        x_rot,
        gate_up,
        w1,
        s1,
        no_weights,
        sorted_ids,
        expert_ids,
        num_post_pad,
        TOP_K,
        0,
        cfg,
    )
    act = torch.empty(rows, INTER, device=device, dtype=torch.half)
    ops.octw3_rotate_cdna(gate_up, act, True, 32)
    out = torch.zeros(num_tokens, HIDDEN, device=device, dtype=torch.half)
    ops.moe_octw3_mfma_gemm_cdna_cfg(
        act,
        out,
        w2,
        s2,
        topk_w,
        sorted_ids,
        expert_ids,
        num_post_pad,
        1,
        TOP_K,
        cfg,
    )

    flat_ids = topk_ids.reshape(-1)
    x_rows = x.float().repeat_interleave(TOP_K, 0)
    act_ref = (
        (F.silu(gate_up[:, :INTER].float()).half().float() * gate_up[:, INTER:].float())
        .half()
        .float()
    )
    ref1 = torch.empty(rows, 2 * INTER, device=device)
    ref2 = torch.empty(rows, HIDDEN, device=device)
    for e in flat_ids.unique().tolist():
        r = (flat_ids == e).nonzero().flatten()
        ref1[r] = x_rows[r] @ dense1[e].t()
        ref2[r] = act_ref[r] @ dense2[e].t()
    ref2 = (ref2 * topk_w[:, None]).view(num_tokens, TOP_K, HIDDEN).sum(1)

    assert _rel(gate_up, ref1) < 2e-3
    assert _rel(out, ref2) < 2e-3


@pytest.mark.parametrize("block_m", [16, 32, 64, 128])
@pytest.mark.parametrize("num_tokens", [1, 40, 300])
def test_moe_octw3_packed_layout_matches_dense_reference(num_tokens: int, block_m: int):
    """The model path: decode-layout weights with the scales inside."""
    torch.manual_seed(0)
    ops = octw3_cdna_ops()
    w1, s1, dense1 = _make_expert_weights(2 * INTER, HIDDEN, 128)
    w2, s2, dense2 = _make_expert_weights(HIDDEN, INTER, 32)
    E_, G1, _, N1 = w1.shape
    E_, G2, _, N2 = w2.shape
    p1 = octw3_dec_pack(w1, s1).view(E_, G1, 4, N1)
    p2 = octw3_dec_pack(w2, s2).view(E_, G2, 4, N2)

    x = torch.randn(num_tokens, HIDDEN, device=device).half()
    topk_w, topk_ids = (
        torch.randn(num_tokens, E, device=device).softmax(-1).topk(TOP_K, -1)
    )
    topk_w = (topk_w / topk_w.sum(-1, keepdim=True)).reshape(-1).float()
    rows = num_tokens * TOP_K
    sorted_ids, expert_ids, num_post_pad = moe_align_block_size(
        topk_ids, block_m, E, None
    )
    no_weights = torch.empty(0, device=device)
    x_rot = torch.empty_like(x)
    ops.octw3_rotate_cdna(x, x_rot, False, 128)
    gate_up = torch.empty(rows, 2 * INTER, device=device, dtype=torch.half)
    ops.moe_octw3_pk_gemm_cdna(
        x_rot,
        gate_up,
        p1,
        no_weights,
        sorted_ids,
        expert_ids,
        num_post_pad,
        TOP_K,
        0,
        block_m,
    )
    act = torch.empty(rows, INTER, device=device, dtype=torch.half)
    ops.octw3_rotate_cdna(gate_up, act, True, 32)
    out = torch.zeros(num_tokens, HIDDEN, device=device, dtype=torch.half)
    ops.moe_octw3_pk_gemm_cdna(
        act,
        out,
        p2,
        topk_w,
        sorted_ids,
        expert_ids,
        num_post_pad,
        1,
        TOP_K,
        block_m,
    )

    flat_ids = topk_ids.reshape(-1)
    x_rows = x.float().repeat_interleave(TOP_K, 0)
    act_ref = (
        (F.silu(gate_up[:, :INTER].float()).half().float() * gate_up[:, INTER:].float())
        .half()
        .float()
    )
    ref1 = torch.empty(rows, 2 * INTER, device=device)
    ref2 = torch.empty(rows, HIDDEN, device=device)
    for e in flat_ids.unique().tolist():
        r = (flat_ids == e).nonzero().flatten()
        ref1[r] = x_rows[r] @ dense1[e].t()
        ref2[r] = act_ref[r] @ dense2[e].t()
    ref2 = (ref2 * topk_w[:, None]).view(num_tokens, TOP_K, HIDDEN).sum(1)

    assert _rel(gate_up, ref1) < 2e-3
    assert _rel(out, ref2) < 2e-3
