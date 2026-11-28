# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct routed tensor-core expert path with fused W1/SiLU epilogue."""

import torch

from vllm import _custom_ops as ops
from vllm.triton_utils import tl, triton


@triton.jit
def fused_gate_up(
    X,
    W,
    Ids,
    A,
    H: tl.constexpr,
    I: tl.constexpr,  # noqa: E741
    TOPK: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    slot = tl.program_id(1)
    ns = tl.program_id(0) * BN + tl.arange(0, BN)
    ms = tl.arange(0, 16)
    ks = tl.arange(0, BK)
    expert = tl.load(Ids + slot).to(tl.int64)
    valid = (expert >= 0) & (expert < 32)
    expert = tl.maximum(0, tl.minimum(expert, 31))
    g = tl.zeros((16, BN), tl.float32)
    u = tl.zeros((16, BN), tl.float32)
    for tile in range(tl.cdiv(H, BK)):
        kth = ks + tile * BK
        x = tl.load(
            X + (slot // TOPK) * H + kth[None, :],
            (ms[:, None] == 0) & (kth[None, :] < H),
            other=0,
        )
        base = W + expert * 2 * I * H + ns[None, :] * H + kth[:, None]
        mask = valid & (ns[None, :] < I) & (kth[:, None] < H)
        wg = tl.load(base, mask, other=0)
        wu = tl.load(base + I * H, mask, other=0)
        g += tl.dot(x, wg)
        u += tl.dot(x, wu)
    gf = g.to(tl.bfloat16).to(tl.float32)
    uf = u.to(tl.bfloat16).to(tl.float32)
    silu = (gf / (1.0 + tl.exp(-gf))).to(tl.bfloat16).to(tl.float32)
    value = tl.where(valid, silu * uf, 0.0)
    tl.store(
        A + slot * I + ms[:, None] * I + ns[None, :],
        value,
        (ms[:, None] == 0) & (ns[None, :] < I),
    )


@triton.jit
def routed_down(
    A,
    W,
    Ids,
    Weights,
    O,  # noqa: E741
    H: tl.constexpr,
    I: tl.constexpr,  # noqa: E741
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    slot = tl.program_id(1)
    ns = tl.program_id(0) * BN + tl.arange(0, BN)
    ms = tl.arange(0, 16)
    ks = tl.arange(0, BK)
    expert = tl.load(Ids + slot).to(tl.int64)
    valid = (expert >= 0) & (expert < 32)
    expert = tl.maximum(0, tl.minimum(expert, 31))
    acc = tl.zeros((16, BN), tl.float32)
    for tile in range(tl.cdiv(I, BK)):
        kth = ks + tile * BK
        a = tl.load(
            A + slot * I + kth[None, :],
            (ms[:, None] == 0) & (kth[None, :] < I),
            other=0,
        )
        w = tl.load(
            W + expert * H * I + ns[None, :] * I + kth[:, None],
            valid & (ns[None, :] < H) & (kth[:, None] < I),
            other=0,
        )
        acc += tl.dot(a, w)
    weight = tl.load(Weights + slot)
    value = tl.where(valid, acc * weight, 0.0)
    tl.store(
        O + slot * H + ms[:, None] * H + ns[None, :],
        value,
        (ms[:, None] == 0) & (ns[None, :] < H),
    )


def small_bf16_silu_experts(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
    rows: int = 32,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Routed BF16 SiLU experts for tiny decode batches without block alignment.

    Each program handles one (token, top-k slot) pair. Only row 0 of the
    16-row tiles is live; the padding satisfies the tl.dot minimum M.

    Args:
        x: [M, H] BF16 hidden states.
        w1: [E, 2 * I, H] gate/up weights.
        w2: [E, H, I] down weights.
        weights: [M, topk] router weights.
        ids: [M, topk] expert ids; ids outside [0, E) contribute zero.
        rows: N-tile size of the gate/up kernel.
        output: optional [M, H] output buffer.

    Returns:
        The [M, H] output tensor.

    """
    assert x.dtype == w1.dtype == w2.dtype == torch.bfloat16
    assert all(t.is_contiguous() for t in (x, w1, w2, weights, ids))
    batch, hidden = x.shape
    experts, twice_intermediate, _ = w1.shape
    assert experts == 32
    intermediate = twice_intermediate // 2
    assert w2.shape == (experts, hidden, intermediate)
    topk = ids.shape[1]
    activated = torch.empty(
        (batch * topk, intermediate), device=x.device, dtype=x.dtype
    )
    weighted = torch.empty((batch, topk, hidden), device=x.device, dtype=x.dtype)
    if output is None:
        output = torch.empty_like(x)
    fused_gate_up[(triton.cdiv(intermediate, rows), batch * topk)](
        x,
        w1,
        ids,
        activated,
        hidden,
        intermediate,
        topk,
        rows,
        128,
        num_warps=4,
        num_stages=4,
    )
    routed_down[(triton.cdiv(hidden, 64), batch * topk)](
        activated,
        w2,
        ids,
        weights,
        weighted,
        hidden,
        intermediate,
        64,
        128,
        num_warps=4,
        num_stages=4,
    )
    ops.moe_sum(weighted, output)
    return output
