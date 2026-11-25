# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP32 A16W4 MoE reference for the MiniMax-M3 specialized decode tests.

AITER's `fused_moe` can't be the reference: its tuned MiniMax-M3 configs
quantize the activations to MXFP4 at decode batch sizes (A4W4)."""

import torch

from vllm.models.minimax_m3.amd.specialized_decode.mono import ffn


def dequant_mxfp4(codes, scales):
    """MXFP4 weights to FP32: `codes` packs two 4-bit values a byte
    (`[rows, K/2]`), and each e8m0 scale in `scales` (`[rows, K/32]`) is a
    power of two shared by 32 consecutive weights. Returns `[rows, K]`."""
    from aiter.utility.fp4_utils import e8m0_to_f32, mxfp4_to_f32

    return mxfp4_to_f32(codes) * e8m0_to_f32(scales).repeat_interleave(32, -1)


def reference_experts(x, raw, ids, wts):
    """Each token's output: the sum of its experts `ids` weighted by `wts`
    (both `[T, k]`). `raw` holds the weights before vLLM shuffles them for
    AITER: `w13` (gate rows, then up rows), `w2`, and their scales `s13`, `s2`.
    Math is FP32, except the SwiGLU output is rounded to BF16 as the kernel
    does."""
    inter = raw.w13.shape[1] // 2
    out = torch.zeros(x.shape[0], raw.w2.shape[1], dtype=torch.float32, device=x.device)
    for e in ids.unique().tolist():
        rows, slot = (ids == e).nonzero(as_tuple=True)
        gate_up = x[rows].float() @ dequant_mxfp4(raw.w13[e], raw.s13[e]).T
        gate = gate_up[:, :inter].clamp(max=ffn.LIMIT)
        up = gate_up[:, inter:].clamp(-ffn.LIMIT, ffn.LIMIT)
        mid = (gate * torch.sigmoid(ffn.ALPHA * gate) * (up + ffn.BETA)).bfloat16()
        y = mid.float() @ dequant_mxfp4(raw.w2[e], raw.s2[e]).T
        out.index_add_(0, rows, wts[rows, slot, None].float() * y)
    return out
