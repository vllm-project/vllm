# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP32 A16W4 MoE reference for the MiniMax-M3 specialized decode tests.

AITER's `fused_moe` can't be the reference: its tuned MiniMax-M3 configs
quantize the activations to MXFP4 at decode batch sizes (A4W4).

Nothing here syncs with the host, since `test_model.py` runs it inside the
engine's forward under `VLLM_GPU_SYNC_CHECK=error`."""

import torch

from vllm.models.minimax_m3.amd.specialized_decode.mono import ffn


def dequant_mxfp4(codes, scales):
    """MXFP4 weights to FP32: `codes` packs two 4-bit e2m1 values (sign, 2
    exponent bits, 1 mantissa bit) a byte, the low 4 bits first
    (`[rows, K/2]`), and each e8m0 scale in `scales` (`[rows, K/32]`) is a power
    of two shared by 32 consecutive weights. Returns `[rows, K]`."""
    fp4 = torch.stack((codes & 0xF, codes >> 4), -1).flatten(-2)
    exp, man = (fp4 >> 1) & 3, (fp4 & 1).float()
    mag = torch.where(exp == 0, man / 2, torch.exp2(exp - 1.0) * (1 + man / 2))
    values = torch.where(fp4 >= 8, -mag, mag)
    return values * torch.exp2(scales.float() - 127).repeat_interleave(32, -1)


def reference_experts(x, raw, ids, wts):
    """Each token's output: the sum of its experts `ids` weighted by `wts`
    (both `[T, k]`). `raw` holds the weights before vLLM shuffles them for
    AITER: `w13` (gate rows, then up rows), `w2`, and their scales `s13`, `s2`.
    Math is FP32, except the SwiGLU output is rounded to BF16 as the kernel
    does. Every expert runs on every token, weighted 0 where not routed, so the
    routing is never read on the host."""
    inter = raw.w13.shape[1] // 2
    x = x.float()
    out = torch.zeros(x.shape[0], raw.w2.shape[1], dtype=torch.float32, device=x.device)
    for e in range(raw.w13.shape[0]):
        weight = (wts.float() * (ids == e)).sum(-1, keepdim=True)
        gate_up = x @ dequant_mxfp4(raw.w13[e], raw.s13[e]).T
        gate = gate_up[:, :inter].clamp(max=ffn.LIMIT)
        up = gate_up[:, inter:].clamp(-ffn.LIMIT, ffn.LIMIT)
        mid = (gate * torch.sigmoid(ffn.ALPHA * gate) * (up + ffn.BETA)).bfloat16()
        out += weight * (mid.float() @ dequant_mxfp4(raw.w2[e], raw.s2[e]).T)
    return out
