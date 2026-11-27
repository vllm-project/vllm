# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP32 A16W4 reference for the fused launch's experts. AITER's fused_moe is
not one: its tuned MiniMax-M3 configs quantize the activations to MXFP4 at
small M (A4W4)."""

import torch

from vllm.models.minimax_m3.amd.specialized_decode.mono import ffn


def dequant_mxfp4(codes, scales):
    from aiter.utility.fp4_utils import e8m0_to_f32, mxfp4_to_f32

    return mxfp4_to_f32(codes) * e8m0_to_f32(scales).repeat_interleave(32, -1)


def reference_experts(x, raw, ids, wts):
    """The experts `ids` weighted by `wts` on the dequantized checkpoint-format
    weights `raw` (`w13` gate rows then up rows, `w2`, and their e8m0 scales
    `s13`, `s2`), in FP32 with a BF16 intermediate."""
    inter = raw.w13.shape[1] // 2
    out = torch.zeros(x.shape[0], raw.w2.shape[1], dtype=torch.float32, device=x.device)
    for e in ids.unique().tolist():
        rows, slot = (ids == e).nonzero(as_tuple=True)
        h = x[rows].float() @ dequant_mxfp4(raw.w13[e], raw.s13[e]).T
        gate = h[:, :inter].clamp(max=ffn.LIMIT)
        up = h[:, inter:].clamp(-ffn.LIMIT, ffn.LIMIT)
        mid = (gate * torch.sigmoid(ffn.ALPHA * gate) * (up + ffn.BETA)).bfloat16()
        y = mid.float() @ dequant_mxfp4(raw.w2[e], raw.s2[e]).T
        out.index_add_(0, rows, wts[rows, slot, None].float() * y)
    return out
