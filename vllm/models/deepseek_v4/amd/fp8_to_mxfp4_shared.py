# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load-time FP8(block-scale) -> MXFP4 re-quant for DSv4 shared experts.

DeepSeek-V4-Flash routes experts in MXFP4 but stores the always-on shared
expert as FP8 (E4M3 + 128x128 block scale). To fold the shared expert into
the routed grouped-GEMM, weights are converted to the MXFP4 checkpoint layout
expected by routed expert slots, then loaded into slots
``[n_routed .. n_routed+n_shared)``.
"""

from __future__ import annotations

import torch


def dequant_fp8_block(
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    block: tuple[int, int] = (128, 128),
) -> torch.Tensor:
    """Dequantize a DeepSeek block-quantized FP8 weight to bf16."""
    assert weight.dim() == 2, f"expected 2D weight, got {tuple(weight.shape)}"
    out, inn = weight.shape
    bn, bk = block
    w = weight.to(torch.float32)
    scale = weight_scale_inv.to(torch.float32)
    scale = scale.repeat_interleave(bn, dim=0)[:out]
    scale = scale.repeat_interleave(bk, dim=1)[:, :inn]
    return (w * scale).to(torch.bfloat16)


def quant_bf16_to_mxfp4(w_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 [out, in] -> (packed_uint8 [out, in//2], scale_uint8 [out, in//32])."""
    from aiter.utility.fp4_utils import dynamic_mxfp4_quant

    assert w_bf16.dim() == 2
    orig_device = w_bf16.device
    w = w_bf16.contiguous()
    if w.device.type != "cuda":
        w = w.cuda()
    packed, scale = dynamic_mxfp4_quant(w, shuffle=False)
    packed = packed.view(torch.uint8).to(orig_device).contiguous()
    scale = scale.view(torch.uint8).to(orig_device).contiguous()
    return packed, scale


def convert_shared_fp8_to_mxfp4(
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    block: tuple[int, int] = (128, 128),
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP8 block-scale [out, in] -> MXFP4 (packed uint8, scale uint8)."""
    return quant_bf16_to_mxfp4(dequant_fp8_block(weight, weight_scale_inv, block))
