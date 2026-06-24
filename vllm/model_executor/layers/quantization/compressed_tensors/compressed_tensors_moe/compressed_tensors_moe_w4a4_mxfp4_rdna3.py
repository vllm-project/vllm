# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CompressedTensors MXFP4 MoE using the fused RDNA3 (gfx1100) HIP kernel.

Uses ``moe_mxfp4_gemm_rdna3`` — one HIP launch per GEMM that fuses expert
routing + E2M1/E8M0 dequant + dot product with atomic output. No zero-point;
E8M0 block scale is ``[E, K/32, N]`` uint8. The two-GEMM forward is shared
with the W4A16 path via RDNA3FusedMoEMixin.
"""

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_mxfp4 import (  # noqa: E501
    CompressedTensorsW4A4Mxfp4MoEMethod,
)

from .rdna3_moe_common import RDNA3FusedMoEMixin, repack_experts


class CompressedTensorsW4A4Mxfp4RDNA3MoEMethod(
    RDNA3FusedMoEMixin, CompressedTensorsW4A4Mxfp4MoEMethod
):
    """MXFP4 MoE via the native RDNA3 HIP kernel (moe_mxfp4_gemm_rdna3)."""

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        device = layer.w13_weight_packed.device
        b_q13, b_s13 = repack_experts(
            layer.w13_weight_packed.data, layer.w13_weight_scale.data, False
        )
        del layer.w13_weight_packed, layer.w13_weight_scale
        layer.w13_weight_packed = torch.nn.Parameter(b_q13, requires_grad=False)
        layer.w13_weight_scale = torch.nn.Parameter(b_s13, requires_grad=False)

        b_q2, b_s2 = repack_experts(
            layer.w2_weight_packed.data, layer.w2_weight_scale.data, False
        )
        del layer.w2_weight_packed, layer.w2_weight_scale
        layer.w2_weight_packed = torch.nn.Parameter(b_q2, requires_grad=False)
        layer.w2_weight_scale = torch.nn.Parameter(b_s2, requires_grad=False)
        layer.rdna3_empty_tw = torch.empty(0, device=device)

    def _gemm_w13(self, layer, a, c, tw, sti, eid, ntp, top_k, block_size_m, mul_tw):
        ops.moe_mxfp4_gemm_rdna3(
            a, c, layer.w13_weight_packed, layer.w13_weight_scale, tw, sti, eid,
            ntp, top_k, block_size_m, mul_tw, 0,
        )

    def _gemm_w2(self, layer, a, c, tw, sti, eid, ntp, block_size_m, mul_tw, output_topk):
        ops.moe_mxfp4_gemm_rdna3(
            a, c, layer.w2_weight_packed, layer.w2_weight_scale, tw, sti, eid,
            ntp, 1, block_size_m, mul_tw, output_topk,
        )
