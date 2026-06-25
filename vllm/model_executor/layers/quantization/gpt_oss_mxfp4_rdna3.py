# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPT-OSS MXFP4 MoE on RDNA3 (gfx1100) via the native HIP kernel.

GPT-OSS is native-mxfp4 (GptOssMxfp4MoEMethod), not compressed-tensors, and
differs from the plain MXFP4 MoE in two ways handled here:
  - per-expert biases (w13_bias, w2_bias)
  - the SwiGLU-OAI clamped activation (alpha=1.702, beta=1.0, limit=7.0)

w13 ships gate/up *interleaved*; we de-interleave at load so the gate_up
output is contiguous [gate || up]. Biases and the moe_sum reduction run in
Python (output_topk=0), so the moe_mxfp4_gemm_rdna3 kernel is reused as-is.
"""

import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import RoutedExperts, SharedExperts
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.rdna3_moe_common import (  # noqa: E501
    repack_experts,
    select_block_size_m,
)
from vllm.model_executor.layers.quantization.mxfp4 import GptOssMxfp4MoEMethod

logger = init_logger(__name__)

_ALPHA, _BETA, _LIMIT = 1.702, 1.0, 7.0


def maybe_rdna3_gpt_oss_method(layer):
    """Return the native RDNA3 GPT-OSS MXFP4 MoE method on gfx1100, else None.

    Keeps the gfx1100 selection out of the generic ``Mxfp4Config`` so the only
    footprint there is a one-line delegation (mirrors ``rocm_moe_rdna`` for the
    compressed-tensors path).
    """
    from vllm.platforms import current_platform

    if not (
        current_platform.is_rocm()
        and hasattr(torch.ops, "_rocm_C")
        and hasattr(torch.ops._rocm_C, "moe_mxfp4_gemm_rdna3")
    ):
        return None
    from vllm.platforms.rocm import on_gfx1100

    if not on_gfx1100():
        return None
    logger.info_once("Using GptOssMxfp4RDNA3MoEMethod (native RDNA3 HIP kernel)")
    return GptOssMxfp4RDNA3MoEMethod(layer.moe_config)


class GptOssMxfp4RDNA3MoEMethod(GptOssMxfp4MoEMethod):
    """GPT-OSS MXFP4 MoE via the native RDNA3 HIP kernel (moe_mxfp4_gemm_rdna3).

    Bypasses the modular MoE kernel: the runner calls apply() directly through
    forward_modular. We report supports_internal_mk (skip MK construction),
    not monolithic (route through forward_modular -> apply), and a null
    fused-moe quant config (apply does the dequant itself).
    """

    @property
    def supports_internal_mk(self) -> bool:
        return True

    @property
    def is_monolithic(self) -> bool:
        return False

    def get_fused_moe_quant_config(self, layer):
        return None

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        device = layer.w13_weight.device
        # w13: de-interleave gate/up + repack, then free the source.
        b_q13, b_s13 = repack_experts(
            layer.w13_weight.data, layer.w13_weight_scale.data, deinterleave=True
        )
        b13 = torch.cat(
            [layer.w13_bias.data[:, ::2], layer.w13_bias.data[:, 1::2]], dim=1
        ).contiguous()
        del layer.w13_weight, layer.w13_weight_scale
        layer.w13_weight_packed = torch.nn.Parameter(b_q13, requires_grad=False)
        layer.w13_weight_scale = torch.nn.Parameter(b_s13, requires_grad=False)
        layer.w13_bias = torch.nn.Parameter(b13, requires_grad=False)

        # w2: plain repack (no de-interleave), then free the source.
        b_q2, b_s2 = repack_experts(
            layer.w2_weight.data, layer.w2_weight_scale.data, deinterleave=False
        )
        del layer.w2_weight, layer.w2_weight_scale
        layer.w2_weight_packed = torch.nn.Parameter(b_q2, requires_grad=False)
        layer.w2_weight_scale = torch.nn.Parameter(b_s2, requires_grad=False)
        layer.rdna3_empty_tw = torch.empty(0, device=device)

    @staticmethod
    def _swiglu_oai(x: torch.Tensor) -> torch.Tensor:
        # gate||up contiguous (w13 de-interleaved at load). SwiGLU-OAI:
        #   out = clamp(gate) * sigmoid(alpha*gate) * (clamp(up) + beta)
        d = x.shape[-1] // 2
        gate = torch.clamp(x[..., :d], max=_LIMIT)
        up = torch.clamp(x[..., d:], min=-_LIMIT, max=_LIMIT)
        return gate * torch.sigmoid(_ALPHA * gate) * (up + _BETA)

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        M = x.shape[0]
        top_k = topk_ids.shape[1]
        total = M * top_k
        N_gate_up = layer.w13_weight_packed.shape[2]
        hidden = layer.w2_weight_packed.shape[2]
        dtype = x.dtype
        device = x.device

        gne = layer.global_num_experts
        if gne <= 0:
            gne = layer.w13_weight_packed.shape[0]
        block_size_m = select_block_size_m(M, top_k, gne)
        sti, eid, ntp = moe_align_block_size(
            topk_ids, block_size_m, gne, layer.expert_map
        )
        flat_experts = topk_ids.reshape(-1).long()
        empty_tw = layer.rdna3_empty_tw

        # gate_up GEMM (no reduction) + per-expert bias
        w1 = torch.zeros(total, N_gate_up, dtype=dtype, device=device)
        ops.moe_mxfp4_gemm_rdna3(
            x, w1, layer.w13_weight_packed, layer.w13_weight_scale, empty_tw,
            sti, eid, ntp, top_k, block_size_m, False, 0,
        )
        w1 = w1 + layer.w13_bias[flat_experts].to(dtype)
        act = self._swiglu_oai(w1)

        # down GEMM (no reduction) + per-expert bias, then weighted moe_sum
        w2o = torch.zeros(total, hidden, dtype=dtype, device=device)
        ops.moe_mxfp4_gemm_rdna3(
            act, w2o, layer.w2_weight_packed, layer.w2_weight_scale, empty_tw,
            sti, eid, ntp, 1, block_size_m, False, 0,
        )
        w2o = w2o + layer.w2_bias[flat_experts].to(dtype)
        w2o = w2o * topk_weights.reshape(-1, 1).to(dtype)
        return w2o.view(M, top_k, hidden).sum(dim=1)
