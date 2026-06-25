# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CompressedTensors MoE W4A16 using the fused RDNA3 (gfx1100) HIP kernel.

Uses ``moe_gptq_gemm_rdna3`` — a single HIP kernel launch per GEMM that
handles expert routing + W4A16 dequant + dot product with atomic output.

Weight format (per expert, same as dense RDNA3 W4A16):
  - Packed int32 ``[E, K/8, N]`` with exllama shuffle
  - Scales ``[E, groups, N]`` in activation dtype
  - Zero points ``[E, groups, N/8]`` packed int32 (synthesized)
"""

import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16 import (  # noqa: E501
    CompressedTensorsWNA16MoEMethod,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32,
)
from vllm.scalar_type import scalar_types

from .rdna3_moe_common import RDNA3FusedMoEMixin

logger = init_logger(__name__)


def _synthesize_qzeros(
    groups: int, out_features: int, device: torch.device
) -> torch.Tensor:
    """Create packed zero-point tensor for symmetric quant.

    GPTQv1 +1 quirk: kernel adds 1 to stored zeros, so encode
    (bias - 1) = 7 for uint4b8 (bias=8).
    """
    zeros = torch.full(
        (groups, out_features),
        scalar_types.uint4b8.bias - 1,
        dtype=torch.int32,
        device=device,
    )
    return pack_quantized_values_into_int32(zeros, scalar_types.uint4b8, packed_dim=1)


class CompressedTensorsWNA16RDNA3MoEMethod(
    RDNA3FusedMoEMixin, CompressedTensorsWNA16MoEMethod
):
    """W4A16 MoE using the fused RDNA3 HIP kernel (moe_gptq_gemm_rdna3).

    Weights are in RDNA3 format (shuffled int32 [E, K/8, N]),
    NOT Triton format (transposed uint8). apply() dispatches through
    the fused HIP kernel directly.
    """

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        device = layer.w13_weight_packed.device
        num_experts = layer.w13_weight_packed.shape[0]
        empty_g_idx = torch.empty(0, dtype=torch.int32, device=device)

        # Shuffle weights in-place per expert (exllama nibble interleave)
        for e in range(num_experts):
            w13_e = layer.w13_weight_packed.data[e].contiguous()
            ops.gptq_shuffle(w13_e, empty_g_idx, 4)
            layer.w13_weight_packed.data[e] = w13_e
            w2_e = layer.w2_weight_packed.data[e].contiguous()
            ops.gptq_shuffle(w2_e, empty_g_idx, 4)
            layer.w2_weight_packed.data[e] = w2_e

        # Keep scales as [E, groups, N] in activation dtype
        act_dtype = layer.w13_weight_scale.dtype
        layer.w13_weight_scale = torch.nn.Parameter(
            layer.w13_weight_scale.to(dtype=act_dtype).contiguous(),
            requires_grad=False,
        )
        layer.w2_weight_scale = torch.nn.Parameter(
            layer.w2_weight_scale.to(dtype=act_dtype).contiguous(),
            requires_grad=False,
        )

        # Synthesize packed zero points: [E, groups, N/8] int32
        w13_groups = (layer.w13_weight_packed.shape[1] * 8) // self.group_size
        w13_N = layer.w13_weight_packed.shape[2]
        w2_groups = (layer.w2_weight_packed.shape[1] * 8) // self.group_size
        w2_N = layer.w2_weight_packed.shape[2]

        w13_qz = _synthesize_qzeros(w13_groups, w13_N, device)
        w2_qz = _synthesize_qzeros(w2_groups, w2_N, device)
        layer.w13_qzeros = torch.nn.Parameter(
            w13_qz.unsqueeze(0).expand(num_experts, -1, -1).contiguous(),
            requires_grad=False,
        )
        layer.w2_qzeros = torch.nn.Parameter(
            w2_qz.unsqueeze(0).expand(num_experts, -1, -1).contiguous(),
            requires_grad=False,
        )

        # Pre-allocate reusable buffers for decode (sizes based on top_k=8)
        N_gate_up = w13_N
        hidden_size = w2_N
        intermediate = N_gate_up // 2  # gated activation
        # Max tokens we expect in decode; prefill will re-allocate if needed
        max_decode_tokens = 16
        top_k = 8  # conservative default
        buf_size = max_decode_tokens * top_k
        layer.rdna3_w1_buf = torch.zeros(
            buf_size, N_gate_up, dtype=act_dtype, device=device
        )
        layer.rdna3_act_buf = torch.empty(
            buf_size, intermediate, dtype=act_dtype, device=device
        )
        layer.rdna3_out_buf = torch.zeros(
            max_decode_tokens, hidden_size, dtype=act_dtype, device=device
        )
        layer.rdna3_empty_tw = torch.empty(0, device=device)

    def _select_block_size_m(self, num_tokens, top_k, num_experts):
        # moe_gptq_gemm_rdna3 only has tiles {1,2,4,8} (no WMMA-16 path); keep
        # the proven W4A16 cap so prefill / batched decode don't hit the
        # kernel's TORCH_CHECK default arm.
        return 1 if num_tokens <= 4 else 4

    def _gemm_w13(self, layer, a, c, tw, sti, eid, ntp, top_k, block_size_m, mul_tw):
        ops.moe_gptq_gemm_rdna3(
            a,
            c,
            layer.w13_weight_packed,
            layer.w13_weight_scale,
            layer.w13_qzeros,
            tw,
            sti,
            eid,
            ntp,
            top_k,
            block_size_m,
            mul_tw,
        )

    def _gemm_w2(
        self, layer, a, c, tw, sti, eid, ntp, block_size_m, mul_tw, output_topk
    ):
        ops.moe_gptq_gemm_rdna3(
            a,
            c,
            layer.w2_weight_packed,
            layer.w2_weight_scale,
            layer.w2_qzeros,
            tw,
            sti,
            eid,
            ntp,
            1,
            block_size_m,
            mul_tw,
            output_topk,
        )
