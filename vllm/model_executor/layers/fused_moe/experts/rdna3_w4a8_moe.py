# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Int4 x int8 (W4A8) MoE decode experts for RDNA3/RDNA3.5 (gfx11).

For small batches, `moe_w4a8_decode_rdna3` replaces the two Triton WNA16
GEMMs, the activation and the top-k reduction with three HIP launches.
Activations and the SiLU-and-mul output are quantized to int8 per 32 values
and reduced with `v_dot4_i32_iu8`, so the numerics differ slightly from
the W4A16 Triton path; this backend is therefore only used when requested
with `--moe-backend rdna3_w4a8`.

Weights stay in the `TritonWNA16Experts` format (uint8, two codes per byte,
optional nibble-packed zero points), so larger batches and unsupported
configurations fall back to the Triton kernels on the same tensors. A request
therefore decodes with int8 activations while the batch has at most
`RDNA3_W4A8_MAX_TOKENS` tokens and with fp16/bf16 activations above that.
"""

from math import prod

import torch

import vllm._custom_ops as ops
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.triton_moe import (
    TritonWNA16Experts,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kInt4Static,
    kInt4Static32,
    kInt4Static32Asym,
    kInt4StaticAsym,
)
from vllm.platforms import current_platform

# Above this many tokens the weight reuse of the Triton GEMMs wins.
RDNA3_W4A8_MAX_TOKENS = 8


def _align16(nbytes: int) -> int:
    return (nbytes + 15) // 16 * 16


def moe_w4a8_workspace_bytes(M: int, K: int, N: int, top_k: int) -> int:
    """Scratch bytes `moe_w4a8_decode_rdna3` carves from its workspace: int8
    activations with fp32 scales and int32 sums per 32 values, for the input
    [M, K] and the activation output [M * top_k, N] (N = intermediate size).
    Mirrors `workspace_bytes` in csrc/rocm/moe_w4a8_decode_rdna3.cu."""
    rows = M * top_k
    return (
        _align16(M * K)
        + 2 * _align16(M * (K // 32) * 4)
        + _align16(rows * N)
        + 2 * _align16(rows * (N // 32) * 4)
    )


def rdna3_w4a8_moe_kernel_available() -> bool:
    """Whether the W4A8 MoE decode kernel is built and the GPU is gfx11."""
    if not current_platform.is_rocm():
        return False

    from vllm.platforms.rocm import on_gfx11

    return (
        on_gfx11()
        and hasattr(torch.ops, "_rocm_C")
        and hasattr(torch.ops._rocm_C, "moe_w4a8_decode_rdna3")
    )


class Rdna3W4A8Experts(TritonWNA16Experts):
    """Triton WNA16 experts with a W4A8 HIP decode path on gfx11."""

    @staticmethod
    def _supports_current_device() -> bool:
        return rdna3_w4a8_moe_kernel_available()

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return activation_key is None and weight_key in (
            kInt4Static,
            kInt4StaticAsym,
            kInt4Static32,
            kInt4Static32Asym,
        )

    def _use_decode_kernel(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        activation: MoEActivation,
        apply_router_weight_on_input: bool,
    ) -> bool:
        if hidden_states.size(0) > RDNA3_W4A8_MAX_TOKENS:
            return False
        if activation != MoEActivation.SILU:
            return False
        if self.activation_config.clamp_limit is not None:
            return False
        if apply_router_weight_on_input or self._lora_context is not None:
            return False
        # The kernel weights and reduces the top-k rows itself.
        if self._uses_post_gemm2_hook(hidden_states.size(0)):
            return False
        if self.w1_bias is not None or self.w2_bias is not None:
            return False
        if hidden_states.dtype not in (torch.float16, torch.bfloat16):
            return False
        if self.w1_scale is None or self.w2_scale is None:
            return False
        if self.w1_scale.dtype != hidden_states.dtype:
            return False
        if not (hidden_states.is_contiguous() and output.is_contiguous()):
            return False
        # Hidden size, intermediate size and both group sizes must be
        # multiples of 32 (one int8 activation group).
        K = hidden_states.size(1)
        N = w2.size(2) * 2
        groups1 = self.w1_scale.size(2)
        groups2 = self.w2_scale.size(2)
        return (
            K % 32 == 0
            and N % 32 == 0
            and K % groups1 == 0
            and (K // groups1) % 32 == 0
            and N % groups2 == 0
            and (N // groups2) % 32 == 0
        )

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        workspace13, workspace2, output = super().workspace_shapes(
            M,
            N,
            K,
            topk,
            global_num_experts,
            local_num_experts,
            expert_tokens_meta,
            activation,
        )
        if M <= RDNA3_W4A8_MAX_TOKENS:
            # The decode kernel's scratch lives in workspace2, which holds
            # fp16/bf16 elements (N is the gate_up width, 2x intermediate).
            needed = -(-moe_w4a8_workspace_bytes(M, K, N // 2, topk) // 2)
            if needed > prod(workspace2):
                workspace2 = (needed,)
        return workspace13, workspace2, output

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        if not self._use_decode_kernel(
            output, hidden_states, w1, w2, activation, apply_router_weight_on_input
        ):
            super().apply(
                output,
                hidden_states,
                w1,
                w2,
                topk_weights,
                topk_ids,
                activation,
                global_num_experts,
                expert_map,
                a1q_scale,
                a2_scale,
                workspace13,
                workspace2,
                expert_tokens_meta,
                apply_router_weight_on_input,
            )
            return

        assert self.w1_scale is not None and self.w2_scale is not None
        # The kernel maps global ids through expert_map itself and skips slots
        # whose local expert is outside [0, E), which is where expert_map
        # sends experts owned by other ranks (-1).
        ops.moe_w4a8_decode_rdna3(
            output,
            hidden_states,
            topk_ids.contiguous(),
            topk_weights.contiguous(),
            expert_map,
            workspace2.view(-1).view(torch.uint8),
            w1,
            self.w1_scale,
            self.w1_zp,
            w2,
            self.w2_scale,
            self.w2_zp,
        )
