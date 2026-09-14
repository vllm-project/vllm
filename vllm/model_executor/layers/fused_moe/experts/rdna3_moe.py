# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused MoE W4A16 experts on the RDNA3 (gfx1100) HIP kernel.

``moe_gptq_gemm_rdna3`` is a single HIP kernel launch per GEMM that handles
expert routing + W4A16 dequant + dot product with atomic output accumulation.

Weight format (per expert, same as the dense RDNA3 W4A16 kernel):
  - Packed int32 ``[E, K/8, N]`` with exllama shuffle
  - Scales ``[E, groups, N]`` in activation dtype
  - Zero points ``[E, groups, N/8]`` packed int32 (synthesized, symmetric only)
"""

import torch

import vllm._custom_ops as ops
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation_supported,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kInt4Static,
    kInt4Static32,
)
from vllm.platforms import current_platform


def rdna3_moe_kernel_available() -> bool:
    """Whether the fused RDNA3 MoE HIP kernel is built into this binary."""
    if not current_platform.is_rocm():
        return False

    from vllm.platforms.rocm import on_gfx1100

    return (
        on_gfx1100()
        and hasattr(torch.ops, "_rocm_C")
        and hasattr(torch.ops._rocm_C, "moe_gptq_gemm_rdna3")
    )


class Rdna3WNA16Experts(mk.FusedMoEExpertsModular):
    """W4A16 experts backed by ``moe_gptq_gemm_rdna3`` (gfx1100).

    Both GEMMs accumulate atomically, so their destinations are zeroed first.
    The second GEMM is given ``output_topk``, which makes it reduce over the
    top-k dimension while it accumulates — the ``moe_sum`` launch and the
    ``[M * top_k, K]`` intermediate are fused away, hence
    ``TopKWeightAndReduceNoOP``.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        assert quant_config.use_int4_w4a16, "Supports only int4_w4a16"
        super().__init__(
            moe_config=moe_config,
            quant_config=quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=num_dispatchers,
        )
        self._empty_topk_weights: torch.Tensor | None = None

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    @staticmethod
    def _supports_current_device() -> bool:
        return rdna3_moe_kernel_available()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        # Symmetric int4 weights only: the kernel consumes synthesized zero
        # points and has no path for checkpoint zero points.
        return activation_key is None and weight_key in (kInt4Static, kInt4Static32)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return apply_moe_activation_supported(activation)

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return not (
            moe_parallel_config.use_fi_nvl_two_sided_kernels
            or moe_parallel_config.use_fi_nvl_one_sided_kernels
        )

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        # Weights are packed K-first: w1 is [E, K // 8, w13_shards * N] and
        # w2 is [E, N // 8, K], so neither N nor K can be read off a trailing
        # dimension the way the base implementation does.
        assert w1.dim() == 3 and w2.dim() == 3
        assert a1.dim() == 2
        assert topk_ids.size(0) == a1.size(0), f"{topk_ids.size(0)} != {a1.size(0)}"
        return w1.size(0), a1.size(0), w2.size(1) * 8, a1.size(-1), topk_ids.size(1)

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
        # The modular kernel provisions the output buffer out of workspace13,
        # so both GEMM-1 buffers have to live in workspace2 — the second GEMM
        # reads the activation output while accumulating into the output.
        gate_up = N if not activation.is_gated else 2 * N
        return ((M, K), (M * topk * (gate_up + N),), (M, K))

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
    ) -> None:
        assert self.w1_scale is not None and self.w2_scale is not None
        assert self.w1_zp is not None and self.w2_zp is not None

        E, M, N, K, top_k = self.moe_problem_size(hidden_states, w1, w2, topk_ids)
        if global_num_experts == -1:
            global_num_experts = E

        gate_up = w1.size(2)
        act_n = self.adjust_N_for_activation(gate_up, activation)
        rows = M * top_k

        scratch = workspace2.view(-1)
        gate_up_out = scratch[: rows * gate_up].view(rows, gate_up)
        act_out = scratch[rows * gate_up : rows * (gate_up + act_n)].view(rows, act_n)

        # BLOCK_SIZE_M=1 for decode (no padding waste), 4 for prefill.
        block_size_m = 1 if M <= 4 else 4
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids,
            block_size_m,
            global_num_experts,
            expert_map,
        )

        if (
            self._empty_topk_weights is None
            or self._empty_topk_weights.device != hidden_states.device
        ):
            self._empty_topk_weights = torch.empty(
                0, dtype=torch.float32, device=hidden_states.device
            )
        topk_weights_f32 = topk_weights.reshape(-1).float()
        no_topk_weights = self._empty_topk_weights

        gate_up_out.zero_()
        ops.moe_gptq_gemm_rdna3(
            hidden_states,
            gate_up_out,
            w1,
            self.w1_scale,
            self.w1_zp,
            topk_weights_f32 if apply_router_weight_on_input else no_topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            top_k,
            block_size_m,
            apply_router_weight_on_input,
        )

        self.activation(activation, act_out, gate_up_out)

        # output_topk=top_k makes the kernel accumulate into out[token_id],
        # fusing the top-k reduction into the atomic write-back.
        output.zero_()
        ops.moe_gptq_gemm_rdna3(
            act_out,
            output,
            w2,
            self.w2_scale,
            self.w2_zp,
            no_topk_weights if apply_router_weight_on_input else topk_weights_f32,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            1,
            block_size_m,
            not apply_router_weight_on_input,
            output_topk=top_k,
        )
