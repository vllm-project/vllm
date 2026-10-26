# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused MoE Octave-W3 experts on the RDNA3 (gfx1100) HIP kernel.

Each expert weight is ``W ~= (s * LV[c]) @ Hb.T``: 3-bit codes ``c`` over the
eight Lloyd-Max levels of the unit normal, an fp16 scale ``s`` per output row
and group of 32 inputs, and a block Hadamard ``Hb`` along K that the kernel
applies to its input. Routing and write-back follow ``Rdna3WNA16Experts``.

Checkpoint format (``octave_w3_planar_v0``), per expert and projection:
``weight_packed`` int32 ``[N, K/32*3]`` (three bit planes per group of 32,
bit j of plane b is bit b of the code of column j) and ``weight_scale`` fp16
``[N, K/32]``. ``octw3_repack`` turns them into the kernel layout.
"""

import torch

import vllm._custom_ops as ops
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.experts.rdna3_moe import (
    _identity_routing,
    rdna3_moe_kernel_available,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey

OCTW3_LEVELS = (-2.1520, -1.3439, -0.7560, -0.2451, 0.2451, 0.7560, 1.3439, 2.1520)


def octw3_codes(packed: torch.Tensor) -> torch.Tensor:
    """Planar ``[..., G*3]`` int32 -> codes ``[..., G, 32]`` int64 in 0..7."""
    planes = packed.to(torch.int64).unflatten(-1, (-1, 3)) & 0xFFFFFFFF
    bits = torch.arange(32, device=packed.device, dtype=torch.int64)
    return sum(((planes[..., b, None] >> bits) & 1) << b for b in range(3))


def octw3_repack(packed: torch.Tensor, chunk: int = 4) -> torch.Tensor:
    """Planar ``[E, N, G*3]`` -> kernel layout ``[E, G, 3, N]`` int32.

    A kernel code holds the sign in bit 2 and the magnitude index (into the
    four positive levels) in bits 0-1. Within the three words of a (group,
    column), code bytes 0-5 take bits 0-2 / 3-5 of each byte of words 0, 0,
    1, 1, 2, 2; codes 24-31 are split over bits 6-7 (see octw3_codes() in
    csrc/rocm/moe_octw3_rdna3.cu).
    """
    E, N, G3 = packed.shape
    out = torch.empty(E, G3 // 3, 3, N, dtype=torch.int32, device=packed.device)
    for e0 in range(0, E, chunk):
        c = octw3_codes(packed[e0 : e0 + chunk])
        c = c ^ torch.where(c >= 4, 4, 7)
        w = torch.zeros(*c.shape[:-1], 3, dtype=torch.int64, device=c.device)
        for j in range(24):
            q, k = divmod(j, 4)
            w[..., q // 2] |= c[..., j] << (8 * k + 3 * (q % 2))
        for k in range(4):
            w[..., 0] |= (c[..., 24 + k] & 3) << (8 * k + 6)
            w[..., 1] |= ((c[..., 24 + k] >> 2) & 1) << (8 * k + 6)
            w[..., 2] |= (c[..., 28 + k] & 3) << (8 * k + 6)
            w[..., 1] |= ((c[..., 28 + k] >> 2) & 1) << (8 * k + 7)
        w = torch.where(w >= 2**31, w - 2**32, w).to(torch.int32)
        out[e0 : e0 + chunk] = w.permute(0, 2, 3, 1)
    return out


class Rdna3OctaveW3Experts(mk.FusedMoEExpertsModular):
    """Octave-W3 experts backed by ``moe_octw3_gemm_rdna3`` (gfx1100).

    ``w1`` is ``[E, K/32, 3, 2N]`` and ``w2`` is ``[E, N/32, 3, K]`` (kernel
    layout); the scales are ``[E, groups, out]`` fp16.
    """

    writes_output_from_zero = True

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        had_w13: int,
        had_w2: int,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        super().__init__(
            moe_config=moe_config,
            quant_config=quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=num_dispatchers,
        )
        self.had_w13 = had_w13
        self.had_w2 = had_w2
        self._empty_topk_weights: torch.Tensor | None = None

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    @staticmethod
    def _supports_current_device() -> bool:
        return rdna3_moe_kernel_available() and hasattr(
            torch.ops._rocm_C, "moe_octw3_gemm_rdna3"
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return activation_key is None

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return not moe_parallel_config.use_ep

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        assert w1.dim() == 4 and w2.dim() == 4 and a1.dim() == 2
        assert topk_ids.size(0) == a1.size(0), f"{topk_ids.size(0)} != {a1.size(0)}"
        return w1.size(0), a1.size(0), w2.size(1) * 32, a1.size(-1), topk_ids.size(1)

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
        return ((M, K), (M * topk * 2 * N,), (M, K))

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
        assert hidden_states.dtype == torch.float16

        E, M, N, K, top_k = self.moe_problem_size(hidden_states, w1, w2, topk_ids)
        if global_num_experts == -1:
            global_num_experts = E
        rows = M * top_k
        gate_up_out = workspace2.view(-1)[: rows * 2 * N].view(rows, 2 * N)

        # Same block choice as Rdna3WNA16Experts.
        block_size_m = 1 if M <= 24 else 4
        if block_size_m == 1 and expert_map is None:
            sorted_token_ids, num_tokens_post_padded = _identity_routing(
                rows, hidden_states.device
            )
            expert_ids = topk_ids.reshape(-1).to(torch.int32)
        else:
            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                topk_ids, block_size_m, global_num_experts, expert_map
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

        torch._foreach_zero_([gate_up_out, output])
        ops.moe_octw3_gemm_rdna3(
            hidden_states,
            gate_up_out,
            w1,
            self.w1_scale,
            topk_weights_f32 if apply_router_weight_on_input else no_topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            top_k,
            block_size_m,
            apply_router_weight_on_input,
            0,
            self.had_w13,
        )
        # Given [gate | up], the kernel applies SiLU-and-mul as it loads it.
        ops.moe_octw3_gemm_rdna3(
            gate_up_out,
            output,
            w2,
            self.w2_scale,
            no_topk_weights if apply_router_weight_on_input else topk_weights_f32,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            1,
            block_size_m,
            not apply_router_weight_on_input,
            top_k,
            self.had_w2,
        )
