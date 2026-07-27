# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native RDNA3 (gfx1100) MXFP4 fused-MoE experts backend.

A modular ``FusedMoEExperts`` that drives the ``moe_mxfp4_gemm_rdna3`` HIP kernel
(E2M1 weights + E8M0 group-32 scale, no zero point). Standard activation format:
``apply`` receives the full hidden states + topk ids/weights and does its own
expert routing (moe_align_block_size), the two GEMMs, the activation, the
per-expert bias and the weighted reduction — so it covers both the
compressed-tensors MXFP4 MoE (no bias, SiLU) and GPT-OSS native-mxfp4 (per-expert
bias + clamped SwiGLU-OAI) by reading everything from ``quant_config``.

Registered as ``Mxfp4MoeBackend.RDNA3_MXFP4`` in the oracle and gated on gfx1100.
"""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEParallelConfig,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Static,
)


def _select_block_size_m(num_tokens: int, top_k: int, num_experts: int) -> int:
    """Pick the GEMM tile from expected occupancy (avg tokens/expert).

    >=16-row steps fill the WMMA-16 tile (clustered routing fills it even on
    256-expert models; also where prefill lives); sub-16 batches use a scalar
    tile sized to occupancy (bsm=1 beats bsm=4 at low concurrency).
    """
    if num_experts <= 0 or top_k <= 0:
        return 1
    if num_tokens >= 16:
        return 16
    tokens_per_expert = num_tokens * top_k / num_experts
    return 4 if tokens_per_expert >= 1.5 else 1


def _rocm_op_available() -> bool:
    return hasattr(torch.ops, "_rocm_C") and hasattr(
        torch.ops._rocm_C, "moe_mxfp4_gemm_rdna3"
    )


def repack_experts_rdna3(
    packed: torch.Tensor, scale: torch.Tensor, deinterleave: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """[E, N, K/2] u8 + [E, N, K/32] u8 -> [E, K/8, N] int32 + [E, K/32, N] u8.

    The kernel reads weights as [E, K/8, N] uint32 (4 consecutive packed bytes
    little-endian = a uint32 of 8 E2M1 codes along K). Per-expert into
    pre-allocated outputs so a multi-GB repack fits next to the loaded weights.
    ``deinterleave`` splits GPT-OSS's gate/up rows ([g0,u0,...] -> [g...,u...]).
    """
    E, N, kh = packed.shape
    dev = packed.device
    b_q = torch.empty(E, kh // 4, N, dtype=torch.int32, device=dev)
    b_scale = torch.empty(E, scale.shape[2], N, dtype=torch.uint8, device=dev)
    for e in range(E):
        pe = packed[e]
        se = scale[e]
        if deinterleave:
            pe = torch.cat([pe[::2], pe[1::2]], dim=0)
            se = torch.cat([se[::2], se[1::2]], dim=0)
        b_q[e] = pe.contiguous().view(torch.int32).t()
        b_scale[e] = se.t()
    return b_q, b_scale


class RDNA3Mxfp4Experts(mk.FusedMoEExpertsModular):
    """MXFP4 MoE experts on the native RDNA3 HIP kernel (gfx1100 only)."""

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        from vllm.platforms import current_platform

        if not current_platform.is_rocm() or not _rocm_op_available():
            return False
        from vllm.platforms.rocm import on_gfx1100

        return on_gfx1100()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False  # gated MLP only (gate_up -> act -> down)

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        # MXFP4 weight-only (W4A16); no activation quantization.
        return (weight_key, activation_key) == (kMxfp4Static, None)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (MoEActivation.SILU, MoEActivation.SWIGLUOAI)

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # The kernel/epilogue applies the topk weights and reduces, so the
        # prepare/finalize step must not reduce again.
        return TopKWeightAndReduceNoOP()

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
        # apply() allocates its own scratch (shapes derived from the weights),
        # so request no framework workspaces; output is the reduced [M, K].
        return ((0,), (0,), (M, K))

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
        qc = self.quant_config
        M = hidden_states.shape[0]
        top_k = topk_ids.shape[1]
        total = M * top_k
        n_gate_up = w1.shape[2]  # [E, K/8, 2*intermediate]
        intermediate = n_gate_up // 2
        hidden = w2.shape[2]
        dtype = hidden_states.dtype
        device = hidden_states.device

        gne = global_num_experts if global_num_experts > 0 else w1.shape[0]
        block_size_m = _select_block_size_m(M, top_k, gne)
        sti, eid, ntp = moe_align_block_size(topk_ids, block_size_m, gne, expert_map)

        topk_w_f32 = topk_weights.reshape(-1).float()
        empty_tw = torch.empty(0, device=device)
        w1_bias = qc.w1_bias
        w2_bias = qc.w2_bias
        flat_experts = (
            topk_ids.reshape(-1).long()
            if (w1_bias is not None or w2_bias is not None)
            else None
        )

        # gate_up GEMM (no reduction); router weight optionally folded in here.
        w1_out = torch.zeros(total, n_gate_up, dtype=dtype, device=device)
        ops.moe_mxfp4_gemm_rdna3(
            hidden_states,
            w1_out,
            w1,
            qc.w1_scale,
            topk_w_f32 if apply_router_weight_on_input else empty_tw,
            sti,
            eid,
            ntp,
            top_k,
            block_size_m,
            apply_router_weight_on_input,
            0,
        )
        if w1_bias is not None:
            w1_out = w1_out + w1_bias[flat_experts].to(dtype)

        act_out = torch.empty(total, intermediate, dtype=dtype, device=device)
        if activation == MoEActivation.SWIGLUOAI:
            self._swiglu_oai(w1_out, act_out, qc)
        else:
            apply_moe_activation(activation, act_out, w1_out)

        # down GEMM. The kernel's fused output_topk reduce is wrong under TP
        # (each rank's down-proj is a partial the layer all-reduces afterwards);
        # a per-expert bias likewise needs adding before the reduction. In both
        # cases write unreduced rows and reduce in Python instead.
        unfused = w2_bias is not None or get_tensor_model_parallel_world_size() > 1
        if unfused:
            w2_out = torch.zeros(total, hidden, dtype=dtype, device=device)
            ops.moe_mxfp4_gemm_rdna3(
                act_out,
                w2_out,
                w2,
                qc.w2_scale,
                empty_tw,
                sti,
                eid,
                ntp,
                1,
                block_size_m,
                False,
                0,
            )
            if w2_bias is not None:
                w2_out = w2_out + w2_bias[flat_experts].to(dtype)
            if not apply_router_weight_on_input:
                w2_out = w2_out * topk_weights.reshape(-1, 1).to(dtype)
            output.copy_(w2_out.view(M, top_k, hidden).sum(dim=1))
        else:
            output.zero_()
            ops.moe_mxfp4_gemm_rdna3(
                act_out,
                output,
                w2,
                qc.w2_scale,
                empty_tw if apply_router_weight_on_input else topk_w_f32,
                sti,
                eid,
                ntp,
                top_k,
                block_size_m,
                not apply_router_weight_on_input,
                top_k,
            )

    @staticmethod
    def _swiglu_oai(x: torch.Tensor, out: torch.Tensor, qc) -> None:
        # gate||up contiguous; OAI clamped SwiGLU with quant-config params.
        alpha = qc.gemm1_alpha if qc.gemm1_alpha is not None else 1.702
        beta = qc.gemm1_beta if qc.gemm1_beta is not None else 1.0
        limit = qc.gemm1_clamp_limit
        d = x.shape[-1] // 2
        gate = x[..., :d]
        up = x[..., d:]
        if limit is not None:
            gate = torch.clamp(gate, max=limit)
            up = torch.clamp(up, min=-limit, max=limit)
        out.copy_(gate * torch.sigmoid(alpha * gate) * (up + beta))
