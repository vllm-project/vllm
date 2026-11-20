# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native RDNA4 (gfx12) MXFP4 fused-MoE experts backend.

A modular `FusedMoEExperts` that drives the `moe_mxfp4_gemm_rdna4` HIP kernels
(E2M1 weights + E8M0 group-32 scale, no zero point) with FP8 activations:
weights are decoded on the fly to E5M2 with the group scale folded into the
exponent, activations are quantized per token to E4M3, and the matrix engine
runs the fp8_bf8 WMMA / SWMMAC forms. Standard activation format: `apply`
receives the full hidden states + topk ids/weights and does its own expert
routing (moe_align_block_size), the two GEMMs, the activation, the per-expert
bias and the weighted reduction, so it covers both the compressed-tensors MXFP4
MoE (no bias, SiLU) and GPT-OSS native MXFP4 (per-expert bias + clamped
SwiGLU-OAI).

Weights are stored fragment-major ([E, K/32, N, 4] uint32, see
`repack_experts_rdna4`) with scales [E, K/32, N] uint8; the per-column fold
base ([E, N] uint8) is derived from the scales once per layer.

The checkpoint is weight-only MXFP4 (activation key None); the FP8 activation
quantization is an implementation detail of this backend, like the dense
RDNA4 path, so it is routed by the W4A16 scheme.

Registered as `Mxfp4MoeBackend.RDNA4_MXFP4` in the oracle and gated on gfx12.
"""

import contextlib
import dataclasses

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
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
    kMxfp4Static,
)

with contextlib.suppress(ImportError):  # non-ROCm builds
    import vllm._rocm_C  # noqa: F401


def _rocm_ops():
    return torch.ops._rocm_C


def repack_experts_rdna4(
    packed: torch.Tensor, scale: torch.Tensor, deinterleave: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """[E, N, K/2] u8 + [E, N, K/32] u8 -> [E, K/32, N, 4] u32 + [E, K/32, N] u8.

    Four consecutive packed bytes along K (low nibble = even k) form one
    uint32 of 8 E2M1 codes; the kernel wants them fragment-major so one lane's
    32-K fragment is a single 16-byte load. Per-expert so a multi-GB repack
    fits next to the loaded weights. `deinterleave` splits GPT-OSS's gate/up
    rows ([g0,u0,...] -> [g...,u...]).
    """
    E, N, kh = packed.shape
    dev = packed.device
    k8 = kh // 4
    b_q = torch.empty(E, k8 // 4, N, 4, dtype=torch.uint32, device=dev)
    b_scale = torch.empty(E, scale.shape[2], N, dtype=torch.uint8, device=dev)
    for e in range(E):
        pe = packed[e]
        se = scale[e]
        if deinterleave:
            pe = torch.cat([pe[::2], pe[1::2]], dim=0)
            se = torch.cat([se[::2], se[1::2]], dim=0)
        kn = pe.contiguous().view(torch.uint32).view(N, k8).t().contiguous()
        b_q[e] = _rocm_ops().mxfp4_moe_repack_rdna4(kn.unsqueeze(0))[0]
        b_scale[e] = se.t()
    return b_q, b_scale


def _first_set(*values: float | None) -> float:
    return next(v for v in values if v is not None)


class RDNA4Mxfp4Experts(mk.FusedMoEExpertsModular):
    """MXFP4 (W4A8) MoE experts on the native RDNA4 HIP kernels (gfx12 only)."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        super().__init__(moe_config, quant_config, max_num_tokens, num_dispatchers)
        # The repack de-interleaves GPT-OSS's gate/up rows, so SWIGLUOAI runs as
        # the packed SWIGLUOAI_UNINTERLEAVE. Not every quant path forwards the
        # SwiGLU constants (gpt_oss.py does not), so fall back to the values
        # swigluoai_and_mul hardcodes.
        self.swigluoai_config = dataclasses.replace(
            self.activation_config,
            clamp_limit=_first_set(
                quant_config.gemm1_clamp_limit, moe_config.swiglu_limit, 7.0
            ),
            alpha=_first_set(quant_config.gemm1_alpha, moe_config.swiglu_alpha, 1.702),
            beta=_first_set(quant_config.gemm1_beta, moe_config.swiglu_beta, 1.0),
        )
        # Per-column E5M2 fold base, derived from the scales once per layer and
        # keyed on the scale tensor (one backend instance may serve one layer,
        # but the quant config can be rebuilt).
        self._fold_base: dict[int, torch.Tensor] = {}

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        from vllm.platforms import current_platform

        if not current_platform.is_rocm():
            return False
        from vllm.platforms.rocm import on_gfx12x

        return on_gfx12x()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False  # gated MLP only (gate_up -> act -> down)

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        # Weight-only MXFP4 checkpoints; the per-token FP8 activation
        # quantization is done inside `apply`.
        return (weight_key, activation_key) == (kMxfp4Static, None)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (MoEActivation.SILU, MoEActivation.SWIGLUOAI)

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # The router weights are applied and the top-k rows reduced here, so
        # the prepare/finalize step must not reduce again.
        return TopKWeightAndReduceNoOP()

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        # Fragment-major weights: w1 is [E, K/32, 2N, 4], w2 is [E, N/32, K, 4].
        assert w1.dim() == 4 and w2.dim() == 4
        assert a1.dim() == 2
        assert topk_ids.size(0) == a1.size(0), f"{topk_ids.size(0)} != {a1.size(0)}"
        return w1.size(0), a1.size(0), w1.size(2) // 2, a1.size(-1), topk_ids.size(1)

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
        # The modular kernel provisions `output` out of workspace13, so every
        # buffer a GEMM reads while another writes `output` has to come out of
        # workspace2: the activation output stays live across the down GEMM.
        # The gate/up GEMM output is dead once the activation has run, so the
        # down GEMM reuses that region.
        scratch = M * topk * (N + max(2 * N, K))
        return ((M, K), (scratch,), (M, K))

    def _base_for(self, scale: torch.Tensor) -> torch.Tensor:
        key = scale.data_ptr()
        base = self._fold_base.get(key)
        if base is None:
            base = _rocm_ops().mxfp4_moe_fold_base_rdna4(scale)
            self._fold_base[key] = base
        return base

    @staticmethod
    def _quant_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q, s = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
        return q.view(torch.uint8), s.reshape(-1).to(torch.float32)

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
        rocm = _rocm_ops()
        E, M, N, K, top_k = self.moe_problem_size(hidden_states, w1, w2, topk_ids)
        n_gate_up = 2 * N
        total = M * top_k
        dtype = hidden_states.dtype
        device = hidden_states.device

        gne = global_num_experts if global_num_experts > 0 else E
        block_size_m = int(rocm.mxfp4_moe_block_size_rdna4(M, top_k, gne))
        sti, eid, ntp = moe_align_block_size(topk_ids, block_size_m, gne, expert_map)

        scratch = workspace2.view(-1)
        act_out = scratch[: total * N].view(total, N)
        gemm_out = scratch[total * N :]
        w1_out = gemm_out[: total * n_gate_up].view(total, n_gate_up)

        topk_w_f32 = topk_weights.reshape(-1).to(torch.float32).contiguous()
        empty_tw = torch.empty(0, device=device, dtype=torch.float32)
        w1_bias = qc.w1_bias
        w2_bias = qc.w2_bias
        fused_w1_bias = None if w1_bias is None else w1_bias.to(dtype).contiguous()
        fused_w2_bias = None if w2_bias is None else w2_bias.to(dtype).contiguous()
        base1 = self._base_for(qc.w1_scale)
        base2 = self._base_for(qc.w2_scale)

        # gate_up GEMM: per-token FP8 activations, bias fused. The epilogue is
        # (acc * topk_w) + bias, which is the right order when the router
        # weight scales the input (bias is not scaled).
        a_fp8, a_sc = self._quant_rows(hidden_states)
        rocm.moe_mxfp4_gemm_rdna4(
            a_fp8,
            a_sc,
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
            fused_w1_bias,
            base1,
        )

        if activation == MoEActivation.SWIGLUOAI:
            apply_moe_activation(
                MoEActivation.SWIGLUOAI_UNINTERLEAVE,
                act_out,
                w1_out,
                activation_config=self.swigluoai_config,
            )
        else:
            self.activation(activation, act_out, w1_out)

        # down GEMM, one row per (token, expert) slot. The kernel adds the bias
        # after the router weight, so with a per-expert bias the weight is
        # applied here, before the top-k reduction; without one the kernel
        # applies it.
        a2_fp8, a2_sc = self._quant_rows(act_out)
        w2_out = gemm_out[: total * K].view(total, K)
        kernel_weights = not apply_router_weight_on_input and fused_w2_bias is None
        rocm.moe_mxfp4_gemm_rdna4(
            a2_fp8,
            a2_sc,
            w2_out,
            w2,
            qc.w2_scale,
            topk_w_f32 if kernel_weights else empty_tw,
            sti,
            eid,
            ntp,
            1,
            block_size_m,
            kernel_weights,
            fused_w2_bias,
            base2,
        )
        if not apply_router_weight_on_input and not kernel_weights:
            w2_out.mul_(topk_weights.reshape(-1, 1).to(dtype))
        torch.sum(w2_out.view(M, top_k, K), dim=1, out=output)
