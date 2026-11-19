# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoE engine operands from vLLM's loaded MiniMax-M3 MoE layer, and the mode-Q FC2
weight prep.

Source of every routed-expert operand: vLLM's NVFP4 MoE with the FLASHINFER_TRTLLM
backend after
`process_weights_after_loading` (`prepare_nvfp4_moe_layer_for_fi_or_cutlass` ->
`prepare_static_weights_for_trtllm_fp4_moe`). The engine reads these buffers in place,
never copies them:

==================  ==========================================  ==============================================
engine operand      vLLM tensor                                  layout
==================  ==========================================  ==============================================
w3_w1_weight        `routed_experts.w13_weight`                uint8 [E, 2 I, H / 2] packed e2m1 (low nibble =
                                                                 even column); rows =
                                                                 logical [w3 (up); w1
                                                                 (gate)]
                                                                 (`reorder_w1w3_to_w3w1`),
                                                                 then the gated-act
                                                                 row interleave and the
                                                                 epilogue-tile-128
                                                                 32-row
                                                                 shuffle
                                                                 (`_maybe_get_cached_w3_w1_permute_indices`)
w3_w1_weight_scale  `routed_experts.w13_weight_scale`          e4m3 [E, 2 I, H / 16], one per 16 elements, rows
                                                                 permuted like the
                                                                 weight, then
                                                                 `nvfp4_block_scale_interleave`
                                                                 (128x4 swizzle)
w2_weight           `routed_experts.w2_weight`                 uint8 [E, H, I / 2], 32-row shuffle (epilogue 128)
w2_weight_scale     `routed_experts.w2_weight_scale`           e4m3 [E, H, I / 16], permuted + 128x4 swizzle
fc31_alpha          `quant_config.g1_alphas`                   f32 [E] = a13_scale * w13_weight_scale_2
fc31_scale_c        `fused_experts.g1_scale_c` (re-derived)    f32 [E] = g1_alphas * fc2_enc
act_alpha           `fused_experts.gemm1_alpha`                f32 [E] (SwiGLU-OAI alpha, raw)
act_beta            `fused_experts.gemm1_beta`                 f32 [E] = beta / g1_alphas (vLLM pre-folds it)
act_clamp           `fused_experts.gemm1_clamp_limit`          f32 [E] = limit / g1_alphas (vLLM pre-folds it)
fc2_alpha           `quant_config.g2_alphas` (re-derived)      f32 [E] = w2_weight_scale_2 / fc2_enc
fc31_input_scale    `quant_config.a1_gscale`                   f32 [1], the checkpoint's 1.0 placeholder
router_weight       `block_sparse_moe.gate.weight`             f32 [E, H]
bias                `block_sparse_moe.e_score_correction_bias` f32 [E]
shared gate_up/down `shared_experts.{gate_up,down}_proj`       e4m3 weight [N, K] + UE8M0 scales, 128x4
                                                                 swizzled (one per 32)
==================  ==========================================  ==============================================

Mode Q (the M3.1 QAT recipe): FC1 input per token (the post's s_t, the checkpoint's FC1
input scale is a 1.0
placeholder) and FC2 input under the fixed global encode scale 16. The FC2 encode scale
lives in two per-expert
scalars (exact powers of two): `fc31_scale_c = g1_alphas * fc2_enc` and `fc2_alpha =
w2_scale_2 / fc2_enc`.
vLLM folds whatever FC2 input scale the checkpoint carries (`a2_gscale`) into both;
`prep_mode_q` removes that
fold (it must be one power of two) and applies 16.
"""  # noqa: E501

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch

from .._mxfp8 import Mxfp8LayoutError, mxfp8_linear_nk
from .host import HIDDEN, INTER, NUM_EXPERTS

FC2_SCALE_Q = 16.0  # mode Q fixed FC2 input global (encode) scale; exact power of two
FC1_QAT_MAX = (
    6.0 * 448.0
)  # s_t = bf16(amax / 2688): NVFP4 e2m1 max x e4m3 block-scale max


class MoeWeightError(ValueError):
    pass


@dataclass(frozen=True)
class MoeWeights:
    w13: torch.Tensor
    w13_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor
    g1_alphas: torch.Tensor
    g1_scale_c: torch.Tensor
    g2_alphas: torch.Tensor
    act_alpha: torch.Tensor
    act_beta: torch.Tensor
    act_clamp: torch.Tensor
    a1_gscale: torch.Tensor
    router: torch.Tensor
    bias: torch.Tensor
    shared_gu_w: torch.Tensor
    shared_gu_s: torch.Tensor
    shared_dn_w: torch.Tensor
    shared_dn_s: torch.Tensor
    fc2_enc: float  # the FC2 input encode scale folded into g1_scale_c / g2_alphas

    def expert_args(self) -> tuple:
        """The 11 positional operands of `M3MoeState.layer`."""
        return (
            self.w13,
            self.w13_scale,
            self.w2,
            self.w2_scale,
            self.g1_alphas,
            self.g1_scale_c,
            self.act_alpha,
            self.act_beta,
            self.act_clamp,
            self.g2_alphas,
            self.bias,
        )

    def shared(self) -> tuple:
        return (self.shared_gu_w, self.shared_gu_s, self.shared_dn_w, self.shared_dn_s)

    def validate(self) -> None:
        e, i, h = NUM_EXPERTS, INTER, HIDDEN
        want = {
            "w13": (self.w13, (e, 2 * i, h // 2)),
            "w13_scale": (self.w13_scale, (e, 2 * i, h // 16)),
            "w2": (self.w2, (e, h, i // 2)),
            "w2_scale": (self.w2_scale, (e, h, i // 16)),
            "router": (self.router, (e, h)),
            "shared_gu_w": (self.shared_gu_w, (2 * i, h)),
            "shared_dn_w": (self.shared_dn_w, (h, i)),
        }
        for name, (t, shape) in want.items():
            if tuple(t.shape) != shape or not t.is_contiguous():
                raise MoeWeightError(
                    f"MoE operand {name}: {tuple(t.shape)} (contiguous "
                    f"{t.is_contiguous()}), "
                    f"expected contiguous {shape}"
                )
        for name in ("w13", "w13_scale", "w2", "w2_scale"):
            if getattr(self, name).element_size() != 1:
                raise MoeWeightError(
                    f"MoE operand {name} must be 1-byte (NVFP4 codes / e4m3 scales)"
                )
        for name in (
            "g1_alphas",
            "g1_scale_c",
            "g2_alphas",
            "act_alpha",
            "act_beta",
            "act_clamp",
            "bias",
        ):
            t = getattr(self, name)
            if t.dtype != torch.float32 or t.numel() != e:
                raise MoeWeightError(
                    f"MoE scalar {name}: {t.dtype} x {t.numel()}, expected float32 x "
                    f"{e}"
                )
        if self.a1_gscale.dtype != torch.float32 or self.a1_gscale.numel() != 1:
            raise MoeWeightError("MoE a1_gscale must be float32 [1]")
        if (
            self.shared_gu_w.dtype != torch.float8_e4m3fn
            or self.shared_dn_w.dtype != torch.float8_e4m3fn
        ):
            raise MoeWeightError("shared expert weights must be float8_e4m3fn (MXFP8)")
        if (
            self.shared_gu_s.numel() != 2 * i * h // 32
            or self.shared_dn_s.numel() != h * i // 32
        ):
            raise MoeWeightError(
                "shared expert scales must be the 128x4-swizzled UE8M0 [N, K / 32] "
                "buffers"
            )


def _pow2_scalar(t: torch.Tensor, what: str) -> float:
    v = t.float().reshape(-1)
    f = float(v[0])
    if not bool((v == f).all()) or f <= 0 or math.frexp(f)[0] != 0.5:
        raise MoeWeightError(f"{what} is not one power-of-two value: {v.tolist()[:4]}")
    return f


def prep_mode_q(w: MoeWeights) -> MoeWeights:
    """Mode Q FC2 prep as new tensors (idempotent): fc31_scale_c = g1_alphas x 16,
    fc2_alpha = w2_scale_2 / 16.
    Refuses non-placeholder routed FC1 input scales (mode Q's FC1 scale is the post's
    per-token s_t)."""
    if w.fc2_enc == FC2_SCALE_Q:
        return w
    a1 = w.a1_gscale.float().reshape(-1)
    if not bool((a1 == 1.0).all()):
        raise MoeWeightError(
            "mode Q assumes the M3.1 1.0 placeholder FC1 input scale, got "
            f"{a1.tolist()[:4]}"
        )
    if not torch.equal(w.g1_scale_c.float(), (w.g1_alphas.float() * w.fc2_enc).float()):
        raise MoeWeightError(
            "g1_scale_c != g1_alphas x fc2_enc: unknown FC2 scale fold"
        )
    base_g2 = w.g2_alphas.float() * w.fc2_enc  # exact: fc2_enc is a power of two
    return replace(
        w,
        g1_scale_c=(w.g1_alphas.float() * FC2_SCALE_Q).contiguous(),
        g2_alphas=(base_g2 / FC2_SCALE_Q).contiguous(),
        fc2_enc=FC2_SCALE_Q,
    )


def _linear_mxfp8(lin, what: str, shape: tuple[int, int]):
    """(e4m3 weight [N, K] contiguous, flat UE8M0 128x4-swizzled scales) of a vLLM MXFP8
    linear after weight
    processing (undoes the CuTe-DSL kernel's [K, N] view; see `_mxfp8`)."""
    try:
        return mxfp8_linear_nk(lin, what, shape=shape)
    except Mxfp8LayoutError as e:
        raise MoeWeightError(str(e)) from e


def from_vllm_moe(moe) -> MoeWeights:
    """MoeWeights from a loaded `MiniMaxM3MoE` (`decoder_layer.block_sparse_moe`) on
    the FLASHINFER_TRTLLM NVFP4
    backend. Raises `MoeWeightError` with the reason when the layer is in any other
    form."""
    ex = getattr(moe.experts, "routed_experts", moe.experts)
    qm = getattr(ex, "quant_method", None)
    kern = getattr(qm, "moe_kernel", None)
    fe = getattr(kern, "fused_experts", None)
    if fe is None or not hasattr(fe, "g1_scale_c"):
        raise MoeWeightError(
            "routed experts are not on the TRTLLM-Gen NVFP4 path (no "
            "fused_experts.g1_scale_c); "
            "serve with the FLASHINFER_TRTLLM NvFP4 MoE backend"
        )
    qc = fe.quant_config
    w13, w13s, w2, w2s = (
        getattr(ex, n)
        for n in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
    )
    if any(t.untyped_storage().nbytes() == 0 for t in (w13, w13s, w2, w2s)):
        raise MoeWeightError(
            "routed expert storage released (localized / freed weights are not "
            "supported)"
        )
    a1 = qc.a1_gscale.float().reshape(-1)
    if a1.numel() > 1 and not bool((a1 == a1[0]).all()):
        raise MoeWeightError("per-expert routed FC1 input scales differ")
    a2g = _pow2_scalar(qc.a2_gscale, "a2_gscale (FC2 input global scale)")
    g1a = qc.g1_alphas.float().contiguous()
    g1c = fe.g1_scale_c.float()
    if not torch.equal(g1c, (g1a * a2g).float()):
        raise MoeWeightError(
            "g1_scale_c != g1_alphas x a2_gscale; unknown FC2 scale fold"
        )
    beta = fe.gemm1_beta if fe.gemm1_beta is not None else torch.zeros_like(g1a)
    if fe.gemm1_alpha is None or fe.gemm1_clamp_limit is None:
        raise MoeWeightError(
            "SwiGLU-OAI alpha / clamp limit missing (not a swigluoai MoE?)"
        )
    gu_w, gu_s = _linear_mxfp8(
        moe.shared_experts.gate_up_proj, "shared gate_up_proj", (2 * INTER, HIDDEN)
    )
    dn_w, dn_s = _linear_mxfp8(
        moe.shared_experts.down_proj, "shared down_proj", (HIDDEN, INTER)
    )
    w = MoeWeights(
        w13=w13.view(torch.uint8),
        w13_scale=w13s.view(torch.float8_e4m3fn),
        w2=w2.view(torch.uint8),
        w2_scale=w2s.view(torch.float8_e4m3fn),
        g1_alphas=g1a,
        g1_scale_c=g1c.contiguous(),
        g2_alphas=qc.g2_alphas.float().contiguous(),
        act_alpha=fe.gemm1_alpha.float().contiguous(),
        act_beta=beta.float().contiguous(),
        act_clamp=fe.gemm1_clamp_limit.float().contiguous(),
        a1_gscale=a1[:1].contiguous(),
        router=moe.gate.weight.float().contiguous(),
        bias=moe.e_score_correction_bias.float().contiguous(),
        shared_gu_w=gu_w,
        shared_gu_s=gu_s,
        shared_dn_w=dn_w,
        shared_dn_s=dn_s,
        fc2_enc=a2g,
    )
    w.validate()
    return w


__all__ = [
    "FC1_QAT_MAX",
    "FC2_SCALE_Q",
    "MoeWeightError",
    "MoeWeights",
    "from_vllm_moe",
    "prep_mode_q",
]
