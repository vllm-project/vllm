# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP4 SiTU MoE on the optional `moonmath_amd` CDNA3 kernels.

AITER's A16W4 grouped GEMM has no in-kernel SiTU, so a SiTU MXFP4 MoE
(Kimi-K3) has no native ROCm kernel without this one, which fuses the
activation into the gate/up epilogue.

`MoonmathW4A16SituExperts` subclasses the AITER W4A16 experts to inherit its
weight format -- the kernels read the layout `oracle/mxfp4.py` already prepares
for AITER's Triton path -- and claims ONLY the SITU activation, and only when
the optional package is installed. The oracle lists it after the AITER class,
which does not claim SiTU, so every other configuration stays on AITER.
"""

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.experts.aiter_mxfp4_w4a16_moe import (
    AiterW4A16ExpertsMonolithic,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import weight_mx_scale
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
from vllm.triton_utils import tl, triton
from vllm.utils.import_utils import has_moonmath_amd

logger = init_logger(__name__)

__all__ = ["MoonmathW4A16SituExperts"]

_REDUCE_BLOCK_H = 512


@triton.jit
def _weighted_topk_sum_kernel(
    down_ptr,
    weight_ptr,
    out_ptr,
    H,
    TOPK: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offs < H
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for k in tl.static_range(TOPK):
        w = tl.load(weight_ptr + t * TOPK + k).to(tl.float32)
        v = tl.load(down_ptr + (t * TOPK + k) * H + offs, mask=mask, other=0.0)
        acc += v.to(tl.float32) * w
    tl.store(out_ptr + t * H + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


class MoonmathW4A16SituExperts(AiterW4A16ExpertsMonolithic):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)
        # DeepSeekV3 (noaux_tc, e.g. Kimi-K3) divides by the selected sum.
        self.renormalize = self.renormalize or (
            moe_config.routing_method == RoutingMethodType.DeepSeekV3
        )
        self._mm = None

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        # Only SiTU, and only with the kernels: anything else must fall through
        # to the AITER class rather than be claimed and then fail at load.
        if activation != MoEActivation.SITU:
            return False
        if not has_moonmath_amd():
            # The generic "does not support SITU" reason would not tell anyone
            # that a kernel exists for it.
            logger.info_once(
                "SiTU MXFP4 MoE: no ROCm kernel is active. Install the optional "
                "`moonmath-amd` package to enable the fused-SiTU A16W4 kernels."
            )
            return False
        return True

    @staticmethod
    def _supports_routing_method(
        routing_method: RoutingMethodType,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return routing_method == RoutingMethodType.DeepSeekV3 or (
            AiterW4A16ExpertsMonolithic._supports_routing_method(
                routing_method, weight_key, activation_key
            )
        )

    def process_weights_after_loading(self, layer) -> None:
        # aiter's swizzle leaves [E, K/2, N] stride(-2)==1; transpose back.
        raw = [
            t.storage.data.view(torch.uint8).transpose(-2, -1)
            for t in (
                layer.w13_weight,
                weight_mx_scale(self.quant_config.w1_precision),
                layer.w2_weight,
                weight_mx_scale(self.quant_config.w2_precision),
            )
        ]
        self._mm = repack_weights(*raw)

    def apply(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        num_expert_group: int | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        routed_scaling_factor: float | None = None,
        topk_group: int | None = None,
    ) -> torch.Tensor:
        # The weights are in moonmath layout, so AITER must never see them.
        assert expert_map is None, (
            "moonmath MXFP4 MoE serves SiTU without expert parallelism only"
        )
        # score_mode None is plain softmax top-k, which IGNORES the router bias.
        score_mode = {
            RoutingMethodType.DeepseekV4: "sqrtsoftplus",
            RoutingMethodType.DeepSeekV3: "sigmoid",
        }.get(self.moe_config.routing_method)
        return situ_moe_forward(
            self._mm,
            hidden_states,
            router_logits,
            self.topk,
            self.renormalize,
            score_mode,
            e_score_correction_bias,
            routed_scaling_factor,
            self.moe_config.activation_situ_beta,
            self.moe_config.activation_situ_linear_beta,
            apply_router_weight_on_input,
        )


def repack_weights(w13, w13s, w2, w2s) -> tuple:
    """Rewrite the four MXFP4 weight tensors in place; return the run plan.

    Must run at weight load: the kernels read moonmath's layout directly and
    stock weights give wrong numbers rather than an error.
    """
    import moonmath_amd as ma

    E, N13, _ = w13.shape
    inter_dim, H = N13 // 2, w2.shape[1]
    return (
        _repack_(w13, ma.repack_mxfp4, (E, H // 32, N13, 16), True),
        _repack_(w13s, ma.repack_mxfp4_scales, (E, H // 32, N13), True),
        _repack_(w2, ma.repack_mxfp4, (E, inter_dim // 32, H, 16)),
        _repack_(w2s, ma.repack_mxfp4_scales, (E, inter_dim // 32, H)),
        E,
        inter_dim,
        H,
        torch.cuda.get_device_properties(w13.device).multi_processor_count,
    )


def _repack_(stock: torch.Tensor, fn, out_shape, deinterleave=False, chunk=32):
    """Rewrite `stock` storage into moonmath's layout IN PLACE, returning a view.

    Out of place would double the MoE weights (~170 GB/rank for K3). Both
    repacks permute within an expert slab, so chunking over E is exact.
    """
    E = stock.shape[0]
    # view(), not reshape(): reshape would silently copy and lose the aliasing.
    flat = stock.view(-1)
    per = flat.numel() // E
    for s in range(0, E, chunk):
        e = min(s + chunk, E)
        src = stock[s:e]
        if deinterleave:
            # The oracle interleaves w13 as (g0,u0,g1,u1,...) for AITER, but
            # the fused SiTU epilogue pairs the [gate | up] HALVES. Backwards
            # here multiplies mismatched columns: garbage, never an error.
            v = src.view(e - s, src.shape[1] // 2, 2, src.shape[2])
            src = torch.cat((v[:, :, 0], v[:, :, 1]), dim=1)
        flat[s * per : e * per].copy_(fn(src).reshape(-1))
    return flat.view(*out_shape)


def situ_moe_forward(
    mm: tuple,
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    score_mode: str | None,
    e_score_correction_bias: torch.Tensor | None,
    routed_scaling_factor: float | None,
    situ_beta: float,
    situ_linear_beta: float,
    apply_router_weight_on_input: bool,
) -> torch.Tensor:
    """One MXFP4 SiTU MoE layer: routing, then the two grouped GEMMs."""
    import moonmath_amd as ma
    from aiter.ops.triton.moe.moe_routing.topk import topk as aiter_topk

    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    w13, w13s, w2, w2s, E, inter_dim, H, num_cus = mm
    # AITER routing()'s math, with its sort replaced by moe_align_block_size.
    scal, indx, _ = aiter_topk(
        gating_output,
        topk,
        apply_softmax=False,
        score_mode=score_mode,
        bias=e_score_correction_bias,
        renorm=renormalize,
        routed_scaling_factor=(
            routed_scaling_factor if routed_scaling_factor is not None else 1.0
        ),
    )
    ids = indx.to(torch.int32)
    T = hidden_states.shape[0]
    rows = T * topk
    dt, dev = hidden_states.dtype, hidden_states.device
    assert hidden_states.is_contiguous(), (
        "moonmath MoE reads A by row; a strided [T, K] reads the wrong ones"
    )
    tw = scal.reshape(-1).to(dt) if apply_router_weight_on_input else None

    bm_g = ma.mxfp4_moe_gateup_block_m(rows, E)
    sg, eg, ng = moe_align_block_size(ids, bm_g, E)
    inter = torch.empty((rows, inter_dim), dtype=dt, device=dev)
    ma.mxfp4_moe_gateup(
        hidden_states,
        w13,
        w13s,
        inter,
        tw,
        sg,
        eg,
        ng,
        eg.numel(),
        bm_g,
        rows,
        topk,
        epilogue=ma.EPI_SITU,
        mul_routed_weight=apply_router_weight_on_input,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
    )

    bm_d = ma.mxfp4_moe_down_block_m(rows, E, inter_dim)
    assert bm_d, f"moonmath has no down tile for K={inter_dim}"
    sd, ed, nd = (sg, eg, ng) if bm_d == bm_g else moe_align_block_size(ids, bm_d, E)
    nt = ma.mxfp4_moe_down_nt(rows, E)
    ns = ma.mxfp4_moe_down_n_steps(ed.numel(), H, num_cus, bm_d, nt)
    down = torch.empty((rows, H), dtype=dt, device=dev)
    # top_k=1 on purpose: down's A already has one row per (token, expert).
    ma.mxfp4_moe_down(
        inter,
        w2,
        w2s,
        down,
        tw,
        sd,
        ed,
        nd,
        ed.numel(),
        bm_d,
        ns,
        nt,
        rows,
        1,
        mul_routed_weight=False,
    )
    # down writes one row per (token, expert); AITER scattered in-kernel.
    if apply_router_weight_on_input:
        return torch.sum(down.view(T, topk, H), dim=1, dtype=torch.float32).to(dt)
    out = torch.empty((T, H), dtype=dt, device=dev)
    _weighted_topk_sum_kernel[(T, triton.cdiv(H, _REDUCE_BLOCK_H))](
        down,
        scal.contiguous(),
        out,
        H,
        TOPK=topk,
        BLOCK_H=_REDUCE_BLOCK_H,
        num_warps=4,
    )
    return out
