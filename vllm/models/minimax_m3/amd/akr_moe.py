# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The amd_kernels `moe` route: the whole MiniMax-M3.1 MoE block on the package of
its per-rank intermediate size (`amd_kernels.moe.mxfp4_h6144_i768_e128_top4` at TP4,
`mxfp4_h6144_i1536_e128_top4` at TP2), then the stock TP all-reduce.

At TP2 every rank holds the whole shared expert (`replicate_shared`): the
sequence-parallel prefill runs it on the rank's own rows only, and the TP path
(decode) computes this rank's TP shard of it from a view of the full copy."""

import os

import torch

from vllm.logger import init_logger
from vllm.models.minimax_m3.amd import akr_config
from vllm.models.minimax_m3.amd.mono.envs import envs as mono_envs

logger = init_logger(__name__)


# per-rank intermediate size -> package
_PKGS = {768: "mxfp4_h6144_i768_e128_top4", 1536: "mxfp4_h6144_i1536_e128_top4"}


def _package(i: int):
    """The package of per-rank I `i`; another I (or an amd_kernels without it) gets
    the TP4 package, whose `unsupported_reason` names the shape."""
    return akr_config.package(f"moe.{_PKGS.get(i, _PKGS[768])}", f"moe.{_PKGS[768]}")


def package(moe: torch.nn.Module):
    """The package of `moe`'s per-rank I (routed w2 [E, H, I / 2]), kept on the
    block."""
    p = getattr(moe, "_akr_pkg", None)
    if p is None:
        i = moe.experts.routed_experts.w2_weight.shape[-1] * 2
        p = moe._akr_pkg = _package(i)
    return p


def replicate_shared(config) -> bool:
    """Whether the MoE blocks load the full shared expert on every rank: the moe route
    at TP2 with plain TP and a package that takes it; `AKR_SHARED_FULL=0` keeps the
    TP-sharded one."""
    from vllm.config import get_current_vllm_config_or_none
    from vllm.distributed import get_tensor_model_parallel_world_size

    if os.environ.get("AKR_SHARED_FULL", "1") == "0" or mono_envs.MONO_ENABLE:
        return False
    tp = get_tensor_model_parallel_world_size()
    if tp != 2 or not akr_config.on("moe"):
        return False
    cfg = get_current_vllm_config_or_none()
    if cfg is not None:
        pc = cfg.parallel_config
        if (
            pc.enable_expert_parallel
            or pc.data_parallel_size > 1
            or pc.prefill_context_parallel_size > 1
            or cfg.kernel_config.monokernel
        ):
            return False
    P = _package(config.intermediate_size // tp)
    return bool(getattr(P, "SHARED_FULL", False))


def setup(moe: torch.nn.Module, config, prefix: str) -> bool:
    """Whether `moe` (a `MiniMaxM3MoE`) runs on the package; if so its weights
    keep the checkpoint layout until `build_weights`."""
    if not akr_config.on("moe"):
        return False
    reason = unsupported_reason(moe, config)
    if reason is not None:
        logger.warning("amd_kernels moe off for %s: %s", prefix, reason)
        return False
    se = moe.shared_experts
    for layer in (moe.experts.routed_experts, se.gate_up_proj, se.down_proj):
        layer.quant_method.process_weights_after_loading = lambda layer: None
    logger.info_once("amd_kernels moe: MoE blocks run on amd_kernels mxfp4_moe")
    if moe.akr_shared_full:
        nbytes = sum(
            p.numel() * p.element_size()
            for p in (
                se.gate_up_proj.weight,
                se.gate_up_proj.weight_scale,
                se.down_proj.weight,
                se.down_proj.weight_scale,
            )
        )
        logger.info_once(
            "amd_kernels moe: the full shared expert on every rank (%.1f MB per layer)",
            nbytes / 1e6,
        )
    return True


def unsupported_reason(moe: torch.nn.Module, config) -> str | None:
    if mono_envs.MONO_ENABLE:
        return "the mono decode path (VLLM_ROCM_MINIMAX_M3_MONO) owns the MoE"
    if moe.is_fused_shared_expert_enabled or moe.shared_experts is None:
        return (
            "needs the unfused shared expert "
            "(VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0)"
        )
    if moe.e_score_correction_bias is None:
        return "needs sigmoid routing with the correction bias"
    rx = moe.experts.routed_experts
    mc = rx.moe_config
    if (
        mc.ep_size > 1
        or mc.dp_size > 1
        or mc.is_sequence_parallel
        or getattr(mc, "pcp_size", 1) > 1
    ):
        return "needs plain TP (no EP, DP, SP or PCP)"
    if mc.experts_per_token != 4:
        return "needs top-4"
    if type(rx.quant_method).__name__ != "QuarkOCP_MX_MoEMethod":
        return f"needs Quark MXFP4 routed experts: {type(rx.quant_method).__name__}"
    se = moe.shared_experts
    return package(moe).unsupported_reason(
        rx.w13_weight,
        rx.w13_weight_scale,
        rx.w2_weight,
        rx.w2_weight_scale,
        se.gate_up_proj.weight,
        se.gate_up_proj.weight_scale,
        se.down_proj.weight,
        se.down_proj.weight_scale,
        swiglu=(mc.swiglu_alpha, mc.swiglu_limit, mc.swiglu_beta),
    )


def build_weights(moe: torch.nn.Module):
    """The package's `MoeWeights`; the checkpoint-layout parameters are released."""
    P = package(moe)
    # Inside DBO microbatches the shared expert stays on the caller's stream.
    from vllm.v1.worker.ubatching import dbo_enabled

    P.set_dbo_query(dbo_enabled)
    rx, se = moe.experts.routed_experts, moe.shared_experts

    def u8(p):
        return p.data.view(torch.uint8)

    shared_args = (
        u8(se.gate_up_proj.weight),
        u8(se.gate_up_proj.weight_scale),
        u8(se.down_proj.weight),
        u8(se.down_proj.weight_scale),
    )
    if moe.akr_shared_full:
        from vllm.distributed import get_tp_group

        shared = P.SharedWeights.from_row_major(
            *shared_args, tp_rank=get_tp_group().rank_in_group
        )
    else:
        shared = P.SharedWeights.from_row_major(*shared_args)
    w = P.MoeWeights.from_row_major(
        moe.gate.weight.data,
        moe.e_score_correction_bias.data,
        u8(rx.w13_weight),
        u8(rx.w13_weight_scale),
        u8(rx.w2_weight),
        u8(rx.w2_weight_scale),
        shared=shared,
        routed_scaling_factor=moe.routed_scaling_factor,
    )
    for layer, names in (
        (rx, ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")),
        (se.gate_up_proj, ("weight", "weight_scale")),
        (se.down_proj, ("weight", "weight_scale")),
    ):
        for n in names:
            p = getattr(layer, n)
            p.data = torch.empty(0, dtype=p.dtype, device=p.device)
    if getattr(moe.gate, "_bf16x3_weight", None) is not None:
        moe.gate._bf16x3_weight = None  # GateLinear's own router split, unused
    torch.accelerator.empty_cache()
    P.warm_up_mxfp4_moe(w)
    return w


def forward(moe: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """The block's output [M, H] bf16 after the stock TP all-reduce."""
    from vllm.distributed import tensor_model_parallel_all_reduce

    if x.shape[0] == 0:
        return torch.empty_like(x)
    out = package(moe).mxfp4_moe(x.contiguous(), moe._akr_weights)
    mc = moe.experts.routed_experts.moe_config
    if mc.tp_size > 1 and not mc.skip_final_all_reduce:
        out = tensor_model_parallel_all_reduce(out)
    return out
