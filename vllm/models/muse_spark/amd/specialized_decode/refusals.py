# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""What the gfx942 specialized decode refuses: the deployment and the modules
(at startup, every reason) and a step (every forward, rank-invariant data
only). Free of FlyDSL, so a missing FlyDSL is a reason, not an import error."""

from collections.abc import Sequence

import torch
from torch import nn

from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.model_executor.kernels.linear import (
    AiterHipbMMPerTokenFp8ScaledMMLinearKernel,
    AiterPreshuffledPerTokenFp8ScaledMMLinearKernel,
)
from vllm.models.muse_spark.common.layers import (
    MuseSpark1xNormPreservingResidualConnection,
)

from ..mono.decoder import _routed_experts, config_refusals
from .wide.layout import (
    BLOCKS,
    E_LOCAL,
    FIRST_LAYER,
    HIDDEN,
    INTER,
    LAST_LAYER,
    LAYERS,
    POOLED,
    Q_ROWS,
    QKVG,
    TP,
)

NUM_CUS = BLOCKS
FP8 = torch.float8_e4m3fnuz
# dense FP8 kernels that keep the weight in AITER's (16, 16) shuffled layout
_SHUFFLING_KERNELS = (
    AiterHipbMMPerTokenFp8ScaledMMLinearKernel,
    AiterPreshuffledPerTokenFp8ScaledMMLinearKernel,
)


# --------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------


def deployment_refusals(lm: nn.Module, tp_rank: int, tp_size: int) -> list[str]:
    """Every reason this deployment cannot run the kernels; empty = supported."""
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return ["not ROCm"]
    from vllm.platforms.rocm import get_cdna_version

    reasons = [] if get_cdna_version() == 3 else ["not gfx942"]
    cus = torch.cuda.get_device_properties(
        torch.accelerator.current_device_index()
    ).multi_processor_count
    if cus != NUM_CUS:
        reasons.append(f"{cus} CUs (kernels: {NUM_CUS})")
    vc = lm.vllm_config
    reasons += config_refusals(vc.model_config.hf_text_config)
    parallel = vc.parallel_config
    if tp_size != TP:
        reasons.append(f"tensor_parallel_size={tp_size} (kernels: {TP})")
    for name in (
        "pipeline_parallel_size",
        "data_parallel_size",
        "decode_context_parallel_size",
        "prefill_context_parallel_size",
    ):
        if getattr(parallel, name, 1) != 1:
            reasons.append(f"{name}={getattr(parallel, name)}")
    if not parallel.enable_expert_parallel:
        reasons.append("expert parallelism off (needs --enable-expert-parallel)")
    if parallel.enable_eplb:
        reasons.append("EPLB on")
    if parallel.expert_placement_strategy != "linear":
        reasons.append(f"expert placement {parallel.expert_placement_strategy}")
    experts = _routed_experts(lm.model.layers[FIRST_LAYER])
    moe_parallel = getattr(
        getattr(experts, "moe_config", None), "moe_parallel_config", None
    )
    if moe_parallel is None or not moe_parallel.use_ep:
        reasons.append("MoE layers not expert parallel")
    elif moe_parallel.ep_size != tp_size or moe_parallel.ep_rank != tp_rank:
        reasons.append(
            f"EP rank {moe_parallel.ep_rank}/{moe_parallel.ep_size} "
            f"!= TP rank {tp_rank}/{tp_size}"
        )
    if getattr(parallel, "use_sequence_parallel_moe", False):
        reasons.append("sequence-parallel MoE on")
    if vc.speculative_config is not None:
        reasons.append("speculative decoding on (the kernels serve plain decode)")
    if vc.lora_config is not None:
        reasons.append("LoRA on")
    if vc.model_config.dtype != torch.bfloat16:
        reasons.append(f"model dtype {vc.model_config.dtype}")
    return reasons


def _nk(w: torch.Tensor, n: int, k: int) -> str | None:
    """Why ``w`` is not an ``[n, k]`` row-major FP8 matrix (or its transposed
    view, as torch's scaled-mm methods keep it)."""
    if w.dtype != FP8:
        return f"{w.dtype}"
    if tuple(w.shape) == (n, k) and w.is_contiguous():
        return None
    if tuple(w.shape) == (k, n) and w.stride() == (1, k):
        return None
    return f"{tuple(w.shape)} strides {w.stride()}"


def _per_channel(s: torch.Tensor | None, n: int) -> bool:
    return (
        s is not None
        and s.dtype == torch.float32
        and s.numel() == n
        and s.is_contiguous()
    )


def _dense_refusal(lin: nn.Module, n: int, k: int, name: str) -> str | None:
    kernel = getattr(getattr(lin, "scheme", None), "fp8_linear", None)
    if isinstance(kernel, _SHUFFLING_KERNELS) or getattr(
        lin, "aiter_bpreshuffled", False
    ):
        return f"{name}: weight pre-shuffled by {type(kernel).__name__}"
    why = _nk(lin.weight, n, k)
    if why is not None:
        return f"{name}: weight {why} (kernels: FP8 [{n}, {k}])"
    if not _per_channel(getattr(lin, "weight_scale", None), n):
        return f"{name}: weight scale not per-channel f32 [{n}]"
    if getattr(lin, "input_scale", None) is not None:
        return f"{name}: static input scale (kernels: dynamic per token)"
    return None


def layer_refusals(model: nn.Module) -> list[str]:
    """Module and tensor guards for layers FIRST_LAYER .. LAST_LAYER + 1."""
    reasons: list[str] = []
    if getattr(model, "residual_dtype", None) != torch.float32:
        reasons.append("residual dtype not fp32")
    if not getattr(model, "fuse_across_layers", False):
        reasons.append("layer fusion off")
    layers = model.layers
    if len(layers) != LAYERS:
        return reasons + [f"{len(layers)} layers"]
    for i in range(FIRST_LAYER, LAST_LAYER + 2):
        layer, attn = layers[i], layers[i].self_attn
        bad = [
            _dense_refusal(attn.qkv_proj, QKVG, HIDDEN, "qkv_proj"),
            _dense_refusal(attn.o_proj, HIDDEN, Q_ROWS, "o_proj"),
        ]
        if not (attn.use_o_gate and attn.fuse_o_norm_gate):
            bad.append("o-path without the fused o-norm gate")
        if i > LAST_LAYER:
            reasons += [f"layer {i}: {b}" for b in bad if b]
            continue
        moe = layer.feed_forward
        experts = _routed_experts(layer)
        for conn, name in ((layer.add_attn, "add_attn"), (layer.add_ffn, "add_ffn")):
            if not isinstance(conn, MuseSpark1xNormPreservingResidualConnection):
                bad.append(f"{name} not a gated residual")
            elif not all(
                isinstance(v := getattr(conn, a, None), torch.Tensor)
                and v.numel() == HIDDEN
                for a in ("alpha", "beta")
            ):
                bad.append(f"{name}: alpha/beta not materialized per channel")
        if layer.post_attn_norm.has_weight:
            bad.append("post_attn_norm has a weight (kernels: weightless)")
        if not (
            layer.post_ffn_norm.has_weight
            and layer.post_ffn_norm.applies_weight_before_norm
        ):
            bad.append("post_ffn_norm not input-scaled")
        if moe.pooler is None or moe.unpooler is None or moe.pre_expert_norm is None:
            bad.append("MoE without pooler / unpooler / pre-expert norm")
        if getattr(moe, "router_output_multiplier", None) is not None and not getattr(
            moe, "_router_multiplier_folded", False
        ):
            bad.append("router output multiplier not folded")
        if moe.router.weight.dtype != torch.float32:
            bad.append(f"router {moe.router.weight.dtype} (kernels: fp32)")
        if experts is None:
            bad.append("no routed experts")
        else:
            for nm, rows, k in (("w13", 2 * INTER, POOLED), ("w2", POOLED, INTER)):
                w = getattr(experts, f"{nm}_weight", None)
                s = getattr(experts, f"{nm}_weight_scale", None)
                if (
                    w is None
                    or w.dtype != FP8
                    or w.dim() != 3
                    or tuple(w.shape[:2]) != (E_LOCAL, rows)
                    or w.shape[2] != k
                    or w.stride(2) != 1
                    or w.stride(0) != rows * w.stride(1)
                ):
                    bad.append(f"experts {nm}: {None if w is None else tuple(w.shape)}")
                elif getattr(w, "is_shuffled", False):
                    bad.append(f"experts {nm}: weight shuffled (kernels: row-major)")
                elif (
                    s is None or s.dtype != torch.float32 or s.numel() != E_LOCAL * rows
                ):
                    bad.append(f"experts {nm} scale not per-channel f32")
        reasons += [f"layer {i}: {b}" for b in bad if b]
    return reasons


def step_refusal(
    rows: int, widths: Sequence[int], names: Sequence[str], aux_layers
) -> str | None:
    """Why this step runs stock, from rank-invariant data only; None = kernels."""
    if rows not in widths:
        return f"rows:{rows}"
    if aux_layers:
        return "aux-hidden-states"
    ctx = get_forward_context()
    if ctx.cudagraph_runtime_mode not in (CUDAGraphMode.NONE, CUDAGraphMode.FULL):
        return f"cudagraph:{ctx.cudagraph_runtime_mode.name}"
    metadata = ctx.attn_metadata
    if not isinstance(metadata, dict) or not metadata:
        return "attn-metadata"
    seen: set[int] = set()
    for name in names:
        md = metadata.get(name)
        if md is None:
            return "attn-metadata"
        if id(md) in seen:
            continue
        seen.add(id(md))
        if getattr(md, "num_prefills", None) != 0 or getattr(
            md, "num_extends", None
        ) not in (0, None):
            return "not-decode"
        if getattr(md, "num_decodes", None) != rows:
            return f"width:{rows}/{getattr(md, 'num_decodes', None)}"
    return None


__all__ = ["deployment_refusals", "layer_refusals", "step_refusal"]
