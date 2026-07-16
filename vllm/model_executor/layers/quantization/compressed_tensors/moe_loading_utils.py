# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight-loading helpers for compressed-tensors per-expert MoE checkpoints.

compressed-tensors / LLM-Compressor MoE checkpoints store experts per-expert
and per-projection (``experts.experts.N.{gate,up,down}_proj.*``). vLLM's
FusedMoE stores them stacked (``w13_*`` = gate+up, ``w2_*`` = down). This helper
routes the per-expert keys into the stacked params with EP/TP slicing, so model
loaders don't have to inline the logic.
"""

import torch

from vllm.model_executor.model_loader.weight_utils import (
    maybe_remap_moe_expert_param_name,
)


def load_per_expert_moe_weight(
    in_name: str,
    in_weight: torch.Tensor,
    *,
    params_dict: dict,
    loaded_params: set,
    use_ep: bool,
    ep_rank_start: int,
    ep_rank_end: int,
    tp_rank: int,
    tp_rank_start: int,
    tp_rank_end: int,
    per_rank_intermediate_size: int,
) -> bool:
    """Route compressed-tensors per-expert keys
    (experts.experts.N.{gate,up,down}_proj.*) into the stacked w13_*/w2_*
    params, honoring EP/TP. Returns True iff the name matched a per-expert
    pattern (caller should `continue`)."""
    if ".mlp.experts.experts." not in in_name:
        return False
    parts = in_name.split(".")
    digits = [int(p) for p in parts if p.isdigit()]
    if len(digits) != 2:
        return False
    layer_id, expert_id = digits

    # EP gating: skip experts owned by other ranks; remap to local id.
    if use_ep:
        if not (ep_rank_start <= expert_id < ep_rank_end):
            return True
        local_expert_id = expert_id - ep_rank_start
    else:
        local_expert_id = expert_id

    # Use the actual TP slice size (shorter than per_rank_intermediate_size
    # only on the last rank when intermediate_size % tp_size != 0).
    local_intermediate = tp_rank_end - tp_rank_start

    # int for the w13 halves (gate/up), None for w2 (down, no dim-1 slice).
    dim1_start: int | None
    dim1_end: int | None
    suffix = "." + parts[-1]
    if suffix in (".w1_weight", ".w1_weight_scale", ".w1_bias"):
        fused_suffix = {
            ".w1_weight": "w13_weight",
            ".w1_weight_scale": "w13_weight_scale",
            ".w1_bias": "w13_bias",
        }[suffix]
        dim1_start, dim1_end = 0, local_intermediate
        if not use_ep:
            in_weight = in_weight[tp_rank_start:tp_rank_end, ...]
    elif suffix in (".w3_weight", ".w3_weight_scale", ".w3_bias"):
        fused_suffix = {
            ".w3_weight": "w13_weight",
            ".w3_weight_scale": "w13_weight_scale",
            ".w3_bias": "w13_bias",
        }[suffix]
        dim1_start, dim1_end = (
            per_rank_intermediate_size,
            per_rank_intermediate_size + local_intermediate,
        )
        if not use_ep:
            in_weight = in_weight[tp_rank_start:tp_rank_end, ...]
    elif suffix in (".w2_weight", ".w2_weight_scale", ".w2_bias"):
        fused_suffix = {
            ".w2_weight": "w2_weight",
            ".w2_weight_scale": "w2_weight_scale",
            ".w2_bias": "w2_bias",
        }[suffix]
        dim1_start = dim1_end = None
        # w2_weight: TP-slice the INPUT (intermediate) dim.
        if not use_ep and suffix == ".w2_weight":
            in_weight = in_weight[:, tp_rank_start:tp_rank_end]
        # w2_bias: replicated across ranks in TP; only rank 0 contributes
        # to the post-all-reduce sum (mirrors the stacked-tensor loader's
        # "if tp_rank != 0: weight.zero_()").
        if not use_ep and suffix == ".w2_bias" and tp_rank != 0:
            in_weight = torch.zeros_like(in_weight)
    else:
        return False

    fused_name = f"layers.{layer_id}.mlp.experts.{fused_suffix}"
    fused_name = maybe_remap_moe_expert_param_name(fused_name, params_dict)
    if fused_name not in params_dict:
        # Model didn't allocate this param (e.g. has_bias=False).
        return True

    param = params_dict[fused_name]
    target_slot = param.data[local_expert_id]

    if dim1_start is None:
        if target_slot.shape != in_weight.shape:
            target_slot.copy_(in_weight.view(target_slot.shape))
        else:
            target_slot.copy_(in_weight)
    else:
        dst = target_slot[dim1_start:dim1_end]
        if dst.shape != in_weight.shape:
            dst.copy_(in_weight.view(dst.shape))
        else:
            dst.copy_(in_weight)
    loaded_params.add(fused_name)
    return True
