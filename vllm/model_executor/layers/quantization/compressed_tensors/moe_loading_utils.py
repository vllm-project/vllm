# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight-loading helper for compressed-tensors per-expert MoE checkpoints.

compressed-tensors / LLM-Compressor MoE checkpoints keep experts separate and
per-projection (``experts.experts.N.{gate,up,down}_proj.*``), while FusedMoE
stores them stacked (``w13_*`` = gate+up, ``w2_*`` = down). This routes the
per-expert keys at the matching stacked param and shard, and hands the tensor
to the layer's own expert weight loader, which owns the TP/EP sharding rules.
"""

import torch

from vllm.model_executor.model_loader.weight_utils import (
    maybe_remap_moe_expert_param_name,
)

# Trailing checkpoint component -> (stacked param suffix, FusedMoE shard id).
# The model's `hf_to_vllm_mapper` has already renamed the HF projection names
# (gate_proj/up_proj/down_proj) to w1/w3/w2 by the time we see them.
_PER_EXPERT_PARAMS = {
    "w1_weight": ("w13_weight", "w1"),
    "w1_weight_scale": ("w13_weight_scale", "w1"),
    "w1_bias": ("w13_bias", "w1"),
    "w3_weight": ("w13_weight", "w3"),
    "w3_weight_scale": ("w13_weight_scale", "w3"),
    "w3_bias": ("w13_bias", "w3"),
    "w2_weight": ("w2_weight", "w2"),
    "w2_weight_scale": ("w2_weight_scale", "w2"),
    "w2_bias": ("w2_bias", "w2"),
}


def load_per_expert_moe_weight(
    name: str,
    weight: torch.Tensor,
    params_dict: dict[str, torch.nn.Parameter],
    loaded_params: set[str],
    *,
    tp_rank: int,
) -> bool:
    """Route a per-expert MoE checkpoint key into its stacked parameter.

    Returns True iff the name belongs to the per-expert layout, in which case
    the caller has nothing left to do for it.
    """
    if ".mlp.experts.experts." not in name:
        return False
    ids = [int(part) for part in name.split(".") if part.isdigit()]
    if len(ids) != 2:
        return False
    layer_id, expert_id = ids

    entry = _PER_EXPERT_PARAMS.get(name.rsplit(".", 1)[-1])
    if entry is None:
        return False
    param_suffix, shard_id = entry

    param_name = maybe_remap_moe_expert_param_name(
        f"layers.{layer_id}.mlp.experts.{param_suffix}", params_dict
    )
    param = params_dict.get(param_name)
    if param is None:
        # The layer never allocated this param, e.g. a biasless checkpoint.
        return True

    # down_proj bias is replicated across TP ranks but its output is summed by
    # the all-reduce, so only rank 0 may contribute it.
    if param_suffix == "w2_bias" and tp_rank != 0:
        weight = torch.zeros_like(weight)

    param.weight_loader(param, weight, param_name, shard_id, expert_id)
    loaded_params.add(param_name)
    return True
