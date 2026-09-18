# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared EPLB MoE registration helpers for DeepSeek V4 family models."""

from collections.abc import Iterable

import torch.nn as nn

from vllm.model_executor.models.interfaces import MixtureOfExperts
from vllm.model_executor.models.utils import PPMissingLayer


def collect_moe_layers(
    moe_model: MixtureOfExperts,
    layers: Iterable[nn.Module],
    config: object,
    *,
    skip_pp_missing: bool = False,
) -> None:
    """Populate ``MixtureOfExperts`` fields from decoder layers for EPLB.

    Works for both target backbones and DSpark draft stacks. Draft models pass
    their shorter ``model.layers`` list; target models enable
    ``skip_pp_missing`` to ignore pipeline-parallel placeholder layers.
    """
    moe_model.num_expert_groups = getattr(config, "n_group", 1)
    moe_model.moe_layers = []
    moe_model.moe_mlp_layers = []
    example_moe = None
    for layer in layers:
        if skip_pp_missing and isinstance(layer, PPMissingLayer):
            continue
        ffn = getattr(layer, "ffn", None)
        if ffn is None:
            continue
        experts = getattr(ffn, "experts", None)
        if experts is None:
            continue
        example_moe = ffn
        moe_model.moe_mlp_layers.append(ffn)
        moe_model.moe_layers.append(experts)

    moe_model.num_moe_layers = len(moe_model.moe_layers)
    moe_model.extract_moe_parameters(example_moe)
