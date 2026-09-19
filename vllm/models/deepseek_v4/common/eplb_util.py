# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared EPLB MoE registration helpers for DeepSeek V4 family models."""

from collections.abc import Iterable
from typing import Protocol

import torch.nn as nn

from vllm.model_executor.models.utils import PPMissingLayer


class _Dsv4CollectableMoE(Protocol):
    num_expert_groups: int
    num_moe_layers: int
    moe_layers: list[nn.Module]
    moe_mlp_layers: list[nn.Module]

    def extract_moe_parameters(self, example_moe: object | None) -> None: ...


def collect_moe_layers(
    moe_model: _Dsv4CollectableMoE,
    layers: Iterable[nn.Module],
    config: object,
    *,
    decoder_layer_type: type[nn.Module],
    moe_type: type[nn.Module],
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
        if not isinstance(layer, decoder_layer_type):
            continue
        if not isinstance(layer.ffn, moe_type):
            continue
        example_moe = layer.ffn
        moe_model.moe_mlp_layers.append(layer.ffn)
        moe_model.moe_layers.append(layer.ffn.experts)

    moe_model.num_moe_layers = len(moe_model.moe_layers)
    moe_model.extract_moe_parameters(example_moe)
