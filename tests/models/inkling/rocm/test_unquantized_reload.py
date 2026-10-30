# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unquantized AITER experts are padded and declare ``weight_loader_numel``;
the method zeroes the padding. The AMD Inkling loader must leave that padding
alone, or layerwise reload counts it and processes the layer before every
expert arrived."""

import torch

from tests.models.inkling.reload_utils import (
    HIDDEN,
    N_ROUTED,
    Checkpoint,
    InklingMoELayer,
    assert_reload_matches_cold_start,
    sink_checkpoint,
)
from vllm.models.inkling.amd.moe import InklingMoE

# AITER rounds an intermediate dim of at most 192 up to a multiple of 64
INTERMEDIATE, PADDED = 96, 128


def _checkpoint(seed: int) -> Checkpoint:
    gen = torch.Generator().manual_seed(seed)
    return {
        "experts.w13_weight": torch.randn(
            N_ROUTED, 2 * INTERMEDIATE, HIDDEN, generator=gen
        ).bfloat16(),
        "experts.w2_weight": torch.randn(
            N_ROUTED, HIDDEN, INTERMEDIATE, generator=gen
        ).bfloat16(),
        **sink_checkpoint(gen, INTERMEDIATE),
    }


def _layer() -> InklingMoELayer:
    return InklingMoELayer(InklingMoE, INTERMEDIATE, None, "aiter")


def test_unquantized_aiter_reload_matches_cold_start(dist_init):
    routed = _layer().moe.experts.routed_experts
    assert routed.w13_weight.shape[1] == 2 * PADDED
    assert routed.w13_weight.weight_loader_numel < routed.w13_weight.numel()

    assert_reload_matches_cold_start(_layer, _checkpoint)
