# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Aggregated loaded-model and device refusals."""

from __future__ import annotations

import torch

from vllm.platforms import current_platform

from .spec import DecodeConfig
from .weights import MonoLayerWeights


def model_refusals(model, config: DecodeConfig) -> list[str]:
    reasons: list[str] = []
    expected = list(range(config.first_layer, config.last_layer + 1))
    layers = [
        layer for layer in model.layers if getattr(layer, "layer_id", -1) in expected
    ]
    actual = [getattr(layer, "layer_id", -1) for layer in layers]
    if actual != expected:
        reasons.append(f"requires sparse layers {expected[0]} through {expected[-1]}")
    for layer in layers:
        layer_id = layer.layer_id
        if not getattr(layer, "is_moe_layer", False):
            reasons.append(f"layer {layer_id} is not an MoE layer")
        if not hasattr(getattr(layer, "self_attn", None), "indexer"):
            reasons.append(f"layer {layer_id} has no sparse indexer")
        try:
            MonoLayerWeights.from_layer(layer)
        except (AttributeError, ValueError) as err:
            reasons.append(str(err))
    if not current_platform.is_rocm():
        reasons.append("requires ROCm")
        return reasons
    try:
        import flydsl  # noqa: F401
        from aiter.ops.flydsl.kernels import buffer_ops  # noqa: F401

        from vllm.platforms.rocm import get_cdna_version
    except ImportError as err:
        reasons.append(f"requires FlyDSL and AITER FlyDSL helpers ({err})")
        return reasons
    if get_cdna_version() != 4:
        reasons.append("requires CDNA4/gfx950")
    try:
        cus = current_platform.num_compute_units(
            torch.accelerator.current_device_index()
        )
    except RuntimeError as err:
        reasons.append(f"cannot determine compute-unit count ({err})")
    else:
        if cus != config.blocks:
            reasons.append(f"requires {config.blocks} compute units, found {cus}")
    return reasons
