# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Refusals that need the loaded model or the device, every reason at once."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.platforms import current_platform

from .spec import GRID, is_dsa, is_kda, kernel_layers
from .weights import experts_backend_reason

if TYPE_CHECKING:
    from torch import nn


def model_refusals(backbone: nn.Module) -> list[str]:
    """Why the loaded layers or this GPU cannot run the kernels (empty: they can)."""
    reasons = []
    for idx in kernel_layers(backbone.config):
        layer = backbone.layers[idx]
        why = experts_backend_reason(layer)
        if why is not None:
            reasons.append(f"layer {idx}: {why}")
        if not getattr(layer, "mhc", False) or getattr(layer, "is_mtp_layer", False):
            reasons.append(f"layer {idx}: needs an mHC decoder layer")
        if getattr(layer, "is_sequence_parallel", False):
            reasons.append(f"layer {idx}: sequence-parallel layer")
        if not hasattr(layer.self_attn, "o_proj"):
            reasons.append(f"layer {idx}: needs the attention's o_proj")
        if is_kda(layer) and getattr(layer.self_attn, "num_spec", 0):
            reasons.append(f"layer {idx}: KDA with speculative decoding")
        if is_dsa(layer) and getattr(layer.self_attn, "indexer", None) is None:
            reasons.append(f"layer {idx}: sparse attention without its indexer")
    if not current_platform.is_rocm():
        return reasons + ["needs ROCm"]
    try:
        import flydsl  # noqa: F401
        from aiter.ops.flydsl.kernels import buffer_ops  # noqa: F401
    except ImportError as err:
        reasons.append(f"needs FlyDSL and AITER's FlyDSL helpers ({err})")
    cus = current_platform.num_compute_units(torch.accelerator.current_device_index())
    if cus < GRID:
        # every CTA of a launch must stay resident (a partitioned GPU has fewer)
        reasons.append(f"needs {GRID} compute units, the GPU has {cus}")
    return reasons
