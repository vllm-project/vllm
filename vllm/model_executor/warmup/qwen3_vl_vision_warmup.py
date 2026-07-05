# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up Qwen3-VL vision Triton kernels."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def _iter_qwen3_vl_visual_modules(
    model: torch.nn.Module,
) -> Iterable[torch.nn.Module]:
    modules = getattr(model, "modules", None)
    if modules is None:
        return
    for module in modules():
        if module.__class__.__name__ == "Qwen3_VisionTransformer":
            yield module


def _warmup_qwen3_vl_visual_module(visual: torch.nn.Module) -> None:
    patch_embed = visual.patch_embed
    in_channels = patch_embed.proj.in_channels
    patch_size = patch_embed.patch_size
    temporal_patch_size = patch_embed.temporal_patch_size
    spatial_merge_size = visual.spatial_merge_size

    # The Triton compile keys for Qwen3-VL pos-embed interpolation and vision
    # rotary do not depend on the real image size, so a minimal image grid is
    # enough to move the first compile out of inference.
    grid_thw = [[1, spatial_merge_size, spatial_merge_size]]
    total_patches = spatial_merge_size * spatial_merge_size
    flattened_patch_size = in_channels * temporal_patch_size * patch_size * patch_size
    pixel_values = torch.empty(
        (total_patches, flattened_patch_size),
        device=visual.device,
        dtype=visual.dtype,
    )
    metadata = visual.prepare_encoder_metadata(grid_thw)
    visual(pixel_values, grid_thw, encoder_metadata=metadata)


def qwen3_vl_vision_warmup(model: torch.nn.Module) -> None:
    """Warm Qwen3-VL vision kernels missed by text-only dummy runs."""
    warmed_keys: set[tuple[Any, ...]] = set()
    warmed_count = 0

    with torch.inference_mode():
        for visual in _iter_qwen3_vl_visual_modules(model):
            warmup_key = (
                str(visual.device),
                visual.dtype,
                visual.hidden_size,
                visual.num_heads,
                visual.patch_embed.patch_size,
                visual.patch_embed.temporal_patch_size,
                visual.spatial_merge_size,
            )
            if warmup_key in warmed_keys:
                continue
            warmed_keys.add(warmup_key)
            try:
                _warmup_qwen3_vl_visual_module(visual)
                warmed_count += 1
            except Exception:
                logger.warning(
                    "Qwen3-VL vision JIT warmup failed. "
                    "First image inference may JIT compile.",
                    exc_info=True,
                )

    if warmed_count:
        logger.info(
            "Warmed up %d Qwen3-VL vision JIT kernel variant(s).",
            warmed_count,
        )
