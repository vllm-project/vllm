# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

from vllm.config.watermarking import WatermarkConfig
from vllm.v1.watermarking.gumbel import (
    DualKeyGumbelWatermarker,
    GumbelWatermarker,
)
from vllm.v1.watermarking.watermarker import Watermarker


def _create_gumbel(config: WatermarkConfig, _is_drafting: bool) -> Watermarker:
    return GumbelWatermarker(config.key, config.context_width, config.prf)


def _create_dual_key_gumbel(config: WatermarkConfig, is_drafting: bool) -> Watermarker:
    return DualKeyGumbelWatermarker(
        config.key,
        config.context_width,
        config.prf,
        is_drafting=is_drafting,
    )


_WATERMARKERS: dict[str, Callable[[WatermarkConfig, bool], Watermarker]] = {
    "gumbel": _create_gumbel,
    "dual_key_gumbel": _create_dual_key_gumbel,
}


def create_watermarker(
    config: WatermarkConfig, *, is_drafting: bool = False
) -> Watermarker:
    watermarker_factory = _WATERMARKERS.get(config.algorithm)
    if watermarker_factory is None:
        raise ValueError(f"Unknown watermarking algorithm: {config.algorithm}")
    return watermarker_factory(config, is_drafting)
