# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

from vllm.config.watermarking import WatermarkConfig
from vllm.v1.watermarking.gumbel import GumbelWatermarker
from vllm.v1.watermarking.watermarker import Watermarker


def _create_gumbel(config: WatermarkConfig) -> Watermarker:
    return GumbelWatermarker(config.key, config.context_width, config.prf)


_WATERMARKERS: dict[str, Callable[[WatermarkConfig], Watermarker]] = {
    "gumbel": _create_gumbel,
}


def create_watermarker(config: WatermarkConfig) -> Watermarker:
    watermarker_factory = _WATERMARKERS.get(config.algorithm)
    if watermarker_factory is None:
        raise ValueError(f"Unknown watermarking algorithm: {config.algorithm}")
    return watermarker_factory(config)
