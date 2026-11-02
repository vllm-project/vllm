# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config.watermarking import WatermarkConfig
from vllm.v1.watermarking.gumbel import (
    DualKeyGumbelWatermarker,
    GumbelWatermarker,
)
from vllm.v1.watermarking.sbw import SBWWatermarker
from vllm.v1.watermarking.watermarker import Watermarker


def create_watermarker(config: WatermarkConfig) -> Watermarker:
    # context_width is set to a scheme-specific default by validate_watermark_settings
    # before this function is called; assert here to satisfy the type checker.
    assert config.context_width is not None
    if config.algorithm == "gumbel":
        return GumbelWatermarker(config.key, config.context_width, config.prf)
    if config.algorithm == "dual_key_gumbel":
        return DualKeyGumbelWatermarker(
            config.key,
            config.context_width,
            config.prf,
            config.alpha,
        )
    if config.algorithm == "sbw":
        return SBWWatermarker(
            key=config.key,
            context_width=config.context_width,
            scheme=config.sbw_scheme,
            gamma=config.sbw_gamma,
            delta=config.sbw_delta,
        )
    raise ValueError(f"Unknown watermarking algorithm: {config.algorithm}")
