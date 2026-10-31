# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config.watermarking import WatermarkConfig
from vllm.v1.watermarking.chi_squared import ChiSquaredWatermarker
from vllm.v1.watermarking.gumbel import (
    DualKeyGumbelWatermarker,
    GumbelWatermarker,
)
from vllm.v1.watermarking.perplexity import PPLHardWatermarker, PPLSoftWatermarker
from vllm.v1.watermarking.watermarker import Watermarker


def create_watermarker(config: WatermarkConfig) -> Watermarker:
    if config.algorithm == "gumbel":
        return GumbelWatermarker(config.key, config.context_width, config.prf)
    if config.algorithm == "dual_key_gumbel":
        return DualKeyGumbelWatermarker(
            config.key,
            config.context_width,
            config.prf,
            config.alpha,
        )
    if config.algorithm == "chi_squared":
        delta = 1.0 if config.delta is None else config.delta
        return ChiSquaredWatermarker(
            config.key, config.context_width, config.prf, delta
        )
    if config.algorithm == "ppl_hard":
        return PPLHardWatermarker(
            config.key, config.context_width, config.prf, config.epsilon
        )
    if config.algorithm == "ppl_soft":
        return PPLSoftWatermarker(
            config.key, config.context_width, config.prf, config.epsilon
        )
    raise ValueError(f"Unknown watermarking algorithm: {config.algorithm}")
