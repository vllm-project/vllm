# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hopper (SM90a) NVFP4 MoE experts for Muse Spark."""

from vllm.models.muse_spark.nvidia.ops.hopper_nvfp4_moe.kernels import (
    MuseSparkHopperNvFp4Experts,
    build_hopper_nvfp4_moe,
    encode_hopper_nvfp4_weights,
    is_hopper_nvfp4_moe_supported,
    try_process_hopper_nvfp4_weights,
)

__all__ = [
    "MuseSparkHopperNvFp4Experts",
    "build_hopper_nvfp4_moe",
    "encode_hopper_nvfp4_weights",
    "is_hopper_nvfp4_moe_supported",
    "try_process_hopper_nvfp4_weights",
]
