# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deprecated environment variables redirected to `--quantization-config`."""

import json

import vllm.envs as envs
from vllm.config.quantization import QuantizationConfigArgs, QuantSpec
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Static

logger = init_logger(__name__)

REMOVAL_VERSION = "v0.35.0"


def _warn(env_names: str, config: dict) -> None:
    logger.warning_once(
        "%s are deprecated and will be removed in vLLM %s. Use "
        "--quantization-config '%s' instead.",
        env_names,
        REMOVAL_VERSION,
        json.dumps(config, separators=(",", ":")),
    )


def fp8_moe_requant_args() -> QuantizationConfigArgs | None:
    """Online quantization args equivalent to `VLLM_FP8_MOE_REQUANT[_ACT]`.

    Only meaningful for block-FP8 checkpoints on ROCm, which the caller checks.
    """
    if envs.VLLM_FP8_MOE_REQUANT != "mxfp4":
        return None
    act = envs.VLLM_FP8_MOE_REQUANT_ACT
    # a4w4 is the default activation of online MXFP4. a8w4 and a16w4 request the
    # backend's bf16-activation kernels (`activation: null`): the interleaved
    # FlyDSL kernels of `AITER_MXFP4_BF16` quantize activations to fp8 per token.
    if act == "a4w4":
        config: dict = {"moe": "mxfp4"}
        spec = QuantSpec(weight=kMxfp4Static)
    else:
        config = {"moe": {"weight": "mxfp4", "activation": None}}
        spec = QuantSpec(weight=kMxfp4Static, activation=None)
    _warn("VLLM_FP8_MOE_REQUANT and VLLM_FP8_MOE_REQUANT_ACT", config)
    if act == "a16w4":
        logger.warning_once(
            "VLLM_FP8_MOE_REQUANT_ACT=a16w4 also needs "
            "--moe-backend aiter_triton_mxfp4_bf16 (it was selected implicitly)."
        )
    return QuantizationConfigArgs(moe=spec)


def muse_moe_online_quant_args(num_hidden_layers: int) -> QuantizationConfigArgs | None:
    """Online quantization args equivalent to
    `VLLM_ROCM_MUSE_MOE_ONLINE_QUANT=mxfp4`: the first and last decoder layers
    stay in BF16."""
    if envs.VLLM_ROCM_MUSE_MOE_ONLINE_QUANT != "mxfp4":
        return None
    ignore = ["*layers.0.*", f"*layers.{num_hidden_layers - 1}.*"]
    _warn("VLLM_ROCM_MUSE_MOE_ONLINE_QUANT", {"moe": "mxfp4", "ignore": ignore})
    return QuantizationConfigArgs(moe=QuantSpec(weight=kMxfp4Static), ignore=ignore)
