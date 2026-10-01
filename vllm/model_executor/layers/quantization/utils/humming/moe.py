# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configure, prepare weights for, and assemble Humming MoE kernels."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    FusedMoEQuantDesc,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
)

if TYPE_CHECKING:
    from vllm.utils.humming import LayerConfig

logger = init_logger(__name__)


@dataclass(kw_only=True)
class HummingMoEQuantConfig(FusedMoEQuantConfig):
    w1_humming_config: "LayerConfig"
    w2_humming_config: "LayerConfig"
    w1_hadamard_block_size: int = 1
    w2_hadamard_block_size: int = 1
    w1_input_scale: torch.Tensor | None = None
    w2_input_scale: torch.Tensor | None = None
    w1_input_scale_2: torch.Tensor | None = None
    w2_input_scale_2: torch.Tensor | None = None


def make_humming_moe_quant_config(
    quant_dtype: torch.dtype | str | None,
    weight_dtype: torch.dtype | str | None,
    weight_group_shape: GroupShape | None = None,
    activation_group_shape: GroupShape | None = None,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    w1_zp: torch.Tensor | None = None,
    w2_zp: torch.Tensor | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    w1_gscale: torch.Tensor | None = None,
    w2_gscale: torch.Tensor | None = None,
    gemm1_alpha: float | None = None,
    gemm1_beta: float | None = None,
    gemm1_clamp_limit: float | None = None,
    humming_configs: dict[str, "LayerConfig"] | None = None,
) -> HummingMoEQuantConfig:
    assert humming_configs is not None
    if quant_dtype is None:
        a_quant_desc = FusedMoEQuantDesc(dtype=None)
    elif activation_group_shape is not None:
        # Pre-dispatch quantization.
        a_quant_desc = FusedMoEQuantDesc(
            dtype=quant_dtype, shape=activation_group_shape
        )
    else:
        # Deferred path: Humming quantizes the activation internally, so the
        # descriptor only needs a non-None dtype to mark it as quantized.
        shape = GroupShape(row=1, col=-1)
        a_quant_desc = FusedMoEQuantDesc(dtype=quant_dtype, shape=shape)

    w1_quant_desc = FusedMoEQuantDesc(
        dtype=weight_dtype,
        shape=weight_group_shape,
        scale=w1_scale,
        alpha_or_gscale=w1_gscale,
        zp=w1_zp,
        bias=w1_bias,
    )

    w2_quant_desc = FusedMoEQuantDesc(
        dtype=weight_dtype,
        shape=weight_group_shape,
        scale=w2_scale,
        alpha_or_gscale=w2_gscale,
        zp=w2_zp,
        bias=w2_bias,
    )

    return HummingMoEQuantConfig(
        _a1=a_quant_desc,
        _a2=a_quant_desc,
        _w1=w1_quant_desc,
        _w2=w2_quant_desc,
        gemm1_alpha=gemm1_alpha,
        gemm1_beta=gemm1_beta,
        gemm1_clamp_limit=gemm1_clamp_limit,
        w1_humming_config=humming_configs["w13"],
        w2_humming_config=humming_configs["w2"],
    )
