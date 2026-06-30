# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.platforms import current_platform

from ..inc_linear import INCLinearMethod
from .inc_scheme import INCScheme

if TYPE_CHECKING:
    import torch

    from ..config_parser import INCLayerConfig
    from ..inc import INCConfig

logger = init_logger(__name__)


class INCMxfp4Scheme(INCScheme):
    """MXFP4 (W4A4) scheme for AutoRound checkpoints.

    Both linear and MoE are supported on XPU only. Linear reuses the shared
    MXFP4 linear kernel; MoE reuses ``CompressedTensorsW4A4Mxfp4MoEMethod``,
    whose registered parameter names (``w13_weight_packed`` / ``w2_weight_packed``
    / ``w13_weight_scale`` / ``w2_weight_scale``) match the names produced by
    AutoRound's ``auto_round:llm_compressor`` MXFP4 export once the per-expert
    ``gate_proj`` / ``up_proj`` / ``down_proj`` ``weight_packed`` / ``weight_scale``
    tensors are folded by ``make_expert_params_mapping``.
    """

    @staticmethod
    def can_handle(layer_config: "INCLayerConfig") -> bool:
        return layer_config.is_mxfp4

    def get_linear_method(
        self,
        config: "INCConfig",
        layer: "torch.nn.Module",
        prefix: str,
        layer_config: "INCLayerConfig",
    ):
        del config, layer, prefix
        if not current_platform.is_xpu():
            raise NotImplementedError(
                f"INC MXFP4: linear only supported on XPU, config {layer_config}"
            )
        from .inc_mxfp4_linear import INCMxfp4LinearMethod

        return INCLinearMethod(INCMxfp4LinearMethod(layer_config))

    def get_moe_method(
        self,
        config: "INCConfig",
        layer: "torch.nn.Module",
        prefix: str,
        layer_config: "INCLayerConfig",
    ):
        del config, prefix
        if not current_platform.is_xpu():
            raise NotImplementedError(
                f"INC MXFP4: MoE only supported on XPU, config {layer_config}"
            )
        from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_mxfp4 import (  # noqa: E501
            CompressedTensorsW4A4Mxfp4MoEMethod,
        )

        return CompressedTensorsW4A4Mxfp4MoEMethod(layer.moe_config)
