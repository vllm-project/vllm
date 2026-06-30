# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from vllm.logger import init_logger

from ..inc_linear import INCLinearMethod
from .inc_scheme import INCScheme

if TYPE_CHECKING:
    import torch

    from ..config_parser import INCLayerConfig
    from ..inc import INCConfig

logger = init_logger(__name__)


class INCMxfp4Scheme(INCScheme):
    """MXFP4 (W4A4) scheme for AutoRound checkpoints.

    Linear uses :class:`INCMxfp4LinearMethod`, which selects the MXFP4 GEMM
    kernel for the current platform via ``init_mxfp4_linear_kernel``
    (FlashInfer / Marlin on CUDA, ``fp4_gemm`` on XPU). MoE uses
    :class:`INCMxfp4MoEMethod`, which registers the ``auto_round:llm_compressor``
    MXFP4 layout (``w13_weight_packed`` / ``w2_weight_packed`` /
    ``w13_weight_scale`` / ``w2_weight_scale``) and dispatches the fused MoE to
    the best backend for the device (CUTLASS / Marlin / XPU). The per-expert
    ``gate_proj`` / ``up_proj`` / ``down_proj`` tensors are folded into those
    stacked parameters by ``make_expert_params_mapping``.
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
        from .inc_mxfp4_linear import INCMxfp4LinearMethod

        return INCLinearMethod(INCMxfp4LinearMethod(layer_config))

    def get_moe_method(
        self,
        config: "INCConfig",
        layer: "torch.nn.Module",
        prefix: str,
        layer_config: "INCLayerConfig",
    ):
        del config, prefix, layer_config
        from ..inc_moe import INCMxfp4MoEMethod

        return INCMxfp4MoEMethod(layer.moe_config)
