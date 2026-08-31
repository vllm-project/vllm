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

    Dispatches to :class:`INCMxfp4LinearMethod` for linear layers and
    :class:`INCMxfp4MoEMethod` for fused MoE layers; see those classes for the
    per-module weight layout and kernel-selection details.
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
        del layer, prefix
        from .inc_mxfp4_linear import INCMxfp4LinearMethod

        if (
            config.rotation_config is not None
            and not current_platform.is_cuda()
            and not current_platform.is_xpu()
        ):
            raise NotImplementedError(
                "AutoRound Hadamard rotation requires CUDA Hadacore or XPU ARK"
            )
        rotation_block_size = (
            config.rotation_config["block_size"]
            if config.rotation_config is not None
            else None
        )
        return INCLinearMethod(INCMxfp4LinearMethod(layer_config, rotation_block_size))

    def get_moe_method(
        self,
        config: "INCConfig",
        layer: "torch.nn.Module",
        prefix: str,
        layer_config: "INCLayerConfig",
    ):
        del prefix, layer_config
        if config.rotation_config is not None:
            raise NotImplementedError(
                "AutoRound Hadamard rotation is not supported for fused MoE"
            )
        from .inc_mxfp4_moe import INCMxfp4MoEMethod

        return INCMxfp4MoEMethod(layer.moe_config)
