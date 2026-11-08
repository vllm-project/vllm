# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from vllm import envs
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
        backend = envs.VLLM_XPU_INC_MXFP4_BACKEND
        if backend != "auto":
            if not current_platform.is_xpu():
                raise NotImplementedError("ARK MXFP4 MoE requires Intel CRI XPU")
            from .inc_ark_mxfp4_moe import INCARKMxfp4MoEMethod

            return INCARKMxfp4MoEMethod(layer.moe_config, backend)
        from .inc_mxfp4_moe import INCMxfp4MoEMethod

        return INCMxfp4MoEMethod(layer.moe_config)
