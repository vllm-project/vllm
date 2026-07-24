# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from ..inc_linear import INCLinearMethod
from .inc_scheme import INCScheme

if TYPE_CHECKING:
    import torch

    from ..config_parser import INCLayerConfig
    from ..inc import INCConfig


class INCMxfp8Scheme(INCScheme):
    @staticmethod
    def can_handle(layer_config: "INCLayerConfig") -> bool:
        return layer_config.is_mxfp8

    def get_linear_method(
        self,
        config: "INCConfig",
        layer: "torch.nn.Module",
        prefix: str,
        layer_config: "INCLayerConfig",
    ):
        del config, layer, prefix, layer_config
        from .inc_mxfp8_linear import INCMxfp8LinearScheme

        return INCLinearMethod(INCMxfp8LinearScheme())
