# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from ..base import LinearKernel, MMLinearLayerConfig


@dataclass(slots=True)
class Int8LinearLayerConfig(MMLinearLayerConfig):
    # TODO: Change to QuantKey like Fp8LinearLayerConfig
    is_static_input_scheme: bool
    is_channelwise: bool
    input_symmetric: bool
    # More than one logical output partition (e.g. QKV). Per-tensor weight
    # scales of such layers hold one scale per partition and are expanded to
    # per-channel scales after loading.
    is_fused_module: bool = False


_Int8ParamsT = tuple[
    torch.Tensor,  # weight
    torch.Tensor,  # weight_scale
    torch.Tensor | None,  # input_scale,
    torch.Tensor | None,  # input_zp
    torch.Tensor | None,  # azp_adj
]


class Int8LinearKernel(LinearKernel[Int8LinearLayerConfig], ABC):
    def __init__(
        self, c: Int8LinearLayerConfig, layer_param_names: Sequence[str]
    ) -> None:
        super().__init__(c)
        self.layer_param_names = layer_param_names

    def _get_layer_params(self, layer) -> _Int8ParamsT:
        w_q, w_s, i_s, i_zp, azp_adj = self.layer_param_names
        return (
            getattr(layer, w_q),
            getattr(layer, w_s),
            getattr(layer, i_s, None),
            # A loaded zero point of a symmetric scheme stays registered, unused.
            None if self.config.input_symmetric else getattr(layer, i_zp, None),
            getattr(layer, azp_adj, None),
        )
