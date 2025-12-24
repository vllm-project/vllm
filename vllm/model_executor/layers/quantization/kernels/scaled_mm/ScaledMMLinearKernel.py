# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class ScaledMMLinearLayerConfig:
    is_channelwise: bool
    is_static_input_scheme: bool
    input_symmetric: bool


class ScaledMMLinearKernel(ABC):
    @classmethod
    @abstractmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def can_implement(cls, c: ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        raise NotImplementedError

    def __init__(
        self,
        c: ScaledMMLinearLayerConfig,
        w_q_param_name: str,
        w_s_param_name: str,
        i_s_param_name: str,
        i_zp_param_name: str,
        azp_adj_param_name: str,
    ) -> None:
        assert self.can_implement(c)
        assert self.is_supported()
        self.config = c
        self.w_q_name = w_q_param_name
        self.w_s_name = w_s_param_name
        self.i_s_name = i_s_param_name
        self.i_zp_name = i_zp_param_name
        self.azp_adj_name = azp_adj_param_name

    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        raise NotImplementedError

    @abstractmethod
    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def _get_weight_params(
        self, layer: torch.nn.Module
    ) -> tuple[
        torch.Tensor,  # weight
        torch.Tensor,  # weight_scale
        torch.Tensor | None,  # input_scale,
        torch.Tensor | None,  # input_zp
        torch.Tensor | None,  # azp_adj
    ]:
        return (
            getattr(layer, self.w_q_name),
            getattr(layer, self.w_s_name),
            getattr(layer, self.i_s_name),
            getattr(layer, self.i_zp_name),
            getattr(layer, self.azp_adj_name),
        )
