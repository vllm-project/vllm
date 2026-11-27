# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from functools import cache
from typing import TYPE_CHECKING

import torch

from vllm.config.mamba import MambaBackendEnum
from vllm.platforms.interface import PlatformEnum

from .base import Conv1dPrefillKernel

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mamba_attn import BaseMambaAttentionMetadata


class CPUConv1dPrefillKernel(Conv1dPrefillKernel):
    backend = MambaBackendEnum.CPU

    @classmethod
    def supports_platform(cls, platform: PlatformEnum) -> bool:
        return platform == PlatformEnum.CPU

    @staticmethod
    @cache
    def _fn():
        from vllm.model_executor.layers.mamba.ops.cpu.causal_conv1d import (
            causal_conv1d_fn_cpu,
        )

        return causal_conv1d_fn_cpu

    @classmethod
    def forward(
        cls,
        x: torch.Tensor,
        conv_states: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        activation: str | None,
        metadata: "BaseMambaAttentionMetadata",
    ) -> torch.Tensor:
        assert metadata.query_start_loc_p is not None
        return cls._fn()(
            x,
            weight,
            bias,
            conv_states=conv_states,
            query_start_loc=metadata.query_start_loc_p,
            cache_indices=metadata.state_indices_tensor_p,
            has_initial_state=metadata.has_initial_states_p,
            activation=activation,
        )
