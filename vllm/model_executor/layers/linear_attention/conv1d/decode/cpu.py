# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from functools import cache
from typing import TYPE_CHECKING

import torch

from vllm.config.mamba import MambaBackendEnum
from vllm.platforms.interface import PlatformEnum

from .base import Conv1dDecodeKernel

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mamba_attn import BaseMambaAttentionMetadata


class CPUConv1dDecodeKernel(Conv1dDecodeKernel):
    backend = MambaBackendEnum.CPU

    @classmethod
    def supports_platform(cls, platform: PlatformEnum) -> bool:
        return platform == PlatformEnum.CPU

    @staticmethod
    @cache
    def _fn():
        from vllm.model_executor.layers.mamba.ops.cpu.causal_conv1d import (
            causal_conv1d_update_cpu,
        )

        return causal_conv1d_update_cpu

    @classmethod
    def forward(
        cls,
        x: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        activation: str | None,
        metadata: "BaseMambaAttentionMetadata",
        *,
        max_query_len: int,
    ) -> torch.Tensor:
        assert max_query_len == 1, "speculative decoding not supported"
        return cls._fn()(
            x,
            conv_state,
            weight,
            bias,
            activation,
            conv_state_indices=metadata.state_indices_tensor_d,
            query_start_loc=metadata.query_start_loc_d,
        )
