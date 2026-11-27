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


class TritonConv1dDecodeKernel(Conv1dDecodeKernel):
    backend = MambaBackendEnum.TRITON

    @classmethod
    def supports_platform(cls, platform: PlatformEnum) -> bool:
        return platform != PlatformEnum.CPU

    @classmethod
    def supports_spec_decode(cls) -> bool:
        return True

    @staticmethod
    @cache
    def _fn():
        from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
            causal_conv1d_update,
        )

        return causal_conv1d_update

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
        return cls._fn()(
            x,
            conv_state,
            weight,
            bias,
            activation,
            conv_state_indices=metadata.state_indices_tensor_d,
            num_accepted_tokens=metadata.num_accepted_tokens,
            query_start_loc=metadata.query_start_loc_d,
            max_query_len=max_query_len,
        )
