# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import replace
from functools import cache
from typing import TYPE_CHECKING

import torch

from vllm.config.mamba import MambaBackendEnum
from vllm.platforms.interface import PlatformEnum
from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

from .base import Conv1dPrefillKernel, M

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mamba_attn import BaseMambaAttentionMetadata


class TritonConv1dPrefillKernel(Conv1dPrefillKernel):
    backend = MambaBackendEnum.TRITON

    @classmethod
    def supports_platform(cls, platform: PlatformEnum) -> bool:
        return platform != PlatformEnum.CPU

    @classmethod
    def prepare_metadata(cls, metadata: M, query_start_loc_p_cpu: torch.Tensor) -> M:
        assert metadata.query_start_loc_p is not None
        nums_dict, batch_ptr, token_chunk_offset_ptr = compute_causal_conv1d_metadata(
            query_start_loc_p_cpu, device=metadata.query_start_loc_p.device
        )
        return replace(
            metadata,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )

    @staticmethod
    @cache
    def _fn():
        from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
            causal_conv1d_fn,
        )

        return causal_conv1d_fn

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
            metadata=metadata,
        )
