# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from functools import cache
from typing import TYPE_CHECKING

import torch

from vllm.config.mamba import MambaBackendEnum
from vllm.platforms.interface import PlatformEnum
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

from .base import MambaSSUKernel, SSULayer

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mamba_attn import BaseMambaAttentionMetadata


class CPUSSUKernel(MambaSSUKernel):
    backend = MambaBackendEnum.CPU

    @classmethod
    def supports_platform(cls, platform: PlatformEnum) -> bool:
        return platform == PlatformEnum.CPU

    @classmethod
    def supports_spec_decode(cls) -> bool:
        return True

    @classmethod
    def supports_pipeline_parallel(cls) -> bool:
        return True

    @staticmethod
    @cache
    def _fn():
        from vllm import _cpu_ops

        return _cpu_ops.selective_state_update_cpu

    @classmethod
    def forward(
        cls,
        layer: SSULayer,
        metadata: "BaseMambaAttentionMetadata",
        state: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        dt_bias: torch.Tensor,
        out: torch.Tensor,
        z: torch.Tensor | None = None,
    ) -> None:
        cls._fn()(
            state,
            x,
            dt,
            A,
            B,
            C,
            D,
            z,
            dt_bias,
            True,
            metadata.state_indices_tensor_d,
            metadata.state_indices_tensor_d,
            NULL_BLOCK_ID,
            out,
            metadata.num_accepted_tokens,
            metadata.query_start_loc_d,
        )
