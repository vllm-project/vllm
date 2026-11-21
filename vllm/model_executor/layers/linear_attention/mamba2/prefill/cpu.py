# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
from typing import TYPE_CHECKING

import torch

from vllm.config.mamba import MambaBackendEnum
from vllm.platforms.interface import PlatformEnum

from .base import MambaSSDKernel, SSDLayerParams, SSDWarmupKey

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadata


class CPUSSDKernel(MambaSSDKernel):
    backend = MambaBackendEnum.CPU

    @classmethod
    def _register_warmups(cls, key: SSDWarmupKey) -> None:
        """Nothing to compile: the CPU kernel launches no Triton."""

    @classmethod
    def supports_platform(cls, platform: PlatformEnum) -> bool:
        return platform == PlatformEnum.CPU

    @staticmethod
    @functools.cache
    def _fn():
        from vllm.model_executor.layers.mamba.ops.cpu.mamba_ssm import (
            _mamba_chunk_scan_combined_fwd_cpu,
        )

        return _mamba_chunk_scan_combined_fwd_cpu

    @classmethod
    def gather_states(
        cls,
        ssm_state: torch.Tensor,
        indices: torch.Tensor,
        has_initial_states: torch.Tensor,
    ) -> torch.Tensor:
        return torch.where(
            has_initial_states[:, None, None, None], ssm_state[indices], 0
        )

    @classmethod
    def write_final_states(
        cls,
        ssm_state: torch.Tensor,
        final_states: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        ssm_state[indices] = final_states

    @classmethod
    def _scan(
        cls,
        layer: SSDLayerParams,
        x: torch.Tensor,
        dt: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        out: torch.Tensor,
        ssm_state: torch.Tensor,
        metadata: "Mamba2AttentionMetadata",
        initial_states: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        assert metadata.checkpoint_chunk_idx is None, (
            "The CPU SSD kernel cannot export prefill checkpoints."
        )
        final_states = cls._fn()(
            x,
            dt,
            layer.A,
            B,
            C,
            metadata.chunk_size,
            out,
            D=layer.D,
            dt_bias=layer.dt_bias,
            initial_states=initial_states,
            cu_seqlens=metadata.query_start_loc_p,
            dt_softplus=True,
            state_dtype=ssm_state.dtype,
        )
        return final_states, None
