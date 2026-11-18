# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
from typing import TYPE_CHECKING

import torch

from vllm.config.mamba import MambaBackendEnum
from vllm.platforms.interface import PlatformEnum

from .base import MambaSSDKernel

if TYPE_CHECKING:
    from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
    from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadata


class TritonSSDKernel(MambaSSDKernel):
    backend = MambaBackendEnum.TRITON

    @classmethod
    def supports_platform(cls, platform: PlatformEnum) -> bool:
        return platform != PlatformEnum.CPU

    @staticmethod
    @functools.cache
    def _fn():
        from vllm.model_executor.layers.mamba.ops.ssd_combined import (
            mamba_chunk_scan_combined_varlen,
        )

        return mamba_chunk_scan_combined_varlen

    @classmethod
    def _scan(
        cls,
        layer: "MambaMixer2",
        x: torch.Tensor,
        dt: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        out: torch.Tensor,
        ssm_state: torch.Tensor,
        metadata: "Mamba2AttentionMetadata",
        initial_states: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        # With checkpoints the kernel returns every chunk's state; otherwise it
        # writes the final states itself.
        exports = metadata.checkpoint_chunk_idx is not None
        states = cls._fn()(
            x,
            dt,
            layer.A,
            B,
            C,
            metadata.chunk_size,
            cu_seqlens=metadata.query_start_loc_p,
            cu_chunk_seqlens=metadata.cu_chunk_seqlen_p,
            last_chunk_indices=metadata.last_chunk_indices_p,
            seq_idx=metadata.seq_idx_p,
            out=out,
            D=layer.D,
            dt_bias=layer.dt_bias,
            initial_states=initial_states,
            dt_softplus=True,
            return_intermediate_states=exports,
            state_dtype=ssm_state.dtype,
            final_states_out=None if exports else ssm_state,
            final_state_indices=None if exports else metadata.state_indices_tensor_p,
        )
        if not exports:
            return None, None
        return states[metadata.last_chunk_indices_p], states
