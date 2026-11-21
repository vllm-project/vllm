# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from functools import cache
from typing import TYPE_CHECKING, cast

import torch

from vllm.config.mamba import MambaBackendEnum
from vllm.platforms.interface import PlatformEnum

from .base import SSULayer
from .replayssm_base import ReplaySSMLayer
from .replayssm_tracker import TrackerReplaySSMKernel

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mamba_attn import BaseMambaAttentionMetadata


class TritonReplaySSMSpecKernel(TrackerReplaySSMKernel):
    """Speculative decoding: every decode row, from one token up to the full
    verify window, through `cu_seqlens`."""

    backend = MambaBackendEnum.TRITON
    name = "Triton ReplaySSM speculative decoding"

    @classmethod
    def supports_platform(cls, platform: PlatformEnum) -> bool:
        return platform != PlatformEnum.CPU

    @classmethod
    def supports_spec_decode(cls) -> bool:
        return True

    @classmethod
    def requires_spec_decode(cls) -> bool:
        return True

    @staticmethod
    @cache
    def _fn():
        from vllm.model_executor.layers.mamba.ops.selective_state_update_replayssm_spec import (  # noqa: E501
            selective_state_update_replayssm_spec,
        )

        return selective_state_update_replayssm_spec

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
        layer = cast(ReplaySSMLayer, layer)
        assert metadata.replayssm_state_indices_d is not None
        assert metadata.query_start_loc_d is not None
        x_cache, dt_cache, B_cache = layer.kv_cache[2:5]
        assert x_cache.size(2) == cls.ring_len(
            layer.ssu_state.buffer_len, layer.ssu_state.num_spec
        )
        cls._fn()(
            state,
            x_cache,
            B_cache,
            dt_cache,
            layer.ssu_state.ring_start,
            layer.ssu_state.prev_num_accepted,
            x,
            dt,
            A,
            B,
            C,
            out,
            cu_seqlens=metadata.query_start_loc_d,
            max_seqlen=1 + layer.ssu_state.num_spec,
            state_batch_indices=metadata.replayssm_state_indices_d,
            D=D,
            dt_bias=dt_bias,
            dt_softplus=True,
        )
