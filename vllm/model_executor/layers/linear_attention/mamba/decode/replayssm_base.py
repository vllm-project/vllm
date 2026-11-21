# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ReplaySSM decode kernels (`--use-replayssm`): each slot caches its recent SSM
inputs in rings (`layer.kv_cache[2:5]`: x, dt and B) and writes its full state
only when its ring fills."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Protocol

import torch

from vllm.model_executor.layers.linear_attention.kernel import LinearAttnSelectorConfig
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

from .base import MambaSSUKernel, SSULayer, SSULayerState

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass
class ReplaySSMLayerState(SSULayerState):
    buffer_len: int
    num_spec: int
    # Per-slot ring position and committed token count, shared by a cache
    # group's layers; bound by `initialize_replayssm_ring_trackers`.
    ring_start: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.int32)
    )
    prev_num_accepted: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.int32)
    )
    # Triton standard decode plans each step in one layer per cache group.
    plans_decode: bool = True
    # FlashInfer's stochastic-rounding seed slot; the group's first layer also
    # holds the whole buffer and refreshes it once per step.
    rand_seed: torch.Tensor | None = None
    rand_seeds: torch.Tensor | None = None

    def release(self) -> None:
        self.ring_start = torch.empty(0, dtype=torch.int32)
        self.prev_num_accepted = torch.empty(0, dtype=torch.int32)


class ReplaySSMLayer(SSULayer, Protocol):
    """What a ReplaySSM kernel reads from its layer: the cache, whose states
    after (conv, ssm) are its rings (x, dt, B), and its runtime state."""

    kv_cache: tuple[torch.Tensor, ...]
    ssu_state: ReplaySSMLayerState


class MambaReplaySSMKernel(MambaSSUKernel):
    name: ClassVar[str]

    @classmethod
    def is_replayssm(cls) -> bool:
        return True

    @classmethod
    def ring_len(cls, buffer_len: int, num_spec: int) -> int:
        """The ReplaySSM rings' physical length for `--replayssm-buffer-len
        buffer_len`; ring-tracker kernels also keep the live and current verify
        windows."""
        return buffer_len + (1 + num_spec if cls.uses_ring_trackers else 0)

    @classmethod
    def extra_state_shapes(
        cls,
        vllm_config: "VllmConfig",
        state_shapes: tuple[tuple[int, ...], ...],
        n_groups: int,
    ) -> tuple[tuple[int, ...], ...]:
        ring_len = cls.ring_len(
            vllm_config.cache_config.replayssm_buffer_len,
            vllm_config.num_speculative_tokens,
        )
        return MambaStateShapeCalculator.replayssm_ring_shapes(
            state_shapes[1],
            n_groups,
            vllm_config.parallel_config.tensor_parallel_size,
            ring_len,
        )

    @classmethod
    def extra_state_dtypes(cls, model_dtype: torch.dtype) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.replayssm_ring_dtypes(model_dtype)

    @classmethod
    def make_layer_state(cls, vllm_config: "VllmConfig") -> ReplaySSMLayerState:
        return ReplaySSMLayerState(
            buffer_len=vllm_config.cache_config.replayssm_buffer_len,
            num_spec=vllm_config.num_speculative_tokens,
        )

    @classmethod
    def supports_mamba_type(cls, mamba_type: MambaAttentionBackendEnum) -> bool:
        return mamba_type == MambaAttentionBackendEnum.MAMBA2

    @classmethod
    def validate_kernel(cls, cfg: LinearAttnSelectorConfig) -> list[str]:
        reasons = []
        query_len = 1 + cfg.num_speculative_tokens
        if query_len > 1 and cfg.replayssm_buffer_len < query_len:
            reasons.append(
                "ReplaySSM speculative decoding requires --replayssm-buffer-len "
                f">= 1 + num_speculative_tokens ({query_len}); got "
                f"{cfg.replayssm_buffer_len}"
            )
        return reasons
