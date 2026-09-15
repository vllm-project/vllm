# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch

from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
    MoEPermuteScratch,
)


@dataclass(frozen=True)
class _MoEPermuteScratchConfig:
    max_num_tokens: int
    topk: int
    num_experts: int
    num_local_experts: int
    device: torch.device
    hidden_size: int | None
    hidden_dtype: torch.dtype | None


class MoEWorkspacePool:
    """Caches reusable MoE workspaces by compatible configuration."""

    def __init__(self):
        self._permute_scratch: dict[_MoEPermuteScratchConfig, MoEPermuteScratch] = {}

    def get_permute_scratch(
        self,
        *,
        max_num_tokens: int,
        topk: int,
        num_experts: int,
        num_local_experts: int,
        device: torch.device,
        hidden_size: int | None = None,
        hidden_dtype: torch.dtype | None = None,
        allow_create: bool = True,
    ) -> MoEPermuteScratch:
        config = _MoEPermuteScratchConfig(
            max_num_tokens=max_num_tokens,
            topk=topk,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            device=device,
            hidden_size=hidden_size,
            hidden_dtype=hidden_dtype,
        )
        scratch = self._permute_scratch.get(config)
        if scratch is None:
            if not allow_create:
                raise RuntimeError(
                    "Workspace is locked but the requested MoE permute scratch "
                    "has not been allocated during warmup."
                )
            scratch = MoEPermuteScratch(
                max_num_tokens=max_num_tokens,
                topk=topk,
                num_experts=num_experts,
                num_local_experts=num_local_experts,
                device=device,
                hidden_size=hidden_size,
                hidden_dtype=hidden_dtype,
            )
            self._permute_scratch[config] = scratch
        return scratch
