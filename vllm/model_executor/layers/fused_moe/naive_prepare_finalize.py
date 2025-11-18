# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Naive prepare/finalize implementation for EP+DP without all2all kernels."""

import torch

from vllm.distributed import get_ep_group
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoEP,
)


class FusedMoENaivePrepareAndFinalize(MoEPrepareAndFinalizeNoEP):
    """Dispatch/combine via prepare/finalize hooks for DP+EP without all2all."""

    def preprocess_inputs(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        layer: torch.nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        is_sequence_parallel = getattr(layer, "is_sequence_parallel", False)
        return get_ep_group().dispatch(
            hidden_states, router_logits, is_sequence_parallel
        )

    def postprocess_output(
        self,
        result,
        layer: torch.nn.Module,
    ):
        shared_experts = getattr(layer, "shared_experts", None)
        zero_expert_num = getattr(layer, "zero_expert_num", 0) or 0
        if isinstance(result, tuple):
            if shared_experts is not None:
                shared_output, expert_output = result
                return shared_output, self._combine(expert_output, layer)
            if zero_expert_num > 0:
                expert_output, aux = result
                return self._combine(expert_output, layer), aux
        return self._combine(result, layer)

    @staticmethod
    def _combine(tensor: torch.Tensor, layer: torch.nn.Module) -> torch.Tensor:
        if tensor.numel() == 0:
            return tensor
        is_sequence_parallel = getattr(layer, "is_sequence_parallel", False)
        return get_ep_group().combine(tensor, is_sequence_parallel)
