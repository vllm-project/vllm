# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from dataclasses import dataclass

import torch

from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEActivationFormat,
    FusedMoEExperts,
)


@dataclass
class _StandardCompactionState:
    retained_rows: torch.Tensor
    original_rows: int


class StandardTokenRowCompaction:
    """Remove rows whose router assignments were all dropped."""

    def __init__(self):
        self._enabled = False
        self._states: list[_StandardCompactionState | None] = [None, None]

    def configure(
        self,
        experts: FusedMoEExperts,
        supports_token_dropping: bool,
    ) -> None:
        self._enabled = (
            supports_token_dropping
            and experts.expert_capacity is not None
            and experts.activation_format() == FusedMoEActivationFormat.Standard
        )

    def compact_dispatch_input(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        ubatch_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._states[ubatch_id] = None
        if not self._enabled:
            return hidden_states, topk_ids, topk_weights
        if hidden_states.is_cuda and torch.cuda.is_current_stream_capturing():
            return hidden_states, topk_ids, topk_weights

        assert hidden_states.dim() == 2
        assert topk_ids.dim() == 2
        assert hidden_states.size(0) == topk_ids.size(0)
        retained_rows = (topk_ids >= 0).any(dim=1)
        if bool(retained_rows.all()):
            return hidden_states, topk_ids, topk_weights

        rows = retained_rows.nonzero(as_tuple=True)[0]
        self._states[ubatch_id] = _StandardCompactionState(
            retained_rows=rows,
            original_rows=hidden_states.size(0),
        )
        return (
            hidden_states.index_select(0, rows).contiguous(),
            topk_ids.index_select(0, rows).contiguous(),
            topk_weights.index_select(0, rows).contiguous(),
        )

    def is_compacted(self, ubatch_id: int) -> bool:
        return self._states[ubatch_id] is not None

    def scatter_output(
        self,
        output: torch.Tensor,
        compact_output: torch.Tensor,
        ubatch_id: int,
    ) -> Callable[[], None]:
        state = self._states[ubatch_id]
        if state is None:
            return lambda: None

        assert output.shape[0] == state.original_rows
        assert compact_output.shape[0] == state.retained_rows.numel()
        assert output.shape[1:] == compact_output.shape[1:]
        assert output.dtype == compact_output.dtype
        assert output.device == compact_output.device
        output.zero_()
        if compact_output.numel() != 0:
            output.index_copy_(0, state.retained_rows, compact_output)

        def receiver():
            _ = compact_output
            self._states[ubatch_id] = None

        return receiver
