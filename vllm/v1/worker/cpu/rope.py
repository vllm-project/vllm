# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import cast

import torch
import torch.nn as nn

from vllm import _custom_ops as ops
from vllm.config import ModelConfig
from vllm.model_executor.models.interfaces import SupportsMRoPE


class RopeState:
    """State for multi-dimensional (M-RoPE) positions.

    `num_dims` is the number of position channels the model consumes, which
    the model config derives from its M-RoPE sections.

    NOTE: `positions` is implemented with one additional dummy position on
    purpose to make it non-contiguous so that it can work with torch compile.
    See detailed explanation in
    https://github.com/vllm-project/vllm/pull/12128#discussion_r1926431923

    NOTE: For text-only inputs, each dimension has identical position IDs,
    making M-RoPE functionally equivalent to 1D-RoPE.
    See page 5 of https://arxiv.org/abs/2409.12191
    """

    def __init__(
        self,
        num_dims: int,
        max_num_reqs: int,
        max_num_tokens: int,
        max_model_len: int,
        device: torch.device,
    ):
        self.num_dims = num_dims
        self.max_num_reqs = max_num_reqs
        self.max_num_tokens = max_num_tokens
        self.max_model_len = max_model_len
        self.device = device

        self.prefill_positions = torch.zeros(
            (max_num_reqs * num_dims, max_model_len),
            dtype=torch.int32,
            device=device,
        )
        self.positions = torch.zeros(
            (num_dims, max_num_tokens + 1), dtype=torch.int64, device=device
        )
        self.prefill_delta = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)

    def init_prefill_positions(
        self,
        req_idx: int,
        model: nn.Module,
        prefill_token_ids: list[int],
        mm_features: list,
    ) -> None:
        mrope_model = cast(SupportsMRoPE, model)
        prefill_positions, delta = mrope_model.get_mrope_input_positions(
            prefill_token_ids, mm_features
        )
        self.prefill_delta[req_idx] = delta

        for i in range(self.num_dims):
            pos = prefill_positions[i]
            self.prefill_positions[self.num_dims * req_idx + i, : pos.shape[0]].copy_(
                pos
            )

    def get_positions(self, num_tokens: int) -> torch.Tensor:
        return self.positions[:, :num_tokens]

    def read_prefill_positions(self, req_idx: int, length: int) -> torch.Tensor:
        """Return per-request prefill positions as [num_dims, length]."""
        base = self.num_dims * req_idx
        return self.prefill_positions[base : base + self.num_dims, :length]

    def update_prefill_positions(
        self, req_idx: int, positions: torch.Tensor, delta: int
    ) -> None:
        """Overwrite a request's prefill positions with recomputed values."""
        base = self.num_dims * req_idx
        length = positions.shape[1]
        self.prefill_positions[base : base + self.num_dims, :length].copy_(positions)
        self.prefill_delta[req_idx] = delta

    def prepare_positions(
        self,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        prefill_lens: torch.Tensor,
        num_computed_tokens: torch.Tensor,
    ) -> None:
        ops.prepare_mrope_positions_cpu(
            self.positions,
            self.prefill_positions,
            self.prefill_delta,
            idx_mapping,
            query_start_loc,
            prefill_lens,
            num_computed_tokens,
        )


def get_rope_state(
    model_config: ModelConfig,
    model: nn.Module,
    max_num_reqs: int,
    max_num_tokens: int,
    max_model_len: int,
    device: torch.device,
) -> RopeState | None:
    """Create a RopeState if the model uses multi-dimensional RoPE."""
    if not model_config.uses_mrope:
        return None

    assert isinstance(model, SupportsMRoPE)
    return RopeState(
        num_dims=model_config.mrope_num_dims,
        max_num_reqs=max_num_reqs,
        max_num_tokens=max_num_tokens,
        max_model_len=max_model_len,
        device=device,
    )
