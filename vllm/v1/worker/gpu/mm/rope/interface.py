# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class RopeState(ABC):
    """Shared interface for multi-dimensional RoPE variants (M-RoPE, XD-RoPE).

    Implementations pre-compute positions during prefill and prepare
    per-step position tensors for the model forward pass.
    """

    @abstractmethod
    def init_prefill_positions(
        self,
        req_idx: int,
        model: nn.Module,
        prefill_token_ids: list[int],
        mm_features: list,
    ) -> None: ...

    @abstractmethod
    def apply_staged_writes(self) -> None: ...

    @abstractmethod
    def prepare_positions(
        self,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        prefill_lens: torch.Tensor,
        num_computed_tokens: torch.Tensor,
    ) -> None: ...

    @abstractmethod
    def get_positions(self, num_tokens: int) -> torch.Tensor: ...
