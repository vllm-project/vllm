# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Protocol

import torch


class FusedMoEDirectBackend(Protocol):
    @property
    def can_overlap_shared_experts(self) -> bool: ...

    @property
    def output_is_reduced(self) -> bool: ...

    @property
    def topk_indices_dtype(self) -> torch.dtype: ...

    @property
    def is_monolithic(self) -> bool: ...

    def __call__(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor: ...
