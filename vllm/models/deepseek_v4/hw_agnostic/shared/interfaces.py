# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generic model-interface protocols for the hw-agnostic path.

Vendored from ``vllm/model_executor/models/interfaces.py``. Currently
holds ``SupportsPP``; additional protocols (``SupportsLoRA``, etc.) can
be added here as more hw_agnostic models are onboarded.

The pp-detection helpers (``supports_pp``, ``_supports_pp_attributes``,
etc.) live upstream and are not used directly by hw-agnostic models;
only the Protocol class is needed.
"""

from typing import ClassVar, Literal, Protocol, runtime_checkable

import torch
from torch import Tensor

from vllm.sequence import IntermediateTensors


@runtime_checkable
class SupportsPP(Protocol):
    """The interface required for all models that support pipeline parallel."""

    supports_pp: ClassVar[Literal[True]] = True

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors: ...

    def forward(
        self,
        input_ids: Tensor | None,
        positions: Tensor,
        *,
        intermediate_tensors: IntermediateTensors | None,
    ) -> IntermediateTensors | None: ...
