# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The local weight-sharing backend contract."""

from abc import ABC, abstractmethod
from typing import Literal, Protocol

import torch

TensorKind = Literal["param", "buffer"]
"""Whether a cached tensor is a model parameter or a buffer."""


class CachedTensorEntry(Protocol):
    kind: TensorKind

    def rebuild(
        self,
        device_index: int,
        buffers: dict[object, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Rebuild a tensor, optionally reusing imported storage buffers."""
        ...


class IPCBackend(ABC):
    """Materialize shared tensor storage and export rebuildable entries.

    Rebuilt tensor views must own their imported memory independently of the
    entries and loader state. Backend selection does not change the processed-
    weight model contract.
    """

    def materialize(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return the canonical tensors to export after warmup."""
        return tensors

    @abstractmethod
    def export_tensor(
        self, tensor: torch.Tensor, kind: TensorKind
    ) -> CachedTensorEntry:
        """Describe a tensor without copying it for each consumer."""
