# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd.

"""Base contract for registry-owned rotation transforms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn


class RotationTransform(nn.Module, ABC):
    """A shared mathematical rotation owned by ``RotationRegistry``."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        if isinstance(hidden_size, bool) or not isinstance(hidden_size, int):
            raise TypeError("hidden_size must be an integer")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        self.hidden_size = hidden_size

    @classmethod
    @abstractmethod
    def from_config(cls, config: Mapping[str, Any]) -> RotationTransform:
        """Build one transform from its type-specific configuration."""
        raise NotImplementedError

    @abstractmethod
    def validate_loaded(self, tag: str) -> None:
        """Verify that checkpoint-backed parameters were loaded."""
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        *,
        inverse: bool = False,
    ) -> torch.Tensor:
        """Apply the transform or its inverse to the last tensor dimension."""
        raise NotImplementedError

    def register_warmup(self, *, inverse: bool, dtype: torch.dtype) -> None:
        """Register any JIT kernels required by this transform."""


__all__ = ["RotationTransform"]
