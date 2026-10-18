# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transformers TPSP backends and runtime contexts."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn

# TPSP v1 supports only standard LayerNorm and RMSNorm.
TPSPNormType = Literal["layer_norm", "rms_norm"]


class TPSPBackend(ABC):
    tpsp_max_microchunk_tokens_per_rank: int | None = None
    supported_norm_types: frozenset[TPSPNormType] = frozenset()

    def __init__(self, group_name: str, device: torch.device) -> None:
        self.group_name = group_name
        self.device = device
        self._closed = False

    @abstractmethod
    def open(
        self,
        *,
        dtype: torch.dtype,
        tp_size: int,
        hidden_size: int,
        max_batched_tokens: int,
        group_name: str,
        device: torch.device,
    ) -> bool:
        """Initialize backend resources once, before configuring ops groups."""

    @abstractmethod
    def close(self, context: Any | None = None) -> None: ...

    @abstractmethod
    def profile(
        self,
        ops_groups: Sequence[TPSPOpsGroup],
        max_batched_tokens: int,
        *,
        microchunk_tokens: int | None = None,
        threshold_tokens: int | None = None,
    ) -> TPSPContext | None:
        """Configure named ops groups on an already-open backend.

        Args:
            ops_groups: Named projection and normalization pairs to configure.
            max_batched_tokens: Maximum scheduler token count.
            microchunk_tokens: Optional global token-size override.
            threshold_tokens: Optional global token threshold override.

        Returns:
            The TPSP context, or None when unsupported.

        """

    @abstractmethod
    def tpsp_matmul_rs_norm_ag(
        self,
        projection_context: object,
        x: torch.Tensor,
        projection: nn.Module,
        residual: torch.Tensor,
        norm: nn.Module,
        *,
        config: object | None = None,
        norm_type: TPSPNormType = "rms_norm",
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def set_config(self, handle: object, config: object) -> None: ...

    def prepare_weights(self, projections: Sequence[nn.Module]) -> None:
        """Prepare backend-specific weights after model weights are loaded."""
        return None

    @abstractmethod
    def _create_context(self) -> object:
        """Create independent ops configuration using an already-open backend."""


@dataclass(frozen=True)
class TPSPProfile:
    """Shared TPSP activation decision."""

    enabled: bool
    threshold_tokens: int | None
    max_batched_tokens: int
    reason: str = ""

    def is_active(self, num_tokens: int) -> bool:
        if not self.enabled:
            return False
        if self.threshold_tokens is None:
            raise RuntimeError("TPSP has an invalid enabled profile")
        if not 1 <= num_tokens <= self.max_batched_tokens:
            raise ValueError("current_batched_tokens must be within the profiled range")
        return num_tokens >= self.threshold_tokens


@dataclass(frozen=True)
class TPSPOpsGroup:
    """A consecutive projection and normalization pair."""

    name: str
    projection: nn.Module
    norm: nn.Module


@dataclass(frozen=True)
class TPSPContext:
    """Shared backend and named per-group handles."""

    profile: TPSPProfile
    backend: Any
    handles: dict[str, object]


class UnsupportedTPSPNormError(ValueError):
    """Normalization not supported by the Transformers TPSP path."""


class UnsupportedTPSPModelError(ValueError):
    """Transformers model structure not supported by the TPSP path."""


def get_tpsp_norm_type(norm: nn.Module) -> TPSPNormType:
    """Identify known normalization implementations without guessing by attributes."""
    from vllm.model_executor.layers.layernorm import (
        LayerNorm,
        RMSNorm,
    )
    from vllm.model_executor.models.transformers.fusers.rms_norm import (
        TPAwareRMSNorm,
    )

    norm_cls = type(norm)
    if norm_cls in (nn.LayerNorm, LayerNorm):
        norm_type: TPSPNormType = "layer_norm"
    elif norm_cls in (RMSNorm, TPAwareRMSNorm):
        norm_type = "rms_norm"
        if norm.variance_size_override is not None:
            raise UnsupportedTPSPNormError(
                "TPSP RMSNorm does not support variance_size_override"
            )
    else:
        raise UnsupportedTPSPNormError(
            f"Unsupported TPSP normalization {type(norm).__name__}"
        )

    eps = (
        getattr(norm, "eps", None)
        if norm_type == "layer_norm"
        else getattr(norm, "variance_epsilon", None)
    )
    weight = getattr(norm, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 1:
        raise UnsupportedTPSPNormError(
            f"Unsupported TPSP normalization {type(norm).__name__}"
        )
    bias = getattr(norm, "bias", None)
    if bias is not None and (
        not isinstance(bias, torch.Tensor)
        or bias.ndim != 1
        or bias.shape != weight.shape
    ):
        raise UnsupportedTPSPNormError(
            f"Unsupported TPSP normalization {type(norm).__name__}"
        )
    if norm_type != "layer_norm" and bias is not None:
        raise UnsupportedTPSPNormError(
            f"Unsupported TPSP normalization {type(norm).__name__}"
        )
    if not isinstance(eps, (float, int)):
        raise UnsupportedTPSPNormError(
            f"Unsupported TPSP normalization {type(norm).__name__}"
        )
    if eps <= 0:
        raise ValueError("TPSP norm_eps must be positive")
    return norm_type


def tpsp_shard_residual(residual: torch.Tensor) -> torch.Tensor:
    """Return the current TP rank's padded shard of a full residual."""
    from vllm.distributed.parallel_state import get_tp_group

    if residual.ndim != 2:
        raise ValueError("TPSP full residual must be a 2D tensor")
    group = get_tp_group()
    rows = math.ceil(residual.size(0) / group.world_size)
    start = group.rank_in_group * rows
    count = min(rows, max(0, residual.size(0) - start))
    if count == rows:
        return residual[start : start + rows].contiguous()
    shard = residual.new_zeros((rows, residual.size(1)))
    shard[:count] = residual[start : start + count]
    return shard
