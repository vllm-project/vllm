# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import torch


class FusedGateRoutingMetadata(NamedTuple):
    """Token metadata shared across MoE layers in a model forward."""

    safe_hash_input_ids: torch.Tensor | None
    image_token_mask: torch.Tensor | None
    hash_token_mask: torch.Tensor | None


def prepare_fused_gate_routing_metadata(
    input_ids: torch.Tensor,
    *,
    has_hash_routing: bool,
    image_sentinel_base_id: int | None,
) -> FusedGateRoutingMetadata:
    if image_sentinel_base_id is None:
        return FusedGateRoutingMetadata(
            input_ids if has_hash_routing else None,
            None,
            torch.ones_like(input_ids, dtype=torch.bool) if has_hash_routing else None,
        )
    image_token_mask = (input_ids >= image_sentinel_base_id) & (
        input_ids < image_sentinel_base_id + 5
    )
    if not has_hash_routing:
        return FusedGateRoutingMetadata(None, image_token_mask, None)
    return FusedGateRoutingMetadata(
        torch.where(image_token_mask, 0, input_ids),
        image_token_mask,
        ~image_token_mask,
    )


@dataclass(frozen=True)
class FusedGateConfig:
    """Routing semantics a fused gate must preserve."""

    top_k: int
    scoring_func: str
    renormalize: bool
    routed_scaling_factor: float
    bias: torch.Tensor | None = None
    image_bias: torch.Tensor | None = None
    has_hash_routing: bool = False
    ep_rank: int = 0


@dataclass(frozen=True)
class FusedGateMetadata:
    """Per-call masks and logical IDs for fixed routing.

    True valid-token entries are active. Image rows use image_bias instead of
    bias or fixed routing. fixed_expert_ids applies only to fixed_routing_mask
    rows and is scratch storage that the backend may overwrite.
    """

    image_token_mask: torch.Tensor | None = None
    valid_token_mask: torch.Tensor | None = None
    fixed_routing_mask: torch.Tensor | None = None
    fixed_expert_ids: torch.Tensor | None = None


class FusedGatePlan(ABC):
    @abstractmethod
    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return contiguous FP32 weights and signed logical expert IDs.

        Both outputs have shape `[tokens, top_k]`, in weights-first order.
        Inactive rows have zero weights and IDs of -1. Shared experts, routing
        capture and EPLB mapping belong to the caller.
        """
        raise NotImplementedError


class FusedGateBackend(ABC):
    @abstractmethod
    def is_supported(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        config: FusedGateConfig,
        *,
        has_padding: bool,
    ) -> bool:
        """Check hardware, tensor metadata and every requested routing mode.

        Must not launch GPU work, allocate device buffers or JIT compile.
        Unsupported modes must be declined before the runner skips the gate.
        """
        raise NotImplementedError

    @abstractmethod
    def prepare(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        config: FusedGateConfig,
        metadata: FusedGateMetadata,
    ) -> FusedGatePlan:
        """Bind this call to a plan owned by the backend.

        Cached plans must account for addresses, layouts, config, device and
        stream. Backends own workspace lifetimes, including captured graphs.
        During capture, reuse warmed resources without device queries or JIT.
        """
        raise NotImplementedError


def default_fused_gate_backends() -> tuple[FusedGateBackend, ...]:
    from .deep_gemm_gate import DeepGemmGateBackend

    return (DeepGemmGateBackend(),)


def select_fused_gate_backend(
    backends: Sequence[FusedGateBackend],
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    config: FusedGateConfig,
    *,
    has_padding: bool,
) -> FusedGateBackend | None:
    for backend in backends:
        if backend.is_supported(hidden_states, weight, config, has_padding=has_padding):
            return backend
    return None
