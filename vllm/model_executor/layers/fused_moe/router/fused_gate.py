# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import torch


class FusedGateRoutingMetadata(NamedTuple):
    """Token metadata that can be prepared once and reused across MoE layers."""

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
    """Routing semantics a gate backend must preserve."""

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
    """Per-batch routing metadata; True valid-token mask entries are active.

    Fixed expert IDs are logical IDs and may be overwritten by the backend.
    They are only used on rows selected by fixed_routing_mask. Image-token
    rows use the image bias instead of the regular bias or fixed routing.
    """

    image_token_mask: torch.Tensor | None = None
    valid_token_mask: torch.Tensor | None = None
    fixed_routing_mask: torch.Tensor | None = None
    fixed_expert_ids: torch.Tensor | None = None


class FusedGatePlan(ABC):
    @abstractmethod
    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return FP32 weights and logical expert IDs, in vLLM output order.

        Outputs have shape [tokens, top_k]. Rows excluded by valid_token_mask
        have zero weights and signed IDs of -1. The caller handles shared
        experts, EPLB mapping and routing capture after this call.
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
        """Check hardware, inputs, routing modes and capture readiness.

        This query must not launch GPU work, allocate device buffers or JIT
        compile. A backend must decline any routing mode it cannot preserve.
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
        """Bind this call or reuse a prepared plan for these inputs.

        Stateful backends own their plans and workspaces. Address-bound plans
        must account for tensor addresses/layouts, config, device and stream,
        and remain alive for captured graphs that use them. During capture,
        prepare must reuse warmed resources without device queries or JIT.
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
    """Select the first backend preserving this call's routing semantics."""
    for backend in backends:
        if backend.is_supported(hidden_states, weight, config, has_padding=has_padding):
            return backend
    return None
