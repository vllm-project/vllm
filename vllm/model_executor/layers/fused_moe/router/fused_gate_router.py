# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence

import torch

import vllm.envs as envs
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.router.base_router import (
    get_moe_padding_mask,
)
from vllm.model_executor.layers.fused_moe.router.fused_gate import (
    FusedGateBackend,
    FusedGateConfig,
    FusedGateMetadata,
    FusedGateRoutingMetadata,
    prepare_fused_gate_routing_metadata,
    select_fused_gate_backend,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    FusedTopKBiasRouter,
)
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear


class FusedGateRouter(FusedTopKBiasRouter):
    """Fuse gate projection and routing, falling back to logits-based top-k."""

    def __init__(
        self,
        top_k: int,
        global_num_experts: int,
        e_score_correction_bias: torch.Tensor | None = None,
        renormalize: bool = True,
        routed_scaling_factor: float = 1.0,
        eplb_state: EplbLayerState | None = None,
        *,
        scoring_func: str = "sigmoid",
        hash_indices_table: torch.Tensor | None = None,
        num_fused_shared_experts: int = 0,
        shared_expert_weight: float = 1.0,
        bias_vl: torch.Tensor | None = None,
        image_sentinel_lo: int = 0,
        fused_gate_backends: Sequence[FusedGateBackend] = (),
        ep_rank: int = 0,
        defer_padding: bool = False,
    ):
        super().__init__(
            top_k=top_k,
            global_num_experts=global_num_experts,
            e_score_correction_bias=e_score_correction_bias,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
            eplb_state=eplb_state,
            scoring_func=scoring_func,
            hash_indices_table=hash_indices_table,
            num_fused_shared_experts=num_fused_shared_experts,
            shared_expert_weight=shared_expert_weight,
            bias_vl=bias_vl,
            image_sentinel_lo=image_sentinel_lo,
        )
        self.gate: GateLinear | None = None
        self._fused_gate_backends = tuple(fused_gate_backends)
        self.ep_rank = ep_rank
        # Native MegaMoE invalidates padded routes during input staging.
        self.defer_padding = defer_padding

    def set_gate(self, gate: torch.nn.Module | None) -> None:
        self.gate = gate if isinstance(gate, GateLinear) else None

    def set_fused_gate_backends(self, backends: Sequence[FusedGateBackend]) -> None:
        """Set per-router backends in priority order before warmup/capture."""
        self._fused_gate_backends = tuple(backends)

    def _fused_gate_config(self) -> FusedGateConfig:
        return FusedGateConfig(
            top_k=self.top_k,
            scoring_func=self.scoring_func,
            renormalize=self.renormalize,
            routed_scaling_factor=self.routed_scaling_factor,
            bias=self.e_score_correction_bias,
            image_bias=self.bias_vl if self.image_sentinel_lo > 0 else None,
            has_hash_routing=self._hash_indices_table is not None,
            ep_rank=self.ep_rank,
        )

    def _select_fused_gate_backend(
        self, hidden_states: torch.Tensor, config: FusedGateConfig
    ) -> FusedGateBackend | None:
        gate = self.gate
        if (
            gate is None
            or envs.VLLM_BATCH_INVARIANT
            or not gate.is_unquantized
            or gate.bias is not None
            or gate.out_dtype != torch.float32
            or gate.weight.shape[0] > self.global_num_experts
        ):
            return None
        has_padding = (
            not self.defer_padding
            and get_moe_padding_mask(hidden_states.shape[0]) is not None
        )
        return select_fused_gate_backend(
            self._fused_gate_backends,
            hidden_states,
            gate.weight,
            config,
            has_padding=has_padding,
        )

    def can_fuse_gate(self, hidden_states: torch.Tensor) -> bool:
        if not self._fused_gate_backends:
            return False
        return (
            self._select_fused_gate_backend(hidden_states, self._fused_gate_config())
            is not None
        )

    def _fused_gate_metadata(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None,
        routing_metadata: FusedGateRoutingMetadata | None,
    ) -> FusedGateMetadata:
        if routing_metadata is None:
            if self._hash_indices_table is not None or self.bias_vl is not None:
                assert input_ids is not None, "Hash/image routing requires input_ids"
                routing_metadata = prepare_fused_gate_routing_metadata(
                    input_ids,
                    has_hash_routing=self._hash_indices_table is not None,
                    image_sentinel_base_id=(
                        self.image_sentinel_lo
                        if self.bias_vl is not None and self.image_sentinel_lo > 0
                        else None
                    ),
                )
            else:
                routing_metadata = FusedGateRoutingMetadata(None, None, None)
        fixed_ids = fixed_mask = None
        if self._hash_indices_table is not None:
            safe_ids = routing_metadata.safe_hash_input_ids
            fixed_mask = routing_metadata.hash_token_mask
            assert safe_ids is not None and fixed_mask is not None, (
                "Hash routing requires prepared input IDs and a routing mask"
            )
            fixed_ids = self._hash_indices_table[safe_ids].to(torch.int64)
        padding = (
            None if self.defer_padding else get_moe_padding_mask(hidden_states.shape[0])
        )
        return FusedGateMetadata(
            image_token_mask=routing_metadata.image_token_mask,
            fixed_routing_mask=fixed_mask,
            fixed_expert_ids=fixed_ids,
            valid_token_mask=~padding if padding is not None else None,
        )

    def _compute_routed_topk(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(
            hidden_states, router_logits, indices_type, input_ids=input_ids
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None = None,
        indices_type: torch.dtype | None = None,
        *,
        input_ids: torch.Tensor | None = None,
        routing_metadata: FusedGateRoutingMetadata | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return routed weights and logical IDs, computing gate logits if needed.

        Native consumers handle shared experts, capture and EPLB themselves.
        Modular consumers use select_experts(), which adds that post-processing.
        """
        config = self._fused_gate_config()
        backend = self._select_fused_gate_backend(hidden_states, config)
        if backend is None:
            if router_logits is None:
                assert self.gate is not None, "Routing requires a gate or gate logits"
                router_logits, _ = self.gate(hidden_states)
            return super()._compute_routed_topk(
                hidden_states, router_logits, indices_type, input_ids=input_ids
            )

        assert self.gate is not None
        metadata = self._fused_gate_metadata(hidden_states, input_ids, routing_metadata)
        plan = backend.prepare(hidden_states, self.gate.weight, config, metadata)
        topk_weights, topk_ids = plan.forward()
        return topk_weights, topk_ids.to(
            torch.int32 if indices_type is None else indices_type
        )
