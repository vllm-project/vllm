# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Platform-neutral MuseSpark1x MoE pieces: routing, config, expert loading."""

from collections.abc import Iterable

import torch

from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    RoutingMethodType,
    get_routing_method_type,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
from vllm.models.muse_spark.common.ops.fused_routing import (
    fused_musespark1x_routing,
)


class MuseSpark1xRouter(BaseRouter):
    """MuseSpark1x top-k router.

    Optional metap output multiplier -> sigmoid/softmax -> add expert bias for
    SELECTION only -> top-k -> gather UNBIASED scores -> divide by
    (sum + division_eps) -> scale by route_scale.
    """

    def __init__(
        self,
        top_k: int,
        num_experts: int,
        expert_bias: torch.Tensor,
        route_scale: float,
        router_output_multiplier: float | None = None,
        router_score_normalization: str = "sigmoid",
        router_score_division_eps: float | None = None,
    ):
        super().__init__(top_k=top_k, global_num_experts=num_experts)
        if router_score_normalization not in ("sigmoid", "softmax"):
            raise ValueError(
                f"Invalid router_score_normalization: {router_score_normalization}"
            )
        self.expert_bias = expert_bias
        self.route_scale = route_scale
        self.router_output_multiplier = router_output_multiplier
        self.router_score_normalization = router_score_normalization
        self.router_score_division_eps = router_score_division_eps

    @property
    def routing_method_type(self) -> RoutingMethodType:
        """Classify this router in vLLM's routing vocabulary."""
        return get_routing_method_type(
            scoring_func=self.router_score_normalization,
            top_k=self.top_k,
            renormalize=True,
            num_expert_group=None,
            has_e_score_bias=True,
        )

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not router_logits.is_cuda:
            return self._compute_routing_native(router_logits, indices_type)

        topk_weights, topk_ids = fused_musespark1x_routing(
            router_logits,
            self.expert_bias,
            self.top_k,
            self.route_scale,
            router_output_multiplier=self.router_output_multiplier,
            router_score_normalization=self.router_score_normalization,
            router_score_division_eps=self.router_score_division_eps,
        )
        if indices_type is not None and indices_type != torch.int32:
            topk_ids = topk_ids.to(indices_type)
        return topk_weights, topk_ids

    def _compute_routing_native(
        self,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The recipe in torch: the kernel's reference, and the CPU path."""
        gating_output = router_logits
        if self.router_output_multiplier is not None:
            gating_output = gating_output * self.router_output_multiplier

        if self.router_score_normalization == "sigmoid":
            gating_output = torch.sigmoid(gating_output)
        else:
            gating_output = torch.softmax(gating_output, dim=-1)

        scores_for_topk = gating_output + self.expert_bias
        _, topk_ids = torch.topk(scores_for_topk, self.top_k, dim=-1)

        topk_weights = torch.gather(gating_output, dim=1, index=topk_ids)
        denom = topk_weights.sum(dim=-1, keepdim=True)
        if self.router_score_division_eps is not None:
            denom = denom + self.router_score_division_eps
        topk_weights = topk_weights / denom * self.route_scale

        return topk_weights.to(torch.float32), topk_ids.to(
            torch.int32 if indices_type is None else indices_type
        )


def needs_pre_norm_all_reduce(moe_config: FusedMoEConfig) -> bool:
    """Whether the GEMM2 rows must be all-reduced before the norm.

    Under TP the GEMM2 rows are partial sums and RMS is not linear in them.
    """
    return moe_config.moe_parallel_config.tp_size > 1


class MuseSparkMoEConfig(FusedMoEConfig):
    """MoE config whose top-k finalize is always deferred (except empty batches).

    MuseSpark1x normalizes every expert's output before the weighted top-k
    reduction, so the kernel must always stop after GEMM2.
    """

    def should_defer_moe_finalize(self, num_tokens: int) -> bool:
        return num_tokens > 0


class MuseSparkConsolidatedRoutedExperts(RoutedExperts):
    """RoutedExperts that load the consolidated (all-experts-stacked) tensors.

    Vendor subclasses supply ``_get_quant_method``.
    """

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[str]:
        for name, tensor in weights:
            param = getattr(self, name, None)
            if param is None:
                raise AttributeError(
                    f"{self.layer_name}: unexpected expert weight {name!r}"
                )
            if name in ("w13_weight", "w13_weight_scale"):
                # [E, 2I, ...] -> per-shard (w1/w3) -> per-expert
                for shard_id, col in (("w1", 0), ("w3", 1)):
                    shard = tensor.chunk(2, dim=1)[col]
                    for expert_id, loaded in enumerate(shard.unbind(0)):
                        param.weight_loader(
                            param=param,
                            loaded_weight=loaded,
                            weight_name=name,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
            elif name in ("w2_weight", "w2_weight_scale"):
                for expert_id, loaded in enumerate(tensor.unbind(0)):
                    param.weight_loader(
                        param=param,
                        loaded_weight=loaded,
                        weight_name=name,
                        shard_id="w2",
                        expert_id=expert_id,
                    )
            elif name in ("w13_weight_scale_2", "w13_input_scale"):
                # [E, 2] -> per-expert, per-shard scalars
                for expert_id in range(tensor.shape[0]):
                    for shard_id, col in (("w1", 0), ("w3", 1)):
                        param.weight_loader(
                            param=param,
                            loaded_weight=tensor[expert_id, col],
                            weight_name=name,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
            elif name in ("w2_weight_scale_2", "w2_input_scale"):
                for expert_id in range(tensor.shape[0]):
                    param.weight_loader(
                        param=param,
                        loaded_weight=tensor[expert_id],
                        weight_name=name,
                        shard_id="w2",
                        expert_id=expert_id,
                    )
            else:
                raise ValueError(
                    f"{self.layer_name}: unsupported expert weight {name!r}"
                )
            yield name


__all__ = [
    "MuseSpark1xRouter",
    "MuseSparkConsolidatedRoutedExperts",
    "MuseSparkMoEConfig",
    "needs_pre_norm_all_reduce",
]
