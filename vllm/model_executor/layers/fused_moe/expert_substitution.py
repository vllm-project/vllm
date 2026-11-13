# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Constant novice experts for MoNE-pruned checkpoints.

`quantization_config.moe_sparsity_config` declares retained expert counts.
Leading router IDs index compact expert MLPs; trailing IDs index the constant
vectors in `.novices`. See `docs/design/expert_substitution.md` for the
checkpoint format and execution paths.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExperts

NOVICES_SUFFIX = ".novices"


def decoder_layer_index(path: str) -> int | None:
    """Return the only integer component of a module path, if unique."""
    indices = [int(part) for part in path.split(".") if part.isdigit()]
    return indices[0] if len(indices) == 1 else None


@dataclass(frozen=True, slots=True)
class ExpertSubstitutionSpec:
    # Retained expert count per decoder layer declared by the checkpoint.
    retained_experts: dict[int, int]


def _parse_moe_sparsity_config(raw: Any) -> ExpertSubstitutionSpec:
    try:
        if raw["method"] != "novice":
            raise ValueError(
                f"unsupported moe_sparsity_config method {raw['method']!r}"
            )
        counts = raw["num_experts"]
        if not isinstance(counts, list):
            raise ValueError("num_experts must be a per-layer list")
        retained: dict[int, int] = {}
        for layer_idx, count in enumerate(counts):
            if count is None:
                continue
            if not isinstance(count, int) or isinstance(count, bool) or count < 1:
                raise ValueError(
                    f"num_experts[{layer_idx}] must be a positive integer or null, "
                    f"got {count!r}"
                )
            retained[layer_idx] = count
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"invalid moe_sparsity_config: {e}") from e
    return ExpertSubstitutionSpec(retained)


def get_expert_substitution_spec(model_config: Any) -> ExpertSubstitutionSpec | None:
    """Parse the novice experts declared in a model's HF config."""
    for config in (
        getattr(model_config, "hf_text_config", None),
        getattr(model_config, "hf_config", None),
    ):
        quantization_config = getattr(config, "quantization_config", None)
        if quantization_config is None:
            continue
        raw = (
            quantization_config.get("moe_sparsity_config")
            if isinstance(quantization_config, Mapping)
            else getattr(quantization_config, "moe_sparsity_config", None)
        )
        if raw is not None:
            return _parse_moe_sparsity_config(raw)
    # A multimodal wrapper may replace hf_config with a text config that lacks
    # quantization_config. The quant config built from the outer checkpoint
    # still carries the pruning metadata.
    from vllm.config import get_current_vllm_config_or_none

    current_config = get_current_vllm_config_or_none()
    if current_config is None:
        return None
    raw = getattr(current_config.quant_config, "moe_sparsity_config", None)
    if raw is not None:
        return _parse_moe_sparsity_config(raw)
    return None


class ConstantExpertSubstitution(nn.Module):
    """Weighted constant outputs for one layer's trailing logical experts."""

    def __init__(
        self,
        layer_idx: int,
        num_logical_experts: int,
        num_compute_experts: int,
        hidden_size: int,
        params_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if not 0 < num_compute_experts < num_logical_experts:
            raise ValueError(
                f"layer {layer_idx} must retain between 1 and "
                f"{num_logical_experts - 1} experts, got {num_compute_experts}"
            )
        self.layer_idx = layer_idx
        self.num_logical_experts = num_logical_experts
        self.num_compute_experts = num_compute_experts

        # NaN marks rows that the checkpoint has not provided yet.
        self.novices = nn.Parameter(
            torch.full(
                (num_logical_experts - num_compute_experts, hidden_size),
                torch.nan,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        self.novices.weight_loader = self.weight_loader

    @staticmethod
    def supports_monolithic_routing(
        routing_method: RoutingMethodType, *, use_grouped_topk: bool
    ) -> bool:
        """Whether captured IDs and logits suffice to reconstruct novice weights."""
        return not use_grouped_topk and routing_method in (
            RoutingMethodType.Renormalize,
            RoutingMethodType.RenormalizeNaive,
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        if loaded_weight.shape != param.data.shape:
            raise ValueError(
                f"novices for layer {self.layer_idx} has shape "
                f"{tuple(loaded_weight.shape)}, expected {tuple(param.data.shape)}"
            )
        param.data.copy_(loaded_weight)

    def validate_loaded_values(self, prefix: str) -> None:
        if torch.isnan(self.novices).any():
            raise ValueError(f"{prefix} is missing or incomplete")

    def transform_routes(
        self,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> None:
        """Turn novice and invalid selections into zero-weight routes in place.

        `topk_ids` and `topk_weights` have shape `[num_tokens, top_k]` and are
        mutated. Compute the novice contribution before calling this method,
        while the original routing IDs and weights are still available.
        """
        retained = (topk_ids >= 0) & (topk_ids < self.num_compute_experts)
        topk_weights.masked_fill_(~retained, 0.0)
        topk_ids.masked_fill_(~retained, 0)

    def compute_output(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute novice output without changing logical routes.

        Return the hidden-state shape and dtype, with zero padding beyond
        the novice vector width. Weights include router normalization/scaling.
        """
        # Keep this dense even for sparse selections to avoid extra dispatch
        # calls. Retained and invalid routes enter a discarded column.
        substituted = (topk_ids >= self.num_compute_experts) & (
            topk_ids < self.num_logical_experts
        )
        num_values, value_size = self.novices.shape
        scales = self.novices.new_zeros((topk_ids.shape[0], num_values + 1))
        scales.scatter_add_(
            1,
            torch.where(
                substituted, topk_ids.long() - self.num_compute_experts, num_values
            ),
            topk_weights.to(scales.dtype),
        )
        output = torch.zeros_like(hidden_states)
        output[..., :value_size] = scales[:, :num_values] @ self.novices
        return output


def make_expert_substitution(
    model_config: Any,
    prefix: str,
    num_experts: int,
    hidden_size: int,
    params_dtype: torch.dtype | None,
) -> ConstantExpertSubstitution | None:
    spec = get_expert_substitution_spec(model_config)
    if spec is None:
        return None
    layer_idx = decoder_layer_index(prefix)
    if layer_idx is None:
        raise ValueError(
            f"cannot match MoE layer {prefix!r} to novice experts: expected "
            "exactly one layer index in its prefix"
        )
    retained = spec.retained_experts.get(layer_idx)
    if retained is None:
        return None
    if retained > num_experts:
        raise ValueError(
            f"moe_sparsity_config retains {retained} experts for layer {layer_idx} "
            f"but the model has {num_experts}"
        )
    if retained == num_experts:
        return None
    return ConstantExpertSubstitution(
        layer_idx,
        num_experts,
        retained,
        hidden_size,
        params_dtype or torch.get_default_dtype(),
    )


class SubstitutedRoutedExperts(RoutedExperts):
    """Routed experts that hold weights only for retained experts."""

    def __init__(
        self, *args: Any, expert_substitution: ConstantExpertSubstitution, **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        self.expert_substitution = expert_substitution
        moe_config = self.moe_config
        parallel_config = moe_config.moe_parallel_config
        unsupported = [
            name
            for name, enabled in (
                (
                    "quantized MoE weights",
                    not self.quant_method.supports_expert_substitution,
                ),
                ("MoE LoRA", moe_config.is_lora_enabled),
                (
                    "EPLB",
                    parallel_config.enable_eplb
                    or moe_config.num_experts
                    != expert_substitution.num_compute_experts,
                ),
                (
                    "fused shared experts",
                    self.expert_map_manager.num_fused_shared_experts > 0,
                ),
                ("expert parallelism", parallel_config.use_ep),
                ("data parallelism", parallel_config.dp_size > 1),
                ("prefill context parallelism", parallel_config.pcp_size > 1),
                ("sequence parallelism", parallel_config.is_sequence_parallel),
                ("router weights on expert inputs", self.apply_router_weight_on_input),
            )
            if enabled
        ]
        if unsupported:
            raise NotImplementedError(
                "expert substitution does not support: " + ", ".join(unsupported)
            )
        if self.quant_method.is_monolithic:
            # Allocation and backend local range still use R. Only the native
            # router's global expert count must cover all E logical experts.
            self.global_num_experts = expert_substitution.num_logical_experts

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        # Retained experts of router rows 0..R-1 already sit in their compact
        # physical rows; collapsed router rows have no local expert.
        if not 0 <= expert_id < self.expert_substitution.num_compute_experts:
            return -1
        return super()._map_global_expert_id_to_local_expert_id(expert_id)

    def get_expert_mapping(
        self,
        ckpt_gate_proj_name: str | None = None,
        ckpt_down_proj_name: str | None = None,
        ckpt_up_proj_name: str | None = None,
        include_fused: bool = False,
    ) -> list[tuple[str, str, int, str]]:
        # Checkpoint names address the full router-row (logical) expert set;
        # collapsed rows map to no local expert via the expert ID mapping.
        return self.build_expert_params_mapping(
            ckpt_gate_proj_name or self.ckpt_gate_proj_name,
            ckpt_down_proj_name or self.ckpt_down_proj_name,
            ckpt_up_proj_name or self.ckpt_up_proj_name,
            num_experts=self.expert_substitution.num_logical_experts,
            num_redundant_experts=0,
            routed_experts_prefix="",
            lora_base_layer_prefix=self.lora_base_layer_prefix,
            include_fused=include_fused,
            is_gated=self.moe_config.is_act_and_mul,
        )

    def forward_modular(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: "SharedExperts | None" = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Each TP rank holds the full novice vectors and routes. Compute/add
        # their output once so the runner's all-reduce does not duplicate it.
        constant_output = (
            self.expert_substitution.compute_output(x, topk_weights, topk_ids)
            if self.moe_config.tp_rank == 0
            else None
        )
        self.expert_substitution.transform_routes(topk_weights, topk_ids)
        output = super().forward_modular(
            x, topk_weights, topk_ids, shared_experts, shared_experts_input
        )
        if isinstance(output, UnfinalizedMoEOutput):
            raise NotImplementedError(
                "expert substitution does not support deferred MoE finalize"
            )
        if constant_output is not None:
            output.add_(constant_output[..., : output.shape[-1]])
        return output

    def forward_monolithic(
        self,
        x: torch.Tensor,
        router_logits: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        x_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert router_logits is not None
        assert x_scale is None, "expert substitution requires unquantized inputs"
        # Capture on the contributing TP rank only. The buffer is allocated
        # during graph capture and reused by graph replay without host sync.
        selected_ids = (
            torch.empty((x.shape[0], self.top_k), dtype=torch.int16, device=x.device)
            if self.moe_config.tp_rank == 0
            else None
        )
        output = self.quant_method.apply_monolithic(
            layer=self,
            x=x,
            router_logits=router_logits,
            input_ids=input_ids,
            routing_replay_out=selected_ids,
        )
        if isinstance(output, UnfinalizedMoEOutput):
            raise NotImplementedError(
                "expert substitution does not support deferred MoE finalize"
            )
        if selected_ids is not None:
            # For selected-softmax renormalization, the full-router denominator
            # cancels. Use the backend's IDs to preserve its top-k tie choices.
            weights = router_logits.gather(1, selected_ids.long()).float().softmax(-1)
            constant_output = self.expert_substitution.compute_output(
                x, weights, selected_ids
            )
            output.add_(constant_output[..., : output.shape[-1]])
        return output
