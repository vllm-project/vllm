# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Constant novice experts for MoNE-pruned checkpoints.

MoNE (https://arxiv.org/abs/2507.00390) collapses low-importance experts into
"novices" that return a constant vector. Checkpoints declare the per-layer
retained expert count in ``quantization_config.moe_sparsity_config`` with
``method="novice"`` and store, per MoE layer:

* the router at its full original width, with rows reordered so the retained
  experts occupy the leading rows and the collapsed experts the trailing rows;
* the retained experts left-shifted into compact stacked tensors
  (``<experts prefix>.gate_up_proj`` and ``<experts prefix>.down_proj``), so
  physical row ``r`` holds the expert of router row ``r``;
* one ``<experts prefix>.novices`` tensor of shape
  ``[num_collapsed, hidden_size]`` whose rows match the trailing router rows.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExperts

NOVICES_SUFFIX = ".novices"


def decoder_layer_index(path: str) -> int | None:
    """Return the only integer component of a module path, if unique."""
    indices = [int(part) for part in path.split(".") if part.isdigit()]
    return indices[0] if len(indices) == 1 else None


@dataclass(frozen=True)
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
    # Sub-model configs (e.g. the text model inside a multimodal wrapper)
    # swap ``hf_config`` for the text config, which drops
    # ``quantization_config``; the quant config object built from the
    # top-level checkpoint config still carries the metadata.
    from vllm.config import get_current_vllm_config_or_none

    current_config = get_current_vllm_config_or_none()
    if current_config is None:
        return None
    raw = getattr(current_config.quant_config, "moe_sparsity_config", None)
    if raw is not None:
        return _parse_moe_sparsity_config(raw)
    return None


class ConstantExpertSubstitution(nn.Module):
    """Constant novice outputs for the collapsed experts of one MoE layer.

    Router outputs keep the full logical expert set. The trailing logical IDs
    are the collapsed experts; ``transform_routes`` maps the leading (retained)
    IDs to their compact physical rows and turns collapsed routes into zero-weight
    routes to physical expert 0, so any decomposed MoE backend can run them
    unchanged.
    """

    def __init__(
        self,
        layer_idx: int,
        num_logical_experts: int,
        substituted_expert_ids: tuple[int, ...],
        hidden_size: int,
        params_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        substituted = set(substituted_expert_ids)
        if not substituted <= set(range(num_logical_experts)):
            raise ValueError(
                f"novice experts for layer {layer_idx} must be within "
                f"[0, {num_logical_experts}), got {sorted(substituted)}"
            )
        if len(substituted) == num_logical_experts:
            raise ValueError(f"layer {layer_idx} must retain at least one expert")
        self.layer_idx = layer_idx
        self.substituted_expert_ids = tuple(sorted(substituted))
        retained = [i for i in range(num_logical_experts) if i not in substituted]
        self.num_compute_experts = len(retained)

        # Retained experts map to their physical row r >= 0, collapsed
        # experts to -1 - (their row in `novices`).
        routes = [0] * num_logical_experts
        for row, expert_id in enumerate(retained):
            routes[expert_id] = row
        for row, expert_id in enumerate(self.substituted_expert_ids):
            routes[expert_id] = -1 - row
        self._routes = tuple(routes)
        self.register_buffer(
            "expert_substitution_routes",
            torch.tensor(routes, dtype=torch.int32),
            persistent=False,
        )

        # NaN marks rows that the checkpoint has not provided yet.
        self.novices = nn.Parameter(
            torch.full((len(substituted), hidden_size), torch.nan, dtype=params_dtype),
            requires_grad=False,
        )
        self.novices.weight_loader = self.weight_loader

    @property
    def num_logical_experts(self) -> int:
        return len(self._routes)

    def physical_expert_id(self, expert_id: int) -> int:
        """Physical row of a retained expert, or -1 for a collapsed one."""
        return max(self._routes[expert_id], -1)

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
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Remap routes in place and return the novice experts' output."""
        valid = (topk_ids >= 0) & (topk_ids < self.num_logical_experts)
        routes = self.expert_substitution_routes[
            torch.where(valid, topk_ids, torch.zeros_like(topk_ids)).long()
        ].long()
        substituted = valid & (routes < 0)

        # Sum each token's router weights per collapsed expert in FP32; other
        # routes land in a trailing column that is dropped.
        num_values, value_size = self.novices.shape
        scales = topk_weights.new_zeros(
            (topk_ids.shape[0], num_values + 1), dtype=torch.float32
        )
        scales.scatter_add_(
            1,
            torch.where(substituted, -1 - routes, num_values),
            topk_weights.float(),
        )
        output = torch.zeros_like(hidden_states)
        output[..., :value_size] = scales[:, :num_values] @ self.novices.float()

        topk_weights.masked_fill_(~valid | substituted, 0.0)
        topk_ids.copy_(routes.clamp_min(0).to(topk_ids.dtype))
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
        tuple(range(retained, num_experts)),
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
                ("monolithic MoE backends", self.quant_method.is_monolithic),
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

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        # Retained experts of router rows 0..R-1 already sit in their compact
        # physical rows; collapsed router rows have no local expert.
        physical_id = self.expert_substitution.physical_expert_id(expert_id)
        if physical_id < 0:
            return -1
        return super()._map_global_expert_id_to_local_expert_id(physical_id)

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
        constant_output = self.expert_substitution.transform_routes(
            x, topk_weights, topk_ids
        )
        output = super().forward_modular(
            x, topk_weights, topk_ids, shared_experts, shared_experts_input
        )
        if isinstance(output, UnfinalizedMoEOutput):
            raise NotImplementedError(
                "expert substitution does not support deferred MoE finalize"
            )
        # The constant output is replicated across TP ranks, whose partial
        # outputs are all-reduced later, so add it on one rank only.
        if self.moe_config.tp_rank == 0:
            output.add_(constant_output[..., : output.shape[-1]])
        return output
