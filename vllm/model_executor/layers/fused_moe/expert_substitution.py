# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Constant expert substitution for MoNE-pruned checkpoints.

MoNE (https://arxiv.org/abs/2507.00390) replaces redundant experts with
"novices" that return a constant vector. Checkpoints list the replaced experts
per decoder layer either in ``config.approximate_experts``, with constants named
``<experts prefix>.<expert id>.approx_value``, or in
``compression_config.transform_config.expert_substitution``, with explicitly
named constants. Retained experts keep their logical checkpoint names and are
loaded into compact physical rows.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExperts

APPROX_VALUE_SUFFIX = ".approx_value"


def decoder_layer_index(path: str) -> int | None:
    """Return the only integer component of a module path, if unique."""
    indices = [int(part) for part in path.split(".") if part.isdigit()]
    return indices[0] if len(indices) == 1 else None


@dataclass(frozen=True)
class ExpertSubstitutionSpec:
    # Sorted substituted expert IDs per decoder layer.
    experts: dict[int, tuple[int, ...]]
    # Explicitly named constant tensors and the (layer, expert) pairs they
    # supply. Without an entry, constants use the ``approx_value`` naming.
    value_names: dict[str, tuple[tuple[int, int], ...]] = field(default_factory=dict)


# The only router semantics that the constant-v1 format supports.
_ROUTER_SEMANTICS = {
    "preserve_logical_expert_ids": True,
    "preserve_router_weights": True,
    "renormalize_after_substitution": False,
}


def _expert_ids(layer_idx: int, expert_ids: Any) -> tuple[int, ...]:
    ids = tuple(sorted(int(expert_id) for expert_id in expert_ids or ()))
    if len(set(ids)) != len(ids) or any(expert_id < 0 for expert_id in ids):
        raise ValueError(
            f"substituted experts of layer {layer_idx} must be unique, "
            f"non-negative expert IDs, got {list(ids)}"
        )
    return ids


def _parse_approximate_experts(raw: Any) -> ExpertSubstitutionSpec:
    if not isinstance(raw, Mapping):
        raise ValueError("approximate_experts must map layer indices to expert IDs")
    experts: dict[int, tuple[int, ...]] = {}
    for layer, expert_ids in raw.items():
        try:
            layer_idx = int(layer)
            experts[layer_idx] = _expert_ids(layer_idx, expert_ids)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"approximate_experts has an invalid entry for layer {layer!r}"
            ) from e
    return ExpertSubstitutionSpec({k: v for k, v in experts.items() if v})


def _parse_expert_substitution(raw: Any) -> ExpertSubstitutionSpec:
    experts: dict[int, tuple[int, ...]] = {}
    value_names: dict[str, list[tuple[int, int]]] = {}
    try:
        if raw["version"] != 1 or raw["router_semantics"] != _ROUTER_SEMANTICS:
            raise ValueError("only version 1 with unchanged routing is supported")
        for path, target in raw["targets"].items():
            layer_idx = decoder_layer_index(path)
            if layer_idx is None or layer_idx in experts:
                raise ValueError(f"target {path!r} must name one decoder layer")
            if target["weight_layout"] != "compact_retained_experts":
                raise ValueError(f"unsupported weight layout for {path!r}")
            for expert_id, replacement in target["replacements"].items():
                if replacement["format"] != "constant-v1":
                    raise ValueError(f"unsupported replacement format for {path!r}")
                value_names.setdefault(replacement["tensors"]["value"], []).append(
                    (layer_idx, int(expert_id))
                )
            experts[layer_idx] = _expert_ids(layer_idx, target["replacements"])
    except (AttributeError, KeyError, TypeError, ValueError) as e:
        raise ValueError(f"invalid expert_substitution metadata: {e}") from e
    return ExpertSubstitutionSpec(
        experts, {name: tuple(pairs) for name, pairs in value_names.items()}
    )


def get_expert_substitution_spec(model_config: Any) -> ExpertSubstitutionSpec | None:
    """Parse the substituted experts declared in a model's HF config."""
    hf_configs = [
        getattr(model_config, name, None) for name in ("hf_text_config", "hf_config")
    ]
    approximate = None
    for config in hf_configs:
        approximate = getattr(config, "approximate_experts", None)
        if approximate is not None:
            break
    compression_config = getattr(hf_configs[1], "compression_config", None)
    transform_config = (
        compression_config.get("transform_config")
        if isinstance(compression_config, Mapping)
        else None
    )
    declared = (
        transform_config.get("expert_substitution")
        if isinstance(transform_config, Mapping)
        else None
    )
    if approximate is not None and declared is not None:
        raise ValueError(
            "config declares both approximate_experts and "
            "compression_config.transform_config.expert_substitution"
        )
    if approximate is not None:
        spec = _parse_approximate_experts(approximate)
    elif declared is not None:
        spec = _parse_expert_substitution(declared)
    else:
        return None
    return spec if spec.experts else None


class ConstantExpertSubstitution(nn.Module):
    """Constant outputs for the substituted experts of one MoE layer.

    Router outputs keep logical expert IDs. ``transform_routes`` maps retained
    experts to compact physical rows and turns substituted routes into
    zero-weight routes to physical expert 0, so any decomposed MoE backend can
    run them unchanged.
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
                f"approximate_experts for layer {layer_idx} must be within "
                f"[0, {num_logical_experts}), got {sorted(substituted)}"
            )
        if len(substituted) == num_logical_experts:
            raise ValueError(
                f"approximate_experts for layer {layer_idx} must retain at least "
                "one expert"
            )
        self.layer_idx = layer_idx
        self.substituted_expert_ids = tuple(sorted(substituted))
        retained = [i for i in range(num_logical_experts) if i not in substituted]
        self.num_compute_experts = len(retained)

        # Retained experts map to their physical row r >= 0, substituted
        # experts to -1 - (their row in `values`).
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

        # NaN marks values that the checkpoint has not provided yet.
        self.values = nn.Parameter(
            torch.full((len(substituted), hidden_size), torch.nan, dtype=params_dtype),
            requires_grad=False,
        )
        self.values.weight_loader = self.weight_loader

    @property
    def num_logical_experts(self) -> int:
        return len(self._routes)

    def physical_expert_id(self, expert_id: int) -> int:
        """Physical row of a retained expert, or -1 for a substituted one."""
        return max(self._routes[expert_id], -1)

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, expert_id: int
    ) -> None:
        if not 0 <= expert_id < self.num_logical_experts or (
            self._routes[expert_id] >= 0
        ):
            raise ValueError(
                f"expert {expert_id} of layer {self.layer_idx} is not listed in "
                "approximate_experts"
            )
        row = param.data[-1 - self._routes[expert_id]]
        if loaded_weight.shape != row.shape:
            raise ValueError(
                f"approx_value for expert {expert_id} of layer {self.layer_idx} has "
                f"shape {tuple(loaded_weight.shape)}, expected {tuple(row.shape)}"
            )
        row.copy_(loaded_weight)

    def validate_loaded_values(self, prefix: str) -> None:
        missing_rows = torch.isnan(self.values).any(dim=1).nonzero().flatten().tolist()
        if missing_rows:
            missing = [self.substituted_expert_ids[row] for row in missing_rows]
            raise ValueError(
                f"{prefix} is missing approx_value for {len(missing)} "
                f"expert(s), first missing expert IDs: {missing[:8]}"
            )

    def transform_routes(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Remap routes in place and return the constant experts' output."""
        valid = (topk_ids >= 0) & (topk_ids < self.num_logical_experts)
        routes = self.expert_substitution_routes[
            torch.where(valid, topk_ids, torch.zeros_like(topk_ids)).long()
        ].long()
        substituted = valid & (routes < 0)

        # Sum each token's router weights per substituted expert in FP32; other
        # routes land in a trailing column that is dropped.
        num_values, value_size = self.values.shape
        scales = topk_weights.new_zeros(
            (topk_ids.shape[0], num_values + 1), dtype=torch.float32
        )
        scales.scatter_add_(
            1,
            torch.where(substituted, -1 - routes, num_values),
            topk_weights.float(),
        )
        output = torch.zeros_like(hidden_states)
        output[..., :value_size] = scales[:, :num_values] @ self.values.float()

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
            f"cannot match MoE layer {prefix!r} to substituted experts: expected "
            "exactly one layer index in its prefix"
        )
    expert_ids = spec.experts.get(layer_idx)
    if expert_ids is None:
        return None
    return ConstantExpertSubstitution(
        layer_idx,
        num_experts,
        expert_ids,
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
                    not isinstance(self.quant_method, UnquantizedFusedMoEMethod),
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
        # Checkpoints name retained experts by their logical ID.
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
        # Fused checkpoint tensors would cover every logical expert at once.
        return self.build_expert_params_mapping(
            ckpt_gate_proj_name or self.ckpt_gate_proj_name,
            ckpt_down_proj_name or self.ckpt_down_proj_name,
            ckpt_up_proj_name or self.ckpt_up_proj_name,
            num_experts=self.expert_substitution.num_logical_experts,
            routed_experts_prefix="",
            lora_base_layer_prefix=self.lora_base_layer_prefix,
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
