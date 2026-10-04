# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from torch import nn
from typing_extensions import Self

from vllm.platforms import current_platform


@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
def _compute_constant_substitution_output(
    output: torch.Tensor,
    values: torch.Tensor,
    substitution_rows: torch.Tensor,
    substitution_mask: torch.Tensor,
    topk_weights: torch.Tensor,
) -> None:
    # Explicit compilation fuses these intermediates inside the opaque MoE op.
    gathered = values[substitution_rows.clamp_min(0)].float()
    scales = topk_weights.float() * substitution_mask
    result = torch.sum(gathered * scales.unsqueeze(-1), dim=1)
    output[..., : values.size(-1)].copy_(result.to(output.dtype))


class _SchemaModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _RouterSemantics(_SchemaModel):
    preserve_logical_expert_ids: bool
    preserve_router_weights: bool
    renormalize_after_substitution: bool

    @model_validator(mode="after")
    def validate_supported_semantics(self) -> Self:
        if (
            self.preserve_logical_expert_ids,
            self.preserve_router_weights,
            self.renormalize_after_substitution,
        ) != (True, True, False):
            raise ValueError("unsupported router semantics")
        return self


class _ValueTensors(_SchemaModel):
    value: str = Field(min_length=1)


class _ReplacementSchema(_SchemaModel):
    format: Literal["constant-v1"]
    tensors: _ValueTensors


class _TargetSchema(_SchemaModel):
    num_logical_experts: int = Field(gt=0)
    weight_layout: Literal["compact_retained_experts"]
    replacements: dict[str, _ReplacementSchema] = Field(min_length=1)


class _SubstitutionSchema(_SchemaModel):
    version: int = Field(ge=1, le=1)
    router_semantics: _RouterSemantics
    targets: dict[str, _TargetSchema] = Field(min_length=1)


@dataclass(frozen=True)
class ConstantExpertSubstitutionSpec:
    logical_expert_id: int
    value_tensor: str


@dataclass(frozen=True)
class ExpertSubstitutionTarget:
    module_path: str
    num_logical_experts: int
    replacements: tuple[ConstantExpertSubstitutionSpec, ...]

    @property
    def substituted_expert_ids(self) -> tuple[int, ...]:
        return tuple(spec.logical_expert_id for spec in self.replacements)


@dataclass(frozen=True)
class ExpertSubstitutionConfig:
    version: int
    targets: tuple[ExpertSubstitutionTarget, ...]

    def get_target(self, module_path: str) -> ExpertSubstitutionTarget | None:
        exact = [target for target in self.targets if target.module_path == module_path]
        if exact:
            return exact[0]

        def is_path_suffix(longer: str, shorter: str) -> bool:
            return longer.endswith(f".{shorter}")

        matches = [
            target
            for target in self.targets
            if is_path_suffix(target.module_path, module_path)
            or is_path_suffix(module_path, target.module_path)
        ]
        if len(matches) > 1:
            paths = sorted(target.module_path for target in matches)
            raise ValueError(
                f"expert substitution module path {module_path!r} is ambiguous; "
                f"matching configured targets: {paths}"
            )
        return matches[0] if matches else None


@dataclass(frozen=True)
class ExpertLayout:
    """Map stable logical expert IDs to compact physical MLP rows."""

    num_logical_experts: int
    compute_expert_ids: tuple[int, ...]
    substituted_expert_ids: tuple[int, ...]
    logical_to_physical: tuple[int, ...]

    @classmethod
    def from_substitutions(
        cls,
        num_logical_experts: int,
        substituted_expert_ids: Sequence[int],
    ) -> "ExpertLayout":
        if num_logical_experts <= 0:
            raise ValueError("num_logical_experts must be positive")

        substituted_ids = tuple(sorted(int(i) for i in substituted_expert_ids))
        if not substituted_ids:
            raise ValueError("substitution expert IDs must not be empty")
        if len(set(substituted_ids)) != len(substituted_ids):
            raise ValueError("substitution expert IDs must be unique")
        invalid = [i for i in substituted_ids if not 0 <= i < num_logical_experts]
        if invalid:
            raise ValueError(
                "substitution expert IDs are outside the logical expert range: "
                f"{invalid[:8]}"
            )

        substituted_set = set(substituted_ids)
        compute_ids = tuple(
            i for i in range(num_logical_experts) if i not in substituted_set
        )
        if not compute_ids:
            raise ValueError("at least one full compute expert must be retained")

        physical_ids = {logical_id: row for row, logical_id in enumerate(compute_ids)}
        return cls(
            num_logical_experts=num_logical_experts,
            compute_expert_ids=compute_ids,
            substituted_expert_ids=substituted_ids,
            logical_to_physical=tuple(
                physical_ids.get(logical_id, -1)
                for logical_id in range(num_logical_experts)
            ),
        )


def parse_expert_substitution_config(
    config: Any,
) -> ExpertSubstitutionConfig | None:
    """Parse the versioned expert-substitution inference representation."""
    compression_config = getattr(config, "compression_config", None)
    if compression_config is None:
        return None
    if not isinstance(compression_config, Mapping):
        raise ValueError("compression_config must be a mapping")
    transform_config = compression_config.get("transform_config")
    if transform_config is None:
        return None
    if not isinstance(transform_config, Mapping):
        raise ValueError("compression_config.transform_config must be a mapping")
    raw_config = transform_config.get("expert_substitution")
    if raw_config is None:
        return None
    try:
        schema = _SubstitutionSchema.model_validate(raw_config)
    except ValidationError as exc:
        raise ValueError(f"invalid expert_substitution metadata: {exc}") from exc

    targets: list[ExpertSubstitutionTarget] = []
    for module_path, target in schema.targets.items():
        if not module_path:
            raise ValueError("expert_substitution target paths must be non-empty")
        replacements: list[ConstantExpertSubstitutionSpec] = []
        seen_expert_ids: set[int] = set()
        for raw_expert_id, substitution in target.replacements.items():
            try:
                expert_id = int(raw_expert_id)
            except ValueError as exc:
                raise ValueError(
                    f"expert_substitution target {module_path!r} has invalid "
                    f"expert ID {raw_expert_id!r}"
                ) from exc
            if expert_id in seen_expert_ids:
                raise ValueError(
                    f"expert_substitution target {module_path!r} declares "
                    f"logical expert {expert_id} twice"
                )
            seen_expert_ids.add(expert_id)
            value_tensor = substitution.tensors.value
            replacements.append(ConstantExpertSubstitutionSpec(expert_id, value_tensor))

        replacements.sort(key=lambda spec: spec.logical_expert_id)
        ExpertLayout.from_substitutions(
            target.num_logical_experts,
            [spec.logical_expert_id for spec in replacements],
        )
        targets.append(
            ExpertSubstitutionTarget(
                module_path=module_path,
                num_logical_experts=target.num_logical_experts,
                replacements=tuple(replacements),
            )
        )

    targets.sort(key=lambda target: target.module_path)
    return ExpertSubstitutionConfig(version=schema.version, targets=tuple(targets))


class ConstantExpertSubstitution(nn.Module):
    """Execute the homogeneous ``constant-v1`` substitution format."""

    def __init__(
        self,
        target: ExpertSubstitutionTarget,
        hidden_size: int,
        params_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.target = target
        layout = ExpertLayout.from_substitutions(
            target.num_logical_experts, target.substituted_expert_ids
        )
        self.num_logical_experts = layout.num_logical_experts
        self.num_compute_experts = len(layout.compute_expert_ids)
        self._compute_expert_ids = layout.compute_expert_ids
        self.substituted_expert_ids = layout.substituted_expert_ids
        self._value_tensor_names = {
            spec.logical_expert_id: spec.value_tensor for spec in target.replacements
        }
        self._substituted_expert_to_row = {
            expert_id: row for row, expert_id in enumerate(self.substituted_expert_ids)
        }

        logical_to_physical = torch.tensor(
            layout.logical_to_physical, dtype=torch.int32
        )
        substitution_index = torch.full(
            (self.num_logical_experts,), -1, dtype=torch.int32
        )
        for expert_id, row in self._substituted_expert_to_row.items():
            substitution_index[expert_id] = row
        # Routing metadata is independent of checkpoint parameters. Keeping it
        # separate lets layerwise reload preserve these buffers unchanged.
        self._routing = nn.Module()
        self._routing.register_buffer(
            "logical_to_physical", logical_to_physical, persistent=False
        )
        self._routing.register_buffer(
            "substitution_index", substitution_index, persistent=False
        )

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.values = nn.Parameter(
            torch.full(
                (len(self.substituted_expert_ids), hidden_size),
                torch.nan,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        self.values.weight_loader = self.weight_loader
        self._layerwise_load: tuple[nn.Parameter, set[int]] | None = None

    @property
    def logical_to_physical(self) -> torch.Tensor:
        return self._routing.logical_to_physical

    @property
    def substitution_index(self) -> torch.Tensor:
        return self._routing.substitution_index

    @property
    def compute_expert_ids(self) -> tuple[int, ...]:
        return self._compute_expert_ids

    def load_value(
        self,
        loaded_weight: torch.Tensor,
        expert_id: int,
    ) -> None:
        self.values.weight_loader(self.values, loaded_weight, expert_id)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        expert_id: int,
    ) -> None:
        row = self._substituted_expert_to_row.get(int(expert_id))
        if row is None:
            raise ValueError(
                f"logical expert {expert_id} is not a constant substitution"
            )
        target = param.data[row]
        if loaded_weight.ndim != 1 or loaded_weight.shape[0] != target.shape[0]:
            raise ValueError(
                f"substitution value for expert {expert_id} has shape "
                f"{tuple(loaded_weight.shape)}, expected {(target.shape[0],)}"
            )
        if param.is_meta:
            if self._layerwise_load is None or self._layerwise_load[0] is not param:
                self._layerwise_load = (param, set())
            self._layerwise_load[1].add(expert_id)
        elif self._layerwise_load is not None:
            missing = set(self.substituted_expert_ids) - self._layerwise_load[1]
            if missing:
                raise ValueError(
                    "layerwise reload requires all constant expert value tensors "
                    f"for {self.target.module_path!r}; missing expert IDs: "
                    f"{sorted(missing)}"
                )
            self._layerwise_load = None
        target.copy_(
            loaded_weight.reshape_as(target).to(
                device=target.device, dtype=target.dtype
            )
        )

    def validate_loaded_values(self, prefix: str) -> None:
        missing_rows = torch.isnan(self.values).any(dim=1).nonzero().flatten().tolist()
        missing = {self.substituted_expert_ids[row] for row in missing_rows}
        if missing:
            preview = sorted(missing)[:8]
            raise ValueError(
                f"{prefix} is missing {len(missing)} constant expert value "
                f"tensor(s), first missing logical expert IDs: {preview}"
            )

    def transform_routes(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.values.size(-1) > hidden_states.size(-1):
            raise ValueError(
                "constant expert hidden size exceeds the routed output size: "
                f"{self.values.size(-1)} > {hidden_states.size(-1)}"
            )
        substitution_output = torch.zeros_like(hidden_states)
        if topk_ids.numel() == 0:
            return topk_weights, topk_ids, substitution_output

        valid = (topk_ids >= 0) & (topk_ids < self.num_logical_experts)
        safe_ids = torch.where(valid, topk_ids, torch.zeros_like(topk_ids)).long()
        substitution_rows = self.substitution_index[safe_ids].long()
        substitution_mask = valid & (substitution_rows >= 0)
        _compute_constant_substitution_output(
            substitution_output,
            self.values,
            substitution_rows,
            substitution_mask,
            topk_weights,
        )

        physical_ids = self.logical_to_physical[safe_ids]
        topk_weights.masked_fill_(~(valid & (physical_ids >= 0)), 0.0)
        topk_ids.copy_(physical_ids.clamp_min(0).to(topk_ids.dtype))

        # Backends consume valid compact physical IDs; substituted slots remain
        # only as zero-weight placeholders.
        return topk_weights, topk_ids, substitution_output

    def make_expert_params_mapping(
        self,
        moe_prefix: str,
        ckpt_gate_proj_name: str,
        ckpt_down_proj_name: str,
        ckpt_up_proj_name: str,
        ckpt_prefix: str | None = None,
        routed_experts_prefix: str = "routed_experts",
        base_layer: str = "",
    ) -> list[tuple[str, str, int, str]]:
        ckpt_prefix = moe_prefix if ckpt_prefix is None else ckpt_prefix
        runtime_prefix = (
            f"{moe_prefix}.{routed_experts_prefix}."
            if routed_experts_prefix
            else f"{moe_prefix}."
        )
        return [
            (
                f"{runtime_prefix}{base_layer}w13_"
                if weight_name in (ckpt_gate_proj_name, ckpt_up_proj_name)
                else f"{runtime_prefix}{base_layer}w2_",
                f"{ckpt_prefix}.{logical_expert_id}.{weight_name}.{base_layer}",
                physical_expert_id,
                shard_id,
            )
            for physical_expert_id, logical_expert_id in enumerate(
                self.compute_expert_ids
            )
            for shard_id, weight_name in (
                ("w1", ckpt_gate_proj_name),
                ("w2", ckpt_down_proj_name),
                ("w3", ckpt_up_proj_name),
            )
        ]


def make_expert_substitution(
    config: Any,
    module_path: str,
    num_logical_experts: int,
    hidden_size: int,
    params_dtype: torch.dtype | None = None,
) -> ConstantExpertSubstitution | None:
    substitution_config = parse_expert_substitution_config(config)
    if substitution_config is None:
        return None
    target = substitution_config.get_target(module_path)
    if target is None:
        return None
    if target.num_logical_experts != num_logical_experts:
        raise ValueError(
            f"expert substitution target {module_path!r} declares "
            f"{target.num_logical_experts} logical experts, but the model has "
            f"{num_logical_experts}"
        )
    return ConstantExpertSubstitution(
        target=target,
        hidden_size=hidden_size,
        params_dtype=params_dtype,
    )


def validate_expert_substitution_model(
    config: Any, module: nn.Module, prefix: str = ""
) -> set[str]:
    """Validate local bindings and return targets owned by other pipeline stages."""
    from vllm.model_executor.models.utils import PPMissingLayer, StageMissingLayer

    substitution_config = parse_expert_substitution_config(config)
    if substitution_config is None:
        return set()

    matched = Counter(
        child.target.module_path
        for child in module.modules()
        if isinstance(child, ConstantExpertSubstitution)
    )
    modules = dict(module.named_modules())

    def belongs_to_missing_stage(path: str) -> bool:
        path = path.removeprefix(f"{prefix}.") if prefix else path
        while path:
            matches = (
                [modules[path]]
                if path in modules
                else [
                    child
                    for name, child in modules.items()
                    if name.endswith(f".{path}") or path.endswith(f".{name}")
                ]
            )
            if matches:
                return len(matches) == 1 and isinstance(
                    matches[0], (PPMissingLayer, StageMissingLayer)
                )
            path = path.rpartition(".")[0]
        return False

    remote = {
        target.module_path
        for target in substitution_config.targets
        if belongs_to_missing_stage(target.module_path)
    }
    missing = sorted(
        target.module_path
        for target in substitution_config.targets
        if matched[target.module_path] == 0 and target.module_path not in remote
    )
    duplicate = sorted(path for path, count in matched.items() if count > 1)
    if missing or duplicate:
        details = []
        if missing:
            details.append(f"unmatched targets: {missing}")
        if duplicate:
            details.append(f"targets matched more than once: {duplicate}")
        raise ValueError(
            "invalid expert_substitution model binding; " + "; ".join(details)
        )
    return remote


def intercept_expert_substitution_weights(
    module: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    ignored_tensor_names: set[str] | None = None,
) -> tuple[Iterator[tuple[str, torch.Tensor]], set[str]]:
    """Load explicitly named substitution tensors before model-specific loaders."""
    mappings: dict[str, list[tuple[str, ConstantExpertSubstitution, int]]] = {}
    for module_name, child in module.named_modules():
        if not isinstance(child, ConstantExpertSubstitution):
            continue
        param_name = f"{module_name}.values" if module_name else "values"
        for expert_id, tensor_name in child._value_tensor_names.items():
            mappings.setdefault(tensor_name, []).append((param_name, child, expert_id))

    loaded_params: set[str] = set()

    def remaining_weights() -> Iterator[tuple[str, torch.Tensor]]:
        for name, loaded_weight in weights:
            substitution_mappings = mappings.get(name)
            if substitution_mappings is None:
                if ignored_tensor_names is None or name not in ignored_tensor_names:
                    yield name, loaded_weight
                continue
            for param_name, substitution, expert_id in substitution_mappings:
                substitution.load_value(loaded_weight, expert_id)
                loaded_params.add(param_name)

    return remaining_weights(), loaded_params


def validate_expert_substitution_weights_loaded(module: nn.Module) -> None:
    """Ensure every declared substitution supplied its constant value."""
    for name, child in module.named_modules():
        if isinstance(child, ConstantExpertSubstitution):
            child.validate_loaded_values(name or child.__class__.__name__)


def make_substituted_expert_params_mapping(
    model: nn.Module,
    ckpt_gate_proj_name: str,
    ckpt_down_proj_name: str,
    ckpt_up_proj_name: str,
) -> list[tuple[str, str, int, str]]:
    """Build per-layer mappings for models containing compact expert layouts."""
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.fused_moe.substituted_routed_experts import (
        SubstitutedRoutedExperts,
    )

    mapping: list[tuple[str, str, int, str]] = []
    for module_name, module in model.named_modules():
        if not isinstance(module, RoutedExperts):
            continue
        if not module_name.endswith(".routed_experts"):
            raise ValueError(
                "expert substitution runtime module path "
                f"{module_name!r} must end with '.routed_experts'"
            )
        moe_prefix = module_name.removesuffix(".routed_experts")
        if isinstance(module, SubstitutedRoutedExperts):
            substitution = module.expert_substitution
            target_prefix = substitution.target.module_path
            if target_prefix == moe_prefix or target_prefix.endswith(f".{moe_prefix}"):
                ckpt_prefix = moe_prefix
            elif moe_prefix.endswith(f".{target_prefix}"):
                ckpt_prefix = target_prefix
            else:
                raise ValueError(
                    "expert substitution target path "
                    f"{target_prefix!r} does not match runtime MoE module "
                    f"{moe_prefix!r}; expected one path to be an "
                    "unambiguous suffix of the other"
                )
            mapping.extend(
                substitution.make_expert_params_mapping(
                    moe_prefix=moe_prefix,
                    ckpt_prefix=ckpt_prefix,
                    ckpt_gate_proj_name=ckpt_gate_proj_name,
                    ckpt_down_proj_name=ckpt_down_proj_name,
                    ckpt_up_proj_name=ckpt_up_proj_name,
                )
            )
            continue

        layer_prefix = moe_prefix.removesuffix(".experts")
        mapping.extend(
            (
                f"{layer_prefix}.{param_name}",
                f"{layer_prefix}.{weight_name}",
                expert_id,
                shard_id,
            )
            for param_name, weight_name, expert_id, shard_id in (
                RoutedExperts.build_expert_params_mapping(
                    ckpt_gate_proj_name,
                    ckpt_down_proj_name,
                    ckpt_up_proj_name,
                    num_experts=module.moe_config.num_logical_experts,
                    routed_experts_prefix="routed_experts",
                )
            )
        )
    return mapping
