# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from torch import nn
from typing_extensions import Self

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


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
class ConstantExpertReplacementSpec:
    logical_expert_id: int
    value_tensor: str


@dataclass(frozen=True)
class ExpertSubstitutionTarget:
    module_path: str
    num_logical_experts: int
    replacements: tuple[ConstantExpertReplacementSpec, ...]

    @property
    def replacement_expert_ids(self) -> tuple[int, ...]:
        return tuple(spec.logical_expert_id for spec in self.replacements)


@dataclass(frozen=True)
class ExpertSubstitutionConfig:
    version: int
    targets: tuple[ExpertSubstitutionTarget, ...]

    def get_target(self, module_path: str) -> ExpertSubstitutionTarget | None:
        return next(
            (target for target in self.targets if target.module_path == module_path),
            None,
        )


@dataclass(frozen=True)
class ExpertLayout:
    """Map stable logical expert IDs to compact physical MLP rows."""

    num_logical_experts: int
    compute_expert_ids: tuple[int, ...]
    replacement_expert_ids: tuple[int, ...]
    logical_to_physical: tuple[int, ...]

    @classmethod
    def from_replacements(
        cls,
        num_logical_experts: int,
        replacement_expert_ids: Sequence[int],
    ) -> "ExpertLayout":
        if num_logical_experts <= 0:
            raise ValueError("num_logical_experts must be positive")

        replacement_ids = tuple(sorted(int(i) for i in replacement_expert_ids))
        if not replacement_ids:
            raise ValueError("replacement expert IDs must not be empty")
        if len(set(replacement_ids)) != len(replacement_ids):
            raise ValueError("replacement expert IDs must be unique")
        invalid = [i for i in replacement_ids if not 0 <= i < num_logical_experts]
        if invalid:
            raise ValueError(
                "replacement expert IDs are outside the logical expert range: "
                f"{invalid[:8]}"
            )

        replacement_set = set(replacement_ids)
        compute_ids = tuple(
            i for i in range(num_logical_experts) if i not in replacement_set
        )
        if not compute_ids:
            raise ValueError("at least one full compute expert must be retained")

        physical_ids = {logical_id: row for row, logical_id in enumerate(compute_ids)}
        return cls(
            num_logical_experts=num_logical_experts,
            compute_expert_ids=compute_ids,
            replacement_expert_ids=replacement_ids,
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
        replacements: list[ConstantExpertReplacementSpec] = []
        seen_expert_ids: set[int] = set()
        for raw_expert_id, replacement in target.replacements.items():
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
            value_tensor = replacement.tensors.value
            replacements.append(ConstantExpertReplacementSpec(expert_id, value_tensor))

        replacements.sort(key=lambda spec: spec.logical_expert_id)
        ExpertLayout.from_replacements(
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


@triton.jit
def _constant_expert_output_kernel(
    top_k: tl.constexpr,
    expert_indices_ptr: tl.tensor,
    expert_scales_ptr: tl.tensor,
    replacement_index_ptr: tl.tensor,
    replacement_values_ptr: tl.tensor,
    output_ptr: tl.tensor,
    num_tokens: int,
    hidden_dim: int,
    value_hidden_dim: int,
    num_logical_experts: int,
    indices_stride: int,
    indices_stride_k: int,
    scales_stride: int,
    scales_stride_k: int,
    values_stride_e: int,
    values_stride_h: int,
    output_stride: int,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    num_dim_blocks = tl.cdiv(hidden_dim, BLOCK_SIZE)
    token_id = pid // num_dim_blocks
    dim_offset = (pid % num_dim_blocks) * BLOCK_SIZE
    offsets = dim_offset + tl.arange(0, BLOCK_SIZE)

    result = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for route_id in range(top_k):
        expert_id = tl.load(
            expert_indices_ptr
            + token_id * indices_stride
            + route_id * indices_stride_k,
            mask=token_id < num_tokens,
            other=-1,
        ).to(tl.int64)
        valid_expert = (expert_id >= 0) & (expert_id < num_logical_experts)
        safe_expert_id = tl.where(valid_expert, expert_id, 0)
        replacement_row = tl.load(
            replacement_index_ptr + safe_expert_id,
            mask=valid_expert,
            other=-1,
        ).to(tl.int64)
        is_replacement = replacement_row >= 0
        safe_replacement_row = tl.where(is_replacement, replacement_row, 0)

        scale = tl.load(
            expert_scales_ptr + token_id * scales_stride + route_id * scales_stride_k,
            mask=(token_id < num_tokens) & is_replacement,
            other=0.0,
        ).to(tl.float32)
        values = tl.load(
            replacement_values_ptr
            + safe_replacement_row * values_stride_e
            + offsets * values_stride_h,
            mask=is_replacement & (offsets < value_hidden_dim),
            other=0.0,
        ).to(tl.float32)
        result += values * scale

    tl.store(
        output_ptr + token_id * output_stride + offsets,
        result,
        mask=(token_id < num_tokens) & (offsets < hidden_dim),
    )


@triton.jit
def _mask_non_compute_expert_scales_kernel(
    expert_indices_ptr: tl.tensor,
    expert_scales_ptr: tl.tensor,
    expert_map_ptr: tl.tensor,
    num_routes: int,
    top_k: tl.constexpr,
    num_logical_experts: int,
    indices_stride: int,
    indices_stride_k: int,
    scales_stride: int,
    scales_stride_k: int,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    route_offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    route_mask = route_offsets < num_routes
    token_id = route_offsets // top_k
    topk_id = route_offsets - token_id * top_k

    expert_id = tl.load(
        expert_indices_ptr + token_id * indices_stride + topk_id * indices_stride_k,
        mask=route_mask,
        other=-1,
    ).to(tl.int64)
    valid_expert = (expert_id >= 0) & (expert_id < num_logical_experts)
    safe_expert_id = tl.where(valid_expert, expert_id, 0)
    physical_expert_id = tl.load(
        expert_map_ptr + safe_expert_id,
        mask=valid_expert,
        other=-1,
    )
    tl.store(
        expert_scales_ptr + token_id * scales_stride + topk_id * scales_stride_k,
        0.0,
        mask=route_mask & (physical_expert_id < 0),
    )


class ConstantExpertReplacement(nn.Module):
    """Execute the homogeneous ``constant-v1`` substitution format."""

    def __init__(
        self,
        target: ExpertSubstitutionTarget,
        hidden_size: int,
        params_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.target = target
        self.layout = ExpertLayout.from_replacements(
            target.num_logical_experts, target.replacement_expert_ids
        )
        self.num_logical_experts = self.layout.num_logical_experts
        self.num_compute_experts = len(self.layout.compute_expert_ids)
        self._compute_expert_ids = self.layout.compute_expert_ids
        self.replacement_expert_ids = self.layout.replacement_expert_ids
        self._value_tensor_names = {
            spec.logical_expert_id: spec.value_tensor for spec in target.replacements
        }
        self._replacement_expert_to_row = {
            expert_id: row for row, expert_id in enumerate(self.replacement_expert_ids)
        }

        logical_to_physical = torch.tensor(
            self.layout.logical_to_physical, dtype=torch.int32
        )
        replacement_index = torch.full(
            (self.num_logical_experts,), -1, dtype=torch.int32
        )
        for expert_id, row in self._replacement_expert_to_row.items():
            replacement_index[expert_id] = row
        self.register_buffer(
            "logical_to_physical", logical_to_physical, persistent=False
        )
        self.register_buffer("replacement_index", replacement_index, persistent=False)

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.values = nn.Parameter(
            torch.full(
                (len(self.replacement_expert_ids), hidden_size),
                torch.nan,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        self.values.weight_loader = self.weight_loader

    @property
    def compute_expert_ids(self) -> tuple[int, ...]:
        return self._compute_expert_ids

    @property
    def expert_map(self) -> torch.Tensor:
        return self.logical_to_physical

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str | None = None,
        shard_id: str | None = None,
        expert_id: int | None = None,
        return_success: bool = False,
        **_: Any,
    ) -> bool | None:
        if expert_id is None:
            raise ValueError("expert_id is required when loading a replacement value")
        row = self._replacement_expert_to_row.get(int(expert_id))
        if row is None:
            raise ValueError(
                f"logical expert {expert_id} is not a constant replacement"
            )
        target = param.data[row]
        if loaded_weight.ndim != 1 or loaded_weight.shape[0] != target.shape[0]:
            raise ValueError(
                f"replacement value for expert {expert_id} has shape "
                f"{tuple(loaded_weight.shape)}, expected {(target.shape[0],)}"
            )
        target.copy_(
            loaded_weight.reshape_as(target).to(
                device=target.device, dtype=target.dtype
            )
        )
        return True if return_success else None

    def clear_loaded_values(self) -> None:
        self.values.data.fill_(torch.nan)

    def validate_loaded_values(self, prefix: str) -> None:
        missing_rows = torch.isnan(self.values).any(dim=1).nonzero().flatten().tolist()
        missing = {self.replacement_expert_ids[row] for row in missing_rows}
        if missing:
            preview = sorted(missing)[:8]
            raise ValueError(
                f"{prefix} is missing {len(missing)} constant expert value "
                f"tensor(s), first missing logical expert IDs: {preview}"
            )

    def _compute_replacement_output_torch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.zeros_like(hidden_states)
        if topk_ids.numel() == 0:
            return output
        valid = (topk_ids >= 0) & (topk_ids < self.num_logical_experts)
        safe_ids = torch.where(valid, topk_ids, torch.zeros_like(topk_ids)).long()
        replacement_rows = self.replacement_index[safe_ids].long()
        replacement_mask = valid & (replacement_rows >= 0)
        gathered = self.values[replacement_rows.clamp_min(0)]
        scales = topk_weights.to(gathered.dtype) * replacement_mask.to(gathered.dtype)
        result = torch.sum(gathered * scales.unsqueeze(-1), dim=1)
        output[..., : result.shape[-1]].copy_(result.to(output.dtype))
        return output

    def _compute_replacement_output(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self.values.size(-1) > hidden_states.size(-1):
            raise ValueError(
                "constant expert hidden size exceeds the routed output size: "
                f"{self.values.size(-1)} > {hidden_states.size(-1)}"
            )
        if (
            not hidden_states.is_cuda
            or not current_platform.is_cuda_alike()
            or topk_ids.numel() == 0
        ):
            return self._compute_replacement_output_torch(
                hidden_states, topk_weights, topk_ids
            )

        output = torch.empty_like(hidden_states)
        num_tokens = hidden_states.size(0)
        hidden_dim = hidden_states.size(-1)
        if num_tokens == 0 or hidden_dim == 0:
            return output
        top_k = topk_ids.size(-1)
        grid = lambda meta: (num_tokens * triton.cdiv(hidden_dim, meta["BLOCK_SIZE"]),)
        _constant_expert_output_kernel[grid](
            top_k,
            topk_ids,
            topk_weights,
            self.replacement_index,
            self.values,
            output,
            num_tokens,
            hidden_dim,
            self.values.size(-1),
            self.num_logical_experts,
            topk_ids.stride(0),
            topk_ids.stride(1),
            topk_weights.stride(0),
            topk_weights.stride(1),
            self.values.stride(0),
            self.values.stride(1),
            output.stride(0),
            BLOCK_SIZE=256,
        )
        return output

    def transform_routes(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        replacement_output = self._compute_replacement_output(
            hidden_states, topk_weights, topk_ids
        )

        if topk_ids.numel() > 0:
            if topk_ids.is_cuda and current_platform.is_cuda_alike():
                num_routes = topk_ids.numel()
                top_k = topk_ids.size(-1)
                grid = lambda meta: (triton.cdiv(num_routes, meta["BLOCK_SIZE"]),)
                _mask_non_compute_expert_scales_kernel[grid](
                    topk_ids,
                    topk_weights,
                    self.logical_to_physical,
                    num_routes,
                    top_k,
                    self.num_logical_experts,
                    topk_ids.stride(0),
                    topk_ids.stride(1),
                    topk_weights.stride(0),
                    topk_weights.stride(1),
                    BLOCK_SIZE=256,
                )
            else:
                valid = (topk_ids >= 0) & (topk_ids < self.num_logical_experts)
                safe_ids = torch.where(
                    valid, topk_ids, torch.zeros_like(topk_ids)
                ).long()
                physical_ids = self.logical_to_physical[safe_ids]
                topk_weights.masked_fill_(~(valid & (physical_ids >= 0)), 0.0)

        # Keep logical IDs: the expert map translates retained routes and marks
        # replacement routes invalid so the backend omits their GEMMs entirely.
        return topk_weights, topk_ids, replacement_output

    def make_expert_params_mapping(
        self,
        moe_prefix: str,
        ckpt_gate_proj_name: str,
        ckpt_down_proj_name: str,
        ckpt_up_proj_name: str,
        ckpt_prefix: str | None = None,
        checkpoint_prefix_to_strip: str = "",
        routed_experts_prefix: str = "routed_experts",
        base_layer: str = "",
    ) -> list[tuple[str, str, int, str]]:
        ckpt_prefix = moe_prefix if ckpt_prefix is None else ckpt_prefix
        runtime_prefix = (
            f"{moe_prefix}.{routed_experts_prefix}."
            if routed_experts_prefix
            else f"{moe_prefix}."
        )
        mapping = [
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
        mapping.extend(
            (
                f"{runtime_prefix}expert_replacement.values",
                self._strip_checkpoint_prefix(
                    self._value_tensor_names[logical_expert_id],
                    checkpoint_prefix_to_strip,
                ),
                logical_expert_id,
                "constant",
            )
            for logical_expert_id in self.replacement_expert_ids
        )
        return mapping

    @staticmethod
    def _strip_checkpoint_prefix(tensor_name: str, prefix: str) -> str:
        if not prefix:
            return tensor_name
        if not tensor_name.startswith(prefix):
            raise ValueError(
                f"replacement tensor {tensor_name!r} does not start with "
                f"the expected checkpoint prefix {prefix!r}"
            )
        return tensor_name.removeprefix(prefix)


def make_expert_replacement(
    config: Any,
    module_path: str,
    num_logical_experts: int,
    hidden_size: int,
    params_dtype: torch.dtype | None = None,
) -> ConstantExpertReplacement | None:
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
    if isinstance(params_dtype, str):
        params_dtype = getattr(torch, params_dtype.removeprefix("torch."))
    if not isinstance(params_dtype, torch.dtype):
        params_dtype = torch.get_default_dtype()
    return ConstantExpertReplacement(
        target=target,
        hidden_size=hidden_size,
        params_dtype=params_dtype,
    )


def validate_expert_substitution_targets(
    config: Any, supported_module_paths: set[str]
) -> None:
    substitution_config = parse_expert_substitution_config(config)
    if substitution_config is None:
        return
    unsupported = sorted(
        target.module_path
        for target in substitution_config.targets
        if target.module_path not in supported_module_paths
    )
    if unsupported:
        raise ValueError(
            "expert_substitution contains targets that are not supported by "
            f"this model: {unsupported}"
        )


def clear_expert_substitution_load_state(module: nn.Module) -> None:
    """Clear replacement load tracking before a checkpoint load."""
    for child in module.modules():
        if isinstance(child, ConstantExpertReplacement):
            child.clear_loaded_values()


def validate_expert_substitution_weights_loaded(module: nn.Module) -> None:
    """Ensure every declared replacement supplied its constant value."""
    for name, child in module.named_modules():
        if isinstance(child, ConstantExpertReplacement):
            child.validate_loaded_values(name or child.__class__.__name__)
