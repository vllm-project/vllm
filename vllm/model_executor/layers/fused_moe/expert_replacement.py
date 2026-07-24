# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


def get_mone_expert_ids(config: Any, layer_idx: int) -> tuple[int, ...]:
    """Read constant novice expert IDs from versioned or legacy metadata."""
    metadata = getattr(config, "mone", None)
    approximate_experts = None
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("config.mone must be a metadata dictionary")
    if isinstance(metadata, dict):
        version = metadata.get("version", 1)
        if version != 1:
            raise ValueError(
                f"unsupported config.mone metadata version {version!r}; expected 1"
            )
        replacement_type = metadata.get("replacement_type", "constant")
        if replacement_type != "constant":
            raise ValueError(
                "vLLM currently supports only constant MoNE replacements, "
                f"but replacement_type={replacement_type!r} was requested"
            )
        approximate_experts = metadata.get(
            "experts_by_layer", metadata.get("approximate_experts")
        )

    if approximate_experts is None:
        approximate_experts = getattr(config, "approximate_experts", None)
    if not approximate_experts:
        return ()

    if isinstance(approximate_experts, dict):
        layer_experts = approximate_experts.get(layer_idx)
        if layer_experts is None:
            layer_experts = approximate_experts.get(str(layer_idx))
    else:
        if layer_idx >= len(approximate_experts):
            return ()
        layer_experts = approximate_experts[layer_idx]

    if not layer_experts:
        return ()
    return tuple(int(expert_id) for expert_id in layer_experts)


class ExpertReplacement(nn.Module, ABC):
    """Transform logical routed experts into a compact compute layout."""

    num_logical_experts: int
    num_compute_experts: int

    @property
    @abstractmethod
    def compute_expert_ids(self) -> tuple[int, ...]:
        """Return retained logical expert IDs in physical-row order."""
        raise NotImplementedError

    @property
    @abstractmethod
    def expert_map(self) -> torch.Tensor:
        """Map logical routed IDs to physical rows; negative rows are skipped."""
        raise NotImplementedError

    @abstractmethod
    def transform_routes(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return compute weights, logical IDs, and replacement output."""
        raise NotImplementedError

    @abstractmethod
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
        """Map checkpoint logical expert weights into compact parameters."""
        raise NotImplementedError


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


class ConstantExpertReplacement(ExpertReplacement):
    """Replace selected logical experts with weighted constant vectors."""

    def __init__(
        self,
        num_logical_experts: int,
        replacement_expert_ids: Sequence[int],
        hidden_size: int,
        params_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if num_logical_experts <= 0:
            raise ValueError("num_logical_experts must be positive")

        replacement_ids = tuple(sorted(int(i) for i in replacement_expert_ids))
        if not replacement_ids:
            raise ValueError("replacement expert IDs must not be empty")
        if len(set(replacement_ids)) != len(replacement_ids):
            raise ValueError("replacement expert IDs must be unique")
        invalid = [
            expert_id
            for expert_id in replacement_ids
            if expert_id < 0 or expert_id >= num_logical_experts
        ]
        if invalid:
            raise ValueError(
                "replacement expert IDs are outside the logical expert range: "
                f"{invalid[:8]}"
            )

        replacement_set = set(replacement_ids)
        compute_ids = tuple(
            expert_id
            for expert_id in range(num_logical_experts)
            if expert_id not in replacement_set
        )
        if not compute_ids:
            raise ValueError("at least one full compute expert must be retained")

        self.num_logical_experts = num_logical_experts
        self.num_compute_experts = len(compute_ids)
        self._compute_expert_ids = compute_ids
        self.replacement_expert_ids = replacement_ids
        self._replacement_expert_to_row = {
            expert_id: row for row, expert_id in enumerate(replacement_ids)
        }

        logical_to_physical = torch.full((num_logical_experts,), -1, dtype=torch.int32)
        replacement_index = torch.full((num_logical_experts,), -1, dtype=torch.int32)
        for row, expert_id in enumerate(compute_ids):
            logical_to_physical[expert_id] = row
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
                (len(replacement_ids), hidden_size),
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
            raise ValueError("expert_id is required when loading a novice value")
        row = self._replacement_expert_to_row.get(int(expert_id))
        if row is None:
            raise ValueError(
                f"logical expert {expert_id} is not declared as a constant novice"
            )
        target = param.data[row]
        if loaded_weight.numel() != target.numel():
            raise ValueError(
                f"novice value for expert {expert_id} has {loaded_weight.numel()} "
                f"elements, expected {target.numel()}"
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
                f"{ckpt_prefix}.{logical_expert_id}.approx_value",
                logical_expert_id,
                "constant",
            )
            for logical_expert_id in self.replacement_expert_ids
        )
        return mapping


def make_mone_replacement(
    config: Any,
    layer_idx: int,
    num_logical_experts: int,
    hidden_size: int,
    params_dtype: torch.dtype | None = None,
) -> ConstantExpertReplacement | None:
    replacement_ids = get_mone_expert_ids(config, layer_idx)
    if not replacement_ids:
        return None
    if isinstance(params_dtype, str):
        params_dtype = getattr(torch, params_dtype.removeprefix("torch."))
    if not isinstance(params_dtype, torch.dtype):
        params_dtype = torch.get_default_dtype()
    return ConstantExpertReplacement(
        num_logical_experts=num_logical_experts,
        replacement_expert_ids=replacement_ids,
        hidden_size=hidden_size,
        params_dtype=params_dtype,
    )


def clear_mone_load_state(module: nn.Module) -> None:
    """Clear replacement load tracking before a checkpoint load."""
    for child in module.modules():
        if isinstance(child, ConstantExpertReplacement):
            child.clear_loaded_values()


def validate_mone_weights_loaded(module: nn.Module) -> None:
    """Ensure every declared novice expert supplied its constant value."""
    for name, child in module.named_modules():
        if isinstance(child, ConstantExpertReplacement):
            child.validate_loaded_values(name or child.__class__.__name__)


__all__ = [
    "ConstantExpertReplacement",
    "ExpertReplacement",
    "clear_mone_load_state",
    "get_mone_expert_ids",
    "make_mone_replacement",
    "validate_mone_weights_loaded",
]
