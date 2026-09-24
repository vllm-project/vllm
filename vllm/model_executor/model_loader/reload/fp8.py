# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 checkpoint-to-runtime policies; never install parameters or kernels."""

import inspect
import math
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
from functools import partial
from typing import Any, Literal

import torch

from vllm.model_executor.layers.quantization.utils.fp8_processing import (
    Fp8MoEProcessingPlan,
    Fp8MoEWeights,
)

from .trace import ReloadError, ReloadState


def _process_moe_weights(
    state: ReloadState, plan: Fp8MoEProcessingPlan
) -> Fp8MoEWeights:
    """Use the cold-load conversion with this round's loader-format inputs."""
    suffix = "weight_scale_inv" if plan.block_shape is not None else "weight_scale"
    return plan.process(
        Fp8MoEWeights(
            state.work("w13_weight"),
            state.work("w2_weight"),
            state.work(f"w13_{suffix}"),
            state.work(f"w2_{suffix}"),
            state.work("w13_input_scale") if "w13_input_scale" in state.roles else None,
            state.work("w2_input_scale") if "w2_input_scale" in state.roles else None,
        )
    )


def _copy_moe_weights(state: ReloadState, weights: Fp8MoEWeights) -> None:
    suffix = (
        "weight_scale_inv" if "w13_weight_scale_inv" in state.roles else "weight_scale"
    )
    for role, value in (
        ("w13_weight", weights.w13),
        ("w2_weight", weights.w2),
        (f"w13_{suffix}", weights.w13_scale),
        (f"w2_{suffix}", weights.w2_scale),
        ("w13_input_scale", weights.w13_input_scale),
        ("w2_input_scale", weights.w2_input_scale),
    ):
        if value is not None and role in state.targets:
            state.copy_(role, value)


class _CanonicalReloadPolicy:
    """Prepare a whole unit only after its first accepted local arrival."""

    def destination(
        self, state: ReloadState, role: str, bound: inspect.BoundArguments
    ) -> torch.Tensor:
        self.validate(state)
        if not state.checkpoint:
            self.prepare_for_load(state)
        return state.checkpoint[role]

    def validate(self, state: ReloadState) -> None:
        raise NotImplementedError

    def prepare_for_load(self, state: ReloadState) -> None:
        raise NotImplementedError


@dataclass
class ModelOptLinearReloadPolicy(_CanonicalReloadPolicy):
    """Reload ModelOpt tensors through the cold-selected processing plan.

    ModelOpt's cold PWAL may transpose weights or squeeze block-scale
    dimensions. Reload receives the original checkpoint layout, so conversion
    is delegated to the kernel's tensor-only processing entry point.
    """

    processing_plan_getter: Callable[[], Any | None] | None = None

    def bind(self, state: ReloadState) -> None:
        self.runtime_shapes = {
            role: tuple(target.tensor.shape) for role, target in state.targets.items()
        }
        method = state.module.quant_method
        self.processing_plan = (
            self.processing_plan_getter()
            if self.processing_plan_getter is not None
            else getattr(method, "processing_plan", None)
        )
        if self.processing_plan is None:
            raise ReloadError(
                f"{state.key}: ModelOpt processing plan was not created "
                "before runtime binding"
            )
        processing_plan = self.processing_plan
        self.kernel = processing_plan.kernel
        if self.kernel is not getattr(method, "kernel", None):
            raise ReloadError(f"{state.key}: ModelOpt processing plan kernel is stale")

    def validate(self, state: ReloadState) -> None:
        if {
            role: tuple(target.tensor.shape) for role, target in state.targets.items()
        } != self.runtime_shapes:
            raise ReloadError("ModelOpt FP8 runtime layout changed since binding")

    def prepare_for_load(self, state: ReloadState) -> None:
        state.prepare_sources(reuse_roles=())

    def _match_runtime_shape(
        self, state: ReloadState, role: str, value: torch.Tensor
    ) -> torch.Tensor:
        target = state.targets[role].tensor
        if tuple(value.shape) == self.runtime_shapes[role]:
            return value
        if role == "weight" and tuple(value.t().shape) == self.runtime_shapes[role]:
            return value.t().contiguous().t()
        if value.numel() == target.numel():
            return value.reshape(target.shape).contiguous()
        raise ReloadError(
            f"{state.key}/{role}: cannot convert checkpoint shape "
            f"{tuple(value.shape)} to runtime shape {tuple(target.shape)}"
        )

    def finish(self, state: ReloadState) -> None:
        self.validate(state)
        values = {role: state.work(role) for role in state.roles}
        processing_plan = self.processing_plan
        if processing_plan is None:
            raise ReloadError(f"{state.key}: ModelOpt processing plan is unavailable")
        converted = processing_plan.process_reload_tensors(state.module, values)
        for role, value in converted.items():
            if role in state.targets:
                state.copy_(role, value)


@dataclass
class TensorFP8LinearReloadPolicy(_CanonicalReloadPolicy):
    """Requantize checkpoint shards into a tensor-scaled linear runtime."""

    cutlass_padding: bool = False

    def bind(self, state: ReloadState) -> None:
        self.kernel = state.module.quant_method.fp8_linear
        self.logical_widths = tuple(state.module.logical_widths)
        self.logical_output_size = getattr(self.kernel, "logical_output_size", None)

    def validate(self, state: ReloadState) -> None:
        if (
            state.module.quant_method.fp8_linear is not self.kernel
            or tuple(state.module.logical_widths) != self.logical_widths
            or getattr(self.kernel, "logical_output_size", None)
            != self.logical_output_size
        ):
            raise ReloadError("Tensor FP8 linear kernel/layout changed since binding")

    def prepare_for_load(self, state: ReloadState) -> None:
        # NK loading views borrow KN/padded storage; reduced scales need staging.
        state.prepare_sources(reuse_roles=("weight", "bias"))

    def _convert(
        self, state: ReloadState
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            process_fp8_weight_tensor_strategy,
        )

        weight, scale, input_scale = process_fp8_weight_tensor_strategy(
            state.work("weight"),
            state.work("weight_scale"),
            list(self.logical_widths),
            state.work("input_scale") if "input_scale" in state.roles else None,
        )
        weight = weight.t()
        if self.cutlass_padding:
            pad_k = (-weight.shape[0]) % 16
            pad_n = (-weight.shape[1]) % 16
            if pad_k or pad_n:
                weight = torch.nn.functional.pad(
                    weight.t().contiguous(), (0, pad_k, 0, pad_n)
                ).t()
                if pad_n and scale.numel() > 1:
                    scale = torch.nn.functional.pad(
                        scale.reshape(-1), (0, (-scale.numel()) % 16), value=1.0
                    ).view(-1, *scale.shape[1:])
        return weight, scale, input_scale

    def finish(self, state: ReloadState) -> None:
        self.validate(state)
        weight, scale, input_scale = self._convert(state)
        state.copy_("weight", weight)
        state.copy_("weight_scale", scale)
        if "input_scale" in state.roles:
            assert input_scale is not None
            state.copy_("input_scale", input_scale.max())
        if "bias" in state.roles:
            state.copy_("bias", state.work("bias"))


@dataclass
class HummingFP8LinearReloadPolicy(TensorFP8LinearReloadPolicy):
    """Replay the cold processing plan without constructing a layer or config."""

    block_quant: bool = False

    def bind(self, state: ReloadState) -> None:
        super().bind(state)
        layer = state.module
        self.layer_config = self.kernel.layer_config
        self.plan = self.kernel.processing_plan
        self.compute_config = self.kernel.compute_config
        self.weight_schema = layer.weight_schema
        self.parameters = tuple(dict(layer.named_parameters(recurse=False)))
        self.layer_metadata = {
            name: deepcopy(getattr(layer, name))
            for name in (
                "output_partition_sizes",
                "input_size_per_partition",
                "params_dtype",
                "has_bias",
                "weight_block_size",
            )
        }
        for name in self.parameters:
            state.bind_target(f"humming.{name}", partial(getattr, layer, name))
        state.bind_target("humming_locks", partial(getattr, self.kernel, "locks"))

    def validate(self, state: ReloadState) -> None:
        super().validate(state)
        if (
            self.kernel.layer_config is not self.layer_config
            or self.kernel.processing_plan is not self.plan
            or self.plan.conversion.config is not self.layer_config
            or self.kernel.compute_config != self.compute_config
            or state.module.weight_schema is not self.weight_schema
            or any(
                getattr(state.module, name) != value
                for name, value in self.layer_metadata.items()
            )
        ):
            raise ReloadError("Humming linear configuration changed since binding")

    def prepare_for_load(self, state: ReloadState) -> None:
        # Packed int32 weights have no same-dtype FP8 loading view. Scales
        # (including renamed block scales) and bias may lend dense storage.
        # Missing, encoded or undersized targets automatically use staging.
        state.prepare_sources(
            reuse_roles=tuple(role for role in state.roles if role != "weight")
        )

    def finish(self, state: ReloadState) -> None:
        self.validate(state)
        if self.block_quant:
            values = {role: state.work(role) for role in state.roles}
        else:
            weight, scale, input_scale = self._convert(state)
            values = {"weight": weight, "weight_scale": scale}
            if input_scale is not None:
                values["input_scale"] = input_scale.max()
            if "bias" in state.roles:
                values["bias"] = state.work("bias")
        converted = self.plan.process(values)
        if set(converted) != set(self.parameters):
            raise ReloadError("Humming conversion changed the runtime schema")
        # Validate every output before the first live write.
        for name, value in converted.items():
            target = state.targets[f"humming.{name}"].tensor
            if value.shape != target.shape or value.dtype != target.dtype:
                raise ReloadError("Humming conversion changed a runtime tensor layout")
        for name, value in converted.items():
            state.copy_(f"humming.{name}", value)


@dataclass
class B12xTensorFP8LinearReloadPolicy(TensorFP8LinearReloadPolicy):
    """Update a B12x packed dataclass while retaining its tensors and metadata."""

    def prepare_for_load(self, state: ReloadState) -> None:
        # Packed dataclass leaves have no audited canonical weight view.
        state.prepare_sources(reuse_roles=("bias",))

    @staticmethod
    def _tensor_paths(value, path=()):
        if isinstance(value, torch.Tensor):
            yield path, value
        elif is_dataclass(value):
            for field in fields(value):
                yield from B12xTensorFP8LinearReloadPolicy._tensor_paths(
                    getattr(value, field.name), (*path, field.name)
                )

    @staticmethod
    def _resolve(packed, path):
        for name in path:
            packed = getattr(packed, name)
        return packed

    @staticmethod
    def _resolve_live(module, path):
        return B12xTensorFP8LinearReloadPolicy._resolve(
            module.b12x_tensor_fp8_packed_weight, path
        )

    def bind(self, state: ReloadState) -> None:
        super().bind(state)
        self.packed = state.module.b12x_tensor_fp8_packed_weight
        self.provider = state.module.b12x_warmup_provider
        leaves = dict(self._tensor_paths(self.packed))
        if not is_dataclass(self.packed) or not leaves:
            raise ReloadError("B12x reload requires a packed tensor dataclass")
        self.paths = tuple(leaves)
        # Snapshot metadata, not a second copy of potentially large packed weights.
        self.layout = deepcopy(self.packed, {id(t): t for t in leaves.values()})
        for path in self.paths:
            state.bind_target(
                f"packed.{'.'.join(path)}",
                partial(self._resolve_live, state.module, path),
            )

    def validate(self, state: ReloadState) -> None:
        from vllm.utils.b12x import _same_packed_layout

        super().validate(state)
        if (
            state.module.b12x_tensor_fp8_packed_weight is not self.packed
            or state.module.b12x_warmup_provider is not self.provider
            or not _same_packed_layout(self.layout, self.packed)
        ):
            raise ReloadError("B12x packed object/layout changed since runtime binding")

    def finish(self, state: ReloadState) -> None:
        from vllm.utils.b12x import (
            _same_packed_layout,
            get_b12x_tensor_fp8_linear,
        )

        self.validate(state)
        weight, scale, input_scale = self._convert(state)
        if input_scale is None or scale.numel() != 1:
            raise ReloadError("B12x tensor FP8 requires static per-tensor scales")
        input_scale = input_scale.max()
        backend = get_b12x_tensor_fp8_linear()
        if backend is None:
            raise ReloadError("B12x tensor FP8 packing backend is unavailable")
        output_scale = (
            input_scale.detach().float().reshape(1) * scale.detach().float().reshape(1)
        ).contiguous()
        replacement = backend.pack_weight(
            weight.detach().t().contiguous(), output_scale
        )
        if not _same_packed_layout(self.layout, replacement):
            raise ReloadError("B12x packing produced an incompatible runtime layout")
        for path in self.paths:
            state.copy_(f"packed.{'.'.join(path)}", self._resolve(replacement, path))
        state.copy_("input_scale", input_scale)
        if "bias" in state.roles:
            state.copy_("bias", state.work("bias"))


@dataclass
class XPUTensorFP8LinearReloadPolicy(TensorFP8LinearReloadPolicy):
    """Keep XPU's KN weights and backend-specific scale dimensions."""

    weight_only: bool = False

    def _convert(
        self, state: ReloadState
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        weight, scale, input_scale = super()._convert(state)
        if self.weight_only:
            scale = scale.t().contiguous()
        elif scale.numel() == 1:
            scale = scale.reshape(1)
        return weight, scale, input_scale


@dataclass
class AiterTensorFP8LinearReloadPolicy(TensorFP8LinearReloadPolicy):
    """Reproduce AITER's NK, shuffled NK, or shuffled KN representation."""

    layout: Literal["plain", "preshuffled", "hipbmm"] = "plain"

    def _convert(
        self, state: ReloadState
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        weight, scale, input_scale = super()._convert(state)
        weight = weight.t()
        if self.layout != "plain":
            from vllm._aiter_ops import rocm_aiter_ops

            weight = rocm_aiter_ops.shuffle_weight(weight.contiguous())
            if self.layout == "hipbmm":
                weight = weight.t()
                if scale.ndim > 1:
                    scale = scale.t().contiguous()
        return weight, scale, input_scale


@dataclass
class BlockFP8LinearReloadPolicy(_CanonicalReloadPolicy):
    """Reload kernels using the standard FP8 block weight representation."""

    def bind(self, state: ReloadState) -> None:
        self.kernel = state.module.quant_method.fp8_linear

    def validate(self, state: ReloadState) -> None:
        if state.module.quant_method.fp8_linear is not self.kernel:
            raise ReloadError("Block FP8 linear kernel changed since binding")

    def prepare_for_load(self, state: ReloadState) -> None:
        state.prepare_sources(reuse_roles=state.roles)

    def _convert(self, state: ReloadState) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            process_fp8_weight_block_strategy,
        )

        return process_fp8_weight_block_strategy(
            state.work("weight"), state.work("weight_scale_inv")
        )

    def finish(self, state: ReloadState) -> None:
        self.validate(state)
        weight, scale = self._convert(state)
        state.copy_("weight", weight)
        state.copy_("weight_scale_inv", scale)
        for role in state.roles:
            if role not in ("weight", "weight_scale_inv"):
                state.copy_(role, state.work(role))


@dataclass
class CPUBlockFP8LinearReloadPolicy(BlockFP8LinearReloadPolicy):
    """Pack AMX weights without GPU padding or FP8 format normalization."""

    def bind(self, state: ReloadState) -> None:
        super().bind(state)
        self.skip_dispatch = getattr(state.module, "_cpu_skip_gemm_dispatch", False)

    def validate(self, state: ReloadState) -> None:
        super().validate(state)
        if (
            getattr(state.module, "_cpu_skip_gemm_dispatch", False)
            != self.skip_dispatch
        ):
            raise ReloadError("CPU FP8 dispatch mode changed since runtime binding")

    def prepare_for_load(self, state: ReloadState) -> None:
        state.prepare_sources(
            reuse_roles=state.roles
            if self.skip_dispatch
            else ("weight_scale_inv", "bias")
        )

    def _convert(self, state: ReloadState) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            _upcast_e8m0_to_fp32,
        )

        weight, scale = state.work("weight"), state.work("weight_scale_inv")
        if self.skip_dispatch:
            return weight, scale
        weight = torch.ops._C.convert_weight_packed(weight)
        if scale.dtype in (torch.float8_e8m0fnu, torch.uint8):
            scale = _upcast_e8m0_to_fp32(scale)
        return weight, scale.contiguous()


@dataclass
class AiterBlockFP8LinearReloadPolicy(BlockFP8LinearReloadPolicy):
    """Preserve AITER's direct-read exception and optional weight shuffle."""

    preshuffled: bool = False

    def bind(self, state: ReloadState) -> None:
        super().bind(state)
        self.direct_read = self._direct_read(state)

    def _direct_read(self, state: ReloadState) -> bool:
        return bool(
            getattr(state.module, "is_bmm", False)
            or getattr(state.module, "skip_weight_relayout", False)
        )

    def validate(self, state: ReloadState) -> None:
        super().validate(state)
        if self._direct_read(state) != self.direct_read:
            raise ReloadError(
                "AITER weight relayout mode changed since runtime binding"
            )

    def _convert(self, state: ReloadState) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            _upcast_e8m0_to_fp32,
        )

        weight, scale = super()._convert(state)
        if self.preshuffled and self.direct_read:
            return weight, scale
        if scale.dtype == torch.float8_e8m0fnu:
            scale = _upcast_e8m0_to_fp32(scale).contiguous()
        if self.preshuffled:
            from vllm._aiter_ops import rocm_aiter_ops

            weight = rocm_aiter_ops.shuffle_weight(weight.contiguous(), layout=(16, 16))
        return weight, scale


@dataclass
class XPUBlockFP8LinearReloadPolicy(BlockFP8LinearReloadPolicy):
    """Expand ragged block scales and refresh XPU BMM's cached scale tensor."""

    def bind(self, state: ReloadState) -> None:
        super().bind(state)
        self.layout = self._layout(state)
        self.block_shape, self.is_bmm, self.batch_size = self.layout
        if self.is_bmm:
            for name in ("bmm_weight", "bmm_scale"):
                state.bind_target(name, partial(getattr, state.module, name))

    def _layout(self, state: ReloadState) -> tuple:
        return (
            tuple(self.kernel.weight_group_shape),
            getattr(state.module, "is_bmm", False),
            getattr(state.module, "bmm_batch_size", 0),
        )

    def validate(self, state: ReloadState) -> None:
        super().validate(state)
        if self._layout(state) != self.layout:
            raise ReloadError("XPU block/BMM layout changed since runtime binding")

    def _convert(self, state: ReloadState) -> tuple[torch.Tensor, torch.Tensor]:
        weight, scale = super()._convert(state)
        n, k = weight.shape
        block_n, block_k = self.block_shape
        if k % block_k:
            raise ReloadError("XPU FP8 reload requires block-aligned K")
        if n % block_n:
            group_n = math.gcd(n, block_n)
            if group_n % 16:
                raise ReloadError("XPU FP8 scale groups require 16-row alignment")
            starts = torch.arange(n // group_n, device=scale.device) * group_n
            indices = torch.div(starts, block_n, rounding_mode="floor")
            scale = scale.index_select(0, indices).contiguous()
        return weight, scale.t().contiguous().t()

    def finish(self, state: ReloadState) -> None:
        self.validate(state)
        weight, scale = self._convert(state)
        bmm_weight = bmm_scale = None
        if self.is_bmm:
            scale_kn = scale.t()
            k_blocks, n_blocks = scale_kn.shape
            batch = self.batch_size
            bmm_scale = (
                scale_kn.reshape(k_blocks, batch, n_blocks // batch)
                .permute(1, 0, 2)
                .contiguous()
            )
            bmm_weight = weight.reshape(
                batch, weight.shape[0] // batch, weight.shape[1]
            ).permute(0, 2, 1)
        state.copy_("weight", weight)
        state.copy_("weight_scale_inv", scale)
        if bmm_weight is not None and bmm_scale is not None:
            state.copy_("bmm_weight", bmm_weight)
            state.copy_("bmm_scale", bmm_scale)
        if "bias" in state.roles:
            state.copy_("bias", state.work("bias"))


@dataclass
class B12xBlockFP8LinearReloadPolicy(BlockFP8LinearReloadPolicy):
    """Keep B12x's block layout and convert encoded scales to FP32."""

    def bind(self, state: ReloadState) -> None:
        super().bind(state)
        self.provider = state.module.b12x_warmup_provider

    def validate(self, state: ReloadState) -> None:
        super().validate(state)
        if state.module.b12x_warmup_provider is not self.provider:
            raise ReloadError("B12x warmup provider changed since runtime binding")

    def _convert(self, state: ReloadState) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            _upcast_e8m0_to_fp32,
        )

        weight, scale = super()._convert(state)
        if scale.dtype in (torch.float8_e8m0fnu, torch.uint8):
            scale = _upcast_e8m0_to_fp32(scale).contiguous()
        return weight, scale


@dataclass
class DeepGEMMReloadPolicy(_CanonicalReloadPolicy):
    pairs: tuple[tuple[str, str], ...]
    block_shape: tuple[int, ...]
    is_bmm: bool = False
    bmm_batch_size: int = 0

    def bind(self, state: ReloadState) -> None:
        method = state.module.quant_method
        self.kernel = getattr(method, "fp8_linear", getattr(method, "moe_kernel", None))
        self.kernel = getattr(self.kernel, "fallback", self.kernel)
        self.plan = getattr(method, "processing_plan", None)
        for weight, scale in self.pairs:
            if state.targets[weight].tensor.dtype != torch.float8_e4m3fn:
                raise ReloadError("DeepGEMM reload requires E4M3FN runtime weights")

    def validate(self, state: ReloadState) -> None:
        self._validate_kernel(state)

    def _validate_kernel(self, state: ReloadState) -> None:
        method = state.module.quant_method
        kernel = getattr(method, "fp8_linear", getattr(method, "moe_kernel", None))
        kernel = getattr(kernel, "fallback", kernel)
        if kernel is not self.kernel:
            raise ReloadError("DeepGEMM kernel changed since runtime binding")
        if getattr(method, "processing_plan", None) is not self.plan:
            raise ReloadError("DeepGEMM processing plan changed since binding")

    def prepare_for_load(self, state: ReloadState) -> None:
        # Encoded/strided scales automatically stage unless a compatible dense
        # runtime allocation can hold their canonical dtype and shape.
        state.prepare_sources(reuse_roles=state.roles)

    def finish(self, state: ReloadState) -> None:
        self._validate_kernel(state)
        if self.plan is not None:
            _copy_moe_weights(state, _process_moe_weights(state, self.plan))
            return
        assert self.kernel is not None
        for weight, scale in self.pairs:
            converted_weight, converted_scale = self.kernel.prepare_weights(
                state.work(weight),
                state.work(scale),
                self.block_shape,
                is_bmm=self.is_bmm,
                bmm_batch_size=self.bmm_batch_size,
            )
            state.copy_(weight, converted_weight)
            state.copy_(scale, converted_scale)
        converted_roles = {role for pair in self.pairs for role in pair}
        for role in state.roles:
            if role not in converted_roles:
                state.copy_(role, state.work(role))


@dataclass
class MarlinFP8LinearReloadPolicy(_CanonicalReloadPolicy):
    """Replay packing with fixed layout metadata and no workspace allocation."""

    block_quant: bool

    def bind(self, state: ReloadState) -> None:
        self.kernel = state.module.quant_method.fp8_linear
        self.plan = state.module.fp8_marlin_processing_plan
        self.input_dtype = self.kernel.marlin_input_dtype
        state.bind_target("workspace", partial(getattr, state.module, "workspace"))
        self.layout = self._layout(state)

    def _layout(self, state: ReloadState) -> tuple:
        layer = state.module
        return (
            layer.input_size_per_partition,
            layer.output_size_per_partition,
            tuple(layer.logical_widths),
            tuple(layer.weight_block_size) if self.block_quant else None,
            layer.orig_dtype,
        )

    def validate(self, state: ReloadState) -> None:
        if (
            state.module.quant_method.fp8_linear is not self.kernel
            or state.module.fp8_marlin_processing_plan is not self.plan
            or self.kernel.marlin_input_dtype != self.input_dtype
            or self._layout(state) != self.layout
        ):
            raise ReloadError("Marlin kernel/layout changed since runtime binding")

    def prepare_for_load(self, state: ReloadState) -> None:
        # Weight packing changes FP8 to int32; scales include an exponent-bias
        # conversion to the activation dtype. Do not reinterpret either one.
        # Bias permutation/padding is repeatable over a canonical loading view.
        state.prepare_sources(reuse_roles=("bias",))

    def finish(self, state: ReloadState) -> None:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            process_fp8_weight_block_strategy,
        )

        self.validate(state)
        scale_name = "weight_scale_inv" if self.block_quant else "weight_scale"
        weight, scale = state.work("weight"), state.work(scale_name)
        if self.block_quant:
            weight, scale = process_fp8_weight_block_strategy(weight, scale)
        else:
            weight = weight.t()
        weight, scale, bias = self.plan.process(
            weight,
            scale,
            state.work("bias") if "bias" in state.roles else None,
        )
        state.copy_("weight", weight)
        state.copy_(scale_name, scale)
        if bias is not None:
            state.copy_("bias", bias)


@dataclass
class PlainMoEReloadPolicy(_CanonicalReloadPolicy):
    """Reload backends whose expert tensors retain the checkpoint layout."""

    block_quant: bool
    is_act_and_mul: bool
    shard_size: int
    num_experts: int

    def bind(self, state: ReloadState) -> None:
        from vllm.platforms import current_platform

        method = state.module.quant_method
        self.kernel = method.moe_kernel
        self.config = method.moe_quant_config
        self.backend = method.fp8_backend
        self.fnuz = current_platform.is_fp8_fnuz()
        if self.kernel is None or self.config is None:
            raise ReloadError("Bind MoE reload only after kernel initialization")

    def validate(self, state: ReloadState) -> None:
        from vllm.platforms import current_platform

        method = state.module.quant_method
        if (
            method.moe_kernel is not self.kernel
            or method.moe_quant_config is not self.config
            or method.fp8_backend != self.backend
            or current_platform.is_fp8_fnuz() != self.fnuz
        ):
            raise ReloadError("MoE backend/kernel/config changed since runtime binding")

    def prepare_for_load(self, state: ReloadState) -> None:
        state.prepare_sources(reuse_roles=state.roles)

    def _checkpoint_values(self, state: ReloadState) -> dict[str, torch.Tensor]:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            process_fp8_input_tensor_strategy_moe,
            process_fp8_weight_tensor_strategy_moe,
        )

        self.validate(state)
        values = {role: state.work(role) for role in state.roles}
        if self.fnuz:
            from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
                normalize_e4m3fn_to_e4m3fnuz,
            )

            scale_name = "weight_scale_inv" if self.block_quant else "weight_scale"
            for prefix in ("w13", "w2"):
                weight, scale, input_scale = normalize_e4m3fn_to_e4m3fnuz(
                    values[f"{prefix}_weight"],
                    values[f"{prefix}_{scale_name}"],
                    values.get(f"{prefix}_input_scale"),
                )
                values[f"{prefix}_weight"] = weight
                values[f"{prefix}_{scale_name}"] = scale
                if input_scale is not None:
                    values[f"{prefix}_input_scale"] = input_scale
        if not self.block_quant:
            weight, scale = process_fp8_weight_tensor_strategy_moe(
                values["w13_weight"],
                values["w13_weight_scale"],
                self.shard_size,
                self.num_experts,
                self.is_act_and_mul,
            )
            values["w13_weight"], values["w13_weight_scale"] = weight, scale
            if "w13_input_scale" in state.roles:
                a1, a2 = process_fp8_input_tensor_strategy_moe(
                    values["w13_input_scale"],
                    values["w2_input_scale"],
                    state.module.moe_config.moe_parallel_config.enable_eplb,
                )
                values["w13_input_scale"], values["w2_input_scale"] = a1, a2
        return values

    def finish(self, state: ReloadState) -> None:
        for role, value in self._checkpoint_values(state).items():
            state.copy_(role, value)


@dataclass
class HummingMoEReloadPolicy(PlainMoEReloadPolicy):
    """Convert expert tensors with cold-bound Humming schemas and configs."""

    def bind(self, state: ReloadState) -> None:
        super().bind(state)
        layer = state.module
        self.plan = layer.fp8_humming_processing_plan
        self.moe_config = layer.moe_config
        self.humming_configs = layer.humming_configs
        self.weight_schemas = layer.weight_schemas
        self.input_schemas = layer.input_schemas
        self.layout = self._layout(layer)
        self.parameters = tuple(
            name
            for name, _ in layer.named_parameters(recurse=False)
            if name.startswith(("w13_", "w2_"))
        )
        for name in self.parameters:
            state.bind_target(f"humming.{name}", partial(getattr, layer, name))

    def _layout(self, layer) -> tuple:
        return (
            layer.moe_config.num_local_experts,
            layer.moe_config.hidden_dim,
            layer.moe_config.intermediate_size_per_partition,
            layer.moe_config.activation,
            layer.moe_config.has_bias,
            layer.params_dtype,
            tuple(layer.weight_block_size) if self.block_quant else None,
            layer.layer_name,
        )

    def validate(self, state: ReloadState) -> None:
        super().validate(state)
        layer = state.module
        if (
            layer.moe_config is not self.moe_config
            or layer.fp8_humming_processing_plan is not self.plan
            or layer.humming_configs is not self.humming_configs
            or layer.weight_schemas is not self.weight_schemas
            or layer.input_schemas is not self.input_schemas
            or self._layout(layer) != self.layout
            or any(
                self.humming_configs.get(prefix) is not plan.config
                or self.weight_schemas.get(prefix) is not plan.weight_schema
                or self.input_schemas.get(prefix) is not plan.input_schema
                for prefix, plan in self.plan.sublayers
            )
        ):
            raise ReloadError("Humming MoE configuration changed since binding")

    def prepare_for_load(self, state: ReloadState) -> None:
        # Only source roles participate: Humming's additional derived scales
        # must remain output targets, never independent checkpoint destinations.
        state.prepare_sources(
            reuse_roles=tuple(
                role for role in state.roles if role not in ("w13_weight", "w2_weight")
            )
        )

    def finish(self, state: ReloadState) -> None:
        values = self._checkpoint_values(state)
        converted = self.plan.process(values)
        if set(converted) != set(self.parameters):
            raise ReloadError("Humming MoE conversion changed the runtime schema")
        for name, value in converted.items():
            target = state.targets[f"humming.{name}"].tensor
            if value.shape != target.shape or value.dtype != target.dtype:
                raise ReloadError("Humming MoE conversion changed a tensor layout")
        for name, value in converted.items():
            state.copy_(f"humming.{name}", value)


class PlannedMoEReloadPolicy(_CanonicalReloadPolicy):
    """Reuse cold-load conversions, updating only the bound runtime tensors."""

    def bind(self, state: ReloadState) -> None:
        method = state.module.quant_method
        self.plan = method.processing_plan
        self.kernel = method.moe_kernel
        self.config = method.moe_quant_config
        self.backend = method.fp8_backend
        if self.kernel is None or self.config is None:
            raise ReloadError("Bind MoE reload only after kernel initialization")
        self.experts = self.kernel.fused_experts
        if self.plan.block_shape is None:
            if self.plan.backend == "hpc":
                for name in ("g1_alphas", "g2_alphas", "a1_gscale", "a2_gscale"):
                    state.bind_target(name, partial(getattr, self.config, name))
            elif self.plan.backend == "flashinfer_trtllm":
                for name in ("_g1_alphas", "_g2_alphas", "_g1_scale_c"):
                    state.bind_target(name, partial(getattr, self.experts, name))

    def validate(self, state: ReloadState) -> None:
        from vllm.platforms import current_platform

        method = state.module.quant_method
        if (
            method.processing_plan is not self.plan
            or method.moe_kernel is not self.kernel
            or method.moe_quant_config is not self.config
            or method.fp8_backend != self.backend
            or self.kernel.fused_experts is not self.experts
            or current_platform.is_fp8_fnuz() != self.plan.fnuz
        ):
            raise ReloadError("MoE processing plan/kernel changed since binding")
        if self.plan.backend == "aiter" and not all(
            getattr(state.targets[name].tensor, "is_shuffled", False)
            for name in ("w13_weight", "w2_weight")
        ):
            raise ReloadError("AITER runtime weights lost the shuffled layout marker")
        if self.plan.backend == "xpu":
            # XpuFusedMoe is created lazily, often after bind_runtime().
            impl = self.experts.fused_moe_impl
            if impl is not None:
                suffix = (
                    "weight_scale_inv"
                    if self.plan.block_shape is not None
                    else "weight_scale"
                )
                for name, role in (
                    ("w13", "w13_weight"),
                    ("w2", "w2_weight"),
                    ("gemm1_wei_scales", f"w13_{suffix}"),
                    ("gemm2_wei_scales", f"w2_{suffix}"),
                ):
                    if getattr(impl, name, None) is not state.targets[role].tensor:
                        raise ReloadError(
                            f"XPU cached {name} no longer references the runtime tensor"
                        )

    def prepare_for_load(self, state: ReloadState) -> None:
        # Shuffle/transpose outputs can lend dense storage before complete
        # replacement. CPU packing is opaque, so keep its weight inputs separate.
        state.prepare_sources(
            reuse_roles=tuple(
                role
                for role in state.roles
                if self.plan.backend != "cpu" or role not in ("w13_weight", "w2_weight")
            )
        )

    def finish(self, state: ReloadState) -> None:
        self.validate(state)
        weights = _process_moe_weights(state, self.plan)
        scales = self.plan.derived_scales(weights)
        _copy_moe_weights(state, weights)
        for name, value in scales.items():
            state.copy_(name, value)


@dataclass
class MarlinMoEReloadPolicy(PlainMoEReloadPolicy):
    """Normalize expert shards and replay the cold-bound packing plan."""

    def bind(self, state: ReloadState) -> None:
        super().bind(state)
        self.plan = state.module.fp8_marlin_processing_plan
        state.bind_target("workspace", partial(getattr, state.module, "workspace"))
        self.layout = self._layout(state)

    def _layout(self, state: ReloadState) -> tuple:
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            get_marlin_input_dtype,
        )

        layer = state.module
        return (
            layer.num_experts,
            layer.hidden_size,
            layer.intermediate_size_per_partition,
            tuple(layer.weight_block_size) if self.block_quant else None,
            layer.orig_dtype,
            get_marlin_input_dtype(),
        )

    def validate(self, state: ReloadState) -> None:
        super().validate(state)
        if (
            state.module.fp8_marlin_processing_plan is not self.plan
            or self._layout(state) != self.layout
        ):
            raise ReloadError("Marlin expert layout changed since runtime binding")

    def prepare_for_load(self, state: ReloadState) -> None:
        # Packed weights and exponent-adjusted FP16/BF16 scales stay separate
        # from FP8/FP32 loader inputs. Activation scales can reuse dense storage.
        state.prepare_sources(
            reuse_roles=tuple(role for role in state.roles if "input_scale" in role)
        )

    def finish(self, state: ReloadState) -> None:
        values = self._checkpoint_values(state)
        suffix = "weight_scale_inv" if self.block_quant else "weight_scale"
        names = ("w13_weight", "w2_weight", f"w13_{suffix}", f"w2_{suffix}")
        converted = self.plan.process(*(values[name] for name in names))
        values.update(zip(names, converted))
        for role, value in values.items():
            state.copy_(role, value)


class CutlassMoEReloadPolicy(_CanonicalReloadPolicy):
    """Load canonical inputs and reuse the cold-load processing plan."""

    @property
    def block_quant(self) -> bool:
        return self.plan.block_shape is not None

    def bind(self, state: ReloadState) -> None:
        method = state.module.quant_method
        self.kernel = method.moe_kernel
        self.config = method.moe_quant_config
        self.plan = method.processing_plan
        if self.kernel is None or self.config is None:
            raise ReloadError("Bind CUTLASS reload only after kernel initialization")
        for name in ("w13_weight", "w2_weight"):
            if state.targets[name].tensor.dtype != torch.float8_e4m3fn:
                raise ReloadError("CUTLASS reload requires E4M3FN runtime weights")
        if not self.block_quant:
            for name in ("g1_alphas", "g2_alphas", "a1_gscale", "a2_gscale"):
                if not isinstance(getattr(self.config, name), torch.Tensor):
                    raise ReloadError(
                        "CUTLASS per-tensor requires static activation scales"
                    )
                state.bind_target(name, partial(getattr, self.config, name))

    def validate(self, state: ReloadState) -> None:
        self._validate_kernel(state)

    def _validate_kernel(self, state: ReloadState) -> None:
        method = state.module.quant_method
        if (
            method.moe_kernel is not self.kernel
            or method.moe_quant_config is not self.config
            or method.processing_plan is not self.plan
        ):
            raise ReloadError("CUTLASS kernel/config changed since runtime binding")

    def prepare_for_load(self, state: ReloadState) -> None:
        """Prepare canonical inputs once, on the first valid local arrival.

        Old values need not be unscrambled: the round must load every required
        shard before conversion. Padded weights can lend their contiguous storage
        prefix to a smaller canonical tensor. This is not a crop of the old
        weights, and the live Parameter's shape/strides remain unchanged.

        Block scales also receive canonical loading views: W13 scale shards
        overwrite the old W31 ordering before finish swaps them back. Per-tensor
        scales stay separate because W1/W3 scales were merged during cold PWAL.
        """
        reusable_roles = {"w13_weight", "w2_weight"}
        if self.block_quant:
            reusable_roles.update(("w13_weight_scale_inv", "w2_weight_scale_inv"))
        state.prepare_sources(reuse_roles=tuple(sorted(reusable_roles)))

    def finish(self, state: ReloadState) -> None:
        self._validate_kernel(state)
        weights = _process_moe_weights(state, self.plan)
        _copy_moe_weights(state, weights)
        for name, value in self.plan.derived_scales(weights).items():
            state.copy_(name, value)
