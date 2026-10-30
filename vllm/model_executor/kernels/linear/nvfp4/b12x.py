# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

from vllm._custom_ops import scaled_fp4_quant
from vllm.config import get_current_vllm_config
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.utils.b12x import B12xWarmupUnit
from vllm.utils.b12x import (
    get_b12x_blockscaled as _import_b12x_blockscaled,
)
from vllm.utils.b12x import get_b12x_intrinsics as _import_b12x_intrinsics

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig


def _apply_b12x_nvfp4_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_storage: torch.Tensor,
    input_global_scale_inv: torch.Tensor,
    alpha: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    blockscaled = _import_b12x_blockscaled()
    assert blockscaled is not None

    output_size = int(weight.shape[0])
    output_shape = [*x.shape[:-1], output_size]
    x_2d = x.reshape(-1, x.shape[-1])
    x_packed, x_scale_swizzled = scaled_fp4_quant(
        x_2d,
        input_global_scale_inv,
        is_sf_swizzled_layout=True,
    )
    output = blockscaled.mm_nvfp4(
        x_packed,
        x_scale_swizzled,
        weight,
        weight_scale_storage,
        alpha,
        out_dtype=x.dtype,
    )
    if bias is not None:
        output = output + bias
    return output.view(*output_shape)


class B12xNvFp4LinearKernel(NvFp4LinearKernel):
    """ModelOpt NVFP4 linear through the native B12X SM120 dense GEMM."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        del compute_capability
        if not current_platform.is_cuda():
            return False, "B12X NVFP4 kernels are only available on CUDA"
        if not current_platform.is_device_capability_family(120):
            return False, "B12X NVFP4 kernels require a Blackwell 12x device"
        blockscaled = _import_b12x_blockscaled()
        if blockscaled is None or _import_b12x_intrinsics() is None:
            return False, "Install the B12X backend with `pip install vllm[b12x]`"
        if not blockscaled.is_supported():
            return False, "b12x native NVFP4 GEMM is not supported"
        return True, None

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        del config
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        intrinsics = _import_b12x_intrinsics()
        assert intrinsics is not None
        replace_parameter(
            layer,
            "weight_scale",
            intrinsics.swizzle_block_scale(layer.weight_scale.data),
        )
        layer.b12x_warmup_provider = self

    def get_b12x_warmup_unit(
        self,
        layer: torch.nn.Module,
        token_counts: tuple[int, ...],
        output_dtype: torch.dtype,
    ) -> B12xWarmupUnit:
        weight = layer.weight
        weight_scale = layer.weight_scale
        n, packed_k = map(int, weight.shape)
        k = packed_k * 2

        def compile() -> None:
            for tokens in token_counts:
                source = torch.zeros(
                    (tokens, k), dtype=output_dtype, device=weight.device
                )
                _apply_b12x_nvfp4_linear(
                    source,
                    weight,
                    weight_scale,
                    layer.input_global_scale_inv,
                    layer.alpha,
                    None,
                )

        return B12xWarmupUnit(
            name="NVFP4",
            key=(
                type(self),
                weight.device,
                n,
                k,
                weight.dtype,
                weight_scale.dtype,
                output_dtype,
            ),
            compile=compile,
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return _apply_b12x_nvfp4_linear(
            x,
            layer.weight,
            layer.weight_scale,
            layer.input_global_scale_inv,
            layer.alpha,
            bias,
        )


class B12xNvFp4W4A16LinearKernel(B12xNvFp4LinearKernel):
    """BF16 x NVFP4 GEMM without activation quantization."""

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        blockscaled = _import_b12x_blockscaled()
        assert blockscaled is not None
        packed = blockscaled.pack_weight(
            layer.weight,
            layer.weight_scale,
            recipe="nvfp4",
            global_scale=layer.weight_global_scale,
        )
        layer.b12x_nvfp4_packed_weight = packed
        # Declare every serving shape now. An undeclared token count is
        # prepared on first use, and that preparation evicts kernels other
        # layers compiled earlier in the same forward, which then fails CUDA
        # graph capture.
        config = get_current_vllm_config()
        capture_sizes = config.compilation_config.cudagraph_capture_sizes or []
        capacity = max(
            config.scheduler_config.max_num_batched_tokens,
            config.scheduler_config.max_num_scheduled_tokens or 0,
            *capture_sizes,
        )
        query = blockscaled.BlockscaledQuery(
            recipe="nvfp4",
            num_tokens=capacity,
            in_features=packed.in_features,
            padded_in_features=packed.padded_in_features,
            out_features=packed.out_features,
            activation_mode="a16",
            global_scale_kind=packed.global_scale_kind,
        )
        # A weight reload keeps the prepared plan when the shapes are unchanged.
        if getattr(layer, "b12x_nvfp4_plan_query", None) != query:
            layer.b12x_nvfp4_plan = blockscaled.plan_regimes(
                query,
                exact_m=tuple(sorted({m for m in capture_sizes if 0 < m < capacity})),
            )
            layer.b12x_nvfp4_plan_query = query

    def get_b12x_warmup_unit(
        self,
        layer: torch.nn.Module,
        token_counts: tuple[int, ...],
        output_dtype: torch.dtype,
    ) -> B12xWarmupUnit:
        plan = layer.b12x_nvfp4_plan

        def compile() -> None:
            for tokens in sorted(set(token_counts) | set(plan.token_counts)):
                source = torch.zeros(
                    (tokens, layer.weight.shape[1] * 2),
                    dtype=output_dtype,
                    device=layer.weight.device,
                )
                self.apply_weights(layer, source)

        return B12xWarmupUnit(
            name="NVFP4 W4A16",
            # Every layer owns a plan that must be prepared before capture.
            key=(type(self), plan.handle, output_dtype),
            compile=compile,
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dtype != torch.bfloat16:
            raise ValueError(f"b12x NVFP4 W4A16 requires BF16 input, got {x.dtype}")
        blockscaled = _import_b12x_blockscaled()
        assert blockscaled is not None
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        # Rows beyond the planned capacity (e.g. an encoder batch) run in
        # capacity-sized chunks; an unplanned call would prepare on first use.
        capacity = layer.b12x_nvfp4_plan_query.num_tokens
        if x_2d.shape[0] <= capacity:
            output = blockscaled.mm(
                x_2d,
                layer.b12x_nvfp4_packed_weight,
                plan=layer.b12x_nvfp4_plan,
                required_mode="a16",
            )
        else:
            output = torch.cat(
                [
                    blockscaled.mm(
                        chunk,
                        layer.b12x_nvfp4_packed_weight,
                        plan=layer.b12x_nvfp4_plan,
                        required_mode="a16",
                    )
                    for chunk in x_2d.split(capacity)
                ]
            )
        if bias is not None:
            output = output + bias
        return output.view(*x.shape[:-1], layer.weight.shape[0])


__all__ = ["B12xNvFp4LinearKernel", "B12xNvFp4W4A16LinearKernel"]
