# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import weakref

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
    """BF16 x NVFP4 GEMM with prepared b12x A16 execution plans."""

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from b12x.preparation import PreparationSession, PreparedCall

        super().process_weights_after_loading(layer)
        blockscaled = _import_b12x_blockscaled()
        assert blockscaled is not None
        config = get_current_vllm_config()
        scheduler = config.scheduler_config
        capture_sizes = config.compilation_config.cudagraph_capture_sizes or []
        capacity = max(
            scheduler.max_num_batched_tokens,
            scheduler.max_num_scheduled_tokens or 0,
            *capture_sizes,
        )
        weight = layer.weight
        scales = layer.weight_scale
        global_scale = layer.weight_global_scale
        n, packed_k = weight.shape
        query = blockscaled.BlockscaledQuery(
            recipe="nvfp4",
            num_tokens=capacity,
            in_features=packed_k * 2,
            padded_in_features=packed_k * 2,
            out_features=n,
            activation_mode="a16",
        )
        plan = blockscaled.plan_regimes(
            query, exact_m=tuple(sorted({m for m in capture_sizes if 0 < m < capacity}))
        )
        source = torch.zeros(
            (capacity, packed_k * 2), dtype=torch.bfloat16, device=weight.device
        )

        def prepare_call(state):
            return PreparedCall(
                run=lambda: state.run(
                    source[: state.query.num_tokens], weight, scales, global_scale
                )
            )

        session = PreparationSession(
            device=weight.device, autotune=False, compile_workers=0
        )
        try:
            session.prepare(
                (
                    plan.request(
                        name="nvfp4_w4a16",
                        prepare_calls={m: prepare_call for m in plan.token_counts},
                    ),
                )
            )
        except BaseException:
            session.close()
            raise
        previous_finalizer = getattr(layer, "b12x_nvfp4_a16_finalizer", None)
        if previous_finalizer is not None:
            previous_finalizer()
        layer.b12x_nvfp4_a16_plan = plan
        layer.b12x_nvfp4_a16_finalizer = weakref.finalize(layer, session.close)
        layer.b12x_nvfp4_a16_finalizer.atexit = False

    def get_b12x_warmup_unit(
        self,
        layer: torch.nn.Module,
        token_counts: tuple[int, ...],
        output_dtype: torch.dtype,
    ) -> B12xWarmupUnit:
        def compile() -> None:
            for tokens in token_counts:
                source = torch.zeros(
                    (tokens, layer.weight.shape[1] * 2),
                    dtype=output_dtype,
                    device=layer.weight.device,
                )
                self.apply_weights(layer, source)

        return B12xWarmupUnit(
            name="NVFP4 W4A16",
            key=(
                type(self),
                layer.weight.device,
                tuple(layer.weight.shape),
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
        if x.dtype != torch.bfloat16:
            raise ValueError(f"b12x NVFP4 W4A16 requires BF16 input, got {x.dtype}")
        blockscaled = _import_b12x_blockscaled()
        assert blockscaled is not None
        output = blockscaled.w4a16(
            x.reshape(-1, x.shape[-1]).contiguous(),
            layer.weight,
            layer.weight_scale,
            layer.weight_global_scale,
            plan=layer.b12x_nvfp4_a16_plan,
        )
        if bias is not None:
            output = output + bias
        return output.view(*x.shape[:-1], layer.weight.shape[0])


__all__ = ["B12xNvFp4LinearKernel", "B12xNvFp4W4A16LinearKernel"]
