# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

from vllm._custom_ops import scaled_fp4_quant
from vllm.model_executor.kernels.linear.phase_split import (
    ActivationMode,
    DualActivationKernel,
    padding_rows,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    FLOAT4_E2M1_MAX,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kNvfp4DynamicTensor,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.utils.b12x import (
    B12xWarmupUnit,
    plan_b12x_linear,
    reuse_packed_weight_storage,
)
from vllm.utils.b12x import (
    get_b12x_blockscaled as _import_b12x_blockscaled,
)
from vllm.utils.b12x import get_b12x_intrinsics as _import_b12x_intrinsics

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig

# Maps a per-call amax to the largest FP8 E4M3 block scale.
_NVFP4_SCALE_RANGE = FLOAT4_E2M1_MAX * torch.finfo(torch.float8_e4m3fn).max


def _activation_global_scale(x: torch.Tensor) -> torch.Tensor:
    """NVFP4 activation multiplier of a call from the absolute maximum of its
    real rows: CUDA graph and data parallel padding rows may hold uninitialized
    memory, which must not set the scale every row shares."""
    if x.shape[0] == 0:
        # No rows to scale; the max over them is undefined.
        return x.new_ones((), dtype=torch.float32)
    padding = padding_rows(x.shape[0])
    if padding is not None:
        amax = torch.linalg.vector_norm(x, float("inf"), dim=1)
        amax = amax.masked_fill(padding, 0).amax().float()
    elif x.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"b12x NVFP4 activations of a {x.shape[0]}-row call in a CUDA graph "
            "need the runner's padding mask (ForwardContext.is_padding), which "
            "this forward pass does not set for these rows"
        )
    else:
        amax = torch.linalg.vector_norm(x, float("inf")).float()
    return _NVFP4_SCALE_RANGE / amax.clamp_min(1e-12)


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


class B12xNvFp4W4A16LinearKernel(B12xNvFp4LinearKernel, DualActivationKernel):
    """BF16 x NVFP4 GEMM. With a phase split, calls in an NVFP4 phase quantize
    their activations with one global scale per call, computed from the
    absolute maximum of the call's real rows."""

    @classmethod
    def activation_modes(cls) -> frozenset[ActivationMode]:
        return frozenset((None, kNvfp4DynamicTensor))

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
        layer.b12x_nvfp4_packed_weight = reuse_packed_weight_storage(
            getattr(layer, "b12x_nvfp4_packed_weight", None), packed
        )
        layer.b12x_nvfp4_plans = plan_b12x_linear(
            blockscaled,
            packed,
            "nvfp4",
            self.phase,
            getattr(layer, "b12x_nvfp4_plans", None),
        )

    def get_b12x_warmup_unit(
        self,
        layer: torch.nn.Module,
        token_counts: tuple[int, ...],
        output_dtype: torch.dtype,
    ) -> B12xWarmupUnit:
        plans = layer.b12x_nvfp4_plans
        mode = kNvfp4DynamicTensor if any(plans.quantized) else None
        planned = {m for plan in plans.plans for m in plan.token_counts}

        def compile() -> None:
            for tokens in sorted(set(token_counts) | planned):
                source = torch.zeros(
                    (tokens, layer.weight.shape[1] * 2),
                    dtype=output_dtype,
                    device=layer.weight.device,
                )
                self.apply_weights(layer, source, mode=mode)

        return B12xWarmupUnit(
            name="NVFP4 W4A16",
            # Every layer owns a plan that must be prepared before capture.
            key=(type(self), *(plan.handle for plan in plans.plans), output_dtype),
            compile=compile,
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        mode: ActivationMode = None,
    ) -> torch.Tensor:
        if x.dtype != torch.bfloat16:
            raise ValueError(f"b12x NVFP4 W4A16 requires BF16 input, got {x.dtype}")
        if mode is not None and mode != kNvfp4DynamicTensor:
            raise ValueError(f"b12x NVFP4 W4A16 does not support activations {mode}")
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        # Rows beyond the planned capacity (e.g. an encoder batch) run in
        # capacity-sized chunks; an unplanned call would prepare on first use.
        capacity = layer.b12x_nvfp4_plans.query.num_tokens
        quantize = mode is not None
        if x_2d.shape[0] <= capacity:
            output = self._mm(layer, x_2d, quantize)
        else:
            output = torch.cat(
                [self._mm(layer, c, quantize) for c in x_2d.split(capacity)]
            )
        if bias is not None:
            output = output + bias
        return output.view(*x.shape[:-1], layer.weight.shape[0])

    @staticmethod
    def _mm(layer: torch.nn.Module, x: torch.Tensor, quantize: bool) -> torch.Tensor:
        blockscaled = _import_b12x_blockscaled()
        assert blockscaled is not None
        packed = layer.b12x_nvfp4_packed_weight
        plans = layer.b12x_nvfp4_plans
        if not any(plans.quantized):
            return blockscaled.mm(x, packed, plan=plans.plan, required_mode="a16")
        plan, quantized = plans.select(x.shape[0], quantize)
        if quantized:
            scale, required_mode = _activation_global_scale(x), "quantized"
        else:
            # The plan's A16 regimes require a scale argument but never read it.
            scale, required_mode = packed.global_scale, "a16"
        return blockscaled.mm(
            x,
            packed,
            plan=plan,
            activation_global_scale=scale,
            required_mode=required_mode,
        )


__all__ = ["B12xNvFp4LinearKernel", "B12xNvFp4W4A16LinearKernel"]
