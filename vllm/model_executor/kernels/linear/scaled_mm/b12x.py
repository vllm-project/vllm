# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _upcast_e8m0_to_fp32,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.utils.b12x import B12xWarmupUnit, reuse_packed_weight_storage
from vllm.utils.b12x import (
    get_b12x_blockscaled as _import_b12x_blockscaled,
)
from vllm.utils.b12x import (
    get_b12x_tensor_fp8_linear as _import_b12x_tensor_fp8,
)
from vllm.utils.torch_utils import current_stream

from .BlockScaledMMLinearKernel import (
    Fp8BlockScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)
from .ScaledMMLinearKernel import FP8ScaledMMLinearKernel


def _run_b12x_fp8_block_scaled_mm(
    a: torch.Tensor,
    weight: torch.Tensor,
    a_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    blockscaled = _import_b12x_blockscaled()
    assert blockscaled is not None

    return blockscaled.mm_block_fp8(
        a,
        a_scale,
        weight,
        weight_scale,
        out_dtype=out_dtype,
    )


class B12xFp8BlockScaledMMKernel(Fp8BlockScaledMMLinearKernel):
    """K128 block-FP8 linear through the native B12X SM120 dense GEMM."""

    @classmethod
    def is_supported(
        cls,
        compute_capability: int | None = None,
    ) -> tuple[bool, str | None]:
        del compute_capability
        if not current_platform.is_cuda():
            return False, "B12X FP8 kernels are only available on CUDA"
        if not current_platform.is_device_capability_family(120):
            return False, "B12X FP8 kernels require a Blackwell 12x device"
        blockscaled = _import_b12x_blockscaled()
        if blockscaled is None:
            return False, "Install the B12X backend with `pip install vllm[b12x]`"
        if not blockscaled.is_supported():
            return False, "B12X regular block-FP8 GEMM is not supported"
        return True, None

    @classmethod
    def can_implement(
        cls,
        config: FP8ScaledMMLinearLayerConfig,
    ) -> tuple[bool, str | None]:
        can_implement_base, reason = super().can_implement(config)
        if not can_implement_base:
            return can_implement_base, reason

        if config.input_dtype not in (torch.bfloat16, torch.float16):
            return False, "Supports only bf16/fp16 input dtype"
        if config.input_dtype != config.out_dtype:
            return False, "Input and output dtype must match"

        act_group_shape = config.activation_quant_key.scale.group_shape
        if act_group_shape != GroupShape(1, 128):
            return (
                False,
                "Supports only dynamic per-token group activation quantization "
                "with group_shape=(1,128)",
            )
        weight_group_shape = config.weight_quant_key.scale.group_shape
        if weight_group_shape != GroupShape(128, 128):
            return False, "Supports only 128x128 block-scaled FP8 weights"

        out_features, in_features = config.weight_shape
        if in_features <= 0 or in_features % 128 != 0:
            return False, "Input features must be a positive multiple of 128"
        if out_features <= 0 or out_features % 128 != 0:
            return False, "Output features must be a positive multiple of 128"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        params = self._get_layer_params(layer)
        if params.weight_scale_inv is not None:
            weight_scale = params.weight_scale_inv
            scale_attr = params.WEIGHT_SCALE_INV
        else:
            weight_scale = params.weight_scale
            scale_attr = params.WEIGHT_SCALE
        if weight_scale is not None and weight_scale.dtype in (
            torch.float8_e8m0fnu,
            torch.uint8,
        ):
            # TODO: Remove once B12X supports 128x128 UE8M0 block scales.
            replace_parameter(
                layer,
                scale_attr,
                _upcast_e8m0_to_fp32(weight_scale).contiguous(),
            )
        layer.b12x_warmup_provider = self

    def get_b12x_warmup_unit(
        self,
        layer: torch.nn.Module,
        token_counts: tuple[int, ...],
        output_dtype: torch.dtype,
    ) -> B12xWarmupUnit:
        weight = layer.weight
        weight_scale = getattr(layer, "weight_scale_inv", None)
        if weight_scale is None:
            weight_scale = layer.weight_scale
        n, k = map(int, weight.shape)

        def compile() -> None:
            for tokens in token_counts:
                a = torch.empty((tokens, k), dtype=weight.dtype, device=weight.device)
                a_scale = torch.empty(
                    (tokens, k // 128),
                    dtype=torch.float32,
                    device=weight.device,
                )
                _run_b12x_fp8_block_scaled_mm(
                    a, weight, a_scale, weight_scale, output_dtype
                )

        return B12xWarmupUnit(
            name="block-FP8",
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

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        return _run_b12x_fp8_block_scaled_mm(
            A,
            B,
            As,
            Bs,
            self.config.out_dtype,
        )


def _apply_b12x_tensor_fp8_packed_linear(
    layer: torch.nn.Module,
    x_q: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    packed_weight = layer.b12x_tensor_fp8_packed_weight

    tensor_fp8 = _import_b12x_tensor_fp8()
    assert tensor_fp8 is not None

    input_2d = x_q.reshape(-1, x_q.shape[-1]).contiguous()
    output_shape = [*x_q.shape[:-1], int(packed_weight.out_features)]
    output = tensor_fp8.mm(
        input_2d,
        packed_weight,
        bias=bias,
        out_dtype=out_dtype,
        expected_m=max(1, int(input_2d.shape[0])),
    )
    return output.view(*output_shape)


class B12xTensorFP8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    """Static per-tensor FP8 linear through the B12X SM12x dense GEMM."""

    @classmethod
    def is_supported(
        cls,
        compute_capability: int | None = None,
    ) -> tuple[bool, str | None]:
        del compute_capability
        if not current_platform.is_cuda():
            return False, "b12x tensor FP8 kernels are only available on CUDA"
        if not current_platform.is_device_capability_family(120):
            return False, "b12x tensor FP8 kernels require a Blackwell 12x device"
        tensor_fp8 = _import_b12x_tensor_fp8()
        if tensor_fp8 is None:
            return False, "Install the B12X backend with `pip install vllm[b12x]`"
        if not tensor_fp8.is_supported():
            return False, "b12x.gemm.tensor_fp8_linear is not supported"
        return True, None

    @classmethod
    def can_implement(
        cls,
        config: FP8ScaledMMLinearLayerConfig,
    ) -> tuple[bool, str | None]:
        activation_scale = config.activation_quant_key.scale
        weight_scale = config.weight_quant_key.scale
        if (
            not activation_scale.static
            or not activation_scale.group_shape.is_per_tensor()
        ):
            return False, "requires static per-tensor activation scales"
        if not weight_scale.static or not weight_scale.group_shape.is_per_tensor():
            return False, "requires static per-tensor weight scales"
        if config.input_dtype not in (torch.bfloat16, torch.float16):
            return False, "supports only bf16/fp16 input dtype"
        if config.out_dtype not in (torch.bfloat16, torch.float16):
            return False, "supports only bf16/fp16 output dtype"
        out_features, in_features = config.weight_shape
        if out_features <= 0 or in_features <= 0 or in_features % 32 != 0:
            return False, "weight dimensions must be positive with K divisible by 32"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight, weight_scale, input_scale, _ = self._get_layer_params(layer)
        assert weight.dtype == torch.float8_e4m3fn
        assert input_scale is not None
        assert weight_scale.numel() == input_scale.numel() == 1

        out_features, in_features = map(int, self.config.weight_shape)
        assert tuple(weight.shape) == (in_features, out_features)

        tensor_fp8 = _import_b12x_tensor_fp8()
        assert tensor_fp8 is not None
        output_scale = (
            input_scale.detach().to(torch.float32).reshape(1)
            * weight_scale.detach().to(torch.float32).reshape(1)
        ).contiguous()
        packed_weight = tensor_fp8.pack_weight(
            weight.detach().T.contiguous(),
            output_scale,
        )
        layer.b12x_tensor_fp8_packed_weight = reuse_packed_weight_storage(
            getattr(layer, "b12x_tensor_fp8_packed_weight", None),
            packed_weight,
        )
        weight_name, weight_scale_name, _, _ = self.layer_param_names
        replace_parameter(layer, weight_name, weight.new_empty((0,)))
        replace_parameter(layer, weight_scale_name, weight_scale.new_empty((0,)))
        layer.b12x_warmup_provider = self

    def get_b12x_warmup_unit(
        self,
        layer: torch.nn.Module,
        token_counts: tuple[int, ...],
        output_dtype: torch.dtype,
    ) -> B12xWarmupUnit:
        packed_weight = layer.b12x_tensor_fp8_packed_weight
        device = torch.device(packed_weight.values.device)

        def compile() -> None:
            tensor_fp8 = _import_b12x_tensor_fp8()
            assert tensor_fp8 is not None
            tensor_fp8.prewarm(
                packed_weight,
                token_counts,
                out_dtype=output_dtype,
                stream=current_stream().cuda_stream,
            )

        return B12xWarmupUnit(
            name="tensor FP8",
            key=(
                type(self),
                device,
                int(packed_weight.in_features),
                int(packed_weight.padded_in_features),
                int(packed_weight.out_features),
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
        assert isinstance(x, torch.Tensor)
        _, _, input_scale, input_scale_ub = self._get_layer_params(layer)
        input_2d = x.reshape(-1, x.shape[-1])
        x_q, _ = self.quant_fp8(input_2d, input_scale, input_scale_ub)
        out_dtype = self.config.out_dtype
        output = _apply_b12x_tensor_fp8_packed_linear(
            layer,
            x_q,
            bias,
            out_dtype,
        )
        return output.view(*x.shape[:-1], output.shape[-1])

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        del A, B, out_dtype, As, Bs, bias, output_shape
        raise NotImplementedError("b12x tensor FP8 linear overrides apply_weights")


__all__ = [
    "B12xFp8BlockScaledMMKernel",
    "B12xTensorFP8ScaledMMLinearKernel",
]
