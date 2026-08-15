# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.utils.b12x import (
    b12x_warmup_token_counts,
    reuse_packed_weight_storage,
)
from vllm.utils.b12x import (
    get_b12x_tensor_fp8_linear as _import_b12x_tensor_fp8,
)
from vllm.utils.torch_utils import current_stream

from .ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)


def _apply_b12x_tensor_fp8_packed_linear(
    layer: torch.nn.Module,
    x_q: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    packed_weight = getattr(layer, "b12x_tensor_fp8_packed_weight", None)
    if packed_weight is None:
        raise RuntimeError(
            "b12x tensor FP8 packed weights are missing; "
            "process_weights_after_loading did not run for this layer"
        )

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


def warmup_b12x_tensor_fp8_linear(
    model: torch.nn.Module,
    *,
    max_tokens: int,
    cudagraph_capture_sizes: Iterable[int] = (),
    output_dtype: torch.dtype = torch.bfloat16,
) -> int:
    if not current_platform.is_cuda():
        return 0
    if not current_platform.is_device_capability_family(120):
        return 0

    if output_dtype not in (torch.bfloat16, torch.float16):
        output_dtype = torch.bfloat16

    layer_map: dict[tuple[Any, ...], Any] = {}
    for layer in model.modules():
        packed_weight = getattr(
            layer,
            "b12x_tensor_fp8_packed_weight",
            None,
        )
        if packed_weight is None:
            continue
        device = torch.device(packed_weight.values.device)
        signature = (
            device,
            int(packed_weight.in_features),
            int(packed_weight.padded_in_features),
            int(packed_weight.out_features),
            output_dtype,
        )
        layer_map.setdefault(signature, packed_weight)
    if not layer_map:
        return 0

    tensor_fp8 = _import_b12x_tensor_fp8()
    if tensor_fp8 is None:
        return 0

    token_counts = b12x_warmup_token_counts(
        max_tokens=max_tokens,
        cudagraph_capture_sizes=cudagraph_capture_sizes,
    )
    warmed = 0
    last_device: torch.device | None = None

    with torch.inference_mode():
        for signature, packed_weight in layer_map.items():
            last_device = signature[0]
            warmed += int(
                tensor_fp8.prewarm(
                    packed_weight,
                    token_counts,
                    out_dtype=output_dtype,
                    stream=current_stream().cuda_stream,
                )
            )

        if warmed > 0 and last_device is not None and last_device.type == "cuda":
            torch.accelerator.synchronize(last_device)

    return warmed


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
        if weight.dtype != torch.float8_e4m3fn:
            raise ValueError(
                f"b12x tensor FP8 requires float8_e4m3fn weight, got {weight.dtype}"
            )
        if weight_scale.numel() != 1 or input_scale is None or input_scale.numel() != 1:
            raise ValueError(
                "b12x tensor FP8 requires scalar weight and activation scales"
            )

        out_features, in_features = map(int, self.config.weight_shape)
        if tuple(weight.shape) != (in_features, out_features):
            raise ValueError(
                "b12x tensor FP8 expects the processed weight in [K,N] layout, "
                f"got {tuple(weight.shape)} for N={out_features}, K={in_features}"
            )

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

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError("b12x tensor FP8 linear requires a Tensor input")
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
    "B12xTensorFP8ScaledMMLinearKernel",
    "warmup_b12x_tensor_fp8_linear",
]
