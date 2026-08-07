# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib
from collections.abc import Iterable
from typing import Any

import torch

import vllm.envs as envs
from vllm.config import get_current_vllm_config_or_none
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _upcast_e8m0_to_fp32,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.utils.torch_utils import current_stream, direct_register_custom_op

from .BlockScaledMMLinearKernel import (
    Fp8BlockScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)

_B12X_BLOCKSCALED: Any | None = None
_B12X_BLOCKSCALED_MISSING = False


def _import_b12x_blockscaled() -> Any | None:
    global _B12X_BLOCKSCALED, _B12X_BLOCKSCALED_MISSING
    if _B12X_BLOCKSCALED is not None:
        return _B12X_BLOCKSCALED
    if _B12X_BLOCKSCALED_MISSING:
        return None
    try:
        _B12X_BLOCKSCALED = importlib.import_module("b12x.gemm.blockscaled")
    except ImportError:
        _B12X_BLOCKSCALED_MISSING = True
        return None
    return _B12X_BLOCKSCALED


def _current_linear_backend() -> str:
    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None:
        return "auto"
    return str(getattr(vllm_config.kernel_config, "linear_backend", "auto")).lower()


def _b12x_block_fp8_warmup_token_counts(
    *,
    max_tokens: int,
    cudagraph_capture_sizes: Iterable[int] = (),
) -> tuple[int, ...]:
    counts = {1}
    counts.update(int(size) for size in cudagraph_capture_sizes if int(size) > 0)
    if int(max_tokens) > 0:
        counts.add(int(max_tokens))
    return tuple(sorted(counts))


def _run_b12x_fp8_block_scaled_mm(
    a: torch.Tensor,
    weight: torch.Tensor,
    a_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    blockscaled = _import_b12x_blockscaled()
    if blockscaled is None:
        raise ImportError("b12x regular block-FP8 GEMM is not importable")

    m, k = map(int, a.shape)
    n = int(weight.shape[0])
    output = blockscaled.mm(
        (a.reshape(m, k, 1), a_scale),
        (weight.reshape(n, k, 1), weight_scale),
        ab_dtype="float8_e4m3fn",
        sf_dtype="float32",
        c_dtype=str(out_dtype).removeprefix("torch."),
        sf_vec_size=128,
        block_fp8=True,
        expected_m=m,
        stream=current_stream().cuda_stream,
    )
    return output[:, :, 0]


def _b12x_fp8_block_scaled_mm(
    a: torch.Tensor,
    weight: torch.Tensor,
    a_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    return _run_b12x_fp8_block_scaled_mm(
        a,
        weight,
        a_scale,
        weight_scale,
        out_dtype,
    )


def _b12x_fp8_block_scaled_mm_fake(
    a: torch.Tensor,
    weight: torch.Tensor,
    a_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    del a_scale, weight_scale
    return torch.empty(
        (a.shape[0], weight.shape[0]),
        dtype=out_dtype,
        device=a.device,
    )


direct_register_custom_op(
    op_name="b12x_fp8_block_scaled_mm",
    op_func=_b12x_fp8_block_scaled_mm,
    fake_impl=_b12x_fp8_block_scaled_mm_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def warmup_b12x_block_fp8_linear(
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

    blockscaled = _import_b12x_blockscaled()
    if blockscaled is None:
        return 0
    token_counts = _b12x_block_fp8_warmup_token_counts(
        max_tokens=max_tokens,
        cudagraph_capture_sizes=cudagraph_capture_sizes,
    )
    seen_signatures: set[tuple[Any, ...]] = set()
    warmed = 0

    with torch.inference_mode():
        for layer in model.modules():
            if not getattr(layer, "b12x_block_fp8_linear", False):
                continue
            weight = layer.weight
            weight_scale = getattr(layer, "weight_scale_inv", None)
            if weight_scale is None:
                weight_scale = layer.weight_scale
            n, k = map(int, weight.shape)
            signature = (
                weight.device,
                n,
                k,
                weight.dtype,
                weight_scale.dtype,
                output_dtype,
            )
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            for tokens in token_counts:
                a = torch.empty(
                    (tokens, k),
                    dtype=weight.dtype,
                    device=weight.device,
                )
                a_scale = torch.empty(
                    (tokens, k // 128),
                    dtype=torch.float32,
                    device=weight.device,
                )
                _run_b12x_fp8_block_scaled_mm(
                    a,
                    weight,
                    a_scale,
                    weight_scale,
                    output_dtype,
                )
                warmed += 1
    return warmed


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

        if _current_linear_backend() != "b12x" and not envs.VLLM_USE_B12X_FP8_GEMM:
            return False, "B12X FP8 GEMM is not enabled"
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
        layer.b12x_block_fp8_linear = True

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.vllm.b12x_fp8_block_scaled_mm(
            A,
            B,
            As,
            Bs,
            self.config.out_dtype,
        )


__all__ = [
    "B12xFp8BlockScaledMMKernel",
    "warmup_b12x_block_fp8_linear",
]
