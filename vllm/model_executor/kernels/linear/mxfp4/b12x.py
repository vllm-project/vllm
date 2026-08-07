# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

import torch

import vllm.envs as envs
from vllm.config import get_current_vllm_config_or_none
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp4Dynamic,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    _USE_LAYERNAME,
    LayerName,
    _encode_layer_name,
    current_stream,
    direct_register_custom_op,
)

from .base import MxFp4LinearKernel, MxFp4LinearLayerConfig

if TYPE_CHECKING:
    from typing import TypeAlias

    _layer_name_type: TypeAlias = str | LayerName
else:
    _layer_name_type = LayerName if _USE_LAYERNAME else str

_MXFP4_GROUP_SIZE = 32
_B12X_BLOCKSCALED: Any | None = None
_B12X_INTRINSICS: Any | None = None


def _import_b12x_blockscaled() -> Any | None:
    global _B12X_BLOCKSCALED
    if _B12X_BLOCKSCALED is None:
        try:
            _B12X_BLOCKSCALED = importlib.import_module("b12x.gemm.blockscaled")
        except ImportError:
            return None
    return _B12X_BLOCKSCALED


def _import_b12x_intrinsics() -> Any | None:
    global _B12X_INTRINSICS
    if _B12X_INTRINSICS is None:
        try:
            _B12X_INTRINSICS = importlib.import_module("b12x._lib.intrinsics")
        except ImportError:
            return None
    return _B12X_INTRINSICS


def _current_linear_backend() -> str:
    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None:
        return "auto"
    return str(getattr(vllm_config.kernel_config, "linear_backend", "auto")).lower()


@torch.compiler.assume_constant_result
def _resolve_layer_name(layer_name: str | LayerName) -> str:
    from torch._library.fake_class_registry import FakeScriptObject

    if isinstance(layer_name, LayerName):
        return layer_name.value
    elif isinstance(layer_name, FakeScriptObject):
        return layer_name.real_obj.value
    return layer_name


def _register_b12x_mxfp4_linear_layer(layer: torch.nn.Module) -> None:
    prefix = getattr(layer, "prefix", "")
    if not prefix:
        return
    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None:
        return
    static_forward_context = vllm_config.compilation_config.static_forward_context
    existing = static_forward_context.get(prefix)
    if existing is not None and existing is not layer:
        raise ValueError(f"Duplicate B12X MXFP4 linear layer name: {prefix}")
    static_forward_context[prefix] = layer


def _apply_b12x_mxfp4_linear(
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    from vllm.utils.flashinfer import flashinfer_mxfp4_quantize

    blockscaled = _import_b12x_blockscaled()
    intrinsics = _import_b12x_intrinsics()
    if blockscaled is None or intrinsics is None:
        raise ImportError("b12x native MXFP4 GEMM is not importable")

    output_size = int(layer.output_size_per_partition)
    output_shape = [*x.shape[:-1], output_size]
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    m, k = map(int, x_2d.shape)
    x_packed, x_scale_swizzled = flashinfer_mxfp4_quantize(x_2d, backend="cute-dsl")
    x_scale = intrinsics.as_grouped_scale_view_mx(
        x_scale_swizzled.view(torch.uint8).unsqueeze(0), m, k
    )
    weight_scale = intrinsics.as_grouped_scale_view_mx(
        layer.weight_scale.view(torch.uint8).unsqueeze(0), output_size, k
    )
    output = blockscaled.mm(
        (x_packed.unsqueeze(-1), x_scale),
        (layer.weight.unsqueeze(-1), weight_scale),
        ab_dtype="float4_e2m1fn",
        sf_dtype="float8_e8m0fnu",
        c_dtype=str(x.dtype).split(".")[-1],
        sf_vec_size=_MXFP4_GROUP_SIZE,
        expected_m=m,
        stream=current_stream().cuda_stream,
    )[:, :, 0]
    if bias is not None:
        output = output + bias
    return output.view(*output_shape)


def _b12x_mxfp4_linear(
    x: torch.Tensor,
    bias: torch.Tensor | None,
    layer_name: _layer_name_type,
    out_features: int,
) -> torch.Tensor:
    del out_features
    layer = get_forward_context().no_compile_layers[_resolve_layer_name(layer_name)]
    return _apply_b12x_mxfp4_linear(layer, x, bias)


def _b12x_mxfp4_linear_fake(
    x: torch.Tensor,
    bias: torch.Tensor | None,
    layer_name: _layer_name_type,
    out_features: int,
) -> torch.Tensor:
    del bias, layer_name
    return x.new_empty((*x.shape[:-1], out_features))


direct_register_custom_op(
    op_name="b12x_mxfp4_linear",
    op_func=_b12x_mxfp4_linear,
    fake_impl=_b12x_mxfp4_linear_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


class B12xMxFp4LinearKernel(MxFp4LinearKernel):
    """MXFP4 linear through the native B12X SM120 dense GEMM."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        del compute_capability
        if not current_platform.is_cuda():
            return False, "B12X MXFP4 kernels are only available on CUDA"
        if not current_platform.is_device_capability_family(120):
            return False, "B12X MXFP4 kernels require a Blackwell 12x device"
        blockscaled = _import_b12x_blockscaled()
        if blockscaled is None or _import_b12x_intrinsics() is None:
            return False, "Install the B12X backend with `pip install vllm[b12x]`"
        if not blockscaled.is_supported():
            return False, "b12x native MXFP4 GEMM is not supported"
        return True, None

    @classmethod
    def can_implement(cls, config: MxFp4LinearLayerConfig) -> tuple[bool, str | None]:
        if _current_linear_backend() != "b12x" and not envs.VLLM_USE_B12X_FP4_GEMM:
            return False, "B12X MXFP4 GEMM is not enabled"
        if config.activation_quant_key != kMxfp4Dynamic:
            return False, "B12X MXFP4 GEMM requires dynamic MXFP4 activations"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        intrinsics = _import_b12x_intrinsics()
        if intrinsics is None:
            raise ImportError("b12x native MXFP4 GEMM is not importable")
        replace_parameter(
            layer,
            "weight_scale",
            intrinsics.swizzle_block_scale(layer.weight_scale.data),
        )
        _register_b12x_mxfp4_linear_layer(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if torch.compiler.is_compiling():
            prefix = getattr(layer, "prefix", "")
            if not prefix:
                raise RuntimeError(
                    "B12X MXFP4 linear requires a layer prefix under torch.compile"
                )
            return torch.ops.vllm.b12x_mxfp4_linear(
                x,
                bias,
                _encode_layer_name(prefix),
                int(layer.output_size_per_partition),
            )
        return _apply_b12x_mxfp4_linear(layer, x, bias)
