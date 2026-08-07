# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import torch

import vllm.envs as envs
from vllm.config import get_current_vllm_config_or_none
from vllm.forward_context import get_forward_context
from vllm.model_executor.kernels.b12x_utils import (
    reuse_packed_weight_storage,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    MXFP8_SCALE_DTYPE,
    MXFP8_VALUE_DTYPE,
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

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig

if TYPE_CHECKING:
    from typing import TypeAlias

    _layer_name_type: TypeAlias = str | LayerName
else:
    _layer_name_type = LayerName if _USE_LAYERNAME else str

_B12X_MXFP8: Any | None = None
_B12X_MXFP8_MISSING = False


def _import_b12x_mxfp8() -> Any | None:
    global _B12X_MXFP8, _B12X_MXFP8_MISSING
    if _B12X_MXFP8 is not None:
        return _B12X_MXFP8
    if _B12X_MXFP8_MISSING:
        return None
    try:
        _B12X_MXFP8 = importlib.import_module("b12x.gemm.mxfp8_linear")
    except ImportError:
        _B12X_MXFP8_MISSING = True
        return None
    return _B12X_MXFP8


def _current_linear_backend() -> str:
    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None:
        return "auto"
    return str(getattr(vllm_config.kernel_config, "linear_backend", "auto")).lower()


def _b12x_mxfp8_enabled() -> bool:
    return _current_linear_backend() == "b12x" or envs.VLLM_USE_B12X_FP8_GEMM


def _b12x_mxfp8_expected_m(tokens: int) -> int:
    return max(1, int(tokens))


def _b12x_mxfp8_warmup_token_counts(
    *,
    max_tokens: int,
    cudagraph_capture_sizes: Iterable[int] = (),
) -> tuple[int, ...]:
    counts = {1}
    counts.update(int(size) for size in cudagraph_capture_sizes if int(size) > 0)
    if int(max_tokens) > 0:
        counts.add(int(max_tokens))
    return tuple(sorted(counts))


@torch.compiler.assume_constant_result
def _resolve_layer_name(layer_name: str | LayerName) -> str:
    from torch._library.fake_class_registry import FakeScriptObject

    if isinstance(layer_name, LayerName):
        return layer_name.value
    elif isinstance(layer_name, FakeScriptObject):
        return layer_name.real_obj.value
    return layer_name


def _register_b12x_mxfp8_linear_layer(layer: torch.nn.Module) -> None:
    prefix = getattr(layer, "prefix", "")
    if not prefix:
        return
    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None:
        return
    static_forward_context = vllm_config.compilation_config.static_forward_context
    existing = static_forward_context.get(prefix)
    if existing is not None and existing is not layer:
        raise ValueError(f"Duplicate B12X MXFP8 linear layer name: {prefix}")
    static_forward_context[prefix] = layer


def _apply_b12x_mxfp8_packed_linear(
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    packed_weight = getattr(layer, "b12x_mxfp8_packed_weight", None)
    if packed_weight is None:
        raise RuntimeError(
            "b12x MXFP8 packed weights are missing; "
            "process_weights_after_loading did not run for this layer"
        )

    input_2d = x.reshape(-1, x.shape[-1]).contiguous()
    output_shape = [*x.shape[:-1], int(packed_weight.out_features)]

    mxfp8 = _import_b12x_mxfp8()
    if mxfp8 is None:
        raise ImportError("b12x.gemm.mxfp8_linear is not importable")
    output = mxfp8.mm(
        input_2d,
        packed_weight,
        bias=bias,
        expected_m=_b12x_mxfp8_expected_m(int(input_2d.shape[0])),
        stream=current_stream().cuda_stream,
    )
    return output.view(*output_shape)


def warmup_b12x_mxfp8_linear(
    model: torch.nn.Module,
    *,
    max_tokens: int,
    cudagraph_capture_sizes: Iterable[int] = (),
    output_dtype: torch.dtype = torch.bfloat16,
) -> int:
    if not _b12x_mxfp8_enabled():
        return 0
    if not current_platform.is_cuda():
        return 0
    if not current_platform.is_device_capability_family(120):
        return 0
    if output_dtype not in (torch.bfloat16, torch.float16):
        output_dtype = torch.bfloat16

    mxfp8 = _import_b12x_mxfp8()
    if mxfp8 is None:
        return 0

    token_counts = _b12x_mxfp8_warmup_token_counts(
        max_tokens=max_tokens,
        cudagraph_capture_sizes=cudagraph_capture_sizes,
    )
    seen_signatures: set[tuple[int, int, int, torch.dtype]] = set()
    warmed = 0
    last_device: torch.device | None = None

    with torch.inference_mode():
        for layer in model.modules():
            packed_weight = getattr(layer, "b12x_mxfp8_packed_weight", None)
            if packed_weight is None:
                continue
            signature = (
                int(packed_weight.in_features),
                int(packed_weight.padded_in_features),
                int(packed_weight.out_features),
                output_dtype,
            )
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)

            device = torch.device(packed_weight.weight.values.device)
            last_device = device
            for tokens in token_counts:
                source = torch.zeros(
                    (tokens, int(packed_weight.in_features)),
                    dtype=output_dtype,
                    device=device,
                )
                mxfp8.mm(
                    source,
                    packed_weight,
                    expected_m=_b12x_mxfp8_expected_m(tokens),
                    stream=current_stream().cuda_stream,
                )
                warmed += 1

        if warmed > 0 and last_device is not None and last_device.type == "cuda":
            torch.accelerator.synchronize(last_device)

    return warmed


def _b12x_mxfp8_linear(
    x: torch.Tensor,
    bias: torch.Tensor | None,
    layer_name: _layer_name_type,
    out_features: int,
) -> torch.Tensor:
    del out_features
    layer = get_forward_context().no_compile_layers[_resolve_layer_name(layer_name)]
    return _apply_b12x_mxfp8_packed_linear(layer, x, bias)


def _b12x_mxfp8_linear_fake(
    x: torch.Tensor,
    bias: torch.Tensor | None,
    layer_name: _layer_name_type,
    out_features: int,
) -> torch.Tensor:
    del bias, layer_name
    return x.new_empty((*x.shape[:-1], out_features))


direct_register_custom_op(
    op_name="b12x_mxfp8_linear",
    op_func=_b12x_mxfp8_linear,
    fake_impl=_b12x_mxfp8_linear_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


class B12xMxfp8LinearKernel(Mxfp8LinearKernel):
    """ModelOpt MXFP8 linear through the native b12x SM120 dense GEMM path."""

    @classmethod
    def is_supported(
        cls,
        compute_capability: int | None = None,
    ) -> tuple[bool, str | None]:
        del compute_capability
        if not current_platform.is_cuda():
            return False, "b12x MXFP8 kernels are only available on CUDA"
        if not current_platform.is_device_capability_family(120):
            return False, "b12x MXFP8 kernels require a Blackwell 12x device"
        if not _b12x_mxfp8_enabled():
            return False, "b12x MXFP8 GEMM is not enabled"
        mxfp8 = _import_b12x_mxfp8()
        if mxfp8 is None:
            return False, "Install the B12X backend with `pip install vllm[b12x]`"
        if not mxfp8.is_supported():
            return False, "b12x.gemm.mxfp8_linear is not supported"
        return True, None

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        del c
        if not _b12x_mxfp8_enabled():
            return False, "b12x MXFP8 GEMM is not enabled"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data
        if weight.dtype != MXFP8_VALUE_DTYPE:
            raise ValueError(
                f"b12x MXFP8 requires {MXFP8_VALUE_DTYPE}, got {weight.dtype}"
            )
        if weight.ndim != 2:
            raise ValueError(f"b12x MXFP8 weight must be 2D, got {weight.ndim}D")
        if not hasattr(layer, "weight_scale"):
            raise ValueError("b12x MXFP8 linear requires weight_scale")

        out_features, in_features = map(int, weight.shape)
        if in_features % MXFP8_BLOCK_SIZE != 0:
            raise ValueError(
                "b12x MXFP8 requires input features divisible by "
                f"{MXFP8_BLOCK_SIZE}, got {in_features}"
            )
        weight_scale = layer.weight_scale.data
        if weight_scale.dtype != MXFP8_SCALE_DTYPE:
            raise ValueError(
                f"b12x MXFP8 requires {MXFP8_SCALE_DTYPE} weight_scale, "
                f"got {weight_scale.dtype}"
            )
        if weight_scale.ndim != 2:
            raise ValueError(
                f"b12x MXFP8 weight_scale must be 2D, got {weight_scale.ndim}D"
            )

        mxfp8 = _import_b12x_mxfp8()
        if mxfp8 is None:
            raise ImportError("b12x.gemm.mxfp8_linear is not importable")
        scale_k = in_features // MXFP8_BLOCK_SIZE
        packed_weight = mxfp8.pack_weight(
            weight[:out_features, :in_features].detach(),
            weight_scale[:out_features, :scale_k].detach(),
        )
        layer.b12x_mxfp8_packed_weight = reuse_packed_weight_storage(
            getattr(layer, "b12x_mxfp8_packed_weight", None),
            packed_weight,
        )
        _register_b12x_mxfp8_linear_layer(layer)
        replace_parameter(layer, "weight", weight.new_empty((0,)))
        replace_parameter(layer, "weight_scale", weight_scale.new_empty((0,)))

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
                    "B12X MXFP8 linear requires a layer prefix under torch.compile"
                )
            packed_weight = getattr(layer, "b12x_mxfp8_packed_weight", None)
            if packed_weight is None:
                raise RuntimeError(
                    "b12x MXFP8 packed weights are missing; "
                    "process_weights_after_loading did not run for this layer"
                )
            return torch.ops.vllm.b12x_mxfp8_linear(
                x,
                bias,
                _encode_layer_name(prefix),
                int(packed_weight.out_features),
            )

        return _apply_b12x_mxfp8_packed_linear(layer, x, bias)
