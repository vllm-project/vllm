# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract tests for the ``QuantizedActivation`` manual-fusion integration.

Pure-Python introspection over the linear-kernel registry -- no model load, no
device -- pinning which backends consume a pre-quantized activation and how the
bridge / consumer enforce the key. Tracks the AR+RMS+Quant bullet under
https://github.com/vllm-project/vllm/issues/43224.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.kernels.linear import (
    _POSSIBLE_FP8_BLOCK_KERNELS,
    _POSSIBLE_FP8_KERNELS,
    _POSSIBLE_INT8_KERNELS,
    _POSSIBLE_NVFP4_KERNELS,
)
from vllm.model_executor.kernels.linear.nvfp4.base import (
    NvFp4LinearKernel,
    NvFp4LinearLayerConfig,
)
from vllm.model_executor.kernels.linear.nvfp4.flashinfer import (
    FlashInferCutlassNvFp4LinearKernel,
    FlashInferTrtllmNvFp4LinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
    CutlassFP8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.flashinfer import (
    FlashInferFP8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    FP8ScaledMMLinearLayerConfig,
    Int8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.fusion.ar_rms_quant import (
    _FUSED_AR_RMS_QUANT_IMPLS,
    _FUSED_RMS_QUANT_IMPLS,
    _flashinfer_ar_rms_fp8,
    _flashinfer_ar_rms_nvfp4,
    _rms_norm_fp8_static,
    _to_fp8_qa,
    _to_nvfp4_qa,
)
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    as_quantized_activation,
    expose_input_quant_key,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8StaticTensorSym,
    kNvfp4Dynamic,
)
from vllm.platforms import current_platform

# The only backends that consume a pre-quantized activation. Anything else must
# quantize its own input, so the bridge must not mark its layer.
SUPPORTING = {
    CutlassFP8ScaledMMLinearKernel,
    FlashInferFP8ScaledMMLinearKernel,
    FlashInferCutlassNvFp4LinearKernel,
}


def _all_kernel_classes() -> list[type]:
    seen: dict[type, None] = {}
    for registry in (
        _POSSIBLE_FP8_KERNELS,
        _POSSIBLE_FP8_BLOCK_KERNELS,
        _POSSIBLE_INT8_KERNELS,
        _POSSIBLE_NVFP4_KERNELS,
    ):
        for kernels in registry.values():
            for cls in kernels:
                seen.setdefault(cls, None)
    return list(seen)


def _probe(cls: type):
    """A bare kernel instance with a plausible config, so ``input_quant_key()``
    can be queried without the hardware-gated ``__init__``."""
    obj = cls.__new__(cls)  # type: ignore[call-overload]
    if issubclass(cls, NvFp4LinearKernel):
        obj.config = NvFp4LinearLayerConfig()
    elif issubclass(cls, Int8ScaledMMLinearKernel):
        obj.config = Int8ScaledMMLinearLayerConfig(
            is_static_input_scheme=True, is_channelwise=False, input_symmetric=True
        )
    else:
        obj.config = FP8ScaledMMLinearLayerConfig(
            weight_quant_key=kFp8StaticTensorSym,
            activation_quant_key=kFp8StaticTensorSym,
            weight_shape=(16, 16),
            input_dtype=torch.bfloat16,
            out_dtype=torch.bfloat16,
        )
    return obj


def _resolved_apply_weights(cls: type):
    for base in cls.__mro__:
        if "apply_weights" in base.__dict__:
            return base.__dict__["apply_weights"]
    raise AssertionError(f"{cls.__name__} has no apply_weights in its MRO")


def test_only_known_backends_support_prequantized_input():
    declarers = {c for c in _all_kernel_classes() if _probe(c).input_quant_key()}
    assert declarers == SUPPORTING


def test_supporting_backend_declares_consume_via_helper():
    for cls in SUPPORTING:
        fn = _resolved_apply_weights(cls)
        assert "as_quantized_activation" in fn.__code__.co_names, cls.__name__


def test_bridge_marks_supporting_and_skips_others():
    supported = _probe(FlashInferCutlassNvFp4LinearKernel)
    layer = torch.nn.Module()
    expose_input_quant_key(layer, supported)
    assert layer.input_quant_key == kNvfp4Dynamic

    unsupported = _probe(FlashInferTrtllmNvFp4LinearKernel)
    assert unsupported.input_quant_key() is None
    layer = torch.nn.Module()
    expose_input_quant_key(layer, unsupported)
    assert not hasattr(layer, "input_quant_key")


def test_fp8_producer_stamps_registered_key():
    qa = _to_fp8_qa(
        torch.zeros(2, 4, dtype=current_platform.fp8_dtype()),
        torch.zeros(2, 4),
        SimpleNamespace(input_scale=torch.tensor(1.0)),
    )
    assert qa.quant_key == kFp8StaticTensorSym
    assert _FUSED_RMS_QUANT_IMPLS[kFp8StaticTensorSym] is _rms_norm_fp8_static
    assert _FUSED_AR_RMS_QUANT_IMPLS[kFp8StaticTensorSym] is _flashinfer_ar_rms_fp8


def test_nvfp4_producer_stamps_registered_key():
    qa = _to_nvfp4_qa(
        torch.zeros(2, 2, dtype=torch.uint8),  # 2 fp4 packed per byte
        torch.zeros(2, 4, dtype=torch.uint8),  # viewed as float8_e4m3fn
        torch.zeros(2, 4),
    )
    assert qa.quant_key == kNvfp4Dynamic
    assert _FUSED_AR_RMS_QUANT_IMPLS[kNvfp4Dynamic] is _flashinfer_ar_rms_nvfp4


def test_as_quantized_activation_validates_key():
    qa = QuantizedActivation(
        data=torch.zeros(2, 4, dtype=current_platform.fp8_dtype()),
        scale=torch.tensor(1.0),
        orig_dtype=torch.bfloat16,
        orig_shape=torch.Size([2, 4]),
        quant_key=kFp8StaticTensorSym,
    )
    with pytest.raises(AssertionError):
        as_quantized_activation(qa, kNvfp4Dynamic)
    with pytest.raises(AssertionError):
        as_quantized_activation(qa, None)
    assert as_quantized_activation(torch.zeros(2, 4), kFp8StaticTensorSym) is None
    assert as_quantized_activation(qa, kFp8StaticTensorSym) is qa
