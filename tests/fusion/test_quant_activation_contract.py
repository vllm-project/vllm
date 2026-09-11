# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract tests for the QuantizedActivation linear-kernel integration."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.kernels.linear import (
    _POSSIBLE_FP8_BLOCK_KERNELS,
    _POSSIBLE_FP8_KERNELS,
    _POSSIBLE_INT8_KERNELS,
    _POSSIBLE_MXFP8_KERNELS,
    _POSSIBLE_NVFP4_KERNELS,
)
from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
    FlashInferCutedslMxfp8LinearKernel,
    FlashInferCutlassMxfp8LinearKernel,
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
from vllm.model_executor.kernels.linear.scaled_mm.pytorch import (
    PerTensorTorchFP8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    FP8ScaledMMLinearLayerConfig,
    Int8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
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

# The only backends that consume a pre-quantized activation.
SUPPORTING = {
    CutlassFP8ScaledMMLinearKernel,
    FlashInferFP8ScaledMMLinearKernel,
    FlashInferCutlassNvFp4LinearKernel,
    PerTensorTorchFP8ScaledMMLinearKernel,
    FlashInferCutedslMxfp8LinearKernel,
    FlashInferCutlassMxfp8LinearKernel,
}


def _all_kernel_classes() -> list[type]:
    seen: dict[type, None] = {}
    for registry in (
        _POSSIBLE_FP8_KERNELS,
        _POSSIBLE_FP8_BLOCK_KERNELS,
        _POSSIBLE_INT8_KERNELS,
        _POSSIBLE_NVFP4_KERNELS,
        _POSSIBLE_MXFP8_KERNELS,
    ):
        for kernels in registry.values():
            for cls in kernels:
                seen.setdefault(cls, None)
    return list(seen)


def _probe(cls: type):
    """A bare kernel instance with a plausible config, so input_quant_key()
    can be queried without the hardware-gated constructor."""
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


@pytest.mark.parametrize("scale", [0.125, 0.25, 0.0, -1.0, float("nan"), float("inf")])
@pytest.mark.parametrize("method_kind", ["modelopt", "fp8", "compressed_tensors"])
def test_gdn_shares_only_equal_valid_static_scales_and_rechecks_reload(
    scale, method_kind
):
    from vllm.model_executor.layers.linear import ColumnParallelLinear
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
        _shared_input_quant,
    )

    def projection(value):
        layer = ColumnParallelLinear.__new__(ColumnParallelLinear)
        torch.nn.Module.__init__(layer)
        kernel = _probe(CutlassFP8ScaledMMLinearKernel)
        kernel.quant_fp8 = torch.nn.Identity()
        kernel.quant_fp8.num_token_padding = None
        if method_kind == "compressed_tensors":
            layer.scheme = SimpleNamespace(fp8_linear=kernel)
            layer.quant_method = SimpleNamespace()
        else:
            attr = "kernel" if method_kind == "modelopt" else "fp8_linear"
            layer.quant_method = SimpleNamespace(**{attr: kernel})
        layer.input_scale = torch.tensor([value], dtype=torch.float32, device="cpu")
        return layer

    qkvz, ba = projection(scale), projection(scale)
    selected = _shared_input_quant(qkvz, ba)
    assert (selected is not None) == (scale > 0 and scale != float("inf"))
    assert qkvz.input_scale.data_ptr() != ba.input_scale.data_ptr()

    layer = SimpleNamespace(
        in_proj_qkvz=qkvz,
        in_proj_ba=ba,
        _allow_shared_input_quant=True,
        _concurrent_input_gemm_requested=False,
    )
    QwenGatedDeltaNetAttention.process_weights_after_loading(layer, torch.bfloat16)
    assert layer._shared_input_quant is selected
    ba.input_scale = torch.tensor([0.5], device="cpu")
    QwenGatedDeltaNetAttention.process_weights_after_loading(layer, torch.bfloat16)
    assert layer._shared_input_quant is None
    ba.input_scale = None  # Dynamic quantization cannot share a static input.
    assert _shared_input_quant(qkvz, ba) is None
    assert _shared_input_quant(None, ba) is None  # Separate/LoRA projections.


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
