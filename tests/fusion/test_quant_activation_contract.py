# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract tests for the QuantizedActivation linear-kernel integration."""

from dataclasses import replace

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
    FlashInferCuteDslNvFp4LinearKernel,
    FlashInferCutlassNvFp4LinearKernel,
    FlashInferTrtllmNvFp4LinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.aiter import (
    AiterHipbMMPerTokenFp8ScaledMMLinearKernel,
    AiterPerTokenFp8ScaledMMLinearKernel,
    AiterPreshuffledPerTokenFp8ScaledMMLinearKernel,
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
from vllm.model_executor.layers.activation import ReLUSquaredActivation, SiluAndMul
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    as_quantized_activation,
    expose_input_quant_key,
    get_fused_act_quant_key,
    get_input_quant_key,
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
    FlashInferCuteDslNvFp4LinearKernel,
    FlashInferCutlassNvFp4LinearKernel,
    PerTensorTorchFP8ScaledMMLinearKernel,
    AiterHipbMMPerTokenFp8ScaledMMLinearKernel,
    AiterPreshuffledPerTokenFp8ScaledMMLinearKernel,
    AiterPerTokenFp8ScaledMMLinearKernel,
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


@pytest.mark.parametrize(
    "kernel_cls",
    [FlashInferCuteDslNvFp4LinearKernel, FlashInferCutlassNvFp4LinearKernel],
)
def test_bridge_marks_supporting_and_skips_others(kernel_cls):
    supported = _probe(kernel_cls)
    layer = torch.nn.Module()
    expose_input_quant_key(layer, supported)
    assert get_input_quant_key(layer) == kNvfp4Dynamic
    layer.requires_unquantized_input = True
    assert get_input_quant_key(layer) is None

    unsupported = _probe(FlashInferTrtllmNvFp4LinearKernel)
    assert unsupported.input_quant_key() is None
    layer = torch.nn.Module()
    expose_input_quant_key(layer, unsupported)
    assert get_input_quant_key(layer) is None


@pytest.mark.parametrize("act_type", [ReLUSquaredActivation, SiluAndMul])
@pytest.mark.parametrize("compile_dispatch", [False, True])
def test_activation_policy_reexposure_preserves_guards(
    default_vllm_config, act_type, compile_dispatch
):
    """Re-exposure resets policy/key, and fullgraph guards observe every change."""

    class LegacyKernel:
        def input_quant_key(self):
            return kNvfp4Dynamic

    class NoManualFusion(LegacyKernel):
        def input_quant_activation_types(self):
            return ()

    layer = torch.nn.Module()
    act = act_type(compile_native=False)

    def select(x):
        return x + 1 if get_fused_act_quant_key(layer, act) is not None else x - 1

    dispatch = (
        torch.compile(select, backend="eager", fullgraph=True)
        if compile_dispatch
        else select
    )
    x = torch.ones(2, device="cpu")
    cute = _probe(FlashInferCuteDslNvFp4LinearKernel)
    for kernel, allowed in (
        (cute, act_type is ReLUSquaredActivation),
        (_probe(FlashInferCutlassNvFp4LinearKernel), act_type is SiluAndMul),
        (NoManualFusion(), False),
        (_probe(FlashInferTrtllmNvFp4LinearKernel), False),
        (cute, act_type is ReLUSquaredActivation),
        (LegacyKernel(), True),
    ):
        expose_input_quant_key(layer, kernel)
        assert get_input_quant_key(layer) == kernel.input_quant_key()
        assert (get_fused_act_quant_key(layer, act) is not None) is allowed
        torch.testing.assert_close(
            dispatch(x), x + (1 if allowed else -1), rtol=0, atol=0
        )
    layer.requires_unquantized_input = True
    assert get_input_quant_key(layer) is get_fused_act_quant_key(layer, act) is None
    torch.testing.assert_close(dispatch(x), x - 1, rtol=0, atol=0)


def test_cutedsl_policy_keeps_cutlass_silu_producer_selection(
    default_vllm_config, monkeypatch
):
    """The new restriction belongs to CuTe, not to the existing SiLU producer."""
    from vllm.model_executor.layers.fusion import fused_act_quant as fusion

    act = SiluAndMul(compile_native=False)
    act._forward_method = act.forward_native
    layer = torch.nn.Module()
    x = torch.ones((2, 4), device="cpu")
    qa = QuantizedActivation(
        data=torch.zeros((2, 1), dtype=torch.uint8, device="cpu"),
        scale=torch.ones((), device="cpu"),
        orig_dtype=x.dtype,
        orig_shape=torch.Size((2, 2)),
        quant_key=kNvfp4Dynamic,
    )
    key = (SiluAndMul, kNvfp4Dynamic)
    monkeypatch.setitem(fusion._FUSED_ACT_QUANT, key, lambda x, linear: qa)
    expose_input_quant_key(layer, _probe(FlashInferCuteDslNvFp4LinearKernel))
    actual = fusion.maybe_fused_act_quant(act, x, layer)
    assert isinstance(actual, torch.Tensor)
    torch.testing.assert_close(actual, act(x), rtol=0, atol=0)
    expose_input_quant_key(layer, _probe(FlashInferCutlassNvFp4LinearKernel))
    assert fusion.maybe_fused_act_quant(act, x, layer) is qa


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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cutedsl_qact_metadata_contract(dtype):
    from vllm.model_executor.kernels.linear.nvfp4.flashinfer import (
        _validate_cutedsl_quantized_activation,
    )

    layer = torch.nn.Module()
    layer.weight = torch.empty((160, 32), dtype=torch.uint8)
    layer.input_size_per_partition = 48
    qa = QuantizedActivation(
        data=torch.empty((3, 24), dtype=torch.uint8),
        scale=torch.empty((128, 4), dtype=torch.float8_e4m3fn),
        orig_dtype=dtype,
        orig_shape=torch.Size([3, 48]),
        quant_key=kNvfp4Dynamic,
    )
    _validate_cutedsl_quantized_activation(qa, layer)
    for bad in (
        replace(qa, scale=torch.empty((3, 3), dtype=torch.float8_e4m3fn)),
        replace(qa, scale=torch.empty((8, 4), dtype=torch.float8_e4m3fn)),
        replace(qa, scale=qa.scale.float()),
        replace(qa, data=torch.empty((3, 32), dtype=torch.uint8)),
        replace(qa, data=qa.data.float()),
        replace(qa, data=torch.empty((24, 3), dtype=torch.uint8).t()),
        replace(qa, orig_dtype=torch.float32),
        replace(qa, orig_shape=torch.Size([3, 64])),
    ):
        with pytest.raises(AssertionError):
            _validate_cutedsl_quantized_activation(bad, layer)
