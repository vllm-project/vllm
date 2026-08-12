# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract tests for the QuantizedActivation linear-kernel integration."""

import pytest
import torch

import vllm.model_executor.kernels.linear.scaled_mm.deep_gemm as deep_gemm
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
from vllm.model_executor.kernels.linear.scaled_mm.BlockScaledMMLinearKernel import (
    Fp8BlockScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
    CutlassFP8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.deep_gemm import (
    DeepGemmFp8BlockScaledMMKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.flashinfer import (
    FlashInferFP8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    FP8ScaledMMLinearLayerConfig,
    Int8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.fusion.fused_act_quant import (
    maybe_allocate_fp8_block_quant,
)
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    as_quantized_activation,
    expose_input_quant_key,
    index_quantized_activation,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
    kFp8StaticTensorSym,
    kNvfp4Dynamic,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import DeepGemmQuantScaleFMT

# The only backends that consume a pre-quantized activation.
SUPPORTING = {
    CutlassFP8ScaledMMLinearKernel,
    DeepGemmFp8BlockScaledMMKernel,
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
    """A bare kernel instance with a plausible config, so input_quant_key()
    can be queried without the hardware-gated constructor."""
    obj = cls.__new__(cls)  # type: ignore[call-overload]
    if issubclass(cls, NvFp4LinearKernel):
        obj.config = NvFp4LinearLayerConfig()
    elif issubclass(cls, Int8ScaledMMLinearKernel):
        obj.config = Int8ScaledMMLinearLayerConfig(
            is_static_input_scheme=True, is_channelwise=False, input_symmetric=True
        )
    elif issubclass(cls, Fp8BlockScaledMMLinearKernel):
        obj.config = FP8ScaledMMLinearLayerConfig(
            weight_quant_key=kFp8Static128BlockSym,
            activation_quant_key=kFp8Dynamic128Sym,
            weight_shape=(128, 128),
            input_dtype=torch.bfloat16,
            out_dtype=torch.bfloat16,
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


def test_deepgemm_custom_op_hides_compiler_scale_storage_padding(monkeypatch):
    q_input = torch.empty(3, 128)
    input_scale = torch.empty_strided((4, 1), (1, 4), dtype=torch.int32)
    weight = torch.empty(128, 128)
    weight_scale = torch.empty(1, 1)
    output = torch.empty(3, 128)
    received_scale = None

    def fake_fp8_gemm_nt(a, b, out, *, is_deep_gemm_e8m0_used):
        nonlocal received_scale
        received_scale = a[1]

    monkeypatch.setattr(deep_gemm, "fp8_gemm_nt", fake_fp8_gemm_nt)
    deep_gemm._fp8_gemm_nt_op(
        q_input,
        input_scale,
        weight,
        weight_scale,
        output,
        True,
    )

    assert received_scale is not None
    assert received_scale.shape == (3, 1)
    assert received_scale.stride() == input_scale.stride()


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


def test_shared_fp8_block_quant_allocation_requires_matching_consumers(monkeypatch):
    monkeypatch.setattr(
        DeepGemmQuantScaleFMT,
        "from_oracle",
        lambda: DeepGemmQuantScaleFMT.UE8M0,
    )
    x = torch.randn(5, 256, dtype=torch.bfloat16)
    first = torch.nn.Module()
    first.input_quant_key = kFp8Dynamic128Sym
    second = torch.nn.Module()
    second.input_quant_key = kFp8Dynamic128Sym

    qa = maybe_allocate_fp8_block_quant(x, first, second)

    assert qa is not None
    assert qa.data.shape == x.shape
    assert qa.scale.shape == (5, 1)
    assert qa.scale.stride() == (1, 8)
    assert qa.orig_shape == x.shape
    assert qa.quant_key == kFp8Dynamic128Sym

    second.input_quant_key = kFp8StaticTensorSym
    assert maybe_allocate_fp8_block_quant(x, first, second) is None


def test_index_fp8_block_activation_repacks_scale_layout():
    data = torch.arange(30).view(5, 6)
    scale = torch.empty_strided((5, 2), (1, 8), dtype=torch.int32)
    scale.copy_(torch.arange(10).view(5, 2))
    activation = QuantizedActivation(
        data=data,
        scale=scale,
        orig_dtype=torch.bfloat16,
        orig_shape=data.shape,
        quant_key=kFp8Dynamic128Sym,
    )

    indexed = index_quantized_activation(activation, torch.tensor([4, 1, 3]))

    assert isinstance(indexed, QuantizedActivation)
    torch.testing.assert_close(indexed.data, data[[4, 1, 3]])
    torch.testing.assert_close(indexed.scale, scale[[4, 1, 3]])
    assert indexed.scale.stride() == (1, 4)
    assert indexed.orig_shape == (3, 6)


def test_logits_processor_forwards_quantized_activation(default_vllm_config):
    activation = QuantizedActivation(
        data=torch.empty(2, 4),
        scale=torch.empty(2, 1),
        orig_dtype=torch.bfloat16,
        orig_shape=torch.Size([2, 4]),
        quant_key=kFp8Dynamic128Sym,
    )

    class RecordingQuantMethod:
        received = None

        def apply(self, layer, hidden_states, bias=None):
            self.received = hidden_states
            return torch.ones(2, 3)

    head = torch.nn.Module()
    head.quant_method = RecordingQuantMethod()
    head.tp_size = 1

    logits = LogitsProcessor(3)(head, activation)

    assert head.quant_method.received is activation
    torch.testing.assert_close(logits, torch.ones(2, 3))
