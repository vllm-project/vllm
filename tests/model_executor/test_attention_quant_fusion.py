# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)

import vllm.model_executor.layers.attention.attention as attention_module
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import (  # noqa: E501
    CompressedTensorsW8A8Fp8,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptFp8Config,
    ModelOptFp8LinearMethod,
)
from vllm.model_executor.layers.quantization.quark.schemes.quark_w8a8_fp8 import (  # noqa: E501
    QuarkW8A8Fp8,
)
from vllm.model_executor.layers.quantization.utils.quant_fusion import (
    get_static_fp8_attn_output_scale,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTokenSym,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform


def _noop_weight_loader(*args, **kwargs):
    pass


@pytest.fixture
def current_vllm_config(monkeypatch):
    import vllm.model_executor.parameter as parameter_module

    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    vllm_config = VllmConfig()
    vllm_config.model_config = SimpleNamespace(dtype=torch.float16)
    with set_current_vllm_config(vllm_config):
        yield


class _AttentionImpl:
    def __init__(self, supported: bool = True) -> None:
        self.supported = supported
        self.seen_quant_key = None

    def fused_output_quant_supported(self, quant_key):
        self.seen_quant_key = quant_key
        return self.supported


def _linear(activation_quant_key=kFp8StaticTensorSym):
    return SimpleNamespace(
        input_scale=torch.tensor([0.5]),
        input_quant_key=activation_quant_key,
    )


def _attention(supported: bool = True, fuse_attn_quant: bool = True):
    attn = object.__new__(attention_module.Attention)
    attn.calculate_kv_scales = False
    attn.query_quant = None
    attn.num_heads = 2
    attn.num_kv_heads = 2
    attn.head_size = 4
    attn.head_size_v = 4
    attn.use_direct_call = True
    attn.attn_backend = SimpleNamespace(forward_includes_kv_cache_update=True)
    attn.kv_sharing_target_layer_name = None
    attn.layer_name = "test_attn"
    attn.impl = _AttentionImpl(supported=supported)
    attn.use_fused_attn_quant = fuse_attn_quant
    return attn


def _create_standard_linear_layer(quant_method):
    layer = torch.nn.Module()
    quant_method.create_weights(
        layer=layer,
        input_size_per_partition=8,
        output_partition_sizes=[8],
        input_size=8,
        output_size=8,
        params_dtype=torch.float16,
        weight_loader=_noop_weight_loader,
    )
    return layer


def _create_quark_linear_layer(quant_method):
    layer = torch.nn.Module()
    quant_method.create_weights(
        layer=layer,
        output_partition_sizes=[8],
        input_size_per_partition=8,
        params_dtype=torch.float16,
        weight_loader=_noop_weight_loader,
    )
    return layer


def test_static_fp8_attn_output_scale_selected_when_supported():
    attn = _attention()
    output_proj = _linear()

    scale = get_static_fp8_attn_output_scale(attn, output_proj)

    assert scale is output_proj.input_scale
    assert attn.impl.seen_quant_key == kFp8StaticTensorSym


def test_static_fp8_attn_output_scale_skipped_without_input_quant_key():
    attn = _attention()
    output_proj = SimpleNamespace(input_scale=torch.tensor([0.5]))

    scale = get_static_fp8_attn_output_scale(attn, output_proj)

    assert scale is None
    assert attn.impl.seen_quant_key is None


def test_static_fp8_attn_output_scale_skipped_when_flag_disabled():
    attn = _attention(fuse_attn_quant=False)
    output_proj = _linear()

    scale = get_static_fp8_attn_output_scale(attn, output_proj)

    assert scale is None
    assert attn.impl.seen_quant_key is None


def test_static_fp8_attn_output_scale_skipped_when_backend_unsupported():
    attn = _attention(supported=False)
    output_proj = _linear()

    scale = get_static_fp8_attn_output_scale(attn, output_proj)

    assert scale is None


def test_static_fp8_attn_output_scale_skipped_when_backend_has_no_support_probe():
    attn = _attention()
    attn.impl = SimpleNamespace()
    output_proj = _linear()

    scale = get_static_fp8_attn_output_scale(attn, output_proj)

    assert scale is None


def test_static_fp8_attn_output_scale_skipped_for_non_static_fp8_linear():
    attn = _attention()
    output_proj = _linear(activation_quant_key=kFp8DynamicTokenSym)

    scale = get_static_fp8_attn_output_scale(attn, output_proj)

    assert scale is None
    assert attn.impl.seen_quant_key is None


def test_static_fp8_attn_output_scale_skipped_without_input_scale():
    attn = _attention()
    output_proj = SimpleNamespace(input_quant_key=kFp8StaticTensorSym)

    scale = get_static_fp8_attn_output_scale(attn, output_proj)

    assert scale is None
    assert attn.impl.seen_quant_key is None


@pytest.mark.parametrize(
    "layer_factory",
    [
        pytest.param(
            lambda: _create_standard_linear_layer(
                CompressedTensorsW8A8Fp8(
                    QuantizationArgs(
                        num_bits=8,
                        type=QuantizationType.FLOAT,
                        strategy=QuantizationStrategy.TENSOR,
                    ),
                    is_static_input_scheme=True,
                )
            ),
            id="compressed_tensors",
        ),
        pytest.param(
            lambda: _create_standard_linear_layer(
                Fp8LinearMethod(
                    Fp8Config(
                        is_checkpoint_fp8_serialized=True,
                        activation_scheme="static",
                    )
                )
            ),
            id="native_fp8",
        ),
        pytest.param(
            lambda: _create_standard_linear_layer(
                ModelOptFp8LinearMethod(
                    ModelOptFp8Config(
                        quant_method="FP8",
                        is_checkpoint_fp8_serialized=True,
                        kv_cache_quant_method=None,
                        exclude_modules=[],
                    )
                )
            ),
            id="modelopt",
        ),
        pytest.param(
            lambda: _create_quark_linear_layer(
                QuarkW8A8Fp8(
                    weight_config={"qscheme": "per_tensor"},
                    input_config={"is_dynamic": False, "qscheme": "per_tensor"},
                )
            ),
            id="quark",
        ),
    ],
)
def test_static_fp8_linear_schemes_mark_input_quant_key(
    current_vllm_config, layer_factory
):
    layer = layer_factory()

    assert layer.input_quant_key == kFp8StaticTensorSym
    assert layer.input_scale is not None


def test_fp8_attention_output_feeds_static_fp8_linear_without_requant(
    current_vllm_config, monkeypatch
):
    quant_method = Fp8LinearMethod(
        Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="static",
        )
    )
    output_proj = _create_standard_linear_layer(quant_method)
    attn = _attention()
    captured = {}

    class FailingQuant(torch.nn.Module):
        def forward(self, *args, **kwargs):
            raise AssertionError("FP8 attention output should not be requantized")

    def fake_apply_scaled_mm(*, A, B, out_dtype, As, Bs, bias, output_shape):
        captured["input_dtype"] = A.dtype
        captured["input_scale"] = As
        return torch.empty(output_shape, dtype=out_dtype, device=A.device)

    monkeypatch.setattr(quant_method.fp8_linear, "quant_fp8", FailingQuant())
    monkeypatch.setattr(
        quant_method.fp8_linear, "apply_scaled_mm", fake_apply_scaled_mm
    )

    output_scale = get_static_fp8_attn_output_scale(attn, output_proj)
    attn_output = torch.empty(3, 8, dtype=current_platform.fp8_dtype())
    output = quant_method.apply(output_proj, attn_output)

    assert output_scale is output_proj.input_scale
    assert captured["input_dtype"] == current_platform.fp8_dtype()
    assert captured["input_scale"] is output_proj.input_scale
    assert output.shape == torch.Size([3, 8])


def test_attention_forward_uses_fp8_output_when_scale_provided(monkeypatch):
    captured = {}

    def fake_unified_attention_with_output(
        query,
        key,
        value,
        output,
        layer_name,
        output_scale=None,
        output_block_scale=None,
        kv_cache_dummy_dep=None,
    ):
        captured["output_dtype"] = output.dtype
        captured["output_scale"] = output_scale

    monkeypatch.setattr(
        attention_module,
        "unified_attention_with_output",
        fake_unified_attention_with_output,
    )

    attn = _attention()
    q = torch.randn(3, 8)
    k = torch.randn(3, 8)
    v = torch.randn(3, 8)
    scale = torch.tensor([0.5])

    result = attention_module.Attention.forward(attn, q, k, v, output_scale=scale)

    assert captured["output_scale"] is scale
    assert captured["output_dtype"] == current_platform.fp8_dtype()
    assert result.dtype == current_platform.fp8_dtype()


def test_attention_forward_uses_fp8_output_with_torch_ops_path(monkeypatch):
    captured = {}

    def fake_unified_attention_with_output(
        query,
        key,
        value,
        output,
        layer_name,
        output_scale=None,
        output_block_scale=None,
        kv_cache_dummy_dep=None,
    ):
        captured["output_dtype"] = output.dtype
        captured["output_scale"] = output_scale

    monkeypatch.setattr(
        torch.ops.vllm,
        "unified_attention_with_output",
        fake_unified_attention_with_output,
    )

    attn = _attention()
    attn.use_direct_call = False
    q = torch.randn(3, 8)
    k = torch.randn(3, 8)
    v = torch.randn(3, 8)
    scale = torch.tensor([0.5])

    result = attention_module.Attention.forward(attn, q, k, v, output_scale=scale)

    assert captured["output_scale"] is scale
    assert captured["output_dtype"] == current_platform.fp8_dtype()
    assert result.dtype == current_platform.fp8_dtype()


def test_attention_forward_rejects_backend_without_support_probe():
    attn = _attention()
    attn.impl = SimpleNamespace()
    q = torch.randn(3, 8)
    k = torch.randn(3, 8)
    v = torch.randn(3, 8)
    scale = torch.tensor([0.5])

    with pytest.raises(ValueError, match="static FP8 attention output quantization"):
        attention_module.Attention.forward(attn, q, k, v, output_scale=scale)
