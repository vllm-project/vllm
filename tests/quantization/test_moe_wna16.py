# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from compressed_tensors.quantization import (
    ActivationOrdering,
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)

from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
    WNA16MoEBackend,
    _backend_incompatibility_reason,
    _convert_moe_wna16_humming_tensors,
    convert_to_wna16_moe_kernel_format,
    map_wna16_backend,
)
from vllm.model_executor.layers.quantization import moe_wna16
from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig
from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQConfig
from vllm.model_executor.layers.quantization.moe_wna16 import (
    MoeWNA16Config,
    MoeWNA16Method,
)
from vllm.platforms import current_platform


def test_map_wna16_backend_supports_triton():
    assert map_wna16_backend("triton") == WNA16MoEBackend.TRITON


@pytest.mark.parametrize(
    ("backend", "quant_config", "may_have_zp", "may_have_bias", "expected"),
    [
        (
            WNA16MoEBackend.TRITON,
            AutoAWQConfig(4, 128, True, False),
            True,
            False,
            "AutoAWQ weight layout",
        ),
        (
            WNA16MoEBackend.TRITON,
            AutoGPTQConfig(4, 128, True, True, False, {}, {}),
            False,
            False,
            "activation ordering",
        ),
        (
            WNA16MoEBackend.TRITON,
            QuantizationArgs(
                num_bits=4,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.GROUP,
                symmetric=True,
                dynamic=False,
                group_size=128,
                actorder=ActivationOrdering.GROUP,
            ),
            False,
            False,
            "activation ordering",
        ),
        (
            WNA16MoEBackend.TRITON,
            AutoGPTQConfig(4, 128, False, True, False, {}, {}),
            False,
            True,
            "bias",
        ),
        (
            WNA16MoEBackend.MARLIN,
            MoeWNA16Config(
                linear_quant_method="gptq",
                weight_bits=4,
                group_size=128,
                has_zp=False,
                lm_head_quantized=False,
                modules_to_not_convert=None,
                full_config={},
            ),
            False,
            False,
            "MoeWNA16 checkpoint layout",
        ),
    ],
)
def test_wna16_oracle_rejects_incompatible_quant_structures(
    backend, quant_config, may_have_zp, may_have_bias, expected
):
    from tests.kernels.moe.utils import make_dummy_moe_config

    moe_config = make_dummy_moe_config()

    reason = _backend_incompatibility_reason(
        backend=backend,
        moe_config=moe_config,
        quant_config=quant_config,
        may_have_zp=may_have_zp,
        may_have_bias=may_have_bias,
        allow_tile_padding=True,
    )

    assert reason is not None
    assert expected in reason


def test_compressed_tensors_weights_are_transposed_for_triton():
    quant_config = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.GROUP,
        symmetric=True,
        dynamic=False,
        group_size=32,
    )
    w13 = torch.arange(16, dtype=torch.int32).reshape(1, 2, 8)
    w2 = torch.arange(12, dtype=torch.int32).reshape(1, 2, 6)
    w13_scale = torch.arange(32, dtype=torch.float16).reshape(1, 4, 8)
    w2_scale = torch.arange(18, dtype=torch.float16).reshape(1, 3, 6)

    converted = convert_to_wna16_moe_kernel_format(
        backend=WNA16MoEBackend.TRITON,
        layer=torch.nn.Module(),
        quant_config=quant_config,
        input_dtype=None,
        w13=w13,
        w2=w2,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
    )

    assert converted is not None
    assert torch.equal(converted[0], w13.transpose(1, 2).contiguous().view(torch.uint8))
    assert torch.equal(converted[1], w2.transpose(1, 2).contiguous().view(torch.uint8))
    assert torch.equal(converted[2], w13_scale.transpose(1, 2).contiguous())
    assert torch.equal(converted[3], w2_scale.transpose(1, 2).contiguous())


def test_moe_wna16_setup_forwards_selected_backend(monkeypatch):
    method = object.__new__(MoeWNA16Method)
    method.experts_cls = object
    method.wna16_backend = WNA16MoEBackend.HUMMING
    method.moe = object()
    quant_config = object()
    method.get_fused_moe_quant_config = lambda layer: quant_config
    layer = SimpleNamespace(_expert_routing_tables=lambda: (None, None, None))
    captured = {}
    kernel = object()

    def fake_make_wna16_moe_kernel(**kwargs):
        captured.update(kwargs)
        return kernel

    monkeypatch.setattr(moe_wna16, "make_wna16_moe_kernel", fake_make_wna16_moe_kernel)

    method._setup_kernel(layer)

    assert method.moe_kernel is kernel
    assert captured["backend"] == WNA16MoEBackend.HUMMING


def test_moe_wna16_humming_adapter_repacks_uint8_tensors():
    qweight = torch.arange(32, dtype=torch.uint8).reshape(1, 4, 8)
    scales = torch.arange(16, dtype=torch.float16).reshape(1, 4, 4)
    qzeros = torch.arange(16, dtype=torch.uint8).reshape(1, 8, 2)

    converted = _convert_moe_wna16_humming_tensors(
        {"qweight": qweight, "scales": scales, "qzeros": qzeros},
        has_zero_point=True,
    )

    assert torch.equal(converted["weight"], qweight.view(torch.int32))
    assert converted["weight"].shape == (1, 4, 2)
    assert torch.equal(converted["weight_scale"], scales)
    expected_qzeros = (
        qzeros.transpose(-1, -2)
        .contiguous()
        .view(torch.int32)
        .transpose(-1, -2)
        .contiguous()
    )
    assert torch.equal(converted["zero_point"], expected_qzeros)
    assert converted["zero_point"].shape == (1, 2, 2)


def test_moe_wna16_uses_humming_quant_config(monkeypatch):
    from vllm.model_executor.layers.quantization.utils import humming_utils

    method = object.__new__(MoeWNA16Method)
    method.wna16_backend = WNA16MoEBackend.HUMMING
    layer = object()
    quant_config = object()
    monkeypatch.setattr(
        humming_utils,
        "get_humming_moe_quant_config",
        lambda actual_layer, *args, **kwargs: (
            quant_config if actual_layer is layer else None
        ),
    )

    assert method.get_fused_moe_quant_config(layer) is quant_config


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="Compressed-tensors Humming WNA16 MoE requires CUDA",
)
@pytest.mark.parametrize("num_bits", [3, 5, 6, 7])
def test_compressed_tensors_wna16_moe_create_weights_uses_ceil_packed_shapes(
    num_bits,
):
    pytest.importorskip("humming")

    from tests.kernels.moe.utils import make_dummy_moe_config
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16 import (  # noqa: E501
        CompressedTensorsWNA16MoEMethod,
    )

    quant_args = QuantizationArgs(
        num_bits=num_bits,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.GROUP,
        symmetric=True,
        dynamic=False,
        group_size=128,
    )
    moe_config = make_dummy_moe_config(
        num_experts=2,
        hidden_dim=256,
        intermediate_size=512,
    )
    moe_config.moe_backend = "humming"
    method = CompressedTensorsWNA16MoEMethod(quant_args, None, moe_config)
    layer = torch.nn.Module()

    method.create_weights(
        layer,
        num_experts=2,
        hidden_size=256,
        intermediate_size_per_partition=512,
        params_dtype=torch.float16,
        intermediate_size_full=512,
    )

    packed_hidden = (256 * num_bits + 31) // 32
    packed_intermediate = (512 * num_bits + 31) // 32
    assert method.wna16_backend == WNA16MoEBackend.HUMMING
    assert layer.w13_weight_packed.shape == (2, 1024, packed_hidden)
    assert layer.w2_weight_packed.shape == (2, 256, packed_intermediate)
    assert layer.w13_weight_scale.shape == (2, 1024, 2)
    assert layer.w2_weight_scale.shape == (2, 256, 4)
    assert layer.w13_weight_packed.dtype is torch.int32
    assert layer.w2_weight_scale.dtype is torch.float16


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="Compressed-tensors Humming WNA16 MoE requires CUDA",
)
def test_compressed_tensors_wna16_moe_converts_and_sets_up_humming_kernel():
    pytest.importorskip("humming")

    from tests.kernels.moe.utils import make_dummy_moe_config
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16 import (  # noqa: E501
        CompressedTensorsWNA16MoEMethod,
    )

    quant_args = QuantizationArgs(
        num_bits=3,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.GROUP,
        symmetric=True,
        dynamic=False,
        group_size=128,
    )
    moe_config = make_dummy_moe_config(
        num_experts=2,
        hidden_dim=256,
        intermediate_size=512,
    )
    moe_config.moe_backend = "humming"
    method = CompressedTensorsWNA16MoEMethod(quant_args, None, moe_config)
    layer = torch.nn.Module()
    layer.moe_config = moe_config
    layer.params_dtype = torch.bfloat16
    layer.layer_name = "test.humming_moe"
    layer._expert_routing_tables = lambda: (None, None, None)

    method.create_weights(
        layer,
        num_experts=2,
        hidden_size=256,
        intermediate_size_per_partition=512,
        intermediate_size_full=512,
        params_dtype=torch.bfloat16,
    )
    layer.cuda()
    for parameter in layer.parameters():
        parameter.data.zero_()

    method.process_weights_after_loading(layer)

    assert method.wna16_backend == WNA16MoEBackend.HUMMING
    assert method.moe_kernel is not None
    assert set(layer.weight_schemas) == {"w13", "w2"}
    assert set(layer.humming_configs) == {"w13", "w2"}
    assert not hasattr(layer, "w13_weight_packed")
    assert not hasattr(layer, "w2_weight_packed")
    assert layer.w13_weight.dtype is torch.int32
    assert layer.w2_weight.dtype is torch.int32
