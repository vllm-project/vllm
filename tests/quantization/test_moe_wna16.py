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

from vllm.model_executor.layers.fused_moe.oracle import int_wna16
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
    WNA16MoEBackend,
    _backend_incompatibility_reason,
    _convert_moe_wna16_humming_tensors,
    _process_awq_weights_marlin,
    _process_weights_marlin,
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


@pytest.mark.parametrize("quant_method", ["awq", "gptq"])
@pytest.mark.parametrize(
    ("tp_size", "intermediate_size"), [(2, 512), (4, 256), (8, 128)]
)
def test_marlin_moe_w13_scale_permutation_uses_weight_k(
    monkeypatch, quant_method, tp_size, intermediate_size
):
    hidden_size = 4096
    group_size = 128
    pack_factor = 8
    num_experts = 1
    w13_scales = torch.arange(
        num_experts * (hidden_size // group_size) * (2 * intermediate_size),
        dtype=torch.float32,
    ).reshape(num_experts, hidden_size // group_size, 2 * intermediate_size)
    w2_scales = torch.arange(
        num_experts * (intermediate_size // group_size) * hidden_size,
        dtype=torch.float32,
    ).reshape(num_experts, intermediate_size // group_size, hidden_size)
    layer = SimpleNamespace(
        intermediate_size_per_partition=intermediate_size,
        num_groups_w13=hidden_size // group_size,
        num_groups_w2=intermediate_size // group_size,
    )

    permute_scales = int_wna16.marlin_moe_permute_scales
    observed_size_ks = []

    def capture_size_k(*, s, size_k, size_n, group_size, is_a_8bit=False):
        observed_size_ks.append(size_k)
        return permute_scales(
            s=s,
            size_k=size_k,
            size_n=size_n,
            group_size=group_size,
            is_a_8bit=is_a_8bit,
        )

    monkeypatch.setattr(int_wna16, "marlin_moe_permute_scales", capture_size_k)

    if quant_method == "awq":
        monkeypatch.setattr(
            int_wna16.ops,
            "awq_marlin_moe_repack",
            lambda qweight, *_args, **_kwargs: qweight,
        )
        monkeypatch.setattr(
            int_wna16,
            "moe_awq_to_marlin_zero_points",
            lambda qzeros, **_kwargs: qzeros,
        )
        w13_qweight = torch.empty(
            num_experts,
            hidden_size,
            2 * intermediate_size // pack_factor,
            dtype=torch.int32,
        )
        w2_qweight = torch.empty(
            num_experts,
            intermediate_size,
            hidden_size // pack_factor,
            dtype=torch.int32,
        )
        w13_qzeros = torch.empty(
            num_experts,
            hidden_size // group_size,
            2 * intermediate_size // pack_factor,
            dtype=torch.int32,
        )
        w2_qzeros = torch.empty(
            num_experts,
            intermediate_size // group_size,
            hidden_size // pack_factor,
            dtype=torch.int32,
        )
        converted = _process_awq_weights_marlin(
            layer,
            4,
            pack_factor,
            group_size,
            None,
            w13_qweight,
            w2_qweight,
            w13_scales,
            w2_scales,
            w13_qzeros,
            w2_qzeros,
        )
    else:
        monkeypatch.setattr(
            int_wna16.ops,
            "gptq_marlin_moe_repack",
            lambda qweight, *_args, **_kwargs: qweight,
        )
        w13_qweight = torch.empty(
            num_experts,
            hidden_size // pack_factor,
            2 * intermediate_size,
            dtype=torch.int32,
        )
        w2_qweight = torch.empty(
            num_experts,
            intermediate_size // pack_factor,
            hidden_size,
            dtype=torch.int32,
        )
        converted = _process_weights_marlin(
            layer,
            None,
            4,
            pack_factor,
            group_size,
            None,
            w13_qweight,
            w2_qweight,
            w13_scales,
            w2_scales,
            torch.empty(num_experts, hidden_size, dtype=torch.int32),
            torch.empty(num_experts, intermediate_size, dtype=torch.int32),
        )

    correctly_permuted_w13 = permute_scales(
        s=w13_scales,
        size_k=hidden_size,
        size_n=w13_scales.shape[2],
        group_size=group_size,
    )
    old_w13_permutation = permute_scales(
        s=w13_scales,
        size_k=intermediate_size,
        size_n=w13_scales.shape[2],
        group_size=group_size,
    )

    assert observed_size_ks == [hidden_size, intermediate_size]
    assert torch.equal(converted[2], correctly_permuted_w13)
    assert torch.equal(old_w13_permutation, correctly_permuted_w13) == (tp_size < 8)
