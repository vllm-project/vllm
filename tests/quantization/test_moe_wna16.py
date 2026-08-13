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


@pytest.mark.parametrize("backend", [WNA16MoEBackend.MARLIN, WNA16MoEBackend.TRITON])
def test_wna16_oracle_handles_unset_group_size(backend):
    """A per-channel compressed-tensors checkpoint leaves group_size unset.

    `QuantizationArgs.group_size` is then None, and the Marlin support probe
    compares it against ints. The oracle must return a verdict (a reason string
    or None), not raise TypeError.
    """
    from tests.kernels.moe.utils import make_dummy_moe_config

    quant_config = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.CHANNEL,
        symmetric=True,
        dynamic=False,
    )
    assert quant_config.group_size is None, (
        "premise: CHANNEL strategy must leave group_size unset"
    )

    # Shapes matter: the default dummy config has hidden_dim=1, which fails the
    # `hidden_size % 128` check before group_size is ever compared, so the probe
    # would return a reason without exercising the None path.
    moe_config = make_dummy_moe_config(
        num_experts=8, hidden_dim=4096, intermediate_size=1024
    )

    reason = _backend_incompatibility_reason(
        backend=backend,
        moe_config=moe_config,
        quant_config=quant_config,
        may_have_zp=False,
        may_have_bias=False,
        allow_tile_padding=True,
    )
    assert reason is None or isinstance(reason, str)


def test_ct_wna16_moe_method_normalises_unset_group_size():
    """The loading path itself: constructing the quant method must not raise.

    This is what a user hits when serving a per-channel compressed-tensors MoE
    checkpoint -- `__init__` feeds group_size into the Marlin support probe.
    """
    from tests.kernels.moe.utils import make_dummy_moe_config
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16 import (  # noqa: E501
        CompressedTensorsWNA16MoEMethod,
    )

    quant_config = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.CHANNEL,
        symmetric=True,
        dynamic=False,
    )
    try:
        method = CompressedTensorsWNA16MoEMethod(
            weight_quant=quant_config,
            input_quant=None,
            moe=make_dummy_moe_config(
                num_experts=8, hidden_dim=4096, intermediate_size=1024
            ),
        )
    except AssertionError as exc:
        # Where the oracle does not select Marlin -- XPU, whose priority list is
        # [XPU] -- __init__ takes the non-Marlin branch and asserts
        # strategy == "group", a deliberate rejection of channelwise that this
        # change does not touch. The Marlin probe is never reached there, so
        # there is nothing for this test to observe.
        pytest.skip(f"oracle selected a non-Marlin backend on this platform: {exc}")

    assert method.group_size == -1


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="gptq_marlin_repack needs a GPU"
)
def test_marlin_weight_prep_accepts_unset_group_size():
    """Weight prep is a second, independent read of the same field.

    The probe fix alone is not enough: once MARLIN is selected, repacking reads
    group_size again to size the scale permutation.
    """
    num_experts, intermediate, hidden = 8, 1024, 4096
    quant_config = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.CHANNEL,
        symmetric=True,
        dynamic=False,
    )
    layer = torch.nn.Module()
    layer.intermediate_size_per_partition = intermediate
    layer.hidden_size = hidden
    layer.num_experts = num_experts

    def i32(*shape):
        return torch.randint(0, 255, shape, dtype=torch.int32, device="cuda")

    result = convert_to_wna16_moe_kernel_format(
        backend=WNA16MoEBackend.MARLIN,
        layer=layer,
        quant_config=quant_config,
        input_dtype=torch.bfloat16,
        w13=i32(num_experts, 2 * intermediate, hidden // 8),
        w2=i32(num_experts, hidden, intermediate // 8),
        w13_scale=torch.ones(
            num_experts, 2 * intermediate, 1, dtype=torch.bfloat16, device="cuda"
        ),
        w2_scale=torch.ones(
            num_experts, hidden, 1, dtype=torch.bfloat16, device="cuda"
        ),
        w13_g_idx=torch.empty(num_experts, 0, dtype=torch.int32, device="cuda"),
        w2_g_idx=torch.empty(num_experts, 0, dtype=torch.int32, device="cuda"),
    )
    assert result is not None
