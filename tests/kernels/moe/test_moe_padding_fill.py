# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Padding refilled by a direct reload of a padded MoE layer (CPU)."""

import importlib
import importlib.util
from collections.abc import Callable
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm.distributed as distributed
from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe import routed_experts
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.quantization.quark import quark_moe
from vllm.model_executor.layers.quantization.utils import quant_utils as qu
from vllm.model_executor.model_loader.reload.direct import (
    begin_direct_reload,
    record_checkpoint_layout,
)
from vllm.utils.math_utils import round_up


def _no_backend(*args, **kwargs):
    return None, None


@pytest.fixture
def mock_platform(monkeypatch):
    """Build quant methods without selecting a kernel or a model config."""
    from vllm.model_executor.layers.quantization import fp8
    from vllm.model_executor.layers.quantization.compressed_tensors import (
        compressed_tensors_moe as ct,
    )

    monkeypatch.setattr(
        quark_moe,
        "get_current_vllm_config",
        lambda: SimpleNamespace(model_config=SimpleNamespace(hf_config=None)),
    )
    monkeypatch.setattr(quark_moe.rocm_aiter_ops, "is_fused_moe_enabled", lambda: True)
    for module in (quark_moe, fp8, *_ct_modules(ct)):
        for name in dir(module):
            if name.startswith("select_") and name.endswith("_backend"):
                monkeypatch.setattr(module, name, _no_backend)
    monkeypatch.setattr(
        quark_moe,
        "select_mxfp4_moe_backend",
        lambda *args, **kwargs: (quark_moe.Mxfp4MoeBackend.NONE, None),
    )
    # Block-shape validation reads the TP size.
    monkeypatch.setattr(fp8, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(distributed, "get_tensor_model_parallel_world_size", lambda: 1)


def _ct_modules(ct):
    return [
        importlib.import_module(f"{ct.__name__}.compressed_tensors_moe_{suffix}")
        for suffix in (
            "w4a8_fp8",
            "w4a8_int8",
            "w8a8_fp8",
            "w8a8_int8",
            "wna16",
        )
    ]


def _make_experts(
    monkeypatch,
    make_method: Callable[[RoutedExperts], object],
    *,
    num_experts: int = 2,
    hidden: int = 256,
    intermediate: int = 256,
) -> RoutedExperts:
    monkeypatch.setattr(
        routed_experts,
        "resolve_quant_method",
        lambda quant_config, layer, prefix: make_method(layer),
    )
    config = make_dummy_moe_config(
        num_experts=num_experts,
        hidden_dim=hidden,
        intermediate_size=intermediate,
        experts_per_token=2,
    )
    config.device = "cpu"
    return RoutedExperts(
        "model.layers.0.mlp.experts",
        torch.bfloat16,
        config,
        SimpleNamespace(get_name=lambda: "test"),
        expert_map_manager=SimpleNamespace(
            local_num_experts=num_experts,
            num_fused_shared_experts=0,
            placement_strategy="linear",
            expert_map=None,
            expert_mask=None,
            routing_tables=None,
            map_global_to_local=lambda index: index,
        ),
    )


def _ct(method_cls: str, weight: dict, input_: dict | None):
    def make(layer):
        from compressed_tensors.quantization import QuantizationArgs

        from vllm.model_executor.layers.quantization.compressed_tensors import (
            compressed_tensors_moe as ct,
        )

        modules = _ct_modules(ct)
        if method_cls == "CompressedTensorsW4A16FlydslMoEMethod":
            from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
                compressed_tensors_moe_w4a16_flydsl as flydsl,
            )

            modules.append(flydsl)
        classes = {
            name: cls for module in modules for name, cls in vars(module).items()
        }
        return classes[method_cls](
            QuantizationArgs(**weight),
            None if input_ is None else QuantizationArgs(**input_),
            layer.moe_config,
        )

    return make


def _quark(method_cls: str, *keys):
    return lambda layer: getattr(quark_moe, method_cls)(layer.moe_config, *keys)


def _fp8_block(layer):
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    config = Fp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128])
    return config.get_quant_method(layer, layer.layer_name)


_INT4_GROUP = dict(num_bits=4, type="int", strategy="group", symmetric=True)
_FP8_TOKEN = dict(num_bits=8, type="float", strategy="token", dynamic=True)
_INT8_TOKEN = dict(num_bits=8, type="int", strategy="token", dynamic=True)

QUANT_METHODS = [
    pytest.param(_fp8_block, id="fp8_block"),
    pytest.param(
        _ct(
            "CompressedTensorsW4A16FlydslMoEMethod",
            {**_INT4_GROUP, "group_size": 32},
            None,
        ),
        id="ct_w4a16_flydsl",
        marks=pytest.mark.skipif(
            importlib.util.find_spec("aiter") is None,
            reason="its module imports AITER (ROCm) at import time",
        ),
    ),
    pytest.param(
        _ct(
            "CompressedTensorsW4A8Fp8MoEMethod",
            {**_INT4_GROUP, "group_size": 128},
            _FP8_TOKEN,
        ),
        id="ct_w4a8_fp8",
    ),
    pytest.param(
        _ct(
            "CompressedTensorsW4A8Int8MoEMethod",
            dict(num_bits=4, type="int", strategy="channel", symmetric=True),
            _INT8_TOKEN,
        ),
        id="ct_w4a8_int8",
    ),
    pytest.param(
        _ct(
            "CompressedTensorsW8A8Fp8MoEMethod",
            dict(num_bits=8, type="float", strategy="channel", symmetric=True),
            _FP8_TOKEN,
        ),
        id="ct_w8a8_fp8_channel",
    ),
    pytest.param(
        _ct(
            "CompressedTensorsW8A8Fp8MoEMethod",
            dict(
                num_bits=8,
                type="float",
                strategy="block",
                block_structure=[128, 128],
                symmetric=True,
            ),
            dict(
                num_bits=8, type="float", strategy="group", group_size=128, dynamic=True
            ),
        ),
        id="ct_w8a8_fp8_block",
    ),
    pytest.param(
        _ct(
            "CompressedTensorsW8A8Int8MoEMethod",
            dict(num_bits=8, type="int", strategy="channel", symmetric=True),
            _INT8_TOKEN,
        ),
        id="ct_w8a8_int8",
    ),
    pytest.param(
        _ct(
            "CompressedTensorsWNA16MoEMethod", {**_INT4_GROUP, "group_size": 128}, None
        ),
        id="ct_wna16",
    ),
    pytest.param(
        _quark(
            "QuarkW8A8Fp8MoEMethod", qu.kFp8StaticChannelSym, qu.kFp8DynamicTokenSym
        ),
        id="quark_w8a8_fp8_channel",
    ),
    pytest.param(
        _quark("QuarkW8A8Fp8MoEMethod", qu.kFp8Static128BlockSym, qu.kFp8Dynamic128Sym),
        id="quark_w8a8_fp8_block",
    ),
    pytest.param(
        _quark(
            "QuarkW8A8Int8MoEMethod", qu.kInt8StaticChannelSym, qu.kInt8DynamicTokenSym
        ),
        id="quark_w8a8_int8_channel",
    ),
    pytest.param(
        _quark("QuarkW4A8Fp8MoEMethod", qu.kFp8DynamicTokenSym), id="quark_w4a8_fp8"
    ),
    pytest.param(
        _quark("QuarkOCP_MX_MoEMethod", qu.kMxfp4Static, None), id="quark_ocp_mx"
    ),
]


@pytest.mark.parametrize("make_method", QUANT_METHODS)
def test_padding_value_marks_ones_initialized_params(
    monkeypatch, mock_platform, make_method
):
    """A padded layer refills each parameter with its initial padding: 1 for
    the ones-initialized scales, 0 for everything else."""
    layer = _make_experts(monkeypatch, make_method)
    layer.moe_config.hidden_dim_unpadded = layer.moe_config.hidden_dim - 1
    fill = dict(layer.padding_fill_params())
    assert any(value == 1 for value in fill.values()), fill
    for name, value in fill.items():
        ones = bool(torch.all(getattr(layer, name) == 1))
        assert (value == 1) == ones, (name, value, ones)


def _make_inkling_moe(layer: RoutedExperts):
    from vllm.models.inkling.amd.moe import InklingMoE

    moe = InklingMoE.__new__(InklingMoE)
    nn.Module.__init__(moe)
    moe.n_routed_experts = layer.global_num_experts
    moe.experts = nn.Module()
    moe.experts.routed_experts = layer
    slots = {e: e for e in range(layer.global_num_experts)}
    moe._local_expert_slots = lambda: slots  # type: ignore[method-assign]
    return moe


def test_inkling_direct_reload_after_dummy_start_resets_aiter_padding(
    monkeypatch, mock_platform
):
    """After a dummy cold start (no loader ran), a direct reload resets the
    AITER padding of an Inkling AMD Quark OCP_MX layer (zero weights, 0x01
    scales) instead of keeping the dummy bytes."""
    # AITER rounds the intermediate size up (to 256; 64 here).
    monkeypatch.setattr(
        quark_moe,
        "mxfp4_round_up_hidden_size_and_intermediate_size",
        lambda backend, hidden, intermediate: (hidden, round_up(intermediate, 64)),
    )
    num_experts, hidden, intermediate = 4, 64, 32

    def make_moe():
        layer = _make_experts(
            monkeypatch,
            _quark("QuarkOCP_MX_MoEMethod", qu.kMxfp4Static, None),
            num_experts=num_experts,
            hidden=hidden,
            intermediate=intermediate,
        )
        assert layer.w13_weight.shape == (num_experts, 128, hidden // 2)
        return _make_inkling_moe(layer), layer

    generator = torch.Generator().manual_seed(0)

    def randbytes(*shape):
        return torch.randint(2, 255, shape, generator=generator, dtype=torch.uint8)

    # Quark flattens the block scales over experts.
    checkpoint = {
        "experts.w13_weight": randbytes(num_experts, 2 * intermediate, hidden // 2),
        "experts.w2_weight": randbytes(num_experts, hidden, intermediate // 2),
        "experts.w13_weight_scale": randbytes(num_experts * 2 * intermediate, 2),
        "experts.w2_weight_scale": randbytes(num_experts * hidden, 1),
    }

    def load(moe):
        for name, weight in checkpoint.items():
            moe.load_expert_weight(name, weight)

    expected_moe, expected = make_moe()
    load(expected_moe)
    assert torch.all(expected.w2_weight_scale[:, :, 1] == 1)

    moe, layer = make_moe()
    for p in layer.parameters():
        p.data.fill_(7)  # dummy weights; no loader ran
    record_checkpoint_layout(moe)

    begin_direct_reload(moe)
    load(moe)

    for name, p in layer.named_parameters():
        torch.testing.assert_close(p, getattr(expected, name), rtol=0, atol=0)
