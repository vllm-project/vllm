# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
from types import SimpleNamespace

import pytest
import torch
from compressed_tensors.quantization import QuantizationArgs

import vllm.distributed as distributed
from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe import routed_experts
from vllm.model_executor.layers.quantization.utils import quant_utils as qu

_CT = "compressed_tensors.compressed_tensors_moe.compressed_tensors_moe"


def _ct(suffix, name, weight, act=None):
    def make(module, layer):
        cls = getattr(module, f"CompressedTensors{name}MoEMethod")
        act_args = act and QuantizationArgs(**act)
        return cls(QuantizationArgs(**weight), act_args, layer.moe_config)

    return (
        f"compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_{suffix}",
        make,
    )


def _quark(name, *keys):
    def make(module, layer):
        return getattr(module, f"Quark{name}MoEMethod")(layer.moe_config, *keys)

    return "quark.quark_moe", make


def _fp8_block(module, layer):
    config = module.Fp8Config(True, weight_block_size=[128, 128])
    return module.Fp8MoEMethod(config, layer)


_W4 = dict(num_bits=4, type="int", symmetric=True)
_FP8 = dict(num_bits=8, type="float", symmetric=True)
_INT8 = dict(num_bits=8, type="int", symmetric=True)
_TOKEN = dict(strategy="token", dynamic=True)

# (module under vllm.model_executor.layers.quantization, method factory)
METHODS = {
    "fp8_block": ("fp8", _fp8_block),
    "ct_w4a16_flydsl": _ct(
        "w4a16_flydsl", "W4A16Flydsl", {**_W4, "strategy": "group", "group_size": 32}
    ),
    "ct_w4a8_fp8": _ct(
        "w4a8_fp8",
        "W4A8Fp8",
        {**_W4, "strategy": "group", "group_size": 128},
        {**_FP8, **_TOKEN},
    ),
    "ct_w4a8_int8": _ct(
        "w4a8_int8", "W4A8Int8", {**_W4, "strategy": "channel"}, {**_INT8, **_TOKEN}
    ),
    "ct_w8a8_fp8_channel": _ct(
        "w8a8_fp8", "W8A8Fp8", {**_FP8, "strategy": "channel"}, {**_FP8, **_TOKEN}
    ),
    "ct_w8a8_fp8_block": _ct(
        "w8a8_fp8",
        "W8A8Fp8",
        {**_FP8, "strategy": "block", "block_structure": [128, 128]},
        {**_FP8, "strategy": "group", "group_size": 128, "dynamic": True},
    ),
    "ct_w8a8_int8": _ct(
        "w8a8_int8", "W8A8Int8", {**_INT8, "strategy": "channel"}, {**_INT8, **_TOKEN}
    ),
    "ct_wna16": _ct("wna16", "WNA16", {**_W4, "strategy": "group", "group_size": 128}),
    "quark_w8a8_fp8_channel": _quark(
        "W8A8Fp8", qu.kFp8StaticChannelSym, qu.kFp8DynamicTokenSym
    ),
    "quark_w8a8_fp8_block": _quark(
        "W8A8Fp8", qu.kFp8Static128BlockSym, qu.kFp8Dynamic128Sym
    ),
    "quark_w8a8_int8": _quark(
        "W8A8Int8", qu.kInt8StaticChannelSym, qu.kInt8DynamicTokenSym
    ),
    "quark_w4a8_fp8": _quark("W4A8Fp8", qu.kFp8DynamicTokenSym),
    "quark_ocp_mx": _quark("OCP_MX_", qu.kMxfp4Static, None),
}


@pytest.mark.parametrize("method", METHODS)
def test_padding_value_marks_ones_initialized_params(monkeypatch, method):
    """A padded MoE layer is refilled by a direct reload with each parameter's
    initial padding: 1 exactly for the ones-initialized (scale) parameters."""
    module_name, make_method = METHODS[method]
    if module_name.endswith("flydsl"):
        pytest.importorskip("aiter", reason="its module imports AITER at import")
    module = importlib.import_module(
        f"vllm.model_executor.layers.quantization.{module_name}"
    )
    # Build the method without selecting a kernel or reading a model config.
    for name in dir(module):
        if name.startswith("select_") and name.endswith("_backend"):
            monkeypatch.setattr(module, name, lambda *args, **kwargs: (None, None))
    if module_name == "quark.quark_moe":
        monkeypatch.setattr(
            module,
            "select_mxfp4_moe_backend",
            lambda *args, **kwargs: (module.Mxfp4MoeBackend.NONE, None),
        )
        monkeypatch.setattr(
            module,
            "get_current_vllm_config",
            lambda: SimpleNamespace(model_config=SimpleNamespace(hf_config=None)),
        )
        monkeypatch.setattr(module.rocm_aiter_ops, "is_fused_moe_enabled", lambda: True)
    for tp_module in (distributed, module):
        if hasattr(tp_module, "get_tensor_model_parallel_world_size"):
            monkeypatch.setattr(
                tp_module, "get_tensor_model_parallel_world_size", lambda: 1
            )
    monkeypatch.setattr(
        routed_experts,
        "resolve_quant_method",
        lambda quant_config, layer, prefix: make_method(module, layer),
    )

    layer = routed_experts.RoutedExperts(
        "model.layers.0.mlp.experts",
        torch.bfloat16,
        make_dummy_moe_config(
            num_experts=2, hidden_dim=256, intermediate_size=256, experts_per_token=2
        ),
        SimpleNamespace(get_name=lambda: "test"),
        expert_map_manager=SimpleNamespace(
            local_num_experts=2,
            num_fused_shared_experts=0,
            placement_strategy="linear",
            expert_map=None,
            expert_mask=None,
            routing_tables=None,
            map_global_to_local=lambda index: index,
        ),
    )
    layer.moe_config.hidden_dim_unpadded = layer.moe_config.hidden_dim - 1

    fill = dict(layer.padding_fill_params())
    assert 1 in fill.values(), fill
    for name, value in fill.items():
        ones = bool(torch.all(getattr(layer, name) == 1))
        assert (value == 1) == ones, (name, value)
