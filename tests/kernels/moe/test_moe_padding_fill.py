# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
from types import SimpleNamespace

import pytest
import torch

import vllm.distributed as distributed
from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe import routed_experts
from vllm.model_executor.layers.quantization.utils import quant_utils as qu


def _quark(name, *keys):
    def make(module, layer):
        return getattr(module, f"Quark{name}MoEMethod")(layer.moe_config, *keys)

    return "quark.quark_moe", make


def _fp8_block(module, layer):
    config = module.Fp8Config(True, weight_block_size=[128, 128])
    return module.Fp8MoEMethod(config, layer)


# (module under vllm.model_executor.layers.quantization, method factory)
METHODS = {
    "fp8_block": ("fp8", _fp8_block),
    "quark_ocp_mx": _quark("OCP_MX_", qu.kMxfp4Static, None),
}


@pytest.mark.parametrize("method", METHODS)
def test_padding_value_marks_ones_initialized_params(monkeypatch, method):
    """A padded MoE layer is refilled by a direct reload with each parameter's
    initial padding: 1 exactly for the ones-initialized (scale) parameters."""
    module_name, make_method = METHODS[method]
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
