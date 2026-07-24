# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.expert_replacement import (
    ConstantExpertReplacement,
    get_mone_expert_ids,
    make_mone_replacement,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.v1.worker.workspace import init_workspace_manager


@pytest.mark.parametrize(
    ("config", "layer_idx", "expected"),
    [
        (
            SimpleNamespace(
                mone={
                    "replacement_type": "constant",
                    "experts_by_layer": {"1": [3, 1]},
                }
            ),
            1,
            (3, 1),
        ),
        (SimpleNamespace(approximate_experts=[[], [2, 4]]), 1, (2, 4)),
        (SimpleNamespace(approximate_experts={"3": [7, 255]}), 3, (7, 255)),
        (SimpleNamespace(), 0, ()),
    ],
)
def test_get_mone_expert_ids(config, layer_idx, expected):
    assert get_mone_expert_ids(config, layer_idx) == expected


def test_constant_replacement_transforms_routes_and_computes_side_output():
    replacement = ConstantExpertReplacement(5, [1, 3], 2, torch.float32)
    replacement.values.data.copy_(torch.tensor([[10.0, 20.0], [30.0, 40.0]]))

    topk_ids = torch.tensor([[1, 2], [4, 3], [99, -1]])
    topk_weights = torch.tensor([[0.25, 0.75], [0.6, 0.4], [0.5, 0.5]])
    compute_weights, compute_ids, replacement_output = replacement.transform_routes(
        torch.zeros(3, 2), topk_weights, topk_ids
    )

    assert replacement.compute_expert_ids == (0, 2, 4)
    assert compute_weights.data_ptr() == topk_weights.data_ptr()
    torch.testing.assert_close(
        compute_weights,
        torch.tensor([[0.0, 0.75], [0.6, 0.0], [0.0, 0.0]]),
    )
    torch.testing.assert_close(compute_ids, topk_ids)
    torch.testing.assert_close(
        replacement_output,
        torch.tensor([[2.5, 5.0], [12.0, 16.0], [0.0, 0.0]]),
    )


def test_constant_replacement_weight_loading_and_mapping():
    replacement = ConstantExpertReplacement(4, [1, 3], 3, torch.float32)
    replacement.values.weight_loader(
        replacement.values,
        torch.tensor([1.0, 2.0, 3.0]),
        expert_id=3,
    )
    with pytest.raises(ValueError, match="logical expert IDs: \\[1\\]"):
        replacement.validate_loaded_values("test layer")

    mapping = replacement.make_expert_params_mapping(
        moe_prefix="layers.0.mlp.experts",
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
    )
    assert (
        "layers.0.mlp.experts.routed_experts.expert_replacement.values",
        "layers.0.mlp.experts.3.approx_value",
        3,
        "constant",
    ) in mapping


def test_glm_mone_uses_compact_deepseek_style_expert_mapping():
    config = SimpleNamespace(
        approximate_experts={"3": [0, 7, 255]},
        dtype=torch.bfloat16,
    )
    replacement = make_mone_replacement(
        config=config,
        layer_idx=3,
        num_logical_experts=256,
        hidden_size=6144,
        params_dtype=config.dtype,
    )
    assert replacement is not None
    assert replacement.num_logical_experts == 256
    assert replacement.num_compute_experts == 253
    assert replacement.compute_expert_ids[:8] == (1, 2, 3, 4, 5, 6, 8, 9)

    mapping = replacement.make_expert_params_mapping(
        moe_prefix="layers.3.mlp.experts",
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
    )
    assert len(mapping) == 253 * 3 + 3
    assert (
        "layers.3.mlp.experts.routed_experts.expert_replacement.values",
        "layers.3.mlp.experts.255.approx_value",
        255,
        "constant",
    ) in mapping
    assert (
        "layers.3.mlp.experts.routed_experts.w13_",
        "layers.3.mlp.experts.1.gate_proj.",
        0,
        "w1",
    ) in mapping


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_invalid_replacement_routes_are_not_scheduled_or_reduced():
    topk_ids = torch.tensor([[0, 1], [2, 3]], device="cuda")
    expert_map = torch.tensor([0, -1, 1, -1], dtype=torch.int32, device="cuda")
    _, expert_ids, num_tokens_post_pad = moe_align_block_size(
        topk_ids,
        block_size=4,
        num_experts=4,
        expert_map=expert_map,
        ignore_invalid_experts=True,
    )
    assert num_tokens_post_pad.item() == 8
    torch.testing.assert_close(
        expert_ids[:2].cpu(), torch.tensor([0, 1], dtype=torch.int32)
    )

    route_outputs = torch.full((2, 2, 16), 1000.0, device="cuda")
    route_outputs[0, 0] = 1.0
    route_outputs[1, 0] = 2.0
    output = torch.empty(2, 16, device="cuda")
    ops.moe_sum(route_outputs, output, topk_ids, expert_map)
    torch.testing.assert_close(
        output, torch.tensor([[1.0] * 16, [2.0] * 16], device="cuda")
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fused_moe_combines_compact_experts_and_replacement(dist_init):
    hidden_size = 256
    replacement = ConstantExpertReplacement(4, [1, 3], hidden_size, torch.bfloat16)
    vllm_config = VllmConfig()
    vllm_config.compilation_config.static_forward_context = {}
    vllm_config.kernel_config.moe_backend = "triton"

    with set_current_vllm_config(vllm_config), set_forward_context(None, vllm_config):
        init_workspace_manager(torch.accelerator.current_device_index())
        layer = FusedMoE(
            num_experts=2,
            num_logical_experts=4,
            expert_replacement=replacement,
            top_k=2,
            hidden_size=hidden_size,
            intermediate_size=512,
            params_dtype=torch.bfloat16,
            prefix="test_constant_expert_replacement",
            renormalize=False,
        ).cuda()
        with torch.no_grad():
            layer.routed_experts.w13_weight.normal_(0, 0.01)
            layer.routed_experts.w2_weight.normal_(0, 0.01)
            replacement.values.normal_(0, 0.01)
        layer._quant_method.process_weights_after_loading(layer.routed_experts)

        hidden_states = torch.randn(8, hidden_size, dtype=torch.bfloat16, device="cuda")
        router_logits = torch.randn(8, 4, dtype=torch.float32, device="cuda")
        topk_weights, topk_ids = layer.router.select_experts(
            hidden_states.clone(), router_logits
        )
        compute_weights, compute_ids, replacement_output = replacement.transform_routes(
            hidden_states, topk_weights, topk_ids
        )
        expected = layer._quant_method.apply(
            layer=layer.routed_experts,
            x=hidden_states.clone(),
            topk_weights=compute_weights,
            topk_ids=compute_ids,
            shared_experts=None,
            shared_experts_input=None,
        )
        expected += replacement_output

        get_forward_context().all_moe_layers = None
        actual = layer(hidden_states.clone(), router_logits)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
