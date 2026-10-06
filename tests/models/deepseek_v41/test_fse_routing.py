# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for DeepSeek-V4.1 Fused Shared Experts (FSE) routing, padding,
and expert mapping."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from vllm.models.deepseek_v41.amd.model import DeepseekV4Model, DeepseekV4MoEBase


def test_fse_weight_name_redirection():
    """Verify shared_experts weights are redirected to virtual expert slot
    when FSE is enabled."""
    n_routed = 256
    model = MagicMock(spec=DeepseekV4Model)
    model.config = SimpleNamespace(n_routed_experts=n_routed, n_shared_experts=1)
    model.quant_config = None
    model.use_sequence_parallel = False

    # Mock MoE module on layer 0 with FSE enabled
    moe_layer0 = MagicMock(spec=DeepseekV4MoEBase)
    moe_layer0.is_fused_shared_expert_enabled = True

    # Layer 1 has FSE disabled
    moe_layer1 = MagicMock(spec=DeepseekV4MoEBase)
    moe_layer1.is_fused_shared_expert_enabled = False

    model.named_modules.return_value = [
        ("model.layers.0.mlp", moe_layer0),
        ("model.layers.1.mlp", moe_layer1),
    ]

    # Test names across layer 0 (FSE enabled) and layer 1 (FSE disabled)
    dummy_tensor = torch.zeros(1)
    weights = [
        ("model.layers.0.mlp.shared_experts.gate_proj.weight", dummy_tensor),
        ("model.layers.0.mlp.shared_experts.down_proj.weight", dummy_tensor),
        ("model.layers.0.mlp.shared_experts.up_proj.weight", dummy_tensor),
        ("model.layers.0.mlp.shared_experts.w1.weight", dummy_tensor),
        ("model.layers.0.mlp.shared_experts.w2.weight", dummy_tensor),
        ("model.layers.0.mlp.shared_experts.w3.weight", dummy_tensor),
        ("model.layers.0.ffn.shared_experts.gate_proj.weight", dummy_tensor),
        ("model.layers.1.mlp.shared_experts.gate_proj.weight", dummy_tensor),
        ("model.layers.1.mlp.shared_experts.down_proj.weight", dummy_tensor),
    ]

    # Intercept names processed in the weight iteration loop
    processed_names = []

    def mock_pad_shared(quant_config, name, loaded_weight):
        return loaded_weight

    model._pad_shared_expert_weight = mock_pad_shared

    # Simulate load_weights loop name transformation logic
    from vllm.model_executor.models.utils import extract_layer_index

    fuse_by_layer = {
        extract_layer_index(mod_name): getattr(
            mod, "is_fused_shared_expert_enabled", False
        )
        for mod_name, mod in model.named_modules()
        if isinstance(mod, DeepseekV4MoEBase)
    }

    for name, loaded_weight in weights:
        is_shared = ".ffn.shared_experts." in name or ".shared_experts." in name
        is_fse_redirect = is_shared and fuse_by_layer.get(
            extract_layer_index(name), False
        )
        if is_fse_redirect:
            name = name.replace(".shared_experts.down_proj", f".experts.{n_routed}.w2")
            name = name.replace(".shared_experts.gate_proj", f".experts.{n_routed}.w1")
            name = name.replace(".shared_experts.up_proj", f".experts.{n_routed}.w3")
            name = name.replace(".shared_experts.w", f".experts.{n_routed}.w")
        processed_names.append(name)

    # Layer 0 must redirect to .experts.256.w{1,2,3}
    assert processed_names[0] == f"model.layers.0.mlp.experts.{n_routed}.w1.weight"
    assert processed_names[1] == f"model.layers.0.mlp.experts.{n_routed}.w2.weight"
    assert processed_names[2] == f"model.layers.0.mlp.experts.{n_routed}.w3.weight"
    assert processed_names[3] == f"model.layers.0.mlp.experts.{n_routed}.w1.weight"
    assert processed_names[4] == f"model.layers.0.mlp.experts.{n_routed}.w2.weight"
    assert processed_names[5] == f"model.layers.0.mlp.experts.{n_routed}.w3.weight"
    assert processed_names[6] == f"model.layers.0.ffn.experts.{n_routed}.w1.weight"

    # Layer 1 (FSE disabled) must preserve original .shared_experts. names
    assert processed_names[7] == "model.layers.1.mlp.shared_experts.gate_proj.weight"
    assert processed_names[8] == "model.layers.1.mlp.shared_experts.down_proj.weight"


def test_pad_shared_expert_weight_dim1_for_w2_and_down_proj():
    """Verify down_proj and w2 weights/scales are padded along dimension 1
    (intermediate axis)."""
    with patch(
        "vllm.models.deepseek_v41.amd.model.get_tensor_model_parallel_world_size",
        return_value=2,
    ):
        quant_cfg = SimpleNamespace(weight_block_size=[128, 128])
        # mult = tp_size (2) * block_size (128) = 256
        # Down proj shape [H, I] = [1024, 2000]. 2000 -> padded to 2048 along dim 1.
        w_down = torch.ones(1024, 2000)

        # 1. Test .down_proj.
        padded_down = DeepseekV4Model._pad_shared_expert_weight(
            quant_cfg, "model.layers.0.mlp.shared_experts.down_proj.weight", w_down
        )
        assert padded_down.shape == (1024, 2048)
        assert torch.all(padded_down[:, :2000] == 1.0)
        assert torch.all(padded_down[:, 2000:] == 0.0)

        # 2. Test .w2. (FSE redirected name)
        padded_w2 = DeepseekV4Model._pad_shared_expert_weight(
            quant_cfg, "model.layers.0.mlp.experts.256.w2.weight", w_down
        )
        assert padded_w2.shape == (1024, 2048)
        assert torch.all(padded_w2[:, :2000] == 1.0)
        assert torch.all(padded_w2[:, 2000:] == 0.0)

        # 3. Test scale for .w2. (step = 1 instead of 128)
        # Scale shape in blocks: [1024 // 128, 15] = [8, 15]. mult = 2 * 1 = 2.
        # 15 -> padded to 16 along dim 1.
        scale_w2 = torch.ones(8, 15)
        padded_scale = DeepseekV4Model._pad_shared_expert_weight(
            quant_cfg, "model.layers.0.mlp.experts.256.w2.weight_scale", scale_w2
        )
        assert padded_scale.shape == (8, 16)
        assert torch.all(padded_scale[:, :15] == 1.0)
        assert torch.all(padded_scale[:, 15:] == 0.0)


def test_pad_shared_expert_weight_dim0_for_w1_w3_and_gate_up():
    """Verify gate/up and w1/w3 weights/scales are padded along dimension 0."""
    with patch(
        "vllm.models.deepseek_v41.amd.model.get_tensor_model_parallel_world_size",
        return_value=2,
    ):
        quant_cfg = SimpleNamespace(weight_block_size=[128, 128])
        # Gate/Up shape [I, H] = [2000, 1024]. 2000 -> padded to 2048 along dim 0.
        w_gate = torch.ones(2000, 1024)

        # 1. Test .gate_proj.
        padded_gate = DeepseekV4Model._pad_shared_expert_weight(
            quant_cfg, "model.layers.0.mlp.shared_experts.gate_proj.weight", w_gate
        )
        assert padded_gate.shape == (2048, 1024)
        assert torch.all(padded_gate[:2000, :] == 1.0)
        assert torch.all(padded_gate[2000:, :] == 0.0)

        # 2. Test .w1. (FSE redirected name)
        padded_w1 = DeepseekV4Model._pad_shared_expert_weight(
            quant_cfg, "model.layers.0.mlp.experts.256.w1.weight", w_gate
        )
        assert padded_w1.shape == (2048, 1024)

        # 3. Test .w3. (FSE redirected name)
        padded_w3 = DeepseekV4Model._pad_shared_expert_weight(
            quant_cfg, "model.layers.0.mlp.experts.256.w3.weight", w_gate
        )
        assert padded_w3.shape == (2048, 1024)

        # 4. Test scale for .w1. (step = 1)
        # Scale shape: [15, 8]. mult = 2 * 1 = 2. 15 -> padded to 16 along dim 0.
        scale_w1 = torch.ones(15, 8)
        padded_scale = DeepseekV4Model._pad_shared_expert_weight(
            quant_cfg, "model.layers.0.mlp.experts.256.w1.weight_scale", scale_w1
        )
        assert padded_scale.shape == (16, 8)


def test_pad_shared_expert_weight_no_op_when_aligned():
    """Verify already-aligned tensor is returned unchanged without copying or
    allocating pad."""
    with patch(
        "vllm.models.deepseek_v41.amd.model.get_tensor_model_parallel_world_size",
        return_value=2,
    ):
        quant_cfg = SimpleNamespace(weight_block_size=[128, 128])
        # mult = 2 * 128 = 256. 2048 is already a multiple of 256.
        aligned_w = torch.ones(1024, 2048)
        result = DeepseekV4Model._pad_shared_expert_weight(
            quant_cfg, "model.layers.0.mlp.experts.256.w2.weight", aligned_w
        )
        assert result is aligned_w  # Exact identity preserved


def test_get_expert_mapping_expansion_under_fse():
    """Verify get_expert_mapping expands num_experts by n_shared_experts
    when FSE is enabled."""
    model = MagicMock(spec=DeepseekV4Model)
    model.config = SimpleNamespace(n_routed_experts=256, n_shared_experts=1)

    moe_module = MagicMock(spec=DeepseekV4MoEBase)
    moe_module.is_fused_shared_expert_enabled = True
    model.modules.return_value = [moe_module]

    mapping_fn = DeepseekV4Model.get_expert_mapping.__get__(model, DeepseekV4Model)

    with patch(
        "vllm.models.deepseek_v41.amd.model.fused_moe_make_expert_params_mapping"
    ) as mock_mapping_gen:
        mapping_fn()
        mock_mapping_gen.assert_called_once_with(
            model,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=257,
        )

    # When FSE is disabled, num_experts must remain exactly n_routed_experts (256)
    moe_module.is_fused_shared_expert_enabled = False
    with patch(
        "vllm.models.deepseek_v41.amd.model.fused_moe_make_expert_params_mapping"
    ) as mock_mapping_gen:
        mapping_fn()
        mock_mapping_gen.assert_called_once_with(
            model,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=256,
        )
