# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Fast MoE Safetensors Bypass Loader (Option C)."""

import os
import tempfile
import pytest
from safetensors.torch import save_file
import torch

from vllm.model_executor.model_loader.moe_fast_loader import (
    SafetensorsMoEIndex,
    fast_bypass_safetensors_iterator,
    _resolve_and_broadcast_mode,
)


@pytest.fixture
def synthetic_moe_checkpoint():
    """Creates a temporary safetensors checkpoint with non-MoE and MoE weights."""
    with tempfile.TemporaryDirectory() as tmpdir:
        num_layers = 2
        num_experts = 8
        hidden_dim = 64
        intermediate_dim = 128

        weights = {}

        # Non-MoE weights
        weights["model.embed_tokens.weight"] = torch.randn(
            100, hidden_dim, dtype=torch.bfloat16
        )
        weights["model.norm.weight"] = torch.ones(hidden_dim, dtype=torch.bfloat16)

        # MoE weights per layer
        for l in range(num_layers):
            weights[f"model.layers.{l}.input_layernorm.weight"] = torch.ones(
                hidden_dim, dtype=torch.bfloat16
            )
            for e in range(num_experts):
                weights[f"model.layers.{l}.mlp.experts.{e}.gate_proj.weight"] = (
                    torch.randn(intermediate_dim, hidden_dim, dtype=torch.bfloat16)
                )
                weights[f"model.layers.{l}.mlp.experts.{e}.up_proj.weight"] = (
                    torch.randn(intermediate_dim, hidden_dim, dtype=torch.bfloat16)
                )
                weights[f"model.layers.{l}.mlp.experts.{e}.down_proj.weight"] = (
                    torch.randn(hidden_dim, intermediate_dim, dtype=torch.bfloat16)
                )
                # 2D block scales
                weights[
                    f"model.layers.{l}.mlp.experts.{e}.gate_proj.weight_scale"
                ] = torch.randn(intermediate_dim // 32, hidden_dim // 32, dtype=torch.float32)
                weights[
                    f"model.layers.{l}.mlp.experts.{e}.up_proj.weight_scale"
                ] = torch.randn(intermediate_dim // 32, hidden_dim // 32, dtype=torch.float32)
                weights[
                    f"model.layers.{l}.mlp.experts.{e}.down_proj.weight_scale"
                ] = torch.randn(hidden_dim // 32, intermediate_dim // 32, dtype=torch.float32)

        shard_path = os.path.join(tmpdir, "model.safetensors")
        save_file(weights, shard_path)

        yield shard_path, weights, num_layers, num_experts, hidden_dim, intermediate_dim


def test_safetensors_moe_index_parsing(synthetic_moe_checkpoint):
    """Verifies that SafetensorsMoEIndex correctly parses and classifies all keys."""
    shard_path, weights, num_layers, num_experts, hidden_dim, intermediate_dim = (
        synthetic_moe_checkpoint
    )

    index = SafetensorsMoEIndex.build([shard_path])

    # Check non-MoE keys count
    # 2 global (embed, norm) + 2 layer norms = 4 non-MoE keys
    assert len(index.non_moe_keys) == 4
    non_moe_names = {k for k, _ in index.non_moe_keys}
    assert "model.embed_tokens.weight" in non_moe_names
    assert "model.norm.weight" in non_moe_names

    # Check MoE layers count
    assert len(index.moe_layers) == num_layers
    for l in range(num_layers):
        prefix = f"model.layers.{l}.mlp.experts"
        assert prefix in index.moe_layers
        plan = index.moe_layers[prefix]
        assert plan.num_total_experts == num_experts

        # Check projections
        assert ("gate", ".weight") in plan.slices
        assert ("up", ".weight") in plan.slices
        assert ("down", ".weight") in plan.slices
        assert ("gate", ".weight_scale") in plan.slices
        assert ("up", ".weight_scale") in plan.slices
        assert ("down", ".weight_scale") in plan.slices

        assert len(plan.slices[("gate", ".weight")]) == num_experts
        assert len(plan.slices[("up", ".weight")]) == num_experts
        assert len(plan.slices[("down", ".weight")]) == num_experts


def test_fast_bypass_iterator_numerical_equivalence(synthetic_moe_checkpoint):
    """Verifies exact bitwise equivalence between fast bypass 3D tensors and 2D source slices."""
    shard_path, weights, num_layers, num_experts, hidden_dim, intermediate_dim = (
        synthetic_moe_checkpoint
    )

    loaded_tensors = {}
    for name, tensor in fast_bypass_safetensors_iterator([shard_path]):
        loaded_tensors[name] = tensor.clone()

    # Total yielded keys:
    # 4 non-MoE keys + (4 consolidated keys * 2 layers) = 12 keys
    # vs original (4 + 6 * 8 * 2) = 100 keys!
    assert len(loaded_tensors) == 12

    for l in range(num_layers):
        prefix = f"model.layers.{l}.mlp.experts"
        gate_up_name = f"{prefix}.gate_up_proj"
        down_name = f"{prefix}.down_proj"
        gate_up_scale_name = f"{prefix}.gate_up_proj.weight_scale"
        down_scale_name = f"{prefix}.down_proj.weight_scale"

        assert gate_up_name in loaded_tensors
        assert down_name in loaded_tensors
        assert gate_up_scale_name in loaded_tensors
        assert down_scale_name in loaded_tensors

        fused_gate_up = loaded_tensors[gate_up_name]
        fused_down = loaded_tensors[down_name]
        fused_gate_up_scale = loaded_tensors[gate_up_scale_name]
        fused_down_scale = loaded_tensors[down_scale_name]

        assert fused_gate_up.shape == (num_experts, 2 * intermediate_dim, hidden_dim)
        assert fused_down.shape == (num_experts, hidden_dim, intermediate_dim)

        # Bitwise verification for each expert
        for e in range(num_experts):
            expected_gate = weights[f"{prefix}.{e}.gate_proj.weight"]
            expected_up = weights[f"{prefix}.{e}.up_proj.weight"]
            expected_down = weights[f"{prefix}.{e}.down_proj.weight"]

            expected_gate_scale = weights[f"{prefix}.{e}.gate_proj.weight_scale"]
            expected_up_scale = weights[f"{prefix}.{e}.up_proj.weight_scale"]
            expected_down_scale = weights[f"{prefix}.{e}.down_proj.weight_scale"]

            # Gate in lower half, up in upper half
            actual_gate = fused_gate_up[e, :intermediate_dim, :]
            actual_up = fused_gate_up[e, intermediate_dim:, :]
            actual_down = fused_down[e]

            assert torch.equal(actual_gate, expected_gate), f"Layer {l} Expert {e} gate mismatch"
            assert torch.equal(actual_up, expected_up), f"Layer {l} Expert {e} up mismatch"
            assert torch.equal(actual_down, expected_down), f"Layer {l} Expert {e} down mismatch"

            # Scales
            scale_gate_dim = intermediate_dim // 32
            actual_gate_scale = fused_gate_up_scale[e, :scale_gate_dim, :]
            actual_up_scale = fused_gate_up_scale[e, scale_gate_dim:, :]
            actual_down_scale = fused_down_scale[e]

            assert torch.equal(actual_gate_scale, expected_gate_scale)
            assert torch.equal(actual_up_scale, expected_up_scale)
            assert torch.equal(actual_down_scale, expected_down_scale)


def test_fast_bypass_iterator_expert_parallelism(synthetic_moe_checkpoint):
    """Verifies that under Expert Parallelism, only local experts are buffered and yielded."""
    shard_path, weights, num_layers, num_experts, hidden_dim, intermediate_dim = (
        synthetic_moe_checkpoint
    )

    # Rank 0 handles experts {0, 1, 2, 3}
    local_eids = {0, 1, 2, 3}
    loaded_tensors = {}
    for name, tensor in fast_bypass_safetensors_iterator(
        [shard_path], local_expert_ids=local_eids
    ):
        loaded_tensors[name] = tensor.clone()

    for l in range(num_layers):
        prefix = f"model.layers.{l}.mlp.experts"
        fused_gate_up = loaded_tensors[f"{prefix}.gate_up_proj"]
        fused_down = loaded_tensors[f"{prefix}.down_proj"]

        # Only 4 local experts
        assert fused_gate_up.shape == (len(local_eids), 2 * intermediate_dim, hidden_dim)
        assert fused_down.shape == (len(local_eids), hidden_dim, intermediate_dim)

        for slot_idx, e in enumerate(sorted(local_eids)):
            expected_gate = weights[f"{prefix}.{e}.gate_proj.weight"]
            expected_up = weights[f"{prefix}.{e}.up_proj.weight"]
            expected_down = weights[f"{prefix}.{e}.down_proj.weight"]

            actual_gate = fused_gate_up[slot_idx, :intermediate_dim, :]
            actual_up = fused_gate_up[slot_idx, intermediate_dim:, :]
            actual_down = fused_down[slot_idx]

            assert torch.equal(actual_gate, expected_gate)
            assert torch.equal(actual_up, expected_up)
            assert torch.equal(actual_down, expected_down)


def _create_simple_routed_experts(num_experts, hidden_dim, intermediate_dim):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
    )
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
        UnquantizedFusedMoEMethod,
    )

    moe_config = FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=1,
        hidden_dim=hidden_dim,
        intermediate_size=intermediate_dim,
        num_local_experts=num_experts,
        num_logical_experts=num_experts,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device="cpu",
        routing_method=RoutingMethodType.TopK,
        max_num_tokens=4096,
    )
    moe_config.moe_parallel_config = FusedMoEParallelConfig.make_no_parallel()
    moe_config.is_fused_checkpoint_transposed = False
    moe_config.enable_expert_parallel = False
    moe_config.expert_placement_strategy = "linear"

    experts = object.__new__(RoutedExperts)
    torch.nn.Module.__init__(experts)
    experts.layer_name = "experts"
    experts.moe_config = moe_config
    experts.global_num_experts = num_experts
    experts.hidden_size = hidden_dim
    experts.is_fused_checkpoint_transposed = False
    experts.lora_base_layer_prefix = ""
    experts.ckpt_gate_proj_name = "gate_proj"
    experts.ckpt_down_proj_name = "down_proj"
    experts.ckpt_up_proj_name = "up_proj"
    experts.quant_config = None

    class DummyMapManager:
        num_fused_shared_experts = 0
        def map_global_to_local(self, expert_id: int) -> int:
            return expert_id

    experts.expert_map_manager = DummyMapManager()
    with set_current_vllm_config(VllmConfig()):
        experts.quant_method = UnquantizedFusedMoEMethod(moe_config)

    experts.w13_weight = torch.nn.Parameter(
        torch.zeros(num_experts, 2 * intermediate_dim, hidden_dim, device="cpu", dtype=torch.bfloat16)
    )
    experts.w2_weight = torch.nn.Parameter(
        torch.zeros(num_experts, hidden_dim, intermediate_dim, device="cpu", dtype=torch.bfloat16)
    )
    experts.w13_weight.weight_loader = experts.weight_loader
    experts.w2_weight.weight_loader = experts.weight_loader
    return experts


def test_routed_experts_integration_with_fast_bypass(synthetic_moe_checkpoint):
    """Verifies that RoutedExperts.load_weights successfully bulk-loads fast bypass 3D tensors."""
    shard_path, weights, num_layers, num_experts, hidden_dim, intermediate_dim = (
        synthetic_moe_checkpoint
    )

    layer = _create_simple_routed_experts(
        num_experts=num_experts,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
    )

    prefix = "model.layers.0.mlp.experts"
    weights_stream = [
        (name, tensor)
        for name, tensor in fast_bypass_safetensors_iterator([shard_path])
        if name.startswith(prefix)
    ]

    # Map the yielded keys (experts.gate_up_proj -> experts.w13, experts.down_proj -> experts.w2)
    # as done by RoutedExperts expert_params_mapping
    mapped_stream = []
    for name, tensor in weights_stream:
        # Strip model.layers.0.mlp.experts. prefix so it matches layer relative name
        rel_name = name[len(f"{prefix}.") :]
        mapped_stream.append((rel_name, tensor))

    # RoutedExperts.load_weights is a generator; exhaust it to execute loading
    list(layer.load_weights(mapped_stream))

    # Verify layer weights match expected bitwise
    for e in range(num_experts):
        expected_gate = weights[f"{prefix}.{e}.gate_proj.weight"]
        expected_up = weights[f"{prefix}.{e}.up_proj.weight"]
        expected_down = weights[f"{prefix}.{e}.down_proj.weight"]

        actual_gate = layer.w13_weight[e, :intermediate_dim, :]
        actual_up = layer.w13_weight[e, intermediate_dim:, :]
        actual_down = layer.w2_weight[e]

        assert torch.equal(actual_gate, expected_gate)
        assert torch.equal(actual_up, expected_up)
        assert torch.equal(actual_down, expected_down)


@pytest.fixture
def synthetic_3d_moe_checkpoint():
    """Creates a temporary safetensors checkpoint with 3D pre-fused MoE weights and negative cases."""
    with tempfile.TemporaryDirectory() as tmpdir:
        num_experts = 8
        hidden_dim = 64
        intermediate_dim = 128

        weights = {}

        # Non-MoE weights
        weights["model.embed_tokens.weight"] = torch.randn(
            100, hidden_dim, dtype=torch.bfloat16
        )
        weights["model.norm.weight"] = torch.ones(hidden_dim, dtype=torch.bfloat16)

        # Negative edge cases that must NOT be parsed as MoE
        weights["model.vision_tower.patch_embedding.weight"] = torch.randn(
            16, hidden_dim, dtype=torch.bfloat16
        )
        weights["model.layers.0.mlp.shared_experts.gate_up_proj.weight"] = torch.randn(
            2 * intermediate_dim, hidden_dim, dtype=torch.bfloat16
        )
        weights["model.layers.0.mlp.experts.bias"] = torch.zeros(
            hidden_dim, dtype=torch.bfloat16
        )

        # Layer 0: Pre-fused 3D weights and scales
        weights["model.layers.0.input_layernorm.weight"] = torch.ones(
            hidden_dim, dtype=torch.bfloat16
        )
        weights["model.layers.0.mlp.experts.gate_up_proj"] = torch.randn(
            num_experts, 2 * intermediate_dim, hidden_dim, dtype=torch.bfloat16
        )
        weights["model.layers.0.mlp.experts.down_proj"] = torch.randn(
            num_experts, hidden_dim, intermediate_dim, dtype=torch.bfloat16
        )
        weights["model.layers.0.mlp.experts.gate_up_proj.weight_scale"] = torch.randn(
            num_experts, (2 * intermediate_dim) // 32, hidden_dim // 32, dtype=torch.float32
        )
        weights["model.layers.0.mlp.experts.down_proj.weight_scale"] = torch.randn(
            num_experts, hidden_dim // 32, intermediate_dim // 32, dtype=torch.float32
        )

        # Layer 1: Separate 3D gate and up weights (DeepSeek-V3 style) + scales
        weights["model.layers.1.input_layernorm.weight"] = torch.ones(
            hidden_dim, dtype=torch.bfloat16
        )
        weights["model.layers.1.mlp.experts.gate_proj.weight"] = torch.randn(
            num_experts, intermediate_dim, hidden_dim, dtype=torch.bfloat16
        )
        weights["model.layers.1.mlp.experts.up_proj.weight"] = torch.randn(
            num_experts, intermediate_dim, hidden_dim, dtype=torch.bfloat16
        )
        weights["model.layers.1.mlp.experts.down_proj.weight"] = torch.randn(
            num_experts, hidden_dim, intermediate_dim, dtype=torch.bfloat16
        )
        weights["model.layers.1.mlp.experts.gate_proj.weight_scale"] = torch.randn(
            num_experts, intermediate_dim // 32, hidden_dim // 32, dtype=torch.float32
        )
        weights["model.layers.1.mlp.experts.up_proj.weight_scale"] = torch.randn(
            num_experts, intermediate_dim // 32, hidden_dim // 32, dtype=torch.float32
        )
        weights["model.layers.1.mlp.experts.down_proj.weight_scale"] = torch.randn(
            num_experts, hidden_dim // 32, intermediate_dim // 32, dtype=torch.float32
        )

        shard_path = os.path.join(tmpdir, "model_3d.safetensors")
        save_file(weights, shard_path)

        yield shard_path, weights, num_experts, hidden_dim, intermediate_dim


def test_3d_safetensors_index_parsing(synthetic_3d_moe_checkpoint):
    """Verifies that SafetensorsMoEIndex correctly parses 3D MoE keys and rejects non-MoE weights."""
    shard_path, weights, num_experts, hidden_dim, intermediate_dim = (
        synthetic_3d_moe_checkpoint
    )

    index = SafetensorsMoEIndex.build([shard_path])

    # Check non-MoE keys:
    # embed, norm, vision patch embedding, shared_experts, experts.bias, layer 0 norm, layer 1 norm = 7 keys
    non_moe_names = {k for k, _ in index.non_moe_keys}
    assert "model.vision_tower.patch_embedding.weight" in non_moe_names
    assert "model.layers.0.mlp.shared_experts.gate_up_proj.weight" in non_moe_names
    assert "model.layers.0.mlp.experts.bias" in non_moe_names
    assert "model.embed_tokens.weight" in non_moe_names

    # Check MoE layers count
    assert len(index.moe_layers) == 2

    # Layer 0 has pre-fused gate_up and down
    plan0 = index.moe_layers["model.layers.0.mlp.experts"]
    assert plan0.num_total_experts == num_experts
    assert ("gate_up", "") in plan0.tensors_3d
    assert ("down", "") in plan0.tensors_3d
    assert ("gate_up", ".weight_scale") in plan0.tensors_3d
    assert ("down", ".weight_scale") in plan0.tensors_3d

    # Layer 1 has separate gate, up, down
    plan1 = index.moe_layers["model.layers.1.mlp.experts"]
    assert plan1.num_total_experts == num_experts
    assert ("gate", ".weight") in plan1.tensors_3d
    assert ("up", ".weight") in plan1.tensors_3d
    assert ("down", ".weight") in plan1.tensors_3d
    assert ("gate", ".weight_scale") in plan1.tensors_3d
    assert ("up", ".weight_scale") in plan1.tensors_3d
    assert ("down", ".weight_scale") in plan1.tensors_3d


def test_3d_fast_bypass_iterator_numerical_equivalence(synthetic_3d_moe_checkpoint):
    """Verifies bitwise numerical equivalence for pre-fused 3D tensors and online 3D gate+up fusion."""
    shard_path, weights, num_experts, hidden_dim, intermediate_dim = (
        synthetic_3d_moe_checkpoint
    )

    loaded_tensors = {}
    for name, tensor in fast_bypass_safetensors_iterator([shard_path]):
        loaded_tensors[name] = tensor.clone()

    # Layer 0: pre-fused
    prefix0 = "model.layers.0.mlp.experts"
    assert torch.equal(
        loaded_tensors[f"{prefix0}.gate_up_proj"],
        weights[f"{prefix0}.gate_up_proj"],
    )
    assert torch.equal(
        loaded_tensors[f"{prefix0}.down_proj"],
        weights[f"{prefix0}.down_proj"],
    )
    assert torch.equal(
        loaded_tensors[f"{prefix0}.gate_up_proj.weight_scale"],
        weights[f"{prefix0}.gate_up_proj.weight_scale"],
    )
    assert torch.equal(
        loaded_tensors[f"{prefix0}.down_proj.weight_scale"],
        weights[f"{prefix0}.down_proj.weight_scale"],
    )

    # Layer 1: fused online from separate gate_proj and up_proj
    prefix1 = "model.layers.1.mlp.experts"
    fused_gate_up = loaded_tensors[f"{prefix1}.gate_up_proj"]
    fused_gate_up_scale = loaded_tensors[f"{prefix1}.gate_up_proj.weight_scale"]

    expected_gate = weights[f"{prefix1}.gate_proj.weight"]
    expected_up = weights[f"{prefix1}.up_proj.weight"]
    assert torch.equal(fused_gate_up[:, :intermediate_dim, :], expected_gate)
    assert torch.equal(fused_gate_up[:, intermediate_dim:, :], expected_up)

    expected_gate_scale = weights[f"{prefix1}.gate_proj.weight_scale"]
    expected_up_scale = weights[f"{prefix1}.up_proj.weight_scale"]
    scale_gate_dim = intermediate_dim // 32
    assert torch.equal(fused_gate_up_scale[:, :scale_gate_dim, :], expected_gate_scale)
    assert torch.equal(fused_gate_up_scale[:, scale_gate_dim:, :], expected_up_scale)

    assert torch.equal(
        loaded_tensors[f"{prefix1}.down_proj"],
        weights[f"{prefix1}.down_proj.weight"],
    )


@pytest.mark.parametrize("local_eids", [{0, 1, 2, 3}, {4, 5, 6, 7}])
def test_3d_fast_bypass_iterator_linear_ep(synthetic_3d_moe_checkpoint, local_eids):
    """Verifies that under Linear EP, 3D tensors are sliced to local experts with zero extra bytes."""
    shard_path, weights, num_experts, hidden_dim, intermediate_dim = (
        synthetic_3d_moe_checkpoint
    )

    loaded_tensors = {}
    for name, tensor in fast_bypass_safetensors_iterator(
        [shard_path], local_expert_ids=local_eids
    ):
        loaded_tensors[name] = tensor.clone()

    start_eid = min(local_eids)
    end_eid = max(local_eids) + 1

    # Layer 0 verification
    prefix0 = "model.layers.0.mlp.experts"
    l0_gate_up = loaded_tensors[f"{prefix0}.gate_up_proj"]
    assert l0_gate_up.shape == (len(local_eids), 2 * intermediate_dim, hidden_dim)
    assert torch.equal(
        l0_gate_up,
        weights[f"{prefix0}.gate_up_proj"][start_eid:end_eid],
    )

    # Layer 1 verification (online fused gate+up)
    prefix1 = "model.layers.1.mlp.experts"
    l1_gate_up = loaded_tensors[f"{prefix1}.gate_up_proj"]
    assert l1_gate_up.shape == (len(local_eids), 2 * intermediate_dim, hidden_dim)
    expected_l1_gate = weights[f"{prefix1}.gate_proj.weight"][start_eid:end_eid]
    expected_l1_up = weights[f"{prefix1}.up_proj.weight"][start_eid:end_eid]
    assert torch.equal(l1_gate_up[:, :intermediate_dim, :], expected_l1_gate)
    assert torch.equal(l1_gate_up[:, intermediate_dim:, :], expected_l1_up)


def test_3d_fast_bypass_iterator_non_contiguous_ep(synthetic_3d_moe_checkpoint):
    """Verifies that arbitrary non-contiguous EP (e.g. round-robin {0, 2, 4, 6}) slices correctly."""
    shard_path, weights, num_experts, hidden_dim, intermediate_dim = (
        synthetic_3d_moe_checkpoint
    )

    local_eids = {0, 2, 4, 6}
    loaded_tensors = {}
    for name, tensor in fast_bypass_safetensors_iterator(
        [shard_path], local_expert_ids=local_eids
    ):
        loaded_tensors[name] = tensor.clone()

    prefix0 = "model.layers.0.mlp.experts"
    l0_gate_up = loaded_tensors[f"{prefix0}.gate_up_proj"]
    assert l0_gate_up.shape == (4, 2 * intermediate_dim, hidden_dim)

    for slot_idx, eid in enumerate(sorted(local_eids)):
        expected_expert = weights[f"{prefix0}.gate_up_proj"][eid]
        assert torch.equal(l0_gate_up[slot_idx], expected_expert)


def test_routed_experts_integration_with_3d_fast_bypass(synthetic_3d_moe_checkpoint):
    """Verifies that RoutedExperts.load_weights successfully ingests pre-fused 3D tensors."""
    shard_path, weights, num_experts, hidden_dim, intermediate_dim = (
        synthetic_3d_moe_checkpoint
    )

    layer = _create_simple_routed_experts(
        num_experts=num_experts,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
    )

    prefix = "model.layers.0.mlp.experts"
    weights_stream = [
        (name, tensor)
        for name, tensor in fast_bypass_safetensors_iterator([shard_path])
        if name.startswith(prefix)
    ]

    mapped_stream = []
    for name, tensor in weights_stream:
        rel_name = name[len(f"{prefix}.") :]
        mapped_stream.append((rel_name, tensor))

    list(layer.load_weights(mapped_stream))

    assert torch.equal(layer.w13_weight, weights[f"{prefix}.gate_up_proj"])
    assert torch.equal(layer.w2_weight, weights[f"{prefix}.down_proj"])


def test_hand_rolled_moe_load_weights_compatibility(synthetic_3d_moe_checkpoint):
    """Verifies that hand-rolled model load_weights loops (like Glm5NextModel and KimiK3)

    cleanly load consolidated 3D weights without KeyError or dropped parameters.
    """
    from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping

    shard_path, weights, num_experts, hidden_dim, intermediate_dim = (
        synthetic_3d_moe_checkpoint
    )

    layer = _create_simple_routed_experts(
        num_experts=num_experts,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
    )

    class DummyModel(torch.nn.Module):
        def __init__(self, routed_experts):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([torch.nn.Module()])
            self.model.layers[0].mlp = torch.nn.Module()
            self.model.layers[0].mlp.experts = torch.nn.Module()
            self.model.layers[0].mlp.experts.routed_experts = routed_experts

    dummy_model = DummyModel(layer)
    params_dict = dict(dummy_model.named_parameters())

    # Test Case 1: GLM / DeepSeek naming (gate_proj, down_proj, up_proj)
    expert_mapping_glm = fused_moe_make_expert_params_mapping(
        dummy_model,
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=num_experts,
    )

    weights_stream = list(fast_bypass_safetensors_iterator([shard_path]))
    for name, loaded_weight in weights_stream:
        if not ("layers.0.mlp.experts" in name and ("gate_up_proj" in name or "down_proj" in name)):
            continue
        if "scale" in name:
            continue
        for param_name, weight_name, expert_id, expert_shard_id in expert_mapping_glm:
            if weight_name not in name:
                continue
            name_mapped = name.replace(weight_name, param_name)
            if name_mapped not in params_dict:
                continue
            param = params_dict[name_mapped]
            weight_loader = getattr(param, "weight_loader", layer.weight_loader)
            success = weight_loader(
                param,
                loaded_weight,
                name_mapped,
                expert_id=expert_id,
                shard_id=expert_shard_id,
                return_success=True,
            )
            if success:
                break
        else:
            raise KeyError(f"Key {name} failed to match expert_mapping")

    prefix0 = "model.layers.0.mlp.experts"
    assert torch.equal(layer.w13_weight, weights[f"{prefix0}.gate_up_proj"])
    assert torch.equal(layer.w2_weight, weights[f"{prefix0}.down_proj"])

    # Test Case 2: Kimi-K3 naming (w1, w2, w3)
    layer.w13_weight.data.zero_()
    layer.w2_weight.data.zero_()
    expert_mapping_kimi = fused_moe_make_expert_params_mapping(
        dummy_model,
        ckpt_gate_proj_name="w1",
        ckpt_down_proj_name="w2",
        ckpt_up_proj_name="w3",
        num_experts=num_experts,
    )

    for name, loaded_weight in weights_stream:
        if not ("layers.0.mlp.experts" in name and ("gate_up_proj" in name or "down_proj" in name)):
            continue
        if "scale" in name:
            continue
        for param_name, weight_name, expert_id, expert_shard_id in expert_mapping_kimi:
            if weight_name not in name:
                continue
            name_mapped = name.replace(weight_name, param_name)
            if name_mapped not in params_dict:
                continue
            param = params_dict[name_mapped]
            weight_loader = getattr(param, "weight_loader", layer.weight_loader)
            success = weight_loader(
                param,
                loaded_weight,
                name_mapped,
                expert_id=expert_id,
                shard_id=expert_shard_id,
                return_success=True,
            )
            if success:
                break
        else:
            raise KeyError(f"Key {name} failed to match expert_mapping")

    assert torch.equal(layer.w13_weight, weights[f"{prefix0}.gate_up_proj"])
    assert torch.equal(layer.w2_weight, weights[f"{prefix0}.down_proj"])


def test_fast_slice_packer_load_inline():
    """Verifies that C++ OpenMP Batch Slice Packer compiles and performs byte-accurate batch copies."""
    from vllm.model_executor.model_loader.moe_fast_loader import _get_fast_slice_packer

    packer = _get_fast_slice_packer()
    assert packer is not None, "Failed to compile/load C++ fast slice packer"

    # Test synthetic multi-slice copy
    num_ops = 50
    slice_size = 128
    src = torch.randint(0, 255, (num_ops * slice_size * 2,), dtype=torch.uint8)
    dst = torch.zeros(num_ops * slice_size, dtype=torch.uint8)

    ops_list = []
    expected_dst = torch.zeros_like(dst)

    for i in range(num_ops):
        dst_off = i * slice_size
        src_off = i * slice_size * 2
        ops_list.append([dst_off, src_off, slice_size])
        expected_dst[dst_off : dst_off + slice_size] = src[src_off : src_off + slice_size]

    ops_tensor = torch.tensor(ops_list, dtype=torch.int64)
    packer.batch_copy_slices(dst.data_ptr(), src.data_ptr(), ops_tensor)

    assert torch.equal(dst, expected_dst), "Batch slice copy data mismatch"



def test_mode2_direct_vram_streaming(synthetic_moe_checkpoint):
    """Verifies that Mode 2 (direct_vram_mode=True) streams all keys with pipelined prefetching."""
    shard_path, weights, num_layers, num_experts, hidden_dim, intermediate_dim = (
        synthetic_moe_checkpoint
    )

    loaded_tensors = {}
    for name, tensor in fast_bypass_safetensors_iterator(
        [shard_path], direct_vram_mode=True
    ):
        loaded_tensors[name] = tensor.clone()

    # All non-MoE and MoE 2D keys must be yielded directly
    for k, v in weights.items():
        assert k in loaded_tensors, f"Key {k} missing in Mode 2 stream"
        assert torch.equal(loaded_tensors[k], v), f"Tensor mismatch for key {k}"


def test_mode1_vs_mode2_numerical_parity(synthetic_moe_checkpoint):
    """Verifies exact bitwise parity of loaded layer weights between Mode 1 and Mode 2."""
    shard_path, weights, num_layers, num_experts, hidden_dim, intermediate_dim = (
        synthetic_moe_checkpoint
    )

    prefix = "model.layers.0.mlp.experts"

    # 1. Ingest via Mode 1 (3D host staging)
    layer_mode1 = _create_simple_routed_experts(
        num_experts=num_experts,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
    )
    stream_mode1 = [
        (name[len(f"{prefix}.") :], tensor)
        for name, tensor in fast_bypass_safetensors_iterator(
            [shard_path], direct_vram_mode=False
        )
        if name.startswith(prefix)
    ]
    list(layer_mode1.load_weights(stream_mode1))

    # 2. Ingest via Mode 2 (Direct-to-VRAM streaming)
    layer_mode2 = _create_simple_routed_experts(
        num_experts=num_experts,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
    )
    stream_mode2 = [
        (name[len(f"{prefix}.") :], tensor)
        for name, tensor in fast_bypass_safetensors_iterator(
            [shard_path], direct_vram_mode=True
        )
        if name.startswith(prefix)
    ]
    list(layer_mode2.load_weights(stream_mode2))

    # Bitwise parity assertion
    assert torch.equal(layer_mode1.w13_weight, layer_mode2.w13_weight)
    assert torch.equal(layer_mode1.w2_weight, layer_mode2.w2_weight)


def test_mode2_fse_shared_expert_routing():
    """Verifies that Mode 2 routes FSE shared experts into virtual expert slot num_routed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        num_routed = 8
        hidden_dim = 64
        intermediate_dim = 128

        weights = {}
        # Routed experts
        for e in range(num_routed):
            weights[f"model.layers.0.mlp.experts.{e}.gate_proj.weight"] = torch.randn(
                intermediate_dim, hidden_dim, dtype=torch.bfloat16
            )
            weights[f"model.layers.0.mlp.experts.{e}.up_proj.weight"] = torch.randn(
                intermediate_dim, hidden_dim, dtype=torch.bfloat16
            )
            weights[f"model.layers.0.mlp.experts.{e}.down_proj.weight"] = torch.randn(
                hidden_dim, intermediate_dim, dtype=torch.bfloat16
            )
        # Shared expert
        weights["model.layers.0.mlp.shared_experts.gate_proj.weight"] = torch.randn(
            intermediate_dim, hidden_dim, dtype=torch.bfloat16
        )
        weights["model.layers.0.mlp.shared_experts.up_proj.weight"] = torch.randn(
            intermediate_dim, hidden_dim, dtype=torch.bfloat16
        )
        weights["model.layers.0.mlp.shared_experts.down_proj.weight"] = torch.randn(
            hidden_dim, intermediate_dim, dtype=torch.bfloat16
        )

        shard_path = os.path.join(tmpdir, "fse_model.safetensors")
        save_file(weights, shard_path)

        stream = list(
            fast_bypass_safetensors_iterator(
                [shard_path],
                direct_vram_mode=True,
                fse_enabled=True,
                n_shared_experts=1,
            )
        )
        yielded_keys = {name: tensor for name, tensor in stream}

        # Shared expert must be remapped to virtual slot num_routed (8)
        assert "model.layers.0.mlp.experts.8.gate_proj.weight" in yielded_keys
        assert "model.layers.0.mlp.experts.8.up_proj.weight" in yielded_keys
        assert "model.layers.0.mlp.experts.8.down_proj.weight" in yielded_keys

        assert torch.equal(
            yielded_keys["model.layers.0.mlp.experts.8.gate_proj.weight"],
            weights["model.layers.0.mlp.shared_experts.gate_proj.weight"],
        )
        assert torch.equal(
            yielded_keys["model.layers.0.mlp.experts.8.down_proj.weight"],
            weights["model.layers.0.mlp.shared_experts.down_proj.weight"],
        )


def test_fse_shared_expert_shard_inversion():
    """Verifies that checkpoints ordering shared experts before routed experts do not duplicate prefix."""
    with tempfile.TemporaryDirectory() as tmpdir:
        num_routed = 4
        hidden_dim = 64
        intermediate_dim = 128

        # Shard 0: contains shared expert weights ONLY
        shard0_weights = {
            "model.layers.0.mlp.shared_experts.gate_proj.weight": torch.randn(
                intermediate_dim, hidden_dim, dtype=torch.bfloat16
            ),
            "model.layers.0.mlp.shared_experts.up_proj.weight": torch.randn(
                intermediate_dim, hidden_dim, dtype=torch.bfloat16
            ),
            "model.layers.0.mlp.shared_experts.down_proj.weight": torch.randn(
                hidden_dim, intermediate_dim, dtype=torch.bfloat16
            ),
        }
        shard0_path = os.path.join(tmpdir, "model-00001-of-00002.safetensors")
        save_file(shard0_weights, shard0_path)

        # Shard 1: contains routed expert weights
        shard1_weights = {}
        for e in range(num_routed):
            shard1_weights[f"model.layers.0.mlp.experts.{e}.gate_proj.weight"] = torch.randn(
                intermediate_dim, hidden_dim, dtype=torch.bfloat16
            )
            shard1_weights[f"model.layers.0.mlp.experts.{e}.up_proj.weight"] = torch.randn(
                intermediate_dim, hidden_dim, dtype=torch.bfloat16
            )
            shard1_weights[f"model.layers.0.mlp.experts.{e}.down_proj.weight"] = torch.randn(
                hidden_dim, intermediate_dim, dtype=torch.bfloat16
            )
        shard1_path = os.path.join(tmpdir, "model-00002-of-00002.safetensors")
        save_file(shard1_weights, shard1_path)

        # Build index from shards ordered [shard0, shard1]
        index = SafetensorsMoEIndex.build(
            [shard0_path, shard1_path],
            fse_enabled=True,
            n_shared_experts=1,
        )

        # Ensure no duplicate prefix like "model.layers.0.mlp.experts.experts" exists
        for layer_prefix in index.moe_layers:
            assert ".experts.experts" not in layer_prefix, f"Found duplicated prefix: {layer_prefix}"

        # Verify streaming in Mode 2 yields virtual slot num_routed
        stream = list(
            fast_bypass_safetensors_iterator(
                [shard0_path, shard1_path],
                direct_vram_mode=True,
                fse_enabled=True,
                n_shared_experts=1,
            )
        )
        yielded_keys = {name: tensor for name, tensor in stream}
        assert f"model.layers.0.mlp.experts.{num_routed}.gate_proj.weight" in yielded_keys
        assert f"model.layers.0.mlp.experts.{num_routed}.down_proj.weight" in yielded_keys
        assert torch.equal(
            yielded_keys[f"model.layers.0.mlp.experts.{num_routed}.gate_proj.weight"],
            shard0_weights["model.layers.0.mlp.shared_experts.gate_proj.weight"],
        )


def test_resolve_and_broadcast_mode_warmth_gating(monkeypatch, tmp_path):
    """Verify that page cache warmth < 80% selects Mode 3 and >= 80% selects Mode 1."""
    # Create a small dummy file
    dummy_file = tmp_path / "shard.safetensors"
    dummy_file.write_bytes(b"\x00" * 4096)
    files = [str(dummy_file)]

    # Mock warmth check to return 70% (cold cache, below 80% threshold)
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.moe_fast_loader.check_page_cache_warmth",
        lambda _: 0.70,
    )
    mode_cold = _resolve_and_broadcast_mode(
        files,
        use_direct_io=None,
        use_direct_vram=None,
        env_threshold_gb=None,
        direct_vram_threshold_gb=None,
        tp_rank=0,
        tp_size=1,
    )
    assert mode_cold == 3, f"Expected Mode 3 under cold cache (<80%), got {mode_cold}"

    # Mock warmth check to return 85% (warm cache, >= 80% threshold)
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.moe_fast_loader.check_page_cache_warmth",
        lambda _: 0.85,
    )
    mode_warm = _resolve_and_broadcast_mode(
        files,
        use_direct_io=None,
        use_direct_vram=None,
        env_threshold_gb=None,
        direct_vram_threshold_gb=None,
        tp_rank=0,
        tp_size=1,
    )
    assert mode_warm == 1, f"Expected Mode 1 under warm cache, got {mode_warm}"


def test_resolve_and_broadcast_mode_structural_inspection(monkeypatch, tmp_path):
    """Verify structural multi-factor decision tree with 2D vs 3D weights, scale, and TP sizing."""
    from vllm.model_executor.model_loader.moe_fast_loader import (
        MoE3DTensorLocation,
        MoELayerPlan,
        MoESliceLocation,
        SafetensorsMoEIndex,
    )

    dummy_file = tmp_path / "shard.safetensors"
    dummy_file.write_bytes(b"\x00" * 4096)
    files = [str(dummy_file)]

    # Always warm cache for structural inspection
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.moe_fast_loader.check_page_cache_warmth",
        lambda _: 0.95,
    )

    # 1. Case: Ultra-large checkpoint (> 300 GB, e.g. Kimi-K3 1.45 TB) with TP=8 -> Mode 2
    monkeypatch.setattr(os.path, "getsize", lambda _: 1450 * (1024**3))
    mode_kimi_k3 = _resolve_and_broadcast_mode(
        files,
        tp_rank=0,
        tp_size=8,
    )
    assert mode_kimi_k3 == 2, f"Expected Mode 2 for ultra-large Kimi-K3 (1.45 TB, TP=8), got {mode_kimi_k3}"

    # 2. Case: Checkpoint <= 300 GB (e.g. Qwen3.8, GLM-5.3, MiniMax-M3) -> Mode 1 across TP1, TP2, TP4, TP8
    monkeypatch.setattr(os.path, "getsize", lambda _: 170 * (1024**3))
    for tp in (1, 2, 4, 8):
        mode_med = _resolve_and_broadcast_mode(
            files,
            tp_rank=0,
            tp_size=tp,
        )
        assert mode_med == 1, f"Expected Mode 1 for checkpoint <= 300 GB under TP={tp}, got {mode_med}"

    # 3. Case: Ultra-large checkpoint (> 300 GB, e.g. DeepSeek-V4.1 475 GB) and TP >= 4 -> Mode 2 (avoid mmap_lock)
    monkeypatch.setattr(os.path, "getsize", lambda _: 475 * (1024**3))
    mode_2d_large_tp4 = _resolve_and_broadcast_mode(
        files,
        tp_rank=0,
        tp_size=4,
    )
    assert mode_2d_large_tp4 == 2, f"Expected Mode 2 for DeepSeek (> 300 GB) under TP=4, got {mode_2d_large_tp4}"

    mode_2d_large_tp8 = _resolve_and_broadcast_mode(
        files,
        tp_rank=0,
        tp_size=8,
    )
    assert mode_2d_large_tp8 == 2, f"Expected Mode 2 for DeepSeek (> 300 GB) under TP=8, got {mode_2d_large_tp8}"

    # 4. Case: Ultra-large checkpoint (> 300 GB) and TP <= 2 -> Mode 1
    mode_2d_large_tp2 = _resolve_and_broadcast_mode(
        files,
        tp_rank=0,
        tp_size=2,
    )
    assert mode_2d_large_tp2 == 1, f"Expected Mode 1 for checkpoint > 300 GB under TP=2, got {mode_2d_large_tp2}"


def test_resolve_and_broadcast_mode_overrides(monkeypatch, tmp_path):
    """Verify that manual overrides take precedence over automatic warmth gating."""
    dummy_file = tmp_path / "shard.safetensors"
    dummy_file.write_bytes(b"\x00" * 4096)
    files = [str(dummy_file)]

    # Under warm cache (which would naturally pick Mode 1), test unified mode strings
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.moe_fast_loader.check_page_cache_warmth",
        lambda _: 0.95,
    )
    mode_str_dio = _resolve_and_broadcast_mode(files, mode="direct_io", tp_rank=0, tp_size=1)
    assert mode_str_dio == 3, f"Expected Mode 3 for mode='direct_io', got {mode_str_dio}"

    mode_str_vram = _resolve_and_broadcast_mode(files, mode="direct_vram", tp_rank=0, tp_size=1)
    assert mode_str_vram == 2, f"Expected Mode 2 for mode='direct_vram', got {mode_str_vram}"

    mode_str_staging = _resolve_and_broadcast_mode(files, mode="host_staging", tp_rank=0, tp_size=1)
    assert mode_str_staging == 1, f"Expected Mode 1 for mode='host_staging', got {mode_str_staging}"

    # Legacy use_direct_io=True forces Mode 3
    mode_dio = _resolve_and_broadcast_mode(
        files,
        use_direct_io=True,
        use_direct_vram=None,
        env_threshold_gb=None,
        direct_vram_threshold_gb=None,
        tp_rank=0,
        tp_size=1,
    )
    assert mode_dio == 3

    # Legacy use_direct_vram=True forces Mode 2
    mode_vram = _resolve_and_broadcast_mode(
        files,
        use_direct_io=None,
        use_direct_vram=True,
        env_threshold_gb=None,
        direct_vram_threshold_gb=None,
        tp_rank=0,
        tp_size=1,
    )
    assert mode_vram == 2

    # Threshold override smaller than file size forces Mode 2
    mode_thresh = _resolve_and_broadcast_mode(
        files,
        use_direct_io=None,
        use_direct_vram=None,
        env_threshold_gb=0.000001,  # ~1 KB, file is 4 KB
        direct_vram_threshold_gb=None,
        tp_rank=0,
        tp_size=1,
    )
    assert mode_thresh == 2


def test_resolve_and_broadcast_mode_cold_large_checkpoint(monkeypatch, tmp_path):
    """Verify that cold checkpoints (> 300 GB) strictly route to Mode 3 Direct-I/O Broadcast."""
    dummy_file = tmp_path / "shard.safetensors"
    dummy_file.write_bytes(b"\x00" * 4096)
    files = [str(dummy_file)]

    # 1. 500 GB checkpoint with cold cache (warmth = 70% < 80%)
    monkeypatch.setattr(os.path, "getsize", lambda _: 500 * (1024**3))
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.moe_fast_loader.check_page_cache_warmth",
        lambda _: 0.70,
    )
    mode_cold_large = _resolve_and_broadcast_mode(files, tp_rank=0, tp_size=8)
    assert mode_cold_large == 3, f"Cold 500 GB checkpoint must route to Mode 3, got {mode_cold_large}"

    # 2. When warm (warmth = 95%), 500 GB checkpoint routes to Mode 2 (under TP=8)
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.moe_fast_loader.check_page_cache_warmth",
        lambda _: 0.95,
    )
    mode_warm_large = _resolve_and_broadcast_mode(files, tp_rank=0, tp_size=8)
    assert mode_warm_large == 2, f"Warm 500 GB checkpoint under TP=8 must route to Mode 2, got {mode_warm_large}"


def test_resolve_and_broadcast_mode_distributed_sync(monkeypatch, tmp_path):
    """Verify that Rank 0 broadcasts selected mode to peer TP ranks when distributed is active."""
    dummy_file = tmp_path / "shard.safetensors"
    dummy_file.write_bytes(b"\x00" * 4096)
    files = [str(dummy_file)]

    # Mock warmth check on rank 0
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.moe_fast_loader.check_page_cache_warmth",
        lambda _: 0.15,
    )
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    broadcasted_data = []

    def mock_broadcast_object_list(object_list, src=0):
        if src == 0 and object_list[0] != 0:
            broadcasted_data.append(object_list[0])
        elif len(broadcasted_data) > 0:
            object_list[0] = broadcasted_data[0]

    monkeypatch.setattr(torch.distributed, "broadcast_object_list", mock_broadcast_object_list)

    # Rank 0 resolves mode 3 and broadcasts
    mode_r0 = _resolve_and_broadcast_mode(
        files,
        use_direct_io=None,
        use_direct_vram=None,
        env_threshold_gb=None,
        direct_vram_threshold_gb=None,
        tp_rank=0,
        tp_size=2,
    )
    assert mode_r0 == 3
    assert broadcasted_data == [3]

    # Rank 1 receives broadcasted mode 3
    mode_r1 = _resolve_and_broadcast_mode(
        files,
        use_direct_io=None,
        use_direct_vram=None,
        env_threshold_gb=None,
        direct_vram_threshold_gb=None,
        tp_rank=1,
        tp_size=2,
    )
    assert mode_r1 == 3


def test_safetensors_moe_index_layer_completion_multi_shard(tmp_path):
    """Verifies that SafetensorsMoEIndex correctly tracks shard spans and completion points."""
    num_experts = 8
    hidden_dim = 32
    intermediate_dim = 64

    # Shard 0: Layer 0 experts 0..3
    shard0_weights = {}
    for e in range(4):
        shard0_weights[f"model.layers.0.mlp.experts.{e}.gate_proj.weight"] = torch.randn(
            intermediate_dim, hidden_dim, dtype=torch.bfloat16
        )
        shard0_weights[f"model.layers.0.mlp.experts.{e}.up_proj.weight"] = torch.randn(
            intermediate_dim, hidden_dim, dtype=torch.bfloat16
        )
        shard0_weights[f"model.layers.0.mlp.experts.{e}.down_proj.weight"] = torch.randn(
            hidden_dim, intermediate_dim, dtype=torch.bfloat16
        )
    p0 = str(tmp_path / "model-00001-of-00003.safetensors")
    save_file(shard0_weights, p0)

    # Shard 1: Layer 0 experts 4..7, Layer 1 experts 0..3
    shard1_weights = {}
    for e in range(4, 8):
        shard1_weights[f"model.layers.0.mlp.experts.{e}.gate_proj.weight"] = torch.randn(
            intermediate_dim, hidden_dim, dtype=torch.bfloat16
        )
        shard1_weights[f"model.layers.0.mlp.experts.{e}.up_proj.weight"] = torch.randn(
            intermediate_dim, hidden_dim, dtype=torch.bfloat16
        )
        shard1_weights[f"model.layers.0.mlp.experts.{e}.down_proj.weight"] = torch.randn(
            hidden_dim, intermediate_dim, dtype=torch.bfloat16
        )
    for e in range(4):
        shard1_weights[f"model.layers.1.mlp.experts.{e}.gate_proj.weight"] = torch.randn(
            intermediate_dim, hidden_dim, dtype=torch.bfloat16
        )
        shard1_weights[f"model.layers.1.mlp.experts.{e}.up_proj.weight"] = torch.randn(
            intermediate_dim, hidden_dim, dtype=torch.bfloat16
        )
        shard1_weights[f"model.layers.1.mlp.experts.{e}.down_proj.weight"] = torch.randn(
            hidden_dim, intermediate_dim, dtype=torch.bfloat16
        )
    p1 = str(tmp_path / "model-00002-of-00003.safetensors")
    save_file(shard1_weights, p1)

    # Shard 2: Layer 1 experts 4..7
    shard2_weights = {}
    for e in range(4, 8):
        shard2_weights[f"model.layers.1.mlp.experts.{e}.gate_proj.weight"] = torch.randn(
            intermediate_dim, hidden_dim, dtype=torch.bfloat16
        )
        shard2_weights[f"model.layers.1.mlp.experts.{e}.up_proj.weight"] = torch.randn(
            intermediate_dim, hidden_dim, dtype=torch.bfloat16
        )
        shard2_weights[f"model.layers.1.mlp.experts.{e}.down_proj.weight"] = torch.randn(
            hidden_dim, intermediate_dim, dtype=torch.bfloat16
        )
    p2 = str(tmp_path / "model-00003-of-00003.safetensors")
    save_file(shard2_weights, p2)

    index = SafetensorsMoEIndex.build([p0, p1, p2])

    pfx0 = "model.layers.0.mlp.experts"
    pfx1 = "model.layers.1.mlp.experts"

    assert index.layer_shard_indices[pfx0] == {0, 1}
    assert index.layer_shard_indices[pfx1] == {1, 2}

    assert index.layers_completed_at_shard[0] == []
    assert index.layers_completed_at_shard[1] == [pfx0]
    assert index.layers_completed_at_shard[2] == [pfx1]


def test_streaming_3d_layer_stager_and_pool(synthetic_moe_checkpoint):
    """Verifies that _Streaming3DLayerStager packs 2D slices into 3D and recycles pinned buffers."""
    from vllm.model_executor.model_loader.moe_fast_loader import (
        PinnedHostStagingPool,
        _Streaming3DLayerStager,
    )

    shard_path, weights, num_layers, num_experts, hidden_dim, intermediate_dim = (
        synthetic_moe_checkpoint
    )
    index = SafetensorsMoEIndex.build([shard_path])
    pool = PinnedHostStagingPool(capacity_per_shape=4)
    stager = _Streaming3DLayerStager(index, pool)

    pfx = "model.layers.0.mlp.experts"
    plan = index.moe_layers[pfx]
    stager.get_or_create_layer(pfx)

    for e in range(num_experts):
        g = weights[f"{pfx}.{e}.gate_proj.weight"]
        u = weights[f"{pfx}.{e}.up_proj.weight"]
        d = weights[f"{pfx}.{e}.down_proj.weight"]
        gs = weights[f"{pfx}.{e}.gate_proj.weight_scale"]
        us = weights[f"{pfx}.{e}.up_proj.weight_scale"]
        ds = weights[f"{pfx}.{e}.down_proj.weight_scale"]

        stager.copy_slice(pfx, "gate", ".weight", e, g)
        stager.copy_slice(pfx, "up", ".weight", e, u)
        stager.copy_slice(pfx, "down", ".weight", e, d)
        stager.copy_slice(pfx, "gate", ".weight_scale", e, gs)
        stager.copy_slice(pfx, "up", ".weight_scale", e, us)
        stager.copy_slice(pfx, "down", ".weight_scale", e, ds)

    yielded = dict(stager.yield_completed_layer(pfx))

    # Check 3D shapes
    assert yielded[f"{pfx}.gate_up_proj"].shape == (num_experts, 2 * intermediate_dim, hidden_dim)
    assert yielded[f"{pfx}.down_proj"].shape == (num_experts, hidden_dim, intermediate_dim)
    assert yielded[f"{pfx}.gate_up_proj.weight_scale"].shape == (
        num_experts,
        2 * (intermediate_dim // 32),
        hidden_dim // 32,
    )
    assert yielded[f"{pfx}.down_proj.weight_scale"].shape == (
        num_experts,
        hidden_dim // 32,
        intermediate_dim // 32,
    )

    # Check numerical bitwise equivalence
    for e in range(num_experts):
        expected_g = weights[f"{pfx}.{e}.gate_proj.weight"]
        expected_u = weights[f"{pfx}.{e}.up_proj.weight"]
        actual_gu = yielded[f"{pfx}.gate_up_proj"][e]
        assert torch.equal(actual_gu[:intermediate_dim, :], expected_g)
        assert torch.equal(actual_gu[intermediate_dim:, :], expected_u)

        expected_d = weights[f"{pfx}.{e}.down_proj.weight"]
        assert torch.equal(yielded[f"{pfx}.down_proj"][e], expected_d)

    # Release layer and check pool recycling
    stager.release_layer(pfx)
    assert len(pool._pool) > 0
    pool.clear()


def test_streaming_3d_direct_io_broadcast_parity(synthetic_moe_checkpoint):
    """Verifies that _stream_direct_io_broadcast yields 3D MoE tensors and accurate non-MoE tensors."""
    from vllm.model_executor.model_loader.moe_fast_loader import _stream_direct_io_broadcast

    shard_path, weights, num_layers, num_experts, hidden_dim, intermediate_dim = (
        synthetic_moe_checkpoint
    )
    index = SafetensorsMoEIndex.build([shard_path])

    loaded_tensors = {}
    for name, tensor in _stream_direct_io_broadcast(
        [shard_path],
        index,
        tp_rank=0,
        tp_size=1,
    ):
        loaded_tensors[name] = tensor.clone()

    # Non-MoE tensors preserved
    assert "model.embed_tokens.weight" in loaded_tensors
    assert torch.equal(loaded_tensors["model.embed_tokens.weight"], weights["model.embed_tokens.weight"])
    assert "model.norm.weight" in loaded_tensors
    assert torch.equal(loaded_tensors["model.norm.weight"], weights["model.norm.weight"])

    # MoE weights emitted as 3D (no individual 2D slice keys!)
    for l in range(num_layers):
        pfx = f"model.layers.{l}.mlp.experts"
        assert f"{pfx}.gate_up_proj" in loaded_tensors
        assert f"{pfx}.down_proj" in loaded_tensors
        assert loaded_tensors[f"{pfx}.gate_up_proj"].dim() == 3
        assert loaded_tensors[f"{pfx}.down_proj"].dim() == 3

        # Zero 2D slice keys emitted
        for e in range(num_experts):
            assert f"{pfx}.{e}.gate_proj.weight" not in loaded_tensors
            assert f"{pfx}.{e}.down_proj.weight" not in loaded_tensors

        # Verify numerical parity for gate_up and down
        for e in range(num_experts):
            eg = weights[f"{pfx}.{e}.gate_proj.weight"]
            eu = weights[f"{pfx}.{e}.up_proj.weight"]
            actual_gu = loaded_tensors[f"{pfx}.gate_up_proj"][e]
            assert torch.equal(actual_gu[:intermediate_dim, :], eg)
            assert torch.equal(actual_gu[intermediate_dim:, :], eu)

            ed = weights[f"{pfx}.{e}.down_proj.weight"]
            assert torch.equal(loaded_tensors[f"{pfx}.down_proj"][e], ed)







