# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest
import torch

from vllm.distributed.eplb.policy.flashlb import FlashlbEplbPolicy


def test_basic_rebalance():
    """Test basic rebalance functionality of FlashLB"""
    weight = torch.tensor(
        [
            [90, 132, 40, 61, 104, 165, 39, 4, 73, 56, 183, 86],
            [20, 107, 104, 64, 19, 197, 187, 157, 172, 86, 16, 27],
        ]
    )
    num_layers, num_expert = weight.shape
    num_replicas = 16
    num_groups = 4
    num_nodes = 2
    num_gpus = 8
    policy = FlashlbEplbPolicy()
    phy2log, log2phy, logcnt = policy.rebalance_experts(
        weight, num_replicas, num_groups, num_nodes, num_gpus
    )
    assert phy2log.shape == (
        2,
        16,
    ), f"Expected `phy2log` shape (2, 16), got {phy2log.shape}"
    assert log2phy.shape[0] == 2, (
        f"Expected `log2phy` first dimension 2, got {log2phy.shape[0]}"
    )
    assert log2phy.shape[1] == 12, (
        f"Expected `log2phy` second dimension 12, got {log2phy.shape[1]}"
    )
    assert logcnt.shape == (
        2,
        12,
    ), f"Expected `logcnt` shape (2, 12), got {logcnt.shape}"
    # Verify physical to logical expert mapping range is correct
    assert torch.all(phy2log >= 0) and torch.all(phy2log < 12), (
        "Physical to logical mapping should be in range [0, 12)"
    )
    # Verify expert count reasonableness
    assert torch.all(logcnt >= 1), "Each logical expert should have at least 1 replica"
    assert torch.sum(logcnt, dim=1).sum() == num_replicas * num_layers, (
        f"Total replicas should be {num_replicas * num_layers}"
    )
    # Verify expected output
    expected_phy2log = torch.tensor(
        [
            [1, 7, 4, 3, 10, 2, 10, 6, 0, 8, 11, 8, 5, 9, 5, 3],
            [1, 3, 2, 11, 5, 0, 5, 4, 6, 7, 6, 7, 9, 8, 8, 10],
        ]
    )
    assert torch.all(phy2log == expected_phy2log)
    expected_logcnt = torch.tensor(
        [[1, 1, 1, 2, 1, 2, 1, 1, 2, 1, 2, 1], [1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1]]
    )
    assert torch.all(logcnt == expected_logcnt)


def test_single_gpu_case():
    """Test case with a single GPU"""
    weight = torch.tensor([[10, 20, 30, 40]])
    num_replicas = 4
    num_groups = 1
    num_nodes = 1
    num_gpus = 1
    policy = FlashlbEplbPolicy()
    phy2log, log2phy, logcnt = policy.rebalance_experts(
        weight, num_replicas, num_groups, num_nodes, num_gpus
    )
    # Verify shapes
    assert phy2log.shape == (1, 4)
    assert log2phy.shape[0] == 1
    assert log2phy.shape[1] == 4
    assert logcnt.shape == (1, 4)
    # Verify all logical experts are mapped
    assert set(phy2log[0].tolist()) == {0, 1, 2, 3}


def test_equal_weights():
    """Test case with equal weights"""
    weight = torch.tensor([[50, 50, 50, 50, 50, 50, 50, 50]])
    num_replicas = 16
    num_groups = 2
    num_nodes = 2
    num_gpus = 4
    policy = FlashlbEplbPolicy()
    phy2log, log2phy, logcnt = policy.rebalance_experts(
        weight, num_replicas, num_groups, num_nodes, num_gpus
    )
    assert phy2log.shape == (1, 16)
    assert logcnt.shape == (1, 8)
    assert torch.all(logcnt == 2), (
        "With equal weights each expert should have exactly 2 replicas"
    )


def test_extreme_weight_imbalance():
    """Test extreme weight imbalance case"""
    weight = torch.tensor([[1000, 1, 1, 1, 1, 1, 1, 1]])
    num_replicas = 12
    num_groups = 2
    num_nodes = 2
    num_gpus = 4
    policy = FlashlbEplbPolicy()
    phy2log, log2phy, logcnt = policy.rebalance_experts(
        weight, num_replicas, num_groups, num_nodes, num_gpus
    )
    assert phy2log.shape == (1, 12)
    assert logcnt.shape == (1, 8)
    # Expert with highest weight (index 0) should have more replicas
    assert logcnt[0, 0] == max(logcnt[0]), (
        "Expert with extreme weight should have more replicas than others"
    )


def test_multiple_layers():
    """Test multiple layers case"""
    weight = torch.tensor(
        [
            [10, 20, 30, 40, 50, 60],  # First layer
            [60, 50, 40, 30, 20, 10],  # Second layer (opposite weight pattern)
            [25, 25, 25, 25, 25, 25],  # Third layer (equal weights)
        ]
    )
    num_replicas = 8
    num_groups = 2
    num_nodes = 2
    num_gpus = 4
    policy = FlashlbEplbPolicy()
    phy2log, log2phy, logcnt = policy.rebalance_experts(
        weight, num_replicas, num_groups, num_nodes, num_gpus
    )
    assert phy2log.shape == (3, 8)
    assert logcnt.shape == (3, 6)
    # Verify expert allocation is reasonable for each layer
    for layer in range(3):
        assert torch.all(phy2log[layer] >= 0) and torch.all(phy2log[layer] < 6), (
            f"Layer {layer} physical to logical mapping should be in range [0, 6)"
        )
        assert torch.all(logcnt[layer] >= 1), (
            "Each logical expert should have at least 1 replica"
        )
        assert torch.sum(logcnt[layer]) == num_replicas, (
            f"Layer {layer} total replicas should be {num_replicas}"
        )


def test_parameter_validation():
    """Test parameter validation"""
    weight = torch.tensor([[10, 20, 30, 40]])
    policy = FlashlbEplbPolicy()
    phy2log, log2phy, logcnt = policy.rebalance_experts(weight, 8, 3, 2, 4)
    assert phy2log.shape == (1, 8)
    assert logcnt.shape == (1, 4)
    with pytest.raises(AssertionError):
        policy.rebalance_experts(weight, 7, 2, 2, 4)


def test_small_scale_hierarchical():
    """Test small-scale hierarchical load balancing"""
    weight = torch.tensor(
        [
            [100, 50, 200, 75, 150, 25, 300, 80],  # 8 experts
        ]
    )
    num_replicas = 12
    num_groups = 4  # 4 groups, 2 experts each
    num_nodes = 2  # 2 nodes
    num_gpus = 4  # 4 GPUs
    policy = FlashlbEplbPolicy()
    phy2log, log2phy, logcnt = policy.rebalance_experts(
        weight, num_replicas, num_groups, num_nodes, num_gpus
    )
    assert phy2log.shape == (1, 12)
    assert logcnt.shape == (1, 8)
    assert torch.sum(logcnt) == num_replicas
    assert torch.all(logcnt >= 1)
    # Expert with highest weight should have more replicas
    max_weight_expert = torch.argmax(weight[0])
    assert logcnt[0, max_weight_expert] >= 2, (
        "Highest weight expert should have multiple replicas"
    )


def test_global_load_balance_fallback():
    """Test global load balancing fallback case"""
    # When num_groups % num_nodes != 0, should fall back to global load
    # balancing
    weight = torch.tensor([[10, 20, 30, 40, 50, 60]])
    num_replicas = 8
    num_groups = 3  # Cannot be divided evenly by num_nodes=2
    num_nodes = 2
    num_gpus = 4
    policy = FlashlbEplbPolicy()
    phy2log, log2phy, logcnt = policy.rebalance_experts(
        weight, num_replicas, num_groups, num_nodes, num_gpus
    )
    assert phy2log.shape == (1, 8)
    assert logcnt.shape == (1, 6)
    assert torch.sum(logcnt) == num_replicas


@pytest.mark.parametrize("device", ["cpu", "npu", "cuda"])
def test_device_compatibility(device):
    """Test device compatibility"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    weight = torch.tensor([[10, 20, 30, 40]], device=device)
    num_replicas = 6
    num_groups = 2
    num_nodes = 1
    num_gpus = 2
    policy = FlashlbEplbPolicy()
    phy2log, log2phy, logcnt = policy.rebalance_experts(
        weight, num_replicas, num_groups, num_nodes, num_gpus
    )
    assert phy2log.shape == (1, 6)
    assert logcnt.shape == (1, 4)


def generate_expert_map(
    global_expert_num, world_size, num_moe_layers, num_of_redundant_expert=0
):
    base_experts = global_expert_num // world_size
    redundant_experts = num_of_redundant_expert // world_size
    remainder = global_expert_num % world_size
    local_num_experts = base_experts + remainder + redundant_experts
    expert_map_tensor = torch.full(
        (num_moe_layers, world_size, global_expert_num), -1, dtype=torch.int32
    )
    for device_id in range(world_size):
        local_ids = torch.arange(base_experts + redundant_experts, dtype=torch.int32)
        expand_ids = torch.arange(
            base_experts + redundant_experts, local_num_experts, dtype=torch.int32
        )

        if device_id < world_size - 1:
            start = device_id * base_experts
            end = start + base_experts + redundant_experts
            expert_map_tensor[:, device_id, start:end] = local_ids.unsqueeze(0).expand(
                num_moe_layers, -1
            )
        else:
            if remainder > 0:
                slice_end = -remainder
                slice_start = slice_end - (base_experts + redundant_experts)
            else:
                slice_start = -(base_experts + redundant_experts)
                slice_end = None
            expert_map_tensor[:, device_id, slice_start:slice_end] = (
                local_ids.unsqueeze(0).expand(num_moe_layers, -1)
            )

        if remainder > 0:
            expert_map_tensor[:, device_id, -remainder:] = expand_ids.unsqueeze(
                0
            ).expand(num_moe_layers, -1)
    expert_map = []
    for layer_id in range(num_moe_layers):
        layer_expert = []
        for device_id in range(world_size):
            valid_global_experts = torch.where(
                expert_map_tensor[layer_id, device_id] != -1
            )[0]
            layer_expert.append(valid_global_experts.tolist())
        expert_map.append(layer_expert)
    return torch.tensor(expert_map, dtype=int)


def test_experts_exchange():
    """Test experts exchange"""
    torch.manual_seed(27)
    num_layers = 10
    num_groups = 4
    num_nodes = 2
    num_gpus = 32
    num_expert = 256
    num_replicas = 288
    replicas_per_gpu = num_replicas // num_gpus

    weight = torch.randint(1, 1000, (num_layers, num_expert))
    expert_map = generate_expert_map(
        global_expert_num=num_expert,
        world_size=num_gpus,
        num_moe_layers=num_layers,
        num_of_redundant_expert=num_replicas - num_expert,
    )
    policy = FlashlbEplbPolicy()
    phy2log, log2phy, logcnt = policy.rebalance_experts(
        weight, num_replicas, num_groups, num_nodes, num_gpus, expert_map
    )
    new_placement = phy2log.reshape((num_layers, num_gpus, replicas_per_gpu))
    for layer_id in range(num_layers):
        num_old_expert = torch.unique(expert_map[layer_id]).numel()
        num_new_expert = torch.unique(new_placement[layer_id]).numel()
        assert num_new_expert == num_old_expert, (
            f"There exists expert not placed on any rank in layer {layer_id}"
        )

        for gpu_id in range(num_gpus):
            new_placement_check = new_placement[layer_id][gpu_id]
            old_placement_check = expert_map[layer_id][gpu_id]

            # Check if same logical experts are placed on the same NPU
            new_unique = torch.unique(new_placement_check)
            assert new_placement_check.numel() == new_unique.numel(), (
                f"Replicated experts are placed on the same NPU, "
                f"expert placement on layer {layer_id}, "
                f"rank {gpu_id} is invalid"
            )

            # Check if there is any experts movement inside one NPU
            expert_not_move = torch.isin(new_placement_check, old_placement_check)
            new_retained = new_placement_check[expert_not_move]
            old_retained = old_placement_check[expert_not_move]
            assert torch.equal(new_retained, old_retained), (
                f"There exists expert movement inside NPU, "
                f"expert placement on layer {layer_id}, "
                f"rank {gpu_id} is invalid"
            )
