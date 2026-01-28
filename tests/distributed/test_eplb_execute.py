# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import random

import pytest
import torch
import torch.distributed

from vllm.distributed.eplb.rebalance_execute import (
    move_from_buffer,
    rearrange_expert_weights_inplace,
    transfer_layer,
)
from vllm.distributed.parallel_state import (
    ensure_model_parallel_initialized,
    get_tp_group,
)

from .eplb_utils import distributed_run, set_env_vars_and_device


def create_expert_indices_with_redundancy(
    num_layers: int,
    num_logical_experts: int,
    total_physical_experts: int,
    redundancy_config: list[int],  # redundancy for each logical expert
) -> torch.Tensor:
    """
    Create expert indices with redundancy.

    Args:
        num_layers: number of layers
        num_logical_experts: number of logical experts
        total_physical_experts: total number of physical experts
        redundancy_config: redundancy for each logical expert

    Returns:
        indices: Shape (num_layers, total_physical_experts)
    """
    assert sum(redundancy_config) == total_physical_experts
    assert len(redundancy_config) == num_logical_experts

    indices = torch.zeros(num_layers, total_physical_experts, dtype=torch.long)

    for layer in range(num_layers):
        physical_pos = 0
        for logical_expert_id, redundancy in enumerate(redundancy_config):
            for _ in range(redundancy):
                indices[layer, physical_pos] = logical_expert_id
                physical_pos += 1

    # Shuffle the indices at dim 1
    for layer in range(num_layers):
        indices[layer] = indices[layer][torch.randperm(indices.shape[1])]

    return indices


def create_expert_weights(
    num_layers: int,
    num_local_experts: int,
    hidden_sizes: list[int],
    rank: int,
    device: torch.device,
    physical_to_logical_mapping: torch.Tensor,
) -> list[list[torch.Tensor]]:
    """
    Create fake expert weights tensor for testing.

    Use `arange` to generate predictable weights values, based on logical
    expert ID.
    All replicas of the same logical expert should have the same weights.

    Args:
        physical_to_logical_mapping: Shape (num_layers, num_local_experts)
            mapping[layer, physical_pos] = logical_expert_id
    """
    expert_weights = []

    for layer in range(num_layers):
        layer_weights = []
        for weight_idx, hidden_size in enumerate(hidden_sizes):
            weight_tensor = torch.zeros(
                num_local_experts, hidden_size, device=device, dtype=torch.float32
            )

            for local_expert in range(num_local_experts):
                # Get the logical expert ID for this physical expert
                global_pos = rank * num_local_experts + local_expert
                logical_expert_id = physical_to_logical_mapping[
                    layer, global_pos
                ].item()

                # Generate weights based on logical expert ID
                # (so that all replicas of the same logical expert have the
                # same weights)
                base_value = logical_expert_id * 1000 + layer * 100 + weight_idx * 10
                weight_tensor[local_expert] = torch.arange(
                    base_value,
                    base_value + hidden_size,
                    device=device,
                    dtype=torch.float32,
                )

            layer_weights.append(weight_tensor)
        expert_weights.append(layer_weights)

    return expert_weights


def create_redundancy_config(
    num_logical_experts: int,
    num_physical_experts: int,
) -> list[int]:
    """Create a redundancy configuration."""
    redundancy_config = [1] * num_logical_experts
    remaining = num_physical_experts - num_logical_experts
    # Randomly assign the remaining physical experts to the logical experts
    for _ in range(remaining):
        redundancy_config[random.choice(range(num_logical_experts))] += 1
    return redundancy_config


def verify_expert_weights_after_shuffle(
    expert_weights: list[list[torch.Tensor]],
    new_indices: torch.Tensor,
    hidden_sizes: list[int],
    ep_rank: int,
    num_local_experts: int,
):
    """Verify the weights after shuffling are correct."""
    num_layers = len(expert_weights)

    for layer in range(num_layers):
        for weight_idx, hidden_size in enumerate(hidden_sizes):
            weight_tensor = expert_weights[layer][weight_idx]

            for local_expert in range(num_local_experts):
                # Calculate the global expert ID for this local expert
                global_pos = ep_rank * num_local_experts + local_expert
                expected_logical_expert = new_indices[layer, global_pos].item()

                # Check if the weights are correct
                actual_weights = weight_tensor[local_expert]
                expected_base = (
                    expected_logical_expert * 1000 + layer * 100 + weight_idx * 10
                )
                expected_weights = torch.arange(
                    expected_base,
                    expected_base + hidden_size,
                    device=actual_weights.device,
                    dtype=actual_weights.dtype,
                )

                torch.testing.assert_close(
                    actual_weights,
                    expected_weights,
                    msg=f"Layer {layer}, weight {weight_idx},"
                    f"local expert {local_expert}: "
                    f"weights do not match. "
                    f"Expected logical expert {expected_logical_expert}",
                )


def verify_redundant_experts_have_same_weights(
    expert_weights: list[list[torch.Tensor]],
    indices: torch.Tensor,
    hidden_sizes: list[int],
    world_size: int,
    num_local_experts: int,
):
    """
    Verify that all replicas of the same logical expert have the same weights.
    """
    num_layers = len(expert_weights)
    total_physical_experts = world_size * num_local_experts

    for layer in range(num_layers):
        # Collect weights for all physical experts for each weight matrix
        all_weights: list[torch.Tensor] = []

        for weight_idx, hidden_size in enumerate(hidden_sizes):
            # Create tensor to store all expert weights
            # Shape: [total_physical_experts, hidden_size]
            gathered_weights = torch.zeros(
                total_physical_experts,
                hidden_size,
                device=expert_weights[layer][weight_idx].device,
                dtype=expert_weights[layer][weight_idx].dtype,
            )

            # Use all_gather to collect expert weights from current node
            # expert_weights[layer][weight_idx] shape:
            # [num_local_experts, hidden_size]
            local_weights = expert_weights[layer][
                weight_idx
            ]  # [num_local_experts, hidden_size]

            # Split tensor along dim 0 into a list for all_gather
            gathered_weights_list = torch.chunk(gathered_weights, world_size, dim=0)

            torch.distributed.all_gather(
                # Output list: each element corresponds to one rank's weights
                list(gathered_weights_list),
                local_weights,  # Input: current rank's local weights
            )

            all_weights.append(gathered_weights)

        # Verify that all replicas of the same logical expert have the same
        # weights
        logical_expert_weights: dict[int, dict[int, torch.Tensor]] = {}

        for physical_pos in range(total_physical_experts):
            logical_expert_id = int(indices[layer, physical_pos].item())

            if logical_expert_id not in logical_expert_weights:
                # First time encountering this logical expert, save its weights
                logical_expert_weights[logical_expert_id] = {
                    weight_idx: all_weights[weight_idx][physical_pos]
                    for weight_idx in range(len(hidden_sizes))
                }
            else:
                # Verify that current physical expert's weights match the
                # previously saved logical expert weights
                for weight_idx in range(len(hidden_sizes)):
                    torch.testing.assert_close(
                        all_weights[weight_idx][physical_pos],
                        logical_expert_weights[logical_expert_id][weight_idx],
                        msg=f"Layer {layer}, weight {weight_idx},"
                        f"logical expert {logical_expert_id}: "
                        f"Physical expert {physical_pos} has different weights"
                        f"than expected",
                    )


def _test_async_transfer_layer_without_mtp_worker(
    env,
    world_size: int,
    num_layers: int,
    num_local_experts: int,
    num_logical_experts: int,
) -> None:
    set_env_vars_and_device(env)
    ensure_model_parallel_initialized(
        tensor_model_parallel_size=world_size, pipeline_model_parallel_size=1
    )

    tp_group = get_tp_group()
    ep_group = tp_group.device_group
    ep_rank = torch.distributed.get_rank()
    device = torch.device(f"cuda:{ep_rank}")

    total_physical_experts = world_size * num_local_experts
    hidden_sizes = [16, 32]

    redundancy_config = create_redundancy_config(
        num_logical_experts,
        total_physical_experts,
    )
    old_indices = create_expert_indices_with_redundancy(
        num_layers,
        num_logical_experts,
        total_physical_experts,
        redundancy_config,
    )

    new_redundancy_config = create_redundancy_config(
        num_logical_experts,
        total_physical_experts,
    )
    new_indices = create_expert_indices_with_redundancy(
        num_layers,
        num_logical_experts,
        total_physical_experts,
        new_redundancy_config,
    )

    expert_weights = create_expert_weights(
        num_layers,
        num_local_experts,
        hidden_sizes,
        ep_rank,
        device,
        old_indices,
    )
    old_indices_cpu = old_indices.cpu()
    new_indices_cpu = new_indices.cpu()

    expert_buffer = [torch.empty_like(w) for w in expert_weights[0]]
    cuda_stream = torch.cuda.Stream(device=device)

    for layer_idx in range(num_layers):
        is_unchanged, is_received_locally, recv_metadata = asyncio.run(
            transfer_layer(
                old_global_expert_indices=old_indices_cpu,
                new_global_expert_indices=new_indices_cpu,
                expert_weights=expert_weights,
                expert_weights_buffer=expert_buffer,
                ep_group=ep_group,
                layer=layer_idx,
                cuda_stream=cuda_stream,
            )
        )
        cuda_stream.synchronize()
        move_from_buffer(
            expert_weights=expert_weights[layer_idx],
            expert_weights_buffers=expert_buffer,
            is_unchanged=is_unchanged,
            is_received_locally=is_received_locally,
            recv_metadata=recv_metadata,
            new_indices=new_indices_cpu[layer_idx].numpy(),
            ep_rank=ep_rank,
        )

    verify_expert_weights_after_shuffle(
        expert_weights,
        new_indices,
        hidden_sizes,
        ep_rank,
        num_local_experts,
    )
    verify_redundant_experts_have_same_weights(
        expert_weights,
        new_indices,
        hidden_sizes,
        world_size,
        num_local_experts,
    )


def _test_rearrange_expert_weights_with_redundancy(
    env, world_size, num_layers, num_local_experts, num_logical_experts
) -> None:
    # Initialize model parallel (using tensor parallel as an entrypoint
    # to expert parallel)
    set_env_vars_and_device(env)
    ensure_model_parallel_initialized(
        tensor_model_parallel_size=world_size, pipeline_model_parallel_size=1
    )

    ep_group = get_tp_group().cpu_group
    ep_rank = torch.distributed.get_rank()
    device = torch.device(f"cuda:{ep_rank}")

    # Test parameters
    total_physical_experts = world_size * num_local_experts
    hidden_sizes = [32, 64]  # Two different weight matrices

    # Create old expert indices (with redundancy)
    redundancy_config = create_redundancy_config(
        num_logical_experts, total_physical_experts
    )

    old_indices = create_expert_indices_with_redundancy(
        num_layers,
        num_logical_experts,
        total_physical_experts,
        redundancy_config,
    )

    # Create new expert indices (with redundancy)
    new_redundancy_config = create_redundancy_config(
        num_logical_experts, total_physical_experts
    )
    new_indices = create_expert_indices_with_redundancy(
        num_layers,
        num_logical_experts,
        total_physical_experts,
        new_redundancy_config,
    )

    # Create expert weights
    expert_weights = create_expert_weights(
        num_layers, num_local_experts, hidden_sizes, ep_rank, device, old_indices
    )

    # Execute weight rearrangement
    rearrange_expert_weights_inplace(
        old_indices,
        new_indices,
        expert_weights,
        ep_group,
        is_profile=False,
    )

    # Verify the rearrangement result
    verify_expert_weights_after_shuffle(
        expert_weights,
        new_indices,
        hidden_sizes,
        ep_rank,
        num_local_experts,
    )

    verify_redundant_experts_have_same_weights(
        expert_weights,
        new_indices,
        hidden_sizes,
        world_size,
        num_local_experts,
    )


@pytest.mark.parametrize(
    "world_size,num_layers,num_local_experts,num_logical_experts",
    [
        # 2 GPU, 2 experts per GPU
        # 3 logical experts, 4 physical experts, 1 redundant experts
        (2, 1, 2, 3),
        # 2 GPU, 3 experts per GPU
        # 4 logical experts, 6 physical experts, 2 redundant experts
        (2, 2, 3, 4),
        # 2 GPU, 8 experts per GPU
        # 16 logical experts, 16 physical experts, 0 redundant experts
        (2, 4, 8, 16),
        # 4 GPU, 2 experts per GPU
        # 6 logical experts, 8 physical experts, 2 redundant experts
        (4, 1, 2, 6),
        # 4 GPU, 2 experts per GPU
        # 5 logical experts, 8 physical experts, 3 redundant experts
        (4, 2, 2, 5),
        # 4 GPU, 8 experts per GPU
        # 16 logical experts, 32 physical experts, 16 redundant experts
        (4, 8, 8, 16),
    ],
)
def test_rearrange_expert_weights_with_redundancy(
    world_size, num_layers, num_local_experts, num_logical_experts
):
    """Test the functionality of rearranging expert weights with redundancy."""

    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Need at least {world_size} GPUs to run the test")
    distributed_run(
        _test_rearrange_expert_weights_with_redundancy,
        world_size,
        num_layers,
        num_local_experts,
        num_logical_experts,
    )


def _test_rearrange_expert_weights_no_change(env, world_size) -> None:
    set_env_vars_and_device(env)
    ensure_model_parallel_initialized(
        tensor_model_parallel_size=world_size, pipeline_model_parallel_size=1
    )

    ep_group = get_tp_group().cpu_group
    ep_rank = torch.distributed.get_rank()
    device = torch.device(f"cuda:{ep_rank}")

    num_layers = 2
    num_local_experts = 2
    total_physical_experts = world_size * num_local_experts
    num_logical_experts = total_physical_experts // 2  # Some redundancy
    hidden_sizes = [32, 64]

    # Create redundancy configuration
    redundancy_config = [2] * num_logical_experts

    # Same indices - no change
    indices = create_expert_indices_with_redundancy(
        num_layers, num_logical_experts, total_physical_experts, redundancy_config
    )

    expert_weights = create_expert_weights(
        num_layers, num_local_experts, hidden_sizes, ep_rank, device, indices
    )

    # Save original weights
    original_weights = []
    for layer_weights in expert_weights:
        layer_copy = []
        for weight in layer_weights:
            layer_copy.append(weight.clone())
        original_weights.append(layer_copy)

    # Execute rearrangement (should be no change)
    rearrange_expert_weights_inplace(
        indices,
        indices,  # Same indices
        expert_weights,
        ep_group,
        is_profile=False,
    )

    # Verify that the weights have not changed
    for layer in range(num_layers):
        for weight_idx in range(len(hidden_sizes)):
            torch.testing.assert_close(
                expert_weights[layer][weight_idx],
                original_weights[layer][weight_idx],
                msg=f"""Layer {layer}, weight {weight_idx}
 should remain unchanged""",
            )


@pytest.mark.parametrize(
    "world_size,num_layers,num_local_experts,num_logical_experts",
    [
        (2, 2, 2, 3),
    ],
)
def test_async_transfer_layer_without_mtp(
    world_size: int,
    num_layers: int,
    num_local_experts: int,
    num_logical_experts: int,
):
    """Exercise async EPLB transfer path without MTP/spec decode."""

    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Need at least {world_size} GPUs to run the test")

    distributed_run(
        _test_async_transfer_layer_without_mtp_worker,
        world_size,
        num_layers,
        num_local_experts,
        num_logical_experts,
    )


@pytest.mark.parametrize("world_size", [2, 4])
def test_rearrange_expert_weights_no_change(world_size):
    """
    Test that when the indices do not change, the weights should remain
    unchanged.
    """

    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Need at least {world_size} GPUs to run the test")
    distributed_run(_test_rearrange_expert_weights_no_change, world_size)


def _test_rearrange_expert_weights_profile_mode(env, world_size) -> None:
    set_env_vars_and_device(env)
    ensure_model_parallel_initialized(
        tensor_model_parallel_size=world_size, pipeline_model_parallel_size=1
    )

    ep_group = get_tp_group().cpu_group
    ep_rank = torch.distributed.get_rank()
    device = torch.device(f"cuda:{ep_rank}")

    num_layers = 1
    num_local_experts = 2
    total_physical_experts = world_size * num_local_experts
    num_logical_experts = total_physical_experts // 2
    hidden_sizes = [32]

    # Create different index distributions
    old_redundancy = create_redundancy_config(
        num_logical_experts, total_physical_experts
    )
    new_redundancy = create_redundancy_config(
        num_logical_experts, total_physical_experts
    )

    old_indices = create_expert_indices_with_redundancy(
        num_layers, num_logical_experts, total_physical_experts, old_redundancy
    )
    new_indices = create_expert_indices_with_redundancy(
        num_layers, num_logical_experts, total_physical_experts, new_redundancy
    )

    expert_weights = create_expert_weights(
        num_layers, num_local_experts, hidden_sizes, ep_rank, device, old_indices
    )

    # Save original weights
    original_weights = []
    for layer_weights in expert_weights:
        layer_copy = []
        for weight in layer_weights:
            layer_copy.append(weight.clone())
        original_weights.append(layer_copy)

    # Execute profile mode rearrangement
    rearrange_expert_weights_inplace(
        old_indices,
        new_indices,
        expert_weights,
        ep_group,
        is_profile=True,  # Profile mode
    )

    # In profile mode, the weights should remain unchanged
    for layer in range(num_layers):
        for weight_idx in range(len(hidden_sizes)):
            torch.testing.assert_close(
                expert_weights[layer][weight_idx],
                original_weights[layer][weight_idx],
                msg="In profile mode, the weights should remain unchanged",
            )


def _test_rearrange_expert_weights_over_limit(env, world_size) -> None:
    set_env_vars_and_device(env)
    ensure_model_parallel_initialized(
        tensor_model_parallel_size=world_size, pipeline_model_parallel_size=1
    )

    ep_group = get_tp_group().cpu_group
    ep_rank = torch.distributed.get_rank()
    device = torch.device(f"cuda:{ep_rank}")

    num_layers = 1
    num_local_experts = 2
    total_physical_experts = world_size * num_local_experts
    num_logical_experts = total_physical_experts // 2
    hidden_sizes = [32]

    # Create different index distributions
    old_redundancy = create_redundancy_config(
        num_logical_experts, total_physical_experts
    )
    new_redundancy = create_redundancy_config(
        num_logical_experts, total_physical_experts
    )

    old_indices = create_expert_indices_with_redundancy(
        num_layers, num_logical_experts, total_physical_experts, old_redundancy
    )
    new_indices = create_expert_indices_with_redundancy(
        num_layers, num_logical_experts, total_physical_experts, new_redundancy
    )

    expert_weights = create_expert_weights(
        num_layers, num_local_experts, hidden_sizes, ep_rank, device, old_indices
    )

    # Save original weights
    original_weights = []
    for layer_weights in expert_weights:
        layer_copy = []
        for weight in layer_weights:
            layer_copy.append(weight.clone())
        original_weights.append(layer_copy)

    # Execute profile mode rearrangement
    rearrange_expert_weights_inplace(
        old_indices,
        new_indices,
        expert_weights,
        ep_group,
    )

    # In profile mode, the weights should remain unchanged
    for layer in range(num_layers):
        for weight_idx in range(len(hidden_sizes)):
            torch.testing.assert_close(
                expert_weights[layer][weight_idx],
                original_weights[layer][weight_idx],
                msg="In profile mode, the weights should remain unchanged",
            )


@pytest.mark.parametrize("world_size", [2, 4])
def test_rearrange_expert_weights_profile_mode(world_size):
    """Test profile mode (should not copy actual weights)"""

    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Need at least {world_size} GPUs to run the test")
    distributed_run(_test_rearrange_expert_weights_profile_mode, world_size)
