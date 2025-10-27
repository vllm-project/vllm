# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math

import pytest
import torch
import torch.multiprocessing as mp

from tests.utils import multi_gpu_test
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.models.vision import (
    get_load_balance_assignment,
    resolve_visual_encoder_outputs,
    run_dp_sharded_mrope_vision_model,
    run_dp_sharded_vision_model,
)
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    ("select_layers", "num_layers_loaded", "max_possible_layers", "expected_features"),
    [
        # All layers loaded
        ([1, 10], 10, 10, [1, 10]),
        ([-10, -1], 10, 10, [1, 10]),
        # Some layers not loaded
        ([1, 10], 10, 20, [1, 10]),
        ([-20, -11], 10, 20, [1, 10]),
    ],
)
def test_resolve_visual_encoder_outputs(
    select_layers, num_layers_loaded, max_possible_layers, expected_features
):
    """
    Test that offsets are correctly handled for vision feature layers.
    """
    encoder_outputs = [torch.tensor([idx]) for idx in range(num_layers_loaded + 1)]
    output_tensor = resolve_visual_encoder_outputs(
        encoder_outputs=encoder_outputs,
        post_layer_norm=None,
        select_layers=select_layers,
        max_possible_layers=max_possible_layers,
    )
    assert torch.equal(torch.tensor(expected_features), output_tensor)


class SimpleLinearModel(torch.nn.Module):
    """A simple linear vision model for testing."""

    def __init__(self, input_dim: int = 3 * 224 * 224, output_dim: int = 32):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor):
        # Flatten the input and apply linear transformation
        x = self.flatten(x)
        return self.linear(x)


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize(
    "batch_size",
    [
        1,  # Single image
        4,  # Small batch
        5,  # Odd batch size (for testing padding)
    ],
)
def test_run_dp_sharded_vision_model(batch_size: int):
    world_size = 2
    # Launch processes
    mp.spawn(
        run_dp_sharded_vision_model_vs_direct,
        args=(
            world_size,
            batch_size,
            get_open_port(),
        ),
        nprocs=world_size,
    )


def run_dp_sharded_vision_model_vs_direct(
    local_rank: int, world_size: int, batch_size: int, master_port: int
):
    """
    Test that run_dp_sharded_vision_model produces the same results as
    calling the model directly.
    """

    # Set random seed for reproducibility
    current_platform.seed_everything(0)

    device = f"{current_platform.device_name}:{local_rank}"
    current_platform.set_device(device)
    torch.set_default_device(device)

    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(master_port),
        }
    )

    # initialize distributed
    init_distributed_environment()
    initialize_model_parallel(tensor_model_parallel_size=world_size)

    # Create a test input tensor
    image_input = torch.randn(batch_size, 3, 224, 224)

    # Create a simple linear model
    vision_model = SimpleLinearModel()

    # Run the model directly on the full input
    with torch.inference_mode():
        direct_output = vision_model(image_input)

    # Run the model through the sharded function
    with torch.inference_mode():
        sharded_output = run_dp_sharded_vision_model(image_input, vision_model)

    # Check that the world size is set up correctly
    assert get_tensor_model_parallel_world_size() == world_size

    # Check that the outputs have the same shape
    assert direct_output.shape == sharded_output.shape

    # Check that the outputs are close (they should be identical)
    assert torch.allclose(direct_output, sharded_output, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "sizes,num_gpus,expected_shuffle_indices,expected_gpu_sample_counts,"
    "expected_grouped_sizes_per_gpu,test_description",
    [
        # Empty input
        ([], 2, [], [0, 0], [0, 0], "empty input"),
        # Fewer samples than GPUs
        (
            [100, 200],
            4,
            [1, 0],
            [1, 1, 0, 0],
            [200, 100, 0, 0],
            "fewer samples than GPUs",
        ),
        # Single GPU
        ([100, 200, 300], 1, [2, 1, 0], [3], [600], "single GPU"),
        # Balanced assignment
        (
            [100, 100, 100, 100],
            2,
            [0, 2, 1, 3],
            [2, 2],
            [200, 200],
            "balanced assignment",
        ),
        # Unbalanced sizes - this one is trickier since the algorithm is greedy
        (
            [1000, 100, 200, 50],
            2,
            [0, 2, 1, 3],
            [1, 3],
            [1000, 350],
            "unbalanced sizes",
        ),
    ],
)
def test_get_load_balance_assignment_cases(
    sizes,
    num_gpus,
    expected_shuffle_indices,
    expected_gpu_sample_counts,
    expected_grouped_sizes_per_gpu,
    test_description,
):
    """Test get_load_balance_assignment with various input cases."""
    result = get_load_balance_assignment(sizes, num_gpus=num_gpus)
    (shuffle_indices, gpu_sample_counts, grouped_sizes_per_gpu) = result

    # Common assertions for all cases
    assert len(shuffle_indices) == len(sizes)
    assert len(gpu_sample_counts) == num_gpus
    assert len(grouped_sizes_per_gpu) == num_gpus
    assert sum(gpu_sample_counts) == len(sizes)

    assert shuffle_indices == expected_shuffle_indices

    assert gpu_sample_counts == expected_gpu_sample_counts
    assert grouped_sizes_per_gpu == expected_grouped_sizes_per_gpu


class SimpleMRopeVisionModel(torch.nn.Module):
    """A simple vision model for testing mrope functionality."""

    def __init__(self, spatial_merge_size: int = 2, out_hidden_size: int = 64):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        self.out_hidden_size = out_hidden_size
        self.linear = torch.nn.Linear(768, out_hidden_size)

    def forward(self, pixel_values: torch.Tensor, grid_thw_list: list[list[int]]):
        """Simple forward pass that simulates spatial merging."""
        # Apply linear transformation
        embeddings = self.linear(pixel_values)

        # Simulate spatial merging by reducing the number of patches
        merge_factor = self.spatial_merge_size * self.spatial_merge_size

        # Group patches and merge spatially
        merged_embeddings = []
        start_idx = 0

        for grid_thw in grid_thw_list:
            num_patches = math.prod(grid_thw)
            end_idx = start_idx + num_patches

            # Get patches for this image
            image_patches = embeddings[start_idx:end_idx]

            # Simulate spatial merging by averaging groups of patches
            merged_patches = num_patches // merge_factor
            if merged_patches > 0:
                # Reshape and average to simulate merging
                reshaped = image_patches[: merged_patches * merge_factor].view(
                    merged_patches, merge_factor, -1
                )
                merged = reshaped.mean(dim=1)
                merged_embeddings.append(merged)

            start_idx = end_idx

        if merged_embeddings:
            return torch.cat(merged_embeddings, dim=0)
        else:
            return torch.empty(
                (0, self.out_hidden_size),
                device=pixel_values.device,
                dtype=pixel_values.dtype,
            )


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize(
    "batch_size",
    [
        1,  # Single image
        3,  # Small batch
        5,  # Odd batch size (for testing padding)
    ],
)
def test_run_dp_sharded_mrope_vision_model(batch_size: int):
    world_size = 2
    # Launch processes
    mp.spawn(
        run_dp_sharded_mrope_vision_model_vs_direct,
        args=(
            world_size,
            batch_size,
            get_open_port(),
        ),
        nprocs=world_size,
    )


def run_dp_sharded_mrope_vision_model_vs_direct(
    local_rank: int, world_size: int, batch_size: int, master_port: int
):
    """
    Test that run_dp_sharded_mrope_vision_model produces the same results as
    calling the model directly.
    """
    # Set random seed for reproducibility
    current_platform.seed_everything(0)
    device = f"{current_platform.device_name}:{local_rank}"
    current_platform.set_device(device)
    torch.set_default_device(device)

    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(master_port),
        }
    )

    # initialize distributed
    init_distributed_environment()
    initialize_model_parallel(tensor_model_parallel_size=world_size)

    # Create test data
    grid_thw_list = []
    pixel_values_list = []

    for i in range(batch_size):
        # Varying image sizes for better testing
        t, h, w = 1, 4 + i, 4 + i
        grid_thw_list.append([t, h, w])

        num_patches = t * h * w
        # Create random pixel values for this image
        image_pixels = torch.randn(num_patches, 768)
        pixel_values_list.append(image_pixels)

    # Concatenate all pixel values
    pixel_values = torch.cat(pixel_values_list, dim=0)

    # Create a simple mrope vision model
    vision_model = SimpleMRopeVisionModel()

    # Run the model directly on the full input (only on rank 0)
    if local_rank == 0:
        with torch.inference_mode():
            direct_output = vision_model(pixel_values, grid_thw_list)

    # Run the model through the sharded function
    with torch.inference_mode():
        sharded_output = run_dp_sharded_mrope_vision_model(
            vision_model, pixel_values, grid_thw_list, rope_type="rope_3d"
        )
        sharded_output = torch.cat(sharded_output, dim=0)

    # Check that the world size is set up correctly
    assert get_tensor_model_parallel_world_size() == world_size

    # Compare outputs (only on rank 0)
    if local_rank == 0:
        # Check that the outputs have the same shape
        assert direct_output.shape == sharded_output.shape
        # Check that the outputs are close (they should be identical)
        assert torch.allclose(direct_output, sharded_output, rtol=1e-5, atol=1e-5)


@multi_gpu_test(num_gpus=2)
def test_run_dp_sharded_mrope_vision_model_empty_input():
    world_size = 2
    mp.spawn(
        run_dp_sharded_mrope_vision_model_empty_input_worker,
        args=(world_size, get_open_port()),
        nprocs=world_size,
    )


def run_dp_sharded_mrope_vision_model_empty_input_worker(
    local_rank: int, world_size: int, master_port: int
):
    """Test run_dp_sharded_mrope_vision_model with empty input."""
    # Set up distributed environment
    device = f"{current_platform.device_name}:{local_rank}"
    current_platform.set_device(device)
    torch.set_default_device(device)

    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(master_port),
        }
    )

    init_distributed_environment()
    initialize_model_parallel(tensor_model_parallel_size=world_size)

    # Create empty inputs
    pixel_values = torch.empty((0, 768))
    grid_thw_list: list[list[int]] = []

    vision_model = SimpleMRopeVisionModel()

    # Should handle empty input gracefully
    with torch.inference_mode():
        output = run_dp_sharded_mrope_vision_model(
            vision_model, pixel_values, grid_thw_list, rope_type="rope_3d"
        )

    assert len(output) == 0


@multi_gpu_test(num_gpus=4)
def test_run_dp_sharded_mrope_vision_model_uneven_load():
    world_size = 4
    mp.spawn(
        run_dp_sharded_mrope_vision_model_uneven_load_worker,
        args=(world_size, get_open_port()),
        nprocs=world_size,
    )


def run_dp_sharded_mrope_vision_model_uneven_load_worker(
    local_rank: int, world_size: int, master_port: int
):
    """Test run_dp_sharded_mrope_vision_model with uneven load distribution."""
    # Set up distributed environment
    current_platform.seed_everything(123)
    device = f"{current_platform.device_name}:{local_rank}"
    current_platform.set_device(device)
    torch.set_default_device(device)

    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(master_port),
        }
    )

    init_distributed_environment()
    initialize_model_parallel(tensor_model_parallel_size=world_size)

    # Create images with very different sizes
    grid_thw_list = [
        [1, 2, 2],  # Small: 4 patches
        [1, 8, 8],  # Large: 64 patches
        [1, 3, 3],  # Medium: 9 patches
    ]

    pixel_values_list = []
    for grid_thw in grid_thw_list:
        num_patches = math.prod(grid_thw)
        image_pixels = torch.randn(num_patches, 768)
        pixel_values_list.append(image_pixels)

    pixel_values = torch.cat(pixel_values_list, dim=0)
    vision_model = SimpleMRopeVisionModel()

    # Should handle uneven distribution without errors
    with torch.inference_mode():
        output_tuple = run_dp_sharded_mrope_vision_model(
            vision_model, pixel_values, grid_thw_list, rope_type="rope_3d"
        )

    # Verify output shape is reasonable
    merge_factor = vision_model.spatial_merge_size**2
    expected_output_patches = list(
        math.prod(grid_thw) // merge_factor for grid_thw in grid_thw_list
    )

    for i, output in enumerate(output_tuple):
        assert output.shape[0] == expected_output_patches[i]
        assert output.shape[1] == vision_model.out_hidden_size


@pytest.mark.parametrize("spatial_merge_size", [2, 4])
def test_simple_mrope_vision_model_spatial_merge(spatial_merge_size: int):
    """Test SimpleMRopeVisionModel with different spatial merge sizes."""
    device = current_platform.device_type

    grid_thw_list = [[1, 4, 4], [1, 6, 6]]  # Two images
    pixel_values_list = []

    for grid_thw in grid_thw_list:
        num_patches = math.prod(grid_thw)
        image_pixels = torch.randn(num_patches, 768, device=device)
        pixel_values_list.append(image_pixels)

    pixel_values = torch.cat(pixel_values_list, dim=0)
    vision_model = SimpleMRopeVisionModel(spatial_merge_size=spatial_merge_size).to(
        device
    )

    with torch.inference_mode():
        output = vision_model(pixel_values, grid_thw_list)

    # Verify output dimensions based on spatial merging
    total_patches = sum(math.prod(grid_thw) for grid_thw in grid_thw_list)
    merge_factor = spatial_merge_size**2
    expected_output_patches = total_patches // merge_factor

    assert output.shape[0] == expected_output_patches
    assert output.shape[1] == vision_model.out_hidden_size
