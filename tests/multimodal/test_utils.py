# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64
import math
import mimetypes
import os
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp
from PIL import Image, ImageChops

from tests.utils import multi_gpu_test
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.parallel_state import (init_distributed_environment,
                                             initialize_model_parallel)
from vllm.multimodal.image import convert_image_mode
from vllm.multimodal.inputs import PlaceholderRange
from vllm.multimodal.utils import (MediaConnector, argsort_mm_positions,
                                   get_load_balance_assignment,
                                   run_dp_sharded_mrope_vision_model,
                                   run_dp_sharded_vision_model)
from vllm.platforms import current_platform
from vllm.utils import get_open_port, update_environment_variables

if TYPE_CHECKING:
    from vllm.multimodal.inputs import MultiModalPlaceholderDict

# Test different image extensions (JPG/PNG) and formats (gray/RGB/RGBA)
TEST_IMAGE_ASSETS = [
    "2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",  # "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    "Grayscale_8bits_palette_sample_image.png",  # "https://upload.wikimedia.org/wikipedia/commons/f/fa/Grayscale_8bits_palette_sample_image.png",
    "1280px-Venn_diagram_rgb.svg.png",  # "https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Venn_diagram_rgb.svg/1280px-Venn_diagram_rgb.svg.png",
    "RGBA_comp.png",  # "https://upload.wikimedia.org/wikipedia/commons/0/0b/RGBA_comp.png",
]

TEST_VIDEO_URLS = [
    "https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4",
    "https://github.com/opencv/opencv/raw/refs/tags/4.12.0/samples/data/vtest.avi",
]


@pytest.fixture(scope="module")
def url_images(local_asset_server) -> dict[str, Image.Image]:

    return {
        image_url: local_asset_server.get_image_asset(image_url)
        for image_url in TEST_IMAGE_ASSETS
    }


def get_supported_suffixes() -> tuple[str, ...]:
    # We should at least test the file types mentioned in GPT-4 with Vision
    OPENAI_SUPPORTED_SUFFIXES = ('.png', '.jpeg', '.jpg', '.webp', '.gif')

    # Additional file types that are supported by us
    EXTRA_SUPPORTED_SUFFIXES = ('.bmp', '.tiff')

    return OPENAI_SUPPORTED_SUFFIXES + EXTRA_SUPPORTED_SUFFIXES


def _image_equals(a: Image.Image, b: Image.Image) -> bool:
    return (np.asarray(a) == np.asarray(convert_image_mode(b, a.mode))).all()


@pytest.mark.asyncio
@pytest.mark.parametrize("image_url", TEST_IMAGE_ASSETS, indirect=True)
async def test_fetch_image_http(image_url: str):
    connector = MediaConnector()

    image_sync = connector.fetch_image(image_url)
    image_async = await connector.fetch_image_async(image_url)
    assert _image_equals(image_sync, image_async)


@pytest.mark.asyncio
@pytest.mark.parametrize("raw_image_url", TEST_IMAGE_ASSETS)
@pytest.mark.parametrize("suffix", get_supported_suffixes())
async def test_fetch_image_base64(url_images: dict[str, Image.Image],
                                  raw_image_url: str, suffix: str):
    connector = MediaConnector()
    url_image = url_images[raw_image_url]

    try:
        mime_type = Image.MIME[Image.registered_extensions()[suffix]]
    except KeyError:
        try:
            mime_type = mimetypes.types_map[suffix]
        except KeyError:
            pytest.skip('No MIME type')

    with NamedTemporaryFile(suffix=suffix) as f:
        try:
            url_image.save(f.name)
        except Exception as e:
            if e.args[0] == 'cannot write mode RGBA as JPEG':
                pytest.skip('Conversion not supported')

            raise

        base64_image = base64.b64encode(f.read()).decode("utf-8")
        data_url = f"data:{mime_type};base64,{base64_image}"

        data_image_sync = connector.fetch_image(data_url)
        if _image_equals(url_image, Image.open(f)):
            assert _image_equals(url_image, data_image_sync)
        else:
            pass  # Lossy format; only check that image can be opened

        data_image_async = await connector.fetch_image_async(data_url)
        assert _image_equals(data_image_sync, data_image_async)


@pytest.mark.asyncio
@pytest.mark.parametrize("image_url", TEST_IMAGE_ASSETS, indirect=True)
async def test_fetch_image_local_files(image_url: str):
    connector = MediaConnector()

    with TemporaryDirectory() as temp_dir:
        local_connector = MediaConnector(allowed_local_media_path=temp_dir)

        origin_image = connector.fetch_image(image_url)
        origin_image.save(os.path.join(temp_dir, os.path.basename(image_url)),
                          quality=100,
                          icc_profile=origin_image.info.get('icc_profile'))

        image_async = await local_connector.fetch_image_async(
            f"file://{temp_dir}/{os.path.basename(image_url)}")
        image_sync = local_connector.fetch_image(
            f"file://{temp_dir}/{os.path.basename(image_url)}")
        # Check that the images are equal
        assert not ImageChops.difference(image_sync, image_async).getbbox()

        with pytest.raises(ValueError, match="must be a subpath"):
            await local_connector.fetch_image_async(
                f"file://{temp_dir}/../{os.path.basename(image_url)}")
        with pytest.raises(RuntimeError, match="Cannot load local files"):
            await connector.fetch_image_async(
                f"file://{temp_dir}/../{os.path.basename(image_url)}")

        with pytest.raises(ValueError, match="must be a subpath"):
            local_connector.fetch_image(
                f"file://{temp_dir}/../{os.path.basename(image_url)}")
        with pytest.raises(RuntimeError, match="Cannot load local files"):
            connector.fetch_image(
                f"file://{temp_dir}/../{os.path.basename(image_url)}")


@pytest.mark.asyncio
@pytest.mark.parametrize("image_url", [TEST_IMAGE_ASSETS[0]], indirect=True)
async def test_fetch_image_local_files_with_space_in_name(image_url: str):
    connector = MediaConnector()

    with TemporaryDirectory() as temp_dir:
        local_connector = MediaConnector(allowed_local_media_path=temp_dir)

        origin_image = connector.fetch_image(image_url)
        filename = "file name with space.jpg"
        origin_image.save(os.path.join(temp_dir, filename),
                          quality=100,
                          icc_profile=origin_image.info.get('icc_profile'))

        try:
            image_async = await local_connector.fetch_image_async(
                f"file://{temp_dir}/{filename}")
            image_sync = local_connector.fetch_image(
                f"file://{temp_dir}/{filename}")
        except FileNotFoundError as e:
            pytest.fail(
                "Failed to fetch image with space in name: {}".format(e))
        # Check that the images are equal
        assert not ImageChops.difference(image_sync, image_async).getbbox()


@pytest.mark.asyncio
async def test_fetch_image_error_conversion():
    connector = MediaConnector()
    broken_img = "data:image/png;base64,aGVsbG9fdmxsbV9jb21tdW5pdHkK"

    # PIL.UnidentifiedImageError should be converted to ValueError
    with pytest.raises(ValueError):
        await connector.fetch_image_async(broken_img)

    with pytest.raises(ValueError):
        connector.fetch_image(broken_img)


@pytest.mark.asyncio
@pytest.mark.parametrize("video_url", TEST_VIDEO_URLS)
@pytest.mark.parametrize("num_frames", [-1, 32, 1800])
async def test_fetch_video_http(video_url: str, num_frames: int):
    connector = MediaConnector(
        media_io_kwargs={"video": {
            "num_frames": num_frames,
        }})

    video_sync, metadata_sync = connector.fetch_video(video_url)
    video_async, metadata_async = await connector.fetch_video_async(video_url)
    assert np.array_equal(video_sync, video_async)
    assert metadata_sync == metadata_async


@pytest.mark.asyncio
@pytest.mark.parametrize("video_url", TEST_VIDEO_URLS)
@pytest.mark.parametrize("max_duration", [1, 60, 1800])
@pytest.mark.parametrize("requested_fps", [2, 24])
async def test_fetch_video_http_with_dynamic_loader(
        video_url: str, max_duration: int, requested_fps: int,
        monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        m.setenv("VLLM_VIDEO_LOADER_BACKEND", "opencv_dynamic")
        connector = MediaConnector(
            media_io_kwargs={
                "video": {
                    "max_duration": max_duration,
                    "requested_fps": requested_fps,
                }
            })

        video_sync, metadata_sync = connector.fetch_video(video_url)
        video_async, metadata_async = await connector.fetch_video_async(
            video_url)

        assert np.array_equal(video_sync, video_async)
        assert metadata_sync == metadata_async
        assert metadata_sync["video_backend"] == "opencv_dynamic"


# Used for `test_argsort_mm_positions`.
class TestCase(NamedTuple):
    mm_positions: "MultiModalPlaceholderDict"
    expected_modality_idxs: list[tuple[str, int]]


def test_argsort_mm_positions():

    test_cases = [
        # Single modality
        ## Internally sorted
        TestCase(
            mm_positions={
                "image": [
                    PlaceholderRange(offset=0, length=2),
                    PlaceholderRange(offset=3, length=2),
                ]
            },
            expected_modality_idxs=[
                ("image", 0),
                ("image", 1),
            ],
        ),
        ## Internally unsorted
        TestCase(
            mm_positions={
                "image": [
                    PlaceholderRange(offset=3, length=2),
                    PlaceholderRange(offset=0, length=2),
                ]
            },
            expected_modality_idxs=[
                ("image", 1),
                ("image", 0),
            ],
        ),

        # Two modalities
        ## Internally sorted
        TestCase(
            mm_positions={
                "image": [
                    PlaceholderRange(offset=7, length=4),
                    PlaceholderRange(offset=11, length=5),
                ],
                "audio": [
                    PlaceholderRange(offset=0, length=2),
                    PlaceholderRange(offset=2, length=3),
                ]
            },
            expected_modality_idxs=[
                ("audio", 0),
                ("audio", 1),
                ("image", 0),
                ("image", 1),
            ],
        ),
        ## Interleaved, internally sorted
        TestCase(
            mm_positions={
                "image": [
                    PlaceholderRange(offset=0, length=4),
                    PlaceholderRange(offset=8, length=2),
                ],
                "audio": [
                    PlaceholderRange(offset=5, length=2),
                    PlaceholderRange(offset=11, length=4),
                ]
            },
            expected_modality_idxs=[
                ("image", 0),
                ("audio", 0),
                ("image", 1),
                ("audio", 1),
            ],
        ),
        ## Interleaved, internally unsorted
        TestCase(
            mm_positions={
                "image": [
                    PlaceholderRange(offset=8, length=2),
                    PlaceholderRange(offset=0, length=4),
                ],
                "audio": [
                    PlaceholderRange(offset=11, length=4),
                    PlaceholderRange(offset=5, length=2),
                ]
            },
            expected_modality_idxs=[
                ("image", 1),
                ("audio", 1),
                ("image", 0),
                ("audio", 0),
            ],
        ),

        # Three modalities
        ## Internally sorted
        TestCase(
            mm_positions={
                "image": [
                    PlaceholderRange(offset=15, length=7),
                    PlaceholderRange(offset=22, length=8),
                ],
                "audio": [
                    PlaceholderRange(offset=0, length=2),
                ],
                "video": [
                    PlaceholderRange(offset=3, length=4),
                    PlaceholderRange(offset=7, length=5),
                    PlaceholderRange(offset=12, length=6),
                ]
            },
            expected_modality_idxs=[
                ("audio", 0),
                ("video", 0),
                ("video", 1),
                ("video", 2),
                ("image", 0),
                ("image", 1),
            ],
        ),
        ## Interleaved, internally sorted
        TestCase(
            mm_positions={
                "image": [
                    PlaceholderRange(offset=0, length=2),
                    PlaceholderRange(offset=2, length=3),
                    PlaceholderRange(offset=20, length=4),
                ],
                "audio": [
                    PlaceholderRange(offset=5, length=2),
                ],
                "video": [
                    PlaceholderRange(offset=8, length=5),
                ]
            },
            expected_modality_idxs=[
                ("image", 0),
                ("image", 1),
                ("audio", 0),
                ("video", 0),
                ("image", 2),
            ],
        ),
        ## Interleaved, internally sunorted
        TestCase(
            mm_positions={
                "image": [
                    PlaceholderRange(offset=0, length=2),
                    PlaceholderRange(offset=20, length=4),
                    PlaceholderRange(offset=2, length=3),
                ],
                "audio": [
                    PlaceholderRange(offset=5, length=2),
                ],
                "video": [
                    PlaceholderRange(offset=8, length=5),
                ]
            },
            expected_modality_idxs=[
                ("image", 0),
                ("image", 2),
                ("audio", 0),
                ("video", 0),
                ("image", 1),
            ],
        ),
    ]

    for mm_positions, expected_modality_idxs in test_cases:
        modality_idxs = argsort_mm_positions(mm_positions)

        assert modality_idxs == expected_modality_idxs


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


def run_dp_sharded_vision_model_vs_direct(local_rank: int, world_size: int,
                                          batch_size: int, master_port: int):
    """
    Test that run_dp_sharded_vision_model produces the same results as 
    calling the model directly.
    """

    # Set random seed for reproducibility
    current_platform.seed_everything(0)

    device = f"{current_platform.device_name}:{local_rank}"
    current_platform.set_device(device)
    torch.set_default_device(device)

    update_environment_variables({
        'RANK': str(local_rank),
        'LOCAL_RANK': str(local_rank),
        'WORLD_SIZE': str(world_size),
        'MASTER_ADDR': 'localhost',
        'MASTER_PORT': str(master_port),
    })

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
        ([100, 200], 4, [1, 0], [1, 1, 0, 0], [200, 100, 0, 0
                                               ], "fewer samples than GPUs"),

        # Single GPU
        ([100, 200, 300], 1, [2, 1, 0], [3], [600], "single GPU"),

        # Balanced assignment
        ([100, 100, 100, 100
          ], 2, [0, 2, 1, 3], [2, 2], [200, 200], "balanced assignment"),

        # Unbalanced sizes - this one is trickier since the algorithm is greedy
        ([1000, 100, 200, 50], 2, [0, 2, 1, 3
                                   ], [1, 3], [1000, 350], "unbalanced sizes"),
    ],
)
def test_get_load_balance_assignment_cases(sizes, num_gpus,
                                           expected_shuffle_indices,
                                           expected_gpu_sample_counts,
                                           expected_grouped_sizes_per_gpu,
                                           test_description):
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

    def forward(self, pixel_values: torch.Tensor,
                grid_thw_list: list[list[int]]):
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
                reshaped = image_patches[:merged_patches * merge_factor].view(
                    merged_patches, merge_factor, -1)
                merged = reshaped.mean(dim=1)
                merged_embeddings.append(merged)

            start_idx = end_idx

        if merged_embeddings:
            return torch.cat(merged_embeddings, dim=0)
        else:
            return torch.empty((0, self.out_hidden_size),
                               device=pixel_values.device,
                               dtype=pixel_values.dtype)


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


def run_dp_sharded_mrope_vision_model_vs_direct(local_rank: int,
                                                world_size: int,
                                                batch_size: int,
                                                master_port: int):
    """
    Test that run_dp_sharded_mrope_vision_model produces the same results as 
    calling the model directly.
    """
    # Set random seed for reproducibility
    current_platform.seed_everything(0)
    device = f"{current_platform.device_name}:{local_rank}"
    current_platform.set_device(device)
    torch.set_default_device(device)

    update_environment_variables({
        'RANK': str(local_rank),
        'LOCAL_RANK': str(local_rank),
        'WORLD_SIZE': str(world_size),
        'MASTER_ADDR': 'localhost',
        'MASTER_PORT': str(master_port),
    })

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
        sharded_output = run_dp_sharded_mrope_vision_model(vision_model,
                                                           pixel_values,
                                                           grid_thw_list,
                                                           rope_type="rope_3d")
        sharded_output = torch.cat(sharded_output, dim=0)

    # Check that the world size is set up correctly
    assert get_tensor_model_parallel_world_size() == world_size

    # Compare outputs (only on rank 0)
    if local_rank == 0:
        # Check that the outputs have the same shape
        assert direct_output.shape == sharded_output.shape
        # Check that the outputs are close (they should be identical)
        assert torch.allclose(direct_output,
                              sharded_output,
                              rtol=1e-5,
                              atol=1e-5)


@multi_gpu_test(num_gpus=2)
def test_run_dp_sharded_mrope_vision_model_empty_input():
    world_size = 2
    mp.spawn(
        run_dp_sharded_mrope_vision_model_empty_input_worker,
        args=(world_size, get_open_port()),
        nprocs=world_size,
    )


def run_dp_sharded_mrope_vision_model_empty_input_worker(
        local_rank: int, world_size: int, master_port: int):
    """Test run_dp_sharded_mrope_vision_model with empty input."""
    # Set up distributed environment
    device = f"{current_platform.device_name}:{local_rank}"
    current_platform.set_device(device)
    torch.set_default_device(device)

    update_environment_variables({
        'RANK': str(local_rank),
        'LOCAL_RANK': str(local_rank),
        'WORLD_SIZE': str(world_size),
        'MASTER_ADDR': 'localhost',
        'MASTER_PORT': str(master_port),
    })

    init_distributed_environment()
    initialize_model_parallel(tensor_model_parallel_size=world_size)

    # Create empty inputs
    pixel_values = torch.empty((0, 768))
    grid_thw_list: list[list[int]] = []

    vision_model = SimpleMRopeVisionModel()

    # Should handle empty input gracefully
    with torch.inference_mode():
        output = run_dp_sharded_mrope_vision_model(vision_model,
                                                   pixel_values,
                                                   grid_thw_list,
                                                   rope_type="rope_3d")

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
        local_rank: int, world_size: int, master_port: int):
    """Test run_dp_sharded_mrope_vision_model with uneven load distribution."""
    # Set up distributed environment
    current_platform.seed_everything(123)
    device = f"{current_platform.device_name}:{local_rank}"
    current_platform.set_device(device)
    torch.set_default_device(device)

    update_environment_variables({
        'RANK': str(local_rank),
        'LOCAL_RANK': str(local_rank),
        'WORLD_SIZE': str(world_size),
        'MASTER_ADDR': 'localhost',
        'MASTER_PORT': str(master_port),
    })

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
        output_tuple = run_dp_sharded_mrope_vision_model(vision_model,
                                                         pixel_values,
                                                         grid_thw_list,
                                                         rope_type="rope_3d")

    # Verify output shape is reasonable
    merge_factor = vision_model.spatial_merge_size**2
    expected_output_patches = list(
        math.prod(grid_thw) // merge_factor for grid_thw in grid_thw_list)

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
    vision_model = SimpleMRopeVisionModel(
        spatial_merge_size=spatial_merge_size).to(device)

    with torch.inference_mode():
        output = vision_model(pixel_values, grid_thw_list)

    # Verify output dimensions based on spatial merging
    total_patches = sum(math.prod(grid_thw) for grid_thw in grid_thw_list)
    merge_factor = spatial_merge_size**2
    expected_output_patches = total_patches // merge_factor

    assert output.shape[0] == expected_output_patches
    assert output.shape[1] == vision_model.out_hidden_size
