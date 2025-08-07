# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64
import mimetypes
import os
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import TYPE_CHECKING, NamedTuple, Optional

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
from vllm.multimodal.utils import (MediaConnector,
                                   merge_and_sort_multimodal_metadata,
                                   run_dp_sharded_vision_model)
from vllm.platforms import current_platform
from vllm.utils import get_open_port, update_environment_variables

if TYPE_CHECKING:
    from vllm.multimodal.hasher import MultiModalHashDict
    from vllm.multimodal.inputs import MultiModalPlaceholderDict

# Test different image extensions (JPG/PNG) and formats (gray/RGB/RGBA)
TEST_IMAGE_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/f/fa/Grayscale_8bits_palette_sample_image.png",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Venn_diagram_rgb.svg/1280px-Venn_diagram_rgb.svg.png",
    "https://upload.wikimedia.org/wikipedia/commons/0/0b/RGBA_comp.png",
]

TEST_VIDEO_URLS = [
    "https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4",
    "https://github.com/opencv/opencv/raw/refs/tags/4.12.0/samples/data/vtest.avi",
]


@pytest.fixture(scope="module")
def url_images() -> dict[str, Image.Image]:
    connector = MediaConnector()

    return {
        image_url: connector.fetch_image(image_url)
        for image_url in TEST_IMAGE_URLS
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
@pytest.mark.parametrize("image_url", TEST_IMAGE_URLS)
async def test_fetch_image_http(image_url: str):
    connector = MediaConnector()

    image_sync = connector.fetch_image(image_url)
    image_async = await connector.fetch_image_async(image_url)
    assert _image_equals(image_sync, image_async)


@pytest.mark.asyncio
@pytest.mark.parametrize("image_url", TEST_IMAGE_URLS)
@pytest.mark.parametrize("suffix", get_supported_suffixes())
async def test_fetch_image_base64(url_images: dict[str, Image.Image],
                                  image_url: str, suffix: str):
    connector = MediaConnector()
    url_image = url_images[image_url]

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
@pytest.mark.parametrize("image_url", TEST_IMAGE_URLS)
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


# Used for the next two tests related to `merge_and_sort_multimodal_metadata`.
class TestCase(NamedTuple):
    mm_positions: "MultiModalPlaceholderDict"
    mm_hashes: Optional["MultiModalHashDict"]
    expected_modalities: list[str]
    expected_ranges: list[PlaceholderRange]
    expected_hashes: Optional[list[str]]


def test_merge_and_sort_multimodal_metadata():

    test_cases = [
        # Single modality should return result as is but flattened
        TestCase(
            mm_positions={
                "image": [
                    PlaceholderRange(offset=0, length=2),
                    PlaceholderRange(offset=3, length=2),
                ]
            },
            mm_hashes={"image": ["hash1", "hash2"]},
            expected_modalities=["image", "image"],
            expected_ranges=[
                PlaceholderRange(offset=0, length=2),
                PlaceholderRange(offset=3, length=2),
            ],
            expected_hashes=["hash1", "hash2"],
        ),

        # Single modality without hashes return None for mm hash.
        TestCase(
            mm_positions={
                "image": [
                    PlaceholderRange(offset=0, length=2),
                    PlaceholderRange(offset=2, length=2),
                ]
            },
            mm_hashes=None,
            expected_modalities=["image", "image"],
            expected_ranges=[
                PlaceholderRange(offset=0, length=2),
                PlaceholderRange(offset=2, length=2),
            ],
            expected_hashes=None,
        ),

        # Multiple modalities with hashes should return sorted modalities
        # and flattened ranges and hashes.
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
            mm_hashes={
                "image": ["image_hash1", "image_hash2"],
                "audio": ["audio_hash1", "audio_hash2"],
            },
            expected_modalities=["audio", "audio", "image", "image"],
            expected_ranges=[
                PlaceholderRange(offset=0, length=2),
                PlaceholderRange(offset=2, length=3),
                PlaceholderRange(offset=7, length=4),
                PlaceholderRange(offset=11, length=5),
            ],
            expected_hashes=[
                "audio_hash1", "audio_hash2", "image_hash1", "image_hash2"
            ],
        ),

        # Multiple modalities without hashes should return sorted modalities
        # and flattened ranges and None.
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
            mm_hashes=None,
            expected_modalities=["audio", "audio", "image", "image"],
            expected_ranges=[
                PlaceholderRange(offset=0, length=2),
                PlaceholderRange(offset=2, length=3),
                PlaceholderRange(offset=7, length=4),
                PlaceholderRange(offset=11, length=5),
            ],
            expected_hashes=None,
        ),

        # Three modalities
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
            mm_hashes={
                "image": ["image_hash1", "image_hash2"],
                "audio": ["audio_hash1"],
                "video": ["video_hash1", "video_hash2", "video_hash3"]
            },
            expected_modalities=[
                "audio", "video", "video", "video", "image", "image"
            ],
            expected_ranges=[
                PlaceholderRange(offset=0, length=2),
                PlaceholderRange(offset=3, length=4),
                PlaceholderRange(offset=7, length=5),
                PlaceholderRange(offset=12, length=6),
                PlaceholderRange(offset=15, length=7),
                PlaceholderRange(offset=22, length=8),
            ],
            expected_hashes=[
                "audio_hash1", "video_hash1", "video_hash2", "video_hash3",
                "image_hash1", "image_hash2"
            ],
        ),
    ]

    for (mm_positions, mm_hashes, expected_modalities, expected_ranges,
         expected_hashes) in test_cases:
        modalities, ranges, hashes = merge_and_sort_multimodal_metadata(
            mm_positions, mm_hashes)

        assert modalities == expected_modalities
        assert ranges == expected_ranges
        assert hashes == expected_hashes


def test_merge_and_sort_multimodal_metadata_with_interleaving():

    test_cases = [

        # <image> <audio> <image> <audio>
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
            mm_hashes={
                "image": ["image_hash1", "image_hash2"],
                "audio": ["audio_hash1", "audio_hash2"],
            },
            expected_modalities=["image", "audio", "image", "audio"],
            expected_ranges=[
                PlaceholderRange(offset=0, length=4),
                PlaceholderRange(offset=5, length=2),
                PlaceholderRange(offset=8, length=2),
                PlaceholderRange(offset=11, length=4),
            ],
            expected_hashes=[
                "image_hash1", "audio_hash1", "image_hash2", "audio_hash2"
            ],
        ),

        # <image> <image> <audio> <video> <image>
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
            mm_hashes=None,
            expected_modalities=["image", "image", "audio", "video", "image"],
            expected_ranges=[
                PlaceholderRange(offset=0, length=2),
                PlaceholderRange(offset=2, length=3),
                PlaceholderRange(offset=5, length=2),
                PlaceholderRange(offset=8, length=5),
                PlaceholderRange(offset=20, length=4),
            ],
            expected_hashes=None,
        ),

        # <image> <audio> <video> <image> with hashes
        TestCase(
            mm_positions={
                "image": [
                    PlaceholderRange(offset=0, length=2),
                    PlaceholderRange(offset=18, length=4),
                ],
                "audio": [
                    PlaceholderRange(offset=6, length=2),
                ],
                "video": [
                    PlaceholderRange(offset=10, length=5),
                ]
            },
            mm_hashes={
                "image": ["image_hash1", "image_hash2"],
                "audio": ["audio_hash1"],
                "video": ["video_hash1"],
            },
            expected_modalities=["image", "audio", "video", "image"],
            expected_ranges=[
                PlaceholderRange(offset=0, length=2),
                PlaceholderRange(offset=6, length=2),
                PlaceholderRange(offset=10, length=5),
                PlaceholderRange(offset=18, length=4),
            ],
            expected_hashes=[
                "image_hash1", "audio_hash1", "video_hash1", "image_hash2"
            ],
        ),
    ]

    for (mm_positions, mm_hashes, expected_modalities, expected_ranges,
         expected_hashes) in test_cases:
        modalities, ranges, hashes = merge_and_sort_multimodal_metadata(
            mm_positions, mm_hashes)

        assert modalities == expected_modalities
        assert ranges == expected_ranges
        assert hashes == expected_hashes


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

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
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

    # Check that the world size is setup correctly
    assert get_tensor_model_parallel_world_size() == world_size

    # Check that the outputs have the same shape
    assert direct_output.shape == sharded_output.shape

    # Check that the outputs are close (they should be identical)
    assert torch.allclose(direct_output, sharded_output, rtol=1e-5, atol=1e-5)
