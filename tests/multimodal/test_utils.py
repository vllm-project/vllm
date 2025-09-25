# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64
import mimetypes
import os
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pytest
from PIL import Image, ImageChops

from vllm.multimodal.image import convert_image_mode
from vllm.multimodal.inputs import PlaceholderRange
from vllm.multimodal.utils import MediaConnector, argsort_mm_positions

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
