# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import base64
import mimetypes
import os
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pytest
import torch
from PIL import Image, ImageChops

from vllm.multimodal.image import convert_image_mode
from vllm.multimodal.inputs import PlaceholderRange
from vllm.multimodal.media import MediaConnector

# Test different image extensions (JPG/PNG) and formats (gray/RGB/RGBA)
TEST_IMAGE_ASSETS = [
    "2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",  # "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    "Grayscale_8bits_palette_sample_image.png",  # "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/Grayscale_8bits_palette_sample_image.png",
    "1280px-Venn_diagram_rgb.svg.png",  # "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/1280px-Venn_diagram_rgb.svg.png",
    "RGBA_comp.png",  # "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/RGBA_comp.png",
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
    OPENAI_SUPPORTED_SUFFIXES = (".png", ".jpeg", ".jpg", ".webp", ".gif")

    # Additional file types that are supported by us
    EXTRA_SUPPORTED_SUFFIXES = (".bmp", ".tiff")

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
async def test_fetch_image_base64(
    url_images: dict[str, Image.Image], raw_image_url: str, suffix: str
):
    connector = MediaConnector(
        # Domain restriction should not apply to data URLs.
        allowed_media_domains=[
            "www.bogotobogo.com",
            "github.com",
        ]
    )
    url_image = url_images[raw_image_url]

    try:
        mime_type = Image.MIME[Image.registered_extensions()[suffix]]
    except KeyError:
        try:
            mime_type = mimetypes.types_map[suffix]
        except KeyError:
            pytest.skip("No MIME type")

    with NamedTemporaryFile(suffix=suffix) as f:
        try:
            url_image.save(f.name)
        except Exception as e:
            if e.args[0] == "cannot write mode RGBA as JPEG":
                pytest.skip("Conversion not supported")

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
        origin_image.save(
            os.path.join(temp_dir, os.path.basename(image_url)),
            quality=100,
            icc_profile=origin_image.info.get("icc_profile"),
        )

        image_async = await local_connector.fetch_image_async(
            f"file://{temp_dir}/{os.path.basename(image_url)}"
        )
        image_sync = local_connector.fetch_image(
            f"file://{temp_dir}/{os.path.basename(image_url)}"
        )
        # Check that the images are equal
        assert not ImageChops.difference(image_sync, image_async).getbbox()

        with pytest.raises(ValueError, match="must be a subpath"):
            await local_connector.fetch_image_async(
                f"file://{temp_dir}/../{os.path.basename(image_url)}"
            )
        with pytest.raises(RuntimeError, match="Cannot load local files"):
            await connector.fetch_image_async(
                f"file://{temp_dir}/../{os.path.basename(image_url)}"
            )

        with pytest.raises(ValueError, match="must be a subpath"):
            local_connector.fetch_image(
                f"file://{temp_dir}/../{os.path.basename(image_url)}"
            )
        with pytest.raises(RuntimeError, match="Cannot load local files"):
            connector.fetch_image(f"file://{temp_dir}/../{os.path.basename(image_url)}")


@pytest.mark.asyncio
@pytest.mark.parametrize("image_url", [TEST_IMAGE_ASSETS[0]], indirect=True)
async def test_fetch_image_local_files_with_space_in_name(image_url: str):
    connector = MediaConnector()

    with TemporaryDirectory() as temp_dir:
        local_connector = MediaConnector(allowed_local_media_path=temp_dir)

        origin_image = connector.fetch_image(image_url)
        filename = "file name with space.jpg"
        origin_image.save(
            os.path.join(temp_dir, filename),
            quality=100,
            icc_profile=origin_image.info.get("icc_profile"),
        )

        try:
            image_async = await local_connector.fetch_image_async(
                f"file://{temp_dir}/{filename}"
            )
            image_sync = local_connector.fetch_image(f"file://{temp_dir}/{filename}")
        except FileNotFoundError as e:
            pytest.fail("Failed to fetch image with space in name: {}".format(e))
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


@pytest.mark.flaky(reruns=3, reruns_delay=5)
@pytest.mark.asyncio
@pytest.mark.parametrize("video_url", TEST_VIDEO_URLS)
@pytest.mark.parametrize("num_frames", [-1, 32, 1800])
async def test_fetch_video_http(video_url: str, num_frames: int):
    connector = MediaConnector(
        media_io_kwargs={
            "video": {
                "num_frames": num_frames,
            }
        }
    )

    try:
        video_sync, metadata_sync = connector.fetch_video(video_url)
        video_async, metadata_async = await connector.fetch_video_async(video_url)
    except (TimeoutError, asyncio.TimeoutError) as e:
        pytest.skip(f"Timeout fetching video (CI network flakiness): {e}")

    assert np.array_equal(video_sync, video_async)
    assert metadata_sync == metadata_async


@pytest.mark.asyncio
@pytest.mark.parametrize("video_url", TEST_VIDEO_URLS)
@pytest.mark.parametrize("max_duration", [1, 60, 1800])
@pytest.mark.parametrize("requested_fps", [2, 24])
async def test_fetch_video_http_with_dynamic_loader(
    video_url: str,
    max_duration: int,
    requested_fps: int,
    monkeypatch: pytest.MonkeyPatch,
):
    with monkeypatch.context() as m:
        m.setenv("VLLM_VIDEO_LOADER_BACKEND", "opencv_dynamic")
        connector = MediaConnector(
            media_io_kwargs={
                "video": {
                    "max_duration": max_duration,
                    "requested_fps": requested_fps,
                }
            }
        )

        video_sync, metadata_sync = connector.fetch_video(video_url)
        video_async, metadata_async = await connector.fetch_video_async(video_url)

        assert np.array_equal(video_sync, video_async)
        assert metadata_sync == metadata_async
        assert metadata_sync["video_backend"] == "opencv_dynamic"


@pytest.mark.parametrize(
    "is_embed,start_idx,end_idx,expected",
    [
        (None, 2, 4, (2, 4)),
        (
            torch.tensor([False, True, False, True, True]),
            3,
            5,
            (1, 3),
        ),
        (
            torch.tensor([False, True, False, True, True]),
            0,
            2,
            (0, 1),
        ),
        (
            torch.tensor([True, False, True, False]),
            2,
            2,
            (1, 1),
        ),
    ],
)
def test_placeholder_range_get_embeds_indices_in_range(
    is_embed, start_idx, end_idx, expected
):
    length = len(is_embed) if is_embed is not None else 5
    pr = PlaceholderRange(offset=0, length=length, is_embed=is_embed)
    assert pr.get_embeds_indices_in_range(start_idx, end_idx) == expected


@pytest.mark.parametrize(
    "offset,is_embed,expected",
    [
        (0, None, [(0, 4)]),
        (
            2,
            torch.tensor([False, True, False, True, True]),
            [(3, 3), (5, 6)],
        ),
        (0, torch.tensor([True, True, True, True]), [(0, 3)]),
        (0, torch.tensor([False, False, False, False]), []),
    ],
)
def test_placeholder_range_extract_embeds_range(offset, is_embed, expected):
    length = len(is_embed) if is_embed is not None else 5
    pr = PlaceholderRange(offset=offset, length=length, is_embed=is_embed)
    assert pr.extract_embeds_range() == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("video_url", TEST_VIDEO_URLS)
@pytest.mark.parametrize("num_frames", [-1, 32, 1800])
async def test_allowed_media_domains(video_url: str, num_frames: int):
    connector = MediaConnector(
        media_io_kwargs={
            "video": {
                "num_frames": num_frames,
            }
        },
        allowed_media_domains=[
            "www.bogotobogo.com",
            "github.com",
        ],
    )

    video_sync, metadata_sync = connector.fetch_video(video_url)
    video_async, metadata_async = await connector.fetch_video_async(video_url)
    assert np.array_equal(video_sync, video_async)
    assert metadata_sync == metadata_async

    disallowed_url = "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png"
    with pytest.raises(ValueError):
        _, _ = connector.fetch_video(disallowed_url)

    with pytest.raises(ValueError):
        _, _ = await connector.fetch_video_async(disallowed_url)
