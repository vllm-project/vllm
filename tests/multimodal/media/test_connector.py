# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import mimetypes
import os
import shutil
import time
from io import BytesIO
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any
from unittest.mock import MagicMock

import aiohttp
import numpy as np
import pybase64 as base64
import pytest
import requests
import torch
from PIL import Image, ImageChops

import vllm.envs as envs
from vllm.assets.base import VLLM_S3_BUCKET_URL
from vllm.multimodal.image import convert_image_mode
from vllm.multimodal.inputs import PlaceholderRange
from vllm.multimodal.media import (
    AudioMediaIO,
    ImageMediaIO,
    MediaConnector,
    MediaRef,
    VideoMediaIO,
)
from vllm.multimodal.processing.context import (
    BaseProcessingInfo,
    InputProcessingContext,
)

# Test different image extensions (JPG/PNG) and formats (gray/RGB/RGBA)
TEST_IMAGE_ASSETS = [
    "2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",  # "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    "Grayscale_8bits_palette_sample_image.png",  # "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/Grayscale_8bits_palette_sample_image.png",
    "1280px-Venn_diagram_rgb.svg.png",  # "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/1280px-Venn_diagram_rgb.svg.png",
    "RGBA_comp.png",  # "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/RGBA_comp.png",
]

TEST_VIDEO_URLS = [
    f"{VLLM_S3_BUCKET_URL}/multimodal_asset/slow_traffic_small.mp4",
    f"{VLLM_S3_BUCKET_URL}/multimodal_asset/vtest.avi",
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


def _processing_info() -> BaseProcessingInfo:
    """A `BaseProcessingInfo` with a placeholder model config.

    `get_media_io` reads the model config only to resolve a video backend,
    which the tests below stub out, so nothing else has to be real.
    """
    return BaseProcessingInfo(InputProcessingContext(MagicMock(), None))


def _image_equals(a: Image.Image, b: Image.Image) -> bool:
    return (np.asarray(a) == np.asarray(convert_image_mode(b, a.mode))).all()


def _decode_media(media):
    """Resolve a MediaRef down to the decoded media object."""
    return media.decode() if isinstance(media, MediaRef) else media


@pytest.mark.asyncio
@pytest.mark.parametrize("image_url", TEST_IMAGE_ASSETS, indirect=True)
async def test_fetch_image_http(image_url: str):
    connector = MediaConnector()

    image_sync = connector.fetch_image(image_url, ImageMediaIO())
    image_async = await connector.fetch_image_async(image_url, ImageMediaIO())

    # Fetches return lazy handles; accessing the decoded media decodes them.
    assert isinstance(image_sync, MediaRef)
    assert not image_sync.is_decoded
    assert isinstance(image_async, MediaRef)

    image_sync = _decode_media(image_sync)
    image_async = _decode_media(image_async)
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
            VLLM_S3_BUCKET_URL.removeprefix("https://"),
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

        data_image_sync = _decode_media(connector.fetch_image(data_url, ImageMediaIO()))
        if _image_equals(url_image, Image.open(f)):
            assert _image_equals(url_image, data_image_sync)
        else:
            pass  # Lossy format; only check that image can be opened

        data_image_async = _decode_media(
            await connector.fetch_image_async(data_url, ImageMediaIO())
        )
        assert _image_equals(data_image_sync, data_image_async)


@pytest.mark.asyncio
async def test_fetch_image_keep_original_mode():
    """`image_mode=None` keeps the original mode instead of converting."""
    # RGBA image: opaque black pixel on a fully transparent background
    rgba_image = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    rgba_image.putpixel((2, 2), (0, 0, 0, 255))
    buffer = BytesIO()
    rgba_image.save(buffer, "PNG")
    data_url = (
        f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"
    )

    # Default behavior: RGBA is composited onto a white background
    default_image = _decode_media(
        MediaConnector().fetch_image(data_url, ImageMediaIO())
    )
    assert default_image.mode == "RGB"
    assert default_image.getpixel((0, 0)) == (255, 255, 255)
    assert default_image.getpixel((2, 2)) == (0, 0, 0)

    # image_mode=None: original mode is preserved
    connector = MediaConnector()
    keep_mode = ImageMediaIO(image_mode=None)
    image_sync = _decode_media(connector.fetch_image(data_url, keep_mode))
    image_async = _decode_media(await connector.fetch_image_async(data_url, keep_mode))
    for image in (image_sync, image_async):
        assert image.mode == "RGBA"
        assert image.getpixel((0, 0)) == (0, 0, 0, 0)
        assert image.getpixel((2, 2)) == (0, 0, 0, 255)


@pytest.mark.asyncio
@pytest.mark.parametrize("image_url", TEST_IMAGE_ASSETS, indirect=True)
async def test_fetch_image_local_files(image_url: str):
    connector = MediaConnector()

    with TemporaryDirectory() as temp_dir:
        local_connector = MediaConnector(allowed_local_media_path=temp_dir)

        origin_image = _decode_media(connector.fetch_image(image_url, ImageMediaIO()))
        origin_image.save(
            os.path.join(temp_dir, os.path.basename(image_url)),
            quality=100,
            icc_profile=origin_image.info.get("icc_profile"),
        )

        image_async = _decode_media(
            await local_connector.fetch_image_async(
                f"file://{temp_dir}/{os.path.basename(image_url)}", ImageMediaIO()
            )
        )
        image_sync = _decode_media(
            local_connector.fetch_image(
                f"file://{temp_dir}/{os.path.basename(image_url)}", ImageMediaIO()
            )
        )
        # Check that the images are equal
        assert not ImageChops.difference(image_sync, image_async).getbbox()

        with pytest.raises(ValueError, match="must be a subpath"):
            await local_connector.fetch_image_async(
                f"file://{temp_dir}/../{os.path.basename(image_url)}", ImageMediaIO()
            )
        with pytest.raises(RuntimeError, match="Cannot load local files"):
            await connector.fetch_image_async(
                f"file://{temp_dir}/../{os.path.basename(image_url)}", ImageMediaIO()
            )

        with pytest.raises(ValueError, match="must be a subpath"):
            local_connector.fetch_image(
                f"file://{temp_dir}/../{os.path.basename(image_url)}", ImageMediaIO()
            )
        with pytest.raises(RuntimeError, match="Cannot load local files"):
            connector.fetch_image(
                f"file://{temp_dir}/../{os.path.basename(image_url)}", ImageMediaIO()
            )


@pytest.mark.asyncio
async def test_fetch_image_local_files_relative_allowed_path(tmp_path, monkeypatch):
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    image_path = media_dir / "image.png"
    Image.new("RGB", (1, 1), color=(255, 0, 0)).save(image_path)

    monkeypatch.chdir(tmp_path)
    local_connector = MediaConnector(allowed_local_media_path="media")

    image_sync = _decode_media(
        local_connector.fetch_image(image_path.as_uri(), ImageMediaIO())
    )
    image_async = _decode_media(
        await local_connector.fetch_image_async(image_path.as_uri(), ImageMediaIO())
    )

    assert image_sync.size == (1, 1)
    assert not ImageChops.difference(image_sync, image_async).getbbox()


@pytest.mark.asyncio
@pytest.mark.parametrize("image_url", [TEST_IMAGE_ASSETS[0]], indirect=True)
async def test_fetch_image_local_files_with_space_in_name(image_url: str):
    connector = MediaConnector()

    with TemporaryDirectory() as temp_dir:
        local_connector = MediaConnector(allowed_local_media_path=temp_dir)

        origin_image = _decode_media(connector.fetch_image(image_url, ImageMediaIO()))
        filename = "file name with space.jpg"
        origin_image.save(
            os.path.join(temp_dir, filename),
            quality=100,
            icc_profile=origin_image.info.get("icc_profile"),
        )

        try:
            image_async = _decode_media(
                await local_connector.fetch_image_async(
                    f"file://{temp_dir}/{filename}", ImageMediaIO()
                )
            )
            image_sync = _decode_media(
                local_connector.fetch_image(
                    f"file://{temp_dir}/{filename}", ImageMediaIO()
                )
            )
        except FileNotFoundError as e:
            pytest.fail("Failed to fetch image with space in name: {}".format(e))
        # Check that the images are equal
        assert not ImageChops.difference(image_sync, image_async).getbbox()


@pytest.mark.asyncio
async def test_fetch_image_data_url_with_params():
    """RFC 2397 allows parameters between the mediatype and the base64
    marker; they must not be rejected or leak into the media type."""
    connector = MediaConnector()

    image = Image.new("RGB", (4, 4), color=(255, 0, 0))
    with NamedTemporaryFile(suffix=".png") as f:
        image.save(f.name)
        base64_image = base64.b64encode(f.read()).decode("utf-8")

    data_url = f"data:image/png;charset=utf-8;base64,{base64_image}"
    image_sync = _decode_media(connector.fetch_image(data_url, ImageMediaIO()))
    image_async = _decode_media(
        await connector.fetch_image_async(data_url, ImageMediaIO())
    )
    assert _image_equals(image_sync, image_async)


def test_fetch_image_data_url_malformed():
    connector = MediaConnector()

    with pytest.raises(ValueError, match="missing ','"):
        connector.fetch_image("data:image/png;base64", ImageMediaIO())

    with pytest.raises(NotImplementedError, match="base64"):
        connector.fetch_image("data:text/plain,hello", ImageMediaIO())

    # ";base64" requires the ";"; here "base64" is a (bogus) media type.
    with pytest.raises(NotImplementedError, match="base64"):
        connector.fetch_image("data:base64,aGVsbG8=", ImageMediaIO())

    # Strict RFC 2397 grammar: lowercase "base64", no whitespace.
    with pytest.raises(NotImplementedError, match="base64"):
        connector.fetch_image("data:image/png;BASE64,aGVsbG8=", ImageMediaIO())

    with pytest.raises(NotImplementedError, match="base64"):
        connector.fetch_image("data:image/png; base64,aGVsbG8=", ImageMediaIO())


@pytest.mark.asyncio
async def test_fetch_image_error_conversion():
    connector = MediaConnector()
    broken_img = "data:image/png;base64,aGVsbG9fdmxsbV9jb21tdW5pdHkK"

    # Fetches return lazy handles, so decode errors surface at decode time
    # (inside the multi-modal processor), not at the fetch site.
    image_async = await connector.fetch_image_async(broken_img, ImageMediaIO())
    assert isinstance(image_async, MediaRef)
    with pytest.raises(ValueError, match="Failed to load image"):
        image_async.decode()

    image_sync = connector.fetch_image(broken_img, ImageMediaIO())
    assert isinstance(image_sync, MediaRef)
    with pytest.raises(ValueError, match="Failed to load image"):
        image_sync.decode()


@pytest.mark.flaky(reruns=3, reruns_delay=5)
@pytest.mark.asyncio
@pytest.mark.parametrize("video_url", TEST_VIDEO_URLS)
@pytest.mark.parametrize("num_frames", [-1, 32, 1800])
async def test_fetch_video_http(video_url: str, num_frames: int):
    connector = MediaConnector()
    video_io = VideoMediaIO(ImageMediaIO(), num_frames=num_frames)

    try:
        lazy_sync = connector.fetch_video(video_url, video_io)
        lazy_async = await connector.fetch_video_async(video_url, video_io)
    except (TimeoutError, asyncio.TimeoutError) as e:
        pytest.skip(f"Timeout fetching video (CI network flakiness): {e}")

    # Fetches return lazy handles; unpacking decodes them.
    assert isinstance(lazy_sync, MediaRef)
    assert not lazy_sync.is_decoded
    assert isinstance(lazy_async, MediaRef)

    video_sync, metadata_sync = lazy_sync.decode()
    video_async, metadata_async = lazy_async.decode()

    assert np.array_equal(video_sync, video_async)
    assert metadata_sync == metadata_async


@pytest.mark.flaky(reruns=3, reruns_delay=5)
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
        connector = MediaConnector()
        video_io = VideoMediaIO(
            ImageMediaIO(),
            max_duration=max_duration,
            requested_fps=requested_fps,
        )

        try:
            video_sync, metadata_sync = connector.fetch_video(
                video_url, video_io
            ).decode()
            video_async, metadata_async = (
                await connector.fetch_video_async(video_url, video_io)
            ).decode()
        except (TimeoutError, asyncio.TimeoutError) as e:
            pytest.skip(f"Timeout fetching video (CI network flakiness): {e}")

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


@pytest.mark.flaky(reruns=3, reruns_delay=5)
@pytest.mark.asyncio
@pytest.mark.parametrize("video_url", TEST_VIDEO_URLS)
@pytest.mark.parametrize("num_frames", [-1, 32, 1800])
async def test_allowed_media_domains(video_url: str, num_frames: int):
    connector = MediaConnector(
        allowed_media_domains=[
            VLLM_S3_BUCKET_URL.removeprefix("https://"),
        ],
    )
    video_io = VideoMediaIO(ImageMediaIO(), num_frames=num_frames)

    try:
        video_sync, metadata_sync = connector.fetch_video(video_url, video_io).decode()
        video_async, metadata_async = (
            await connector.fetch_video_async(video_url, video_io)
        ).decode()
    except (TimeoutError, asyncio.TimeoutError) as e:
        pytest.skip(f"Timeout fetching video (CI network flakiness): {e}")

    assert np.array_equal(video_sync, video_async)
    assert metadata_sync == metadata_async

    disallowed_url = "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png"
    with pytest.raises(ValueError):
        _, _ = connector.fetch_video(disallowed_url, video_io)

    with pytest.raises(ValueError):
        _, _ = await connector.fetch_video_async(disallowed_url, video_io)


@pytest.mark.asyncio
async def test_ssrf_bypass_backslash_in_url(local_asset_server):
    """Verify that backslash-@ URL parsing confusion cannot bypass the
    allowed_media_domains check (GHSA-v359-jj2v-j536).

    urllib3.parse_url() and aiohttp/yarl disagree on how to parse a
    backslash before ``@``.  urllib3 treats ``\\`` as part of the path
    (encoding it as ``%5C``), while yarl treats it as a userinfo
    separator, changing the effective host.  The fix normalises the URL
    through urllib3 *before* handing it to aiohttp so both layers agree.
    """
    port = local_asset_server.port
    asset = TEST_IMAGE_ASSETS[0]

    # Craft the bypass payload: urllib3 sees host=127.0.0.1, but an
    # un-patched aiohttp would see host=example.com.
    bypass_url = f"http://127.0.0.1:{port}\\@example.com/{asset}"

    connector = MediaConnector(
        allowed_media_domains=["127.0.0.1"],
    )

    # After the fix the request is made to 127.0.0.1 (the local asset
    # server) using the normalised URL.  The normalised path will be
    # /%5C@example.com/<asset> which won't match any file the server
    # knows about, so we expect an HTTP error — but crucially NOT a
    # successful fetch from example.com.
    with pytest.raises(requests.exceptions.HTTPError):
        connector.fetch_image(bypass_url, ImageMediaIO())

    with pytest.raises(aiohttp.ClientResponseError):
        await connector.fetch_image_async(bypass_url, ImageMediaIO())


@pytest.mark.asyncio
async def test_ssrf_bypass_backslash_disallowed_domain():
    """The reverse direction: even when the *attacker-controlled* host
    appears in the urllib3-parsed hostname position the allowlist must
    still block it.
    """
    # urllib3.parse_url sees host=example.com which is NOT in the
    # allowlist, so this must be rejected before any request is made.
    bypass_url = "https://example.com\\@safe.example.org/image.png"

    connector = MediaConnector(
        allowed_media_domains=["safe.example.org"],
    )

    with pytest.raises(ValueError, match="allowed domains"):
        connector.fetch_image(bypass_url, ImageMediaIO())

    with pytest.raises(ValueError, match="allowed domains"):
        await connector.fetch_image_async(bypass_url, ImageMediaIO())


def test_fetch_video_and_audio_downloads_once():
    """One payload, two decoders: the URL is fetched a single time.

    `use_audio_in_video` reads the audio track out of the video container, so
    both refs must come from one download, capped by the strictest limit any
    of the decoders would have applied on its own.
    """
    calls: list[dict[str, Any]] = []

    class _Connection:
        def get_bytes(self, url: str, **kwargs):
            calls.append({"url": url, **kwargs})
            return b"encoded-container"

    connector = MediaConnector(connection=_Connection())  # type: ignore[arg-type]
    video, audio = connector.fetch_video_and_audio(
        "https://example.com/video.mp4",
        VideoMediaIO(ImageMediaIO()),
        AudioMediaIO(),
    )

    assert [call["url"] for call in calls] == ["https://example.com/video.mp4"]
    assert calls[0]["max_bytes"] == AudioMediaIO().get_max_bytes()

    # Same bytes, different decoders -- and neither has decoded yet.
    assert video.data == audio.data == b"encoded-container"
    assert not video.is_decoded
    assert not audio.is_decoded
    assert video.key != audio.key


def test_info_resolves_the_video_backend_from_the_model(
    monkeypatch: pytest.MonkeyPatch,
):
    """The decoder backend is a model-side decision.

    It is resolved from the model's HF video processor by the processing info,
    not chosen by the fetch layer from a class name threaded through the
    request parser.
    """
    monkeypatch.setattr(
        "vllm.multimodal.processing.context.get_video_processor_cls_name",
        lambda model_config: "Qwen2VLVideoProcessor",
    )
    info = _processing_info()

    assert info.get_media_io("video").video_loader_backend == "qwen2_vl"

    # An explicitly requested backend still wins over the model's default.
    override = {"video": {"video_backend": "opencv"}}
    assert info.get_media_io("video", override).video_loader_backend == "opencv"

    # No binding for the model: fall back to the configured default backend.
    monkeypatch.setattr(
        "vllm.multimodal.processing.context.get_video_processor_cls_name",
        lambda model_config: None,
    )
    assert info.get_media_io("video").video_loader_backend == (
        envs.VLLM_VIDEO_LOADER_BACKEND
    )


def test_fetched_ref_carries_the_spec_info_declares():
    """`info` owns the decoder *and* the spec, so they cannot drift.

    A ref fetched with the decoder `info` resolved must carry exactly the spec
    `info` reports -- that spec is what the item's cache key is derived from.
    """
    image = Image.new("RGB", (4, 4), color=(255, 0, 0))
    buffer = BytesIO()
    image.save(buffer, "PNG")
    data_url = f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"

    info = _processing_info()
    media_io_kwargs = {
        "image": {"image_mode": None, "rgba_background_color": [1, 2, 3]},
    }
    media_io = info.get_media_io("image", media_io_kwargs)

    ref = MediaConnector().fetch_image(data_url, media_io)

    assert ref.spec == media_io.get_decode_spec()


def _make_cached_connector(cache_dir, *, max_mb=10, ttl_hours=24):
    """Create a MediaConnector with caching enabled via monkeypatched internals.

    We bypass __init__'s env-var path and wire up the cache fields directly
    so tests don't depend on environment variables. URLs in these tests are
    only used as cache keys (hashed to derive filenames); no HTTP requests
    are made.
    """
    connector = MediaConnector()
    connector._media_cache_dir = cache_dir
    connector._media_cache_max_bytes = max_mb * 1024 * 1024
    connector._media_cache_ttl_secs = ttl_hours * 3600
    return connector


def test_cache_put_and_get():
    """Basic round-trip: put bytes, get them back."""
    with TemporaryDirectory() as cache_dir:
        connector = _make_cached_connector(cache_dir)
        url = "https://example.com/image.png"
        data = b"fake-image-bytes"

        connector._put_cached_bytes(url, data)
        cached = connector._get_cached_bytes(url)
        assert cached == data


def test_cache_ttl_expiry():
    """Entries older than TTL are evicted on read."""
    with TemporaryDirectory() as cache_dir:
        connector = _make_cached_connector(cache_dir, ttl_hours=24)
        url = "https://example.com/old.png"
        data = b"old-data"

        connector._put_cached_bytes(url, data)

        # Backdate the file's mtime so it appears expired
        cache_path = connector._media_cache_path(url)
        expired_time = time.time() - (25 * 3600)  # 25 hours ago
        os.utime(cache_path, (expired_time, expired_time))

        assert connector._get_cached_bytes(url) is None
        assert not cache_path.exists()


def test_cache_lru_eviction():
    """Oldest entries are evicted when cache exceeds size budget."""
    with TemporaryDirectory() as cache_dir:
        # Set a very small max size: 100 bytes
        connector = _make_cached_connector(cache_dir, max_mb=0)
        connector._media_cache_max_bytes = 100

        # Write three 50-byte entries (total 150 > 100 budget)
        urls = [f"https://example.com/{i}.png" for i in range(3)]
        for i, url in enumerate(urls):
            connector._put_cached_bytes(url, b"x" * 50)
            # Stagger mtime so eviction order is deterministic
            path = connector._media_cache_path(url)
            os.utime(path, (time.time() + i, time.time() + i))

        # The oldest entry (urls[0]) should have been evicted
        assert connector._get_cached_bytes(urls[0]) is None
        # The newest entries should still be present
        assert connector._get_cached_bytes(urls[2]) == b"x" * 50


def test_cache_ttl_eviction_during_write():
    """_maybe_evict removes expired files even if under size budget."""
    with TemporaryDirectory() as cache_dir:
        connector = _make_cached_connector(cache_dir, ttl_hours=1)
        url_old = "https://example.com/stale.png"
        url_new = "https://example.com/fresh.png"

        connector._put_cached_bytes(url_old, b"stale")
        # Backdate old entry past TTL
        old_path = connector._media_cache_path(url_old)
        expired_time = time.time() - (2 * 3600)
        os.utime(old_path, (expired_time, expired_time))

        # Writing a new entry triggers _maybe_evict
        connector._put_cached_bytes(url_new, b"fresh")

        assert not old_path.exists()
        assert connector._get_cached_bytes(url_new) == b"fresh"


def test_put_cached_bytes_missing_dir():
    """_put_cached_bytes does not crash when the cache dir disappears."""
    with TemporaryDirectory() as cache_dir:
        connector = _make_cached_connector(cache_dir)
        # Remove the directory to simulate it disappearing at runtime
        shutil.rmtree(cache_dir)

        # Should not raise (graceful degradation)
        connector._put_cached_bytes("https://example.com/x.png", b"data")


def test_get_cached_bytes_file_deleted_before_read():
    """_get_cached_bytes returns None if the file vanishes mid-read."""
    with TemporaryDirectory() as cache_dir:
        connector = _make_cached_connector(cache_dir)
        url = "https://example.com/vanish.png"

        connector._put_cached_bytes(url, b"data")
        # Delete the file to simulate concurrent eviction
        connector._media_cache_path(url).unlink()

        assert connector._get_cached_bytes(url) is None
