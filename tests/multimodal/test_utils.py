import base64
import mimetypes
import os
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Dict, Tuple

import numpy as np
import pytest
from PIL import Image, ImageChops
from transformers import AutoConfig, AutoTokenizer

from vllm.multimodal.utils import (async_fetch_image, fetch_image,
                                   repeat_and_pad_placeholder_tokens)

# Test different image extensions (JPG/PNG) and formats (gray/RGB/RGBA)
TEST_IMAGE_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/f/fa/Grayscale_8bits_palette_sample_image.png",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Venn_diagram_rgb.svg/1280px-Venn_diagram_rgb.svg.png",
    "https://upload.wikimedia.org/wikipedia/commons/0/0b/RGBA_comp.png",
]


@pytest.fixture(scope="module")
def url_images() -> Dict[str, Image.Image]:
    return {image_url: fetch_image(image_url) for image_url in TEST_IMAGE_URLS}


def get_supported_suffixes() -> Tuple[str, ...]:
    # We should at least test the file types mentioned in GPT-4 with Vision
    OPENAI_SUPPORTED_SUFFIXES = ('.png', '.jpeg', '.jpg', '.webp', '.gif')

    # Additional file types that are supported by us
    EXTRA_SUPPORTED_SUFFIXES = ('.bmp', '.tiff')

    return OPENAI_SUPPORTED_SUFFIXES + EXTRA_SUPPORTED_SUFFIXES


def _image_equals(a: Image.Image, b: Image.Image) -> bool:
    return (np.asarray(a) == np.asarray(b.convert(a.mode))).all()


@pytest.mark.asyncio
@pytest.mark.parametrize("image_url", TEST_IMAGE_URLS)
async def test_fetch_image_http(image_url: str):
    image_sync = fetch_image(image_url)
    image_async = await async_fetch_image(image_url)
    assert _image_equals(image_sync, image_async)


@pytest.mark.asyncio
@pytest.mark.parametrize("image_url", TEST_IMAGE_URLS)
@pytest.mark.parametrize("suffix", get_supported_suffixes())
async def test_fetch_image_base64(url_images: Dict[str, Image.Image],
                                  image_url: str, suffix: str):
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

        data_image_sync = fetch_image(data_url)
        if _image_equals(url_image, Image.open(f)):
            assert _image_equals(url_image, data_image_sync)
        else:
            pass  # Lossy format; only check that image can be opened

        data_image_async = await async_fetch_image(data_url)
        assert _image_equals(data_image_sync, data_image_async)


@pytest.mark.asyncio
@pytest.mark.parametrize("image_url", TEST_IMAGE_URLS)
async def test_fetch_image_local_files(image_url: str):
    with TemporaryDirectory() as temp_dir:
        origin_image = fetch_image(image_url)
        origin_image.save(os.path.join(temp_dir, os.path.basename(image_url)),
                          quality=100,
                          icc_profile=origin_image.info.get('icc_profile'))

        image_async = await async_fetch_image(
            f"file://{temp_dir}/{os.path.basename(image_url)}",
            allowed_local_media_path=temp_dir)

        image_sync = fetch_image(
            f"file://{temp_dir}/{os.path.basename(image_url)}",
            allowed_local_media_path=temp_dir)
        # Check that the images are equal
        assert not ImageChops.difference(image_sync, image_async).getbbox()

        with pytest.raises(ValueError):
            await async_fetch_image(
                f"file://{temp_dir}/../{os.path.basename(image_url)}",
                allowed_local_media_path=temp_dir)
        with pytest.raises(ValueError):
            await async_fetch_image(
                f"file://{temp_dir}/../{os.path.basename(image_url)}")

        with pytest.raises(ValueError):
            fetch_image(f"file://{temp_dir}/../{os.path.basename(image_url)}",
                        allowed_local_media_path=temp_dir)
        with pytest.raises(ValueError):
            fetch_image(f"file://{temp_dir}/../{os.path.basename(image_url)}")


@pytest.mark.parametrize("model", ["llava-hf/llava-v1.6-mistral-7b-hf"])
def test_repeat_and_pad_placeholder_tokens(model):
    config = AutoConfig.from_pretrained(model)
    image_token_id = config.image_token_index

    tokenizer = AutoTokenizer.from_pretrained(model)

    test_cases = [
        (
            "<image>",
            2,
            "<image><image>",
            [32000, 32000],
            [{ "offset": 0, "length": 2 }],
        ),
        (
            "<image><image>",
            2,
            "<image><image><image>",
            [32000, 32000, 32000],
            [{ "offset": 0, "length": 2 }]),
        (
            "<image><image>",
            [3, 2],
            "<image><image><image><image><image>",
            [32000, 32000, 32000, 32000, 32000],
            [{ "offset": 0, "length": 3 }, { "offset": 3, "length": 2 }],
        ),
        (
            "Image:<image>Image:<image>!",
            [3, 2],
            "Image:<image><image><image>Image:<image><image>!",
            [9833, 28747, 32000, 32000, 32000, 9833, 28747, 32000, 32000, 918],
            [{ "offset": 2, "length": 3 }, { "offset": 7, "length": 2 }],
        ),
        (
            "<image>",
            [3, 2],
            "<image><image><image>",
            [32000, 32000, 32000],
            [{ "offset": 0, "length": 3 }],
        ),
    ]  # yapf: disable

    for (
            prompt,
            repeat_count,
            expected_prompt,
            expected_token_ids,
            expected_ranges,
    ) in test_cases:
        new_prompt, new_token_ids, ranges = repeat_and_pad_placeholder_tokens(
            tokenizer=tokenizer,
            prompt=prompt,
            prompt_token_ids=tokenizer.encode(prompt,
                                              add_special_tokens=False),
            placeholder_token_id=image_token_id,
            repeat_count=repeat_count,
        )
        assert new_prompt == expected_prompt
        assert new_token_ids == expected_token_ids
        assert ranges == expected_ranges
