# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import transformers.image_utils
from PIL import Image

from vllm.transformers_utils.processors.pixtral import MistralCommonImageProcessor


@pytest.fixture(scope="module")
def image_processor() -> MistralCommonImageProcessor:
    return MistralCommonImageProcessor(mm_encoder=None)


def test_fetch_images_passes_through_decoded_image(
    image_processor: MistralCommonImageProcessor,
):
    image = Image.new("RGB", (4, 4))
    result = image_processor.fetch_images(image)
    assert result is image


def test_fetch_images_recurses_over_list(
    image_processor: MistralCommonImageProcessor,
):
    a = Image.new("RGB", (4, 4))
    b = Image.new("RGB", (8, 8))
    result = image_processor.fetch_images([a, b])
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] is a
    assert result[1] is b


def test_fetch_images_recurses_over_nested_list(
    image_processor: MistralCommonImageProcessor,
):
    a = Image.new("RGB", (4, 4))
    b = Image.new("RGB", (8, 8))
    result = image_processor.fetch_images([[a], [b]])
    assert result == [[a], [b]]


def test_fetch_images_str_delegates_to_load_image(
    monkeypatch, image_processor: MistralCommonImageProcessor
):
    sentinel = Image.new("RGB", (2, 2))
    received: dict[str, object] = {}

    def fake_load_image(path):
        received["path"] = path
        return sentinel

    monkeypatch.setattr(transformers.image_utils, "load_image", fake_load_image)

    result = image_processor.fetch_images("/tmp/fake.png")
    assert result is sentinel
    assert received["path"] == "/tmp/fake.png"


def test_fetch_images_rejects_unsupported_type(
    image_processor: MistralCommonImageProcessor,
):
    with pytest.raises(TypeError, match="only a single or a list"):
        image_processor.fetch_images(42)
