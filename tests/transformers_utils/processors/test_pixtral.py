# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import pytest
from mistral_common.tokens.tokenizers.base import SpecialTokens

from vllm.transformers_utils.processors import pixtral as pixtral_module
from vllm.transformers_utils.processors.pixtral import (
    MistralCommonImageProcessor,
    MistralCommonPixtralProcessor,
)

pytestmark = pytest.mark.skip_global_cleanup


@dataclass
class FakeSpecialIds:
    img_break: int = 10
    img: int = 11
    img_end: int = 12


class FakeImageEncoder:
    special_ids = FakeSpecialIds()


class FakeImageProcessor:
    mm_encoder = FakeImageEncoder()


class FakeTransformersTokenizer:
    pass


class FakeTokenizer:
    transformers_tokenizer = FakeTransformersTokenizer()


@pytest.fixture()
def processor():
    return MistralCommonPixtralProcessor(
        tokenizer=FakeTokenizer(),
        image_processor=FakeImageProcessor(),
    )


def test_token_attributes(processor):
    """Token IDs, string names, and HF-compatible aliases are all correct."""
    assert processor.image_token_id == 11
    assert processor.image_break_id == 10
    assert processor.image_end_id == 12

    assert processor.image_token == SpecialTokens.img.value
    assert processor.image_break_token == SpecialTokens.img_break.value
    assert processor.image_end_token == SpecialTokens.img_end.value

    assert processor.image_break_token_id == processor.image_break_id
    assert processor.image_end_token_id == processor.image_end_id


def test_image_processor_fetch_images_accepts_images_and_lists(monkeypatch):
    image_processor = MistralCommonImageProcessor(FakeImageEncoder())
    image = object()
    monkeypatch.setattr(
        pixtral_module, "is_valid_image", lambda candidate: candidate is image
    )

    assert image_processor.fetch_images(image) is image
    assert image_processor.fetch_images([image, (image,)]) == [image, [image]]


def test_image_processor_fetch_images_rejects_invalid_input(monkeypatch):
    image_processor = MistralCommonImageProcessor(FakeImageEncoder())
    monkeypatch.setattr(pixtral_module, "is_valid_image", lambda candidate: False)

    with pytest.raises(TypeError, match="only a single or a list of entries"):
        image_processor.fetch_images(object())


def test_replace_image_token_is_noop_for_mistral_template_tokens(processor):
    assert (
        processor.replace_image_token(processed_images={}, image_idx=0)
        == SpecialTokens.img.value
    )
