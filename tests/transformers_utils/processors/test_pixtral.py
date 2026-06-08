# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import pytest

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


def test_integer_token_ids(processor):
    """Integer IDs derived from mistral-common special_ids."""
    assert processor.image_token_id == 11
    assert processor.image_break_id == 10
    assert processor.image_end_id == 12


def test_string_token_names(processor):
    """String tokens match canonical Pixtral vocabulary names."""
    assert processor.image_token == "[IMG]"
    assert processor.image_break_token == "[IMG_BREAK]"
    assert processor.image_end_token == "[IMG_END]"


def test_hf_compatible_token_id_aliases(processor):
    """HF PixtralProcessor-style _token_id aliases agree with base IDs."""
    assert processor.image_break_token_id == processor.image_break_id
    assert processor.image_end_token_id == processor.image_end_id


def test_token_properties_are_instance_attributes(processor):
    """All token attributes must be plain instance attributes, not properties,
    so that code doing ``vocab[processor.image_break_token]`` or direct
    assignment works identically to HF PixtralProcessor."""
    cls = type(processor)
    for attr in (
        "image_token",
        "image_break_token",
        "image_end_token",
        "image_token_id",
        "image_break_token_id",
        "image_end_token_id",
    ):
        assert not isinstance(getattr(cls, attr, None), property), (
            f"{attr} must be an instance attribute, not a @property"
        )
        assert attr in processor.__dict__, (
            f"{attr} must be set in __init__, not inherited"
        )


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
    assert processor.replace_image_token(processed_images={}, image_idx=0) == "[IMG]"
