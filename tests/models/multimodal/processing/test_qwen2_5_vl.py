# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen2-VL family prompt expansion tests without downloading a checkpoint."""

from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from vllm.model_executor.models.openpangu_vl import (
    OpenPanguVLMultiModalProcessor,
)
from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VLMultiModalProcessor
from vllm.model_executor.models.qwen2_vl import (
    Qwen2VLDummyInputsBuilder,
    Qwen2VLMultiModalProcessor,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


@pytest.fixture(autouse=True)
def mock_get_model_cls(monkeypatch):
    """Route get_model_cls to the mock model class carried by model_config."""
    import vllm.model_executor.model_loader as loader

    monkeypatch.setattr(
        loader,
        "get_model_cls",
        lambda model_config: model_config._mock_model_cls,
    )


def _make_processor(processor_cls, special_tokens, hf_processor, placeholder_strs):
    vocab = {"[UNK]": 0, "before": 1, "after": 2}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        additional_special_tokens=special_tokens,
    )

    def _placeholder_fn(modality, i):
        result = placeholder_strs.get(modality)
        if isinstance(result, BaseException):
            raise result
        return result

    model_cls = SimpleNamespace(get_placeholder_str=_placeholder_fn)

    processor = object.__new__(processor_cls)
    processor.info = SimpleNamespace(
        get_hf_processor=lambda **kwargs: hf_processor,
        get_image_processor=lambda **kwargs: SimpleNamespace(merge_size=2),
        get_tokenizer=lambda: tokenizer,
        ctx=SimpleNamespace(
            model_config=SimpleNamespace(_mock_model_cls=model_cls),
            get_mm_config=lambda: SimpleNamespace(video_pruning_rate=None),
        ),
    )
    return processor


@pytest.fixture
def processor():
    return _make_processor(
        Qwen2_5_VLMultiModalProcessor,
        [
            "<|vision_start|>",
            "<|image_pad|>",
            "<|vision_end|>",
            "<|video_pad|>",
        ],
        SimpleNamespace(
            image_token="<|image_pad|>",
            video_token="<|video_pad|>",
        ),
        {
            "image": "<|vision_start|><|image_pad|><|vision_end|>",
            "video": "<|vision_start|><|video_pad|><|vision_end|>",
        },
    )


@pytest.fixture
def qwen2_vl_processor():
    return _make_processor(
        Qwen2VLMultiModalProcessor,
        [
            "<|vision_start|>",
            "<|image_pad|>",
            "<|vision_end|>",
            "<|video_pad|>",
        ],
        SimpleNamespace(
            image_token="<|image_pad|>",
            video_token="<|video_pad|>",
        ),
        {
            "image": "<|vision_start|><|image_pad|><|vision_end|>",
            "video": "<|vision_start|><|video_pad|><|vision_end|>",
        },
    )


@pytest.fixture
def exaone_style_processor():
    # EXAONE 4.5 reuses Qwen2VLMultiModalProcessor but its tokenizer has
    # <vision>/</vision> instead of <|vision_start|>/<|vision_end|>.
    return _make_processor(
        Qwen2VLMultiModalProcessor,
        [
            "<vision>",
            "</vision>",
            "<|image_pad|>",
            "<|video_pad|>",
        ],
        SimpleNamespace(
            image_token="<|image_pad|>",
            video_token="<|video_pad|>",
        ),
        {
            "image": "<vision><|image_pad|></vision>",
            "video": "<vision><|video_pad|></vision>",
        },
    )


@pytest.fixture
def dots_ocr_style_processor():
    # dots.ocr registers Qwen2VLMultiModalProcessor directly with its own
    # <|img|><|imgpad|><|endofimg|> wrapper.
    return _make_processor(
        Qwen2VLMultiModalProcessor,
        [
            "<|img|>",
            "<|imgpad|>",
            "<|endofimg|>",
            "<|video_pad|>",
        ],
        SimpleNamespace(
            image_token="<|imgpad|>",
            video_token="<|video_pad|>",
        ),
        {
            "image": "<|img|><|imgpad|><|endofimg|>",
            "video": None,
        },
    )


@pytest.fixture
def jina_style_processor():
    # Jina-VL inherits Qwen2VLMultiModalProcessor; its get_placeholder_str
    # raises for video instead of returning None.
    return _make_processor(
        Qwen2VLMultiModalProcessor,
        [
            "<|vision_start|>",
            "<|image_pad|>",
            "<|vision_end|>",
            "<|video_pad|>",
        ],
        SimpleNamespace(
            image_token="<|image_pad|>",
            video_token="<|video_pad|>",
        ),
        {
            "image": "<|vision_start|><|image_pad|><|vision_end|>",
            "video": ValueError("Only image modality is supported"),
        },
    )


@pytest.fixture
def openpangu_vl_processor():
    return _make_processor(
        OpenPanguVLMultiModalProcessor,
        ["[unused18]", "[unused19]", "[unused20]", "[unused32]"],
        SimpleNamespace(
            image_token="[unused19]",
            video_token="[unused32]",
        ),
        {
            "image": "[unused18][unused19][unused20]",
            "video": "[unused18][unused32][unused20]",
        },
    )


def _resolve_image_update(processor):
    updates = processor._get_prompt_updates(
        mm_items={},
        hf_processor_mm_kwargs={},
        out_mm_kwargs={
            "image": [
                {"image_grid_thw": SimpleNamespace(data=torch.tensor([1, 2, 4]))}
            ],
            "video": [],
        },
    )
    return [updates[0].resolve(0)]


def test_qwen25_vl_does_not_match_bare_image_pad_in_user_text(processor):
    tokenizer = processor.info.get_tokenizer()
    vocab = tokenizer.get_vocab()
    image_pad = vocab["<|image_pad|>"]
    vision_start = vocab["<|vision_start|>"]
    vision_end = vocab["<|vision_end|>"]

    image_update = _resolve_image_update(processor)
    prompt = tokenizer.encode(
        "before <|image_pad|> <|vision_start|><|image_pad|><|vision_end|> after",
        add_special_tokens=False,
    )

    new_prompt, placeholders = processor._apply_prompt_updates(
        prompt,
        {"image": [image_update]},
    )

    assert new_prompt.count(image_pad) == 3
    assert new_prompt.count(vision_start) == 1
    assert new_prompt.count(vision_end) == 1
    assert placeholders["image"][0].tokens == [
        vision_start,
        image_pad,
        image_pad,
        vision_end,
    ]
    assert placeholders["image"][0].is_embed.tolist() == [False, True, True, False]


def test_qwen2_vl_does_not_match_bare_image_pad_in_user_text(
    qwen2_vl_processor,
):
    tokenizer = qwen2_vl_processor.info.get_tokenizer()
    vocab = tokenizer.get_vocab()
    image_pad = vocab["<|image_pad|>"]
    vision_start = vocab["<|vision_start|>"]
    vision_end = vocab["<|vision_end|>"]

    image_update = _resolve_image_update(qwen2_vl_processor)
    prompt = tokenizer.encode(
        "before <|image_pad|> <|vision_start|><|image_pad|><|vision_end|> after",
        add_special_tokens=False,
    )

    new_prompt, placeholders = qwen2_vl_processor._apply_prompt_updates(
        prompt,
        {"image": [image_update]},
    )

    assert new_prompt.count(image_pad) == 3
    assert new_prompt.count(vision_start) == 1
    assert new_prompt.count(vision_end) == 1
    assert placeholders["image"][0].tokens == [
        vision_start,
        image_pad,
        image_pad,
        vision_end,
    ]
    assert placeholders["image"][0].is_embed.tolist() == [False, True, True, False]


def test_exaone_style_wrapper_derived_from_placeholder_str(
    exaone_style_processor,
):
    # EXAONE 4.5's tokenizer lacks <|vision_start|>; the target must be
    # derived from its own placeholder string, not hardcoded.
    tokenizer = exaone_style_processor.info.get_tokenizer()
    vocab = tokenizer.get_vocab()
    image_pad = vocab["<|image_pad|>"]
    vision_start = vocab["<vision>"]
    vision_end = vocab["</vision>"]

    image_update = _resolve_image_update(exaone_style_processor)
    prompt = tokenizer.encode(
        "before <|image_pad|> <vision><|image_pad|></vision> after",
        add_special_tokens=False,
    )

    new_prompt, placeholders = exaone_style_processor._apply_prompt_updates(
        prompt,
        {"image": [image_update]},
    )

    assert new_prompt.count(image_pad) == 3
    assert new_prompt.count(vision_start) == 1
    assert new_prompt.count(vision_end) == 1
    assert placeholders["image"][0].tokens == [
        vision_start,
        image_pad,
        image_pad,
        vision_end,
    ]
    assert placeholders["image"][0].is_embed.tolist() == [False, True, True, False]


def test_dots_ocr_style_wrapper_derived_from_placeholder_str(
    dots_ocr_style_processor,
):
    # dots.ocr uses <|img|><|imgpad|><|endofimg|>; the shared processor must
    # not assume the Qwen wrapper.
    tokenizer = dots_ocr_style_processor.info.get_tokenizer()
    vocab = tokenizer.get_vocab()
    image_pad = vocab["<|imgpad|>"]
    img_start = vocab["<|img|>"]
    img_end = vocab["<|endofimg|>"]

    image_update = _resolve_image_update(dots_ocr_style_processor)
    prompt = tokenizer.encode(
        "before <|imgpad|> <|img|><|imgpad|><|endofimg|> after",
        add_special_tokens=False,
    )

    new_prompt, placeholders = dots_ocr_style_processor._apply_prompt_updates(
        prompt,
        {"image": [image_update]},
    )

    assert new_prompt.count(image_pad) == 3
    assert new_prompt.count(img_start) == 1
    assert new_prompt.count(img_end) == 1
    assert placeholders["image"][0].tokens == [
        img_start,
        image_pad,
        image_pad,
        img_end,
    ]
    assert placeholders["image"][0].is_embed.tolist() == [False, True, True, False]


def test_jina_style_video_falls_back_to_bare_pad(jina_style_processor):
    # get_placeholder_str raises for video; the processor must fall back
    # to the bare pad token instead of crashing.
    updates = jina_style_processor._get_prompt_updates(
        mm_items={},
        hf_processor_mm_kwargs={},
        out_mm_kwargs={"image": [], "video": []},
    )
    tokenizer = jina_style_processor.info.get_tokenizer()
    video_pad = tokenizer.get_vocab()["<|video_pad|>"]

    assert updates[1].target == [video_pad]


def test_openpangu_vl_does_not_match_bare_image_pad_in_user_text(
    openpangu_vl_processor,
):
    tokenizer = openpangu_vl_processor.info.get_tokenizer()
    vocab = tokenizer.get_vocab()
    image_pad = vocab["[unused19]"]
    vision_start = vocab["[unused18]"]
    vision_end = vocab["[unused20]"]

    image_update = _resolve_image_update(openpangu_vl_processor)
    prompt = tokenizer.encode(
        "before [unused19] [unused18][unused19][unused20] after",
        add_special_tokens=False,
    )

    new_prompt, placeholders = openpangu_vl_processor._apply_prompt_updates(
        prompt,
        {"image": [image_update]},
    )

    assert new_prompt.count(image_pad) == 3
    assert new_prompt.count(vision_start) == 1
    assert new_prompt.count(vision_end) == 1
    assert placeholders["image"][0].tokens == [
        vision_start,
        image_pad,
        image_pad,
        vision_end,
    ]
    assert placeholders["image"][0].is_embed.tolist() == [False, True, True, False]


def test_dummy_text_emits_full_wrapper(qwen2_vl_processor):
    # The dummy prompt must contain the complete wrapper so that it matches
    # the replacement targets (used for profiling).
    builder = Qwen2VLDummyInputsBuilder(qwen2_vl_processor.info)

    assert builder.get_dummy_text({"image": 1}) == (
        "<|vision_start|><|image_pad|><|vision_end|>"
    )
    assert builder.get_dummy_text({"image": 1, "video": 1}) == (
        "<|vision_start|><|image_pad|><|vision_end|>"
        "<|vision_start|><|video_pad|><|vision_end|>"
    )


def test_dummy_text_emits_full_wrapper_dots_ocr(dots_ocr_style_processor):
    builder = Qwen2VLDummyInputsBuilder(dots_ocr_style_processor.info)

    assert builder.get_dummy_text({"image": 2}) == (
        "<|img|><|imgpad|><|endofimg|><|img|><|imgpad|><|endofimg|>"
    )
