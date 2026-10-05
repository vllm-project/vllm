# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen2.5-VL prompt expansion tests without downloading a checkpoint."""

from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from vllm.model_executor.models.openpangu_vl import (
    OpenPanguVLMultiModalProcessor,
)
from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VLMultiModalProcessor
from vllm.model_executor.models.qwen2_vl import Qwen2VLMultiModalProcessor

pytestmark = pytest.mark.cpu_test


def _make_processor(processor_cls, special_tokens, hf_processor):
    vocab = {"[UNK]": 0, "before": 1, "after": 2}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        additional_special_tokens=special_tokens,
    )

    processor = object.__new__(processor_cls)
    processor.info = SimpleNamespace(
        get_hf_processor=lambda **kwargs: hf_processor,
        get_image_processor=lambda **kwargs: SimpleNamespace(merge_size=2),
        get_tokenizer=lambda: tokenizer,
        ctx=SimpleNamespace(
            get_mm_config=lambda: SimpleNamespace(video_pruning_rate=None)
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
    )


@pytest.fixture
def openpangu_vl_processor():
    return _make_processor(
        OpenPanguVLMultiModalProcessor,
        ["[unused18]", "[unused19]", "[unused20]", "[unused32]"],
        SimpleNamespace(
            image_token="[unused19]",
            video_token="[unused32]",
            vision_start_token="[unused18]",
            vision_end_token="[unused20]",
        ),
    )


def test_qwen25_vl_does_not_match_bare_image_pad_in_user_text(processor):
    tokenizer = processor.info.get_tokenizer()
    vocab = tokenizer.get_vocab()
    image_pad = vocab["<|image_pad|>"]
    vision_start = vocab["<|vision_start|>"]
    vision_end = vocab["<|vision_end|>"]

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
    image_update = [updates[0].resolve(0)]
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

    updates = qwen2_vl_processor._get_prompt_updates(
        mm_items={},
        hf_processor_mm_kwargs={},
        out_mm_kwargs={
            "image": [
                {"image_grid_thw": SimpleNamespace(data=torch.tensor([1, 2, 4]))}
            ],
            "video": [],
        },
    )
    image_update = [updates[0].resolve(0)]
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


def test_openpangu_vl_does_not_match_bare_image_pad_in_user_text(
    openpangu_vl_processor,
):
    tokenizer = openpangu_vl_processor.info.get_tokenizer()
    vocab = tokenizer.get_vocab()
    image_pad = vocab["[unused19]"]
    vision_start = vocab["[unused18]"]
    vision_end = vocab["[unused20]"]

    updates = openpangu_vl_processor._get_prompt_updates(
        mm_items={},
        hf_processor_mm_kwargs={},
        out_mm_kwargs={
            "image": [
                {"image_grid_thw": SimpleNamespace(data=torch.tensor([1, 2, 4]))}
            ],
            "video": [],
        },
    )
    image_update = [updates[0].resolve(0)]
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
