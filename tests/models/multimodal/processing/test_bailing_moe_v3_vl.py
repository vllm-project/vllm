# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bailing prompt expansion tests without downloading a checkpoint."""

from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from vllm.model_executor.models.bailing_moe_v3_vl import (
    IMAGE_PLACEHOLDER,
    IMAGE_TOKEN,
    BailingMoeV3VLMultiModalProcessor,
)
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import ImageProcessorItems, MultiModalDataItems

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def processor():
    vocab = {
        token: index
        for index, token in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))
    }
    # A newline added by the processor must merge with an existing newline.
    vocab["ĊĊ"] = len(vocab)
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[("Ċ", "Ċ")]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=False
    )
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        additional_special_tokens=[
            "<|vision_start|>",
            IMAGE_TOKEN,
            "<|vision_end|>",
        ],
        clean_up_tokenization_spaces=False,
    )
    processor = object.__new__(BailingMoeV3VLMultiModalProcessor)
    processor.info = SimpleNamespace(
        get_tokenizer=lambda: tokenizer,
        get_hf_config=lambda: SimpleNamespace(
            vision_config=SimpleNamespace(spatial_merge_size=2)
        ),
    )
    return processor


@pytest.mark.parametrize(
    ("prompt", "expected_text", "token_counts"),
    [
        (
            f"before{IMAGE_PLACEHOLDER}after",
            f"before<|vision_start|>{IMAGE_TOKEN * 2}<|vision_end|>\nafter",
            [2],
        ),
        (
            f"before{IMAGE_TOKEN}after",
            f"before<|vision_start|>{IMAGE_TOKEN * 2}<|vision_end|>\nafter",
            [2],
        ),
        (
            f"before{IMAGE_PLACEHOLDER}\nafter",
            f"before<|vision_start|>{IMAGE_TOKEN * 2}<|vision_end|>\n\nafter",
            [2],
        ),
        (
            f"before{IMAGE_PLACEHOLDER}{IMAGE_PLACEHOLDER}after",
            f"before<|vision_start|>{IMAGE_TOKEN * 2}<|vision_end|>\n"
            f"<|vision_start|>{IMAGE_TOKEN * 3}<|vision_end|>\nafter",
            [2, 3],
        ),
    ],
    ids=["wrapped", "bare", "existing_newline", "adjacent_images"],
)
def test_image_expansion_matches_wrappers_and_newline_tokenization(
    processor, prompt, expected_text, token_counts
):
    tokenizer = processor.info.get_tokenizer()
    image_id = tokenizer.get_vocab()[IMAGE_TOKEN]
    grids = torch.tensor([[1, 2, 2 * count] for count in token_counts])
    mm_kwargs = MultiModalKwargsItems.from_hf_inputs(
        {"image_grid_thw": grids},
        {"image_grid_thw": MultiModalFieldConfig.batched("image")},
    )
    mm_items = MultiModalDataItems(
        {"image": ImageProcessorItems([Image.new("RGB", (1, 1))] * len(token_counts))}
    )
    updates = processor._get_mm_prompt_updates(mm_items, {}, mm_kwargs)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    expected_ids = tokenizer.encode(expected_text, add_special_tokens=False)

    actual_ids, placeholders = processor._apply_prompt_updates(prompt_ids, updates)

    assert actual_ids == expected_ids
    assert len(placeholders["image"]) == len(token_counts)
    found = processor._find_mm_placeholders(actual_ids, updates)
    search_offset = 0
    for index, count in enumerate(token_counts):
        placeholder = placeholders["image"][index]
        expected_offset = expected_ids.index(image_id, search_offset)
        assert placeholder.start_idx == expected_offset
        assert placeholder.tokens == [image_id] * count
        assert placeholder.to_range().get_num_embeds() == count
        assert placeholder.to_range() == found["image"][index].to_range()
        search_offset = expected_offset + count


def test_text_only_prompt_preserves_original_token_ids(processor):
    tokenizer = processor.info.get_tokenizer()
    prompt_ids = tokenizer.encode("before\n", add_special_tokens=False)
    prompt_ids += tokenizer.encode("\nafter", add_special_tokens=False)
    # Deliberately use two newline tokens instead of their merged BPE token.
    assert prompt_ids != tokenizer.encode("before\n\nafter", add_special_tokens=False)

    actual_ids, placeholders = processor._apply_prompt_updates(prompt_ids, {})

    assert actual_ids == prompt_ids
    assert placeholders == {}
