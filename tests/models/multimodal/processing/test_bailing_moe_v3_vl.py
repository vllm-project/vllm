# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bailing prompt expansion tests without downloading a checkpoint."""

import numpy as np
import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import TokenizersBackend

from vllm.config import ModelConfig
from vllm.model_executor.models.bailing_moe_v3_vl import (
    IMAGE_PLACEHOLDER,
    IMAGE_TOKEN,
    VIDEO_PLACEHOLDER,
    VIDEO_TOKEN,
    BailingMoeV3VLDummyInputsBuilder,
    BailingMoeV3VLMultiModalProcessor,
    BailingMoeV3VLProcessingInfo,
    _get_bailing_mrope_positions,
)
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    PlaceholderRange,
)
from vllm.multimodal.parse import (
    ImageProcessorItems,
    MultiModalDataItems,
    VideoProcessorItems,
)
from vllm.multimodal.processing.context import InputProcessingContext
from vllm.multimodal.processing.inputs import ProcessorInputs
from vllm.transformers_utils.configs.bailing_moe_v3_vl import BailingMoeV3VLConfig

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def processor(tmp_path):
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
    tokenizer = TokenizersBackend(
        tokenizer_object=backend,
        additional_special_tokens=[
            "<|vision_start|>",
            IMAGE_TOKEN,
            "<|vision_end|>",
            "<|video_start|>",
            VIDEO_TOKEN,
            "<|video_end|>",
        ],
        clean_up_tokenization_spaces=False,
    )
    BailingMoeV3VLConfig().save_pretrained(tmp_path)
    model_config = ModelConfig(
        model=str(tmp_path),
        dtype="float32",
        max_model_len=128,
        skip_tokenizer_init=True,
    )
    context = InputProcessingContext(model_config, tokenizer)
    info = BailingMoeV3VLProcessingInfo(context)
    return BailingMoeV3VLMultiModalProcessor(
        info, BailingMoeV3VLDummyInputsBuilder(info)
    )


@pytest.mark.parametrize(
    ("modality", "case"),
    [
        ("image", "wrapped"),
        ("image", "bare"),
        ("image", "existing_newline"),
        ("image", "adjacent"),
        ("video", "bare"),
        ("video", "adjacent"),
    ],
)
def test_expansion_matches_wrappers_and_newline_tokenization(processor, modality, case):
    token = IMAGE_TOKEN if modality == "image" else VIDEO_TOKEN
    wrapper = "vision" if modality == "image" else "video"
    start, end = f"<|{wrapper}_start|>", f"<|{wrapper}_end|>"
    placeholder = f"{start}{token}{end}"
    prompt, expected_text, token_counts = {
        "wrapped": (placeholder, f"{start}{token * 2}{end}\n", [2]),
        "bare": (token, f"{start}{token * 2}{end}\n", [2]),
        "existing_newline": (placeholder + "\n", f"{start}{token * 2}{end}\n\n", [2]),
        "adjacent": (
            placeholder * 2,
            f"{start}{token * 2}{end}\n{start}{token * 3}{end}\n",
            [2, 3],
        ),
    }[case]
    tokenizer = processor.info.get_tokenizer()
    token_id = tokenizer.get_vocab()[token]
    grids = torch.tensor(
        [
            [1, 2, 2 * count] if modality == "image" else [count, 2, 2]
            for count in token_counts
        ]
    )
    grid_key = f"{modality}_grid_thw"
    mm_kwargs = MultiModalKwargsItems.from_hf_inputs(
        {grid_key: grids},
        {grid_key: MultiModalFieldConfig.batched(modality)},
    )
    items = (
        ImageProcessorItems([Image.new("RGB", (1, 1))] * len(token_counts))
        if modality == "image"
        else VideoProcessorItems(
            [np.zeros((count * 2, 32, 32, 3), dtype=np.uint8) for count in token_counts]
        )
    )
    mm_items = MultiModalDataItems({modality: items})
    updates = processor._get_mm_prompt_updates(mm_items, {}, mm_kwargs)
    prompt_ids = tokenizer.encode(f"before{prompt}after", add_special_tokens=False)
    expected_ids = tokenizer.encode(
        f"before{expected_text}after", add_special_tokens=False
    )

    actual_ids, placeholders = processor._apply_prompt_updates(prompt_ids, updates)

    assert actual_ids == expected_ids
    assert len(placeholders[modality]) == len(token_counts)
    found = processor._find_mm_placeholders(actual_ids, updates)
    search_offset = 0
    for index, count in enumerate(token_counts):
        placeholder = placeholders[modality][index]
        expected_offset = expected_ids.index(token_id, search_offset)
        assert placeholder.start_idx == expected_offset
        assert placeholder.tokens == [token_id] * count
        assert placeholder.to_range().get_num_embeds() == count
        assert placeholder.to_range() == found[modality][index].to_range()
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


def _feature(modality, grid, offset):
    key = f"{modality}_grid_thw"
    item = MultiModalKwargsItems.from_hf_inputs(
        {key: torch.tensor([grid])}, {key: MultiModalFieldConfig.batched(modality)}
    )[modality][0]
    return MultiModalFeatureSpec(
        data=item,
        modality=modality,
        identifier=f"{modality}-{offset}",
        mm_position=PlaceholderRange(offset, int(np.prod(grid)) // 4),
    )


def test_video_mrope_centers_space_and_advances_time_by_two():
    feature = _feature("video", [2, 4, 6], 2)
    expected = torch.tensor(
        [
            [0, 1, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 5, 6],
            [0, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6],
            [0, 1, 1, 2, 3, 1, 2, 3, 3, 4, 5, 3, 4, 5, 5, 6],
        ]
    )
    positions, delta = _get_bailing_mrope_positions([0] * 16, [feature], 2)
    assert torch.equal(positions, expected)
    assert delta == -9
    # Ending directly on video must use temporal extent, not max spatial index.
    positions, delta = _get_bailing_mrope_positions([0] * 14, [feature], 2)
    assert torch.equal(positions, expected[:, :14])
    assert delta == -9


def test_mixed_mrope_orders_features_and_continues_text_time():
    # Keep features out of order and make the image wide enough for negative W.
    features = [_feature("video", [2, 2, 2], 7), _feature("image", [1, 2, 10], 1)]
    positions, delta = _get_bailing_mrope_positions([0] * 10, features, 2)
    assert torch.equal(
        positions,
        torch.tensor(
            [
                [0, 1, 1, 1, 1, 1, 2, 3, 5, 6],
                [0, 1, 1, 1, 1, 1, 2, 3, 5, 6],
                [0, -1, 0, 1, 2, 3, 2, 3, 5, 6],
            ]
        ),
    )
    assert delta == -3


def test_mixed_prompt_preserves_order_and_video_timestamp_mask(processor):
    tokenizer = processor.info.get_tokenizer()
    expanded = (
        "<|video_start|><0.2 seconds><|vision_start|>"
        + VIDEO_TOKEN * 6
        + "<|vision_end|><1.2 seconds><|vision_start|>"
        + VIDEO_TOKEN * 6
        + "<|vision_end|><|video_end|>"
    )
    prompt_ids = torch.tensor(tokenizer.encode(expanded, add_special_tokens=False))
    kwargs = MultiModalKwargsItems.from_hf_inputs(
        {
            "image_grid_thw": torch.tensor([[1, 2, 4], [1, 2, 4]]),
            "video_grid_thw": torch.tensor([[2, 4, 6]]),
            "video_prompt_ids": [prompt_ids],
        },
        {
            key: MultiModalFieldConfig.batched(key.split("_")[0])
            for key in ("image_grid_thw", "video_grid_thw", "video_prompt_ids")
        },
    )
    items = MultiModalDataItems(
        {
            "image": ImageProcessorItems([Image.new("RGB", (32, 32))] * 2),
            "video": VideoProcessorItems([np.zeros((4, 64, 96, 3), dtype=np.uint8)]),
        }
    )
    updates = processor._get_mm_prompt_updates(items, {}, kwargs)
    prompt = f"{IMAGE_PLACEHOLDER}then{VIDEO_PLACEHOLDER}then{IMAGE_PLACEHOLDER}after"
    image = f"<|vision_start|>{IMAGE_TOKEN * 2}<|vision_end|>\n"
    ids, placeholders = processor._apply_prompt_updates(
        tokenizer.encode(prompt, add_special_tokens=False), updates
    )
    assert ids == tokenizer.encode(
        f"{image}then{expanded}then{image}after", add_special_tokens=False
    )
    location = placeholders["video"][0].to_range()
    images = [item.to_range() for item in placeholders["image"]]
    assert images[0].offset < location.offset < images[1].offset
    assert [item.get_num_embeds() for item in images] == [2, 2]
    assert location.get_num_embeds() == 12
    selected = torch.tensor(ids[location.offset : location.offset + location.length])
    assert torch.equal(
        location.is_embed, selected == tokenizer.get_vocab()[VIDEO_TOKEN]
    )
    features = [MultiModalFeatureSpec(kwargs["video"][0], "video", "clip", location)]
    features += [
        MultiModalFeatureSpec(kwargs["image"][i], "image", str(i), item)
        for i, item in enumerate(images)
    ]
    positions, delta = _get_bailing_mrope_positions(ids, features, 2)
    assert positions.shape == (3, len(ids))
    assert delta == -12  # Two six-token frames and two two-token images.
    indices = torch.where(torch.tensor(ids) == tokenizer.get_vocab()[VIDEO_TOKEN])[0]
    for frame in indices.reshape(2, 6):
        t = positions[0, frame[0]]
        assert positions[0, frame].tolist() == [t.item()] * 6
        assert (positions[1, frame] - t).tolist() == [0, 0, 0, 1, 1, 1]
        assert (positions[2, frame] - t).tolist() == [-1, 0, 1, -1, 0, 1]


def test_video_timestamp_metadata_changes_cache_key(processor):
    frames = np.zeros((4, 32, 32, 3), dtype=np.uint8)
    hashes = []
    for fps in (2.0, 4.0):
        items = processor.info.parse_mm_data(
            {
                "video": [
                    (
                        frames,
                        {
                            "fps": fps,
                            "frames_indices": [0, 1, 2, 3],
                        },
                    )
                ]
            }
        )
        inputs = ProcessorInputs(prompt=[], mm_data_items=items)
        hashes.append(inputs.get_mm_hashes("bailing", "sha256")["video"][0])
    assert hashes[0] != hashes[1]
