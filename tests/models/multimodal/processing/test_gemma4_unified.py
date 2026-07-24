# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from collections.abc import Mapping

import numpy as np
import pytest
import torch
from PIL import Image as PILImage

from vllm.model_executor.models.gemma4_mm import (
    Gemma4AudioInputs,
    Gemma4ImagePixelInputs,
    _pad_ragged_audio_features,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig

from ....conftest import ImageTestAssets
from ...utils import build_model_context

# The Unified model ID for testing purposes
GEMMA4_UNIFIED_MODEL_ID = "google/gemma-4-12B-it"


def test_gemma4_unified_image_schema_accepts_variable_patch_counts():
    Gemma4ImagePixelInputs(
        pixel_values=[
            torch.randn(10080, 768),
            torch.randn(2520, 768),
        ],
        pixel_position_ids=[
            torch.zeros(10080, 2, dtype=torch.long),
            torch.zeros(2520, 2, dtype=torch.long),
        ],
    )


def test_gemma4_unified_image_batching_keeps_variable_patch_counts_unstacked():
    field = MultiModalFieldConfig.batched("image").field
    elems = field.build_elems(
        "image",
        "pixel_values",
        [torch.randn(10080, 768), torch.randn(2520, 768)],
    )

    reduced = field.reduce_data(list(elems))

    assert isinstance(reduced, list)
    assert [tensor.shape for tensor in reduced] == [
        torch.Size([10080, 768]),
        torch.Size([2520, 768]),
    ]


@pytest.mark.parametrize(
    "image_width,image_height,max_soft_tokens",
    [
        (900, 3, 280),
        (3, 900, 280),
        (900, 3, 70),
        (4000, 2, 1120),
    ],
)
@pytest.mark.parametrize("model_id", [GEMMA4_UNIFIED_MODEL_ID])
def test_compute_num_soft_tokens_does_not_exceed_max_soft_tokens(
    model_id: str,
    image_width: int,
    image_height: int,
    max_soft_tokens: int,
):
    """Verify ``_compute_num_soft_tokens`` caps output at ``max_soft_tokens``."""
    ctx = build_model_context(
        model_id,
        mm_processor_kwargs={"do_pan_and_scan": True},
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    num_soft_tokens = processor.info._compute_num_soft_tokens(
        image_width=image_width,
        image_height=image_height,
        max_soft_tokens=max_soft_tokens,
    )

    assert num_soft_tokens <= max_soft_tokens, (
        f"_compute_num_soft_tokens returned {num_soft_tokens} for "
        f"image_width={image_width}, image_height={image_height}, "
        f"max_soft_tokens={max_soft_tokens} — exceeds the cap."
    )


@pytest.mark.parametrize(
    ("mm_processor_kwargs", "expected_image_tokens"),
    [
        ({}, 280),
        ({"max_soft_tokens": 70}, 70),
        ({"max_soft_tokens": 280}, 280),
        ({"max_soft_tokens": 1120}, 1120),
        ({"images_kwargs": {"max_soft_tokens": 560}}, 560),
        ({"images_kwargs": None}, 280),
        ({"images_kwargs": "not-a-dict"}, 280),
    ],
)
@pytest.mark.parametrize("model_id", [GEMMA4_UNIFIED_MODEL_ID])
def test_get_mm_max_tokens_per_item_respects_configured_max_soft_tokens(
    model_id: str,
    mm_processor_kwargs: dict[str, object],
    expected_image_tokens: int,
):
    ctx = build_model_context(
        model_id,
        mm_processor_kwargs=mm_processor_kwargs,
        limit_mm_per_prompt={"image": 1, "video": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    tokens = processor.info.get_mm_max_tokens_per_item(
        seq_len=ctx.model_config.max_model_len,
        mm_counts={"image": 1, "video": 1},
    )

    assert tokens is not None
    assert tokens["image"] == expected_image_tokens
    assert tokens["video"] == 32 * (70 + 2 + 6)


@pytest.mark.parametrize(
    ("limit_mm_per_prompt", "expected_video_tokens"),
    [
        ({"video": 1}, 32 * (70 + 2 + 6)),
        ({"video": {"count": 1}}, 32 * (70 + 2 + 6)),
        ({"video": {"count": 1, "num_frames": 1}}, 1 * (70 + 2 + 6)),
        ({"video": {"count": 1, "num_frames": 8}}, 8 * (70 + 2 + 6)),
        ({"video": {"count": 1, "num_frames": 32}}, 32 * (70 + 2 + 6)),
        ({"video": {"count": 1, "num_frames": 40}}, 32 * (70 + 2 + 6)),
    ],
)
@pytest.mark.parametrize("model_id", [GEMMA4_UNIFIED_MODEL_ID])
def test_get_mm_max_tokens_per_item_respects_configured_video_num_frames(
    model_id: str,
    limit_mm_per_prompt: Mapping[str, int | Mapping[str, int]],
    expected_video_tokens: int,
):
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt=limit_mm_per_prompt,
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    tokens = processor.info.get_mm_max_tokens_per_item(
        seq_len=ctx.model_config.max_model_len,
        mm_counts={"video": 1},
    )

    assert tokens is not None
    assert tokens["image"] == 280
    assert tokens["video"] == expected_video_tokens


@pytest.mark.parametrize("model_id", [GEMMA4_UNIFIED_MODEL_ID])
def test_get_prompt_updates_respects_nested_max_soft_tokens(model_id: str):
    ctx = build_model_context(
        model_id,
        mm_processor_kwargs={"images_kwargs": {"max_soft_tokens": 560}},
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    image = PILImage.new("RGB", (1000, 1000), color="white")
    image_size = image.size
    mm_items = processor.info.parse_mm_data({"image": image})

    prompt_update = processor._get_prompt_updates(mm_items, {}, {})[0]
    replacement = prompt_update.resolve(0).content.full
    expected = processor.info.get_image_repl(
        image_width=image_size[0],
        image_height=image_size[1],
        processor=processor.info.get_hf_processor(),
        max_soft_tokens=560,
    ).full

    assert replacement == expected


@pytest.mark.parametrize("model_id", [GEMMA4_UNIFIED_MODEL_ID])
def test_limit_mm_per_prompt(
    image_assets: ImageTestAssets,
    model_id: str,
):
    """Test that limit_mm_per_prompt restricts multiple images correctly."""
    ctx = build_model_context(
        model_id,
        mm_processor_kwargs={},
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    prompt = "<image><image>"
    images = [asset.pil_image for asset in image_assets][:2]
    if len(images) < 2:
        images = [images[0], images[0]]

    mm_data = {"image": images}

    with pytest.raises(ValueError, match="At most 1 image"):
        processor(
            prompt,
            mm_items=processor.info.parse_mm_data(mm_data),
            hf_processor_mm_kwargs={},
        )


def test_audio_field_batching_repads_ragged_lengths():
    """Audios with different frame counts arrive at the model as a plain
    list (batched() does not re-pad); _pad_ragged_audio_features must
    normalize them back to padded tensors."""
    features = [torch.randn(3, 640), torch.randn(6, 640)]
    masks = [torch.ones(3, dtype=torch.bool), torch.ones(6, dtype=torch.bool)]

    field = MultiModalFieldConfig.batched("audio").field
    reduced_features = field.reduce_data(
        list(field.build_elems("audio", "input_features_padded", features))
    )
    reduced_masks = field.reduce_data(
        list(field.build_elems("audio", "input_features_mask", masks))
    )
    assert isinstance(reduced_features, list)

    padded_features, padded_masks = _pad_ragged_audio_features(
        reduced_features, reduced_masks
    )
    assert padded_features.shape == torch.Size([2, 6, 640])
    assert padded_masks.shape == torch.Size([2, 6])
    assert padded_masks.sum(-1).tolist() == [3, 6]
    assert torch.equal(padded_features[0, :3], features[0])
    assert torch.equal(padded_features[1], features[1])

    Gemma4AudioInputs(
        input_features_padded=padded_features,
        input_features_mask=padded_masks,
    )


@pytest.mark.parametrize(
    "residue",
    # residues 1..160 are exactly where the tower-based mel/conv formula
    # undercounts ceil(L / 640) by one
    [0, 1, 160, 161, 639],
)
@pytest.mark.parametrize("model_id", [GEMMA4_UNIFIED_MODEL_ID])
def test_audio_repl_token_count_matches_hf_processor(model_id: str, residue: int):
    """Unified audio placeholder count must equal the HF processor's
    per-audio valid frame count, ceil(L / 640)."""
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"audio": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    hf_processor = processor.info.get_hf_processor()

    audio_len = 640 * 5 + residue
    repl = processor.info.get_audio_repl(
        audio_len=audio_len,
        processor=hf_processor,
    )
    num_repl_tokens = repl.full.count(hf_processor.audio_token_id)

    hf_features = hf_processor.feature_extractor(
        [np.zeros(audio_len, dtype=np.float32)]
    )
    num_hf_tokens = int(hf_features["input_features_mask"][0].sum())

    assert num_hf_tokens == math.ceil(audio_len / 640)
    assert num_repl_tokens == num_hf_tokens


@pytest.mark.parametrize("model_id", [GEMMA4_UNIFIED_MODEL_ID])
def test_multi_audio_apply_different_lengths(model_id: str):
    """Two audios of different lengths must produce self-contained
    unpadded per-item features whose frame counts match the placeholders."""
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"audio": 2},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    hf_processor = processor.info.get_hf_processor()

    sr = hf_processor.feature_extractor.sampling_rate
    audios = [
        (np.zeros(640 * 3, dtype=np.float32), sr),
        (np.zeros(640 * 5 + 100, dtype=np.float32), sr),
    ]
    expected_frames = [3, 6]

    prompt = hf_processor.audio_token * 2
    processed_inputs = processor(
        prompt,
        mm_items=processor.info.parse_mm_data({"audio": audios}),
        hf_processor_mm_kwargs={},
    )

    audio_placeholders = processed_inputs["mm_placeholders"]["audio"]
    assert [p.get_num_embeds() for p in audio_placeholders] == expected_frames

    mm_items = processed_inputs["mm_kwargs"]["audio"]
    for item, num_frames in zip(mm_items, expected_frames, strict=True):
        assert item["input_features_padded"].data.shape == torch.Size(
            [num_frames, 640]
        )
        assert int(item["input_features_mask"].data.sum()) == num_frames
