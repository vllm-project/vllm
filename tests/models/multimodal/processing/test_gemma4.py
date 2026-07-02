# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping

import pytest
import torch
from PIL import Image as PILImage

from vllm.model_executor.models.gemma4_mm import (
    Gemma4ForConditionalGeneration,
    Gemma4ImagePixelInputs,
    _pad_and_stack,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.utils.mem_constants import GiB_bytes

from ....conftest import ImageTestAssets
from ...utils import build_model_context

# TODO: to be updated to "google/gemma-4-e2b-it" once the models are available
GEMMA4_MODEL_ID = "google/gemma-4-E2B-it"


def test_gemma4_image_schema_accepts_variable_patch_counts():
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


def test_gemma4_image_batching_keeps_variable_patch_counts_unstacked():
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


def test_gemma4_audio_batching_keeps_variable_lengths_unstacked():
    field = MultiModalFieldConfig.batched("audio").field
    elems = field.build_elems(
        "audio",
        "input_features_padded",
        [torch.randn(3, 80), torch.randn(5, 80)],
    )

    reduced = field.reduce_data(list(elems))

    # Clips of differing frame counts can't be stacked, so batched() hands the
    # model a plain list -- the input that used to crash _process_audio_input
    # via `list.squeeze(1)`.
    assert isinstance(reduced, list)
    assert [tensor.shape for tensor in reduced] == [
        torch.Size([3, 80]),
        torch.Size([5, 80]),
    ]


def test_gemma4_pad_and_stack_pads_variable_length_audio_to_batch_max():
    features = [torch.randn(3, 80), torch.randn(5, 80)]
    masks = [
        torch.ones(3, dtype=torch.bool),
        torch.ones(5, dtype=torch.bool),
    ]

    padded_features = _pad_and_stack(features)
    padded_masks = _pad_and_stack(masks)

    assert padded_features.shape == (2, 5, 80)
    assert torch.equal(padded_features[0, :3], features[0])  # real frames kept
    assert torch.all(padded_features[0, 3:] == 0)  # short clip zero-padded
    assert padded_masks.shape == (2, 5)
    assert padded_masks[0].tolist() == [True, True, True, False, False]  # pad=False


@pytest.mark.parametrize(
    "image_width,image_height,max_soft_tokens",
    [
        # Production repro: a 3x900 image (extreme aspect ratio) made the
        # prompt-side estimator return 289 while the HF Gemma 4 image
        # processor's vision tower output capped at 280, producing the
        # "Attempted to assign 280 multimodal tokens to 289 placeholders"
        # mismatch that crashed EngineCore.
        (900, 3, 280),
        (3, 900, 280),
        # Same pathology should hold for the video-frame budget (70 tokens).
        (900, 3, 70),
        # And for any other supported budget.
        (4000, 2, 1120),
    ],
)
@pytest.mark.parametrize("model_id", [GEMMA4_MODEL_ID])
def test_compute_num_soft_tokens_does_not_exceed_max_soft_tokens(
    model_id: str,
    image_width: int,
    image_height: int,
    max_soft_tokens: int,
):
    """Regression for the Gemma 3/4 multimodal crash.

    `_compute_num_soft_tokens` must never return a value larger than
    `max_soft_tokens`. The HF Gemma 4 image processor clamps its vision
    tower output to that value; if the prompt-side estimator returns more,
    the prompt has more `image` placeholder tokens than the encoder will
    fill, and `_merge_multimodal_embeddings` raises `ValueError` deep in
    the model forward.
    """
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
        f"max_soft_tokens={max_soft_tokens} — exceeds the cap that the HF "
        f"image processor enforces on its vision tower output. This is "
        f"the placeholder/encoder count mismatch that crashes EngineCore."
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
@pytest.mark.parametrize("model_id", [GEMMA4_MODEL_ID])
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
@pytest.mark.parametrize("model_id", [GEMMA4_MODEL_ID])
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


@pytest.mark.parametrize("model_id", [GEMMA4_MODEL_ID])
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


@pytest.mark.parametrize("model_id", [GEMMA4_MODEL_ID])
def test_limit_mm_per_prompt(
    image_assets: ImageTestAssets,
    model_id: str,
):
    """Test that limit_mm_per_prompt accurately restricts multiple images."""
    # We only allow 1 image
    ctx = build_model_context(
        model_id,
        mm_processor_kwargs={},
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    # Provide 2 images in the prompt
    prompt = "<image><image>"
    # image_assets usually has multiple images
    images = [asset.pil_image for asset in image_assets][:2]
    if len(images) < 2:
        images = [images[0], images[0]]

    mm_data = {"image": images}

    # Expect ValueError when exceeding limit
    with pytest.raises(ValueError, match="At most 1 image"):
        processor(
            prompt,
            mm_items=processor.info.parse_mm_data(mm_data),
            hf_processor_mm_kwargs={},
        )


# Regression guard for PR #43169 follow-up: the batched Gemma4 vision encoder
# admitted ``chunk ~= 53`` on a 22 GiB L4 with a 26B AWQ model loaded,
# allocating 2.43 GiB int64 inside
# ``F.one_hot(num_classes=position_embedding_size)`` and OOMing because only
# 2.41 GiB was actually free.  The fix sizes ``chunk`` from currently-free GPU
# memory and counts the ``F.one_hot`` transient as the dominant cost.

_encoder_chunk = Gemma4ForConditionalGeneration._encoder_chunk

# Gemma4 vision config default (HF: configuration_gemma4.py).
_POSITION_EMBEDDING_SIZE = 10240
# Video frame: max_soft_tokens=70, pooling_kernel_size=2 -> 70 * 4 patches.
_VIDEO_PATCHES_PER_FRAME = 280


def test_encoder_chunk_tight_budget_fits_in_free():
    free = 3 * GiB_bytes  # L4 22 GiB after 26B AWQ load.
    total = 22 * GiB_bytes
    chunk = _encoder_chunk(
        _VIDEO_PATCHES_PER_FRAME, free, total, _POSITION_EMBEDDING_SIZE
    )
    one_hot_bytes = chunk * _VIDEO_PATCHES_PER_FRAME * 2 * _POSITION_EMBEDDING_SIZE * 8
    assert one_hot_bytes <= free // 2


def test_encoder_chunk_roomy_gpu_keeps_batching():
    chunk = _encoder_chunk(
        _VIDEO_PATCHES_PER_FRAME,
        60 * GiB_bytes,
        80 * GiB_bytes,
        _POSITION_EMBEDDING_SIZE,
    )
    assert chunk > 8


def test_encoder_chunk_zero_patches_is_safe():
    assert (
        _encoder_chunk(0, 60 * GiB_bytes, 80 * GiB_bytes, _POSITION_EMBEDDING_SIZE) == 1
    )


def test_encoder_chunk_zero_position_embedding_size_is_safe():
    # Degenerate config: must not raise ZeroDivisionError.
    assert (
        _encoder_chunk(_VIDEO_PATCHES_PER_FRAME, 60 * GiB_bytes, 80 * GiB_bytes, 0) == 1
    )


def test_encoder_chunk_no_free_memory_falls_back_to_one():
    assert (
        _encoder_chunk(
            _VIDEO_PATCHES_PER_FRAME, 0, 22 * GiB_bytes, _POSITION_EMBEDDING_SIZE
        )
        == 1
    )
