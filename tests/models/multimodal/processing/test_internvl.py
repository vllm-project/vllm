# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for InternVL's multimodal preprocessing kwargs."""

from collections.abc import Mapping

import numpy as np
import pytest
from PIL import Image
from transformers import PreTrainedConfig

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.image import rescale_image_size
from vllm.multimodal.processing import BaseMultiModalProcessor

from ....conftest import ImageTestAssets
from ...utils import build_model_context


def _get_expected_num_patches(
    config: PreTrainedConfig,
    image: Image.Image,
    num_imgs: int,
    min_num: int,
    max_num: int,
):
    from vllm.transformers_utils.processors.internvl import (
        calculate_internvl_targets,
        get_internvl_target_ratios,
    )

    width, height = image.size

    blocks, _, _ = calculate_internvl_targets(
        orig_width=width,
        orig_height=height,
        target_ratios=get_internvl_target_ratios(
            min_num,
            max_num,
        ),
        image_size=config.vision_config.image_size,
        use_thumbnail=False,
    )
    expected_num_patches = blocks

    if config.use_thumbnail and expected_num_patches > 1:
        expected_num_patches += 1

    return expected_num_patches


def _run_check(
    processor: BaseMultiModalProcessor,
    images: list[Image.Image],
    min_num: int,
    max_num: int,
    mm_processor_kwargs: Mapping[str, object],
):
    tokenizer = processor.info.get_tokenizer()
    config = processor.info.get_hf_config()

    prompt = "<image>" * len(images)
    mm_data = {"image": images}

    total_expected_num_patches = sum(
        _get_expected_num_patches(config, image, len(images), min_num, max_num)
        for image in images
    )

    processed_inputs = processor(
        prompt,
        mm_items=processor.info.parse_mm_data(mm_data),
        hf_processor_mm_kwargs=mm_processor_kwargs,
    )

    # Ensure we have the right number of placeholders per num_crops size
    image_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    img_tok_count = processed_inputs["prompt_token_ids"].count(image_token_id)
    pixel_shape = processed_inputs["mm_kwargs"].get_data()["pixel_values_flat"].shape

    assert img_tok_count == 256 * total_expected_num_patches
    assert pixel_shape[0] == total_expected_num_patches


@pytest.mark.parametrize("model_id", ["OpenGVLab/InternVL2-2B"])
@pytest.mark.parametrize(
    "size_factors",
    [
        # Single-scale
        [1.0],
        # Single-scale, batched
        [1.0, 1.0, 1.0],
        # Multi-scale
        [0.25, 0.5, 1.0],
        [4.0, 2.0, 1.0],
    ],
)
@pytest.mark.parametrize(
    ("min_dynamic_patch", "max_dynamic_patch"),
    [(1, 1), (1, 2), (1, 4), (1, 8), (2, 4), (4, 8)],
)
@pytest.mark.parametrize("dynamic_image_size", [True, False])
@pytest.mark.parametrize("kwargs_on_init", [True, False])
def test_processor_override(
    model_id: str,
    image_assets: ImageTestAssets,
    size_factors: list[int],
    min_dynamic_patch: int,
    max_dynamic_patch: int,
    dynamic_image_size: bool | None,
    kwargs_on_init: bool,
):
    mm_processor_kwargs = {
        "min_dynamic_patch": min_dynamic_patch,
        "max_dynamic_patch": max_dynamic_patch,
        "dynamic_image_size": dynamic_image_size,
    }

    ctx = build_model_context(
        model_id,
        mm_processor_kwargs=mm_processor_kwargs if kwargs_on_init else None,
        limit_mm_per_prompt={"image": len(size_factors)},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    hf_processor_mm_kwargs = {} if kwargs_on_init else mm_processor_kwargs

    min_num = min_dynamic_patch if dynamic_image_size else 1
    max_num = max_dynamic_patch if dynamic_image_size else 1

    _run_check(
        processor,
        [rescale_image_size(image_assets[0].pil_image, f) for f in size_factors],
        min_num,
        max_num,
        hf_processor_mm_kwargs,
    )


@pytest.mark.parametrize("model_id", ["OpenGVLab/InternVL3-1B"])
def test_video_placeholder_adjacent_to_text(model_id: str):
    """A video placeholder that directly follows text must survive processing.

    InternVL's chat template emits "<video>" right after the user text. Since
    "<video>" is not a vocabulary token, BPE merges it with the surrounding
    text (".<" and ">\\n"), so the token-level target match fails and the
    text-based fallback is used. The video replacement starts with ordinary
    text ("Frame1: ..."), so the fallback must encode the prompt segments
    separately; otherwise BPE merges across the new boundary (".Frame") and
    the placeholder can no longer be located, which used to fail requests
    with "found 0 prompt placeholders".
    """
    ctx = build_model_context(model_id, limit_mm_per_prompt={"video": 1})
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    tokenizer = processor.info.get_tokenizer()

    # Shaped like the actual chat-template output for a text+video message.
    prompt = (
        "<|im_start|>user\nWhat color is the moving shape? Answer briefly."
        "<video>\n<|im_end|>\n<|im_start|>assistant\n"
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

    # Sanity check: this prompt really exercises the text-based fallback,
    # i.e. the bare target is not a token subsequence of the prompt.
    target = tokenizer.encode("<video>", add_special_tokens=False)
    assert not any(
        prompt_ids[i : i + len(target)] == target
        for i in range(len(prompt_ids) - len(target) + 1)
    )

    video = np.zeros((2, 224, 224, 3), dtype=np.uint8)
    mm_items = processor.info.parse_mm_data({"video": video})

    processed_inputs = processor(
        prompt_ids,
        mm_items=mm_items,
        hf_processor_mm_kwargs={},
    )

    # The video placeholder must be found exactly once, and the video context
    # tokens must all be present in the final prompt.
    video_placeholders = processed_inputs["mm_placeholders"].get("video")
    assert video_placeholders is not None
    assert len(video_placeholders) == 1

    video_token_id = tokenizer.convert_tokens_to_ids("<|video_pad|>")
    num_frames = len(video)
    image_seq_length = processor.info.get_hf_processor().image_seq_length
    video_tok_count = processed_inputs["prompt_token_ids"].count(video_token_id)
    assert video_tok_count == num_frames * image_seq_length
