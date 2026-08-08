# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
from functools import partial

import pytest
import torch
from PIL import Image
from pqdm.threads import pqdm
from transformers.models.llava_next.modeling_llava_next import (
    get_anyres_image_grid_shape,
)

from vllm.model_executor.models.llava_next import (
    LlavaNextForConditionalGeneration,
    LlavaNextProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldElem,
    MultiModalKwargsItem,
    MultiModalSharedField,
)
from vllm.multimodal.parse import ImageSize
from vllm.multimodal.processing import BaseMultiModalProcessor

from ...utils import build_model_context


def _validate_image_max_tokens_one(
    processor: BaseMultiModalProcessor,
    max_tokens: int,
    failed_size_excs: list[tuple[ImageSize, Exception]],
    image_size: ImageSize,
) -> None:
    info = processor.info
    feature_size = info.get_num_image_tokens(
        image_width=image_size.width, image_height=image_size.height
    )

    try:
        assert feature_size <= max_tokens, f"{feature_size} <= {max_tokens}"
    except Exception as exc:
        failed_size_excs.append((image_size, exc))


@pytest.mark.skip(
    "This test takes around 5 minutes to run. Comment this out to run it manually."
)
@pytest.mark.parametrize("model_id", ["llava-hf/llava-v1.6-mistral-7b-hf"])
def test_processor_max_tokens(model_id):
    ctx = build_model_context(
        model_id,
        mm_processor_kwargs=None,
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    info = processor.info

    seen_aspect_ratios = set[float]()
    image_sizes = list[ImageSize]()

    # The aspect ratio of the grid layout is between 1 and 2
    # NOTE: Assumes that feature size calculation is the same if we
    # swap the width and height of the image
    for w, h in itertools.product(range(32, 4096), repeat=2):
        aspect_ratio = w / h
        if 1 <= aspect_ratio <= 2 and aspect_ratio not in seen_aspect_ratios:
            image_sizes.append(ImageSize(w, h))
            seen_aspect_ratios.add(aspect_ratio)

    failed_size_excs = list[tuple[ImageSize, Exception]]()

    validate_one = partial(
        _validate_image_max_tokens_one,
        processor,
        info.get_max_image_tokens(),  # type: ignore
        failed_size_excs,
    )
    pqdm(image_sizes, validate_one, n_jobs=8, desc="Validating image sizes")

    if failed_size_excs:
        msg = "Found failing image sizes:" + "\n========\n".join(
            f"[{size}]\n{exc}" for size, exc in failed_size_excs
        )
        raise AssertionError(msg)


def _validate_image_prompt_replacements_one(
    processor: BaseMultiModalProcessor,
    num_imgs: int,
    failed_size_excs: list[tuple[ImageSize, Exception]],
    image_size: ImageSize,
) -> None:
    prompt = "<image>" * num_imgs
    image = Image.new("RGB", size=image_size)
    mm_data = {"image": [image] * num_imgs}

    try:
        # The processor will throw an error if there is a mismatch
        # in the prompt replacements
        processed_inputs = processor(
            prompt,
            mm_items=processor.info.parse_mm_data(mm_data),
            hf_processor_mm_kwargs={},
        )

        image_placeholders = processed_inputs["mm_placeholders"]["image"]
        assert len(image_placeholders) == num_imgs

        first_placeholder = image_placeholders[0]

        # NOTE: There is a BOS token
        assert first_placeholder.offset == 1
        assert (
            first_placeholder.length
            == (len(processed_inputs["prompt_token_ids"]) - 1) // num_imgs
        )

    except Exception as exc:
        failed_size_excs.append((image_size, exc))


def _test_image_prompt_replacements(
    processor,
    *,
    num_imgs: int,
    image_sizes: list[ImageSize],
) -> None:
    """
    Ensure LlavaNextMultiModalProcessor
    handles prompt replacement properly for input images.
    """
    failed_size_excs = list[tuple[ImageSize, Exception]]()

    validate_one = partial(
        _validate_image_prompt_replacements_one,
        processor,
        num_imgs,
        failed_size_excs,
    )
    pqdm(image_sizes, validate_one, n_jobs=8, desc="Validating image sizes")

    if failed_size_excs:
        msg = "Found failing image sizes:" + "\n========\n".join(
            f"[{size}]\n{exc}" for size, exc in failed_size_excs
        )
        raise AssertionError(msg)


@pytest.mark.parametrize("model_id", ["llava-hf/llava-v1.6-mistral-7b-hf"])
@pytest.mark.parametrize("num_imgs", [1, 2])
def test_processor_prompt_replacements_regression(model_id, num_imgs):
    ctx = build_model_context(
        model_id,
        mm_processor_kwargs=None,
        limit_mm_per_prompt={"image": num_imgs},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    image_ratios = [
        (171, 152),
        (184, 161),
        (198, 176),
        (333, 296),
        (369, 328),
        (488, 183),
        (2560, 1669),
    ]
    image_sizes = [
        size for w, h in image_ratios for size in [ImageSize(w, h), ImageSize(h, w)]
    ]

    _test_image_prompt_replacements(
        processor,
        num_imgs=num_imgs,
        image_sizes=image_sizes,
    )


@pytest.mark.skip(
    "This test takes around 2 hours to run. Comment this out to run it manually."
)
@pytest.mark.parametrize("model_id", ["llava-hf/llava-v1.6-mistral-7b-hf"])
@pytest.mark.parametrize("num_imgs", [1])
def test_processor_prompt_replacements_all(model_id, num_imgs):
    ctx = build_model_context(
        model_id,
        mm_processor_kwargs=None,
        limit_mm_per_prompt={"image": num_imgs},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    seen_aspect_ratios = set[float]()
    image_sizes = list[ImageSize]()

    # The aspect ratio of the grid layout is between 1 and 2
    # NOTE: Assumes that feature size calculation is the same if we
    # swap the width and height of the image
    for w, h in itertools.product(range(64, 1024), repeat=2):
        aspect_ratio = w / h
        if 1 <= aspect_ratio <= 2 and aspect_ratio not in seen_aspect_ratios:
            image_sizes.append(ImageSize(w, h))
            seen_aspect_ratios.add(aspect_ratio)

    _test_image_prompt_replacements(
        processor,
        num_imgs=num_imgs,
        image_sizes=image_sizes,
    )


class _StubModel:
    """Carries only the attribute the token helpers read from `self`
    (the HF config, used to look up the vision encoder's patch grid),
    so the real methods can be exercised without constructing the full
    `nn.Module` (vision tower, language model, etc.)."""

    config: object


def _make_mm_data(num_tiles: int, image_size: int) -> MultiModalKwargsItem:
    """Build a `pixel_values`-only `MultiModalKwargsItem` shaped the way
    the real processor produces it: `(num_tiles, C, H, W)`, one tile per
    anyres grid cell plus the base (global) tile."""
    pixel_values = torch.zeros(num_tiles, 3, image_size, image_size)
    return MultiModalKwargsItem(
        {
            "pixel_values": MultiModalFieldElem(
                data=pixel_values, field=MultiModalSharedField(batch_size=1)
            )
        }
    )


@pytest.mark.parametrize("model_id", ["llava-hf/llava-v1.6-mistral-7b-hf"])
@pytest.mark.parametrize(
    ("image_width", "image_height"),
    [
        (672, 672),  # 2x2 grid
        (1024, 768),  # 2x2 grid, different aspect ratio -> different unpad crop
        (768, 1024),
        (1344, 336),  # 1x3 grid
        (336, 1344),  # 3x1 grid (same tile count as 1x3, different final
        # placeholder length after unpad -- this is the
        # non-invertibility case)
        (1000, 500),  # 1x2 grid
        (500, 1000),  # 2x1 grid
    ],
)
def test_num_mm_encoder_tokens_matches_real_tile_count(
    model_id, image_width, image_height
):
    """`get_num_mm_encoder_tokens` must report the *actual* number of
    tokens the vision tower processes for this image (tiles x
    patches-per-tile), not the LLM-side placeholder count. LLaVA-NeXT's
    anyres/unpad tiling makes those two numbers genuinely different
    (unpad crops rows/cols based on aspect ratio and appends newline
    tokens), so this must be computed forward from the real per-item
    `pixel_values` shape via `mm_data`, not backward from
    `num_image_tokens` alone.
    """
    ctx = build_model_context(model_id, limit_mm_per_prompt={"image": 1})
    info = LlavaNextProcessingInfo(ctx)
    hf_config = info.get_hf_config()
    vision_encoder_info = info.get_vision_encoder_info()
    patch_grid_length = vision_encoder_info.get_patch_grid_length()
    image_size = vision_encoder_info.get_image_size()

    num_patch_height, num_patch_width = get_anyres_image_grid_shape(
        (image_height, image_width), hf_config.image_grid_pinpoints, image_size
    )
    num_tiles = num_patch_height * num_patch_width + 1  # +1 base (global) tile
    true_tower_tokens = num_tiles * patch_grid_length**2

    # The final LLM-side placeholder count -- what a naive backward-looking
    # implementation would (incorrectly) treat as the tower token count.
    placeholder_len = info.get_num_image_tokens(
        image_width=image_width, image_height=image_height
    )

    stub = _StubModel()
    stub.config = hf_config
    mm_data = _make_mm_data(num_tiles, image_size)

    get_num_mm_encoder_tokens = (
        LlavaNextForConditionalGeneration.get_num_mm_encoder_tokens
    )
    get_num_mm_connector_tokens = (
        LlavaNextForConditionalGeneration.get_num_mm_connector_tokens
    )

    encoder_tokens = get_num_mm_encoder_tokens(stub, placeholder_len, mm_data=mm_data)
    assert encoder_tokens == true_tower_tokens

    connector_tokens = get_num_mm_connector_tokens(
        stub, encoder_tokens, mm_data=mm_data
    )
    assert connector_tokens == true_tower_tokens


@pytest.mark.parametrize("model_id", ["llava-hf/llava-v1.6-mistral-7b-hf"])
def test_num_mm_encoder_tokens_falls_back_without_mm_data(model_id):
    """Without `mm_data` (e.g. a caller that hasn't been updated to pass
    it through), the helper falls back to treating `num_image_tokens` as
    the token count -- correct only for the single-tile case, but at
    least not a crash."""
    ctx = build_model_context(model_id, limit_mm_per_prompt={"image": 1})
    info = LlavaNextProcessingInfo(ctx)
    hf_config = info.get_hf_config()

    stub = _StubModel()
    stub.config = hf_config

    get_num_mm_encoder_tokens = (
        LlavaNextForConditionalGeneration.get_num_mm_encoder_tokens
    )
    assert get_num_mm_encoder_tokens(stub, 577) == 577
    assert get_num_mm_encoder_tokens(stub, 0, mm_data=None) == 0
