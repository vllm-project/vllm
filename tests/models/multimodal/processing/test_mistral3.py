# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Mistral3's multimodal preprocessing kwargs."""

import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, BatchFeature
from transformers.models.pixtral import PixtralProcessor

from vllm.model_executor.layers.fusion.mm_input_norm import (
    FusedMMInputNorm,
    IdentityInputNorm,
    build_mm_input_norm,
)
from vllm.model_executor.models.lightonocr import (
    LightOnOCRForConditionalGeneration,
    LightOnOCRProcessingInfo,
)
from vllm.model_executor.models.mistral3 import Mistral3HFEncoderInfo
from vllm.model_executor.models.pixtral import (
    PixtralHFEncoderInfo,
    pixtral_patch_embed,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalKwargsItems
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

from ...utils import build_model_context

# This repo ships both params.json (Mistral) and config.json (HF). Auto config
# selects PixtralForConditionalGeneration; force HF to exercise Mistral3.
_MODEL_CONFIG_KWARGS = {"config_format": "hf"}
_MODEL_ID = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
_LIGHTON_MODEL_ID = "lightonai/LightOnOCR-1B-1025"

_RGB_MEAN = [0.48145466, 0.4578275, 0.40821073]
_RGB_STD = [0.26862954, 0.26130258, 0.27577711]
_RGB_RESCALE = 1.0 / 255.0


@pytest.mark.usefixtures("default_vllm_config")
@pytest.mark.skipif(
    not HAS_TRITON or not current_platform.is_cuda(), reason="CUDA Triton kernel"
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize(("channels", "patch_shape"), [(3, (4, 4)), (2, (2, 4))])
def test_pixtral_patch_embed_preserves_image_order_and_strides(
    dtype: torch.dtype,
    normalize: bool,
    channels: int,
    patch_shape: tuple[int, int],
):
    """Packed projection matches CHW normalization and patch convolution."""
    device = torch.device(current_platform.device_type)
    norm = (
        FusedMMInputNorm(
            _RGB_MEAN[:channels],
            _RGB_STD[:channels],
            _RGB_RESCALE,
            channel=channels,
        ).to(device)
        if normalize
        else IdentityInputNorm()
    )
    weight = torch.randn(32, channels, *patch_shape, device=device, dtype=dtype)
    images = [
        torch.randint(0, 256, (channels, 17, 18), device=device, dtype=torch.uint8),
        torch.randint(0, 256, (channels, 25, 17), device=device, dtype=torch.uint8)[
            :, 1:17, 1:17
        ],
        torch.randint(0, 256, (channels, 19, 20), device=device, dtype=torch.uint8)[
            :, 1:17, 2:18
        ],
    ]
    reference = []
    for image in images:
        if normalize:
            normalized = (
                image.float() * norm.weight[:, None, None] + norm.bias[:, None, None]
            )
        else:
            normalized = image
        reference.append(
            F.conv2d(normalized.to(dtype).unsqueeze(0), weight, stride=patch_shape)
        )
    packed, actual = pixtral_patch_embed(images, weight, norm)
    for expected, result in zip(reference, actual):
        torch.testing.assert_close(result, expected, rtol=0.01, atol=0.02)
    expected_packed = torch.cat(
        [result.flatten(2).transpose(1, 2) for result in reference], dim=1
    )
    torch.testing.assert_close(packed, expected_packed, rtol=0.01, atol=0.02)


def _process_images_with_hf(
    hf_processor,
    images: list[Image.Image],
    mm_processor_kwargs: dict[str, object],
) -> tuple[list[torch.Tensor], list[int]]:
    """Process images and placeholders through the public HF processor."""
    hf_out = hf_processor(
        text=hf_processor.image_token * len(images),
        images=images,
        return_tensors="pt",
        **mm_processor_kwargs,
    )
    pixel_values = hf_out["pixel_values"]
    image_sizes = hf_out["image_sizes"]
    unpadded = [p[:, :h, :w] for p, (h, w) in zip(pixel_values, image_sizes)]

    special_token_ids = {
        hf_processor.image_token_id,
        hf_processor.image_break_token_id,
        hf_processor.image_end_token_id,
    }
    placeholder_tokens = [
        token_id
        for token_id in hf_out["input_ids"][0].tolist()
        if token_id in special_token_ids
    ]
    return unpadded, placeholder_tokens


def _placeholder_tokens_from_prompt_updates(
    processor,
    images: list[Image.Image],
    pixel_values: list[torch.Tensor],
    mm_processor_kwargs: dict[str, object],
) -> list[int]:
    hf_inputs = BatchFeature({"pixel_values": pixel_values})
    fields_config = processor._get_mm_fields_config(hf_inputs, mm_processor_kwargs)
    out_mm_kwargs = MultiModalKwargsItems.from_hf_inputs(hf_inputs, fields_config)
    # Prompt updates use raw PIL sizes and must predict the processed grid.
    mm_items = processor.info.parse_mm_data({"image": images})
    updates = processor._get_prompt_updates(
        mm_items, mm_processor_kwargs, out_mm_kwargs
    )
    placeholder_tokens: list[int] = []
    for item_idx in range(len(images)):
        details = updates[0].resolve(item_idx).content
        placeholder_tokens.extend(details.full)
    return placeholder_tokens


def _expected_placeholder_tokens_per_image(
    hf_processor,
    pixel_values: torch.Tensor,
) -> int:
    """Count projected tokens from the actual HF-processed H×W."""
    image_h, image_w = pixel_values.shape[-2:]
    patch_size = hf_processor.image_processor.patch_size
    if isinstance(patch_size, dict):
        patch_h = patch_size["height"]
        patch_w = patch_size["width"]
    else:
        patch_h = patch_w = int(patch_size)

    spatial_merge_size = getattr(hf_processor, "spatial_merge_size", 1)
    merged_patch_h = patch_h * spatial_merge_size
    merged_patch_w = patch_w * spatial_merge_size
    assert image_h % merged_patch_h == 0
    assert image_w % merged_patch_w == 0

    return (image_h // merged_patch_h) * (image_w // merged_patch_w)


@pytest.mark.parametrize("model_id", [_MODEL_ID])
@pytest.mark.parametrize(
    ("mm_processor_kwargs", "image_size", "expected_toks_per_img"),
    [
        ({}, (448, 448), 256),
        ({"size": {"longest_edge": 1008}}, (1540, 1540), 1296),
        (
            {"images_kwargs": {"size": {"longest_edge": 1008}}},
            (1540, 1540),
            1296,
        ),
        ({"size": {"longest_edge": 1288}}, (1536, 1187), 1656),
        ({"size": {"longest_edge": 1008}}, (29, 29), 4),
        ({"size": {"longest_edge": 1000}}, (1540, 1700), 1188),
    ],
)
@pytest.mark.parametrize("num_imgs", [1, 2])
@pytest.mark.parametrize("kwargs_on_init", [True, False])
def test_processor_size_override(
    model_id: str,
    mm_processor_kwargs: dict[str, object],
    image_size: tuple[int, int],
    expected_toks_per_img: int,
    num_imgs: int,
    kwargs_on_init: bool,
):
    ctx = build_model_context(
        model_id,
        mm_processor_kwargs=mm_processor_kwargs if kwargs_on_init else None,
        limit_mm_per_prompt={"image": num_imgs},
        model_config_kwargs=_MODEL_CONFIG_KWARGS,
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    hf_processor_mm_kwargs = {} if kwargs_on_init else mm_processor_kwargs
    vllm_hf_processor = processor.info.get_hf_processor(**hf_processor_mm_kwargs)
    assert isinstance(vllm_hf_processor, PixtralProcessor)

    hf_processor = AutoProcessor.from_pretrained(
        model_id,
        fix_mistral_regex=True,
    )

    dummy_image = Image.new("RGB", image_size, color=(127, 127, 127))
    images = [dummy_image] * num_imgs
    merged_mm_kwargs = processor.info.ctx.get_merged_mm_kwargs(hf_processor_mm_kwargs)
    pixel_values, hf_placeholder_tokens = _process_images_with_hf(
        hf_processor, images, merged_mm_kwargs
    )

    prompt_update_tokens = _placeholder_tokens_from_prompt_updates(
        processor, images, pixel_values, hf_processor_mm_kwargs
    )
    expected_from_pixel_values = _expected_placeholder_tokens_per_image(
        hf_processor, pixel_values[0]
    )
    assert expected_from_pixel_values == expected_toks_per_img
    assert hf_placeholder_tokens.count(hf_processor.image_token_id) == (
        expected_from_pixel_values * num_imgs
    )
    assert prompt_update_tokens == hf_placeholder_tokens


@pytest.mark.usefixtures("default_vllm_config")
def test_mm_device_do_normalize():
    ctx = build_model_context(
        _MODEL_ID,
        limit_mm_per_prompt={"image": 2},
        model_config_kwargs=_MODEL_CONFIG_KWARGS,
    )
    assert ctx.model_config.multimodal_config.mm_device_do_normalize

    ctx.model_config.multimodal_config.mm_device_do_normalize = False
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    images = [
        Image.new("RGB", (31, 47), color=(17, 89, 231)),
        Image.new("RGB", (48, 32), color=(201, 13, 127)),
    ]
    prompt = [processor.info.get_hf_config().image_token_index] * len(images)
    mm_items = processor.info.parse_mm_data({"image": images})

    normalized_inputs = processor(prompt, mm_items=mm_items)
    normalized_values = normalized_inputs["mm_kwargs"].get_data()["pixel_values"]

    ctx.model_config.multimodal_config.mm_device_do_normalize = True
    raw_inputs = processor(prompt, mm_items=mm_items)
    raw_values = raw_inputs["mm_kwargs"].get_data()["pixel_values"]
    assert all(value.dtype == torch.uint8 for value in raw_values)

    input_norm = build_mm_input_norm(ctx.model_config)
    patch_size = processor.info.get_hf_config().vision_config.patch_size

    def pack_patches(image: torch.Tensor) -> torch.Tensor:
        channels, height, width = image.shape
        rows = height // patch_size
        cols = width // patch_size
        return (
            image[:, : rows * patch_size, : cols * patch_size]
            .reshape(channels, rows, patch_size, cols, patch_size)
            .permute(1, 3, 0, 2, 4)
            .reshape(-1, channels * patch_size**2)
        )

    for raw, normalized in zip(raw_values, normalized_values):
        output = input_norm(pack_patches(raw), normalized.dtype)
        torch.testing.assert_close(
            output, pack_patches(normalized), rtol=1e-5, atol=1e-6
        )


def test_scoped_request_size_overrides_configured_flat_size():
    ctx = build_model_context(
        _MODEL_ID,
        mm_processor_kwargs={"size": {"longest_edge": 1008}},
        limit_mm_per_prompt={"image": 1},
        model_config_kwargs=_MODEL_CONFIG_KWARGS,
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    hf_processor = AutoProcessor.from_pretrained(
        _MODEL_ID,
        fix_mistral_regex=True,
    )

    hf_processor_mm_kwargs: dict[str, object] = {
        "images_kwargs": {"size": {"longest_edge": 1288}}
    }
    image = Image.new("RGB", (1536, 1187), color=(127, 127, 127))

    merged_mm_kwargs = processor.info.ctx.get_merged_mm_kwargs(hf_processor_mm_kwargs)
    pixel_values, hf_placeholder_tokens = _process_images_with_hf(
        hf_processor, [image], merged_mm_kwargs
    )
    prompt_update_tokens = _placeholder_tokens_from_prompt_updates(
        processor, [image], pixel_values, hf_processor_mm_kwargs
    )

    expected_from_pixel_values = _expected_placeholder_tokens_per_image(
        hf_processor, pixel_values[0]
    )
    assert expected_from_pixel_values == 1656
    assert hf_placeholder_tokens.count(hf_processor.image_token_id) == 1656
    assert prompt_update_tokens == hf_placeholder_tokens


def test_lightonocr_keeps_vision_config_image_size():
    assert not LightOnOCRForConditionalGeneration.supports_mm_device_do_normalize

    ctx = build_model_context(
        _LIGHTON_MODEL_ID,
        mm_processor_kwargs={"size": {"longest_edge": 1008}},
        model_config_kwargs=_MODEL_CONFIG_KWARGS,
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    assert isinstance(processor.info, LightOnOCRProcessingInfo)
    encoder_info = processor.info.get_vision_encoder_info()
    assert isinstance(encoder_info, PixtralHFEncoderInfo)
    assert not isinstance(encoder_info, Mistral3HFEncoderInfo)
    assert encoder_info.get_image_size() == (
        processor.info.get_hf_config().vision_config.image_size
    )
