# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image as PILImage

from vllm.exceptions import VLLMValidationError
from vllm.model_executor.models.gemma4_mm import (
    Gemma4ForConditionalGeneration,
    Gemma4ImagePixelInputs,
    _strip_trailing_padding,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import MultiModalProcessorOnlyCache
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
@pytest.mark.parametrize(
    "image_kwargs", [{"rescale_factor": 1 / 127.5}, {"max_soft_tokens": 560}]
)
@pytest.mark.parametrize("video_uuid", [None, "same-video"])
def test_video_cache_is_independent_of_image_kwargs(
    model_id: str,
    image_kwargs: dict[str, object],
    video_uuid: str | None,
):
    """Request image overrides must not change video frames or make results
    depend on cache warmth.
    """
    request_kwargs = {"images_kwargs": image_kwargs}
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"image": 1, "video": 1},
        mm_processor_cache_gb=1,
    )
    cache = MultiModalProcessorOnlyCache(ctx.model_config)
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    hf_processor = processor.info.get_hf_processor()
    image = PILImage.new("RGB", (48, 48), color=(128, 128, 128))
    frames = np.stack([np.asarray(image)] * 2)
    metadata = {"fps": 2.0, "frames_indices": [0, 1]}
    mm_items = processor.info.parse_mm_data(
        {"image": image, "video": [(frames, metadata)]}
    )

    def process(mm_kwargs, cache):
        return processor(
            hf_processor.image_token + hf_processor.video_token,
            mm_items,
            mm_uuid_items={"video": [video_uuid]},
            hf_processor_mm_kwargs=mm_kwargs,
            cache=cache,
        )

    # Populate the video cache without image overrides, then change only
    # images_kwargs on the next request. Since images_kwargs are excluded from
    # the video hash, and must not affect the processed video, the cached result
    # must match a fresh recomputation.
    baseline = process({}, cache=cache)
    cached = process(request_kwargs, cache=cache)
    cache.clear_cache()
    fresh = process(request_kwargs, cache=cache)

    def pixels(result, modality, field):
        return result["mm_kwargs"][modality][0][field].data

    video_pixels = pixels(fresh, "video", "pixel_values_videos")
    assert torch.equal(pixels(cached, "video", "pixel_values_videos"), video_pixels)
    assert torch.equal(pixels(baseline, "video", "pixel_values_videos"), video_pixels)
    assert baseline["mm_hashes"]["video"] == fresh["mm_hashes"]["video"]
    assert not torch.equal(
        pixels(baseline, "image", "pixel_values"),
        pixels(fresh, "image", "pixel_values"),
    )


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
    with pytest.raises(VLLMValidationError, match="At most 1 image"):
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


def _padded_patches(
    num_valid: int, num_padded: int, width: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Patches laid out like the HF processor output: valid rows first, then
    zero pixels at position (-1, -1)."""
    height = num_valid // width
    pos = torch.stack(
        torch.meshgrid(torch.arange(width), torch.arange(height), indexing="xy"), -1
    ).reshape(-1, 2)
    pixel_values = torch.zeros(num_padded, 3 * 16 * 16, dtype=torch.float64)
    pixel_values[:num_valid] = torch.randn(num_valid, 3 * 16 * 16, dtype=torch.float64)
    position_ids = torch.full((num_padded, 2), -1, dtype=torch.long)
    position_ids[:num_valid] = pos
    return pixel_values, position_ids


@pytest.mark.parametrize("case", ["trailing", "none", "gap"])
def test_strip_trailing_padding(case: str):
    pixel_values, position_ids = _padded_patches(54, 81, width=9)
    if case == "none":
        pixel_values, position_ids = pixel_values[:54], position_ids[:54]
    elif case == "gap":
        position_ids[3] = -1  # padding inside the valid block: leave untouched
    out_pv, out_pp, padded_len = _strip_trailing_padding(pixel_values, position_ids)
    if case == "trailing":
        assert padded_len == 81
        assert torch.equal(out_pv, pixel_values[:54])
        assert torch.equal(out_pp, position_ids[:54])
    else:
        assert padded_len is None
        assert out_pv is pixel_values and out_pp is position_ids


@pytest.mark.parametrize(
    ("num_valid", "free_bytes", "expected_lens"),
    [
        ((54, 81, 63), 0, {54, 81, 63}),
        ((54, 81, 63), 1 << 34, {54, 81, 63}),
        ((54, 54, 54), 1 << 34, {54}),
    ],
)
@torch.inference_mode()
def test_gemma4_image_unpadding_matches_padded_encoding(
    monkeypatch, num_valid, free_bytes, expected_lens
):
    """Unpadding images (which lets SDPA pick FlashAttention) leaves the image
    embeddings unchanged."""
    from transformers import AutoModel
    from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig

    import vllm.model_executor.models.gemma4_mm as gemma4_mm

    torch.manual_seed(0)
    config = Gemma4VisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        position_embedding_size=64,
    )
    config._attn_implementation = "sdpa"
    tower = AutoModel.from_config(config).eval().double()
    encoder_lens: list[int] = []
    encoder_forward = tower.encoder.forward

    def record_encoder(inputs_embeds, **kwargs):
        encoder_lens.append(inputs_embeds.shape[1])
        return encoder_forward(inputs_embeds=inputs_embeds, **kwargs)

    monkeypatch.setattr(tower.encoder, "forward", record_encoder)
    monkeypatch.setattr(
        torch.accelerator, "get_memory_info", lambda: (free_bytes, 1 << 35)
    )
    model = SimpleNamespace(
        vision_tower=tower,
        config=SimpleNamespace(vision_config=config),
        _enable_mm_lora=False,
        _encoder_chunk=Gemma4ForConditionalGeneration._encoder_chunk,
        model_dtype=torch.float64,
        embed_vision=lambda inputs_embeds: inputs_embeds,
    )
    pixel_values, position_ids = zip(*(_padded_patches(n, 81, 9) for n in num_valid))
    image_input = {
        "pixel_values": torch.stack(pixel_values),
        "pixel_position_ids": torch.stack(position_ids),
    }
    process = Gemma4ForConditionalGeneration._process_image_input

    unpadded = process(model, image_input)
    assert set(encoder_lens) == expected_lens
    monkeypatch.setattr(
        gemma4_mm, "_strip_trailing_padding", lambda pv, pp: (pv, pp, None)
    )
    padded = process(model, image_input)
    assert [e.shape for e in unpadded] == [e.shape for e in padded]
    for got, want in zip(unpadded, padded):
        torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)
