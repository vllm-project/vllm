# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image as PILImage

from vllm.exceptions import VLLMValidationError
from vllm.model_executor.layers.fusion.mm_input_norm import (
    FusedMMInputNorm,
    build_mm_input_norm,
)
from vllm.model_executor.model_loader import get_model_cls
from vllm.model_executor.models import gemma4_mm
from vllm.model_executor.models.gemma4_mm import (
    Gemma4ForConditionalGeneration,
    Gemma4ImagePixelInputs,
    Gemma4ProcessingInfo,
)
from vllm.model_executor.models.gemma4_vision import Gemma4VisionModel
from vllm.model_executor.models.interfaces import supports_multimodal
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import MultiModalProcessorOnlyCache
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from ....conftest import ImageTestAssets
from ...utils import build_model_context

# TODO: to be updated to "google/gemma-4-e2b-it" once the models are available
GEMMA4_MODEL_ID = "google/gemma-4-E2B-it"
DIFFUSION_GEMMA_MODEL_ID = "google/diffusiongemma-26B-A4B-it"


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
    assert isinstance(processor.info, Gemma4ProcessingInfo)

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
    assert isinstance(processor.info, Gemma4ProcessingInfo)
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


@pytest.mark.parametrize(
    ("model_id", "modality"),
    [
        (GEMMA4_MODEL_ID, "image"),
        (GEMMA4_MODEL_ID, "audio"),
        (GEMMA4_MODEL_ID, "video"),
        (DIFFUSION_GEMMA_MODEL_ID, "image"),
        (DIFFUSION_GEMMA_MODEL_ID, "video"),
    ],
)
def test_placeholder_str_is_replaced(model_id: str, modality: str):
    """With `--chat-template-content-format string`, the chat frontend writes
    `get_placeholder_str` into the prompt text, so the prompt replacement must
    find it there.
    """
    ctx = build_model_context(model_id, limit_mm_per_prompt={modality: 1})
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    model_cls = get_model_cls(ctx.model_config)
    assert supports_multimodal(model_cls)
    image = PILImage.new("RGB", (64, 64))
    mm_data = {
        "image": image,
        "audio": (np.zeros(16000, dtype=np.float32), 16000),
        "video": [
            (
                np.stack([np.asarray(image)] * 2),
                {"fps": 2.0, "frames_indices": [0, 1]},
            )
        ],
    }
    placeholder = model_cls.get_placeholder_str(modality, 1)

    processed = processor(
        f"{placeholder}\nDescribe this.",
        mm_items=processor.info.parse_mm_data({modality: mm_data[modality]}),
        hf_processor_mm_kwargs={},
    )

    assert len(processed["mm_placeholders"][modality]) == 1


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
    # A rescale_factor override only changes pixels when the processor
    # rescales, so test it with device-side normalization off.
    model_config_kwargs = (
        {"mm_device_do_normalize": False} if "rescale_factor" in image_kwargs else None
    )
    ctx = build_model_context(
        model_id,
        model_config_kwargs=model_config_kwargs,
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


@pytest.mark.usefixtures("default_vllm_config")
def test_mm_device_do_normalize():
    device = current_platform.device_type
    ctx = build_model_context(
        GEMMA4_MODEL_ID, limit_mm_per_prompt={"image": 2, "video": 1}
    )
    assert ctx.model_config.multimodal_config.mm_device_do_normalize

    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    hf_processor = processor.info.get_hf_processor()
    rng = np.random.default_rng(0)
    images = [
        PILImage.fromarray(rng.integers(0, 256, (470, 310, 3), dtype=np.uint8)),
        PILImage.fromarray(rng.integers(0, 256, (320, 480, 3), dtype=np.uint8)),
    ]
    frames = rng.integers(0, 256, (2, 240, 320, 3), dtype=np.uint8)
    mm_items = processor.info.parse_mm_data(
        {"image": images, "video": [(frames, {"fps": 2.0, "frames_indices": [0, 1]})]}
    )
    prompt = hf_processor.image_token * len(images) + hf_processor.video_token

    def pixels(mm_device_do_normalize: bool) -> list[torch.Tensor]:
        ctx.model_config.multimodal_config.mm_device_do_normalize = (
            mm_device_do_normalize
        )
        data = processor(prompt, mm_items=mm_items)["mm_kwargs"].get_data()
        return [*data["pixel_values"], *data["pixel_values_videos"]]

    normalized = pixels(False)
    raw = pixels(True)

    input_norm = build_mm_input_norm(ctx.model_config).to(device)
    for raw_values, normalized_values in zip(raw, normalized, strict=True):
        assert raw_values.dtype == torch.uint8
        output = input_norm(raw_values.to(device), normalized_values.dtype)
        torch.testing.assert_close(output, normalized_values.to(device))


class _IdentityTower(torch.nn.Module):
    dtype = torch.float32

    def forward(self, pixel_values, encoder_metadata):
        return pixel_values


@pytest.mark.usefixtures("default_vllm_config")
def test_embed_vision_normalizes_channel_last_patches():
    """Gemma 4 patches are (patch, patch, channel); per-channel parameters must
    reach the right pixels even when they differ across channels."""
    device = current_platform.device_type
    mean, std = [0.1, 0.4, 0.7], [0.5, 0.25, 0.125]
    model = object.__new__(Gemma4ForConditionalGeneration)  # type: ignore[type-abstract]
    torch.nn.Module.__init__(model)
    model.vision_input_norm = FusedMMInputNorm(
        image_mean=mean, image_std=std, rescale_factor=1 / 255
    ).to(device)
    model.config = SimpleNamespace(vision_config=SimpleNamespace(num_channels=3))
    model.vision_tower = _IdentityTower()
    model.embed_vision = torch.nn.Identity()

    pixel_values = torch.randint(
        0, 256, (12, 16 * 16 * 3), dtype=torch.uint8, device=device
    )
    output = model._embed_vision(pixel_values, {})

    expected = (
        pixel_values.view(12, -1, 3) / 255 - torch.tensor(mean, device=device)
    ) / torch.tensor(std, device=device)
    torch.testing.assert_close(output, expected.view(pixel_values.shape))


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


def _gemma4_for_encoder_cudagraph(
    attn_backend: AttentionBackendEnum = AttentionBackendEnum.TRITON_ATTN,
) -> Gemma4ForConditionalGeneration:
    model = object.__new__(Gemma4ForConditionalGeneration)  # type: ignore[type-abstract]
    vision_config = SimpleNamespace(
        patch_size=16,
        pooling_kernel_size=3,
        default_output_length=280,
        position_embedding_size=10240,
        num_attention_heads=12,
        head_dim=64,
    )
    vision_tower = SimpleNamespace(
        config=vision_config,
        attn_backend=attn_backend,
        fp8_padded_hidden_size=None,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    vision_tower.prepare_encoder_metadata = partial(
        Gemma4VisionModel.prepare_encoder_metadata, vision_tower
    )
    model.vision_tower = vision_tower
    model.vision_input_norm = SimpleNamespace(input_dtype=None)
    model.config = SimpleNamespace(
        vision_config=vision_config, text_config=SimpleNamespace(hidden_size=8)
    )
    return model


def _padded_image(
    columns: int, rows: int, max_patches: int = 2520
) -> tuple[torch.Tensor, torch.Tensor]:
    """Patches laid out like the processor's: row-major, padded at the end."""
    grid = torch.meshgrid(torch.arange(columns), torch.arange(rows), indexing="xy")
    positions = torch.full((max_patches, 2), -1, dtype=torch.long)
    positions[: columns * rows] = torch.stack(grid, -1).reshape(-1, 2)
    return torch.zeros(max_patches, 768), positions


@pytest.mark.parametrize(
    ("tower_lora", "expected_calls"), [(False, [200, 106]), (True, [306])]
)
def test_eager_vision_calls_are_bounded(
    monkeypatch, tower_lora: bool, expected_calls: list[int]
):
    """Packed eager calls are capped in soft tokens, bounding the tower's peak
    memory, except with tower LoRA: its mapping covers the whole batch."""
    monkeypatch.setattr(gemma4_mm, "_MAX_SOFT_TOKENS_PER_CALL", 250)
    model = _gemma4_for_encoder_cudagraph()
    model._tower_lora = tower_lora
    calls: list[int] = []

    def embed_vision(pixel_values, metadata):
        calls.append(pixel_values.shape[0] // 9)
        return torch.full((calls[-1], 8), float(len(calls)))

    model._embed_vision = embed_vision  # type: ignore[method-assign]
    images = [_padded_image(30, 30)] * 3 + [_padded_image(9, 6)]
    out = model._process_image_input_local(
        {
            "pixel_values": [pixel_values for pixel_values, _ in images],
            "pixel_position_ids": [positions for _, positions in images],
        }  # type: ignore[arg-type]
    )

    assert calls == expected_calls
    assert [o.shape[0] for o in out] == [100, 100, 100, 6]
    # Outputs keep the input order across calls.
    expected_call_index = [1, 1, 2, 2] if not tower_lora else [1, 1, 1, 1]
    assert [int(o[0, 0]) for o in out] == expected_call_index


@pytest.mark.parametrize("token_budget", [70, 4480])
def test_encoder_cudagraph_capture_buffers_fit_token_budget(token_budget: int):
    # Regression for #61045: buffers sized by max_frames_per_batch rows
    # instead of the token budget took 6.69 GiB at a 4096-token budget.
    model = _gemma4_for_encoder_cudagraph()
    values = model.prepare_encoder_cudagraph_capture_inputs(
        token_budget=token_budget,
        max_batch_size=1,
        max_frames_per_batch=32,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    ).values

    assert values["pixel_values"].shape == (token_budget * 9, 768)
    assert values["pixel_position_ids"].shape == (token_budget * 9, 2)
    assert values["pool_gather_idx"].shape == (token_budget * 9,)
    # A slot per image or frame that fits, each of at least 35 tokens: the
    # manager does not bound frames, e.g. a 64-frame video in 4480 tokens.
    assert values["cu_seqlens"].shape == (max(1, token_budget // 35) + 1,)
    # The largest image of the configured max_soft_tokens (280).
    assert values["max_seqlen"].item() == min(token_budget, 280) * 9


def test_encoder_cudagraph_replay_packs_images_without_padding():
    model = _gemma4_for_encoder_cudagraph()
    images = [
        _padded_image(9, 6),
        _padded_image(60, 42),
        _padded_image(3, 3),
        _padded_image(90, 42, max_patches=3780),
    ]
    mm_kwargs = {
        "pixel_values": [pixel_values for pixel_values, _ in images],
        "pixel_position_ids": [positions for _, positions in images],
    }

    specs = model.get_encoder_cudagraph_item_specs(mm_kwargs)
    assert [spec.output_tokens for spec in specs] == [6, 280, 1, 420]
    # Below the 35 tokens the graphs' sequence slots assume, or above their
    # max_seqlen (configured 280 tokens): eager.
    assert [spec.eager_only for spec in specs] == [True, False, True, True]

    selected = model.select_encoder_cudagraph_items(mm_kwargs, [0, 2])
    values = model.prepare_encoder_cudagraph_replay_buffers(
        selected, max_batch_size=4, max_frames_per_batch=4
    ).values
    pixel_values = values["pixel_values"]
    cu_seqlens = values["cu_seqlens"]
    pool_gather_idx = values["pool_gather_idx"]
    assert pixel_values is not None and pixel_values.shape == (54 + 9, 768)
    assert cu_seqlens is not None and cu_seqlens.tolist() == [0, 54, 63]
    # Each 3x3 block of the 9x6 image, then the 3x3 image.
    assert pool_gather_idx is not None
    assert pool_gather_idx[:9].tolist() == [0, 1, 2, 9, 10, 11, 18, 19, 20]
    assert pool_gather_idx[9:18].tolist() == [3, 4, 5, 12, 13, 14, 21, 22, 23]
    assert pool_gather_idx[54:].tolist() == list(range(54, 63))


def test_encoder_metadata_rejects_grids_not_divisible_by_pooling():
    model = _gemma4_for_encoder_cudagraph()
    _, positions = _padded_image(4, 3)
    with pytest.raises(ValueError, match="multiples of 3"):
        model.vision_tower.prepare_encoder_metadata([positions[:12].numpy()])
