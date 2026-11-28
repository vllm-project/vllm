# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import re
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from vllm.config.multimodal import MultiModalConfig
from vllm.entrypoints.scale_out.token_in_token_out.mm_features import (
    extract_mm_features,
)
from vllm.model_executor.layers.fusion.mm_input_norm import build_mm_input_norm
from vllm.model_executor.models.qwen2_vl import Qwen2VLProcessingInfo
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import MultiModalProcessorOnlyCache
from vllm.multimodal.inputs import batched_tensors_equal
from vllm.multimodal.processing.context import InputProcessingContext, TimingContext
from vllm.multimodal.processing.inputs import ProcessorInputs
from vllm.platforms import current_platform

from ....conftest import ImageTestAssets
from ...utils import build_model_context


@pytest.mark.parametrize(
    ("mm_kwargs", "expected"),
    [
        (
            {"size": {"shortest_edge": 64, "longest_edge": 1024}},
            {
                "size": {"shortest_edge": 64, "longest_edge": 1024},
                "min_pixels": 64,
                "max_pixels": 1024,
            },
        ),
        (
            {"min_pixels": 64, "max_pixels": 1024},
            {
                "min_pixels": 64,
                "max_pixels": 1024,
                "size": {"shortest_edge": 64, "longest_edge": 1024},
            },
        ),
        (
            {
                "size": {"shortest_edge": 64, "longest_edge": 1024},
                "min_pixels": 128,
                "max_pixels": 2048,
            },
            {
                "size": {"shortest_edge": 64, "longest_edge": 1024},
                "min_pixels": 128,
                "max_pixels": 2048,
            },
        ),
        (
            {
                "min_pixels": None,
                "size": {"shortest_edge": 0},
                "images_kwargs": {"max_pixels": 4096},
                "videos_kwargs": {"size": {"longest_edge": 8192}},
            },
            {
                "min_pixels": None,
                "size": {"shortest_edge": 0},
                "images_kwargs": {
                    "max_pixels": 4096,
                    "size": {"longest_edge": 4096},
                },
                "videos_kwargs": {
                    "size": {"longest_edge": 8192},
                    "max_pixels": 8192,
                },
            },
        ),
        (
            {"min_pixels": 0},
            {"min_pixels": 0, "size": {"shortest_edge": 0}},
        ),
        (
            {"size": None, "min_pixels": 64},
            {"size": {"shortest_edge": 64}, "min_pixels": 64},
        ),
    ],
)
def test_complete_mm_processor_size_aliases(
    mm_kwargs: dict[str, object],
    expected: dict[str, object],
) -> None:
    original = deepcopy(mm_kwargs)

    actual = Qwen2VLProcessingInfo._complete_mm_processor_size_aliases(mm_kwargs)

    assert actual == expected
    assert mm_kwargs == original


def test_merge_and_resolve_mm_processor_kwargs_preserves_request_alias_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mm_config = MultiModalConfig(
        mm_processor_kwargs={
            "images_kwargs": {"min_pixels": 64},
        },
        mm_device_do_normalize=False,
    )
    model_config = SimpleNamespace(get_multimodal_config=lambda: mm_config)
    ctx = InputProcessingContext(model_config, tokenizer=None)
    info = Qwen2VLProcessingInfo(ctx)
    monkeypatch.setattr(
        info,
        "get_supported_mm_processor_kwargs",
        lambda: {"images_kwargs": {"size", "min_pixels"}},
    )

    assert info._merge_and_resolve_mm_processor_kwargs(
        {"size": {"shortest_edge": 128}}
    ) == {
        "images_kwargs": {
            "size": {"shortest_edge": 128},
            "min_pixels": 128,
        },
    }


@pytest.mark.parametrize(
    ("mm_kwargs", "default_size", "expected"),
    [
        (
            {},
            {"shortest_edge": 32, "longest_edge": 1024},
            {"shortest_edge": 32, "longest_edge": 1024},
        ),
        (
            {"size": {"shortest_edge": 64}},
            {"shortest_edge": 32, "longest_edge": 1024},
            {"shortest_edge": 64, "longest_edge": 1024},
        ),
        (
            {
                "size": {"shortest_edge": 64, "longest_edge": 2048},
                "min_pixels": 128,
                "max_pixels": 4096,
            },
            {"shortest_edge": 32, "longest_edge": 1024},
            {"shortest_edge": 128, "longest_edge": 4096},
        ),
        (
            {
                "size": {"shortest_edge": 64},
                "max_pixels": 4096,
            },
            None,
            {"shortest_edge": 64, "longest_edge": 4096},
        ),
        (
            {
                "size": {"shortest_edge": 64, "longest_edge": 2048},
                "min_pixels": None,
                "max_pixels": 0,
            },
            {"shortest_edge": 32, "longest_edge": 1024},
            {"shortest_edge": 64, "longest_edge": 0},
        ),
    ],
)
def test_get_vision_size(
    mm_kwargs: dict[str, object],
    default_size: dict[str, object] | None,
    expected: dict[str, object],
) -> None:
    original_mm_kwargs = {
        key: dict(value) if isinstance(value, dict) else value
        for key, value in mm_kwargs.items()
    }
    original_default_size = dict(default_size) if default_size is not None else None

    actual = Qwen2VLProcessingInfo._get_vision_size(
        mm_kwargs,
        default_size=default_size,
    )

    assert actual == expected
    assert mm_kwargs == original_mm_kwargs
    assert default_size == original_default_size


@pytest.mark.parametrize(
    ("modality", "scope"),
    [
        ("image", "images_kwargs"),
        ("video", "videos_kwargs"),
    ],
)
def test_get_vision_info_reads_merged_modality_scope(
    monkeypatch: pytest.MonkeyPatch,
    modality: str,
    scope: str,
) -> None:
    vision_config = SimpleNamespace(
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
    )
    ctx = SimpleNamespace(
        get_hf_config=lambda *_: SimpleNamespace(vision_config=vision_config)
    )
    info = Qwen2VLProcessingInfo(ctx)
    merged = {
        "images_kwargs": {"size": {"shortest_edge": 111}},
        "videos_kwargs": {"size": {"shortest_edge": 222}},
    }
    monkeypatch.setattr(
        info,
        "_merge_and_resolve_mm_processor_kwargs",
        lambda _: merged,
    )

    seen: list[dict[str, object]] = []

    def get_size(
        mm_kwargs: dict[str, object],
        default_size: dict[str, object] | None = None,
    ) -> dict[str, object]:
        seen.append(mm_kwargs)
        return dict(default_size or {})

    monkeypatch.setattr(info, "_get_vision_size", get_size)
    image_processor = SimpleNamespace(size={"shortest_edge": 32, "longest_edge": 1024})

    info._get_vision_info(
        image_width=28,
        image_height=28,
        num_frames=2,
        do_resize=False,
        image_processor=image_processor,
        mm_kwargs={},
        modality=modality,
    )

    assert seen == [merged[scope]]


def test_jina_vl_processing_order() -> None:
    """Jina's document-first prompt keeps cached features and hashes aligned."""
    ctx = build_model_context(
        "jinaai/jina-reranker-m0",
        runner="pooling",
        limit_mm_per_prompt={"image": 2},
        mm_processor_cache_gb=1,
    )
    cache = MultiModalProcessorOnlyCache(ctx.model_config)
    processor = MULTIMODAL_REGISTRY.create_processor(
        ctx.model_config,
        tokenizer=ctx.tokenizer,
    )

    placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
    query_image = Image.new("RGB", (128, 160), color=(255, 0, 0))
    document_image = Image.new("RGB", (192, 128), color=(0, 255, 0))

    def process(images: list[Image.Image]):
        return processor(
            placeholder * len(images),
            mm_items=processor.info.parse_mm_data({"image": images}),
            cache=cache,
        )

    query = process([query_image])
    document = process([document_image])
    pair = process([query_image, document_image])

    pair_items = pair["mm_kwargs"]["image"]
    assert pair["mm_hashes"]["image"] == [
        document["mm_hashes"]["image"][0],
        query["mm_hashes"]["image"][0],
    ]
    assert batched_tensors_equal(
        pair_items[0].get_data(),
        document["mm_kwargs"]["image"][0].get_data(),
    )
    assert batched_tensors_equal(
        pair_items[1].get_data(),
        query["mm_kwargs"]["image"][0].get_data(),
    )
    assert [item.length for item in pair["mm_placeholders"]["image"]] == [
        document["mm_placeholders"]["image"][0].length,
        query["mm_placeholders"]["image"][0].length,
    ]


@pytest.mark.parametrize("model_id", ["Qwen/Qwen2-VL-2B-Instruct"])
@pytest.mark.parametrize(
    ("mm_processor_kwargs", "expected_toks_per_img", "expected_pixels_shape"),
    [
        ({}, 1426, (5704, 1176)),
        ({"min_pixels": 64**2, "max_pixels": 512**2}, 330, (1320, 1176)),
        (
            {
                "size": {
                    "shortest_edge": 64**2,
                    "longest_edge": 512**2,
                },
            },
            330,
            (1320, 1176),
        ),
    ],
)
@pytest.mark.parametrize("num_imgs", [1, 2])
@pytest.mark.parametrize("kwargs_on_init", [True, False])
def test_processor_override(
    image_assets: ImageTestAssets,
    model_id: str,
    mm_processor_kwargs: dict[str, object],
    expected_toks_per_img: int,
    expected_pixels_shape: tuple[int, int],
    num_imgs: int,
    kwargs_on_init: bool,
):
    """Ensure Qwen2VLMultiModalProcessor handles min/max pixels properly."""
    ctx = build_model_context(
        model_id,
        mm_processor_kwargs=mm_processor_kwargs if kwargs_on_init else None,
        limit_mm_per_prompt={"image": num_imgs},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    tokenizer = processor.info.get_tokenizer()
    hf_processor_mm_kwargs = {} if kwargs_on_init else mm_processor_kwargs

    # Build the image str / prompt based on the number of images we pass
    prompt = "<|vision_start|><|image_pad|><|vision_end|>" * num_imgs
    mm_data = {"image": [image_assets[0].pil_image] * num_imgs}

    processed_inputs = processor(
        prompt,
        mm_items=processor.info.parse_mm_data(mm_data),
        hf_processor_mm_kwargs=hf_processor_mm_kwargs,
    )

    # Ensure we have the right number of placeholders per num_crops size
    hf_processor = processor.info.get_hf_processor(**hf_processor_mm_kwargs)
    image_token_id = tokenizer.convert_tokens_to_ids(hf_processor.image_token)
    img_tok_count = processed_inputs["prompt_token_ids"].count(image_token_id)
    pixel_values = processed_inputs["mm_kwargs"].get_data()["pixel_values"]
    assert isinstance(pixel_values, torch.Tensor)
    pixel_shape = pixel_values.shape

    assert img_tok_count == expected_toks_per_img * num_imgs
    assert pixel_shape[0] == expected_pixels_shape[0] * num_imgs
    assert pixel_shape[1] == expected_pixels_shape[1]


@pytest.mark.parametrize("model_id", ["Qwen/Qwen2-VL-2B-Instruct"])
@pytest.mark.parametrize(
    "mm_processor_kwargs",
    [
        {"min_pixels": 28 * 28, "max_pixels": 1280 * 28 * 28},
        {"min_pixels": 28 * 28, "max_pixels": 1283 * 28 * 28},
        {"size": {"shortest_edge": 28 * 28, "longest_edge": 1280 * 28 * 28}},
        {"size": {"shortest_edge": 28 * 28, "longest_edge": 1283 * 28 * 28}},
    ],
)
def test_get_image_size_with_most_features(
    image_assets: ImageTestAssets,
    model_id: str,
    mm_processor_kwargs: dict[str, object],
):
    ctx = build_model_context(
        model_id,
        mm_processor_kwargs=mm_processor_kwargs,
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    assert isinstance(processor.info, Qwen2VLProcessingInfo)

    hf_processor = processor.info.get_hf_processor(**mm_processor_kwargs)
    merge_size = processor.info.get_hf_config().vision_config.spatial_merge_size

    max_image_size = processor.info.get_image_size_with_most_features()
    max_tokens = processor.info.get_num_image_tokens(
        image_width=max_image_size.width,
        image_height=max_image_size.height,
        image_processor=hf_processor.image_processor,
        mm_kwargs=mm_processor_kwargs,
    )

    prompt = "<|vision_start|><|image_pad|><|vision_end|>"
    for asset in image_assets:
        mm_data = {"image": [asset.pil_image]}
        processed_inputs = processor(
            prompt,
            mm_items=processor.info.parse_mm_data(mm_data),
            hf_processor_mm_kwargs=mm_processor_kwargs,
        )
        image_grid_thw = processed_inputs["mm_kwargs"].get_data()["image_grid_thw"]
        assert isinstance(image_grid_thw, torch.Tensor)
        grid_thw = image_grid_thw.tolist()
        t, h, w = grid_thw[0]
        tokens = (t * h * w) // (merge_size**2)
        assert tokens < max_tokens


@pytest.mark.parametrize(
    ("model_id", "max_model_len", "mm_processor_kwargs"),
    [
        ("Qwen/Qwen2-VL-2B-Instruct", 20000, None),
        ("Qwen/Qwen2.5-VL-3B-Instruct", 20000, None),
        ("Qwen/Qwen2.5-VL-3B-Instruct", 20000, {"max_pixels": 4096 * 28 * 28}),
        (
            "Qwen/Qwen2.5-VL-3B-Instruct",
            20000,
            {
                "videos_kwargs": {
                    "size": {
                        "shortest_edge": 4 * 28 * 28,
                        "longest_edge": 4096 * 28 * 28,
                    }
                }
            },
        ),
        ("Qwen/Qwen2.5-VL-3B-Instruct", 20000, {"videos_kwargs": {"do_resize": False}}),
        ("LGAI-EXAONE/EXAONE-4.5-33B", 20000, None),
    ],
)
def test_max_video_tokens_fills_context(
    model_id: str,
    max_model_len: int,
    mm_processor_kwargs: dict[str, object] | None,
) -> None:
    ctx = build_model_context(
        model_id,
        mm_processor_kwargs=mm_processor_kwargs,
        model_config_kwargs={"max_model_len": max_model_len},
        limit_mm_per_prompt={"image": 0, "video": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    assert isinstance(processor.info, Qwen2VLProcessingInfo)

    max_video_tokens = processor.info.get_mm_max_tokens_per_item(
        max_model_len, {"video": 1}
    )["video"]
    dummy_inputs = processor.get_dummy_mm_inputs({"video": 1})
    dummy_video_tokens = sum(
        placeholder.get_num_embeds()
        for placeholder in dummy_inputs["mm_placeholders"]["video"]
    )

    assert max_video_tokens == max_model_len
    assert dummy_video_tokens == max_video_tokens


def _count_video_tokens(processor_inputs) -> int:
    return sum(
        placeholder.get_num_embeds()
        for placeholder in processor_inputs["mm_placeholders"]["video"]
    )


def test_dummy_video_follows_size_hint() -> None:
    width, height, num_frames = 1400, 2800, 2
    ctx = build_model_context(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        model_config_kwargs={"max_model_len": 40000},
        limit_mm_per_prompt={
            "image": 0,
            "video": {
                "count": 1,
                "width": width,
                "height": height,
                "num_frames": num_frames,
            },
        },
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    video = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
    metadata = {
        "fps": 2.0,
        "duration": num_frames / 2.0,
        "total_num_frames": num_frames,
        "frames_indices": list(range(num_frames)),
        "video_backend": "opencv",
        "do_sample_frames": False,
    }
    processed = processor(
        "<|vision_start|><|video_pad|><|vision_end|>",
        mm_items=processor.info.parse_mm_data({"video": [(video, metadata)]}),
        hf_processor_mm_kwargs={},
    )

    assert _count_video_tokens(
        processor.get_dummy_mm_inputs({"video": 1})
    ) == _count_video_tokens(processed)


@pytest.mark.parametrize(
    ("max_model_len", "mm_processor_kwargs", "video_options"),
    [
        (40000, None, {"count": 1, "width": 3584, "height": 3584}),
        (40000, None, {"count": 1, "num_frames": 4}),
        (5000, {"min_pixels": 4096 * 28 * 28, "max_pixels": 4096 * 28 * 28}, 1),
    ],
)
def test_max_video_tokens_matches_dummy_video(
    max_model_len: int,
    mm_processor_kwargs: dict[str, object] | None,
    video_options: dict[str, int] | int,
) -> None:
    ctx = build_model_context(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        mm_processor_kwargs=mm_processor_kwargs,
        model_config_kwargs={"max_model_len": max_model_len},
        limit_mm_per_prompt={"image": 0, "video": video_options},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    assert isinstance(processor.info, Qwen2VLProcessingInfo)

    max_video_tokens = processor.info.get_mm_max_tokens_per_item(
        max_model_len, {"video": 1}
    )["video"]

    assert max_video_tokens <= max_model_len
    assert (
        _count_video_tokens(processor.get_dummy_mm_inputs({"video": 1}))
        == max_video_tokens
    )


def test_max_video_tokens_with_pruning_keeps_full_size_frames() -> None:
    max_model_len = 40000
    ctx = build_model_context(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        model_config_kwargs={"max_model_len": max_model_len},
        limit_mm_per_prompt={"image": 0, "video": 1},
    )
    ctx.model_config.multimodal_config.video_pruning_rate = 0.75
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    info = processor.info
    assert isinstance(info, Qwen2VLProcessingInfo)

    max_size = info.get_image_size_with_most_features()
    assert info.get_max_video_tokens(
        max_model_len, {"video": 1}
    ) == info.get_num_video_tokens(
        image_width=max_size.width,
        image_height=max_size.height,
        num_frames=info.get_num_frames_with_most_features(max_model_len, 1),
        image_processor=info.get_image_processor(),
        mm_kwargs={},
    )


def _build_qwen2_5_vl_video_mm_data(num_frames: int, fps: float) -> dict[str, Any]:
    video = np.zeros((num_frames, 56, 56, 3), dtype=np.uint8)
    metadata = {
        "fps": fps,
        "duration": num_frames / fps,
        "total_num_frames": num_frames,
        "frames_indices": list(range(num_frames)),
        "video_backend": "opencv",
        "do_sample_frames": False,
    }
    return {"video": [(video, metadata)]}


@pytest.mark.parametrize("model_id", ["Qwen/Qwen2.5-VL-3B-Instruct"])
def test_qwen2_5_vl_video_fps_to_second_per_grid_ts(model_id: str) -> None:
    ctx = build_model_context(model_id, limit_mm_per_prompt={"image": 0, "video": 1})
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    temporal_patch_size = (
        processor.info.get_hf_processor().video_processor.temporal_patch_size
    )

    prompt = "<|vision_start|><|video_pad|><|vision_end|>"
    for fps in (1.0, 30.0):
        mm_data = _build_qwen2_5_vl_video_mm_data(num_frames=32, fps=fps)
        processed = processor(prompt, mm_items=processor.info.parse_mm_data(mm_data))
        item = processed["mm_kwargs"]["video"][0]
        assert item is not None
        spg = item.get_data()["second_per_grid_ts"]
        assert float(spg) == pytest.approx(temporal_patch_size / fps, rel=2e-2)


@pytest.mark.usefixtures("default_vllm_config")
@pytest.mark.parametrize(
    "model_id", ["Qwen/Qwen2-VL-2B-Instruct", "Qwen/Qwen2.5-VL-3B-Instruct"]
)
@pytest.mark.parametrize("num_imgs", [1, 2])
def test_mm_device_do_normalize(
    image_assets: ImageTestAssets, model_id: str, num_imgs: int
) -> None:
    """Device-side normalisation must reproduce the on-CPU processor result.

    Runs on any platform: the CPU platform exercises the `forward_native`
    fallback, accelerators exercise the fused kernel.
    """
    device = current_platform.device_type
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"image": num_imgs},
    )
    ctx.model_config.multimodal_config.mm_device_do_normalize = False
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    # Build the image str / prompt based on the number of images we pass
    prompt = "<|vision_start|><|image_pad|><|vision_end|>" * num_imgs
    mm_data = {"image": [image_assets[0].pil_image] * num_imgs}

    processed_inputs_with_normalize = processor(
        prompt,
        mm_items=processor.info.parse_mm_data(mm_data),
    )
    pixel_values_with_normalize = processed_inputs_with_normalize[
        "mm_kwargs"
    ].get_data()["pixel_values"]
    assert isinstance(pixel_values_with_normalize, torch.Tensor)
    dtype = pixel_values_with_normalize.dtype

    processed_inputs_without_normalize = processor(
        prompt,
        mm_items=processor.info.parse_mm_data(mm_data),
        hf_processor_mm_kwargs={"do_normalize": False, "do_rescale": False},
    )
    pixel_values_without_normalize = processed_inputs_without_normalize[
        "mm_kwargs"
    ].get_data()["pixel_values"]
    assert isinstance(pixel_values_without_normalize, torch.Tensor)

    ctx.model_config.multimodal_config.mm_device_do_normalize = True
    input_norm = build_mm_input_norm(ctx.model_config).to(device)

    # With normalisation disabled, the processor emits raw uint8 pixels,
    # matching the production mm_device_do_normalize path.
    assert pixel_values_without_normalize.dtype == torch.uint8
    pixel_values_do_input_norm = input_norm(
        pixel_values_without_normalize.to(device), dtype
    )

    torch.testing.assert_close(
        pixel_values_with_normalize.to(device), pixel_values_do_input_norm
    )


def _process_images(
    processor,
    images: list[Image.Image | None],
    mm_processor_kwargs: dict[str, object],
    *,
    metadata_only: bool,
    cache: MultiModalProcessorOnlyCache | None = None,
    uuids: list[str | None] | None = None,
):
    tokenizer = processor.info.get_tokenizer()
    placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
    prompt = tokenizer.encode(f"<|im_start|>user\n{placeholder * len(images)}hi")
    return processor.apply(
        ProcessorInputs(
            prompt,
            processor.info.parse_mm_data({"image": images}),
            mm_uuid_items=None if uuids is None else {"image": uuids},
            hf_processor_mm_kwargs=mm_processor_kwargs,
            cache=cache,
            metadata_only=metadata_only,
        ),
        TimingContext(),
    )


def _rendered_features(engine_input):
    return extract_mm_features(engine_input, mm_payload="metadata")


def _process_outcome(processor, images, mm_processor_kwargs, *, metadata_only):
    # Some Transformers versions accept an input that others reject.
    # Callers compare the two paths instead of assuming one outcome.
    try:
        engine_input = _process_images(
            processor, images, mm_processor_kwargs, metadata_only=metadata_only
        )
    except Exception as exc:
        # The message repeats the input, including the object address of each image.
        return None, (type(exc), re.sub(r" at 0x[0-9a-fA-F]+", "", str(exc)))
    return engine_input, None


@pytest.mark.parametrize(
    "model_id",
    [
        "Qwen/Qwen2-VL-2B-Instruct",
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "Qwen/Qwen3-VL-4B-Instruct",
    ],
)
def test_metadata_only_matches_full_processing(
    image_assets: ImageTestAssets,
    model_id: str,
):
    """Metadata-only processing skips pixel_values and renders identically."""
    ctx = build_model_context(model_id, limit_mm_per_prompt={"image": 2})
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    images = [image_assets[0].pil_image, image_assets[1].pil_image.resize((300, 900))]

    full = _process_images(processor, images, {}, metadata_only=False)
    meta = _process_images(processor, images, {}, metadata_only=True)

    assert all("pixel_values" in item for item in full["mm_kwargs"]["image"])
    assert not any("pixel_values" in item for item in meta["mm_kwargs"]["image"])
    assert meta["prompt_token_ids"] == full["prompt_token_ids"]
    assert meta["mm_hashes"] == full["mm_hashes"]
    assert meta["mm_placeholders"] == full["mm_placeholders"]
    # The serialized metadata (image_grid_thw) must match byte for byte.
    assert _rendered_features(meta) == _rendered_features(full)


@pytest.mark.parametrize(
    ("image_size", "mm_processor_kwargs"),
    [
        ((1120, 672), {"do_resize": False}),
        ((640, 480), {"images_kwargs": {"patch_size": 14}}),
        ((640, 480), {"images_kwargs": {"image_mean": [0.5, 0.5, 0.5]}}),
        # Transformers versions resolve call-time size options differently.
        ((896, 896), {"max_pixels": 512**2}),
        ((640, 480), {"images_kwargs": {"min_pixels": 1024**2}}),
        (
            (896, 896),
            {
                "images_kwargs": {
                    "size": {"shortest_edge": 3136, "longest_edge": 1024**2},
                    "max_pixels": 512**2,
                }
            },
        ),
    ],
)
def test_metadata_only_falls_back_for_unmodeled_options(
    image_size: tuple[int, int],
    mm_processor_kwargs: dict[str, object],
):
    """Options that change the grid and are not modeled run the HF processor."""
    ctx = build_model_context(
        "Qwen/Qwen2.5-VL-3B-Instruct", limit_mm_per_prompt={"image": 1}
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    images = [Image.new("RGB", image_size, color=(0, 128, 255))]

    full = _process_images(processor, images, mm_processor_kwargs, metadata_only=False)
    meta = _process_images(processor, images, mm_processor_kwargs, metadata_only=True)

    assert "pixel_values" in meta["mm_kwargs"]["image"][0]
    assert meta["prompt_token_ids"] == full["prompt_token_ids"]
    assert _rendered_features(meta) == _rendered_features(full)


@pytest.mark.parametrize("metadata_first", [True, False])
def test_metadata_only_results_do_not_enter_the_cache(metadata_first: bool):
    """A tensor request after a metadata-only one still gets pixel_values."""
    ctx = build_model_context(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        limit_mm_per_prompt={"image": 1},
        mm_processor_cache_gb=1,
    )
    cache = MultiModalProcessorOnlyCache(ctx.model_config)
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    images = [Image.new("RGB", (640, 480), color=(10, 200, 30))]

    order = [True, False] if metadata_first else [False, True]
    results = {
        metadata_only: _process_images(
            processor, images, {}, metadata_only=metadata_only, cache=cache
        )
        for metadata_only in order
    }

    assert "pixel_values" in results[False]["mm_kwargs"]["image"][0]
    assert "pixel_values" not in results[True]["mm_kwargs"]["image"][0]
    assert results[True]["mm_hashes"] == results[False]["mm_hashes"]


@pytest.mark.parametrize(
    ("attribute", "value"),
    [("do_resize", False), ("input_data_format", "channels_last")],
)
def test_metadata_only_falls_back_for_saved_processor_settings(
    monkeypatch: pytest.MonkeyPatch,
    attribute: str,
    value: object,
):
    """Saved settings that change how HF sizes the image make HF build the grids."""
    ctx = build_model_context(
        "Qwen/Qwen2.5-VL-3B-Instruct", limit_mm_per_prompt={"image": 1}
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    info = processor.info
    assert isinstance(info, Qwen2VLProcessingInfo)
    image_processor = info.get_image_processor()
    assert info.get_image_processor() is image_processor
    monkeypatch.setattr(image_processor, attribute, value)
    # Patch-aligned and above the default max_pixels, so resizing changes the grid.
    images = [Image.new("RGB", (4480, 3136), color=(0, 128, 255))]

    full, full_error = _process_outcome(processor, images, {}, metadata_only=False)
    meta, meta_error = _process_outcome(processor, images, {}, metadata_only=True)

    # Some Transformers versions reject this layout for the image.
    assert meta_error == full_error
    if full_error is not None:
        return

    assert "pixel_values" in meta["mm_kwargs"]["image"][0]
    assert _rendered_features(meta) == _rendered_features(full)


@pytest.mark.parametrize("mode", ["L", "RGBA"])
def test_metadata_only_falls_back_without_rgb_conversion(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
):
    """Without RGB conversion, HF rejects a non-RGB image, so render must too."""
    ctx = build_model_context(
        "Qwen/Qwen2.5-VL-3B-Instruct", limit_mm_per_prompt={"image": 1}
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    info = processor.info
    assert isinstance(info, Qwen2VLProcessingInfo)
    monkeypatch.setattr(info.get_image_processor(), "do_convert_rgb", False)
    images = [Image.new(mode, (280, 280))]

    full, full_error = _process_outcome(processor, images, {}, metadata_only=False)
    meta, meta_error = _process_outcome(processor, images, {}, metadata_only=True)

    assert meta_error == full_error
    if full_error is not None:
        return

    assert "pixel_values" in meta["mm_kwargs"]["image"][0]
    assert _rendered_features(meta) == _rendered_features(full)


def test_metadata_only_keeps_uuid_reuse_through_the_cache():
    """A metadata-only request may name a cached image by UUID alone."""
    ctx = build_model_context(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        limit_mm_per_prompt={"image": 1},
        mm_processor_cache_gb=1,
    )
    cache = MultiModalProcessorOnlyCache(ctx.model_config)
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    image = Image.new("RGB", (640, 480), color=(10, 200, 30))

    # A metadata-only request with a user UUID still fills the cache.
    first = _process_images(
        processor, [image], {}, metadata_only=True, cache=cache, uuids=["img-1"]
    )
    reused = _process_images(
        processor, [None], {}, metadata_only=True, cache=cache, uuids=["img-1"]
    )

    assert "pixel_values" in first["mm_kwargs"]["image"][0]
    assert reused["mm_hashes"] == first["mm_hashes"]
    assert reused["prompt_token_ids"] == first["prompt_token_ids"]
    assert _rendered_features(reused) == _rendered_features(first)


def test_metadata_only_keeps_the_fast_path_for_uuids_without_a_cache():
    """Without a cache, UUIDs cannot be reused, so they do not force the full path."""
    ctx = build_model_context(
        "Qwen/Qwen2.5-VL-3B-Instruct", limit_mm_per_prompt={"image": 1}
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    images = [Image.new("RGB", (640, 480), color=(10, 200, 30))]

    full = _process_images(processor, images, {}, metadata_only=False, uuids=["r-0"])
    meta = _process_images(processor, images, {}, metadata_only=True, uuids=["r-0"])

    assert "pixel_values" not in meta["mm_kwargs"]["image"][0]
    assert meta["mm_hashes"] == full["mm_hashes"]
    assert _rendered_features(meta) == _rendered_features(full)


def test_metadata_only_falls_back_for_tensor_images():
    """HF reads a (3, H, 3) tensor as CHW, so tensors run the HF processor."""
    ctx = build_model_context(
        "Qwen/Qwen2.5-VL-3B-Instruct", limit_mm_per_prompt={"image": 1}
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    images = [torch.zeros(3, 280, 3, dtype=torch.uint8)]

    full = _process_images(processor, images, {}, metadata_only=False)
    meta = _process_images(processor, images, {}, metadata_only=True)

    assert "pixel_values" in meta["mm_kwargs"]["image"][0]
    assert _rendered_features(meta) == _rendered_features(full)


@pytest.mark.parametrize(
    "mm_processor_kwargs",
    [
        {"max_pixels": 0},
        {"images_kwargs": {"size": {"shortest_edge": 0, "longest_edge": 1024**2}}},
        {"images_kwargs": {"do_normalize": "invalid"}},
        {"common_kwargs": {"size": {"shortest_edge": 3136, "longest_edge": 0}}},
        {"videos_kwargs": {"do_resize": "invalid"}},
        {"images_kwargs": []},
    ],
)
def test_metadata_only_keeps_the_hf_result_for_invalid_options(
    mm_processor_kwargs: dict[str, object],
):
    """Invalid image options give the same HF result with and without metadata only."""
    ctx = build_model_context(
        "Qwen/Qwen2.5-VL-3B-Instruct", limit_mm_per_prompt={"image": 1}
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    images = [Image.new("RGB", (640, 480))]

    full, full_error = _process_outcome(
        processor, images, mm_processor_kwargs, metadata_only=False
    )
    meta, meta_error = _process_outcome(
        processor, images, mm_processor_kwargs, metadata_only=True
    )

    # Transformers 5.16 ignores a lone max_pixels, so not every case fails.
    assert meta_error == full_error
    if full_error is None:
        assert "pixel_values" in meta["mm_kwargs"]["image"][0]
        assert _rendered_features(meta) == _rendered_features(full)


def test_metadata_only_falls_back_for_configured_sizes():
    """Server-configured size options take the HF processor."""
    ctx = build_model_context(
        "Qwen/Qwen2-VL-2B-Instruct",
        mm_processor_kwargs={"images_kwargs": {"max_pixels": 512**2}},
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    images = [Image.new("RGB", (896, 896))]
    request = {
        "images_kwargs": {"size": {"shortest_edge": 3136, "longest_edge": 1024**2}}
    }

    full = _process_images(processor, images, request, metadata_only=False)
    meta = _process_images(processor, images, request, metadata_only=True)
    configured_full = _process_images(processor, images, {}, metadata_only=False)
    configured_meta = _process_images(processor, images, {}, metadata_only=True)

    assert "pixel_values" in meta["mm_kwargs"]["image"][0]
    assert _rendered_features(meta) == _rendered_features(full)
    assert "pixel_values" in configured_meta["mm_kwargs"]["image"][0]
    assert _rendered_features(configured_meta) == _rendered_features(configured_full)


def test_metadata_only_accepts_an_empty_image_list():
    """An empty image list renders like the full path."""
    ctx = build_model_context(
        "Qwen/Qwen2.5-VL-3B-Instruct", limit_mm_per_prompt={"image": 1}
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    full = _process_images(processor, [], {}, metadata_only=False)
    meta = _process_images(processor, [], {}, metadata_only=True)

    assert meta["prompt_token_ids"] == full["prompt_token_ids"]
    assert meta["mm_hashes"] == full["mm_hashes"]
    assert meta["mm_placeholders"] == full["mm_placeholders"]
