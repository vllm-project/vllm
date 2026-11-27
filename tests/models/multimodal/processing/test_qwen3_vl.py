# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for Qwen3-VL multimodal processing."""

import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import numpy as np
import pytest
import torch
from transformers.models.qwen3_vl import Qwen3VLVideoProcessor

from vllm import envs
from vllm.config import ModelConfig
from vllm.config.multimodal import MultiModalDummyOptions
from vllm.exceptions import VLLMValidationError
from vllm.model_executor.models.qwen3_vl import Qwen3VLProcessingInfo
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import MultiModalProcessorOnlyCache
from vllm.multimodal.processing import ProcessorInputs
from vllm.multimodal.processing.context import BaseProcessingInfo

from ...registry import HF_EXAMPLE_MODELS
from ...utils import build_model_context

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"


def test_qwen3_video_routing_schema_adds_vllm_compatibility_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native_video_kwargs = set(Qwen3VLVideoProcessor.valid_kwargs.__annotations__)
    assert "size" in native_video_kwargs
    assert "min_pixels" not in native_video_kwargs
    assert "max_pixels" not in native_video_kwargs

    info = Qwen3VLProcessingInfo(SimpleNamespace())
    parent_supported = {
        "text_kwargs": {"padding"},
        "images_kwargs": {"size", "min_pixels", "max_pixels"},
        "videos_kwargs": set(native_video_kwargs),
    }
    monkeypatch.setattr(
        BaseProcessingInfo,
        "get_supported_mm_processor_kwargs",
        lambda self: {scope: set(kwargs) for scope, kwargs in parent_supported.items()},
    )

    supported = info.supported_mm_processor_kwargs

    assert {"min_pixels", "max_pixels"} <= supported["videos_kwargs"]
    assert parent_supported["videos_kwargs"] == native_video_kwargs


def _build_video_mm_data(
    num_frames: int,
    width: int = 128,
    height: int = 128,
    original_fps: float = 30.0,
) -> dict[str, Any]:
    """Create synthetic video data with metadata indicating that
    HF processor should re-sample frames (do_sample_frames=True).

    `total_num_frames` is set equal to the ndarray frame count so
    that HF's `sample_frames` indices stay within bounds of the
    actual tensor that is passed."""
    video = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
    metadata = {
        "fps": original_fps,
        "duration": num_frames / original_fps,
        "total_num_frames": num_frames,
        "frames_indices": list(range(num_frames)),
        "video_backend": "opencv",
        "do_sample_frames": True,
    }
    return {"video": [(video, metadata)]}


@pytest.mark.parametrize("model_id", [MODEL_ID])
def test_processor_partial_scoped_image_size(model_id: str) -> None:
    """Partial image `size` must be completed and applied before the HF call.

    Keep video compatibility and per-video kwargs in an image-only call as well:
    Qwen3 must sanitize the video scope before HF validates it. Regression for
    #56363.
    """
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"image": 1, "video": 0},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    assert isinstance(processor.info, Qwen3VLProcessingInfo)

    image = np.zeros((128, 128, 3), dtype=np.uint8)
    min_pixels = 256 * 32 * 32
    default_size = dict(processor.info.get_image_processor().size)
    complete_size = default_size | {"shortest_edge": min_pixels}

    def process(hf_mm_kwargs: dict[str, Any]) -> torch.Tensor:
        mm_items = processor.info.parse_mm_data({"image": [image]})
        grid = processor._apply_hf_processor_main(mm_items, hf_mm_kwargs)[
            "image_grid_thw"
        ]
        assert isinstance(grid, torch.Tensor)
        return grid

    stock_grid = process({})
    complete_grid = process({"images_kwargs": {"size": complete_size}})
    partial_grid = process(
        {
            "images_kwargs": {
                "size": {"shortest_edge": min_pixels},
            },
            "videos_kwargs": {
                "min_pixels": 4096,
                "max_pixels": 469_762_048,
                "fps": [2.0],
                "num_frames": [8],
            },
        }
    )

    assert torch.equal(partial_grid, complete_grid)
    assert not torch.equal(partial_grid, stock_grid)


@pytest.mark.parametrize("model_id", [MODEL_ID])
def test_processor_partial_scoped_video_size(model_id: str) -> None:
    """Partial scoped video `size` must be completed and applied before HF."""
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"image": 0, "video": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    assert isinstance(processor.info, Qwen3VLProcessingInfo)

    min_pixels = 512 * 32 * 32
    default_size = dict(processor.info.get_video_processor().size)
    complete_size = default_size | {"shortest_edge": min_pixels}

    def process(hf_mm_kwargs: dict[str, Any]) -> torch.Tensor:
        mm_items = processor.info.parse_mm_data(_build_video_mm_data(num_frames=8))
        grid = processor._apply_hf_processor_main(mm_items, hf_mm_kwargs)[
            "video_grid_thw"
        ]
        assert isinstance(grid, torch.Tensor)
        return grid

    stock_grid = process({})
    complete_grid = process({"videos_kwargs": {"size": complete_size}})
    partial_grid = process(
        {
            "videos_kwargs": {
                "size": {"shortest_edge": min_pixels},
            },
        }
    )

    assert torch.equal(partial_grid, complete_grid)
    assert not torch.equal(partial_grid, stock_grid)


@pytest.mark.parametrize("model_id", [MODEL_ID])
def test_processor_flat_video_pixel_aliases_match_scoped_size(model_id: str) -> None:
    """Flat min/max pixel aliases keep their historical Qwen3 video effect."""
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"image": 0, "video": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    pixel_budget = 512 * 32 * 32

    def process(hf_mm_kwargs: dict[str, Any]) -> torch.Tensor:
        mm_items = processor.info.parse_mm_data(_build_video_mm_data(num_frames=8))
        return processor._apply_hf_processor_main(mm_items, hf_mm_kwargs)[
            "video_grid_thw"
        ]

    stock_grid = process({})
    scoped_grid = process(
        {
            "videos_kwargs": {
                "size": {
                    "shortest_edge": pixel_budget,
                    "longest_edge": pixel_budget,
                },
            },
        }
    )
    flat_grid = process(
        {
            "min_pixels": pixel_budget,
            "max_pixels": pixel_budget,
        }
    )

    assert torch.equal(flat_grid, scoped_grid)
    assert not torch.equal(flat_grid, stock_grid)


@pytest.mark.parametrize("model_id", [MODEL_ID])
@pytest.mark.parametrize(
    "num_frames",
    [8, 16],
)
def test_processor_num_frames_timestamp(
    model_id: str,
    num_frames: int,
) -> None:
    """Regression test: using `num_frames` (without `fps`) must not
    cause a timestamp / token-count mismatch.

    Before the fix, `_get_video_second_idx` ignored the explicit
    `num_frames` and fell back to an fps-based calculation, which
    produced a different number of timestamp entries and ultimately led
    to shape mismatches in downstream token construction.

    We deliberately choose `num_frames` values (8, 16) that differ
    from what the default fps-based path would compute (which clamps
    to `min_frames=4` for a short video at 30 fps), so this test
    would fail without the fix.
    """
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"image": 0, "video": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    prompt = "<|vision_start|><|video_pad|><|vision_end|>"
    mm_data = _build_video_mm_data(num_frames=num_frames)

    # Process with explicit num_frames (no fps) -- this is the path
    # that was broken before the fix.
    hf_mm_kwargs: dict[str, Any] = {"num_frames": num_frames}
    processed = processor(
        prompt,
        mm_items=processor.info.parse_mm_data(mm_data),
        hf_processor_mm_kwargs=hf_mm_kwargs,
    )

    # Basic sanity: the processor must produce video tokens.
    token_ids = processed["prompt_token_ids"]
    assert len(token_ids) > 0, "Processor produced empty token list"

    # Verify that video placeholders were actually inserted.
    assert "mm_placeholders" in processed
    video_phs = processed["mm_placeholders"].get("video", [])
    assert len(video_phs) == 1, (
        f"Expected exactly 1 video placeholder, got {len(video_phs)}"
    )


@pytest.mark.parametrize("model_id", [MODEL_ID])
@pytest.mark.parametrize("num_videos", [2, 4])
def test_processor_multi_video(
    model_id: str,
    num_videos: int,
) -> None:
    """Verify that multi-video processing produces correct placeholders.

    This exercises the token-level replacement path in
    `_apply_hf_processor_main` which avoids the quadratic text-level
    prompt expansion.
    """
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"image": 0, "video": num_videos},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    prompt = "<|vision_start|><|video_pad|><|vision_end|>" * num_videos
    mm_data = {"video": [_build_video_mm_data(num_frames=8)["video"][0]] * num_videos}

    processed = processor(
        prompt,
        mm_items=processor.info.parse_mm_data(mm_data),
        hf_processor_mm_kwargs={"num_frames": 8},
    )

    token_ids = processed["prompt_token_ids"]
    assert len(token_ids) > 0

    video_phs = processed["mm_placeholders"].get("video", [])
    assert len(video_phs) == num_videos, (
        f"Expected {num_videos} video placeholders, got {len(video_phs)}"
    )

    # All placeholders should have the same length (same video params)
    # and must not overlap.
    lengths = {ph.length for ph in video_phs}
    assert len(lengths) == 1, f"Placeholder lengths differ: {lengths}"
    for i in range(1, len(video_phs)):
        prev_end = video_phs[i - 1].offset + video_phs[i - 1].length
        assert video_phs[i].offset >= prev_end, (
            f"Placeholder {i} overlaps with placeholder {i - 1}"
        )


# Qwen3-VL / Qwen3.8 "Long Video Understanding" pixel budget from the
# model card. Used to check --mm-processor-kwargs scoping (#52834).
_LONG_VIDEO_SIZE = {"longest_edge": 469762048, "shortest_edge": 4096}


def _probe_mm_token_budgets(
    model_id: str, mm_processor_kwargs: dict[str, Any] | None
) -> tuple[int, int]:
    ctx = build_model_context(
        model_id,
        mm_processor_kwargs=mm_processor_kwargs,
        limit_mm_per_prompt={"image": 1, "video": 1},
    )
    info = MULTIMODAL_REGISTRY.create_processor(ctx.model_config).info
    assert isinstance(info, Qwen3VLProcessingInfo)
    video = info.get_max_video_tokens(
        seq_len=500000, mm_counts={"video": 1, "image": 1}
    )
    return video, info.get_max_image_tokens()


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("model_id", [MODEL_ID])
def test_processor_kwargs_videos_kwargs_does_not_leak_into_image_budget(
    model_id: str,
) -> None:
    """`videos_kwargs.size` must raise only the video token budget.

    A flat `size` override still applies to both modalities (the previous
    shared-namespace behavior). Regression for #52834.
    """
    stock_video, stock_image = _probe_mm_token_budgets(model_id, None)
    scoped_video, scoped_image = _probe_mm_token_budgets(
        model_id, {"videos_kwargs": {"size": _LONG_VIDEO_SIZE}}
    )
    flat_video, flat_image = _probe_mm_token_budgets(
        model_id, {"size": _LONG_VIDEO_SIZE}
    )

    assert scoped_video == flat_video
    assert scoped_video > stock_video
    assert scoped_image == stock_image
    assert flat_image > stock_image


@pytest.mark.parametrize("model_id", [MODEL_ID])
@pytest.mark.parametrize(
    ("hf_mm_kwargs", "scalar_kwargs"),
    [
        ({"num_frames": [8, 16]}, [{"num_frames": 8}, {"num_frames": 16}]),
        ({"fps": [2.0, 4.0]}, [{"fps": 2.0}, {"fps": 4.0}]),
        ({"fps": [2, 4]}, [{"fps": 2}, {"fps": 4}]),
        ({"num_frames": [8, 16, 32]}, [{"num_frames": 8}, {"num_frames": 16}]),
        ({"fps": [2.0, 4.0, 8.0]}, [{"fps": 2.0}, {"fps": 4.0}]),
    ],
)
@pytest.mark.parametrize("model_kwargs", [False, True])
@pytest.mark.parametrize("use_uuids", [False, True])
@pytest.mark.parametrize("scoped", [False, True])
@pytest.mark.skip_global_cleanup
def test_processor_multi_video_list_kwargs(
    model_id: str,
    hf_mm_kwargs: dict[str, Any],
    scalar_kwargs: list[dict[str, Any]],
    model_kwargs: bool,
    use_uuids: bool,
    scoped: bool,
) -> None:
    """Per-video options must survive duplicate items and partial cache hits."""
    if scoped:
        hf_mm_kwargs = {"videos_kwargs": hf_mm_kwargs}
    hf_mm_kwargs_before = deepcopy(hf_mm_kwargs)
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"image": 0, "video": 2},
        mm_processor_kwargs=hf_mm_kwargs if model_kwargs else None,
        mm_processor_cache_gb=1,
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    cache = MultiModalProcessorOnlyCache(ctx.model_config)
    prompt = "<|vision_start|><|video_pad|><|vision_end|>" * 2
    videos = [
        _build_video_mm_data(num_frames=32, original_fps=8.0)["video"][0]
        for _ in range(3)
    ]
    for i, (frames, _) in enumerate(videos):
        frames.fill(i)

    def process_single(hf_kwargs: dict[str, Any]) -> torch.Tensor:
        single_prompt = "<|vision_start|><|video_pad|><|vision_end|>"
        single = processor(
            single_prompt,
            mm_items=processor.info.parse_mm_data({"video": [videos[0]]}),
            hf_processor_mm_kwargs=hf_kwargs,
        )
        grids = single["mm_kwargs"].get_data()["video_grid_thw"]
        assert isinstance(grids, torch.Tensor)
        return grids[0]

    expected_grids = [process_single(kwargs) for kwargs in scalar_kwargs]
    assert not torch.equal(expected_grids[0], expected_grids[1])

    def run(indices, *, cache=None, cached_only=False):
        return processor(
            prompt,
            mm_items=processor.info.parse_mm_data(
                {"video": [None if cached_only else videos[i] for i in indices]}
            ),
            mm_uuid_items=(
                {"video": [f"video-{i}" for i in indices]} if use_uuids else None
            ),
            hf_processor_mm_kwargs={} if model_kwargs else hf_mm_kwargs,
            cache=cache,
        )

    for indices in ([0, 0], [0, 1], [0, 2], [2, 0]):
        expected = run(indices)
        actual = run(indices, cache=cache)
        assert actual["prompt_token_ids"] == expected["prompt_token_ids"]
        assert actual["mm_placeholders"] == expected["mm_placeholders"]
        assert actual["mm_kwargs"] == expected["mm_kwargs"]
        assert actual["mm_hashes"] == expected["mm_hashes"]
        assert len(set(actual["mm_hashes"]["video"])) == 2
        multi_grids = actual["mm_kwargs"].get_data()["video_grid_thw"]
        assert torch.equal(multi_grids[0], expected_grids[0])
        assert torch.equal(multi_grids[1], expected_grids[1])

    if use_uuids:
        cached_only = run([2, 0], cache=cache, cached_only=True)
        assert cached_only["prompt_token_ids"] == actual["prompt_token_ids"]
        assert cached_only["mm_kwargs"] == actual["mm_kwargs"]
    assert hf_mm_kwargs == hf_mm_kwargs_before


@pytest.mark.parametrize(("key", "value"), [("fps", 2.0), ("num_frames", 8)])
@pytest.mark.parametrize("length", [0, 1])
@pytest.mark.parametrize("use_cache", [False, True])
@pytest.mark.parametrize("scoped", [False, True])
def test_processor_rejects_short_per_video_options(
    key: str, value: float | int, length: int, use_cache: bool, scoped: bool
) -> None:
    """Reject malformed option lists with and without a warm processor cache."""
    ctx = build_model_context(
        MODEL_ID,
        limit_mm_per_prompt={"image": 0, "video": 2},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    cache = MultiModalProcessorOnlyCache(ctx.model_config) if use_cache else None
    prompt = "<|vision_start|><|video_pad|><|vision_end|>" * 2
    videos = _build_video_mm_data(num_frames=32, original_fps=8.0)["video"] * 2

    def process(values: list[float | int]) -> None:
        kwargs = {key: values}
        processor(
            prompt,
            mm_items=processor.info.parse_mm_data({"video": videos}),
            hf_processor_mm_kwargs={"videos_kwargs": kwargs} if scoped else kwargs,
            cache=cache,
        )

    if use_cache:
        process([value, value])

    with pytest.raises(ValueError, match=f"{key} must have at least 2 entries"):
        process([value] * length)


@pytest.mark.parametrize(
    "kwargs", [{"fps": [2.0, 4.0]}, {"fps": [2, 4]}, {"num_frames": [8, 16]}]
)
def test_processor_profiles_fewer_videos_than_configured_options(
    kwargs: dict[str, list[float] | list[int]],
) -> None:
    """Profiling may use fewer videos than the configured per-video list."""
    ctx = build_model_context(
        MODEL_ID,
        limit_mm_per_prompt={"image": 0, "video": 2},
        mm_processor_kwargs=kwargs,
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    profiled = processor.get_dummy_mm_inputs({"video": 1})

    assert len(profiled["mm_placeholders"]["video"]) == 1
    video_grid_thw = profiled["mm_kwargs"].get_data()["video_grid_thw"]
    assert isinstance(video_grid_thw, torch.Tensor)
    assert video_grid_thw.shape[0] == 1


def _build_video_embeds_mm_data(
    hidden_size: int,
    grid_thw: tuple[int, int, int] = (2, 4, 4),
) -> dict[str, Any]:
    """Create an embeds-only video item as an EPD consumer receives it:
    pre-computed embeddings plus the metadata published by the encoder."""
    import torch

    t, h, w = grid_thw
    num_tokens = t * h * w // 4  # spatial_merge_size ** 2
    return {
        "video": {
            "video_embeds": torch.zeros(num_tokens, hidden_size),
            "video_grid_thw": torch.tensor([grid_thw]),
            "timestamps": torch.tensor([[float(i) for i in range(t)]]),
        }
    }


@pytest.mark.parametrize("model_id", [MODEL_ID])
def test_processor_video_embeds_with_timestamps(model_id: str) -> None:
    """Embeds-only video input must size the placeholder range from the
    grid and the real timestamps published by the encoder (EC consumer
    path); synthesized or missing timestamps would change the token count
    and break embedding merging downstream."""
    # `build_model_context` forces enable_mm_embeds off, so build the
    # config directly; keep the same online-availability skip behavior.
    HF_EXAMPLE_MODELS.find_hf_info(model_id).check_available_online(on_fail="skip")
    model_config = ModelConfig(
        model_id,
        runner="generate",
        limit_mm_per_prompt={"image": 0, "video": 1},
        enable_mm_embeds=True,
    )
    processor = MULTIMODAL_REGISTRY.create_processor(model_config)
    tokenizer = processor.info.get_tokenizer()

    grid_thw = (2, 4, 4)
    mm_data = _build_video_embeds_mm_data(
        model_config.get_inputs_embeds_size(), grid_thw
    )

    prompt = "<|vision_start|><|video_pad|><|vision_end|>"
    processed = processor(
        prompt,
        mm_items=processor.info.parse_mm_data(mm_data),
        hf_processor_mm_kwargs={},
    )

    video_phs = processed["mm_placeholders"].get("video", [])
    assert len(video_phs) == 1, (
        f"Expected exactly 1 video placeholder, got {len(video_phs)}"
    )

    t, h, w = grid_thw
    tokens_per_frame = h * w // 4
    timestamp_tokens = sum(
        len(tokenizer.encode(f"<{float(i):.1f} seconds>", add_special_tokens=False))
        for i in range(t)
    )
    # per frame: timestamp tokens + vision_start + video tokens + vision_end
    expected_len = timestamp_tokens + t * (tokens_per_frame + 2)
    assert video_phs[0].length == expected_len


@pytest.mark.parametrize("model_id", [MODEL_ID])
def test_processor_video_embeds_missing_timestamps(model_id: str) -> None:
    """Timestamps are required metadata for video embeds: they size the
    placeholder range, so omitting them must fail loudly at parse time
    instead of silently producing a wrong prompt."""
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"image": 0, "video": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    mm_data = _build_video_embeds_mm_data(ctx.model_config.get_inputs_embeds_size())
    del mm_data["video"]["timestamps"]

    with pytest.raises(ValueError, match="timestamps"):
        processor.info.parse_mm_data(mm_data)


@pytest.mark.parametrize(
    ("video_kwargs", "expected_frames"),
    [
        ({}, 2),
        ({"cap_pixels_per_frame": True}, 4),
        ({"cap_pixels_per_frame": True, "max_video_tokens": 1536}, 4),
    ],
)
def test_dummy_video_spreads_budget_when_frame_cap_enabled(
    video_kwargs: dict[str, object], expected_frames: int
) -> None:
    """Spread the profiling budget over frames using the actual HF ceiling."""
    # The processor instance uses 768 tokens per frame at patch 16, merge 2.
    per_frame_cap = 768 * (16 * 2) ** 2
    size = {"longest_edge": per_frame_cap * 4, "shortest_edge": 4096}
    model_config = ModelConfig(
        model="Qwen/Qwen3-VL-Reranker-2B",
        mm_processor_kwargs={
            "videos_kwargs": {
                "size": size,
                **video_kwargs,
            },
        },
        limit_mm_per_prompt={"image": 0, "video": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(model_config)
    dummy = processor.dummy_inputs.get_dummy_mm_data(
        1024, {"video": 1}, MultiModalDummyOptions()
    )
    videos = dummy["video"]
    assert isinstance(videos, list)
    assert videos[0][0].shape[0] == expected_frames


def test_dummy_video_covers_request_when_shortest_edge_floor_exceeds_cap() -> None:
    """The shortest-edge floor can raise the cost of each frame in a real clip."""
    model_config = ModelConfig(
        model="Qwen/Qwen3-VL-Reranker-2B",
        mm_processor_kwargs={
            "max_video_tokens": 4,
            "videos_kwargs": {
                "cap_pixels_per_frame": True,
                "size": {"shortest_edge": 5120, "longest_edge": 10240},
            },
        },
        limit_mm_per_prompt={"image": 0, "video": 1},
        mm_processor_cache_gb=0,
    )
    processor = MULTIMODAL_REGISTRY.create_processor(model_config)
    profiled = processor.get_dummy_mm_inputs({"video": 1})
    data = _build_video_mm_data(num_frames=4, width=64, height=64, original_fps=2.0)
    data["video"][0][1]["do_sample_frames"] = False
    processed = processor(
        "<|vision_start|><|video_pad|><|vision_end|>",
        mm_items=processor.info.parse_mm_data(data),
        hf_processor_mm_kwargs={},
    )

    (profiled_video,) = profiled["mm_placeholders"]["video"]
    (processed_video,) = processed["mm_placeholders"]["video"]
    assert processed_video.get_num_embeds() == 8
    assert profiled_video.get_num_embeds() >= processed_video.get_num_embeds()


@pytest.mark.parametrize("model_id", [MODEL_ID])
@pytest.mark.parametrize("scoped", [False, True])
@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.skip_global_cleanup
def test_processor_per_video_hashes_survive_partitioning(
    model_id: str, scoped: bool, configured: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Actual encoder children produce the corresponding parent keys and grids."""
    from examples.disaggregated.disaggregated_encoder import disagg_epd_proxy

    def scoped_form(flat: dict[str, Any]) -> dict[str, Any]:
        return {"videos_kwargs": dict(flat)} if scoped else dict(flat)

    options = scoped_form({"num_frames": [8, 16]})
    kwargs = {} if configured else options
    if scoped and not configured:
        kwargs["num_frames"] = [24]  # shadowed by the scoped list
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"image": 0, "video": 2},
        mm_processor_kwargs=options
        if configured
        else scoped_form({"num_frames": [24]}),
        mm_processor_cache_gb=1,
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    cache = MultiModalProcessorOnlyCache(ctx.model_config)
    videos = [
        _build_video_mm_data(num_frames=32, original_fps=8.0)["video"][0]
        for _ in range(3)
    ]
    for i, (frames, _) in enumerate(videos):
        frames.fill(i)

    def run(items, hf_kwargs, uuids, *, cache=None):
        return processor(
            "<|vision_start|><|video_pad|><|vision_end|>" * len(items),
            mm_items=processor.info.parse_mm_data({"video": items}),
            mm_uuid_items={"video": uuids},
            hf_processor_mm_kwargs=hf_kwargs,
            cache=cache,
        )

    response = SimpleNamespace(status=200, read=AsyncMock(return_value=b"{}"))
    post = AsyncMock(return_value=response)
    monkeypatch.setattr(disagg_epd_proxy, "encode_session", SimpleNamespace(post=post))
    monkeypatch.setattr(disagg_epd_proxy, "NO_REWRITE", False)
    body: dict[str, Any] = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": f"http://media/v{i}"},
                        "uuid": f"v{i}",
                    }
                ],
            }
            for i in range(2)
        ],
    }
    if kwargs:
        body["mm_processor_kwargs"] = kwargs
    asyncio.run(
        disagg_epd_proxy.fanout_encoder_primer(
            body, ["http://encoder"], "partition", enable_per_video_kwargs_slicing=True
        )
    )
    full = run(videos[:2], kwargs, ["v0", "v1"])
    full_grids = full["mm_kwargs"].get_data()["video_grid_thw"]
    assert not torch.equal(full_grids[0], full_grids[1])
    assert post.await_count == 2
    for call in post.await_args_list:
        child_request = json.loads(call.kwargs["data"])
        uuid = child_request["messages"][0]["content"][0]["uuid"]
        index = int(uuid[1:])
        for child_cache in (None, cache):
            child = run(
                [videos[index]],
                child_request["mm_processor_kwargs"],
                [uuid],
                cache=child_cache,
            )
            assert child["mm_hashes"]["video"] == [full["mm_hashes"]["video"][index]]
            assert torch.equal(
                child["mm_kwargs"].get_data()["video_grid_thw"][0], full_grids[index]
            )

    # Reuse v0 from the child while the new video still uses position 1.
    partial = run([videos[0], videos[2]], kwargs, ["v0", "v2"], cache=cache)
    assert partial["mm_hashes"]["video"][0] == full["mm_hashes"]["video"][0]
    assert partial["mm_hashes"]["video"][1] != full["mm_hashes"]["video"][1]
    assert torch.equal(partial["mm_kwargs"].get_data()["video_grid_thw"], full_grids)

    if not configured:
        tail = run(
            videos[:2],
            {**kwargs, **scoped_form({"num_frames": [8, 16, 32]})},
            ["v0", "v1"],
        )
        assert tail["mm_hashes"] == full["mm_hashes"]
        changed = run(
            videos[:2],
            {**kwargs, **scoped_form({"num_frames": [8, 24]})},
            ["v0", "v1"],
        )
        assert changed["mm_hashes"]["video"][0] == full["mm_hashes"]["video"][0]
        assert changed["mm_hashes"]["video"][1] != full["mm_hashes"]["video"][1]


@pytest.mark.parametrize("partition", [[-1, 2], [2, 2], [True, 2], None])
@pytest.mark.skip_global_cleanup
def test_processor_rejects_invalid_epd_video_position(partition: object) -> None:
    ctx = build_model_context(MODEL_ID, limit_mm_per_prompt={"video": 1})
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    with pytest.raises(VLLMValidationError, match="Invalid EPD video position"):
        processor(
            "<|vision_start|><|video_pad|><|vision_end|>",
            mm_items=processor.info.parse_mm_data(_build_video_mm_data(num_frames=8)),
            hf_processor_mm_kwargs={"_epd_video_partition": partition},
        )


@pytest.mark.parametrize("model_id", [MODEL_ID])
@pytest.mark.parametrize("form", ["flat", "scoped", "shadowed"])
@pytest.mark.parametrize(("key", "value"), [("num_frames", 16), ("fps", 4.0)])
@pytest.mark.parametrize("use_uuid", [False, True])
@pytest.mark.parametrize("slicing", [False, True])
@pytest.mark.skip_global_cleanup
def test_processor_image_hash_survives_video_partitioning(
    model_id: str,
    form: str,
    key: str,
    value: int | float,
    use_uuid: bool,
    slicing: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mixed parent and image child both preserve the upstream image key."""
    from examples.disaggregated.disaggregated_encoder import disagg_epd_proxy

    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"image": 1, "video": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    video = _build_video_mm_data(num_frames=32, original_fps=8.0)["video"][0]
    kwargs: dict[str, Any] = {key: [value]}
    if form != "flat":
        kwargs = {"videos_kwargs": kwargs}
    if form == "shadowed":
        kwargs[key] = [value / 2 if key == "fps" else 8]

    items = []
    uuids = {}
    for modality in ("image", "video"):
        media_type = f"{modality}_url"
        item: dict[str, Any] = {
            "type": media_type,
            media_type: {"url": f"http://media/{modality}"},
        }
        if use_uuid:
            item["uuid"] = modality
            uuids[modality] = [modality]
        items.append(item)
    response = SimpleNamespace(status=200, read=AsyncMock(return_value=b"{}"))
    post = AsyncMock(return_value=response)
    monkeypatch.setattr(disagg_epd_proxy, "encode_session", SimpleNamespace(post=post))
    monkeypatch.setattr(disagg_epd_proxy, "NO_REWRITE", not use_uuid)
    asyncio.run(
        disagg_epd_proxy.fanout_encoder_primer(
            {
                "messages": [{"role": "user", "content": items}],
                "mm_processor_kwargs": kwargs,
            },
            ["http://encoder"],
            "hash-regression",
            enable_per_video_kwargs_slicing=slicing,
        )
    )
    children = [json.loads(call.kwargs["data"]) for call in post.await_args_list]
    child = next(
        body
        for body in children
        if body["messages"][0]["content"][0]["type"] == "image_url"
    )

    media = processor.info.parse_mm_data({"image": [image], "video": [video]})
    image_media = media.select({"image"})
    child_uuid = child["messages"][0]["content"][0].get("uuid")
    child_uuids = {"image": [child_uuid]} if child_uuid is not None else {}
    image_prompt = "<|vision_start|><|image_pad|><|vision_end|>"
    video_prompt = "<|vision_start|><|video_pad|><|vision_end|>"
    parent_hashes = processor(
        image_prompt + video_prompt,
        mm_items=media,
        mm_uuid_items=uuids,
        hf_processor_mm_kwargs=kwargs,
    )["mm_hashes"]
    child_hashes = processor(
        image_prompt,
        mm_items=image_media,
        mm_uuid_items=child_uuids,
        hf_processor_mm_kwargs=child["mm_processor_kwargs"],
    )["mm_hashes"]

    # Before per-video normalization, Qwen3-VL rehashed only the video keys.
    upstream_image_hash = ProcessorInputs(
        prompt=[],
        mm_data_items=image_media,
        mm_uuid_items=uuids,
        hf_processor_mm_kwargs=kwargs,
    ).get_mm_hashes(
        processor.info.model_id, processor.info.ctx.get_mm_config().mm_hasher_algorithm
    )["image"]
    assert list(parent_hashes) == list(media)
    assert parent_hashes["image"] == child_hashes["image"] == upstream_image_hash


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_frames": 2},
        {"num_frames": 3},
        {"num_frames": 16},
        {"max_frames": 4},
        {"min_frames": 8},
    ],
)
@pytest.mark.parametrize("kwargs_on_init", [False, True])
@pytest.mark.parametrize("token_prompt", [False, True])
def test_video_sampling_matches_hf_timestamps(
    kwargs: dict[str, Any],
    kwargs_on_init: bool,
    token_prompt: bool,
) -> None:
    """Frame counts, limits and timestamps must follow the same HF sampling."""
    ctx = build_model_context(
        MODEL_ID,
        limit_mm_per_prompt={"video": 1},
        mm_processor_cache_gb=1,
        mm_processor_kwargs=kwargs if kwargs_on_init else None,
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    prompt = "<|vision_start|><|video_pad|><|vision_end|>"
    mm_data = _build_video_mm_data(num_frames=12, original_fps=4.0)
    frames, metadata = mm_data["video"][0]
    hf_processor = processor.info.get_hf_processor(**kwargs)
    reference = hf_processor(
        text=prompt,
        videos=[frames],
        video_metadata=[{k: v for k, v in metadata.items() if k != "do_sample_frames"}],
        do_sample_frames=True,
        return_tensors="pt",
        **({"fps": None} | kwargs if "num_frames" in kwargs else kwargs),
    )
    if token_prompt:
        prompt = ctx.tokenizer.encode(prompt, add_special_tokens=False)
    cache = MultiModalProcessorOnlyCache(ctx.model_config)
    for _ in range(2):
        processed = processor(
            prompt,
            mm_items=processor.info.parse_mm_data(deepcopy(mm_data)),
            hf_processor_mm_kwargs={} if kwargs_on_init else kwargs,
            cache=cache,
        )
        data = processed["mm_kwargs"].get_data()
        assert torch.equal(data["video_grid_thw"], reference["video_grid_thw"])
        expected_embeds = int(reference["video_grid_thw"].prod()) // 4
        assert (
            processed["mm_placeholders"]["video"][0].get_num_embeds() == expected_embeds
        )
        assert processed["prompt_token_ids"] == reference["input_ids"][0].tolist()


@pytest.mark.parametrize(
    "engine_kwargs,request_kwargs,expected_frames",
    [
        ({"max_frames": 4}, {}, 4),
        ({"max_frames": 4}, {"max_frames": 2}, 2),
        ({"max_frames": 4}, {"videos_kwargs": {"max_frames": 2}}, 2),
        ({"max_frames": 4}, {"max_frames": 8}, 6),
        ({"max_frames": 4}, {"videos_kwargs": {"max_frames": 8}}, 6),
        ({"min_frames": 8}, {}, 8),
        ({"min_frames": 8}, {"min_frames": 10}, 10),
        ({"min_frames": 8}, {"videos_kwargs": {"min_frames": 10}}, 10),
    ],
)
@pytest.mark.parametrize("token_prompt", [False, True])
def test_video_scoped_frame_limits(
    engine_kwargs, request_kwargs, expected_frames, token_prompt
):
    """Engine video limits and request overrides must change the sampled frames."""
    ctx = build_model_context(
        MODEL_ID,
        limit_mm_per_prompt={"video": 1},
        mm_processor_cache_gb=1,
        mm_processor_kwargs={"videos_kwargs": engine_kwargs},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    prompt = "<|vision_start|><|video_pad|><|vision_end|>"
    if token_prompt:
        prompt = ctx.tokenizer.encode(prompt, add_special_tokens=False)
    mm_data = _build_video_mm_data(num_frames=12, original_fps=4.0)
    cache = MultiModalProcessorOnlyCache(ctx.model_config)
    # At 2 fps this three-second clip has six frames before applying limits.
    # Check a request override, then the engine default, including cache hits.
    default_frames = engine_kwargs.get("max_frames", engine_kwargs.get("min_frames"))
    for kwargs, num_frames in [(request_kwargs, expected_frames), ({}, default_frames)]:
        indices = np.linspace(0, 11, num_frames).round().astype(int)
        timestamps = indices.reshape(-1, 2).mean(axis=1) / 4.0
        for _ in range(2):
            processed = processor(
                prompt,
                mm_items=processor.info.parse_mm_data(deepcopy(mm_data)),
                hf_processor_mm_kwargs=kwargs,
                cache=cache,
            )
            data = processed["mm_kwargs"].get_data()
            grid = data["video_grid_thw"][0]
            pixels = data["pixel_values_videos"]
            assert isinstance(grid, torch.Tensor)
            assert isinstance(pixels, torch.Tensor)
            # Qwen3-VL groups two sampled frames into each temporal patch.
            assert int(grid[0]) * 2 == num_frames
            assert pixels.shape[0] == int(grid.prod())
            assert data["timestamps"][0] == timestamps.tolist()
            assert processed["mm_placeholders"]["video"][0].get_num_embeds() == (
                int(grid.prod()) // 4
            )


@pytest.mark.parametrize("scoped", [False, True])
def test_video_budget_rejects_num_frames_before_sampling(
    monkeypatch: pytest.MonkeyPatch, scoped: bool
) -> None:
    """An explicit frame count over the budget must be rejected unsampled."""
    ctx = build_model_context(MODEL_ID, limit_mm_per_prompt={"video": 1})
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    # A 128x128 RGB frame takes 49152 uint8 or 196608 float32 bytes and is counted
    # twice, so 8 frames fit in 4 MiB and 64 do not in either format.
    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_PREPROCESSING_BYTES", 4 << 20)

    def process(num_frames: int):
        kwargs: dict[str, Any] = {"num_frames": num_frames}
        return processor(
            "<|vision_start|><|video_pad|><|vision_end|>",
            mm_items=processor.info.parse_mm_data(_build_video_mm_data(num_frames=4)),
            hf_processor_mm_kwargs={"videos_kwargs": kwargs} if scoped else kwargs,
        )

    # Within the budget, HF's explicit count applies beyond the 4 decoded frames.
    grid = process(8)["mm_kwargs"].get_data()["video_grid_thw"]
    assert isinstance(grid, torch.Tensor)
    assert grid.tolist() == [[4, 8, 8]]

    def fail_sampling(*args: object, **kwargs: object):
        raise AssertionError("Frames were sampled before the budget check.")

    # vLLM's timestamps and HF's sampler both build their indices with linspace.
    monkeypatch.setattr(np, "linspace", fail_sampling)
    with pytest.raises(VLLMValidationError, match="VLLM_MAX_MEDIA_PREPROCESSING_BYTES"):
        process(64)
