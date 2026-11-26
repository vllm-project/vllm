# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image as PILImage

from vllm import envs
from vllm.config import ModelConfig
from vllm.exceptions import VLLMValidationError
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.encoder_budget import get_mm_max_toks_per_item
from vllm.multimodal.processing.context import (
    TimingContext,
    get_video_frame_dimensions,
)
from vllm.multimodal.processing.inputs import ProcessorInputs

from ...utils import build_model_context


def test_video_preflight_accepts_pil_and_channel_first_frames():
    pil_frames = [PILImage.new("RGB", (128, 64)) for _ in range(2)]
    chw_frames = torch.zeros((2, 3, 64, 128), dtype=torch.uint8)

    assert get_video_frame_dimensions(pil_frames) == (64, 128, 3)
    assert get_video_frame_dimensions(chw_frames) == (64, 128, 3)


@pytest.mark.parametrize(
    "model_id",
    [
        "Qwen/Qwen3-VL-Reranker-2B",
        "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
    ],
)
def test_video_item_limit_defaults_to_32_and_honors_explicit_limit(model_id: str):
    for video_options in (None, {"num_frames": 16}, {"allow_embeds": False}):
        limit = {} if video_options is None else {"video": video_options}
        model_config = ModelConfig(model=model_id, limit_mm_per_prompt=limit)
        info = MULTIMODAL_REGISTRY.get_processing_info(model_config)
        info.validate_num_items("video", 32)
        with pytest.raises(VLLMValidationError, match="At most 32 video"):
            info.validate_num_items("video", 33)

    for video_limit in (33, {"count": 999, "num_frames": 16}):
        model_config = ModelConfig(
            model=model_id, limit_mm_per_prompt={"video": video_limit}
        )
        info = MULTIMODAL_REGISTRY.get_processing_info(model_config)
        info.validate_num_items("video", 33)


@pytest.mark.parametrize(
    ("num_frames", "side", "video_kwargs", "grid", "budget"),
    [
        (2, 84, {"patch_size": 4, "merge_size": 4}, [1, 20, 20], 307_200),
        (1, 64, {"do_resize": False, "temporal_patch_size": 1}, [1, 4, 4], 98_304),
    ],
)
def test_qwen3_vl_video_preflight_uses_patch_overrides(
    monkeypatch: pytest.MonkeyPatch,
    num_frames: int,
    side: int,
    video_kwargs: dict[str, int | bool],
    grid: list[int],
    budget: int,
):
    """Only the resized and padded tensor with the overrides fits the budget."""
    monkeypatch.setenv("VLLM_MAX_MEDIA_PREPROCESSING_BYTES", str(budget))
    config = ModelConfig(
        model="Qwen/Qwen3-VL-Reranker-2B",
        dtype="float32",
        mm_device_do_normalize=False,
        mm_processor_cache_gb=0,
        limit_mm_per_prompt={"image": 0, "video": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(config)
    frames = np.zeros((num_frames, side, side, 3), dtype=np.uint8)
    metadata = {
        "fps": 2.0,
        "duration": num_frames / 2,
        "total_num_frames": num_frames,
        "frames_indices": list(range(num_frames)),
        "video_backend": "opencv",
        "do_sample_frames": False,
    }
    inputs = ProcessorInputs(
        prompt=processor.info.get_tokenizer().encode(
            "<|vision_start|><|video_pad|><|vision_end|>"
        ),
        mm_data_items=processor.info.parse_mm_data({"video": [(frames, metadata)]}),
        hf_processor_mm_kwargs={"videos_kwargs": video_kwargs},
    )

    outputs = processor.apply(inputs, TimingContext(enabled=False))
    (video,) = outputs["mm_kwargs"]["video"]
    assert video is not None
    tensors = video.get_data()
    actual_grid = tensors["video_grid_thw"]
    pixels = tensors["pixel_values_videos"]
    assert isinstance(actual_grid, torch.Tensor)
    assert isinstance(pixels, torch.Tensor)
    assert actual_grid.tolist() == grid
    assert pixels.nbytes * 2 == budget


@pytest.mark.parametrize(
    ("decoded_frames", "sampling_kwargs", "processed_frames"),
    [
        (2, {"do_sample_frames": False}, 2),
        (6, {"do_sample_frames": True, "fps": 1, "max_frames": 4}, 3),
        (6, {"do_sample_frames": True, "fps": 1, "max_frames": 2}, 2),
        (2, {"do_sample_frames": True, "fps": 3, "overflow_strategy": None}, 3),
    ],
    ids=["unsampled", "fps", "frame-cap", "upsample"],
)
def test_embedding_gemma2_video_budget_uses_effective_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decoded_frames: int,
    sampling_kwargs: dict[str, object],
    processed_frames: int,
):
    pytest.importorskip("transformers.models.embedding_gemma2")
    from transformers import (
        AutoTokenizer,
        EmbeddingGemma2Config,
        EmbeddingGemma2Processor,
        Gemma4AudioFeatureExtractor,
        Gemma4ImageProcessor,
        Gemma4VisionConfig,
    )
    from transformers.models.embedding_gemma2.video_processing_embedding_gemma2 import (
        EmbeddingGemma2VideoProcessor,
    )

    hf_processor = EmbeddingGemma2Processor(
        feature_extractor=Gemma4AudioFeatureExtractor(),
        image_processor=Gemma4ImageProcessor(),
        tokenizer=AutoTokenizer.from_pretrained("google/gemma-4-E2B-it"),
        video_processor=EmbeddingGemma2VideoProcessor(max_soft_tokens=140),
    )
    hf_processor.save_pretrained(tmp_path)
    EmbeddingGemma2Config(
        architectures=["EmbeddingGemma2Model"],
        vision_config=Gemma4VisionConfig(),
        image_token_id=hf_processor.image_token_id,
        video_token_id=hf_processor.video_token_id,
    ).save_pretrained(tmp_path)

    # Only the effective frame count and soft-token limit fit the byte budget.
    # Both the output tensor and its concatenated copy count toward the budget.
    pixel_shape = (processed_frames, 70 * 3**2, 16**2 * 3)
    budget = 2 * processed_frames * pixel_shape[1] * pixel_shape[2] * 4
    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_PREPROCESSING_BYTES", budget)
    config = ModelConfig(
        model=str(tmp_path),
        dtype="float32",
        max_model_len=4096,
        limit_mm_per_prompt={"image": 0, "video": 1, "audio": 0},
        mm_device_do_normalize=False,
    )
    processor = MULTIMODAL_REGISTRY.create_processor(config)
    frames = np.zeros((decoded_frames, 64, 64, 3), dtype=np.uint8)
    metadata = {
        "fps": 2.0,
        "duration": decoded_frames / 2.0,
        "total_num_frames": decoded_frames,
        "frames_indices": list(range(decoded_frames)),
    }
    inputs = ProcessorInputs(
        prompt=[hf_processor.video_token_id],
        mm_data_items=processor.info.parse_mm_data({"video": [(frames, metadata)]}),
        hf_processor_mm_kwargs={
            "videos_kwargs": {"max_soft_tokens": 70, **sampling_kwargs}
        },
    )

    (video,) = processor.apply(inputs, TimingContext(enabled=False))["mm_kwargs"][
        "video"
    ]
    assert video is not None
    pixels = video.get_data()["pixel_values_videos"]
    assert isinstance(pixels, torch.Tensor)
    assert pixels.shape == pixel_shape
    assert pixels.nbytes * 2 == budget


@pytest.mark.parametrize(
    ("model_id", "max_model_len"),
    [
        ("zai-org/GLM-4.1V-9B-Thinking", 16384),
        ("AIDC-AI/Ovis2.5-2B", 32768),
        ("OpenGVLab/InternVL3-1B", 16384),
        ("internlm/Intern-S1", 16384),
        ("baidu/ERNIE-4.5-VL-28B-A3B-PT", 32768),
    ],
)
def test_video_budget_matches_profiled_video(model_id: str, max_model_len: int):
    ctx = build_model_context(
        model_id,
        model_config_kwargs={"max_model_len": max_model_len},
        limit_mm_per_prompt={"image": 1, "video": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    budget = get_mm_max_toks_per_item(processor, {"image": 1, "video": 1}, None)
    dummy = processor.get_dummy_mm_inputs({"video": 1})
    num_embeds = sum(p.get_num_embeds() for p in dummy["mm_placeholders"]["video"])

    assert budget["video"] == num_embeds


@pytest.mark.parametrize(
    ("model_id", "max_model_len"),
    [("baidu/ERNIE-4.5-VL-28B-A3B-PT", 16384)],
)
def test_video_budget_ignores_image_count(model_id: str, max_model_len: int):
    ctx = build_model_context(
        model_id,
        model_config_kwargs={"max_model_len": max_model_len},
        limit_mm_per_prompt={"image": 1, "video": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    with_image = get_mm_max_toks_per_item(processor, {"image": 1, "video": 1}, None)
    video_only = get_mm_max_toks_per_item(processor, {"video": 1}, None)

    assert with_image["video"] == video_only["video"]


ERNIE = "baidu/ERNIE-4.5-VL-28B-A3B-PT"


def _video_embeds(outputs) -> int:
    return sum(p.get_num_embeds() for p in outputs["mm_placeholders"]["video"])


@pytest.mark.parametrize(
    ("hint_width", "hint_height", "width", "height", "mm_processor_kwargs"),
    [
        (1288, 728, 1288, 728, None),
        (728, 1288, 728, 1288, None),
        (1288, None, 1288, 728, None),
        (None, 1288, 728, 1288, None),
        (1920, None, 1288, 728, None),
        (1920, 1080, 1288, 728, None),
        (896, None, 896, 448, {"video_min_pixels": 401408, "video_max_pixels": 401408}),
    ],
)
def test_ernie45_vl_dummy_video_follows_size_hint(
    hint_width: int | None,
    hint_height: int | None,
    width: int,
    height: int,
    mm_processor_kwargs: dict | None,
):
    hint = {"count": 1, "num_frames": 2, "width": hint_width, "height": hint_height}
    ctx = build_model_context(
        ERNIE,
        model_config_kwargs={"max_model_len": 32768},
        mm_processor_kwargs=mm_processor_kwargs,
        limit_mm_per_prompt={
            "image": 1,
            "video": {k: v for k, v in hint.items() if v is not None},
        },
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    frames = np.zeros((2, height, width, 3), dtype=np.uint8)
    metadata = {
        "fps": 2.0,
        "duration": 1.0,
        "total_num_frames": 2,
        "frames_indices": [0, 1],
        "video_backend": "opencv",
        "do_sample_frames": False,
    }
    inputs = ProcessorInputs(
        prompt=processor.info.get_tokenizer().encode("<|video@placeholder|>"),
        mm_data_items=processor.info.parse_mm_data({"video": [(frames, metadata)]}),
    )
    real = processor.apply(inputs, TimingContext(enabled=False))
    dummy = processor.get_dummy_mm_inputs({"video": 1})

    assert _video_embeds(dummy) == _video_embeds(real)


def test_ernie45_vl_video_budget_ignores_image_pixel_limit():
    ctx = build_model_context(
        ERNIE,
        model_config_kwargs={"max_model_len": 2048},
        mm_processor_kwargs={"max_pixels": 153664},
        limit_mm_per_prompt={"image": 1, "video": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    budget = get_mm_max_toks_per_item(processor, {"image": 1, "video": 1}, None)
    dummy = processor.get_dummy_mm_inputs({"video": 1})

    assert budget["video"] == _video_embeds(dummy)
