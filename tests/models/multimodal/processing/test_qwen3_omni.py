# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Qwen3 Omni audio processing and sample rate handling."""

from typing import Any

import numpy as np
import pytest

from vllm.multimodal import MULTIMODAL_REGISTRY

from ...utils import build_model_context


@pytest.mark.parametrize("model_id", ["Qwen/Qwen3-Omni-30B-A3B-Instruct"])
@pytest.mark.parametrize(
    ("audio_sample_rate", "audio_duration_sec"),
    [
        (16000, 1.0),  # Native Whisper sample rate, 1 second
        (16000, 2.0),  # Native Whisper sample rate, 2 seconds
    ],
)
def test_processor_with_audio_sample_rate(
    model_id: str,
    audio_sample_rate: int,
    audio_duration_sec: float,
) -> None:
    """Test that vLLM's processor generates expected outputs with audio_sample_rate.

    This validates that the processor correctly handles audio_sample_rate
    passed via hf_processor_mm_kwargs and generates audio tokens.
    """
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"audio": 1, "image": 0, "video": 0},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    tokenizer = processor.info.get_tokenizer()

    # Create audio data at the specified sample rate
    audio_length = int(audio_sample_rate * audio_duration_sec)
    rng = np.random.RandomState(42)
    audio_data = rng.rand(audio_length).astype(np.float32)

    # Build prompt with audio placeholder
    prompt = "<|audio_start|><|audio_pad|><|audio_end|>"
    mm_data = {"audio": [(audio_data, audio_sample_rate)]}

    # Apply processor with audio_sample_rate in mm_kwargs
    hf_processor_mm_kwargs: dict[str, Any] = {
        "audio_sample_rate": audio_sample_rate,
    }
    processed_inputs = processor(
        prompt,
        mm_items=processor.info.parse_mm_data(mm_data),
        hf_processor_mm_kwargs=hf_processor_mm_kwargs,
    )

    # Verify audio tokens are generated
    hf_processor = processor.info.get_hf_processor(**hf_processor_mm_kwargs)
    audio_token_id = tokenizer.convert_tokens_to_ids(hf_processor.audio_token)
    aud_tok_count = processed_inputs["prompt_token_ids"].count(audio_token_id)

    assert aud_tok_count >= 1, (
        f"Expected at least 1 audio token but got {aud_tok_count}. "
        f"sample_rate: {audio_sample_rate}Hz, duration: {audio_duration_sec}s"
    )


@pytest.mark.parametrize("model_id", ["Qwen/Qwen3-Omni-30B-A3B-Instruct"])
def test_longer_audio_generates_more_tokens(model_id: str) -> None:
    """Test that longer audio generates more tokens than shorter audio.

    This validates that audio_sample_rate is being used correctly by checking
    that audio duration affects token count as expected.
    """
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"audio": 1, "image": 0, "video": 0},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    tokenizer = processor.info.get_tokenizer()

    audio_sample_rate = 16000
    rng = np.random.RandomState(42)

    def get_token_count(duration: float) -> int:
        audio_length = int(audio_sample_rate * duration)
        audio_data = rng.rand(audio_length).astype(np.float32)
        prompt = "<|audio_start|><|audio_pad|><|audio_end|>"
        mm_data = {"audio": [(audio_data, audio_sample_rate)]}
        hf_processor_mm_kwargs: dict[str, Any] = {
            "audio_sample_rate": audio_sample_rate,
        }
        processed = processor(
            prompt,
            mm_items=processor.info.parse_mm_data(mm_data),
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
        )
        hf_proc = processor.info.get_hf_processor(**hf_processor_mm_kwargs)
        audio_token_id = tokenizer.convert_tokens_to_ids(hf_proc.audio_token)
        return processed["prompt_token_ids"].count(audio_token_id)

    short_tokens = get_token_count(1.0)
    long_tokens = get_token_count(2.0)

    assert long_tokens > short_tokens, (
        f"Expected longer audio (2s) to have more tokens than shorter (1s). "
        f"Got short={short_tokens}, long={long_tokens}"
    )


@pytest.mark.parametrize("model_id", ["Cloudflare/clef-omni"])
def test_audio_in_video_follows_requested_fps(model_id: str) -> None:
    """Audio and video tokens interleave by the fps the video was processed
    at, as in HF, not by the default of one frame a second."""
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"audio": 1, "image": 0, "video": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    rng = np.random.RandomState(42)
    video = rng.randint(0, 255, (4, 64, 64, 3), dtype=np.uint8)
    audio = rng.rand(32000).astype(np.float32)
    prompt = "<|vision_start|><|video_pad|><|vision_end|>"
    mm_kwargs: dict[str, Any] = {
        "use_audio_in_video": True,
        "videos_kwargs": {"fps": 2.0, "do_sample_frames": False},
    }
    processed = processor(
        prompt,
        mm_items=processor.info.parse_mm_data(
            {"video": [video], "audio": [(audio, 16000)]}
        ),
        hf_processor_mm_kwargs=mm_kwargs,
    )
    expected = processor.info.get_hf_processor()(
        text=[prompt], videos=[video], audio=[audio], return_tensors="pt", **mm_kwargs
    )
    assert processed["prompt_token_ids"] == expected["input_ids"][0].tolist()


def _create_video_processor(
    max_model_len: int,
    mm_processor_kwargs: dict[str, Any] | None = None,
    model_id: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
):
    ctx = build_model_context(
        model_id,
        model_config_kwargs={"max_model_len": max_model_len},
        mm_processor_kwargs=mm_processor_kwargs,
        limit_mm_per_prompt={"audio": 0, "image": 0, "video": 1},
    )
    return MULTIMODAL_REGISTRY.create_processor(ctx.model_config)


def _num_video_embeds(processed) -> int:
    return sum(p.get_num_embeds() for p in processed["mm_placeholders"]["video"])


@pytest.mark.parametrize("max_model_len", [16384, 32768])
def test_video_budget_reaches_max_model_len(max_model_len: int) -> None:
    processor = _create_video_processor(max_model_len)
    budget = processor.info.get_mm_max_tokens_per_item(max_model_len, {"video": 1})
    assert budget["video"] == max_model_len


def test_fps_sampled_video_fits_video_budget() -> None:
    max_model_len = 16384
    processor = _create_video_processor(max_model_len)
    video = np.zeros((60, 504, 896, 3), dtype=np.uint8)
    processed = processor(
        "<|vision_start|><|video_pad|><|vision_end|>",
        mm_items=processor.info.parse_mm_data({"video": [video]}),
        hf_processor_mm_kwargs={
            "videos_kwargs": {"fps": 2.0, "do_sample_frames": False}
        },
    )
    budget = processor.info.get_mm_max_tokens_per_item(max_model_len, {"video": 1})
    assert len(processed["prompt_token_ids"]) <= max_model_len
    assert _num_video_embeds(processed) == 13440
    assert _num_video_embeds(processed) <= budget["video"]


@pytest.mark.parametrize(
    "model_id", ["Qwen/Qwen3-Omni-30B-A3B-Instruct", "Cloudflare/clef-omni"]
)
@pytest.mark.parametrize("max_model_len", [4096, 16384])
def test_dummy_video_fills_video_budget(model_id: str, max_model_len: int) -> None:
    processor = _create_video_processor(max_model_len, model_id=model_id)
    budget = processor.info.get_mm_max_tokens_per_item(max_model_len, {"video": 1})
    num_embeds = _num_video_embeds(processor.get_dummy_mm_inputs({"video": 1}))
    assert budget["video"] - 768 < num_embeds <= budget["video"]


@pytest.mark.parametrize(
    ("model_id", "mm_processor_kwargs"),
    [
        ("Qwen/Qwen3-Omni-30B-A3B-Instruct", None),
        ("Qwen/Qwen3-Omni-30B-A3B-Instruct", {"max_pixels": 256 * 32 * 32}),
        (
            "Qwen/Qwen3-Omni-30B-A3B-Instruct",
            {
                "videos_kwargs": {
                    "size": {
                        "shortest_edge": 128 * 32 * 32,
                        "longest_edge": 256 * 32 * 32,
                    }
                }
            },
        ),
        ("Cloudflare/clef-omni", None),
    ],
)
def test_video_canvas_matches_processed_frame_budget(
    model_id: str,
    mm_processor_kwargs: dict[str, Any] | None,
) -> None:
    processor = _create_video_processor(16384, mm_processor_kwargs, model_id)
    canvas = processor.info.get_video_size_with_most_features()
    video = np.zeros((2, canvas.height * 2, canvas.width * 2, 3), dtype=np.uint8)
    processed = processor(
        "<|vision_start|><|video_pad|><|vision_end|>",
        mm_items=processor.info.parse_mm_data({"video": [video]}),
        hf_processor_mm_kwargs={"videos_kwargs": {"do_sample_frames": False}},
    )
    assert _num_video_embeds(processed) == canvas.width * canvas.height // (32 * 32)


def test_image_budget_is_unchanged() -> None:
    ctx = build_model_context(
        "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        limit_mm_per_prompt={"audio": 0, "image": 1, "video": 0},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    budget = processor.info.get_mm_max_tokens_per_item(4096, {"image": 1})
    assert budget is not None
    assert budget["image"] == 12544
