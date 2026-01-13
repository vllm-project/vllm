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
    """
    Test that vLLM's processor generates expected outputs with audio_sample_rate.

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
    processed_inputs = processor.apply(prompt, mm_data, hf_processor_mm_kwargs)

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
    """
    Test that longer audio generates more tokens than shorter audio.

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
        processed = processor.apply(prompt, mm_data, hf_processor_mm_kwargs)
        hf_proc = processor.info.get_hf_processor(**hf_processor_mm_kwargs)
        audio_token_id = tokenizer.convert_tokens_to_ids(hf_proc.audio_token)
        return processed["prompt_token_ids"].count(audio_token_id)

    short_tokens = get_token_count(1.0)
    long_tokens = get_token_count(2.0)

    assert long_tokens > short_tokens, (
        f"Expected longer audio (2s) to have more tokens than shorter (1s). "
        f"Got short={short_tokens}, long={long_tokens}"
    )
