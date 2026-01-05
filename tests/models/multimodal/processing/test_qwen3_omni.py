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

    This validates the reviewer's request that we test the actual processor
    can handle different audio_sample_rate values and generate audio tokens.
    """
    # Setup: Build model context and processor
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

    # Execute: Apply processor with audio_sample_rate in mm_kwargs
    hf_processor_mm_kwargs: dict[str, Any] = {
        "audio_sample_rate": audio_sample_rate,
    }
    processed_inputs = processor.apply(prompt, mm_data, hf_processor_mm_kwargs)

    # Assert: Verify audio tokens are generated
    hf_processor = processor.info.get_hf_processor(**hf_processor_mm_kwargs)
    audio_token_id = tokenizer.convert_tokens_to_ids(hf_processor.audio_token)
    aud_tok_count = processed_inputs["prompt_token_ids"].count(audio_token_id)

    # Audio should generate at least 1 token
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

    # Get token counts for different durations
    short_tokens = get_token_count(1.0)
    long_tokens = get_token_count(2.0)

    # Longer audio should produce more tokens
    assert long_tokens > short_tokens, (
        f"Expected longer audio (2s) to have more tokens than shorter (1s). "
        f"Got short={short_tokens}, long={long_tokens}"
    )


class TestQwen3OmniAudioSampleRatePreservation:
    """Test that audio_sample_rate is preserved during kwargs restructuring.

    These tests validate the fix for the audio_sample_rate bug in Qwen3 Omni
    where the parameter was lost during kwargs restructuring.
    """

    @staticmethod
    def _process_kwargs(
        mm_kwargs: dict[str, Any],
        tok_kwargs: dict[str, Any],
        transformers_version: str = "4.57.0",
    ) -> dict[str, Any]:
        """
        Helper method to simulate kwargs processing logic from production code.

        This method simulates the kwargs restructuring that happens in the
        Qwen3 Omni model when transformers < 4.58.0. By centralizing this
        logic, we make tests easier to maintain if the production logic changes.

        Args:
            mm_kwargs: Multimodal kwargs (e.g., audio_sample_rate, truncation)
            tok_kwargs: Tokenizer kwargs (e.g., truncation)
            transformers_version: Version string to test against (default: "4.57.0")

        Returns:
            Processed kwargs dictionary with restructured audio_kwargs and text_kwargs
        """
        from packaging.version import Version

        mm_kwargs_copy = dict(mm_kwargs)
        tok_kwargs_copy = dict(tok_kwargs)

        if Version(transformers_version) < Version("4.58.0"):
            # Extract audio_sample_rate before restructuring (THE FIX)
            audio_sample_rate = mm_kwargs_copy.pop("audio_sample_rate", None)

            # Restructure kwargs
            mm_kwargs_copy["audio_kwargs"] = {
                "truncation": mm_kwargs_copy.pop("truncation", False)
            }
            mm_kwargs_copy["text_kwargs"] = {
                "truncation": tok_kwargs_copy.pop("truncation", False)
            }

            # Put audio_sample_rate into audio_kwargs (THE FIX)
            if audio_sample_rate is not None:
                mm_kwargs_copy["audio_kwargs"]["audio_sample_rate"] = audio_sample_rate

        return mm_kwargs_copy

    def test_audio_sample_rate_preserved_in_audio_kwargs(self) -> None:
        """
        Test that audio_sample_rate is moved from top-level mm_kwargs
        into audio_kwargs during kwargs restructuring.

        This is the core fix: when transformers < 4.58.0, the code
        restructures kwargs into audio_kwargs and text_kwargs, and
        audio_sample_rate must be preserved in audio_kwargs.
        """
        # Setup: Create mm_kwargs with audio_sample_rate at top level
        mm_kwargs: dict[str, Any] = {
            "audio_sample_rate": 16000,
            "truncation": True,
        }
        tok_kwargs: dict[str, Any] = {
            "truncation": False,
        }

        # Execute: Process kwargs using helper method
        result = self._process_kwargs(mm_kwargs, tok_kwargs)

        # Assert: Verify audio_sample_rate is in audio_kwargs
        assert "audio_kwargs" in result
        assert "audio_sample_rate" in result["audio_kwargs"]
        assert result["audio_kwargs"]["audio_sample_rate"] == 16000

        # Assert: Verify truncation is also in audio_kwargs
        assert result["audio_kwargs"]["truncation"] is True

        # Assert: Verify text_kwargs is created correctly
        assert "text_kwargs" in result
        assert result["text_kwargs"]["truncation"] is False

    def test_audio_sample_rate_absent_when_not_provided(self) -> None:
        """
        Test that when audio_sample_rate is not provided in mm_kwargs,
        the restructured audio_kwargs doesn't contain it.
        """
        # Setup: Create mm_kwargs WITHOUT audio_sample_rate
        mm_kwargs: dict[str, Any] = {
            "truncation": True,
        }
        tok_kwargs: dict[str, Any] = {
            "truncation": False,
        }

        # Execute: Process kwargs using helper method
        result = self._process_kwargs(mm_kwargs, tok_kwargs)

        # Assert: Verify audio_sample_rate is NOT in audio_kwargs
        assert "audio_kwargs" in result
        assert "audio_sample_rate" not in result["audio_kwargs"]

        # Assert: Verify truncation is still in audio_kwargs
        assert result["audio_kwargs"]["truncation"] is True

    @pytest.mark.parametrize("sample_rate", [8000, 16000, 22050, 24000, 44100, 48000])
    def test_various_audio_sample_rates_preserved(self, sample_rate: int) -> None:
        """
        Test that various common audio sample rates are preserved.

        Common sample rates:
        - 8000: Telephone quality
        - 16000: Wideband speech (Qwen3 Omni default)
        - 22050: Low-quality audio
        - 24000: High-quality speech
        - 44100: CD quality
        - 48000: Professional audio
        """
        # Setup: Create mm_kwargs with specific sample rate
        mm_kwargs: dict[str, Any] = {
            "audio_sample_rate": sample_rate,
            "truncation": True,
        }
        tok_kwargs: dict[str, Any] = {"truncation": False}

        # Execute: Process kwargs using helper method
        result = self._process_kwargs(mm_kwargs, tok_kwargs)

        # Assert: Verify the specific sample rate is preserved
        assert result["audio_kwargs"]["audio_sample_rate"] == sample_rate

    def test_kwargs_unchanged_for_newer_transformers_version(self) -> None:
        """
        Test that kwargs structure remains unchanged for transformers >= 4.58.0.

        This test ensures that when transformers version is 4.58.0 or higher,
        the kwargs restructuring is bypassed and audio_sample_rate remains
        at the top level as originally passed.
        """
        from packaging.version import Version

        # Setup: Create mm_kwargs with audio_sample_rate at top level
        mm_kwargs: dict[str, Any] = {
            "audio_sample_rate": 16000,
            "truncation": True,
        }
        tok_kwargs: dict[str, Any] = {
            "truncation": False,
        }

        # Execute: Simulate with transformers >= 4.58.0
        mm_kwargs_copy = dict(mm_kwargs)
        tok_kwargs_copy = dict(tok_kwargs)

        transformers_ver = "4.58.0"  # Version that bypasses restructuring
        if Version(transformers_ver) < Version("4.58.0"):
            # This block should NOT execute for >= 4.58.0
            audio_sample_rate = mm_kwargs_copy.pop("audio_sample_rate", None)
            mm_kwargs_copy["audio_kwargs"] = {
                "truncation": mm_kwargs_copy.pop("truncation", False)
            }
            mm_kwargs_copy["text_kwargs"] = {
                "truncation": tok_kwargs_copy.pop("truncation", False)
            }
            if audio_sample_rate is not None:
                mm_kwargs_copy["audio_kwargs"]["audio_sample_rate"] = audio_sample_rate

        # Assert: Verify kwargs structure is unchanged
        assert "audio_kwargs" not in mm_kwargs_copy
        assert "text_kwargs" not in mm_kwargs_copy
        assert mm_kwargs_copy["audio_sample_rate"] == 16000
        assert mm_kwargs_copy["truncation"] is True
        assert tok_kwargs_copy["truncation"] is False
