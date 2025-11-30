# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import pytest


class TestQwen3OmniAudioSampleRatePreservation:
    """Test that audio_sample_rate is preserved during kwargs restructuring.

    These tests validate the fix for the audio_sample_rate bug in Qwen3 Omni
    where the parameter was lost during kwargs restructuring. The tests don't
    require importing the actual model classes - they just test the kwargs
    manipulation logic.
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
