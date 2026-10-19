# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Whisper's multimodal preprocessing."""

import numpy as np
import pytest

from vllm.multimodal import MULTIMODAL_REGISTRY

from ...utils import build_model_context


@pytest.mark.parametrize("model_id", ["openai/whisper-large-v3-turbo"])
@pytest.mark.parametrize("audio_duration_s", [5, 30, 35])
def test_audio_features_fit_encoder_window(
    model_id: str,
    audio_duration_s: int,
) -> None:
    """The encoder only has positions for one 30 s window, so a longer clip
    has to be cut to that window like the HF feature extractor does by default.
    """
    ctx = build_model_context(model_id, limit_mm_per_prompt={"audio": 1})
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    feature_extractor = processor.info.get_feature_extractor()

    sampling_rate = feature_extractor.sampling_rate
    rng = np.random.RandomState(0)
    audio = rng.rand(sampling_rate * audio_duration_s).astype(np.float32)

    processed_inputs = processor(
        "<|startoftranscript|>",
        mm_items=processor.info.parse_mm_data({"audio": [(audio, sampling_rate)]}),
        hf_processor_mm_kwargs={},
    )

    input_features = processed_inputs["mm_kwargs"].get_data()["input_features"]
    assert input_features.shape[-1] == feature_extractor.nb_max_frames
