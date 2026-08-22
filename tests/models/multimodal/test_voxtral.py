# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.model_executor.models.voxtral import VoxtralForConditionalGeneration


def _make_voxtral_model(downsample_factor: int) -> VoxtralForConditionalGeneration:
    model = VoxtralForConditionalGeneration.__new__(VoxtralForConditionalGeneration)
    model.downsample_factor = downsample_factor
    return model


@pytest.mark.parametrize(
    ("downsample_factor", "num_audio_tokens"),
    [(2, 1), (2, 50), (4, 1), (4, 375)],
)
def test_voxtral_mm_lora_token_counts(
    downsample_factor: int, num_audio_tokens: int
) -> None:
    # The whisper encoder runs on `downsample_factor` tokens per language-model
    # audio token; the adapter (connector) maps them back down. The tower and
    # connector LoRA mappings rely on this round-trip (see gpu_model_runner).
    model = _make_voxtral_model(downsample_factor)

    num_encoder_tokens = num_audio_tokens * downsample_factor
    assert model.get_num_mm_encoder_tokens(num_audio_tokens) == num_encoder_tokens
    assert model.get_num_mm_connector_tokens(num_encoder_tokens) == num_audio_tokens
