# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import (
    Gemma4Config,
    Gemma4TextConfig,
    Gemma4VisionConfig,
    LlamaConfig,
    WhisperConfig,
)

from vllm.transformers_utils.configs.aprielvoice import AprielVoiceConfig


@pytest.mark.parametrize("with_vision", [False, True])
def test_aprielvoice_config_roundtrip_preserves_backbone(tmp_path, with_vision):
    """Saving a wrapper must not replace its backbone with the default Llama."""
    backbone = (
        Gemma4Config(
            text_config=Gemma4TextConfig(hidden_size=128).to_dict(),
            vision_config=Gemma4VisionConfig().to_dict(),
        )
        if with_vision
        else LlamaConfig(hidden_size=128)
    )
    config = AprielVoiceConfig(
        text_config=backbone.to_dict(),
        audio_config=WhisperConfig(d_model=32).to_dict(),
        stack_factor=4,
        audio_token_index=123,
    )
    config.save_pretrained(tmp_path)
    restored = AprielVoiceConfig.from_pretrained(tmp_path)
    assert type(restored.wrapped_model_config) is type(backbone)
    assert restored.text_config.hidden_size == 128
    assert restored.audio_config.d_model == 32
    assert restored.stack_factor == 4
    assert restored.audio_token_index == 123
    assert (restored.vision_config is not None) == with_vision
