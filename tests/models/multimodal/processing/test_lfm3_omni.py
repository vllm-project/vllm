# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.transformers_utils.configs.lfm3_omni import Lfm3OmniConfig

LAYER_TYPES = ["conv", "conv", "full_attention"] * 5 + ["full_attention"] * 2
ENCODER = {"num_mel_bins": 128, "conv_channels": 256}
LFM2_ENCODER = {
    "model_type": "lfm2",
    "vocab_size": 2,
    "hidden_size": 768,
    "intermediate_size": 2560,
    "num_hidden_layers": 17,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "layer_types": LAYER_TYPES,
    "rope_parameters": {"rope_type": "default", "rope_theta": 1e6},
}
LFM3_ENCODER = {
    **LFM2_ENCODER,
    "model_type": "lfm3",
    "intermediate_size": 1792,
    "head_dim": 96,
    "sliding_window": 32,
    "use_attention_gate": False,
    "use_sandwich_norm": False,
    "layer_types": [
        "sliding_attention" if t == "full_attention" else t for t in LAYER_TYPES
    ],
    "rope_parameters": {
        "sliding_attention": {"rope_type": "default", "rope_theta": 1e6}
    },
}
# The hub checkpoints keep an LFM2 encoder; the transformers-native export an
# LFM3 one without the attention gate and sandwich norms
HUB = {
    **ENCODER,
    "model_type": "lfm3_omni_encoder",
    "sliding_window": 32,
    "text_config": LFM2_ENCODER,
}
TRANSFORMERS_NATIVE = {**ENCODER, "text_config": LFM3_ENCODER}


@pytest.mark.parametrize("encoder", [HUB, TRANSFORMERS_NATIVE])
def test_config_layouts(encoder):
    config = Lfm3OmniConfig(
        text_config={"model_type": "lfm3"},
        vision_config={"model_type": "lfm3_vl_vision"},
        audio_encoder_config=encoder,
        depth_config={"model_type": "lfm3"},
    )
    assert (config.image_token_id, config.video_token_id) == (125030, 125029)
    assert config.audio_token_id == 125031
    assert config.vision_config.spatial_merge_size == 2
    audio_encoder = config.audio_encoder_config
    assert audio_encoder.sliding_window == 32
    assert audio_encoder.text_config.model_type == "lfm2"
    assert audio_encoder.text_config.layer_types == LAYER_TYPES
    restored = Lfm3OmniConfig(**config.to_dict())
    assert restored.audio_encoder_config.to_dict() == audio_encoder.to_dict()
    assert restored.vision_config.to_dict() == config.vision_config.to_dict()
