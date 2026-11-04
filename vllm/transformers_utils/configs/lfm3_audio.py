# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from transformers import Lfm2Config, PreTrainedConfig

from .lfm3 import Lfm3Config


def _lfm2_config(values: dict[str, Any]) -> Lfm2Config:
    values = dict(values)
    # The training export stores the encoder as LFM2; the transformers-native
    # export as an LFM3 stack without the attention gate and sandwich norms,
    # which is the same network.
    if values.pop("model_type", "lfm2") == "lfm3":
        if values.pop("use_attention_gate", True) or values.pop(
            "use_sandwich_norm", True
        ):
            raise ValueError("The LFM3-Audio encoder has no attention gate")
        values.pop("head_dim", None)
        values.pop("sliding_window", None)
        (values["rope_parameters"],) = values["rope_parameters"].values()
        values["layer_types"] = [
            "full_attention" if layer_type == "sliding_attention" else layer_type
            for layer_type in values["layer_types"]
        ]
        values["block_auto_adjust_ff_dim"] = False
    values.pop("dtype", None)
    return Lfm2Config(**values)


class Lfm3AudioEncoderConfig(PreTrainedConfig):
    model_type = "lfm3_audio_encoder"
    sub_configs = {"text_config": Lfm2Config}

    def __init__(
        self,
        text_config: dict | Lfm2Config | None = None,
        num_mel_bins: int = 128,
        conv_channels: int = 256,
        hidden_size: int = 768,
        sliding_window: int | None = None,
        **kwargs: Any,
    ) -> None:
        num_mel_bins = kwargs.pop("dim_in", num_mel_bins)
        conv_channels = kwargs.pop("dim_conv", conv_channels)
        hidden_size = kwargs.pop("dim_out", hidden_size)
        backbone = kwargs.pop("lfm", text_config)
        if isinstance(backbone, dict):
            sliding_window = sliding_window or backbone.get("sliding_window")
            backbone = _lfm2_config(backbone)
        self.text_config: Lfm2Config = backbone or Lfm2Config()
        self.num_mel_bins = num_mel_bins
        self.conv_channels = conv_channels
        self.hidden_size = hidden_size
        # Past frames each frame attends to, itself included
        self.sliding_window = sliding_window or 32
        super().__init__(**kwargs)


class Lfm3AudioConfig(PreTrainedConfig):
    model_type = "lfm3_audio"
    sub_configs = {
        "text_config": Lfm3Config,
        "audio_encoder_config": Lfm3AudioEncoderConfig,
    }

    def __init__(
        self,
        text_config: dict | Lfm3Config | None = None,
        audio_encoder_config: dict | Lfm3AudioEncoderConfig | None = None,
        audio_token_id: int = 125_031,
        audio_start_token_id: int = 125_012,
        mixed_start_token_id: int = 125_015,
        **kwargs: Any,
    ) -> None:
        # The hub checkpoints keep the training export's `lfm` / `encoder` keys
        text_config = kwargs.pop("lfm", text_config)
        audio_encoder_config = kwargs.pop("encoder", audio_encoder_config)
        self.text_config: Lfm3Config = (
            Lfm3Config(**text_config)
            if isinstance(text_config, dict)
            else text_config or Lfm3Config()
        )
        self.audio_encoder_config: Lfm3AudioEncoderConfig = (
            Lfm3AudioEncoderConfig(**audio_encoder_config)
            if isinstance(audio_encoder_config, dict)
            else audio_encoder_config or Lfm3AudioEncoderConfig()
        )
        self.audio_token_id = audio_token_id
        self.audio_start_token_id = audio_start_token_id
        self.mixed_start_token_id = mixed_start_token_id
        super().__init__(**kwargs)
