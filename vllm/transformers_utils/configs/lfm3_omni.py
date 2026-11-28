# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from .lfm3_audio import Lfm3AudioEncoderConfig
from .lfm3_vl import Lfm3VlConfig


class Lfm3OmniConfig(Lfm3VlConfig):
    model_type = "lfm3_omni"
    sub_configs = {
        **Lfm3VlConfig.sub_configs,
        "audio_encoder_config": Lfm3AudioEncoderConfig,
    }

    def __init__(
        self,
        audio_encoder_config: dict | Lfm3AudioEncoderConfig | None = None,
        audio_token_id: int = 125_031,
        audio_start_token_id: int = 125_012,
        mixed_start_token_id: int = 125_015,
        **kwargs: Any,
    ) -> None:
        if audio_encoder_config is None:
            audio_encoder_config = Lfm3AudioEncoderConfig()
        elif isinstance(audio_encoder_config, dict):
            audio_encoder_config = Lfm3AudioEncoderConfig(**audio_encoder_config)
        self.audio_encoder_config: Lfm3AudioEncoderConfig = audio_encoder_config
        self.audio_token_id = audio_token_id
        self.audio_start_token_id = audio_start_token_id
        self.mixed_start_token_id = mixed_start_token_id
        super().__init__(**kwargs)
