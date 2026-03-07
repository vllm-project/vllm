# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
try:
    # Optional dependency: use MusicFlamingo classes when transformers provides them.
    from transformers.models.musicflamingo import (
        MusicFlamingoConfig,
        MusicFlamingoEncoderConfig,
        MusicFlamingoProcessor,
    )
except ImportError:  # pragma: no cover - optional dependency
    from transformers import CONFIG_MAPPING, AutoConfig, PretrainedConfig
    from transformers.models.audioflamingo3 import (
        AudioFlamingo3EncoderConfig,
        AudioFlamingo3Processor,
    )

    class MusicFlamingoProcessor(AudioFlamingo3Processor):  # type: ignore[no-redef]
        pass

    class MusicFlamingoEncoderConfig(AudioFlamingo3EncoderConfig):  # type: ignore[no-redef]
        model_type = "musicflamingo_encoder"

    class MusicFlamingoConfig(PretrainedConfig):  # type: ignore[no-redef]
        model_type = "musicflamingo"
        sub_configs = {
            "audio_config": MusicFlamingoEncoderConfig,
            "text_config": AutoConfig,
        }

        def __init__(
            self,
            audio_config=None,
            text_config=None,
            audio_token_id=151669,
            projector_hidden_act="gelu",
            projector_bias=True,
            **kwargs,
        ):
            self.audio_token_id = audio_token_id

            if isinstance(audio_config, dict):
                audio_config["model_type"] = audio_config.get(
                    "model_type", "musicflamingo_encoder"
                )
                audio_config = MusicFlamingoEncoderConfig(**audio_config)
            elif audio_config is None:
                audio_config = MusicFlamingoEncoderConfig()

            self.audio_config = audio_config

            if isinstance(text_config, dict):
                text_config["model_type"] = text_config.get("model_type", "qwen2")
                text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
            elif text_config is None:
                text_config = CONFIG_MAPPING["qwen2"]()

            self.text_config = text_config
            self.projector_hidden_act = projector_hidden_act
            self.projector_bias = projector_bias

            super().__init__(**kwargs)
