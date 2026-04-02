# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from transformers import PretrainedConfig, SiglipVisionConfig
from transformers.models.qwen2 import Qwen2Config


class CheersConfig(PretrainedConfig):
    """Configuration class for Cheers (UMM) model."""

    model_type = "umm"

    def __init__(
        self,
        text_config: dict | Qwen2Config | None = None,
        vision_representation_config: dict | SiglipVisionConfig | None = None,
        vae_encoder_config: dict | None = None,
        vae_decoder_config: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(text_config, dict):
            self.text_config = Qwen2Config(**text_config)
        else:
            self.text_config = text_config or Qwen2Config()

        if isinstance(vision_representation_config, dict):
            self.vision_representation_config = SiglipVisionConfig(
                **vision_representation_config
            )
        else:
            self.vision_representation_config = (
                vision_representation_config or SiglipVisionConfig()
            )

        self.vae_encoder_config = vae_encoder_config or {"resolution": 512}
        self.vae_decoder_config = vae_decoder_config or {"resolution": 512}

    @property
    def hidden_size(self) -> int:
        """Return the hidden size of the language model."""
        return self.text_config.hidden_size
