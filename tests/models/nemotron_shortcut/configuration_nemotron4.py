# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A stand-in for the `configuration_nemotron4.py` that Nemotron Shortcut
checkpoints ship as remote code, for the tests' tiny random-weight models.
vLLM reads the model's fields as attributes, so these configs keep every field
they are given. `utils.save_checkpoint_config` saves this module next to a
tiny model's config.json, so the engine loads it as the checkpoint's own."""

from transformers import PreTrainedConfig
from transformers.models.pixtral import PixtralVisionConfig


class Nemotron4TextConfig(PreTrainedConfig):
    model_type = "nemotron4_text"
    base_config_key = "text_config"

    @property
    def num_hidden_layers(self) -> int:
        return len(self.layer_types)

    def validate_layer_type(self):
        # The model's own block types, e.g. "mlp" and "shortcut_moe_gdp".
        pass


class Nemotron4Config(PreTrainedConfig):
    model_type = "nemotron4"
    sub_configs = {
        "text_config": Nemotron4TextConfig,
        "vision_config": PixtralVisionConfig,
    }

    def __init__(
        self,
        text_config: dict | Nemotron4TextConfig | None = None,
        vision_config: dict | PixtralVisionConfig | None = None,
        **kwargs,
    ):
        kwargs.setdefault("architectures", ["Nemotron4ForConditionalGeneration"])
        super().__init__(**kwargs)
        if not isinstance(text_config, Nemotron4TextConfig):
            text_config = Nemotron4TextConfig(**(text_config or {}))
        if not isinstance(vision_config, PixtralVisionConfig):
            vision_config = PixtralVisionConfig(**(vision_config or {}))
        self.text_config = text_config
        self.vision_config = vision_config
