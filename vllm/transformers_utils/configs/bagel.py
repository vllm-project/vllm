# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from transformers import PretrainedConfig, SiglipVisionConfig
from transformers.models.qwen2 import Qwen2Config


class BagelConfig(PretrainedConfig):
    """Configuration class for BAGEL model."""

    model_type = "bagel"

    def __init__(
        self,
        visual_gen: bool = True,
        visual_und: bool = True,
        llm_config: dict | Qwen2Config | None = None,
        vit_config: dict | SiglipVisionConfig | None = None,
        vae_config: dict | None = None,
        latent_patch_size: int = 2,
        max_latent_size: int = 32,
        vit_max_num_patch_per_side: int = 70,
        connector_act: str = "gelu_pytorch_tanh",
        interpolate_pos: bool = False,
        timestep_shift: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.visual_gen = visual_gen
        self.visual_und = visual_und

        # Convert dict configs to proper config objects
        if isinstance(llm_config, dict):
            self.llm_config = Qwen2Config(**llm_config)
        else:
            self.llm_config = llm_config or Qwen2Config()

        if isinstance(vit_config, dict):
            self.vit_config = SiglipVisionConfig(**vit_config)
        else:
            self.vit_config = vit_config or SiglipVisionConfig()

        self.vae_config = vae_config or {"z_channels": 16, "downsample": 8}
        self.latent_patch_size = latent_patch_size
        self.max_latent_size = max_latent_size
        self.vit_max_num_patch_per_side = vit_max_num_patch_per_side
        self.connector_act = connector_act
        self.interpolate_pos = interpolate_pos
        self.timestep_shift = timestep_shift

    @property
    def hidden_size(self) -> int:
        """Return the hidden size of the language model."""
        return self.llm_config.hidden_size
