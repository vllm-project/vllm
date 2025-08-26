# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# --------------------------------------------------------
# Adapted from
# https://huggingface.co/OpenGVLab/InternVL2-Llama3-76B
# under MIT License. LICENSE is in incl_licenses directory.
# --------------------------------------------------------

from transformers.configuration_utils import PretrainedConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.utils import logging

logger = logging.get_logger(__name__)


class NemotronH_Nano_VL_Config(PretrainedConfig):
    model_type = "NemotronH_Nano_VL"
    is_composition = True

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        force_image_size=None,
        downsample_ratio=0.5,
        template=None,
        ps_version="v1",
        image_tag_type="internvl",
        projector_hidden_size=4096,
        vit_hidden_size=1280,
        attn_implementation="flash_attention_2",
        **kwargs,
    ):
        super().__init__(**kwargs)

        if vision_config is not None:
            assert (
                "auto_map" in vision_config
                and "AutoConfig" in vision_config["auto_map"]
            )
            vision_auto_config = get_class_from_dynamic_module(
                *vision_config["auto_map"]["AutoConfig"].split("--")[::-1]
            )
            self.vision_config = vision_auto_config(**vision_config)
        else:
            self.vision_config = PretrainedConfig()

        if text_config is None:
            text_config = {}
        self.text_config = PretrainedConfig(**text_config)

        # Assign configuration values
        self.force_image_size = force_image_size
        self.downsample_ratio = downsample_ratio
        self.template = template  # TODO move out of here and into the tokenizer
        self.ps_version = ps_version  # Pixel shuffle version
        self.image_tag_type = image_tag_type  # TODO: into the tokenizer too?
        self.projector_hidden_size = projector_hidden_size
        self.vit_hidden_size = vit_hidden_size

        self._attn_implementation = attn_implementation or ""
        self.vision_config.use_flash_attn = isinstance(
            self._attn_implementation, str
        ) and ("flash_attention" in self._attn_implementation)
        self.text_config._attn_implementation = self._attn_implementation

    def get_text_config(self):
        return self.text_config
