# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.transformers_utils.configs.deepseek_vl2 import DeepseekVLV2Config


class UnlimitedOCRConfig(DeepseekVLV2Config):
    model_type = "unlimited-ocr"

    def __init__(self, **kwargs):
        kwargs.setdefault("architectures", ["UnlimitedOCRForCausalLM"])
        language_config = kwargs.get("language_config")
        if isinstance(language_config, dict):
            language_config["architectures"] = ["DeepseekV2ForCausalLM"]
            language_config.setdefault("model_type", "deepseek_v2")
        super().__init__(**kwargs)
        self.model_type = "unlimited-ocr"
