# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import Optional, Union

from transformers import AutoConfig, PretrainedConfig

from vllm.transformers_utils.configs.deepseek_vl2 import DeepseekV2Config


class EAGLEConfig(PretrainedConfig):
    model_type = "eagle"

    def __init__(self,
                 model: Union[PretrainedConfig, dict, None] = None,
                 truncated_vocab_size: Optional[int] = None,
                 method: Optional[str] = 'eagle',
                 **kwargs):

        model_config: Union[PretrainedConfig, DeepseekV2Config, None]
        if isinstance(model, dict):
            archs = model.get("architectures", [])
            target_archs = ["DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM"]
            if any(target_arch in archs for target_arch in target_archs):
                # AutoConfig does not support DeepSeek MoE models yet
                model_config = DeepseekV2Config(**model)
            else:
                model_config = AutoConfig.for_model(**model)
        else:
            model_config = model

        for k, v in kwargs.items():
            if k != "architectures" and k != "model_type" and hasattr(
                    model_config, k):
                setattr(model_config, k, v)

        self.model = model_config

        if self.model is None:
            self.truncated_vocab_size = None
        else:
            self.truncated_vocab_size = self.model.vocab_size if \
                truncated_vocab_size is None else truncated_vocab_size

        # Eagle model name should follow naming convention of
        # LlamaForCausalLM  EagleLlamaForCausalLM
        # LlamaForCausalLM -> Eagle3LlamaForCausalLM
        # LlamaForCausalLMEagle3 -> LlamaForCausalLMEagle3
        # Qwen2ForCausalLM          → EagleQwen2ForCausalLM
        # Qwen2ForCausalLMEagle     → EagleQwen2ForCausalLM
        # EagleQwen2ForCausalLM     → EagleQwen2ForCausalLM
        # LlamaForCausalLM          → EagleLlamaForCausalLM
        # LlamaForCausalLMEagle     → EagleLlamaForCausalLM
        # LlamaForCausalLM      ->  EagleLlamaForCausalLM
        # Qwen2ForCausalLM      ->  EagleQwen2ForCausalLM
        # Qwen2ForCausalLMEagle ->  EagleQwen2ForCausalLM
        # EagleQwen2ForCausalLM ->  EagleQwen2ForCausalLM
        if method == "eagle":
            assert self.model is not None, \
                "model should not be None when method is eagle"

            def normalize_eagle_arch(arch):
                # Remove Eagle suffix if present, then add Eagle prefix
                if arch.endswith("Eagle"):
                    base_arch = arch[:-5]  # Remove "Eagle" suffix
                elif arch.startswith("Eagle"):
                    return arch  # Already has Eagle prefix
                else:
                    base_arch = arch
                return f"Eagle{base_arch}"

            kwargs["architectures"] = [
                normalize_eagle_arch(arch) for arch in self.model.architectures
            ]
        # Eagle3 model name should follow naming convention of
        # LlamaForCausalLM      ->  Eagle3LlamaForCausalLM
        elif method == "eagle3":
            assert self.model is not None, \
                "model should not be None when method is eagle3"
            kwargs["architectures"] = [
                arch if arch.startswith("Eagle3") or arch.endswith("Eagle3")
                else f"Eagle3{arch}" for arch in self.model.architectures
            ]
        else:
            raise ValueError(f"Invalid method {method}. "
                             "Supported methods are eagle and eagle3.")

        super().__init__(**kwargs)

        if self.model is not None:
            for k, v in self.model.to_dict().items():
                if k not in kwargs:
                    setattr(self, k, v)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        **kwargs,
    ) -> "EAGLEConfig":
        config_dict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path, **kwargs)
        return cls.from_dict(config_dict, **kwargs)
