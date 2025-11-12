# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

from transformers import AutoConfig, DeepseekV2Config, PretrainedConfig


class EAGLEConfig(PretrainedConfig):
    model_type = "eagle"

    def __init__(
        self,
        model: PretrainedConfig | dict | None = None,
        truncated_vocab_size: int | None = None,
        method: str | None = "eagle",
        **kwargs,
    ):
        model_config: PretrainedConfig | DeepseekV2Config | None
        if isinstance(model, dict):
            model_config = AutoConfig.for_model(**model)
        else:
            model_config = model

        for k, v in kwargs.items():
            if k != "architectures" and k != "model_type" and hasattr(model_config, k):
                setattr(model_config, k, v)

        self.model = model_config

        if self.model is None:
            self.truncated_vocab_size = None
        else:
            self.truncated_vocab_size = (
                self.model.vocab_size
                if truncated_vocab_size is None
                else truncated_vocab_size
            )

        # Eagle model name should follow naming convention of
        # LlamaForCausalLM -> EagleLlamaForCausalLM
        # LlamaForCausalLM -> Eagle3LlamaForCausalLM
        # LlamaForCausalLMEagle3 -> LlamaForCausalLMEagle3
        if method == "eagle":
            assert self.model is not None, (
                "model should not be None when method is eagle"
            )
            kwargs["architectures"] = [
                f"Eagle{arch}" if not arch.startswith("Eagle") else arch
                for arch in self.model.architectures
            ]

        elif method == "eagle3":
            assert self.model is not None, (
                "model should not be None when method is eagle3"
            )
            kwargs["architectures"] = [
                arch
                if arch.startswith("Eagle3") or arch.endswith("Eagle3")
                else f"Eagle3{arch}"
                for arch in self.model.architectures
            ]
        else:
            raise ValueError(
                f"Invalid method {method}. Supported methods are eagle and eagle3."
            )

        super().__init__(**kwargs)

        if self.model is not None:
            for k, v in self.model.to_dict().items():
                if k not in kwargs:
                    setattr(self, k, v)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        **kwargs,
    ) -> "EAGLEConfig":
        config_dict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path, **kwargs
        )
        return cls.from_dict(config_dict, **kwargs)
