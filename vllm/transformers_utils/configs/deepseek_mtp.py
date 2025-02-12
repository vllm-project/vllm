# SPDX-License-Identifier: Apache-2.0
import os
from typing import Union

from transformers import PretrainedConfig

from vllm.transformers_utils.configs.deepseek_v3 import DeepseekV3Config


class DeepSeekMTPConfig(PretrainedConfig):
    model_type = "deepseek_mtp"

    def __init__(self,
                 model: Union[PretrainedConfig, dict, None] = None,
                 **kwargs):
        print("model: %s", model)
        if model is not None:
            self.model = DeepseekV3Config.from_dict(model, **kwargs)
        else:
            self.model = None

        if self.model is not None:
            for k, v in kwargs.items():
                if k != "architectures" and k != "model_type" and hasattr(
                        self.model, k):
                    setattr(self.model, k, v)

        if "architectures" not in kwargs:
            kwargs["architectures"] = ["DeepSeekMTPModel"]

        super().__init__(**kwargs)

        if self.model is not None:
            for k, v in self.model.to_dict().items():
                if not hasattr(self, k):
                    setattr(self, k, v)
            # for loading MTP kv cache
            self.model.num_hidden_layers = self.model.num_nextn_predict_layers

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        **kwargs,
    ) -> "DeepSeekMTPConfig":
        config_dict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path, **kwargs)
        return cls.from_dict(config_dict, **kwargs)
