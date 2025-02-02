# SPDX-License-Identifier: Apache-2.0

import os
from typing import Optional, Union

from transformers import AutoConfig, PretrainedConfig


class EAGLEConfig(PretrainedConfig):
    model_type = "eagle"

    def __init__(self,
                 model: Union[PretrainedConfig, dict, None] = None,
                 truncated_vocab_size: Optional[int] = None,
                 **kwargs):

        model_config = None if model is None else (AutoConfig.for_model(
            **model) if isinstance(model, dict) else model)

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

        if "architectures" not in kwargs:
            kwargs["architectures"] = ["EAGLEModel"]

        super().__init__(**kwargs)

        if self.model is not None:
            for k, v in self.model.to_dict().items():
                if not hasattr(self, k):
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
