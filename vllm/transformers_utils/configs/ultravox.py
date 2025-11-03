# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from https://github.com/fixie-ai/ultravox/blob/ecd58c4041030bae2ad15aa6bcf04ab43199ea02/ultravox/model/ultravox_config.py
from typing import Any

import transformers


class UltravoxConfig(transformers.PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a
    [`UltravoxForConditionalGeneration`]. It is used to instantiate an
    Ultravox model according to the specified arguments, defining the model
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to
    control the model outputs. Read the documentation from [`PretrainedConfig`]
    for more information.

    Args:
        audio_config (`Union[AutoConfig, dict]`,  *optional*):
            Custom audio config or dict.
        text_config (`Union[AutoConfig, dict]`, *optional*):
            The config object of the text backbone.
        audio_model_id (`str`, *optional*):
            The model ID of the audio backbone.
        text_model_id (`str`, *optional*):
            The model ID of the text backbone.
        ignore_index (`int`, *optional*, defaults to -100):
            The ignore index for the loss function.
        audio_token_index (`int`, *optional*, defaults to 32000):
            The audio token index to encode the audio prompt.
        stack_factor (`int`, *optional*, defaults to 8):
            Audio downsampling factor for the multimodal projector.
        norm_init (`float`, *optional*, defaults to 0.4):
            The initialization value for the layer normalization.
        projector_act (`str`, *optional*, defaults to `"swiglu"`):
            The activation function used by the multimodal projector.
        projector_ln_mid (`bool`, *optional*, defaults to `False`):
            Whether to apply layer normalization at the middle of the
            projector or at the end. Versions v0.4.1 and below
            use `False`, but v0.5 and above use `True`.
    """

    wrapped_model_config: transformers.PretrainedConfig
    model_type = "ultravox"
    audio_token = "<|audio|>"
    is_composition = False

    def __init__(
        self,
        audio_config: dict[str, Any] | None = None,
        text_config: dict[str, Any] | None = None,
        audio_model_id: str | None = None,
        text_model_id: str | None = None,
        ignore_index: int = -100,
        audio_token_index: int = 32000,
        hidden_size: int = 4096,
        stack_factor: int = 8,
        norm_init: float = 0.4,
        projector_act: str = "swiglu",
        projector_ln_mid: bool = False,
        **kwargs,
    ):
        self.ignore_index = ignore_index
        self.audio_token_index = audio_token_index

        self.hidden_size = hidden_size
        self.stack_factor = stack_factor
        self.norm_init = norm_init
        self.projector_act = projector_act
        self.projector_ln_mid = projector_ln_mid

        # N.B. May set the wrapped_model_config below.
        self.text_model_id = text_model_id
        if text_model_id is None:
            text_config = text_config or {}
            self.wrapped_model_config = transformers.CONFIG_MAPPING[
                text_config.get("model_type", "llama")
            ](**text_config)

        # N.B. May set the audio_config below.
        self.audio_model_id = audio_model_id
        if audio_model_id is None:
            self.audio_model_id = None
            audio_config = audio_config or {}
            self.audio_config = transformers.CONFIG_MAPPING[
                audio_config.get("model_type", "whisper")
            ](**audio_config)

        super().__init__(**kwargs)

    def __setattr__(self, key, value):
        # Since --hf-overrides are applied _after_ the UltravoxConfig is
        # instantiated, load the configs implicitly when assigning text_model_id
        # or audio_model_id. This allows:
        #
        #   --hf-overrides.text_model_id=<quantized variant>
        #
        # to behave as intended.
        if key == "text_model_id" and value is not None:
            from vllm.transformers_utils.config import get_config

            self.wrapped_model_config = get_config(value, trust_remote_code=False)
        elif key == "audio_model_id" and value is not None:
            from vllm.transformers_utils.config import get_config

            self.audio_config = get_config(value, trust_remote_code=False)

        return super().__setattr__(key, value)

    @property
    def text_config(self) -> transformers.PretrainedConfig:
        # When Ultravox wraps a multi-modal model (e.g. Gemma), we instantiate
        # the full model, but the text config is the text config of the inner
        # model.
        return self.wrapped_model_config.get_text_config()
