# Adapted from https://github.com/fixie-ai/ultravox/blob/ecd58c4041030bae2ad15aa6bcf04ab43199ea02/ultravox/model/ultravox_config.py
from typing import Any, Dict, Optional

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
            Custom audio config or dict
        text_config (`Union[AutoConfig, dict]`, *optional*):
            The config object of the text backbone. Can be any of `LlamaConfig`
            or `MistralConfig`.
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
        text_model_lora_config (`LoraConfigSimplified`, *optional*):
            The LoRA configuration for finetuning the text model.
        audio_model_lora_config (`LoraConfigSimplified`, *optional*):
            The LoRA configuration for finetuning the audio model.
    """

    model_type = "ultravox"
    is_composition = False

    def __init__(
        self,
        audio_config: Optional[Dict[str, Any]] = None,
        text_config: Optional[Dict[str, Any]] = None,
        audio_model_id: Optional[str] = None,
        text_model_id: Optional[str] = None,
        ignore_index: int = -100,
        audio_token_index: int = 32000,
        hidden_size: int = 4096,
        stack_factor: int = 8,
        norm_init: float = 0.4,
        projector_act: str = "swiglu",
        text_model_lora_config: Optional[Dict[str, Any]] = None,
        audio_model_lora_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.ignore_index = ignore_index

        self.audio_model_id = audio_model_id
        self.text_model_id = text_model_id
        self.audio_token_index = audio_token_index

        self.hidden_size = hidden_size
        self.stack_factor = stack_factor
        self.norm_init = norm_init
        self.projector_act = projector_act

        if text_model_id is not None:
            # Avoid circular import
            from vllm.transformers_utils.config import get_config

            self.text_config = get_config(text_model_id,
                                          trust_remote_code=False)
        else:
            text_config = text_config or {}
            self.text_config = transformers.CONFIG_MAPPING[text_config.get(
                "model_type", "llama")](**text_config)

        if audio_model_id is not None:
            # Avoid circular import
            from vllm.transformers_utils.config import get_config

            self.audio_config = get_config(audio_model_id,
                                           trust_remote_code=False)
        else:
            audio_config = audio_config or {}
            self.audio_config = transformers.CONFIG_MAPPING[audio_config.get(
                "model_type", "whisper")](**audio_config)

        self.text_model_lora_config = text_model_lora_config or {}
        self.audio_model_lora_config = audio_model_lora_config or {}

        self.vocab_size = self.text_config.vocab_size

        self.initializer_range = self.text_config.initializer_range

        super().__init__(**kwargs)
