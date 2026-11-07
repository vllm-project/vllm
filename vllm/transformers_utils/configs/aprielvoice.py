# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import transformers


class AprielVoiceConfig(transformers.PreTrainedConfig):
    """Configuration for AprielVoice with audio and optional Gemma4 vision.

    Args:
        audio_config: Whisper encoder config (dict or AutoConfig).
        text_config: LLM backbone config (dict or AutoConfig).
        audio_model_id: HF model id for pretrained audio encoder.
        text_model_id: HF model id for pretrained LLM.
        ignore_index: Label ignore index for loss computation.
        audio_token_index: Token id for the ``<|audio|>`` placeholder.
        hidden_size: Projector hidden dimension.
        stack_factor: Number of audio frames stacked before projection.
        norm_init: Initial value for RMSNorm weights.
        projector_act: Activation in the projector (``"swiglu"``).
        projector_ln_mid: If True, layer norm after first linear (v0.5+).

    """

    model_type = "aprielvoice"
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
        wrapped_model_config = kwargs.pop("wrapped_model_config", None)
        if text_config is None:
            text_config = wrapped_model_config

        self.ignore_index = ignore_index
        self.audio_token_index = audio_token_index

        self.hidden_size = hidden_size
        self.stack_factor = stack_factor
        self.norm_init = norm_init
        self.projector_act = projector_act
        self.projector_ln_mid = projector_ln_mid

        self.wrapped_model_config: transformers.PreTrainedConfig
        self.text_model_id = text_model_id
        if text_model_id is None:
            text_config = text_config or {}
            self.wrapped_model_config = transformers.CONFIG_MAPPING[
                text_config.get("model_type", "llama")
            ](**text_config)

        self.audio_model_id = audio_model_id
        if audio_model_id is None:
            self.audio_model_id = None
            audio_config = audio_config or {}
            self.audio_config = transformers.CONFIG_MAPPING[
                audio_config.get("model_type", "whisper")
            ](**audio_config)

        # Used for image placeholders and copied to the speculative drafter
        # by SpecDecodeBaseProposer, including for audio-only backbones.
        self.image_token_index = kwargs.pop(
            "image_token_index",
            getattr(self.wrapped_model_config, "image_token_id", -1),
        )

        super().__init__(**kwargs)

    def __setattr__(self, key, value):
        if key == "text_model_id" and value is not None:
            from vllm.transformers_utils.config import get_config

            self.wrapped_model_config = get_config(value, trust_remote_code=False)
        elif key == "audio_model_id" and value is not None:
            from vllm.transformers_utils.config import get_config

            self.audio_config = get_config(value, trust_remote_code=False)

        return super().__setattr__(key, value)

    @property
    def text_config(self) -> transformers.PreTrainedConfig:
        return self.wrapped_model_config.get_text_config()

    @property
    def vision_config(self) -> transformers.PreTrainedConfig | None:
        if self.wrapped_model_config.model_type == "gemma4":
            return getattr(self.wrapped_model_config, "vision_config", None)
        return None
