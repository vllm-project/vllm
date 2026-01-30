# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from transformers import PretrainedConfig


def _build_text_config(text_config: dict) -> PretrainedConfig:
    # VibeVoice checkpoints store the underlying LLM config under
    # `decoder_config` (typically Qwen2).
    import transformers

    model_type = text_config.get("model_type", "qwen2")
    try:
        text_cls = transformers.CONFIG_MAPPING[model_type]
    except KeyError:
        # Be permissive: keep the raw dict if transformers doesn't recognize
        # the nested model_type in this environment.
        return PretrainedConfig(**text_config)
    return text_cls(**text_config)


class VibeVoiceASRConfig(PretrainedConfig):
    """VibeVoice-ASR config shim for vLLM.

    The HuggingFace checkpoint uses `model_type="vibevoice"` without an
    `auto_map` entry, so vLLM must provide a local config mapping.

    Note: VibeVoice inference requires the external `vibevoice` package at
    runtime, but we keep this config importable without it so vLLM can be
    installed and its CI can run without optional deps.
    """

    model_type = "vibevoice"

    def __init__(
        self,
        decoder_config: PretrainedConfig | dict | None = None,
        acoustic_tokenizer_config: dict | None = None,
        semantic_tokenizer_config: dict | None = None,
        acoustic_vae_dim: int | None = None,
        semantic_vae_dim: int | None = None,
        diffusion_head_config: dict | None = None,
        **kwargs,
    ) -> None:
        if isinstance(decoder_config, dict):
            decoder_config = _build_text_config(decoder_config)

        # vLLM expects multimodal configs to expose their underlying text config
        # via `get_text_config()` (provided by Transformers' PretrainedConfig).
        # Use the same object for both names.
        self.decoder_config = decoder_config
        self.text_config = decoder_config

        self.acoustic_tokenizer_config = acoustic_tokenizer_config
        self.semantic_tokenizer_config = semantic_tokenizer_config
        self.acoustic_vae_dim = acoustic_vae_dim
        self.semantic_vae_dim = semantic_vae_dim
        self.diffusion_head_config = diffusion_head_config

        super().__init__(**kwargs)
