# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

from transformers import PretrainedConfig, Qwen3Config
from transformers.models.whisper.configuration_whisper import WhisperConfig


class MossTranscribeDiarizeConfig(PretrainedConfig):
    """Configuration for MOSS-Transcribe-Diarize."""

    model_type = "moss_transcribe_diarize"
    sub_configs = {"text_config": Qwen3Config, "audio_config": WhisperConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config: dict[str, Any] | Qwen3Config | None = None,
        audio_config: dict[str, Any] | WhisperConfig | None = None,
        audio_token_id: int = 151671,
        audio_merge_size: int = 4,
        adaptor_input_dim: int | None = None,
        tie_word_embeddings: bool = True,
        **kwargs: Any,
    ) -> None:
        text_config_obj: Qwen3Config
        if text_config is None:
            text_config_obj = Qwen3Config(
                vocab_size=151936,
                hidden_size=1024,
                intermediate_size=3072,
                num_hidden_layers=28,
                num_attention_heads=16,
                num_key_value_heads=8,
                head_dim=128,
                max_position_embeddings=40960,
                tie_word_embeddings=tie_word_embeddings,
                rope_theta=1_000_000.0,
                layer_types=["full_attention"] * 28,
            )
        elif isinstance(text_config, dict):
            text_config_obj = Qwen3Config(**text_config)
        else:
            text_config_obj = text_config

        audio_config_obj: WhisperConfig
        if audio_config is None:
            audio_config_obj = WhisperConfig(
                num_mel_bins=80,
                d_model=1024,
                encoder_layers=24,
                encoder_attention_heads=16,
                encoder_ffn_dim=4096,
                max_source_positions=1500,
                dropout=0.0,
                attention_dropout=0.0,
                activation_dropout=0.0,
                activation_function="gelu",
                encoder_layerdrop=0.0,
                scale_embedding=False,
            )
        elif isinstance(audio_config, dict):
            audio_config_obj = WhisperConfig(**audio_config)
        else:
            audio_config_obj = audio_config

        text_config_obj.tie_word_embeddings = tie_word_embeddings
        if not getattr(text_config_obj, "layer_types", None):
            text_config_obj.layer_types = [
                "full_attention"
            ] * text_config_obj.num_hidden_layers

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

        self.text_config = text_config_obj
        self.audio_config = audio_config_obj
        self.audio_token_id = int(audio_token_id)
        self.audio_merge_size = int(audio_merge_size)
        self.adaptor_input_dim = (
            int(adaptor_input_dim)
            if adaptor_input_dim is not None
            else int(audio_config_obj.d_model) * int(audio_merge_size)
        )

        self.vocab_size = int(text_config_obj.vocab_size)
        self.hidden_size = int(text_config_obj.hidden_size)
        self.intermediate_size = int(text_config_obj.intermediate_size)
        self.num_hidden_layers = int(text_config_obj.num_hidden_layers)
        self.num_attention_heads = int(text_config_obj.num_attention_heads)
        self.num_key_value_heads = int(text_config_obj.num_key_value_heads)
        self.head_dim = int(text_config_obj.head_dim)
        self.hidden_act = text_config_obj.hidden_act
        self.max_position_embeddings = int(text_config_obj.max_position_embeddings)
        self.rms_norm_eps = float(text_config_obj.rms_norm_eps)
        rope_parameters = getattr(text_config_obj, "rope_parameters", None)
        rope_theta = float(getattr(text_config_obj, "rope_theta", 1_000_000.0))
        if rope_parameters is None:
            rope_parameters = {
                "rope_type": "default",
                "rope_theta": rope_theta,
            }
            text_config_obj.rope_parameters = rope_parameters
        self.rope_parameters = rope_parameters
        self.rope_theta = float(rope_parameters.get("rope_theta", rope_theta))
        self.attention_bias = bool(text_config_obj.attention_bias)
        self.attention_dropout = float(text_config_obj.attention_dropout)
        self.is_causal = True
