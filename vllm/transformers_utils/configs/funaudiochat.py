# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from transformers import PretrainedConfig

# NOTE: Temporary shim for FunAudioChat checkpoints.
# These checkpoints use `model_type="funaudiochat"`, which is not currently
# recognized by released Transformers, and the public checkpoint does not
# provide an `auto_map` to enable `trust_remote_code=True`.
# Remove this file once Transformers adds native support (or the checkpoint
# provides an `auto_map`) and vLLM can rely on `AutoConfig.from_pretrained()`.


class FunAudioChatAudioEncoderConfig(PretrainedConfig):
    model_type = "funaudiochat_audio_encoder"

    def __init__(
        self,
        _attn_implementation: str | None = None,
        num_mel_bins: int = 128,
        encoder_layers: int = 32,
        encoder_attention_heads: int = 20,
        encoder_ffn_dim: int = 5120,
        d_model: int = 1280,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation_function: str = "gelu",
        activation_dropout: float = 0.0,
        scale_embedding: bool = False,
        initializer_range: float = 0.02,
        max_source_positions: int = 1500,
        n_window: int = 100,
        output_dim: int = 3584,
        bos_token_id: int | None = None,
        codebook_size: int | None = None,
        continuous_features_mode: str = "replace",
        crq_transformer_config: dict | None = None,
        eos_token_id: int | None = None,
        group_size: int = 5,
        enable_audio_invert_tower: bool = True,
        pad_token_id: int | None = None,
        **kwargs,
    ) -> None:
        attn_impl = kwargs.pop("_attn_implementation", None) or _attn_implementation
        super().__init__(**kwargs)
        # Match HF default for attention implementation selection.
        self._attn_implementation = attn_impl or "sdpa"

        self.num_mel_bins = num_mel_bins
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_function = activation_function
        self.activation_dropout = activation_dropout
        self.num_hidden_layers = encoder_layers
        self.initializer_range = initializer_range
        self.scale_embedding = scale_embedding
        self.max_source_positions = max_source_positions
        self.n_window = n_window
        self.output_dim = output_dim

        self.bos_token_id = bos_token_id
        self.codebook_size = codebook_size
        self.continuous_features_mode = continuous_features_mode
        self.crq_transformer_config = crq_transformer_config
        self.eos_token_id = eos_token_id
        self.group_size = group_size
        self.enable_audio_invert_tower = enable_audio_invert_tower
        self.pad_token_id = pad_token_id


class FunAudioChatConfig(PretrainedConfig):
    model_type = "funaudiochat"
    attribute_map = {
        "audio_token_id": "audio_token_index",
    }

    def __init__(
        self,
        audio_config: PretrainedConfig | dict | None = None,
        text_config: PretrainedConfig | dict | None = None,
        audio_token_index: int = 151646,
        ignore_index: int = -100,
        hidden_size: int | None = None,
        **kwargs,
    ) -> None:
        self.audio_token_index = audio_token_index
        self.ignore_index = ignore_index

        if isinstance(audio_config, dict):
            audio_config.setdefault(
                "model_type", FunAudioChatAudioEncoderConfig.model_type
            )
            audio_config = FunAudioChatAudioEncoderConfig(**audio_config)
        elif audio_config is None:
            audio_config = FunAudioChatAudioEncoderConfig()
        self.audio_config = audio_config

        if isinstance(text_config, dict):
            # Default to qwen2 for backwards compatibility; FunAudioChat uses
            # qwen3 in practice for recent checkpoints.
            text_config.setdefault("model_type", "qwen2")
            import transformers

            text_cls = transformers.CONFIG_MAPPING[text_config["model_type"]]
            text_config = text_cls(**text_config)
        elif text_config is None:
            import transformers

            text_config = transformers.CONFIG_MAPPING["qwen2"]()
        self.text_config = text_config

        self.hidden_size = (
            int(self.text_config.hidden_size)
            if hidden_size is None
            else int(hidden_size)
        )

        super().__init__(**kwargs)
