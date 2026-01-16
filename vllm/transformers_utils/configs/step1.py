# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for Step1 text-only models."""

from __future__ import annotations

from transformers.configuration_utils import PretrainedConfig


class Step1Config(PretrainedConfig):
    model_type = "step1"
    architectures = ["Step1ForCausalLM"]
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        *,
        hidden_size: int = 3072,
        intermediate_size: int = 8192,
        num_attention_heads: int = 48,
        num_attention_groups: int = 4,
        num_hidden_layers: int = 32,
        max_seq_len: int = 32768,
        vocab_size: int = 74752,
        rms_norm_eps: float = 1e-5,
        bos_token_id: int = 1,
        eos_token_id: int = 3,
        pad_token_id: int = 0,
        tie_word_embeddings: bool = True,
        initializer_range: float = 0.02,
        **kwargs,
    ) -> None:
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_attention_groups = num_attention_groups
        self.num_hidden_layers = num_hidden_layers
        self.max_seq_len = max_seq_len
        # Align with common config key used by scheduling logic.
        self.max_position_embeddings = kwargs.pop(
            "max_position_embeddings", max_seq_len
        )
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        # Some downstream components expect num_key_value_heads; alias to groups
        # so grouped KV attention can be derived even if the checkpoint omits it.
        self.num_key_value_heads = kwargs.pop(
            "num_key_value_heads", num_attention_groups
        )
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            initializer_range=initializer_range,
            **kwargs,
        )


__all__ = ["Step1Config"]
