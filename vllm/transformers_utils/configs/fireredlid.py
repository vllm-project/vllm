# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import contextlib

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig


class FireRedLIDConfig(PretrainedConfig):
    """Minimal config class for native vLLM FireRedLID support."""

    model_type = "fireredlid"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 120,
        lid_odim: int = 120,
        idim: int = 80,
        d_model: int = 1280,
        n_head: int = 20,
        n_layers_enc: int = 16,
        n_layers_lid_dec: int = 6,
        kernel_size: int = 33,
        residual_dropout: float = 0.05,
        dropout_rate: float = 0.05,
        pe_maxlen: int = 5000,
        pad_token_id: int = 2,
        bos_token_id: int = 3,
        eos_token_id: int = 4,
        decoder_start_token_id: int = 3,
        tie_word_embeddings: bool = True,
        is_encoder_decoder: bool = True,
        architectures: list[str] | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.lid_odim = lid_odim
        self.idim = idim
        self.d_model = d_model
        self.hidden_size = d_model
        self.n_head = n_head
        self.num_attention_heads = n_head
        self.n_layers_enc = n_layers_enc
        self.encoder_layers = n_layers_enc
        self.n_layers_lid_dec = n_layers_lid_dec
        self.decoder_layers = n_layers_lid_dec
        self.num_hidden_layers = n_layers_lid_dec
        self.kernel_size = kernel_size
        self.residual_dropout = residual_dropout
        self.dropout_rate = dropout_rate
        self.pe_maxlen = pe_maxlen
        self.tie_word_embeddings = tie_word_embeddings
        self.is_encoder_decoder = is_encoder_decoder
        self.architectures = architectures or ["FireRedLIDForConditionalGeneration"]

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            tie_word_embeddings=tie_word_embeddings,
            is_encoder_decoder=is_encoder_decoder,
            architectures=self.architectures,
            **kwargs,
        )


with contextlib.suppress(ValueError):
    AutoConfig.register(FireRedLIDConfig.model_type, FireRedLIDConfig)
