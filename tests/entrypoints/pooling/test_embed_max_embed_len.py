# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the ``max_embed_len`` cap on embedding tokenization.

``EmbeddingTokenizeParamsMixin.build_tok_params`` used to enforce
``pooler_config.max_embed_len`` indirectly, by setting
``max_output_tokens = max_model_len - max_embed_len`` and relying on
``max_input_tokens == max_total_tokens - max_output_tokens``. That formula no
longer holds, so the cap has to be expressed directly.
"""

from unittest.mock import Mock

from vllm.config import ModelConfig, PoolerConfig
from vllm.entrypoints.pooling.embed.protocol import EmbeddingCompletionRequest


def _model_config(
    *,
    max_model_len: int = 128,
    max_embed_len: int | None = None,
    enable_chunked_processing: bool = False,
) -> Mock:
    model_config = Mock(spec=ModelConfig)
    model_config.max_model_len = max_model_len
    model_config.encoder_config = None
    model_config.pooler_config = PoolerConfig(
        pooling_target="embed",
        max_embed_len=max_embed_len,
        enable_chunked_processing=enable_chunked_processing,
    )
    return model_config


def test_non_chunked_max_embed_len_caps_input():
    """The regression: with max_embed_len < max_model_len and chunking off,
    the input cap must still be max_embed_len, not the full context."""
    cfg = _model_config(max_model_len=128, max_embed_len=64)

    tok_params = EmbeddingCompletionRequest(model="m", input="hi").build_tok_params(cfg)

    assert tok_params.max_input_tokens == 64
    assert tok_params.max_output_tokens == 0


def test_chunked_max_embed_len_caps_input():
    cfg = _model_config(
        max_model_len=128, max_embed_len=64, enable_chunked_processing=True
    )

    tok_params = EmbeddingCompletionRequest(model="m", input="hi").build_tok_params(cfg)

    assert tok_params.max_input_tokens == 64
    assert tok_params.max_output_tokens == 0


def test_no_max_embed_len_falls_back_to_model_len():
    """Unset max_embed_len must leave the cap at the full context length."""
    cfg = _model_config(max_model_len=128)

    tok_params = EmbeddingCompletionRequest(model="m", input="hi").build_tok_params(cfg)

    assert tok_params.max_input_tokens == 128
    assert tok_params.max_output_tokens == 0
