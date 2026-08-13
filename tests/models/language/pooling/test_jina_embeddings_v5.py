# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backbone validation for Jina Embeddings V5.

The V5 family ships two backbones under one `architectures` entry: `-small` is
a Qwen3 decoder, while `-nano` is a bidirectional EuroBERT encoder. Upstream
ships a separate `configuration_*.py` per repository, so the only signal
distinguishing them is `is_decoder`, which the encoder variant sets to False.
`JinaEmbeddingsV5ModelConfig` uses it to enable bidirectional attention for the
encoder variant; `JinaEmbeddingsV5Model` then dispatches to the correct backbone.
"""

from types import SimpleNamespace
from typing import cast

import pytest
from transformers import PretrainedConfig

from vllm.config import ModelConfig
from vllm.model_executor.models.config import (
    MODELS_CONFIG_MAP,
    JinaEmbeddingsV5ModelConfig,
)


def _model_config(hf_config: PretrainedConfig) -> ModelConfig:
    """Minimal stand-in for ModelConfig; only hf_config is read."""
    return cast(ModelConfig, SimpleNamespace(hf_config=hf_config))


@pytest.mark.cpu_test
def test_registered_for_the_architecture():
    """The handler only runs if it is wired to the architecture name."""
    assert MODELS_CONFIG_MAP["JinaEmbeddingsV5Model"] is JinaEmbeddingsV5ModelConfig


@pytest.mark.cpu_test
def test_encoder_backbone_enables_bidirectional_attention():
    """An encoder checkpoint (is_decoder=False) is supported.

    The handler sets is_causal=False so the Llama backbone uses
    EncoderOnlyAttention; JinaEmbeddingsV5Model then dispatches to the encoder
    implementation.
    """
    hf_config = PretrainedConfig(is_decoder=False)

    JinaEmbeddingsV5ModelConfig.verify_and_update_model_config(_model_config(hf_config))

    assert hf_config.is_causal is False


@pytest.mark.cpu_test
def test_supported_decoder_backbone_is_accepted():
    """The Qwen3-based variants must keep loading.

    `-small` omits `is_decoder` entirely, so an absent attribute has to be
    treated as a decoder. The first assertion pins that assumption: if
    PretrainedConfig ever gains an `is_decoder=False` default, this fails here
    rather than silently rejecting a supported checkpoint.
    """
    absent = PretrainedConfig()
    assert not hasattr(absent, "is_decoder")

    JinaEmbeddingsV5ModelConfig.verify_and_update_model_config(_model_config(absent))
    JinaEmbeddingsV5ModelConfig.verify_and_update_model_config(
        _model_config(PretrainedConfig(is_decoder=True))
    )
