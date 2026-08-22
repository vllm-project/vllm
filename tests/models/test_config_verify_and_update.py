# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for VerifyAndUpdateConfig subclasses in models/config.py.

These use lightweight mock config objects so they run on CPU without
downloading any model.
"""

from types import SimpleNamespace

import pytest

from vllm.model_executor.models.config import LlamaBidirectionalConfig


def _mock_model_config(hf_config):
    return SimpleNamespace(
        hf_config=hf_config,
        pooler_config=SimpleNamespace(seq_pooling_type=None),
    )


def test_llama_bidirectional_missing_pooling_raises_value_error():
    # NV-Embed-v2 style config: no top-level ``pooling`` attribute. Before the
    # guard this raised an opaque ``AttributeError`` deep in engine-core spawn.
    model_config = _mock_model_config(SimpleNamespace())
    with pytest.raises(ValueError, match="pooling"):
        LlamaBidirectionalConfig.verify_and_update_model_config(model_config)


def test_llama_bidirectional_unsupported_pooling_raises_value_error():
    model_config = _mock_model_config(SimpleNamespace(pooling="latent"))
    with pytest.raises(ValueError, match="pool_type"):
        LlamaBidirectionalConfig.verify_and_update_model_config(model_config)


@pytest.mark.parametrize(
    "pooling,expected",
    [("avg", "MEAN"), ("cls", "CLS"), ("last", "LAST")],
)
def test_llama_bidirectional_valid_pooling(pooling, expected):
    model_config = _mock_model_config(SimpleNamespace(pooling=pooling))
    LlamaBidirectionalConfig.verify_and_update_model_config(model_config)
    assert model_config.pooler_config.seq_pooling_type == expected
    assert model_config.hf_config.is_causal is False
