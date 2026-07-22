# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This test file includes some cases where it is inappropriate to
only get the `eos_token_id` from the tokenizer as defined by
`BaseRenderer.get_eos_token_id`.
"""

from vllm.tokenizers import get_tokenizer
from vllm.transformers_utils.config import try_get_generation_config


def test_get_llama3_eos_token():
    model_name = "meta-llama/Llama-3.2-1B-Instruct"

    tokenizer = get_tokenizer(model_name)
    assert tokenizer.eos_token_id == 128009

    generation_config = try_get_generation_config(model_name, trust_remote_code=False)
    assert generation_config is not None
    assert generation_config.eos_token_id == [128001, 128008, 128009]


def test_get_blip2_eos_token():
    model_name = "Salesforce/blip2-opt-2.7b"

    tokenizer = get_tokenizer(model_name)
    assert tokenizer.eos_token_id == 2

    generation_config = try_get_generation_config(model_name, trust_remote_code=False)
    assert generation_config is not None
    assert generation_config.eos_token_id == 50118


def test_allow_global_per_layer_attribute_access():
    """Heterogeneous configs raise if a per-layer attribute is read from the
    global config, and vLLM's engine initialization does such reads before
    the model exists (e.g. head size derivation). The helper must downgrade
    them to global-value reads, and leave homogeneous configs untouched."""
    import pytest
    from transformers import PreTrainedConfig

    heterogeneity = pytest.importorskip(
        "transformers.integrations.heterogeneity.configuration_utils",
        reason="requires transformers with heterogeneous config support",
    )

    from vllm.transformers_utils.config import allow_global_per_layer_attribute_access

    config = PreTrainedConfig(num_hidden_layers=2, head_dim=256)
    config.per_layer_config = {1: {"head_dim": 512}}
    assert config.is_heterogeneous

    with pytest.raises(heterogeneity.AmbiguousGlobalPerLayerAttributeError):
        _ = config.head_dim

    allow_global_per_layer_attribute_access(config)
    assert config.head_dim == 256
    # Per-layer reads are unaffected
    assert config.per_layer_config[1].head_dim == 512

    homogeneous = PreTrainedConfig(num_hidden_layers=2, head_dim=256)
    allow_global_per_layer_attribute_access(homogeneous)
    assert "allow_global_per_layer_attribute_access" not in homogeneous.__dict__
