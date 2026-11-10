# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``--mamba-ssm-cache-dtype`` must reach the KDA recurrent state of the models
built on the shared KDA layers (it used to be dropped, leaving the state
float32 unconditionally).

Mirrors ``tests/models/glm5next/test_kda_state_dtype.py`` for the KDA
implementations under ``vllm/model_executor``.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
    KimiGatedDeltaNetAttention,
)
from vllm.model_executor.models.alice_ai import AliceAIForCausalLM
from vllm.model_executor.models.bailing_moe_v3 import (
    BailingMoeV3ForCausalLM,
    BailingMoeV3KimiDeltaAttention,
)

SSM_DTYPES = [
    ("auto", torch.float32),
    ("float32", torch.float32),
    ("bfloat16", torch.bfloat16),
]


def _configs(ssm_dtype):
    model_config = SimpleNamespace(dtype=torch.bfloat16)
    cache_config = SimpleNamespace(
        mamba_cache_dtype="auto",
        mamba_ssm_cache_dtype=ssm_dtype,
    )
    return model_config, cache_config


@pytest.mark.parametrize(("ssm_dtype", "expected"), SSM_DTYPES)
@pytest.mark.parametrize(
    "layer_cls", [KimiGatedDeltaNetAttention, BailingMoeV3KimiDeltaAttention]
)
def test_kda_layer_state_dtype_follows_ssm_cache_dtype(layer_cls, ssm_dtype, expected):
    model_config, cache_config = _configs(ssm_dtype)
    layer = layer_cls.__new__(layer_cls)
    layer.model_config, layer.cache_config = model_config, cache_config
    assert layer.get_state_dtype() == (torch.bfloat16, expected)


@pytest.mark.parametrize(("ssm_dtype", "expected"), SSM_DTYPES)
@pytest.mark.parametrize("model_cls", [AliceAIForCausalLM, BailingMoeV3ForCausalLM])
def test_kda_model_state_dtype_follows_ssm_cache_dtype(model_cls, ssm_dtype, expected):
    # The model-level dtype feeds the mamba page size used to pick
    # ``mamba_block_size``; it must agree with the layer's own dtype or the
    # allocated page overflows the planned one.
    model_config, cache_config = _configs(ssm_dtype)
    vllm_config = SimpleNamespace(model_config=model_config, cache_config=cache_config)
    assert model_cls.get_mamba_state_dtype_from_config(vllm_config) == (
        torch.bfloat16,
        expected,
    )
