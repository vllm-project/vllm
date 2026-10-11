# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``--mamba-ssm-cache-dtype`` must reach GLM-5.3-Flash's KDA recurrent state
(it used to be dropped, leaving the state float32 unconditionally)."""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.glm5next.common.kda import Glm5NextLinearAttention
from vllm.models.glm5next.common.model import (
    Glm5NextForCausalLM,
    Glm5NextForConditionalGeneration,
)


@pytest.mark.parametrize(
    ("ssm_dtype", "expected"),
    [("auto", torch.float32), ("float32", torch.float32), ("bfloat16", torch.bfloat16)],
)
def test_kda_recurrent_state_dtype_follows_ssm_cache_dtype(ssm_dtype, expected):
    model_config = SimpleNamespace(dtype=torch.bfloat16)
    cache_config = SimpleNamespace(
        mamba_cache_dtype="auto",
        mamba_ssm_cache_dtype=ssm_dtype,
        use_kda_recoverssm=False,
    )
    vllm_config = SimpleNamespace(model_config=model_config, cache_config=cache_config)
    want = (torch.bfloat16, expected)

    assert Glm5NextForCausalLM.get_mamba_state_dtype_from_config(vllm_config) == want
    assert (
        Glm5NextForConditionalGeneration.get_mamba_state_dtype_from_config(vllm_config)
        == want
    )
    # The layer's own dtype must agree with the model-level one, since the
    # KV-cache spec and the kernels read them separately.
    layer = Glm5NextLinearAttention.__new__(Glm5NextLinearAttention)
    layer.model_config, layer.cache_config = model_config, cache_config
    assert layer.get_state_dtype() == want
