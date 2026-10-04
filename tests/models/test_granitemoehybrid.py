# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration tests for Granite hybrid Mamba state shapes."""

from types import SimpleNamespace

import pytest

from vllm.model_executor.models.granitemoehybrid import GraniteMoeHybridForCausalLM

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("num_speculative_tokens", [0, 2])
def test_granite_mamba_state_shape_includes_speculative_tokens(
    num_speculative_tokens: int,
):
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                hidden_size=8,
                mamba_expand=2,
                mamba_n_groups=1,
                mamba_n_heads=2,
                mamba_d_head=4,
                mamba_d_state=3,
                mamba_d_conv=4,
            )
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        num_speculative_tokens=num_speculative_tokens,
    )

    conv_shape, temporal_shape = (
        GraniteMoeHybridForCausalLM.get_mamba_state_shape_from_config(vllm_config)
    )

    expected_conv_state_len = 3 + num_speculative_tokens
    assert conv_shape == (expected_conv_state_len, 22)
    assert temporal_shape == (2, 4, 3)
