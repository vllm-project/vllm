# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.activation import (
    GeluAndMul,
    SiluAndMul,
    get_act_and_mul_fn,
    get_act_fn,
)
from vllm.model_executor.models.gemma3 import Gemma3MLP
from vllm.model_executor.models.gemma4 import Gemma4MLP


@pytest.mark.parametrize(
    ("activation_name", "expected_type"),
    [
        ("gelu_pytorch_tanh", GeluAndMul),
        ("silu", SiluAndMul),
        ("swish", SiluAndMul),
    ],
)
def test_get_act_and_mul_fn_supports_gemma_hidden_act_aliases(
    activation_name: str,
    expected_type: type[torch.nn.Module],
    default_vllm_config,
) -> None:
    assert isinstance(get_act_and_mul_fn(activation_name), expected_type)


def test_get_act_fn_supports_swish_alias() -> None:
    assert isinstance(get_act_fn("swish"), torch.nn.SiLU)


@pytest.mark.parametrize("mlp_cls", [Gemma3MLP, Gemma4MLP])
@pytest.mark.parametrize(
    ("activation_name", "expected_type"),
    [
        ("gelu_pytorch_tanh", GeluAndMul),
        ("silu", SiluAndMul),
        ("swish", SiluAndMul),
    ],
)
def test_gemma_mlp_supports_hidden_act_variants(
    mlp_cls: type[torch.nn.Module],
    activation_name: str,
    expected_type: type[torch.nn.Module],
    default_vllm_config,
    dist_init,
) -> None:
    mlp = mlp_cls(
        hidden_size=16,
        intermediate_size=32,
        hidden_activation=activation_name,
    )

    assert isinstance(mlp.act_fn, expected_type)
    assert mlp(torch.randn(3, 16)).shape == (3, 16)
