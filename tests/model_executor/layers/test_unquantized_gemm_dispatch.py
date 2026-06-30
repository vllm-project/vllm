# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from vllm.config import (
    KernelConfig,
    VllmConfig,
    get_current_vllm_config,
    set_current_vllm_config,
)
from vllm.model_executor.layers import linear, vocab_parallel_embedding

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize(
    ("module", "method_type"),
    [
        pytest.param(linear, linear.UnquantizedLinearMethod, id="linear"),
        pytest.param(
            vocab_parallel_embedding,
            vocab_parallel_embedding.UnquantizedEmbeddingMethod,
            id="embedding",
        ),
    ],
)
def test_unquantized_gemm_is_bound_during_initialization(
    monkeypatch, module, method_type
):
    config = VllmConfig(kernel_config=KernelConfig(bf16_linear_backend="flashinfer"))
    output = object()
    gemm_impl = MagicMock(return_value=output)

    def select_gemm():
        assert get_current_vllm_config() is config
        return gemm_impl

    selector = MagicMock(side_effect=select_gemm)
    monkeypatch.setattr(module, "dispatch_unquantized_gemm", selector)
    monkeypatch.setattr(module.envs, "VLLM_BATCH_INVARIANT", False)

    with set_current_vllm_config(config):
        method = method_type()

    selector.assert_called_once_with()

    weight = object()
    layer = SimpleNamespace(weight=weight)
    x = object()
    bias = object()
    for _ in range(2):
        assert method.apply(layer, x, bias) is output

    selector.assert_called_once_with()
    assert gemm_impl.call_args_list == [
        call(layer, x, weight, bias),
        call(layer, x, weight, bias),
    ]
