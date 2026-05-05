# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test model set-up and inference for quantized HF models supported
on the AutoRound.

Validating the configuration and printing results for manual checking.

Run `pytest tests/quantization/test_auto_round.py`.
"""

import pytest

from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.gptq_marlin import GPTQMarlinLinearMethod
from vllm.model_executor.layers.quantization.inc import (
    INCConfig,
    INCGPTQRowParallelTailLinearMethod,
)
from vllm.model_executor.layers.quantization.utils import marlin_utils
from vllm.platforms import current_platform

MODELS = [
    "OPEA/Qwen2.5-0.5B-Instruct-int4-sym-inc",  ##auto_round:auto_gptq
    "Intel/Qwen2-0.5B-Instruct-int4-sym-AutoRound",  ##auto_round:auto_awq
]


@pytest.mark.skipif(
    not current_platform.is_cpu()
    and not current_platform.is_xpu()
    and not current_platform.is_cuda(),
    reason="only supports CPU/XPU/CUDA backend.",
)
@pytest.mark.parametrize("model", MODELS)
def test_auto_round(vllm_runner, model):
    with vllm_runner(model, enforce_eager=True) as llm:
        output = llm.generate_greedy(["The capital of France is"], max_tokens=8)
    assert output
    print(f"{output[0][1]}")


def _make_linear_base_stub(
    *,
    input_size: int,
    output_size: int,
    input_size_per_partition: int,
    output_size_per_partition: int,
) -> LinearBase:
    layer = object.__new__(LinearBase)
    layer.input_size = input_size
    layer.output_size = output_size
    layer.input_size_per_partition = input_size_per_partition
    layer.output_size_per_partition = output_size_per_partition
    return layer


def test_inc_gptq_dense_marlin_allows_kernel_padding_for_small_output_dim(
    monkeypatch,
):
    """Dense INC Marlin selection should not reject small output shards early.

    The dense Marlin kernel handles sub-tile output dims by padding at load time.
    INC must not run the older pre-kernel shape check that rejects N < 64 before
    the padding-aware kernel can be selected.
    """
    monkeypatch.setattr(
        marlin_utils,
        "check_marlin_supported",
        lambda *args, **kwargs: True,
    )
    layer = _make_linear_base_stub(
        input_size=256,
        output_size=32,
        input_size_per_partition=256,
        output_size_per_partition=32,
    )
    config = INCConfig(weight_bits=4, group_size=128, sym=True)

    method = config.apply_gptq_quant_layer(layer, "model.layers.0.mlp.down_proj")

    assert isinstance(method, GPTQMarlinLinearMethod)


def test_inc_gptq_row_tail_fallback_still_precedes_marlin(monkeypatch):
    """Group-misaligned row shards still use the correctness-first tail path."""
    monkeypatch.setattr(
        marlin_utils,
        "check_marlin_supported",
        lambda *args, **kwargs: True,
    )
    layer = _make_linear_base_stub(
        input_size=256,
        output_size=64,
        input_size_per_partition=192,
        output_size_per_partition=64,
    )
    config = INCConfig(weight_bits=4, group_size=128, sym=True)

    method = config.apply_gptq_quant_layer(layer, "model.layers.0.mlp.down_proj")

    assert isinstance(method, INCGPTQRowParallelTailLinearMethod)
