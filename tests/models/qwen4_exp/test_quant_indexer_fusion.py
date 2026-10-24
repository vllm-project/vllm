# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for https://github.com/vllm-project/vllm/issues/62308."""

from types import SimpleNamespace

import pytest

from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    should_ignore_layer,
)
from vllm.models.qwen4_exp.nvidia.model import (
    Qwen4ExpForCausalLM,
    Qwen4ExpForConditionalGeneration,
)
from vllm.models.qwen4_exp.nvidia.qsa import _same_quant_method


@pytest.mark.parametrize(
    "model_cls", [Qwen4ExpForCausalLM, Qwen4ExpForConditionalGeneration]
)
def test_qkv_proj_mapping_allows_bf16_indexer(model_cls) -> None:
    # Layout of e.g. ukisai/Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ: q/k/v are
    # quantized and only the indexer projection is in `ignore`.
    assert not should_ignore_layer(
        "model.layers.3.self_attn.qkv_proj",
        ignore=["model.layers.3.self_attn.indexer.index_qk_proj"],
        fused_mapping=model_cls.packed_modules_mapping,
    )


class _QuantMethod:
    def __init__(self, scheme: object = None) -> None:
        self.scheme = scheme


class _SchemeA:
    pass


class _SchemeB:
    pass


def _layer(method: object, scheme: object = None) -> SimpleNamespace:
    return SimpleNamespace(quant_method=method, scheme=scheme)


@pytest.mark.parametrize(
    ("qkv_proj", "index_qk_proj", "expected"),
    [
        (
            _layer(UnquantizedLinearMethod()),
            _layer(UnquantizedLinearMethod()),
            True,
        ),
        (_layer(_QuantMethod(), _SchemeA()), _layer(_QuantMethod(), _SchemeA()), True),
        # Quantized q/k/v with a BF16 indexer, as in #62308.
        (_layer(_QuantMethod(), _SchemeA()), _layer(UnquantizedLinearMethod()), False),
        # Scheme on the layer (compressed-tensors) or on the method (INC).
        (_layer(_QuantMethod(), _SchemeA()), _layer(_QuantMethod(), _SchemeB()), False),
        (_layer(_QuantMethod(_SchemeA())), _layer(_QuantMethod(_SchemeB())), False),
    ],
)
def test_same_quant_method(
    qkv_proj: SimpleNamespace, index_qk_proj: SimpleNamespace, expected: bool
) -> None:
    assert _same_quant_method(qkv_proj, index_qk_proj) is expected
