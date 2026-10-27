# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for https://github.com/vllm-project/vllm/issues/62308."""

import pytest

from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    should_ignore_layer,
)
from vllm.models.qwen4_exp.nvidia.model import (
    Qwen4ExpForCausalLM,
    Qwen4ExpForConditionalGeneration,
)


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
