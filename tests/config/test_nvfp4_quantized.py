# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Detection of NVFP4 in compressed-tensors quantization configs.

Mixed-scheme checkpoints (e.g. NVFP4 + FP8) declare the top-level ``format``
as a list of scheme names instead of a single string; the check must accept
both shapes without crashing.
"""

from types import SimpleNamespace

import pytest

from vllm.config.model import ModelConfig


def _is_nvfp4(quantization, quant_config):
    """The method only reads these two attributes; avoid a full ModelConfig."""
    fake = SimpleNamespace(
        quantization=quantization,
        model_arch_config=SimpleNamespace(quantization_config=quant_config),
    )
    return ModelConfig.is_nvfp4_quantized(fake)


@pytest.mark.parametrize(
    "fmt, expected",
    [
        ("nvfp4-pack-quantized", True),
        ("float-quantized", False),
        (["nvfp4-pack-quantized", "float-quantized"], True),
        (["float-quantized", "int-quantized"], False),
        ([], False),
    ],
)
def test_compressed_tensors_format_shapes(fmt, expected):
    assert _is_nvfp4("compressed-tensors", {"format": fmt}) is expected


def test_compressed_tensors_missing_format_key():
    assert _is_nvfp4("compressed-tensors", {}) is False


def test_compressed_tensors_none_config():
    assert _is_nvfp4("compressed-tensors", None) is False


def test_modelopt_fp4_shortcut():
    assert _is_nvfp4("modelopt_fp4", None) is True


def test_non_compressed_tensors_ignores_format():
    assert _is_nvfp4("awq", {"format": "nvfp4-pack-quantized"}) is False
