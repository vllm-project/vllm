# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 detection for compressed-tensors checkpoints.

A single-scheme checkpoint declares ``format`` as a string, while one that mixes
schemes (e.g. NVFP4 MLP with FP8 attention) declares it as a list.
"""

import pytest

from vllm.config.model import _compressed_tensors_has_nvfp4


def test_detects_single_scheme_nvfp4():
    assert _compressed_tensors_has_nvfp4({"format": "nvfp4-pack-quantized"})


def test_detects_nvfp4_in_a_mixed_scheme_format_list():
    """The shipped mixed shape, which used to raise AttributeError on .lower()."""
    assert _compressed_tensors_has_nvfp4(
        {"format": ["nvfp4-pack-quantized", "float-quantized"]}
    )


@pytest.mark.parametrize(
    "fmt", ["float-quantized", ["float-quantized", "pack-quantized"]]
)
def test_ignores_formats_without_nvfp4(fmt):
    assert not _compressed_tensors_has_nvfp4({"format": fmt})


@pytest.mark.parametrize(
    "quant_config",
    [None, {}, {"format": None}, {"format": 4}, {"format": [None, 4]}],
)
def test_absent_or_malformed_config_is_not_nvfp4(quant_config):
    assert not _compressed_tensors_has_nvfp4(quant_config)
