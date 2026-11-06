# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 detection for ModelOpt MIXED_PRECISION checkpoints.

A checkpoint that quantizes some layers to NVFP4 and others to FP8 declares
``quant_algo: MIXED_PRECISION`` and resolves to ``modelopt_mixed``, so NVFP4
has to be detected from the per-layer algorithms.
"""

import pytest

from vllm.config.model import _modelopt_mixed_has_nvfp4


def _quantized_layers(*algos: str) -> dict:
    return {
        "quantized_layers": {
            f"model.layers.{i}.proj": {"quant_algo": algo}
            for i, algo in enumerate(algos)
        }
    }


def test_detects_nvfp4_alongside_another_algorithm():
    """The shipped shape: FP8 attention projections, NVFP4 MLP."""
    assert _modelopt_mixed_has_nvfp4(_quantized_layers("FP8", "NVFP4", "FP8"))


def test_ignores_a_checkpoint_with_no_nvfp4_layer():
    assert not _modelopt_mixed_has_nvfp4(_quantized_layers("FP8", "MXFP8"))


def test_weight_only_nvfp4_does_not_count():
    """W4A16_NVFP4 leaves activations in high precision, so nothing to fuse."""
    assert not _modelopt_mixed_has_nvfp4(_quantized_layers("FP8", "W4A16_NVFP4"))


@pytest.mark.parametrize(
    "quant_config",
    [None, {}, {"quantized_layers": None}, {"quantized_layers": ["NVFP4"]}],
)
def test_absent_or_malformed_config_is_not_nvfp4(quant_config):
    assert not _modelopt_mixed_has_nvfp4(quant_config)
