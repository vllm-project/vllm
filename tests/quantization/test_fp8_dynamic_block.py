# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Fp8Config dynamic block-size auto-discovery from Quark schemas."""

import pytest
from vllm.model_executor.layers.quantization.fp8 import Fp8Config


def test_fp8_config_dynamic_block_size_quark_schema():
    """Verify Fp8Config extracts weight_block_size from nested layer_quant_config."""
    quark_dict = {
        "quant_method": "quark",
        "layer_quant_config": {
            "model.layers.0.self_attn.q_proj": {
                "weight": {"block_size": [128, 128], "dtype": "fp8"}
            }
        },
        "global_quant_config": {"input_tensors": {"is_dynamic": True}},
        "exclude": ["lm_head"],
    }

    cfg = Fp8Config.from_config(quark_dict)
    assert cfg.is_checkpoint_fp8_serialized is True
    assert cfg.weight_block_size == [128, 128]
    assert cfg.activation_scheme == "dynamic"
    assert cfg.ignored_layers == ["lm_head"]


def test_fp8_config_dynamic_block_size_32_and_dynamic_activation():
    """Verify [32, 32] block size discovery with dynamic activation."""
    quark_dict = {
        "quant_method": "quark",
        "layer_quant_config": {
            "layers.0.mlp.gate_proj": {
                "weight": {"block_size": [32, 32], "dtype": "fp8"}
            }
        },
        "global_quant_config": {"input_tensors": {"is_dynamic": True}},
    }

    cfg = Fp8Config.from_config(quark_dict)
    assert cfg.weight_block_size == [32, 32]
    assert cfg.activation_scheme == "dynamic"


def test_fp8_config_explicit_block_size_precedence():
    """Verify top-level weight_block_size overrides nested layer_quant_config."""
    quark_dict = {
        "quant_method": "quark",
        "weight_block_size": [64, 64],
        "layer_quant_config": {
            "model.layers.0.self_attn.q_proj": {
                "weight": {"block_size": [128, 128], "dtype": "fp8"}
            }
        },
        "global_quant_config": {"input_tensors": {"is_dynamic": True}},
    }

    cfg = Fp8Config.from_config(quark_dict)
    assert cfg.weight_block_size == [64, 64]
