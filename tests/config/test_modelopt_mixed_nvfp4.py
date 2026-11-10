# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 detection for ModelOpt MIXED_PRECISION checkpoints.

A checkpoint that quantizes some layers to NVFP4 and others to FP8 declares
`quant_algo: MIXED_PRECISION` and resolves to `modelopt_mixed`, so NVFP4
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


def test_mixed_precision_backfills_missing_quantized_layers_from_hf_quant_config(
    tmp_path,
):
    """When both config.json and hf_quant_config.json exist for a ModelOpt
    MIXED_PRECISION checkpoint, layer entries present only in
    hf_quant_config.json (such as Nemotron-H MTP MoE layers) must be preserved
    while config.json entries take precedence."""
    import json

    from transformers import LlamaConfig

    from vllm.config.load import LoadConfig
    from vllm.config.model import ModelConfig
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptMixedPrecisionConfig,
    )
    from vllm.model_executor.model_loader.weight_utils import get_quant_config
    from vllm.model_executor.models.nemotron_h_mtp import NemotronHMTP

    hf_config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    hf_config.quantization_config = {
        "producer": {"name": "modelopt"},
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "ignore": ["language_model.mtp.layers.0.eh_proj"],
        "quantized_layers": {
            "language_model.backbone.layers.0.mixer.in_proj": {
                "quant_algo": "FP8",
            },
        },
    }
    hf_config.save_pretrained(tmp_path)

    hf_quant_config = {
        "producer": {"name": "modelopt"},
        "quantization": {
            "quant_algo": "MIXED_PRECISION",
            "exclude_modules": ["language_model.mtp.layers.0.eh_proj"],
            "quantized_layers": {
                "language_model.backbone.layers.0.mixer.in_proj": {
                    "quant_algo": "NVFP4",
                    "group_size": 16,
                },
                "language_model.mtp.layers.1.mixer.experts": {
                    "quant_algo": "NVFP4",
                    "group_size": 16,
                },
                "language_model.mtp.layers.1.mixer.shared_experts.up_proj": {
                    "quant_algo": "NVFP4",
                    "group_size": 16,
                },
                "language_model.mtp.layers.1.mixer.shared_experts.down_proj": {
                    "quant_algo": "NVFP4",
                    "group_size": 16,
                },
            },
        },
    }
    (tmp_path / "hf_quant_config.json").write_text(json.dumps(hf_quant_config))

    model_config = ModelConfig(
        str(tmp_path), dtype="bfloat16", skip_tokenizer_init=True
    )
    quant_config = get_quant_config(model_config, LoadConfig())
    assert isinstance(quant_config, ModelOptMixedPrecisionConfig)

    quant_config.apply_vllm_mapper(NemotronHMTP.hf_to_vllm_mapper.get_rename_mapper())

    # Existing config.json entry takes precedence over hf_quant_config.json
    assert quant_config._resolve_quant_algo("backbone.layers.0.mixer.in_proj") == "FP8"
    # Missing MTP entries from hf_quant_config.json are backfilled
    assert quant_config._resolve_quant_algo("mtp.layers.1.mixer.experts") == "NVFP4"
    assert (
        quant_config._resolve_quant_algo("mtp.layers.1.mixer.shared_experts.up_proj")
        == "NVFP4"
    )
    assert (
        quant_config._resolve_quant_algo("mtp.layers.1.mixer.shared_experts.down_proj")
        == "NVFP4"
    )
    assert quant_config.is_layer_excluded("mtp.layers.0.eh_proj")
