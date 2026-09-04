# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Detection of NVFP4 in compressed-tensors quantization configs.

Regression tests for mixed-precision checkpoints (format "mixed-precision"),
whose top-level format string never names nvfp4 even when config groups are
nvfp4 -- and which must NOT match W4A4-float MXFP4 groups.
"""

from types import SimpleNamespace

from vllm.config.model import ModelConfig, _is_nvfp4_quant_group

NVFP4_GROUP = {
    # unsloth/Qwen3.8-27B-NVFP4 config_groups.group_1 (trimmed)
    "format": "nvfp4-pack-quantized",
    "targets": [r"re:.*mlp\.(gate|up|down)_proj$"],
    "weights": {
        "num_bits": 4,
        "type": "float",
        "strategy": "tensor_group",
        "group_size": 16,
        "symmetric": True,
    },
    "input_activations": {
        "num_bits": 4,
        "type": "float",
        "strategy": "tensor_group",
        "group_size": 16,
        "dynamic": "local",
    },
}

MXFP4_GROUP = {
    # INCModel3/GLM-5.3-Flash-MXFP4-Mixed-CT-AutoRound group_0 (trimmed):
    # also W4A4 float, but strategy "group" with group_size 32
    "format": "mxfp4-pack-quantized",
    "targets": ["Linear"],
    "weights": {
        "num_bits": 4,
        "type": "float",
        "strategy": "group",
        "group_size": 32,
        "symmetric": True,
    },
    "input_activations": {
        "num_bits": 4,
        "type": "float",
        "strategy": "group",
        "group_size": 32,
    },
}

FP8_GROUP = {
    "weights": {"num_bits": 8, "type": "float", "strategy": "channel"},
    "input_activations": {"num_bits": 8, "type": "float"},
}


def _model_config(quantization, quant_config):
    """The method only touches these two attributes; avoid a full ModelConfig."""
    fake = SimpleNamespace(
        quantization=quantization,
        model_arch_config=SimpleNamespace(quantization_config=quant_config),
    )
    return ModelConfig.is_nvfp4_quantized(fake)


def test_group_scheme_match_without_per_group_format():
    group = {k: v for k, v in NVFP4_GROUP.items() if k != "format"}
    assert _is_nvfp4_quant_group(group)


def test_group_per_group_format_match():
    group = {"format": "nvfp4-pack-quantized", "weights": {}, "targets": ["Linear"]}
    assert _is_nvfp4_quant_group(group)


def test_group_mxfp4_not_matched():
    assert not _is_nvfp4_quant_group(MXFP4_GROUP)
    # even without its per-group format, the scheme fields must not match
    group = {k: v for k, v in MXFP4_GROUP.items() if k != "format"}
    assert not _is_nvfp4_quant_group(group)


def test_group_preset_name_value_does_not_crash():
    # compressed-tensors allows {"NVFP4": ["Linear"]}-style preset groups;
    # vLLM cannot load those configs, so they must simply not match.
    assert not _is_nvfp4_quant_group(["Linear"])


def test_modelopt_fp4():
    assert _model_config("modelopt_fp4", None)


def test_ct_packed_format():
    assert _model_config("compressed-tensors", {"format": "nvfp4-pack-quantized"})


def test_ct_mixed_precision_with_nvfp4_group():
    qc = {
        "format": "mixed-precision",
        "config_groups": {"group_0": FP8_GROUP, "group_1": NVFP4_GROUP},
    }
    assert _model_config("compressed-tensors", qc)


def test_ct_mixed_precision_mxfp4_only():
    qc = {
        "format": "mixed-precision",
        "config_groups": {"group_0": FP8_GROUP, "group_1": MXFP4_GROUP},
    }
    assert not _model_config("compressed-tensors", qc)


def test_ct_pure_mxfp4_unchanged():
    # non-mixed formats must not consult config_groups at all
    qc = {
        "format": "mxfp4-pack-quantized",
        "config_groups": {"group_0": MXFP4_GROUP},
    }
    assert not _model_config("compressed-tensors", qc)


def test_ct_mixed_precision_fp8_only():
    qc = {"format": "mixed-precision", "config_groups": {"group_0": FP8_GROUP}}
    assert not _model_config("compressed-tensors", qc)


def test_non_ct_quantization():
    assert not _model_config("fp8", {"format": "nvfp4-pack-quantized"})


def test_missing_config():
    assert not _model_config("compressed-tensors", None)
