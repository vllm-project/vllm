# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFly accepts compressed-tensors NVFP4 W4A4 HY4 targets (routed experts
only, e.g. RedHatAI/Hy4-NVFP4); HY4 Flash's own checkpoint check does not."""

import copy

import pytest

from vllm.models.hy_v4.dfly_config import _validate_checkpoint_quantization

NVFP4_ARGS = {
    "num_bits": 4,
    "type": "float",
    "strategy": "tensor_group",
    "group_size": 16,
    "symmetric": True,
    "scale_dtype": "torch.float8_e4m3fn",
}

# quantization_config of RedHatAI/Hy4-NVFP4, without its long ignore list
NVFP4_CONFIG = {
    "quant_method": "compressed-tensors",
    "format": "nvfp4-pack-quantized",
    "quantization_status": "compressed",
    "kv_cache_scheme": None,
    "transform_config": {},
    "ignore": ["lm_head"],
    "config_groups": {
        "group_0": {
            "format": "nvfp4-pack-quantized",
            "targets": [r"re:.*mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$"],
            "weights": {**NVFP4_ARGS, "dynamic": False},
            "input_activations": {**NVFP4_ARGS, "dynamic": "local"},
            "output_activations": None,
        }
    },
}


def test_dfly_accepts_nvfp4_target():
    _validate_checkpoint_quantization(NVFP4_CONFIG, allow_nvfp4=True)


def _weight_only(config):
    config["config_groups"]["group_0"]["input_activations"] = None


def _kv_cache_scheme(config):
    config["kv_cache_scheme"] = {"num_bits": 8, "type": "float"}


@pytest.mark.parametrize(
    "allow_nvfp4,mutate",
    [
        (False, None),
        (True, _weight_only),
        (True, _kv_cache_scheme),
    ],
    ids=["hy4-flash-checkpoint", "weight-only-nvfp4", "kv-cache-scheme"],
)
def test_rejected_nvfp4_configs(allow_nvfp4, mutate):
    config = copy.deepcopy(NVFP4_CONFIG)
    if mutate is not None:
        mutate(config)
    with pytest.raises(ValueError, match="HYV4 requires"):
        _validate_checkpoint_quantization(config, allow_nvfp4=allow_nvfp4)
