# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFly accepts any compressed-tensors target scheme, while draft layers stay
BF16 or blockwise FP8, the formats their weight loader reads."""

import importlib
import json

import pytest
from torch import nn

from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.hy_v4 import HYV4Config

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="HY4 DFly requires CUDA or ROCm"
)


def _dfly_engine_supported() -> bool:
    if not current_platform.is_rocm():
        return True
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


requires_dfly_engine = pytest.mark.skipif(
    not _dfly_engine_supported(), reason="HYV4 DFly on ROCm requires gfx950"
)

EXPERTS = r"re:.*mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$"
FLOAT_ARGS = {"type": "float", "symmetric": True}
NVFP4_ARGS = {**FLOAT_ARGS, "num_bits": 4, "strategy": "tensor_group", "group_size": 16}
NVFP4 = {
    "format": "nvfp4-pack-quantized",
    "weights": {**NVFP4_ARGS, "dynamic": False},
    "input_activations": {**NVFP4_ARGS, "dynamic": "local"},
}
FP8_BLOCK = {
    "format": "float-quantized",
    "weights": {
        **FLOAT_ARGS,
        "num_bits": 8,
        "strategy": "block",
        "block_structure": [128, 128],
        "dynamic": False,
    },
    "input_activations": {
        **FLOAT_ARGS,
        "num_bits": 8,
        "strategy": "group",
        "group_size": 128,
        "dynamic": True,
    },
}
INT_ARGS = {"type": "int", "strategy": "group", "group_size": 32, "symmetric": True}
INT4 = {"format": "pack-quantized", "weights": {**INT_ARGS, "num_bits": 4}}
INT8 = {"format": "pack-quantized", "weights": {**INT_ARGS, "num_bits": 8}}


def _quantization_config(scheme, target, **kwargs):
    return {
        "quant_method": "compressed-tensors",
        "format": scheme["format"],
        "quantization_status": "compressed",
        "ignore": ["lm_head"],
        "config_groups": {"group_0": {**scheme, "targets": [target]}},
        "transform_config": {},
        **kwargs,
    }


def _create_engine_config(path, quantization_config):
    config = HYV4Config(num_hidden_layers=2).to_dict()
    config.update(
        architectures=["HYV4ForCausalLM"],
        quantization_config=quantization_config,
        dfly_num_hidden_layers=1,
        dfly_target_index_layer=[1],
        dfly_target_kv_layers=[1],
        dfly_target_index_topk=2048,
        dfly_block_size=7,
        dfly_mask_token_id=9,
        dfly_mlp_type="dense",
    )
    (path / "config.json").write_text(json.dumps(config))
    return EngineArgs(
        model=str(path),
        skip_tokenizer_init=True,
        load_format="dummy",
        dtype="bfloat16",
        max_model_len=4096,
        speculative_config={"method": "dfly"},
    ).create_engine_config()


@requires_dfly_engine
@pytest.mark.parametrize("scheme", [NVFP4, INT4], ids=["nvfp4", "int4"])
def test_target_scheme_is_unrestricted(tmp_path, scheme):
    config = _create_engine_config(tmp_path, _quantization_config(scheme, EXPERTS))
    assert config.speculative_config.method == "dfly"


@requires_dfly_engine
def test_target_kv_cache_scheme_is_rejected(tmp_path):
    kv_scheme = {**FLOAT_ARGS, "num_bits": 8, "strategy": "tensor"}
    quant = _quantization_config(NVFP4, EXPERTS, kv_cache_scheme=kv_scheme)
    with pytest.raises(ValueError, match="kv_cache_scheme or transforms"):
        _create_engine_config(tmp_path, quant)


@pytest.mark.parametrize(
    "scheme,supported",
    [(FP8_BLOCK, True), (NVFP4, False), (INT8, False)],
    ids=["fp8-block", "nvfp4", "int8"],
)
def test_draft_scheme_is_restricted(scheme, supported):
    from vllm.model_executor.layers.quantization.compressed_tensors import (
        compressed_tensors,
    )

    vendor = "amd" if current_platform.is_rocm() else "nvidia"
    dfly = importlib.import_module(f"vllm.models.hy_v4.{vendor}.dfly")
    HYV4DFlyQuantizationConfig = dfly.HYV4DFlyQuantizationConfig

    target = r"re:model\.dfly_layers\.\d+\.mlp\.down_proj$"
    source = compressed_tensors.CompressedTensorsConfig.from_config(
        _quantization_config(scheme, target)
    )
    config = HYV4DFlyQuantizationConfig.for_model(source, "dfly.model")
    layer = nn.Linear(2, 2)
    name = "dfly.model.layers.0.mlp.down_proj"
    if supported:
        assert config.get_scheme_dict(layer, name)["weights"].num_bits == 8
    else:
        with pytest.raises(ValueError, match="BF16 or blockwise FP8"):
            config.get_scheme_dict(layer, name)
