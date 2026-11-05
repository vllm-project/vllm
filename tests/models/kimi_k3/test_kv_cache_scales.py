# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint FP8 KV-cache scales must reach Kimi-Linear's MLA layers.

Kimi's ``MultiHeadLatentAttention`` owns its KV cache directly (it is neither
``Attention`` nor ``MLAAttention``), and its scales live at
``<prefix>.self_attn.{k,v}_scale``.
"""

import pytest
import torch

from tests.quantization.utils import (
    is_quant_method_supported,
    load_model_without_vllm_runner,
)
from vllm.config import CacheConfig
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.layers.quantization.modelopt import ModelOptFp8Config
from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader

MODEL = "moonshotai/Kimi-Linear-48B-A3B-Instruct"

QUANT_CONFIGS = {
    "compressed-tensors": {
        "quant_method": "compressed-tensors",
        "format": "float-quantized",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 8,
                    "type": "float",
                    "strategy": "channel",
                    "dynamic": False,
                    "symmetric": True,
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "float",
                    "strategy": "token",
                    "dynamic": True,
                    "symmetric": True,
                },
            }
        },
        "ignore": ["lm_head"],
        "kv_cache_scheme": {
            "num_bits": 8,
            "type": "float",
            "strategy": "tensor",
            "dynamic": False,
            "symmetric": True,
        },
    },
    "modelopt": {
        "quant_method": "modelopt",
        "quant_algo": "FP8",
        "kv_cache_quant_algo": "FP8",
        "exclude_modules": ["lm_head"],
    },
}

# How each exporter names the KV-cache scales of an MLA layer.
SCALE_NAMES = {
    "compressed-tensors": ("{prefix}.k_scale", "{prefix}.v_scale"),
    "modelopt": ("{prefix}.k_proj.k_scale", "{prefix}.v_proj.v_scale"),
}


class _SelfContainedAttention(torch.nn.Module, AttentionLayerBase):
    """An attention layer that is neither ``Attention`` nor ``MLAAttention``."""

    def get_attn_backend(self):
        raise NotImplementedError

    def get_kv_cache_spec(self, vllm_config):
        return None


@pytest.mark.parametrize(
    "config_cls,fmt",
    [
        (CompressedTensorsConfig, "compressed-tensors"),
        (ModelOptFp8Config, "modelopt"),
    ],
)
def test_kv_cache_method_for_any_attention_layer(config_cls, fmt):
    config = config_cls.from_config(QUANT_CONFIGS[fmt])
    method = config.get_quant_method(
        _SelfContainedAttention(), prefix="model.layers.0.self_attn"
    )
    assert isinstance(method, BaseKVCacheMethod)


@pytest.mark.parametrize(
    "config_cls,fmt",
    [
        (CompressedTensorsConfig, "compressed-tensors"),
        (ModelOptFp8Config, "modelopt"),
    ],
)
def test_kv_cache_method_creates_scales_on_any_attention_layer(config_cls, fmt):
    # With a per-tensor KV-cache scheme, `create_weights` reads nothing from the
    # layer, so it works on attention layers other than `Attention`.
    config = config_cls.from_config(QUANT_CONFIGS[fmt])
    layer = _SelfContainedAttention()
    method = config.get_quant_method(layer, prefix="model.layers.0.self_attn")
    method.create_weights(layer)
    for name in ("q_scale", "k_scale", "v_scale"):
        param = getattr(layer, name)
        assert isinstance(param, torch.nn.Parameter)
        assert param.numel() == 1


@pytest.mark.skipif(
    not is_quant_method_supported("fp8"),
    reason="FP8 is not supported on this GPU type.",
)
@pytest.mark.parametrize("fmt", ["compressed-tensors", "modelopt"])
def test_kimi_linear_loads_checkpoint_kv_scales(fmt, dist_init, workspace_init):
    # 3 KDA layers + 1 MLA layer (layer index 3).
    hf_overrides = {
        "num_hidden_layers": 4,
        "num_experts": 8,
        "linear_attn_config": {
            "full_attn_layers": [4],
            "kda_layers": [1, 2, 3],
            "head_dim": 128,
            "num_heads": 32,
            "short_conv_kernel_size": 4,
        },
        "quantization_config": QUANT_CONFIGS[fmt],
    }
    model, _ = load_model_without_vllm_runner(
        MODEL,
        model_config_kwargs={"hf_overrides": hf_overrides, "max_model_len": 1024},
        vllm_config_kwargs={"cache_config": CacheConfig(cache_dtype="fp8")},
        model_loader_cls=DummyModelLoader,
    )
    prefix = "model.layers.3.self_attn"
    attn = model.get_submodule(prefix)

    k_name, v_name = (n.format(prefix=prefix) for n in SCALE_NAMES[fmt])
    model.load_weights([(k_name, torch.tensor(0.25)), (v_name, torch.tensor(0.5))])
    # The model loader's post-load steps for this module.
    assert isinstance(getattr(attn, "quant_method", None), BaseKVCacheMethod)
    attn.quant_method.process_weights_after_loading(attn)
    attn.process_weights_after_loading()

    assert attn._k_scale_float == 0.25
    assert attn._v_scale_float == 0.5
