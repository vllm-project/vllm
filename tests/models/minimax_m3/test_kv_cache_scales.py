# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint FP8 KV-cache scales must reach MiniMax-M3's sparse attention.

``MiniMaxM3SparseAttention`` owns its KV cache (it is not ``Attention``), so
its scales live at ``<prefix>.self_attn.{k,v}_scale``.
"""

import pytest
import torch

from tests.quantization.utils import (
    is_quant_method_supported,
    load_model_without_vllm_runner,
)
from vllm.config import CacheConfig
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
from vllm.transformers_utils.configs.minimax_m3 import MiniMaxM3Config

QUANT_CONFIGS = {
    "compressed-tensors": {
        "quant_method": "compressed-tensors",
        "format": "dense",
        "config_groups": {},
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

# How each exporter names the KV-cache scales of an attention layer.
SCALE_NAMES = {
    "compressed-tensors": ("{prefix}.k_scale", "{prefix}.v_scale"),
    "modelopt": ("{prefix}.k_proj.k_scale", "{prefix}.v_proj.v_scale"),
}


@pytest.mark.skipif(
    not is_quant_method_supported("fp8"),
    reason="FP8 is not supported on this GPU type.",
)
@pytest.mark.parametrize("fmt", ["compressed-tensors", "modelopt"])
def test_sparse_attention_loads_checkpoint_kv_scales(
    fmt, tmp_path, dist_init, workspace_init
):
    MiniMaxM3Config(architectures=["MiniMaxM3SparseForCausalLM"]).save_pretrained(
        tmp_path
    )
    # Layers 0-2 use dense attention and layer 3 sparse attention.
    hf_overrides = {
        "text_config": {
            "num_hidden_layers": 4,
            "num_local_experts": 8,
            "vocab_size": 8192,
        },
        "quantization_config": QUANT_CONFIGS[fmt],
    }
    model, _ = load_model_without_vllm_runner(
        str(tmp_path),
        model_config_kwargs={"hf_overrides": hf_overrides, "max_model_len": 1024},
        vllm_config_kwargs={"cache_config": CacheConfig(cache_dtype="fp8")},
        model_loader_cls=DummyModelLoader,
    )
    prefix = "model.layers.3.self_attn"
    attn = model.get_submodule(prefix)

    k_name, v_name = (n.format(prefix=prefix) for n in SCALE_NAMES[fmt])
    model.load_weights([(k_name, torch.tensor(0.25)), (v_name, torch.tensor(0.5))])
    # The model loader's post-load step for this module.
    assert isinstance(getattr(attn, "quant_method", None), BaseKVCacheMethod)
    attn.quant_method.process_weights_after_loading(attn)

    assert attn._k_scale_float == 0.25
    assert attn._v_scale_float == 0.5
