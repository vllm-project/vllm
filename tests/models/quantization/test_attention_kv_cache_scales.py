# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static FP8 KV-cache scales must reach the attention layers of hybrid and
MoE models whose attention layer is built by the model file itself."""

import json

import pytest
import torch

from tests.quantization.utils import is_quant_method_supported
from vllm.config import (
    CacheConfig,
    ModelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.model_loader.utils import initialize_model
from vllm.utils.torch_utils import set_default_torch_dtype

K_SCALE, V_SCALE = 0.5, 0.25

COMMON = {
    "vocab_size": 256,
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
}


def _lfm2():
    from transformers import Lfm2Config

    return Lfm2Config(
        **COMMON,
        num_hidden_layers=2,
        layer_types=["conv", "full_attention"],
        conv_dim=128,
        block_dim=128,
        architectures=["Lfm2ForCausalLM"],
    ), ["model.layers.1.self_attn"]


def _lfm2_moe():
    from transformers import Lfm2MoeConfig

    return Lfm2MoeConfig(
        **COMMON,
        num_hidden_layers=2,
        layer_types=["conv", "full_attention"],
        num_dense_layers=1,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        architectures=["Lfm2MoeForCausalLM"],
    ), ["model.layers.1.self_attn"]


def _jamba():
    from transformers import JambaConfig

    return JambaConfig(
        **COMMON,
        num_hidden_layers=2,
        attn_layer_period=2,
        attn_layer_offset=1,
        expert_layer_period=2,
        expert_layer_offset=1,
        num_experts=4,
        architectures=["JambaForCausalLM"],
    ), ["model.layers.1.self_attn"]


def _zamba2():
    from transformers import Zamba2Config

    # Two hybrid layers share one attention block, and the checkpoint stores
    # its scales once.
    return Zamba2Config(
        **COMMON,
        num_hidden_layers=4,
        layers_block_type=["mamba", "hybrid", "mamba", "hybrid"],
        num_mem_blocks=1,
        mamba_d_state=16,
        mamba_headdim=32,
        n_mamba_heads=8,
        architectures=["Zamba2ForCausalLM"],
    ), ["model.layers.1.shared_transformer.self_attn"]


def _step3_text():
    from vllm.transformers_utils.configs.step3_vl import Step3TextConfig

    return Step3TextConfig(
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_attention_groups=1,
        head_dim=32,
        share_q_dim=64,
        num_hidden_layers=2,
        vocab_size=256,
        moe_layers_enum="1",
        moe_num_experts=4,
        moe_top_k=2,
        moe_intermediate_size=64,
        share_expert_dim=64,
        architectures=["Step3TextForCausalLM"],
    ), ["model.layers.0.self_attn", "model.layers.1.self_attn"]


@pytest.mark.skipif(
    not is_quant_method_supported("fp8"),
    reason="fp8 is not supported on this GPU type.",
)
@pytest.mark.parametrize(
    "make_config", [_lfm2, _lfm2_moe, _jamba, _zamba2, _step3_text]
)
def test_kv_cache_scales_reach_attention(
    tmp_path, dist_init, workspace_init, make_config
):
    hf_config, attention_modules = make_config()
    hf_config.save_pretrained(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    # The KV-cache scheme LLM Compressor writes, without weight quantization.
    config["quantization_config"] = {
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
    }
    (tmp_path / "config.json").write_text(json.dumps(config))

    vllm_config = VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path),
            dtype="bfloat16",
            skip_tokenizer_init=True,
            max_model_len=256,
        ),
        cache_config=CacheConfig(cache_dtype="fp8"),
    )
    with (
        set_current_vllm_config(vllm_config),
        set_default_torch_dtype(torch.bfloat16),
        torch.device("cuda"),
    ):
        model = initialize_model(vllm_config=vllm_config)
        model.load_weights(
            (f"{module}.{name}", torch.tensor([scale]))
            for module in attention_modules
            for name, scale in (("k_scale", K_SCALE), ("v_scale", V_SCALE))
        )
        layers = [m for m in model.modules() if isinstance(m, Attention)]
        for layer in layers:
            layer.quant_method.process_weights_after_loading(layer)

    assert layers
    for layer in layers:
        assert (layer._k_scale_float, layer._v_scale_float) == (K_SCALE, V_SCALE)
