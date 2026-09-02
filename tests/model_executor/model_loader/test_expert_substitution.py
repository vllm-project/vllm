# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from transformers import DeepseekV2Config, DeepseekV2ForCausalLM

from vllm.config import (
    AttentionConfig,
    DeviceConfig,
    KernelConfig,
    LoadConfig,
    ModelConfig,
    VllmConfig,
)
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def _write_tiny_deepseek_substitution_checkpoint(model_dir: Path) -> torch.Tensor:
    config = DeepseekV2Config(
        architectures=["DeepseekV2ForCausalLM"],
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        first_k_dense_replace=0,
        n_routed_experts=2,
        n_shared_experts=1,
        num_experts_per_tok=1,
        n_group=1,
        topk_group=1,
        norm_topk_prob=False,
        q_lora_rank=None,
        kv_lora_rank=4,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        tie_word_embeddings=False,
        dtype="float16",
    )
    value_name = "model.layers.0.mlp.expert_replacements.1.value"
    config.compression_config = {
        "producer": {"name": "llm-compressor"},
        "transform_config": {
            "expert_substitution": {
                "version": 1,
                "router_semantics": {
                    "preserve_logical_expert_ids": True,
                    "preserve_router_weights": True,
                    "renormalize_after_substitution": False,
                },
                "targets": {
                    "model.layers.0.mlp.experts": {
                        "num_logical_experts": 2,
                        "weight_layout": "compact_retained_experts",
                        "replacements": {
                            "1": {
                                "format": "constant-v1",
                                "tensors": {"value": value_name},
                            }
                        },
                    }
                },
            }
        },
    }

    hf_model = DeepseekV2ForCausalLM(config).to(torch.float16)
    hf_weights = hf_model.state_dict()
    fused_gate_up_name = "model.layers.0.mlp.experts.gate_up_proj"
    fused_down_name = "model.layers.0.mlp.experts.down_proj"
    weights = {
        name: value.contiguous()
        for name, value in hf_weights.items()
        if name not in (fused_gate_up_name, fused_down_name)
    }
    retained_gate_up = hf_weights[fused_gate_up_name][0]
    retained_gate, retained_up = retained_gate_up.chunk(2, dim=0)
    retained_prefix = "model.layers.0.mlp.experts.0"
    weights[f"{retained_prefix}.gate_proj.weight"] = retained_gate.contiguous()
    weights[f"{retained_prefix}.up_proj.weight"] = retained_up.contiguous()
    weights[f"{retained_prefix}.down_proj.weight"] = hf_weights[fused_down_name][
        0
    ].contiguous()
    substitution_value = torch.arange(config.hidden_size, dtype=torch.float16)
    weights[value_name] = substitution_value
    config.save_pretrained(model_dir)
    save_file(weights, model_dir / "model.safetensors")
    return substitution_value


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like device"
)
@pytest.mark.usefixtures("dist_init", "workspace_init")
def test_transform_only_config_loads_substituted_expert_checkpoint(tmp_path: Path):
    expected_value = _write_tiny_deepseek_substitution_checkpoint(tmp_path)
    model_config = ModelConfig(
        model=str(tmp_path),
        tokenizer=str(tmp_path),
        skip_tokenizer_init=True,
        dtype="float16",
        max_model_len=32,
        enforce_eager=True,
    )

    assert model_config.quantization is None
    assert model_config.model_arch_config.quantization_config is None
    assert "quant_method" not in model_config.hf_config.compression_config

    load_config = LoadConfig(load_format="safetensors", use_tqdm_on_load=False)
    vllm_config = VllmConfig(
        model_config=model_config,
        device_config=DeviceConfig(device="cuda"),
        load_config=load_config,
        attention_config=AttentionConfig(backend=AttentionBackendEnum.TRITON_MLA),
        kernel_config=KernelConfig(moe_backend="triton"),
    )
    model = DefaultModelLoader(load_config).load_model(vllm_config, model_config)

    substitution = model.model.layers[0].mlp.experts.routed_experts.expert_substitution
    assert substitution is not None
    assert substitution.num_compute_experts == 1
    assert substitution.logical_to_physical.tolist() == [0, -1]
    torch.testing.assert_close(substitution.values.cpu(), expected_value.unsqueeze(0))
