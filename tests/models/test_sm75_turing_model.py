# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Turing (SM75) DeepSeek-V4 model assembly smoke tests."""

import json

import pytest
import torch

from vllm.config import (
    CompilationConfig,
    KernelConfig,
    LoadConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.platforms import current_platform

_capability = current_platform.get_device_capability()

pytestmark = pytest.mark.skipif(
    _capability is None or (_capability.major, _capability.minor) != (7, 5),
    reason="SM75 only",
)


def _write_dsv4_config(tmp_path) -> str:
    config = {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "vocab_size": 32000,
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "n_routed_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 256,
        "swiglu_limit": None,
        "norm_topk_prob": True,
        "scoring_func": "sqrtsoftplus",
        "rms_norm_eps": 1e-6,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 3,
        "hc_eps": 1e-2,
        "index_topk": 2,
        "num_hash_layers": 0,
        "n_shared_experts": 1,
        "hidden_act": "silu",
        "expert_dtype": "fp4",
        "head_dim": 512,
        "nope_head_dim": 448,
        "rope_head_dim": 64,
        "q_lora_rank": 128,
        "o_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "o_groups": 1,
        "kv_lora_rank": 128,
        "num_key_value_heads": 1,
        "use_mla": True,
        "moe_quant_algo": None,
        "first_k_dense_replace": 0,
        "sliding_window": 2048,
        "compress_ratios": [4, 4],
        "max_position_embeddings": 163840,
        "index_n_heads": 4,
        "index_head_dim": 128,
        "rope_theta": 10000.0,
        "compress_rope_theta": 10000.0,
        "rope_parameters": {
            "rope_type": "default",
            "factor": 32.0,
            "original_max_position_embeddings": 163840,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "apply_yarn_scaling": True,
        },
        "torch_dtype": "bfloat16",
        "quantization_config": {
            "quant_method": "deepseek_v4_fp8",
            "activation_scheme": "dynamic",
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return str(tmp_path)


def _build_vllm_config(model_path: str) -> VllmConfig:
    return VllmConfig(
        model_config=ModelConfig(
            model=model_path,
            tokenizer=model_path,
            tokenizer_mode="auto",
            dtype="half",
            seed=0,
        ),
        parallel_config=ParallelConfig(
            tensor_parallel_size=1, pipeline_parallel_size=1
        ),
        kernel_config=KernelConfig(moe_backend="auto"),
        scheduler_config=SchedulerConfig(
            max_num_batched_tokens=64,
            max_num_seqs=8,
            max_model_len=2048,
            is_encoder_decoder=False,
        ),
        compilation_config=CompilationConfig(),
        load_config=LoadConfig(load_format="dummy"),
    )


def test_turing_model_builds_and_uses_fp16_fused_moe(
    dist_init, tmp_path, default_vllm_config
):
    from vllm.models.deepseek_v4.turing.attention import TuringMLAAttention
    from vllm.models.deepseek_v4.turing.model import DeepseekV4ForCausalLM

    model_path = _write_dsv4_config(tmp_path)
    vllm_config = _build_vllm_config(model_path)
    with set_current_vllm_config(vllm_config):
        model = DeepseekV4ForCausalLM(vllm_config=vllm_config, prefix="")

    assert isinstance(model, DeepseekV4ForCausalLM)
    # MegaMoE (deep-gemm, sm90+) must never activate on SM75.
    assert model.model.use_mega_moe is False
    assert len(model.model.layers) == 2
    layer = model.model.layers[0]

    assert isinstance(layer.attn, TuringMLAAttention)
    # FP8 experts stay packed on device through the fused MoE runner.
    assert layer.ffn.experts.__class__.__name__ == "MoERunner"

    # Quantized weights must stay packed on device (no dequant to fp16).
    attn_weight_dtypes = {p.dtype for p in layer.attn.parameters() if p.dim() >= 2}
    assert torch.float8_e4m3fn in attn_weight_dtypes
    expert_dtypes = {p.dtype for p in layer.ffn.experts.parameters()}
    assert torch.uint8 in expert_dtypes
    shared_dtypes = {p.dtype for p in layer.ffn.shared_experts.parameters()}
    assert torch.float8_e4m3fn in shared_dtypes


def test_kv_bytes_per_token_fp16():
    from vllm.models.deepseek_v4.turing.weights import kv_bytes_per_token_fp16

    # DeepSeek-V4-Flash compressed MLA KV row: 512 NoPE + 64 RoPE, FP16.
    assert kv_bytes_per_token_fp16(512, 64) == 1152
    assert kv_bytes_per_token_fp16(448, 64) == 1024


def test_turing_model_rejects_mega_moe_backend(dist_init, tmp_path):
    from vllm.models.deepseek_v4.turing.model import DeepseekV4ForCausalLM

    model_path = _write_dsv4_config(tmp_path)
    vllm_config = _build_vllm_config(model_path)
    vllm_config.kernel_config.moe_backend = "deep_gemm_mega_moe"
    with (
        set_current_vllm_config(vllm_config),
        pytest.raises(NotImplementedError, match="not supported on Turing"),
    ):
        DeepseekV4ForCausalLM(vllm_config=vllm_config, prefix="")
