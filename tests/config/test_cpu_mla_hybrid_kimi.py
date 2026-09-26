# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for hybrid MLA+Mamba models on the CPU backend.

KimiLinear / Kimi-K3 is both MLA and hybrid (KDA/Mamba). Prefix caching is on
by default, which flips mamba cache mode to ``align`` and sets
``mamba_block_size``. The CPU platform then disables prefix caching because
CPU MLA does not support it, leaving a contradictory config that used to
raise ``--mamba-block-size can only be set with --enable-prefix-caching``.

See https://github.com/vllm-project/vllm/issues/52008
"""

import json

import pytest

from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import current_platform

# Minimal text_config taken from inference-optimization/Kimi-K3-0.40B.
# kv_lora_rank + model_type=kimi_linear makes ModelConfig.use_mla True.
# linear_attn_config makes the resolved class hybrid (IsHybrid).
_KIMI_K3_CONFIG = {
    "architectures": ["KimiK3ForConditionalGeneration"],
    "dtype": "float32",
    "model_type": "kimi_k3",
    "pad_token_id": 0,
    "text_config": {
        "attn_res_block_size": 4,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "first_k_dense_replace": 1,
        "head_dim": 74,
        "hidden_act": "situ",
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "intermediate_size": 2048,
        "kv_lora_rank": 128,
        "latent_moe_use_norm": True,
        "linear_attn_config": {
            "full_attn_layers": [4, 8],
            "head_dim": 32,
            "kda_layers": [1, 2, 3, 5, 6, 7],
            "num_heads": 8,
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": True,
        },
        "max_position_embeddings": 4096,
        "mla_use_nope": True,
        "mla_use_output_gate": True,
        "model_type": "kimi_linear",
        "moe_intermediate_size": 256,
        "moe_layer_freq": 1,
        "moe_renormalize": True,
        "moe_router_activation_func": "sigmoid",
        "num_attention_heads": 8,
        "num_expert_group": 1,
        "num_experts": 8,
        "num_experts_per_token": 2,
        "num_hidden_layers": 8,
        "num_key_value_heads": 8,
        "num_nextn_predict_layers": 0,
        "num_shared_experts": 1,
        "pad_token_id": 0,
        "q_lora_rank": 256,
        "qk_nope_head_dim": 64,
        "qk_rope_head_dim": 32,
        "rms_norm_eps": 1e-05,
        "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
        "rope_theta": 10000.0,
        "routed_expert_hidden_size": 512,
        "routed_scaling_factor": 1.0,
        "tie_word_embeddings": False,
        "topk_group": 1,
        "topk_method": "noaux_tc",
        "use_cache": True,
        "use_grouped_topk": True,
        "v_head_dim": 64,
        "vocab_size": 163840,
    },
}


def _write_kimi_config(tmp_path) -> str:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_KIMI_K3_CONFIG))
    return str(tmp_path)


@pytest.mark.cpu_test
@pytest.mark.skipif(not current_platform.is_cpu(), reason="CPU-backend only")
@pytest.mark.parametrize(
    "extra",
    [
        {},
        {
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
            "hf_overrides": {
                "architectures": ["KimiLinearForCausalLM"],
                "use_mla": False,
            },
        },
    ],
    ids=["defaults", "issue-52008-flags"],
)
def test_cpu_mla_hybrid_kimi_config_constructs(tmp_path, extra):
    """Engine config must construct; CPU MLA falls back to mamba mode none."""
    model = _write_kimi_config(tmp_path)
    args = EngineArgs(
        model=model,
        trust_remote_code=True,
        skip_tokenizer_init=True,
        load_format="dummy",
        **extra,
    )
    cfg = args.create_engine_config()

    assert cfg.model_config.use_mla
    assert not cfg.cache_config.enable_prefix_caching
    assert not cfg.scheduler_config.enable_chunked_prefill
    assert cfg.cache_config.mamba_cache_mode == "none"
    assert (
        cfg.cache_config.mamba_block_size is None
        or cfg.cache_config.mamba_block_size == cfg.model_config.max_model_len
    )


@pytest.mark.cpu_test
@pytest.mark.skipif(not current_platform.is_cpu(), reason="CPU-backend only")
def test_cpu_mla_hybrid_rejects_explicit_mamba_block_size(tmp_path):
    """An explicit --mamba-block-size asked for align-mode prefix caching."""
    model = _write_kimi_config(tmp_path)
    args = EngineArgs(
        model=model,
        trust_remote_code=True,
        skip_tokenizer_init=True,
        load_format="dummy",
        enable_prefix_caching=True,
        mamba_block_size=16,
    )
    with pytest.raises(ValueError, match="CPU MLA"):
        args.create_engine_config()
