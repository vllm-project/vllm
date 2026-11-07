# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tiny local target configurations for Uno admission tests."""

import json

_COMMON = dict(
    vocab_size=128,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=1,
    num_attention_heads=1,
    max_position_embeddings=64,
)
_MLA = dict(
    architectures=["DeepseekV2ForCausalLM"],
    model_type="deepseek_v2",
    kv_lora_rank=8,
    q_lora_rank=8,
    qk_nope_head_dim=8,
    qk_rope_head_dim=8,
    v_head_dim=8,
)
TINY_MODEL_CONFIGS = {
    "opt": dict(architectures=["OPTForCausalLM"], model_type="opt", ffn_dim=128),
    "deepseek_v2": _MLA,
    "rswa": dict(_MLA, rswa_window=16),
    "llama4_text": dict(
        architectures=["Llama4ForCausalLM"],
        model_type="llama4_text",
        num_key_value_heads=1,
        head_dim=64,
        attention_chunk_size=16,
    ),
    "openpangu": dict(
        architectures=["PanguEmbeddedForCausalLM"],
        model_type="llama",
        num_key_value_heads=1,
        param_sink_number=16,
        qk_nope_dim=32,
        qk_rope_dim=32,
        v_channels=64,
    ),
}


def write_tiny_model_config(path, case, **overrides):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(
        json.dumps(_COMMON | TINY_MODEL_CONFIGS[case] | overrides)
    )
    return path
