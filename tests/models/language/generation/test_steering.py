# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for activation steering on Gemma 3.

Covers global steering via the worker API, per-request steering via
SamplingParams, concurrent batching with CUDA graphs, and three-tier
prefill/decode phase-specific steering.
"""

import math

import pytest
import requests
import torch

from vllm import SamplingParams
from vllm.model_executor.layers.steering import (
    DEFAULT_HOOK_POINT,
    HOOK_POINT_TABLE_ATTR,
)

from ...registry import HF_EXAMPLE_MODELS

MODEL = "google/gemma-3-4b-it"

_SMALL_DECODER_OVERRIDES = {
    "num_hidden_layers": 1,
    "hidden_size": 512,
    "intermediate_size": 1024,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
}

_QWEN_MOE_OVERRIDES = {
    **_SMALL_DECODER_OVERRIDES,
    "num_experts": 2,
    "num_experts_per_tok": 2,
    "decoder_sparse_step": 1,
    "moe_intermediate_size": 1024,
    "shared_expert_intermediate_size": 0,
}

_QWEN3_NEXT_OVERRIDES = {
    **_QWEN_MOE_OVERRIDES,
    "num_hidden_layers": 2,
    "head_dim": 64,
    "linear_key_head_dim": 64,
    "linear_value_head_dim": 64,
    "linear_num_key_heads": 4,
    "linear_num_value_heads": 8,
    "layer_types": ["full_attention", "linear_attention"],
}

_LOOPCODER_OVERRIDES = {
    **_SMALL_DECODER_OVERRIDES,
    "loop_num": 2,
    "loop_window_size": 32,
}

_STABLELM_OVERRIDES = {
    **_SMALL_DECODER_OVERRIDES,
    "num_hidden_layers": 2,
}

_GROK_OVERRIDES = {
    **_SMALL_DECODER_OVERRIDES,
    "num_local_experts": 2,
    "num_experts_per_tok": 2,
}

_STEP3P5_OVERRIDES = {
    **_SMALL_DECODER_OVERRIDES,
    "num_hidden_layers": 2,
    "num_attention_groups": 8,
    "head_dim": 64,
    "moe_layers_enum": "1",
    "moe_num_experts": 2,
    "moe_top_k": 1,
    "moe_intermediate_size": 1024,
    "share_expert_dim": 0,
}

_MINIMAX_OVERRIDES = {
    **_SMALL_DECODER_OVERRIDES,
    "head_dim": 64,
    "rotary_dim": 64,
    "num_local_experts": 2,
    "num_experts_per_tok": 2,
}

_MINIMAX_TEXT_OVERRIDES = {
    **_MINIMAX_OVERRIDES,
    "num_local_experts": 1,
}

_AXK1_OVERRIDES = {
    **_SMALL_DECODER_OVERRIDES,
    "num_hidden_layers": 2,
    "qk_nope_head_dim": 0,
    "qk_rope_head_dim": 0,
    "v_head_dim": 64,
    "first_k_dense_replace": 999,
    "moe_layer_freq": 1,
    "n_routed_experts": None,
}

_GENERIC_MOE_OVERRIDES = {
    **_SMALL_DECODER_OVERRIDES,
    "num_hidden_layers": 2,
    "num_local_experts": 2,
    "num_experts_per_tok": 2,
}

_EXAONE_MOE_OVERRIDES = {
    **_SMALL_DECODER_OVERRIDES,
    "num_hidden_layers": 2,
    "num_experts": 2,
    "num_experts_per_tok": 1,
    "moe_intermediate_size": 1024,
    "num_shared_experts": 0,
    "routed_scaling_factor": 1.0,
    "norm_topk_prob": True,
    "n_group": 1,
    "topk_group": 1,
    "is_moe_layer": [True, True],
}

_GLM4_MOE_OVERRIDES = {
    **_SMALL_DECODER_OVERRIDES,
    "num_hidden_layers": 2,
    "head_dim": 64,
    "n_routed_experts": 2,
    "n_shared_experts": 0,
    "num_experts_per_tok": 1,
    "moe_intermediate_size": 1024,
    "first_k_dense_replace": 0,
    "norm_topk_prob": True,
    "n_group": 1,
    "topk_group": 1,
    "routed_scaling_factor": 1.0,
    "use_qk_norm": False,
}

_ERNIE45_MOE_OVERRIDES = {
    **_SMALL_DECODER_OVERRIDES,
    "num_hidden_layers": 2,
    "moe_num_experts": 2,
    "moe_k": 1,
    "moe_intermediate_size": 1024,
    "moe_num_shared_experts": 0,
    "moe_layer_start_index": 0,
    "moe_layer_end_index": 1,
    "moe_layer_interval": 1,
    "use_moe": True,
}

_GRANITEMOE_OVERRIDES = {
    **_GENERIC_MOE_OVERRIDES,
    "attention_multiplier": 1.0,
    "residual_multiplier": 1.0,
    "embedding_multiplier": 1.0,
}

_GRANITEMOE_SHARED_OVERRIDES = {
    **_GRANITEMOE_OVERRIDES,
    "shared_intermediate_size": 1024,
}

_DEEPSEEK_MOE_OVERRIDES = {
    **_SMALL_DECODER_OVERRIDES,
    "num_hidden_layers": 2,
    "model_type": "deepseek",
    "n_routed_experts": 2,
    "n_shared_experts": 0,
    "num_experts_per_tok": 1,
    "first_k_dense_replace": 0,
    "moe_layer_freq": 1,
    "moe_intermediate_size": 1024,
    "norm_topk_prob": True,
    "qk_rope_head_dim": 0,
}

_EXAONE_OVERRIDES = {
    **_SMALL_DECODER_OVERRIDES,
    "activation_function": "silu",
    "layer_norm_epsilon": 1e-5,
}

_EXAONE4_OVERRIDES = {
    **_SMALL_DECODER_OVERRIDES,
    "num_hidden_layers": 2,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-5,
    "layer_types": ["full_attention", "full_attention"],
    "sliding_window": 32,
}

_GRANITE_OVERRIDES = {
    **_SMALL_DECODER_OVERRIDES,
    "hidden_act": "silu",
    "attention_multiplier": 1.0,
    "residual_multiplier": 1.0,
    "embedding_multiplier": 1.0,
}

_OPT_OVERRIDES = {
    **_SMALL_DECODER_OVERRIDES,
    "ffn_dim": 1024,
    "enable_bias": False,
    "do_layer_norm_before": True,
    "layer_norm_elementwise_affine": True,
    "_remove_final_layer_norm": False,
    "word_embed_proj_dim": 512,
    "activation_function": "relu",
}

_GPT_NEOX_OVERRIDES = {
    **_SMALL_DECODER_OVERRIDES,
    "hidden_act": "gelu",
    "intermediate_size": 1024,
    "layer_norm_eps": 1e-5,
    "use_parallel_residual": True,
}

_PHI_OVERRIDES = {
    **_SMALL_DECODER_OVERRIDES,
    "hidden_act": "gelu_new",
    "tie_word_embeddings": False,
}

_PERSIMMON_OVERRIDES = {
    **_SMALL_DECODER_OVERRIDES,
    "qk_layernorm": False,
    "layer_norm_eps": 1e-5,
}

_STARCODER2_OVERRIDES = {
    **_SMALL_DECODER_OVERRIDES,
    "hidden_act": "gelu_pytorch_tanh",
    "norm_epsilon": 1e-5,
    "use_bias": True,
}

_JAIS2_OVERRIDES = {
    **_SMALL_DECODER_OVERRIDES,
    "hidden_act": "relu2",
    "layer_norm_eps": 1e-5,
    "attention_bias": False,
    "mlp_bias": False,
}

_OLMO_OVERRIDES = {
    **_SMALL_DECODER_OVERRIDES,
    "hidden_act": "silu",
    "attention_bias": False,
    "clip_qkv": None,
}

_FALCON_OVERRIDES = {
    **_SMALL_DECODER_OVERRIDES,
    "ffn_hidden_size": 1024,
    "bias": False,
}

_ARCTIC_OVERRIDES = {
    **_SMALL_DECODER_OVERRIDES,
    "num_local_experts": 2,
    "num_experts_per_tok": 2,
    "moe_layer_frequency": 1,
    "use_residual": True,
    "parallel_attn_mlp_res": True,
    "enable_expert_tensor_parallelism": False,
}

_FLEX_OLMO_OVERRIDES = {
    **_SMALL_DECODER_OVERRIDES,
    "num_experts": 2,
    "num_experts_per_tok": 2,
}

_HUNYUAN_DENSE_OVERRIDES = {
    **_SMALL_DECODER_OVERRIDES,
    "head_dim": 64,
    "attention_head_dim": 64,
    "use_cla": False,
    "cla_share_factor": 1,
    "use_qk_norm": True,
    "mlp_bias": False,
}


def _mimo_v2_flash_overrides(config):
    """Callable override: shrink dims and strip block-scaled fp8 quant.

    Block-scaled fp8 (``quantization_config: fp8``) only works on Hopper/Ada;
    the test targets must run without a real checkpoint on any CUDA card,
    so we clear the quantization config entirely.
    """
    config.num_hidden_layers = 1
    config.hidden_size = 512
    config.intermediate_size = 1024
    config.num_attention_heads = 8
    config.num_key_value_heads = 4
    config.head_dim = 64
    config.v_head_dim = 64
    config.partial_rotary_factor = 1.0
    config.hybrid_layer_pattern = [0]
    config.moe_layer_freq = [0]
    config.attention_value_scale = None
    if hasattr(config, "quantization_config"):
        delattr(config, "quantization_config")
    return config


def _param2moe_overrides(config):
    """Callable override: shrink dims, force MoE on layer 1, strip fp8."""
    config.num_hidden_layers = 2
    config.hidden_size = 512
    config.intermediate_size = 1024
    config.num_attention_heads = 8
    config.num_key_value_heads = 8
    config.head_dim = 64
    config.first_k_dense_replace = 1
    config.num_experts = 2
    config.num_experts_per_tok = 2
    config.num_shared_experts = 2
    config.moe_intermediate_size = 1024
    config.moe_shared_expert_intermediate_size = 1024
    config.n_group = 1
    config.topk_group = 1
    config.num_nextn_predict_layers = 0
    config.routed_scaling_factor = 1.0
    config.partial_rotary_factor = 1.0
    config.use_qk_norm = True
    if hasattr(config, "quantization_config"):
        delattr(config, "quantization_config")
    return config


_TRUST_REMOTE_EAGER = {"trust_remote_code": True, "enforce_eager": True}
_EAGER_ONLY = {"enforce_eager": True}
_NO_REMOTE_EAGER = {"trust_remote_code": False, "enforce_eager": True}

PHASE1_DISCOVERY_CASES = [
    pytest.param("Qwen/Qwen3-0.6B", None, None, id="qwen3"),
    pytest.param(
        "Qwen/Qwen3-Next-80B-A3B-Instruct",
        _QWEN3_NEXT_OVERRIDES,
        {"enforce_eager": True, "enable_chunked_prefill": True},
        id="qwen3-next",
    ),
    pytest.param(
        "Qwen/Qwen1.5-MoE-A2.7B-Chat",
        _QWEN_MOE_OVERRIDES,
        {"enforce_eager": True},
        id="qwen2-moe",
    ),
    pytest.param(
        "Qwen/Qwen3-30B-A3B",
        _QWEN_MOE_OVERRIDES,
        {"enforce_eager": True},
        id="qwen3-moe",
    ),
    pytest.param(
        "ByteDance/Ouro-1.4B",
        _SMALL_DECODER_OVERRIDES,
        {"enforce_eager": True},
        id="ouro",
    ),
    pytest.param(
        "ByteDance-Seed/Seed-OSS-36B-Instruct",
        _SMALL_DECODER_OVERRIDES,
        {"enforce_eager": True},
        id="seed-oss",
    ),
    pytest.param(
        "IQuestLab/IQuest-Coder-V1-40B-Loop-Instruct",
        _LOOPCODER_OVERRIDES,
        {"enforce_eager": True},
        id="loopcoder",
    ),
]

PHASE1_GENERATION_CASES = [
    pytest.param("Qwen/Qwen3-0.6B", None, None, id="qwen3"),
    pytest.param(
        "Qwen/Qwen3-Next-80B-A3B-Instruct",
        _QWEN3_NEXT_OVERRIDES,
        {"enforce_eager": True, "enable_chunked_prefill": True},
        id="qwen3-next",
    ),
    pytest.param(
        "Qwen/Qwen1.5-MoE-A2.7B-Chat",
        _QWEN_MOE_OVERRIDES,
        {"enforce_eager": True},
        id="qwen2-moe",
    ),
    pytest.param(
        "Qwen/Qwen3-30B-A3B",
        _QWEN_MOE_OVERRIDES,
        {"enforce_eager": True},
        id="qwen3-moe",
    ),
    pytest.param(
        "ByteDance-Seed/Seed-OSS-36B-Instruct",
        _SMALL_DECODER_OVERRIDES,
        {"enforce_eager": True},
        id="seed-oss",
    ),
    pytest.param(
        "ByteDance/Ouro-1.4B",
        _SMALL_DECODER_OVERRIDES,
        {"enforce_eager": True},
        id="ouro",
    ),
    pytest.param(
        "IQuestLab/IQuest-Coder-V1-40B-Loop-Instruct",
        _LOOPCODER_OVERRIDES,
        {"enforce_eager": True},
        id="loopcoder",
    ),
]

PHASE2_DISCOVERY_CASES = [
    pytest.param(
        "baichuan-inc/Baichuan2-7B-chat",
        _SMALL_DECODER_OVERRIDES,
        _TRUST_REMOTE_EAGER,
        id="baichuan",
    ),
    pytest.param(
        "internlm/internlm2-chat-7b",
        _SMALL_DECODER_OVERRIDES,
        _TRUST_REMOTE_EAGER,
        id="internlm2",
    ),
    pytest.param(
        "OrionStarAI/Orion-14B-Chat",
        _SMALL_DECODER_OVERRIDES,
        _TRUST_REMOTE_EAGER,
        id="orion",
    ),
    pytest.param(
        "upstage/solar-pro-preview-instruct",
        _SMALL_DECODER_OVERRIDES,
        _EAGER_ONLY,
        id="solar",
    ),
    pytest.param(
        "stabilityai/stablelm-3b-4e1t",
        _SMALL_DECODER_OVERRIDES,
        _EAGER_ONLY,
        id="stablelm",
    ),
    pytest.param(
        "nvidia/Minitron-8B-Base",
        _SMALL_DECODER_OVERRIDES,
        _EAGER_ONLY,
        id="nemotron",
    ),
    pytest.param(
        "arcee-ai/AFM-4.5B-Base",
        _SMALL_DECODER_OVERRIDES,
        _EAGER_ONLY,
        id="arcee",
    ),
    pytest.param(
        "naver-hyperclovax/HyperCLOVAX-SEED-Think-14B",
        _SMALL_DECODER_OVERRIDES,
        _TRUST_REMOTE_EAGER,
        id="hyperclovax",
    ),
    pytest.param(
        "zai-org/GLM-4-9B-0414",
        _SMALL_DECODER_OVERRIDES,
        _EAGER_ONLY,
        id="glm4",
    ),
    pytest.param(
        "hpcai-tech/grok-1",
        _GROK_OVERRIDES,
        _TRUST_REMOTE_EAGER,
        id="grok1",
    ),
    pytest.param(
        "CohereLabs/c4ai-command-r7b-12-2024",
        _SMALL_DECODER_OVERRIDES,
        _TRUST_REMOTE_EAGER,
        id="commandr",
    ),
    pytest.param(
        "FreedomIntelligence/openPangu-Embedded-7B-V1.1",
        _SMALL_DECODER_OVERRIDES,
        _TRUST_REMOTE_EAGER,
        id="openpangu",
    ),
]

PHASE2_GENERATION_CASES = [
    pytest.param(
        "baichuan-inc/Baichuan2-7B-chat",
        _SMALL_DECODER_OVERRIDES,
        _TRUST_REMOTE_EAGER,
        500.0,
        id="baichuan",
    ),
    pytest.param(
        "OrionStarAI/Orion-14B-Chat",
        _SMALL_DECODER_OVERRIDES,
        _TRUST_REMOTE_EAGER,
        500.0,
        id="orion",
    ),
    pytest.param(
        "CohereLabs/c4ai-command-r7b-12-2024",
        _SMALL_DECODER_OVERRIDES,
        _TRUST_REMOTE_EAGER,
        500.0,
        id="commandr",
    ),
    pytest.param(
        "zai-org/GLM-4-9B-0414",
        _SMALL_DECODER_OVERRIDES,
        _EAGER_ONLY,
        500.0,
        id="glm4",
    ),
    pytest.param(
        "hpcai-tech/grok-1",
        _GROK_OVERRIDES,
        _TRUST_REMOTE_EAGER,
        500.0,
        id="grok1",
    ),
]

PHASE3_DISCOVERY_CASES = [
    pytest.param(
        "swiss-ai/Apertus-8B-Instruct-2509",
        _SMALL_DECODER_OVERRIDES,
        _TRUST_REMOTE_EAGER,
        id="apertus",
    ),
    pytest.param(
        "MiniMaxAI/MiniMax-Text-01",
        _MINIMAX_TEXT_OVERRIDES,
        _EAGER_ONLY,
        id="minimax-text",
    ),
    pytest.param(
        "MiniMaxAI/MiniMax-M2",
        _MINIMAX_OVERRIDES,
        _EAGER_ONLY,
        id="minimax-m2",
    ),
    pytest.param(
        "skt/A.X-K1",
        _AXK1_OVERRIDES,
        _EAGER_ONLY,
        id="axk1",
    ),
]

PHASE3_GENERATION_CASES = [
    pytest.param(
        "swiss-ai/Apertus-8B-Instruct-2509",
        _SMALL_DECODER_OVERRIDES,
        _TRUST_REMOTE_EAGER,
        500.0,
        id="apertus",
    ),
    pytest.param(
        "MiniMaxAI/MiniMax-M2",
        _MINIMAX_OVERRIDES,
        _EAGER_ONLY,
        500.0,
        id="minimax-m2",
    ),
]

PHASE4_DISCOVERY_CASES = [
    pytest.param(
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        _GENERIC_MOE_OVERRIDES,
        _EAGER_ONLY,
        id="mixtral",
    ),
    pytest.param(
        "baidu/ERNIE-4.5-21B-A3B-PT",
        _ERNIE45_MOE_OVERRIDES,
        _EAGER_ONLY,
        id="ernie45-moe",
    ),
    pytest.param(
        "ibm/PowerMoE-3b",
        _GRANITEMOE_OVERRIDES,
        _EAGER_ONLY,
        id="granitemoe",
    ),
    pytest.param(
        "ibm-research/moe-7b-1b-active-shared-experts",
        _GRANITEMOE_SHARED_OVERRIDES,
        _EAGER_ONLY,
        id="granitemoe-shared",
    ),
]

PHASE4_GENERATION_CASES = [
    pytest.param(
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        _GENERIC_MOE_OVERRIDES,
        _EAGER_ONLY,
        500.0,
        id="mixtral",
    ),
    pytest.param(
        "ibm-research/moe-7b-1b-active-shared-experts",
        _GRANITEMOE_SHARED_OVERRIDES,
        _EAGER_ONLY,
        500.0,
        id="granitemoe-shared",
    ),
]

PHASE5_DISCOVERY_CASES = [
    pytest.param(
        "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct",
        _EXAONE_OVERRIDES,
        _TRUST_REMOTE_EAGER,
        id="exaone",
    ),
    pytest.param(
        "LGAI-EXAONE/EXAONE-4.0-32B",
        _EXAONE4_OVERRIDES,
        _EAGER_ONLY,
        id="exaone4",
    ),
    pytest.param(
        "ibm/PowerLM-3b",
        _GRANITE_OVERRIDES,
        _EAGER_ONLY,
        id="granite",
    ),
    pytest.param(
        "facebook/opt-350m",
        _OPT_OVERRIDES,
        _EAGER_ONLY,
        id="opt",
    ),
    pytest.param(
        "EleutherAI/pythia-70m",
        _GPT_NEOX_OVERRIDES,
        _EAGER_ONLY,
        id="gpt-neox",
    ),
    pytest.param(
        "microsoft/phi-2",
        _PHI_OVERRIDES,
        _EAGER_ONLY,
        id="phi",
    ),
    pytest.param(
        "adept/persimmon-8b-chat",
        _PERSIMMON_OVERRIDES,
        _EAGER_ONLY,
        id="persimmon",
    ),
    pytest.param(
        "bigcode/starcoder2-3b",
        _STARCODER2_OVERRIDES,
        _EAGER_ONLY,
        id="starcoder2",
    ),
    pytest.param(
        "inceptionai/Jais-2-8B-Chat",
        _JAIS2_OVERRIDES,
        _EAGER_ONLY,
        id="jais2",
    ),
    pytest.param(
        "allenai/OLMo-1B-hf",
        _OLMO_OVERRIDES,
        _EAGER_ONLY,
        id="olmo",
    ),
    pytest.param(
        "allenai/OLMo-2-0425-1B",
        _SMALL_DECODER_OVERRIDES,
        _EAGER_ONLY,
        id="olmo2",
    ),
    pytest.param(
        "tiiuae/falcon-7b",
        _FALCON_OVERRIDES,
        _NO_REMOTE_EAGER,
        id="falcon",
    ),
]

PHASE5_GENERATION_CASES = [
    pytest.param(
        "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct",
        _EXAONE_OVERRIDES,
        _TRUST_REMOTE_EAGER,
        500.0,
        id="exaone",
    ),
    pytest.param(
        "LGAI-EXAONE/EXAONE-4.0-32B",
        _EXAONE4_OVERRIDES,
        _EAGER_ONLY,
        500.0,
        id="exaone4",
    ),
    pytest.param(
        "ibm/PowerLM-3b",
        _GRANITE_OVERRIDES,
        _EAGER_ONLY,
        500.0,
        id="granite",
    ),
    pytest.param(
        "facebook/opt-350m",
        _OPT_OVERRIDES,
        _EAGER_ONLY,
        500.0,
        id="opt",
    ),
    pytest.param(
        "EleutherAI/pythia-70m",
        _GPT_NEOX_OVERRIDES,
        _EAGER_ONLY,
        500.0,
        id="gpt-neox",
    ),
    pytest.param(
        "microsoft/phi-2",
        _PHI_OVERRIDES,
        _EAGER_ONLY,
        500.0,
        id="phi",
    ),
    pytest.param(
        "bigcode/starcoder2-3b",
        _STARCODER2_OVERRIDES,
        _EAGER_ONLY,
        500.0,
        id="starcoder2",
    ),
    pytest.param(
        "inceptionai/Jais-2-8B-Chat",
        _JAIS2_OVERRIDES,
        _EAGER_ONLY,
        500.0,
        id="jais2",
    ),
    pytest.param(
        "allenai/OLMo-2-0425-1B",
        _SMALL_DECODER_OVERRIDES,
        _EAGER_ONLY,
        500.0,
        id="olmo2",
    ),
    pytest.param(
        "tiiuae/falcon-7b",
        _FALCON_OVERRIDES,
        _NO_REMOTE_EAGER,
        500.0,
        id="falcon",
    ),
]

PHASE6_DISCOVERY_CASES = [
    # arctic is steering-wired but cannot be loaded under torch >= 2.9 because
    # ``arctic.local_moe_fused`` passes ``inplace=True`` to ``fused_experts``
    # unconditionally. Re-enable once that pre-existing issue is fixed.
    pytest.param(
        "allenai/Flex-reddit-2x7B-1T",
        _FLEX_OLMO_OVERRIDES,
        _EAGER_ONLY,
        id="flex-olmo",
    ),
    pytest.param(
        "tencent/Hunyuan-7B-Instruct",
        _HUNYUAN_DENSE_OVERRIDES,
        _EAGER_ONLY,
        id="hunyuan-dense",
    ),
    pytest.param(
        "XiaomiMiMo/MiMo-V2-Flash",
        _mimo_v2_flash_overrides,
        _TRUST_REMOTE_EAGER,
        id="mimo-v2-flash",
    ),
    pytest.param(
        "bharatgenai/Param2-17B-A2.4B-Thinking",
        _param2moe_overrides,
        _TRUST_REMOTE_EAGER,
        id="param2moe",
    ),
]

PHASE6_GENERATION_CASES = [
    # arctic skipped — see PHASE6_DISCOVERY_CASES comment above.
    pytest.param(
        "allenai/Flex-reddit-2x7B-1T",
        _FLEX_OLMO_OVERRIDES,
        _EAGER_ONLY,
        500.0,
        id="flex-olmo",
    ),
    pytest.param(
        "tencent/Hunyuan-7B-Instruct",
        _HUNYUAN_DENSE_OVERRIDES,
        _EAGER_ONLY,
        500.0,
        id="hunyuan-dense",
    ),
    pytest.param(
        "XiaomiMiMo/MiMo-V2-Flash",
        _mimo_v2_flash_overrides,
        _TRUST_REMOTE_EAGER,
        500.0,
        id="mimo-v2-flash",
    ),
    pytest.param(
        "bharatgenai/Param2-17B-A2.4B-Thinking",
        _param2moe_overrides,
        _TRUST_REMOTE_EAGER,
        500.0,
        id="param2moe",
    ),
]

# Shorthand
_HP = DEFAULT_HOOK_POINT.value
_TABLE_ATTR = HOOK_POINT_TABLE_ATTR[DEFAULT_HOOK_POINT]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _discover_layers(llm):
    """Return (target_layer, hidden_size) for the default hook point."""

    def _discover(worker):
        layers = {}
        model_inst = worker.model_runner.get_model()
        for mod in model_inst.modules():
            if hasattr(mod, _TABLE_ATTR) and hasattr(mod, "layer_idx"):
                layers[mod.layer_idx] = getattr(mod, _TABLE_ATTR).shape[1]
        return layers

    layer_info = llm.llm.collective_rpc(_discover)[0]
    target_layer = max(layer_info.keys()) // 2
    hidden_size = layer_info[target_layer]
    return target_layer, hidden_size


def _gen_tokens(llm, prompt, sampling):
    """Generate and return token ids list."""
    result = llm.llm.generate([prompt], sampling)
    return list(result[0].outputs[0].token_ids)


def _gen_tokens_and_cumulative_logprob(llm, prompt, sampling):
    """Generate and return token ids with cumulative logprob."""
    result = llm.llm.generate([prompt], sampling)
    output = result[0].outputs[0]
    return list(output.token_ids), output.cumulative_logprob


def _runner_kwargs(
    hf_overrides: dict | None, extra_runner_kwargs: dict | None = None
) -> dict:
    kwargs = {
        "load_format": "dummy",
        "max_model_len": 256,
        "enable_steering": True,
        "max_steering_configs": 4,
    }
    if hf_overrides is not None:
        kwargs["hf_overrides"] = hf_overrides
    if extra_runner_kwargs is not None:
        kwargs.update(extra_runner_kwargs)
    return kwargs


def _skip_if_cuda_unavailable_or_below(min_memory_gib: float, model: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip(f"{model} real-weights steering test requires CUDA.")
    total_memory_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if total_memory_gib < min_memory_gib:
        pytest.skip(
            f"{model} real-weights steering test requires a GPU with at least "
            f"{min_memory_gib:.0f} GiB of memory."
        )


def _maybe_skip_model_access_failure(exc: Exception, model: str) -> None:
    if isinstance(exc, requests.exceptions.RequestException):
        pytest.skip(f"{model} test skipped due to model download timeout/error: {exc}")
    if isinstance(exc, OSError):
        msg = str(exc).lower()
        if "gated repo" in msg or "connection error" in msg or "read timeout" in msg:
            pytest.skip(f"{model} test skipped due to model access error: {exc}")
    if isinstance(exc, TypeError):
        msg = str(exc)
        if "expected str, bytes or os.PathLike object, not NoneType" in msg:
            pytest.skip(f"{model} test skipped due to model setup issue: {exc}")


@pytest.mark.parametrize(
    ("model", "hf_overrides", "extra_runner_kwargs"), PHASE1_DISCOVERY_CASES
)
def test_steering_layers_discovered_for_supported_families(
    vllm_runner,
    monkeypatch,
    model: str,
    hf_overrides: dict | None,
    extra_runner_kwargs: dict | None,
) -> None:
    """Steering buffers should be discoverable beyond Gemma3."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        with vllm_runner(
            model, **_runner_kwargs(hf_overrides, extra_runner_kwargs)
        ) as llm:
            target_layer, hidden_size = _discover_layers(llm)

            assert target_layer >= 0
            assert hidden_size > 0


@pytest.mark.parametrize(
    ("model", "hf_overrides", "extra_runner_kwargs"), PHASE1_GENERATION_CASES
)
def test_steering_changes_output_qwen_and_bytedance_family(
    vllm_runner,
    monkeypatch,
    model: str,
    hf_overrides: dict | None,
    extra_runner_kwargs: dict | None,
) -> None:
    """Qwen and ByteDance decoder families should respond to steering."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        prompt = "What does the fox say? " * 32
        sampling = SamplingParams(max_tokens=10, temperature=0.0)

        runner_kwargs = _runner_kwargs(hf_overrides, extra_runner_kwargs)
        runner_kwargs["enable_prefix_caching"] = True

        with vllm_runner(model, **runner_kwargs) as llm:
            baseline_tokens = _gen_tokens(llm, prompt, sampling)

            assert llm.llm.reset_prefix_cache()

            target_layer, hidden_size = _discover_layers(llm)

            vec = [500.0] * hidden_size
            llm.llm.collective_rpc(
                "set_steering_vectors",
                kwargs={"vectors": {_HP: {target_layer: vec}}},
            )

            steered_tokens = _gen_tokens(llm, prompt, sampling)

            assert steered_tokens != baseline_tokens, (
                f"Steering should change output for {model}"
            )

            llm.llm.collective_rpc("clear_steering_vectors")
            assert llm.llm.reset_prefix_cache()

            restored_tokens = _gen_tokens(llm, prompt, sampling)

            assert restored_tokens == baseline_tokens, (
                f"Clearing steering should restore baseline for {model}"
            )


@pytest.mark.parametrize(
    ("model", "hf_overrides", "extra_runner_kwargs"), PHASE2_DISCOVERY_CASES
)
def test_steering_layers_discovered_small_decoder_family(
    vllm_runner,
    monkeypatch,
    model: str,
    hf_overrides: dict | None,
    extra_runner_kwargs: dict | None,
) -> None:
    """Small decoder families (baichuan, internlm2, orion, solar, stablelm,
    nemotron, arcee, hyperclovax, glm4, grok1, commandr, openpangu) should
    expose steerable layers."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        with vllm_runner(
            model, **_runner_kwargs(hf_overrides, extra_runner_kwargs)
        ) as llm:
            target_layer, hidden_size = _discover_layers(llm)

            assert target_layer >= 0
            assert hidden_size > 0


@pytest.mark.parametrize(
    ("model", "hf_overrides", "extra_runner_kwargs", "vector_scale"),
    PHASE2_GENERATION_CASES,
)
def test_steering_changes_output_small_decoder_family(
    vllm_runner,
    monkeypatch,
    model: str,
    hf_overrides: dict | None,
    extra_runner_kwargs: dict | None,
    vector_scale: float,
) -> None:
    """Small decoder families should respond to steering."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        prompt = "What does the fox say? " * 32
        sampling = SamplingParams(max_tokens=10, temperature=0.0, logprobs=5)

        runner_kwargs = _runner_kwargs(hf_overrides, extra_runner_kwargs)
        runner_kwargs["enable_prefix_caching"] = True

        with vllm_runner(model, **runner_kwargs) as llm:
            baseline_tokens, baseline_logprob = _gen_tokens_and_cumulative_logprob(
                llm, prompt, sampling
            )

            assert llm.llm.reset_prefix_cache()

            target_layer, hidden_size = _discover_layers(llm)

            vec = [vector_scale] * hidden_size
            llm.llm.collective_rpc(
                "set_steering_vectors",
                kwargs={"vectors": {_HP: {target_layer: vec}}},
            )

            steered_tokens, steered_logprob = _gen_tokens_and_cumulative_logprob(
                llm, prompt, sampling
            )

            assert steered_tokens != baseline_tokens or not math.isclose(
                steered_logprob,
                baseline_logprob,
                rel_tol=0.0,
                abs_tol=1e-6,
            ), f"Steering should change output or logprob for {model}"

            llm.llm.collective_rpc("clear_steering_vectors")
            assert llm.llm.reset_prefix_cache()

            restored_tokens, restored_logprob = _gen_tokens_and_cumulative_logprob(
                llm, prompt, sampling
            )

            assert restored_tokens == baseline_tokens, (
                f"Clearing steering should restore baseline for {model}"
            )
            assert math.isclose(
                restored_logprob,
                baseline_logprob,
                rel_tol=0.0,
                abs_tol=1e-6,
            ), f"Clearing steering should restore baseline logprob for {model}"


def test_stablelm_steering_changes_output_real_weights(
    vllm_runner, monkeypatch
) -> None:
    """StableLM needs a real checkpoint to produce a meaningful steering signal."""
    model = "stabilityai/stablelm-3b-4e1t"
    model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
    model_info.check_available_online(on_fail="skip")

    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        prompt = "What does the fox say? " * 32
        sampling = SamplingParams(max_tokens=10, temperature=0.0, logprobs=5)

        with vllm_runner(
            model,
            max_model_len=256,
            enable_steering=True,
            max_steering_configs=4,
            enable_prefix_caching=True,
            enforce_eager=True,
        ) as llm:
            baseline_tokens, baseline_logprob = _gen_tokens_and_cumulative_logprob(
                llm, prompt, sampling
            )

            assert llm.llm.reset_prefix_cache()

            target_layer, hidden_size = _discover_layers(llm)

            vec = [500.0] * hidden_size
            llm.llm.collective_rpc(
                "set_steering_vectors",
                kwargs={"vectors": {_HP: {target_layer: vec}}},
            )

            steered_tokens, steered_logprob = _gen_tokens_and_cumulative_logprob(
                llm, prompt, sampling
            )

            assert steered_tokens != baseline_tokens or not math.isclose(
                steered_logprob,
                baseline_logprob,
                rel_tol=0.0,
                abs_tol=1e-6,
            ), "Steering should change output or logprob for StableLM"

            llm.llm.collective_rpc("clear_steering_vectors")
            assert llm.llm.reset_prefix_cache()

            restored_tokens, restored_logprob = _gen_tokens_and_cumulative_logprob(
                llm, prompt, sampling
            )

            assert restored_tokens == baseline_tokens
            assert math.isclose(
                restored_logprob,
                baseline_logprob,
                rel_tol=0.0,
                abs_tol=1e-6,
            )


@pytest.mark.parametrize(
    ("model", "hf_overrides", "extra_runner_kwargs"), PHASE3_DISCOVERY_CASES
)
def test_steering_layers_discovered_apertus_minimax_family(
    vllm_runner,
    monkeypatch,
    model: str,
    hf_overrides: dict | None,
    extra_runner_kwargs: dict | None,
) -> None:
    """Apertus and MiniMax norm-variant decoder families should expose
    steerable layers."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        with vllm_runner(
            model, **_runner_kwargs(hf_overrides, extra_runner_kwargs)
        ) as llm:
            target_layer, hidden_size = _discover_layers(llm)

            assert target_layer >= 0
            assert hidden_size > 0


@pytest.mark.parametrize(
    ("model", "hf_overrides", "extra_runner_kwargs", "vector_scale"),
    PHASE3_GENERATION_CASES,
)
def test_steering_changes_output_apertus_minimax_family(
    vllm_runner,
    monkeypatch,
    model: str,
    hf_overrides: dict | None,
    extra_runner_kwargs: dict | None,
    vector_scale: float,
) -> None:
    """Apertus and MiniMax decoder families should respond to steering."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        prompt = "What does the fox say? " * 32
        sampling = SamplingParams(max_tokens=10, temperature=0.0, logprobs=5)

        runner_kwargs = _runner_kwargs(hf_overrides, extra_runner_kwargs)
        runner_kwargs["enable_prefix_caching"] = True

        with vllm_runner(model, **runner_kwargs) as llm:
            baseline_tokens, baseline_logprob = _gen_tokens_and_cumulative_logprob(
                llm, prompt, sampling
            )

            assert llm.llm.reset_prefix_cache()

            target_layer, hidden_size = _discover_layers(llm)

            vec = [vector_scale] * hidden_size
            llm.llm.collective_rpc(
                "set_steering_vectors",
                kwargs={"vectors": {_HP: {target_layer: vec}}},
            )

            steered_tokens, steered_logprob = _gen_tokens_and_cumulative_logprob(
                llm, prompt, sampling
            )

            assert steered_tokens != baseline_tokens or not math.isclose(
                steered_logprob,
                baseline_logprob,
                rel_tol=0.0,
                abs_tol=1e-6,
            ), f"Steering should change output or logprob for {model}"

            llm.llm.collective_rpc("clear_steering_vectors")
            assert llm.llm.reset_prefix_cache()

            restored_tokens, restored_logprob = _gen_tokens_and_cumulative_logprob(
                llm, prompt, sampling
            )

            assert restored_tokens == baseline_tokens, (
                f"Clearing steering should restore baseline for {model}"
            )
            assert math.isclose(
                restored_logprob,
                baseline_logprob,
                rel_tol=0.0,
                abs_tol=1e-6,
            ), f"Clearing steering should restore baseline logprob for {model}"


def test_step3p5_steering_changes_output_real_weights(vllm_runner, monkeypatch) -> None:
    """Step-3.5 needs a real checkpoint for stable validation."""
    model = "stepfun-ai/Step-3.5-Flash"
    model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
    model_info.check_available_online(on_fail="skip")
    if not torch.cuda.is_available():
        pytest.skip("Step-3.5 real-weights steering test requires CUDA.")
    total_memory_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if total_memory_gib < 40:
        pytest.skip(
            "Step-3.5 real-weights steering test requires a GPU with at least "
            "40 GiB of memory."
        )

    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        prompt = "What does the fox say? " * 32
        sampling = SamplingParams(max_tokens=10, temperature=0.0, logprobs=5)

        with vllm_runner(
            model,
            max_model_len=256,
            enable_steering=True,
            max_steering_configs=4,
            enable_prefix_caching=True,
            enforce_eager=True,
            trust_remote_code=True,
        ) as llm:
            baseline_tokens, baseline_logprob = _gen_tokens_and_cumulative_logprob(
                llm, prompt, sampling
            )

            assert llm.llm.reset_prefix_cache()

            target_layer, hidden_size = _discover_layers(llm)

            vec = [500.0] * hidden_size
            llm.llm.collective_rpc(
                "set_steering_vectors",
                kwargs={"vectors": {_HP: {target_layer: vec}}},
            )

            steered_tokens, steered_logprob = _gen_tokens_and_cumulative_logprob(
                llm, prompt, sampling
            )

            assert steered_tokens != baseline_tokens or not math.isclose(
                steered_logprob,
                baseline_logprob,
                rel_tol=0.0,
                abs_tol=1e-6,
            ), "Steering should change output or logprob for Step-3.5"

            llm.llm.collective_rpc("clear_steering_vectors")
            assert llm.llm.reset_prefix_cache()

            restored_tokens, restored_logprob = _gen_tokens_and_cumulative_logprob(
                llm, prompt, sampling
            )

            assert restored_tokens == baseline_tokens
            assert math.isclose(
                restored_logprob,
                baseline_logprob,
                rel_tol=0.0,
                abs_tol=1e-6,
            )


@pytest.mark.parametrize(
    ("model", "hf_overrides", "extra_runner_kwargs"), PHASE4_DISCOVERY_CASES
)
def test_steering_layers_discovered_moe_family(
    vllm_runner,
    monkeypatch,
    model: str,
    hf_overrides: dict | None,
    extra_runner_kwargs: dict | None,
) -> None:
    """MoE families (mixtral, ernie45-moe, granitemoe, granitemoe-shared)
    should expose steerable layers."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        with vllm_runner(
            model, **_runner_kwargs(hf_overrides, extra_runner_kwargs)
        ) as llm:
            target_layer, hidden_size = _discover_layers(llm)

            assert target_layer >= 0
            assert hidden_size > 0


@pytest.mark.parametrize(
    ("model", "hf_overrides", "extra_runner_kwargs", "vector_scale"),
    PHASE4_GENERATION_CASES,
)
def test_steering_changes_output_moe_family(
    vllm_runner,
    monkeypatch,
    model: str,
    hf_overrides: dict | None,
    extra_runner_kwargs: dict | None,
    vector_scale: float,
) -> None:
    """MoE families should respond to steering."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        prompt = "What does the fox say? " * 32
        sampling = SamplingParams(max_tokens=10, temperature=0.0, logprobs=5)

        runner_kwargs = _runner_kwargs(hf_overrides, extra_runner_kwargs)
        runner_kwargs["enable_prefix_caching"] = True

        with vllm_runner(model, **runner_kwargs) as llm:
            baseline_tokens, baseline_logprob = _gen_tokens_and_cumulative_logprob(
                llm, prompt, sampling
            )

            assert llm.llm.reset_prefix_cache()

            target_layer, hidden_size = _discover_layers(llm)

            vec = [vector_scale] * hidden_size
            llm.llm.collective_rpc(
                "set_steering_vectors",
                kwargs={"vectors": {_HP: {target_layer: vec}}},
            )

            steered_tokens, steered_logprob = _gen_tokens_and_cumulative_logprob(
                llm, prompt, sampling
            )

            assert steered_tokens != baseline_tokens or not math.isclose(
                steered_logprob,
                baseline_logprob,
                rel_tol=0.0,
                abs_tol=1e-6,
            ), f"Steering should change output or logprob for {model}"

            llm.llm.collective_rpc("clear_steering_vectors")
            assert llm.llm.reset_prefix_cache()

            restored_tokens, restored_logprob = _gen_tokens_and_cumulative_logprob(
                llm, prompt, sampling
            )

            assert restored_tokens == baseline_tokens, (
                f"Clearing steering should restore baseline for {model}"
            )
            assert math.isclose(
                restored_logprob,
                baseline_logprob,
                rel_tol=0.0,
                abs_tol=1e-6,
            ), f"Clearing steering should restore baseline logprob for {model}"


def test_mixtral_steering_changes_output_real_weights(vllm_runner, monkeypatch) -> None:
    """Mixtral gets a small real-weights check via the tiny HF variant."""
    model = "TitanML/tiny-mixtral"
    model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
    model_info.check_available_online(on_fail="skip")
    _skip_if_cuda_unavailable_or_below(16, "Mixtral")

    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        prompt = "What does the fox say? " * 32
        sampling = SamplingParams(max_tokens=10, temperature=0.0, logprobs=5)

        with vllm_runner(
            model,
            max_model_len=256,
            enable_steering=True,
            max_steering_configs=4,
            enable_prefix_caching=True,
            enforce_eager=True,
        ) as llm:
            baseline_tokens, baseline_logprob = _gen_tokens_and_cumulative_logprob(
                llm, prompt, sampling
            )

            assert llm.llm.reset_prefix_cache()

            target_layer, hidden_size = _discover_layers(llm)

            vec = [500.0] * hidden_size
            llm.llm.collective_rpc(
                "set_steering_vectors",
                kwargs={"vectors": {_HP: {target_layer: vec}}},
            )

            steered_tokens, steered_logprob = _gen_tokens_and_cumulative_logprob(
                llm, prompt, sampling
            )

            assert steered_tokens != baseline_tokens or not math.isclose(
                steered_logprob,
                baseline_logprob,
                rel_tol=0.0,
                abs_tol=1e-6,
            ), "Steering should change output or logprob for Mixtral"

            llm.llm.collective_rpc("clear_steering_vectors")
            assert llm.llm.reset_prefix_cache()

            restored_tokens, restored_logprob = _gen_tokens_and_cumulative_logprob(
                llm, prompt, sampling
            )

            assert restored_tokens == baseline_tokens
            assert math.isclose(
                restored_logprob,
                baseline_logprob,
                rel_tol=0.0,
                abs_tol=1e-6,
            )


def test_deepseek_v2_steering_changes_output_real_weights(
    vllm_runner, monkeypatch
) -> None:
    """DeepSeek-V2 real-weights coverage is hardware-gated."""
    model = "deepseek-ai/DeepSeek-V2-Lite-Chat"
    model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
    model_info.check_available_online(on_fail="skip")
    _skip_if_cuda_unavailable_or_below(40, "DeepSeek-V2-Lite")

    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        prompt = "What does the fox say? " * 32
        sampling = SamplingParams(max_tokens=10, temperature=0.0, logprobs=5)

        with vllm_runner(
            model,
            max_model_len=256,
            enable_steering=True,
            max_steering_configs=4,
            enable_prefix_caching=True,
            enforce_eager=True,
            trust_remote_code=True,
        ) as llm:
            baseline_tokens, baseline_logprob = _gen_tokens_and_cumulative_logprob(
                llm, prompt, sampling
            )

            assert llm.llm.reset_prefix_cache()

            target_layer, hidden_size = _discover_layers(llm)

            vec = [500.0] * hidden_size
            llm.llm.collective_rpc(
                "set_steering_vectors",
                kwargs={"vectors": {_HP: {target_layer: vec}}},
            )

            steered_tokens, steered_logprob = _gen_tokens_and_cumulative_logprob(
                llm, prompt, sampling
            )

            assert steered_tokens != baseline_tokens or not math.isclose(
                steered_logprob,
                baseline_logprob,
                rel_tol=0.0,
                abs_tol=1e-6,
            ), "Steering should change output or logprob for DeepSeek-V2-Lite"

            llm.llm.collective_rpc("clear_steering_vectors")
            assert llm.llm.reset_prefix_cache()

            restored_tokens, restored_logprob = _gen_tokens_and_cumulative_logprob(
                llm, prompt, sampling
            )

            assert restored_tokens == baseline_tokens
            assert math.isclose(
                restored_logprob,
                baseline_logprob,
                rel_tol=0.0,
                abs_tol=1e-6,
            )


def test_phimoe_steering_changes_output_real_weights(vllm_runner, monkeypatch) -> None:
    """PhiMoE gets real-weights coverage because the dummy path is not stable."""
    model = "microsoft/Phi-3.5-MoE-instruct"
    model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
    model_info.check_available_online(on_fail="skip")
    _skip_if_cuda_unavailable_or_below(24, "Phi-3.5-MoE")

    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        prompt = "What does the fox say? " * 32
        sampling = SamplingParams(max_tokens=10, temperature=0.0, logprobs=5)

        with vllm_runner(
            model,
            max_model_len=256,
            enable_steering=True,
            max_steering_configs=4,
            enable_prefix_caching=True,
            enforce_eager=True,
        ) as llm:
            baseline_tokens, baseline_logprob = _gen_tokens_and_cumulative_logprob(
                llm, prompt, sampling
            )

            assert llm.llm.reset_prefix_cache()

            target_layer, hidden_size = _discover_layers(llm)

            vec = [500.0] * hidden_size
            llm.llm.collective_rpc(
                "set_steering_vectors",
                kwargs={"vectors": {_HP: {target_layer: vec}}},
            )

            steered_tokens, steered_logprob = _gen_tokens_and_cumulative_logprob(
                llm, prompt, sampling
            )

            assert steered_tokens != baseline_tokens or not math.isclose(
                steered_logprob,
                baseline_logprob,
                rel_tol=0.0,
                abs_tol=1e-6,
            ), "Steering should change output or logprob for PhiMoE"

            llm.llm.collective_rpc("clear_steering_vectors")
            assert llm.llm.reset_prefix_cache()

            restored_tokens, restored_logprob = _gen_tokens_and_cumulative_logprob(
                llm, prompt, sampling
            )

            assert restored_tokens == baseline_tokens
            assert math.isclose(
                restored_logprob,
                baseline_logprob,
                rel_tol=0.0,
                abs_tol=1e-6,
            )


def test_glm4_moe_steering_changes_output_real_weights(
    vllm_runner, monkeypatch
) -> None:
    """GLM-4.5 is covered with real weights and a hardware gate."""
    model = "zai-org/GLM-4.5"
    model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
    model_info.check_available_online(on_fail="skip")
    _skip_if_cuda_unavailable_or_below(40, "GLM-4.5")

    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        prompt = "What does the fox say? " * 32
        sampling = SamplingParams(max_tokens=10, temperature=0.0, logprobs=5)

        with vllm_runner(
            model,
            max_model_len=256,
            enable_steering=True,
            max_steering_configs=4,
            enable_prefix_caching=True,
            enforce_eager=True,
            trust_remote_code=True,
        ) as llm:
            baseline_tokens, baseline_logprob = _gen_tokens_and_cumulative_logprob(
                llm, prompt, sampling
            )

            assert llm.llm.reset_prefix_cache()

            target_layer, hidden_size = _discover_layers(llm)

            vec = [500.0] * hidden_size
            llm.llm.collective_rpc(
                "set_steering_vectors",
                kwargs={"vectors": {_HP: {target_layer: vec}}},
            )

            steered_tokens, steered_logprob = _gen_tokens_and_cumulative_logprob(
                llm, prompt, sampling
            )

            assert steered_tokens != baseline_tokens or not math.isclose(
                steered_logprob,
                baseline_logprob,
                rel_tol=0.0,
                abs_tol=1e-6,
            ), "Steering should change output or logprob for GLM-4.5"

            llm.llm.collective_rpc("clear_steering_vectors")
            assert llm.llm.reset_prefix_cache()

            restored_tokens, restored_logprob = _gen_tokens_and_cumulative_logprob(
                llm, prompt, sampling
            )

            assert restored_tokens == baseline_tokens
            assert math.isclose(
                restored_logprob,
                baseline_logprob,
                rel_tol=0.0,
                abs_tol=1e-6,
            )


def test_exaone_moe_steering_changes_output_real_weights(
    vllm_runner, monkeypatch
) -> None:
    """EXAONE MoE is too large for the dummy harness, so it is hardware-gated."""
    model = "LGAI-EXAONE/K-EXAONE-236B-A23B"
    model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
    model_info.check_available_online(on_fail="skip")
    _skip_if_cuda_unavailable_or_below(80, "EXAONE MoE")

    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        prompt = "What does the fox say? " * 32
        sampling = SamplingParams(max_tokens=10, temperature=0.0, logprobs=5)

        with vllm_runner(
            model,
            max_model_len=256,
            enable_steering=True,
            max_steering_configs=4,
            enable_prefix_caching=True,
            enforce_eager=True,
            trust_remote_code=True,
        ) as llm:
            baseline_tokens, baseline_logprob = _gen_tokens_and_cumulative_logprob(
                llm, prompt, sampling
            )

            assert llm.llm.reset_prefix_cache()

            target_layer, hidden_size = _discover_layers(llm)

            vec = [500.0] * hidden_size
            llm.llm.collective_rpc(
                "set_steering_vectors",
                kwargs={"vectors": {_HP: {target_layer: vec}}},
            )

            steered_tokens, steered_logprob = _gen_tokens_and_cumulative_logprob(
                llm, prompt, sampling
            )

            assert steered_tokens != baseline_tokens or not math.isclose(
                steered_logprob,
                baseline_logprob,
                rel_tol=0.0,
                abs_tol=1e-6,
            ), "Steering should change output or logprob for EXAONE MoE"

            llm.llm.collective_rpc("clear_steering_vectors")
            assert llm.llm.reset_prefix_cache()

            restored_tokens, restored_logprob = _gen_tokens_and_cumulative_logprob(
                llm, prompt, sampling
            )

            assert restored_tokens == baseline_tokens
            assert math.isclose(
                restored_logprob,
                baseline_logprob,
                rel_tol=0.0,
                abs_tol=1e-6,
            )


@pytest.mark.parametrize(
    ("model", "hf_overrides", "extra_runner_kwargs"), PHASE5_DISCOVERY_CASES
)
def test_steering_layers_discovered_legacy_decoder_family(
    vllm_runner,
    monkeypatch,
    model: str,
    hf_overrides: dict | None,
    extra_runner_kwargs: dict | None,
) -> None:
    """Legacy decoder families (exaone, granite, opt, gpt-neox, phi,
    persimmon, starcoder2, jais2, olmo, olmo2, falcon) should expose
    steerable layers."""
    try:
        model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
        model_info.check_available_online(on_fail="skip")
    except ValueError:
        pass

    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        try:
            with vllm_runner(
                model, **_runner_kwargs(hf_overrides, extra_runner_kwargs)
            ) as llm:
                target_layer, hidden_size = _discover_layers(llm)
        except Exception as exc:
            _maybe_skip_model_access_failure(exc, model)
            raise

        assert target_layer >= 0
        assert hidden_size > 0


@pytest.mark.parametrize(
    ("model", "hf_overrides", "extra_runner_kwargs", "vector_scale"),
    PHASE5_GENERATION_CASES,
)
def test_steering_changes_output_legacy_decoder_family(
    vllm_runner,
    monkeypatch,
    model: str,
    hf_overrides: dict | None,
    extra_runner_kwargs: dict | None,
    vector_scale: float,
) -> None:
    """Legacy decoder families should respond to steering."""
    try:
        model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
        model_info.check_available_online(on_fail="skip")
    except ValueError:
        pass

    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        prompt = "What does the fox say? " * 32
        sampling = SamplingParams(max_tokens=10, temperature=0.0, logprobs=5)

        runner_kwargs = _runner_kwargs(hf_overrides, extra_runner_kwargs)
        runner_kwargs["enable_prefix_caching"] = True

        try:
            with vllm_runner(model, **runner_kwargs) as llm:
                baseline_tokens, baseline_logprob = _gen_tokens_and_cumulative_logprob(
                    llm, prompt, sampling
                )

                assert llm.llm.reset_prefix_cache()

                target_layer, hidden_size = _discover_layers(llm)

                vec = [vector_scale] * hidden_size
                llm.llm.collective_rpc(
                    "set_steering_vectors",
                    kwargs={"vectors": {_HP: {target_layer: vec}}},
                )

                steered_tokens, steered_logprob = _gen_tokens_and_cumulative_logprob(
                    llm, prompt, sampling
                )

                assert steered_tokens != baseline_tokens or not math.isclose(
                    steered_logprob,
                    baseline_logprob,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                ), f"Steering should change output or logprob for {model}"

                llm.llm.collective_rpc("clear_steering_vectors")
                assert llm.llm.reset_prefix_cache()

                restored_tokens, restored_logprob = _gen_tokens_and_cumulative_logprob(
                    llm, prompt, sampling
                )
        except Exception as exc:
            _maybe_skip_model_access_failure(exc, model)
            raise

        assert restored_tokens == baseline_tokens, (
            f"Clearing steering should restore baseline for {model}"
        )
        assert math.isclose(
            restored_logprob,
            baseline_logprob,
            rel_tol=0.0,
            abs_tol=1e-6,
        ), f"Clearing steering should restore baseline logprob for {model}"


@pytest.mark.parametrize(
    ("model", "hf_overrides", "extra_runner_kwargs"), PHASE6_DISCOVERY_CASES
)
def test_steering_layers_discovered_new_decoder_family(
    vllm_runner,
    monkeypatch,
    model: str,
    hf_overrides: dict | None,
    extra_runner_kwargs: dict | None,
) -> None:
    """Newly-wired decoder families (arctic, flex_olmo, hunyuan_v1,
    mimo_v2_flash, param2moe) should expose steerable layers."""
    try:
        model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
        model_info.check_available_online(on_fail="skip")
    except ValueError:
        pass

    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        try:
            with vllm_runner(
                model, **_runner_kwargs(hf_overrides, extra_runner_kwargs)
            ) as llm:
                target_layer, hidden_size = _discover_layers(llm)
        except Exception as exc:
            _maybe_skip_model_access_failure(exc, model)
            raise

        assert target_layer >= 0
        assert hidden_size > 0


@pytest.mark.parametrize(
    ("model", "hf_overrides", "extra_runner_kwargs", "vector_scale"),
    PHASE6_GENERATION_CASES,
)
def test_steering_changes_output_new_decoder_family(
    vllm_runner,
    monkeypatch,
    model: str,
    hf_overrides: dict | None,
    extra_runner_kwargs: dict | None,
    vector_scale: float,
) -> None:
    """Newly-wired decoder families should respond to steering."""
    try:
        model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
        model_info.check_available_online(on_fail="skip")
    except ValueError:
        pass

    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        prompt = "What does the fox say? " * 32
        sampling = SamplingParams(max_tokens=10, temperature=0.0, logprobs=5)

        runner_kwargs = _runner_kwargs(hf_overrides, extra_runner_kwargs)
        runner_kwargs["enable_prefix_caching"] = True

        try:
            with vllm_runner(model, **runner_kwargs) as llm:
                baseline_tokens, baseline_logprob = _gen_tokens_and_cumulative_logprob(
                    llm, prompt, sampling
                )

                assert llm.llm.reset_prefix_cache()

                target_layer, hidden_size = _discover_layers(llm)

                vec = [vector_scale] * hidden_size
                llm.llm.collective_rpc(
                    "set_steering_vectors",
                    kwargs={"vectors": {_HP: {target_layer: vec}}},
                )

                steered_tokens, steered_logprob = _gen_tokens_and_cumulative_logprob(
                    llm, prompt, sampling
                )

                assert steered_tokens != baseline_tokens or not math.isclose(
                    steered_logprob,
                    baseline_logprob,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                ), f"Steering should change output or logprob for {model}"

                llm.llm.collective_rpc("clear_steering_vectors")
                assert llm.llm.reset_prefix_cache()

                restored_tokens, restored_logprob = _gen_tokens_and_cumulative_logprob(
                    llm, prompt, sampling
                )
        except Exception as exc:
            _maybe_skip_model_access_failure(exc, model)
            raise

        assert restored_tokens == baseline_tokens, (
            f"Clearing steering should restore baseline for {model}"
        )
        assert math.isclose(
            restored_logprob,
            baseline_logprob,
            rel_tol=0.0,
            abs_tol=1e-6,
        ), f"Clearing steering should restore baseline logprob for {model}"


# ---------------------------------------------------------------------------
# Existing tests (updated for kwargs-based collective_rpc API)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", [MODEL])
def test_steering_changes_output(vllm_runner, monkeypatch, model: str) -> None:
    """Verify that non-zero steering vectors change model output
    and that clearing them restores the original behaviour."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        prompt = "What does the fox say? " * 32
        sampling = SamplingParams(max_tokens=10, temperature=0.0)

        with vllm_runner(
            model,
            load_format="dummy",
            max_model_len=512,
            enable_prefix_caching=True,
        ) as llm:
            # 1. Baseline (zero steering buffers)
            baseline_tokens = _gen_tokens(llm, prompt, sampling)

            # Clear the clean prompt from APC so the steered run has to
            # prefill and write its own KV entries before the unsteered
            # replay.
            assert llm.llm.reset_prefix_cache()

            # 2. Discover hidden_size and pick a middle layer.
            target_layer, hidden_size = _discover_layers(llm)

            # 3. Set steering via WorkerBase (same path as HTTP API).
            #    With dummy (random) weights the magnitude must be large
            #    enough to overcome noise in the logit space.
            vec = [500.0] * hidden_size
            llm.llm.collective_rpc(
                "set_steering_vectors",
                kwargs={"vectors": {_HP: {target_layer: vec}}},
            )

            steered_tokens = _gen_tokens(llm, prompt, sampling)

            assert steered_tokens != baseline_tokens, (
                "Non-zero steering should change model output"
            )

            # 4. Clear steering and verify output matches baseline
            llm.llm.collective_rpc("clear_steering_vectors")
            assert llm.llm.reset_prefix_cache()

            restored_tokens = _gen_tokens(llm, prompt, sampling)

            assert restored_tokens == baseline_tokens, (
                "Clearing steering should restore original output"
            )


@pytest.mark.parametrize("model", [MODEL])
def test_per_request_steering_via_sampling_params(
    vllm_runner, monkeypatch, model: str
) -> None:
    """Verify that per-request steering_vectors in SamplingParams
    changes output and that different steering produces different results."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        prompt = "What does the fox say? " * 32
        sampling = SamplingParams(max_tokens=10, temperature=0.0)

        with vllm_runner(
            model,
            load_format="dummy",
            max_model_len=512,
            enable_prefix_caching=True,
            enable_steering=True,
            max_steering_configs=4,
        ) as llm:
            # 1. Baseline (no steering)
            baseline_tokens = _gen_tokens(llm, prompt, sampling)

            assert llm.llm.reset_prefix_cache()

            # 2. Discover steerable layers
            target_layer, hidden_size = _discover_layers(llm)

            # 3. Generate with per-request steering via SamplingParams
            steered_sampling = SamplingParams(
                max_tokens=10,
                temperature=0.0,
                steering_vectors={
                    _HP: {target_layer: [500.0] * hidden_size},
                },
            )

            steered_tokens = _gen_tokens(llm, prompt, steered_sampling)

            assert steered_tokens != baseline_tokens, (
                "Per-request steering should change model output"
            )

            # 4. Verify baseline is unchanged (no contamination)
            assert llm.llm.reset_prefix_cache()
            restored_tokens = _gen_tokens(llm, prompt, sampling)

            assert restored_tokens == baseline_tokens, (
                "Per-request steering should not contaminate other requests"
            )


@pytest.mark.parametrize("model", [MODEL])
def test_per_request_steering_concurrent_with_cuda_graphs(
    vllm_runner, monkeypatch, model: str
) -> None:
    """Test that different per-request steering configs in the same batch
    produce different outputs, and that CUDA graph replays correctly pick
    up updated steering buffers between steps.

    This sends multiple requests simultaneously so they land in the same
    batch during decode, exercising the request-indexed gather with
    CUDA graphs active.
    """
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        prompt = "What does the fox say? " * 32

        with vllm_runner(
            model,
            load_format="dummy",
            max_model_len=512,
            enable_prefix_caching=False,
            enable_steering=True,
            max_steering_configs=4,
        ) as llm:
            # 1. Discover steerable layers
            target_layer, hidden_size = _discover_layers(llm)

            # 2. Create three requests: no steering, positive, negative
            no_steer = SamplingParams(max_tokens=10, temperature=0.0)
            steer_pos = SamplingParams(
                max_tokens=10,
                temperature=0.0,
                steering_vectors={
                    _HP: {target_layer: [500.0] * hidden_size},
                },
            )
            steer_neg = SamplingParams(
                max_tokens=10,
                temperature=0.0,
                steering_vectors={
                    _HP: {target_layer: [-500.0] * hidden_size},
                },
            )

            # 3. Send all three simultaneously so they batch together
            outputs = llm.llm.generate(
                [prompt, prompt, prompt],
                [no_steer, steer_pos, steer_neg],
            )

            tokens_none = list(outputs[0].outputs[0].token_ids)
            tokens_pos = list(outputs[1].outputs[0].token_ids)
            tokens_neg = list(outputs[2].outputs[0].token_ids)

            # Positive and negative steering should produce different output
            assert tokens_pos != tokens_neg, (
                "Opposite steering vectors should produce different outputs"
            )

            # At least one steered output should differ from unsteered
            assert tokens_pos != tokens_none or tokens_neg != tokens_none, (
                "At least one steered request should differ from unsteered"
            )

            # 4. Run again without steering to verify CUDA graph replays
            #    pick up updated (cleared) buffer contents
            outputs2 = llm.llm.generate(
                [prompt, prompt],
                [no_steer, no_steer],
            )

            tokens_none2 = list(outputs2[0].outputs[0].token_ids)
            tokens_none3 = list(outputs2[1].outputs[0].token_ids)

            # Unsteered should be consistent across runs
            assert tokens_none2 == tokens_none, (
                "Unsteered output should be deterministic across runs"
            )
            assert tokens_none3 == tokens_none, (
                "Both unsteered requests should match baseline"
            )


# ---------------------------------------------------------------------------
# Prefill steering tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", [MODEL])
def test_prefill_steering_changes_output(vllm_runner, monkeypatch, model: str) -> None:
    """Verify that prefill-specific steering via SamplingParams changes
    output and does not contaminate subsequent unsteered requests."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        prompt = "What does the fox say? " * 32
        sampling = SamplingParams(max_tokens=10, temperature=0.0)

        with vllm_runner(
            model,
            load_format="dummy",
            max_model_len=512,
            enable_prefix_caching=True,
            enable_steering=True,
            max_steering_configs=4,
        ) as llm:
            # 1. Baseline (no steering)
            baseline_tokens = _gen_tokens(llm, prompt, sampling)

            assert llm.llm.reset_prefix_cache()

            # 2. Discover steerable layers
            target_layer, hidden_size = _discover_layers(llm)

            # 3. Generate with prefill-specific steering
            prefill_steered = SamplingParams(
                max_tokens=10,
                temperature=0.0,
                prefill_steering_vectors={
                    _HP: {target_layer: [500.0] * hidden_size},
                },
            )

            steered_tokens = _gen_tokens(llm, prompt, prefill_steered)

            assert steered_tokens != baseline_tokens, (
                "Prefill steering should change model output"
            )

            # 4. Reset and verify no contamination
            assert llm.llm.reset_prefix_cache()
            restored_tokens = _gen_tokens(llm, prompt, sampling)

            assert restored_tokens == baseline_tokens, (
                "Prefill steering should not contaminate subsequent requests"
            )


@pytest.mark.parametrize("model", [MODEL])
def test_decode_only_steering_via_new_field(
    vllm_runner, monkeypatch, model: str
) -> None:
    """Verify that decode_steering_vectors (the new field) changes output
    compared to an unsteered baseline."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        prompt = "What does the fox say? " * 32
        sampling = SamplingParams(max_tokens=10, temperature=0.0)

        with vllm_runner(
            model,
            load_format="dummy",
            max_model_len=512,
            enable_prefix_caching=True,
            enable_steering=True,
            max_steering_configs=4,
        ) as llm:
            # 1. Baseline (no steering)
            baseline_tokens = _gen_tokens(llm, prompt, sampling)

            assert llm.llm.reset_prefix_cache()

            # 2. Discover steerable layers
            target_layer, hidden_size = _discover_layers(llm)

            # 3. Generate with decode-specific steering
            decode_steered = SamplingParams(
                max_tokens=10,
                temperature=0.0,
                decode_steering_vectors={
                    _HP: {target_layer: [500.0] * hidden_size},
                },
            )

            steered_tokens = _gen_tokens(llm, prompt, decode_steered)

            assert steered_tokens != baseline_tokens, (
                "Decode-only steering should change model output"
            )


@pytest.mark.parametrize("model", [MODEL])
def test_prefill_and_decode_different_steering(
    vllm_runner, monkeypatch, model: str
) -> None:
    """Verify that using different vectors for prefill vs decode produces
    different output than using the same vector for both phases."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        prompt = "What does the fox say? " * 32

        with vllm_runner(
            model,
            load_format="dummy",
            max_model_len=512,
            enable_prefix_caching=True,
            enable_steering=True,
            max_steering_configs=4,
        ) as llm:
            target_layer, hidden_size = _discover_layers(llm)

            # 1. Same vector for both phases via base steering_vectors
            both_same = SamplingParams(
                max_tokens=10,
                temperature=0.0,
                steering_vectors={
                    _HP: {target_layer: [500.0] * hidden_size},
                },
            )

            result_both = _gen_tokens(llm, prompt, both_same)

            assert llm.llm.reset_prefix_cache()

            # 2. Different vectors for prefill vs decode
            split = SamplingParams(
                max_tokens=10,
                temperature=0.0,
                prefill_steering_vectors={
                    _HP: {target_layer: [500.0] * hidden_size},
                },
                decode_steering_vectors={
                    _HP: {target_layer: [-500.0] * hidden_size},
                },
            )

            result_split = _gen_tokens(llm, prompt, split)

            assert result_both != result_split, (
                "Different prefill vs decode steering should produce "
                "different output than uniform steering"
            )


@pytest.mark.parametrize("model", [MODEL])
def test_additive_composition(vllm_runner, monkeypatch, model: str) -> None:
    """Verify the three-tier additive model works correctly.

    To test additive composition we must ensure BOTH prefill and decode
    effective vectors match between the two approaches.  We use:

    Approach A:  prefill_steering=P, steering_vectors=X, decode_steering=Y
        → prefill_effective = P + X,  decode_effective = X + Y

    Approach B:  prefill_steering=P+X, decode_steering=X+Y
        → prefill_effective = P + X,  decode_effective = X + Y

    Both should produce identical output.
    """
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        prompt = "What does the fox say? " * 32

        with vllm_runner(
            model,
            load_format="dummy",
            max_model_len=512,
            enable_prefix_caching=True,
            enable_steering=True,
            max_steering_configs=4,
        ) as llm:
            target_layer, hidden_size = _discover_layers(llm)

            H = hidden_size

            # Approach A: three-tier additive
            # prefill = P(200) + base(300) = 500
            # decode  = base(300) + D(200) = 500
            approach_a = SamplingParams(
                max_tokens=10,
                temperature=0.0,
                prefill_steering_vectors={
                    _HP: {target_layer: [200.0] * H},
                },
                steering_vectors={
                    _HP: {target_layer: [300.0] * H},
                },
                decode_steering_vectors={
                    _HP: {target_layer: [200.0] * H},
                },
            )

            result_a = _gen_tokens(llm, prompt, approach_a)

            assert llm.llm.reset_prefix_cache()

            # Approach B: phase-specific only (no base), same totals
            # prefill = 500, decode = 500
            approach_b = SamplingParams(
                max_tokens=10,
                temperature=0.0,
                prefill_steering_vectors={
                    _HP: {target_layer: [500.0] * H},
                },
                decode_steering_vectors={
                    _HP: {target_layer: [500.0] * H},
                },
            )

            result_b = _gen_tokens(llm, prompt, approach_b)

            assert result_a == result_b, (
                "Three-tier additive (P=200 + base=300 + D=200) should "
                "produce same output as direct (P=500, D=500)"
            )


@pytest.mark.parametrize("model", [MODEL])
def test_prefix_cache_respects_prefill_steering(
    vllm_runner, monkeypatch, model: str
) -> None:
    """Verify that prefix cache correctly separates different prefill
    steering: same prompt with different prefill steering should produce
    different outputs, but same prompt with same prefill steering should
    hit the cache and produce identical output."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        prompt = "What does the fox say? " * 32
        sampling_no_steer = SamplingParams(max_tokens=10, temperature=0.0)

        with vllm_runner(
            model,
            load_format="dummy",
            max_model_len=512,
            enable_prefix_caching=True,
            enable_steering=True,
            max_steering_configs=4,
        ) as llm:
            target_layer, hidden_size = _discover_layers(llm)

            # 1. Request A: with prefill steering
            steered_sampling = SamplingParams(
                max_tokens=10,
                temperature=0.0,
                prefill_steering_vectors={
                    _HP: {target_layer: [500.0] * hidden_size},
                },
            )

            tokens_a = _gen_tokens(llm, prompt, steered_sampling)

            # 2. Request B: same prompt, NO prefill steering
            tokens_b = _gen_tokens(llm, prompt, sampling_no_steer)

            # Different prefill steering means different KV cache
            assert tokens_a != tokens_b, (
                "Different prefill steering should produce different output "
                "even with prefix caching enabled"
            )

            # 3. Request C: same prompt, same prefill steering as A
            #    Should hit prefix cache and produce identical output
            tokens_c = _gen_tokens(llm, prompt, steered_sampling)

            assert tokens_c == tokens_a, (
                "Same prefill steering should hit prefix cache and produce "
                "identical output"
            )


@pytest.mark.parametrize("model", [MODEL])
def test_co_located_scale(vllm_runner, monkeypatch, model: str) -> None:
    """Verify that the co-located scale format produces the same result
    as a pre-scaled bare vector: [500]*H should equal {"vector": [250]*H,
    "scale": 2.0}."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        prompt = "What does the fox say? " * 32

        with vllm_runner(
            model,
            load_format="dummy",
            max_model_len=512,
            enable_prefix_caching=True,
            enable_steering=True,
            max_steering_configs=4,
        ) as llm:
            target_layer, hidden_size = _discover_layers(llm)

            H = hidden_size

            # 1. Bare vector at magnitude 500
            bare = SamplingParams(
                max_tokens=10,
                temperature=0.0,
                steering_vectors={
                    _HP: {target_layer: [500.0] * H},
                },
            )

            result_bare = _gen_tokens(llm, prompt, bare)

            assert llm.llm.reset_prefix_cache()

            # 2. Co-located scale: vector=250, scale=2.0 => effective 500
            scaled = SamplingParams(
                max_tokens=10,
                temperature=0.0,
                steering_vectors={
                    _HP: {
                        target_layer: {
                            "vector": [250.0] * H,
                            "scale": 2.0,
                        },
                    },
                },
            )

            result_scaled = _gen_tokens(llm, prompt, scaled)

            assert result_bare == result_scaled, (
                "Co-located scale (250 * 2.0) should produce same output "
                "as bare vector (500)"
            )


@pytest.mark.parametrize("model", [MODEL])
def test_global_prefill_steering_via_worker_api(
    vllm_runner, monkeypatch, model: str
) -> None:
    """Verify global three-tier steering via the worker API: setting
    prefill-specific global vectors changes output, and clearing them
    restores the baseline."""
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        prompt = "What does the fox say? " * 32
        sampling = SamplingParams(max_tokens=10, temperature=0.0)

        with vllm_runner(
            model,
            load_format="dummy",
            max_model_len=512,
            enable_prefix_caching=True,
        ) as llm:
            # 1. Baseline (no global steering)
            baseline_tokens = _gen_tokens(llm, prompt, sampling)

            assert llm.llm.reset_prefix_cache()

            # 2. Discover steerable layers
            target_layer, hidden_size = _discover_layers(llm)

            # 3. Set global prefill-specific vectors via worker API
            vec = [500.0] * hidden_size
            llm.llm.collective_rpc(
                "set_steering_vectors",
                kwargs={
                    "prefill_vectors": {_HP: {target_layer: vec}},
                },
            )

            steered_tokens = _gen_tokens(llm, prompt, sampling)

            assert steered_tokens != baseline_tokens, (
                "Global prefill steering should change model output"
            )

            # 4. Clear and verify restoration
            llm.llm.collective_rpc("clear_steering_vectors")
            assert llm.llm.reset_prefix_cache()

            restored_tokens = _gen_tokens(llm, prompt, sampling)

            assert restored_tokens == baseline_tokens, (
                "Clearing global prefill steering should restore baseline"
            )
