# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU-smoke Recirculation adapters with tiny local dummy models.

This benchmark downloads no model weights or tokenizers. It generates a small
four-layer Hugging Face configuration in a temporary directory, asks vLLM to
initialize random weights, and runs token-ID generation through the real engine.
"""

import argparse
import json
import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")

import torch
from transformers import (
    DeepseekV3Config,
    Glm4MoeLiteConfig,
    GptOssConfig,
    Llama4TextConfig,
    LlamaConfig,
    MixtralConfig,
    PretrainedConfig,
    Qwen3Config,
)
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

from vllm.transformers_utils.configs.minimax_m3 import MiniMaxM3TextConfig
from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5TextConfig
from vllm.transformers_utils.configs.qwen3_5_moe import Qwen3_5MoeTextConfig
from vllm.transformers_utils.configs.qwen3_next import Qwen3NextConfig
from vllm.transformers_utils.configs.step3p5 import Step3p5Config

ConfigBuilder = Callable[[], PretrainedConfig]


def deepseek_v3_config() -> PretrainedConfig:
    return DeepseekV3Config(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=4,
        kv_lora_rank=512,
        q_lora_rank=None,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        max_position_embeddings=128,
        tie_word_embeddings=False,
    )


def qwen3_config() -> PretrainedConfig:
    return Qwen3Config(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=128,
        tie_word_embeddings=False,
    )


def mixtral_config() -> PretrainedConfig:
    return MixtralConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=128,
        num_local_experts=4,
        num_experts_per_tok=2,
        tie_word_embeddings=False,
    )


def llama4_config() -> PretrainedConfig:
    return Llama4TextConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=128,
        intermediate_size_mlp=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=128,
        num_local_experts=4,
        num_experts_per_tok=1,
        attention_chunk_size=64,
        attn_temperature_tuning=False,
        tie_word_embeddings=False,
    )


def gemma4_config() -> PretrainedConfig:
    return Gemma4TextConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        global_head_dim=32,
        num_global_key_value_heads=2,
        max_position_embeddings=128,
        sliding_window=64,
        layer_types=["sliding_attention"] * 3 + ["full_attention"],
        hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=256,
        enable_moe_block=True,
        num_experts=4,
        top_k_experts=2,
        moe_intermediate_size=128,
        tie_word_embeddings=False,
    )


def gpt_oss_config() -> PretrainedConfig:
    return GptOssConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=128,
        sliding_window=64,
        num_local_experts=4,
        num_experts_per_tok=2,
        tie_word_embeddings=False,
    )


def qwen3_5_config() -> PretrainedConfig:
    return Qwen3_5TextConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=128,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        tie_word_embeddings=False,
    )


def qwen3_5_moe_config() -> PretrainedConfig:
    return Qwen3_5MoeTextConfig(
        vocab_size=256,
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=128,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        tie_word_embeddings=False,
    )


def qwen3_next_config() -> PretrainedConfig:
    return Qwen3NextConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=128,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        tie_word_embeddings=False,
    )


def minimax_m3_config() -> PretrainedConfig:
    return MiniMaxM3TextConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=64,
        dense_intermediate_size=256,
        shared_intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=128,
        max_position_embeddings=128,
        rotary_dim=64,
        num_local_experts=4,
        num_experts_per_tok=2,
        moe_layer_freq=[0, 1, 1, 1],
        sparse_attention_config={},
        tie_word_embeddings=False,
    )


def glm4_moe_lite_config() -> PretrainedConfig:
    config = Glm4MoeLiteConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=4,
        kv_lora_rank=512,
        q_lora_rank=None,
        qk_rope_head_dim=64,
        qk_nope_head_dim=192,
        v_head_dim=256,
        num_experts_per_tok=2,
        max_position_embeddings=128,
        mlp_layer_types=["dense", "sparse", "sparse", "sparse"],
        tie_word_embeddings=False,
    )
    config.first_k_dense_replace = 1
    config.moe_layer_freq = 1
    return config


def mimo_v2_config() -> PretrainedConfig:
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=128,
        hidden_act="silu",
        tie_word_embeddings=False,
    )
    config.layernorm_epsilon = 1e-6
    config.hybrid_layer_pattern = [0, 0, 0, 0]
    config.attention_bias = False
    config.moe_layer_freq = [0, 1, 1, 1]
    config.n_routed_experts = 4
    config.num_experts_per_tok = 2
    config.moe_intermediate_size = 64
    config.norm_topk_prob = True
    config.n_group = 1
    config.topk_group = 1
    return config


def step3p5_config() -> PretrainedConfig:
    return Step3p5Config(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_attention_groups=2,
        head_dim=32,
        max_seq_len=128,
        max_position_embeddings=128,
        moe_intermediate_size=64,
        moe_num_experts=4,
        moe_top_k=2,
        share_expert_dim=64,
        tie_word_embeddings=False,
    )


FAMILIES: dict[str, tuple[str, ConfigBuilder, bool]] = {
    "deepseek-v3": ("DeepseekV3ForCausalLM", deepseek_v3_config, False),
    "gemma4": ("Gemma4ForCausalLM", gemma4_config, True),
    "glm4-moe-lite": (
        "Glm4MoeLiteForCausalLM",
        glm4_moe_lite_config,
        False,
    ),
    "gpt-oss": ("GptOssForCausalLM", gpt_oss_config, False),
    "llama4": ("Llama4ForCausalLM", llama4_config, True),
    "minimax-m3": (
        "MiniMaxM3SparseForCausalLM",
        minimax_m3_config,
        False,
    ),
    "mimo-v2": ("MiMoV2ForCausalLM", mimo_v2_config, True),
    "mixtral": ("MixtralForCausalLM", mixtral_config, True),
    "qwen3": ("Qwen3ForCausalLM", qwen3_config, True),
    "qwen3.5": ("Qwen3_5ForCausalLM", qwen3_5_config, False),
    "qwen3.5-moe": (
        "Qwen3_5MoeForCausalLM",
        qwen3_5_moe_config,
        False,
    ),
    "qwen3-next": ("Qwen3NextForCausalLM", qwen3_next_config, False),
    "step3.5": ("Step3p5ForCausalLM", step3p5_config, True),
}


def run(args: argparse.Namespace) -> dict[str, Any]:
    from vllm import LLM, SamplingParams

    architecture, build_config, supports_wavefront = FAMILIES[args.family]
    wavefront = args.mode == "wavefront"
    if wavefront and not supports_wavefront:
        raise ValueError(f"{args.family} only supports serial Recirculation")

    config = build_config()
    config.architectures = [architecture]
    config.recirculation_config = {
        "source_layer": 2,
        "destination_layer": 1,
        "alpha": 0.15,
        "wavefront": wavefront,
    }

    torch.accelerator.reset_peak_memory_stats()
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=f"recirculation-{args.family}-") as path:
        config.save_pretrained(path)
        llm = LLM(
            model=path,
            load_format="dummy",
            skip_tokenizer_init=True,
            dtype="bfloat16",
            max_model_len=64,
            max_num_seqs=1,
            max_num_batched_tokens=64,
            long_prefill_token_threshold=1,
            enable_prefix_caching=False,
            enforce_eager=not args.compile,
            gpu_memory_utilization=0.25,
            kv_cache_memory_bytes=args.kv_cache_mib * 1024 * 1024,
        )
        try:
            output = llm.generate(
                [{"prompt_token_ids": list(range(1, 9))}],
                SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True),
                use_tqdm=False,
            )[0]
            token_ids = list(output.outputs[0].token_ids)
        finally:
            llm.llm_engine.engine_core.shutdown()

    return {
        "family": args.family,
        "architecture": architecture,
        "mode": args.mode,
        "compiled": args.compile,
        "load_format": "dummy",
        "weights_downloaded": False,
        "num_layers": 4,
        "hidden_size": 128,
        "kv_cache_mib": args.kv_cache_mib,
        "peak_torch_gpu_mib": torch.accelerator.max_memory_allocated() / (1024 * 1024),
        "elapsed_s": time.perf_counter() - started,
        "output_token_ids": token_ids,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=sorted(FAMILIES), required=True)
    parser.add_argument("--mode", choices=("serial", "wavefront"), required=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--kv-cache-mib", type=int, default=128)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args)
    payload = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
