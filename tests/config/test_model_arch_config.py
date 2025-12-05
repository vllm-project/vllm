# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from pathlib import Path

import torch

from vllm.config import ModelConfig


def test_model_arch_config():
    trust_remote_code_models = [
        "nvidia/Llama-3_3-Nemotron-Super-49B-v1",
        "XiaomiMiMo/MiMo-7B-RL",
        # Not available online right now
        # "FreedomIntelligence/openPangu-Ultra-MoE-718B-V1.1",
        "meituan-longcat/LongCat-Flash-Chat",
    ]
    models_to_test = [
        "Zyphra/Zamba2-7B-instruct",
        "mosaicml/mpt-7b",
        "databricks/dbrx-instruct",
        "tiiuae/falcon-7b",
        "tiiuae/falcon-40b",
        "luccafong/deepseek_mtp_main_random",
        "luccafong/deepseek_mtp_draft_random",
        "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "tiny-random/qwen3-next-moe",
        "zai-org/GLM-4.5",
        "baidu/ERNIE-4.5-21B-A3B-PT",
        # Select some models using base convertor for testing
        "lmsys/gpt-oss-20b-bf16",
        "deepseek-ai/DeepSeek-V3.2-Exp",
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    ] + trust_remote_code_models

    groundtruth_path = Path(__file__).parent / "model_arch_groundtruth.json"
    with open(groundtruth_path) as f:
        model_arch_groundtruth = json.load(f)

    for model in models_to_test:
        print(f"testing {model=}")
        model_config = ModelConfig(
            model, trust_remote_code=model in trust_remote_code_models
        )

        model_arch_config = model_config.model_arch_config
        expected = model_arch_groundtruth[model]
        assert model_arch_config.architectures == expected["architectures"]
        assert model_arch_config.model_type == expected["model_type"]
        assert model_arch_config.text_model_type == expected["text_model_type"]
        assert model_arch_config.hidden_size == expected["hidden_size"]
        assert (
            model_arch_config.total_num_hidden_layers
            == expected["total_num_hidden_layers"]
        )
        assert (
            model_arch_config.total_num_attention_heads
            == expected["total_num_attention_heads"]
        )
        assert model_arch_config.head_size == expected["head_size"]
        assert model_arch_config.vocab_size == expected["vocab_size"]
        assert model_arch_config.total_num_kv_heads == expected["total_num_kv_heads"]
        assert model_arch_config.num_experts == expected["num_experts"]
        assert model_arch_config.is_deepseek_mla == expected["is_deepseek_mla"]
        assert model_arch_config.is_multimodal_model == expected["is_multimodal_model"]

        dtype = model_arch_config.torch_dtype
        assert str(dtype) == expected["dtype"]
        if expected["dtype_original_type"] == "str":
            assert isinstance(dtype, str)
        elif expected["dtype_original_type"] == "torch.dtype":
            assert isinstance(dtype, torch.dtype)
        else:
            raise ValueError(f"Unknown dtype_original_type: {expected['dtype']}")

        # Test that model_config methods return expected values
        assert model_config.architectures == expected["architectures"]
        assert model_config.get_vocab_size() == expected["vocab_size"]
        assert model_config.get_hidden_size() == expected["hidden_size"]
        assert model_config.get_head_size() == expected["head_size"]
        assert model_config.get_total_num_kv_heads() == expected["total_num_kv_heads"]
        assert model_config.get_num_experts() == expected["num_experts"]
        assert (
            model_config.get_total_num_hidden_layers()
            == expected["total_num_hidden_layers"]
        )
