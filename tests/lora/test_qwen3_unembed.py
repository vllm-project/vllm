# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for Qwen3 unembed LoRA support.

This test creates synthetic LoRA weights that include lm_head (output embedding)
to verify that Qwen3 properly supports LoRA on the unembed/lm_head layer.
"""

import json
import os
import tempfile

import numpy as np
import torch
from safetensors.torch import save_file

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

MODEL_PATH = "Qwen/Qwen3-0.6B"
HIDDEN_SIZE = 1024
VOCAB_SIZE = 151936


def create_qwen3_lora_with_lm_head(save_dir: str, rank: int = 8) -> None:
    """Create synthetic Qwen3 LoRA weights with lm_head."""
    lora_weights = {}
    for module in ["q_proj", "v_proj"]:
        lora_A = torch.from_numpy(
            np.random.randn(rank, HIDDEN_SIZE).astype(np.float16) * 0.01
        )
        lora_B = torch.zeros(HIDDEN_SIZE, rank, dtype=torch.float16)
        key_prefix = f"base_model.model.model.layers.0.self_attn.{module}"
        lora_weights[f"{key_prefix}.lora_A.weight"] = lora_A
        lora_weights[f"{key_prefix}.lora_B.weight"] = lora_B

    # lm_head LoRA weights
    lora_weights["base_model.model.lm_head.lora_A.weight"] = torch.from_numpy(
        np.random.randn(rank, HIDDEN_SIZE).astype(np.float16) * 0.01
    )
    lora_weights["base_model.model.lm_head.lora_B.weight"] = torch.zeros(
        VOCAB_SIZE, rank, dtype=torch.float16
    )

    adapter_config = {
        "peft_type": "LORA",
        "base_model_name_or_path": MODEL_PATH,
        "task_type": "CAUSAL_LM",
        "inference_mode": True,
        "r": rank,
        "lora_alpha": rank * 2,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": ["q_proj", "v_proj", "lm_head"],
    }

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "adapter_config.json"), "w") as f:
        json.dump(adapter_config, f)
    save_file(lora_weights, os.path.join(save_dir, "adapter_model.safetensors"))


def test_qwen3_unembed_lora():
    """Verify Qwen3 can load and generate with LoRA adapters with lm_head."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize engine first (before creating torch tensors)
        llm = LLM(
            model=MODEL_PATH,
            enable_lora=True,
            max_loras=4,
            max_lora_rank=8,
            max_model_len=128,
            gpu_memory_utilization=0.8,
            enforce_eager=True,
        )

        # Create LoRA weights after engine init
        create_qwen3_lora_with_lm_head(tmpdir, rank=8)

        lora_request = LoRARequest("lm_head_lora", 1, tmpdir)
        llm.llm_engine.add_lora(lora_request)

        assert 1 in llm.llm_engine.list_loras(), "lm_head LoRA should be loaded"

        # Test generation
        sampling_params = SamplingParams(temperature=0, max_tokens=32)
        prompts = ["Hello, my name is"]

        # Generate with base model (no LoRA)
        base_outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        assert len(base_outputs) == 1
        assert len(base_outputs[0].outputs[0].text) > 0

        # Generate with lm_head LoRA
        lora_outputs = llm.generate(
            prompts, sampling_params, lora_request=lora_request, use_tqdm=False
        )
        assert len(lora_outputs) == 1
        assert len(lora_outputs[0].outputs[0].text) > 0
