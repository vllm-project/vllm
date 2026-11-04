# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test: Cohere2 (CohereForCausalLM) must serve with LoRA enabled."""

import json
import os

import torch
from safetensors.torch import save_file

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

MODEL_PATH = "CohereLabs/tiny-aya-global"
HIDDEN_SIZE = 2048
Q_SIZE = 16 * 128
KV_SIZE = 4 * 128


def _zero_delta_lora(save_dir: str, rank: int = 8) -> None:
    """LoRA on layer-0 q/v projections with B = 0, so outputs must equal base."""
    weights = {}
    for module, out_size in (("q_proj", Q_SIZE), ("v_proj", KV_SIZE)):
        prefix = f"base_model.model.model.layers.0.self_attn.{module}"
        weights[f"{prefix}.lora_A.weight"] = torch.randn(rank, HIDDEN_SIZE) * 0.01
        weights[f"{prefix}.lora_B.weight"] = torch.zeros(out_size, rank)
    os.makedirs(save_dir, exist_ok=True)
    save_file(weights, os.path.join(save_dir, "adapter_model.safetensors"))
    with open(os.path.join(save_dir, "adapter_config.json"), "w") as f:
        json.dump(
            {
                "peft_type": "LORA",
                "base_model_name_or_path": MODEL_PATH,
                "task_type": "CAUSAL_LM",
                "r": rank,
                "lora_alpha": rank,
                "lora_dropout": 0.0,
                "bias": "none",
                "target_modules": ["q_proj", "v_proj"],
            },
            f,
        )


def test_cohere2_serves_with_lora(tmp_path):
    lora_dir = str(tmp_path / "zero_lora")
    _zero_delta_lora(lora_dir)

    llm = LLM(
        MODEL_PATH,
        enable_lora=True,
        max_lora_rank=8,
        max_loras=1,
        max_model_len=1024,
        gpu_memory_utilization=0.6,
    )
    params = SamplingParams(temperature=0, max_tokens=16)
    prompts = ["The capital of France is", "def fibonacci(n):"]

    base = [o.outputs[0].text for o in llm.generate(prompts, params)]
    lora = [
        o.outputs[0].text
        for o in llm.generate(
            prompts, params, lora_request=LoRARequest("zero", 1, lora_dir)
        )
    ]
    assert base == lora
