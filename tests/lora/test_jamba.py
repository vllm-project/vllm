# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LoRA on Jamba's Mamba-1 projections, in batches that mix prefill and decode."""

import json

import pytest
import torch
from safetensors.torch import save_file

from tests.models.utils import check_logprobs_close
from vllm.lora.request import LoRARequest

MODEL_PATH = "ai21labs/Jamba-tiny-dev"
SHORT_PROMPT = "The capital of France is"
# Prefilled in chunks of max_num_batched_tokens while SHORT_PROMPT decodes.
LONG_PROMPT = " ".join(f"Item {i} is a number." for i in range(300))


@pytest.fixture(scope="module")
def mamba_lora_files(tmp_path_factory) -> str:
    """A seeded random LoRA on the projections of every Mamba layer."""
    hidden, inner, dt_rank, d_state, rank = 512, 1024, 32, 16, 8
    shapes = {
        "in_proj": (hidden, 2 * inner),
        "x_proj": (inner, dt_rank + 2 * d_state),
        "dt_proj": (dt_rank, inner),
        "out_proj": (inner, hidden),
    }
    generator = torch.Generator().manual_seed(0)
    tensors = {}
    for layer in range(16):
        if layer % 8 == 4:  # attention layers
            continue
        for module, (fan_in, fan_out) in shapes.items():
            prefix = f"base_model.model.model.layers.{layer}.mamba.{module}"
            lora_a = torch.randn(rank, fan_in, generator=generator)
            lora_b = torch.randn(fan_out, rank, generator=generator)
            tensors[f"{prefix}.lora_A.weight"] = (0.05 * lora_a).bfloat16()
            tensors[f"{prefix}.lora_B.weight"] = (0.05 * lora_b).bfloat16()
    path = tmp_path_factory.mktemp("jamba_mamba_lora")
    save_file(tensors, path / "adapter_model.safetensors")
    config = {
        "peft_type": "LORA",
        "r": rank,
        "lora_alpha": 2 * rank,
        "target_modules": list(shapes),
        "task_type": "CAUSAL_LM",
    }
    (path / "adapter_config.json").write_text(json.dumps(config))
    return str(path)


def test_mamba_lora_mixed_prefill_decode(vllm_runner, mamba_lora_files):
    """A request decoding next to another request's prefill chunks gets the same
    output as alone: the Mamba LoRA layers must see every row of the batch."""
    lora = LoRARequest("mamba", 1, mamba_lora_files)
    with vllm_runner(
        MODEL_PATH,
        enable_lora=True,
        max_lora_rank=8,
        max_model_len=4096,
        enable_chunked_prefill=True,
        max_num_batched_tokens=256,
        gpu_memory_utilization=0.5,
    ) as vllm_model:
        alone = vllm_model.generate_greedy_logprobs(
            [SHORT_PROMPT], 32, 5, lora_request=lora
        )
        mixed = vllm_model.generate_greedy_logprobs(
            [SHORT_PROMPT, LONG_PROMPT], 32, 5, lora_request=[lora, lora]
        )

    check_logprobs_close(
        outputs_0_lst=alone,
        outputs_1_lst=mixed[:1],
        name_0="alone",
        name_1="mixed",
    )
