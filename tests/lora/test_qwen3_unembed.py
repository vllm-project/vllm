# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Qwen3 unembed LoRA support.

This test creates synthetic LoRA weights that include lm_head (output embedding)
to verify that Qwen3 properly supports LoRA on the unembed/lm_head layer.
"""

import json
import os
import tempfile

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
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


def _create_lm_head_lora(save_dir: str, seed: int, rank: int = 8) -> None:
    """Like create_qwen3_lora_with_lm_head, but with a non-zero lm_head delta."""
    create_qwen3_lora_with_lm_head(save_dir, rank=rank)
    path = os.path.join(save_dir, "adapter_model.safetensors")
    weights = load_file(path)
    gen = torch.Generator().manual_seed(seed)
    weights["base_model.model.lm_head.lora_B.weight"] = torch.randn(
        VOCAB_SIZE, rank, generator=gen
    ).to(torch.float16)
    save_file(weights, path)


def _prompt_lps(output, token_ids: list[int], last_n: int) -> torch.Tensor:
    pairs = zip(output.prompt_logprobs[-last_n:], token_ids[-last_n:])
    return torch.tensor([d[t].logprob for d, t in pairs])


@pytest.mark.parametrize("use_v2_model_runner", [True, False])
def test_qwen3_unembed_lora_prompt_logprobs(monkeypatch, use_v2_model_runner):
    """prompt_logprobs must include the lm_head LoRA delta of each request's
    own adapter, and only that one (regression test for #51594)."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", str(int(use_v2_model_runner)))
    num_gen = 16
    with tempfile.TemporaryDirectory() as tmpdir:
        llm = LLM(
            model=MODEL_PATH,
            enable_lora=True,
            max_loras=2,
            max_lora_rank=8,
            max_model_len=256,
            gpu_memory_utilization=0.5,
            enforce_eager=True,
        )
        loras = []
        for i in range(2):
            path = os.path.join(tmpdir, f"lora{i}")
            _create_lm_head_lora(path, seed=i)
            loras.append(LoRARequest(f"lm_head_lora{i}", i + 1, path))

        prompts = ["Hello, my name is", "The capital of France is"]
        gen_params = SamplingParams(temperature=0, max_tokens=num_gen, logprobs=0)
        gen_outputs = llm.generate(
            prompts, gen_params, lora_request=loras, use_tqdm=False
        )

        # Teacher-force prompt + generated tokens. Mix both adapters and no
        # adapter in one batch so a mapping mix-up between requests shows up.
        seqs, expected = [], []
        for out in gen_outputs:
            gen_ids = list(out.outputs[0].token_ids)
            seqs.append(list(out.prompt_token_ids) + gen_ids)
            expected.append(
                torch.tensor(
                    [lp[t].logprob for lp, t in zip(out.outputs[0].logprobs, gen_ids)]
                )
            )
        score_params = SamplingParams(max_tokens=1, prompt_logprobs=0)
        token_prompts = [TokensPrompt(prompt_token_ids=s) for s in seqs]
        scored = llm.generate(
            token_prompts + token_prompts,
            score_params,
            lora_request=loras + [None, None],
            use_tqdm=False,
        )
        base_alone = llm.generate(token_prompts, score_params, use_tqdm=False)

        for i, seq in enumerate(seqs):
            lora_lps = _prompt_lps(scored[i], seq, num_gen)
            base_lps = _prompt_lps(scored[2 + i], seq, num_gen)
            # The adapter changes the scores by a lot ...
            assert (lora_lps - base_lps).abs().mean() > 0.5
            # ... prompt logprobs agree with the generation logprobs ...
            torch.testing.assert_close(lora_lps, expected[i], atol=0.1, rtol=0)
            # ... and no-adapter requests do not pick up a neighbour's delta.
            torch.testing.assert_close(
                base_lps, _prompt_lps(base_alone[i], seq, num_gen), atol=0.1, rtol=0
            )

        if use_v2_model_runner:
            # Fixed-ID prompt scoring uses the same logits path.
            id_params = [
                SamplingParams(
                    max_tokens=1,
                    prompt_logprob_token_ids=[[t] for t in seq[1:]],
                )
                for seq in seqs
            ]
            id_scored = llm.generate(
                token_prompts, id_params, lora_request=loras, use_tqdm=False
            )
            for out, exp in zip(id_scored, expected):
                lps = torch.from_numpy(out.prompt_token_id_logprobs[-num_gen:, 0])
                torch.testing.assert_close(lps.float(), exp, atol=0.1, rtol=0)
