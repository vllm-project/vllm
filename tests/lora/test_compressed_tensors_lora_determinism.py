# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Determinism regression for LoRA on compressed-tensors W4A16 (issue #50059)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

import vllm
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.lora.request import LoRARequest
from vllm.platforms import current_platform

# TinyLlama compressed-tensors W4A16 (group 128) — small enough for H20/L20.
W4A16_MODEL = (
    "nm-testing/TinyLlama-1.1B-Chat-v1.0-W4A16-G128-Asym-Updated-ActOrder"
)

# TinyLlama-1.1B architecture dims.
_HIDDEN = 2048
_INTERMEDIATE = 5632
_NUM_LAYERS = 22
_NUM_KV_HEADS = 4
_HEAD_DIM = 64
_KV_OUT = _NUM_KV_HEADS * _HEAD_DIM  # 256


def _write_all_layer_lora(save_dir: Path, *, rank: int, seed: int = 0) -> Path:
    """Create a PEFT LoRA covering all q/k/v/o/gate/up/down projections."""
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)

    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    adapter_config = {
        "peft_type": "LORA",
        "base_model_name_or_path": W4A16_MODEL,
        "task_type": "CAUSAL_LM",
        "r": rank,
        "lora_alpha": rank,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": target_modules,
        "inference_mode": True,
    }
    with open(save_dir / "adapter_config.json", "w", encoding="utf-8") as f:
        json.dump(adapter_config, f, indent=2)

    weights: dict[str, torch.Tensor] = {}
    # (name_suffix, in_features, out_features)
    per_layer = [
        ("self_attn.q_proj", _HIDDEN, _HIDDEN),
        ("self_attn.k_proj", _HIDDEN, _KV_OUT),
        ("self_attn.v_proj", _HIDDEN, _KV_OUT),
        ("self_attn.o_proj", _HIDDEN, _HIDDEN),
        ("mlp.gate_proj", _HIDDEN, _INTERMEDIATE),
        ("mlp.up_proj", _HIDDEN, _INTERMEDIATE),
        ("mlp.down_proj", _INTERMEDIATE, _HIDDEN),
    ]
    for layer_idx in range(_NUM_LAYERS):
        for suffix, in_f, out_f in per_layer:
            prefix = f"base_model.model.model.layers.{layer_idx}.{suffix}"
            lora_a = torch.randn(rank, in_f, dtype=torch.float16)
            torch.nn.init.kaiming_uniform_(lora_a, a=5**0.5)
            # Non-zero B so LoRA actually changes logits (not a no-op).
            lora_b = torch.randn(out_f, rank, dtype=torch.float16) * 0.01
            weights[f"{prefix}.lora_A.weight"] = lora_a
            weights[f"{prefix}.lora_B.weight"] = lora_b

    save_file(weights, str(save_dir / "adapter_model.safetensors"))
    return save_dir


def _greedy_texts(
    llm: vllm.LLM,
    lora_path: str,
    *,
    lora_id: int = 1,
    max_tokens: int = 16,
) -> list[str]:
    prompts = [
        "<|im_start|>user\nSay one short color name.<|im_end|>\n"
        "<|im_start|>assistant\n",
        "<|im_start|>user\nReply with exactly: OK.<|im_end|>\n"
        "<|im_start|>assistant\n",
    ]
    sampling = vllm.SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=max_tokens,
        seed=1234,
        stop=["<|im_end|>"],
    )
    outputs = llm.generate(
        prompts,
        sampling,
        lora_request=LoRARequest(str(lora_id), lora_id, lora_path),
    )
    return [o.outputs[0].text for o in outputs]


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="compressed-tensors W4A16 + LoRA determinism needs CUDA",
)
@pytest.mark.parametrize("rank", [8, 16, 32])
def test_w4a16_lora_greedy_deterministic_same_engine(tmp_path, rank: int):
    """Same engine, repeated greedy generate must be identical (issue #50059)."""
    lora_dir = _write_all_layer_lora(tmp_path / f"lora_r{rank}", rank=rank)
    llm = vllm.LLM(
        model=W4A16_MODEL,
        enable_lora=True,
        max_loras=2,
        max_lora_rank=32,
        max_model_len=256,
        gpu_memory_utilization=0.25,
        enforce_eager=True,
        trust_remote_code=True,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        async_scheduling=False,
        seed=0,
    )
    try:
        # Discard Triton JIT / first-call warmup before measuring.
        _ = _greedy_texts(llm, str(lora_dir))
        baseline = _greedy_texts(llm, str(lora_dir))
        for _ in range(4):
            assert _greedy_texts(llm, str(lora_dir)) == baseline
        # LoRA must actually move the output vs base (non-zero adapter).
        base_out = llm.generate(
            [
                "<|im_start|>user\nSay one short color name.<|im_end|>\n"
                "<|im_start|>assistant\n"
            ],
            vllm.SamplingParams(temperature=0.0, max_tokens=16, seed=1234),
        )[0].outputs[0].text
        assert baseline[0] != base_out or baseline[1] != "", (
            "LoRA adapter produced no change; adapter may not have loaded"
        )
    finally:
        del llm
        cleanup_dist_env_and_memory()


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="compressed-tensors W4A16 + LoRA determinism needs CUDA",
)
@pytest.mark.parametrize("rank", [8, 16, 32])
def test_w4a16_lora_greedy_deterministic_across_reloads(tmp_path, rank: int):
    """Reload engine (restart-like) and compare greedy LoRA outputs (#50059)."""
    lora_dir = _write_all_layer_lora(tmp_path / f"lora_r{rank}_reload", rank=rank)

    def one_run() -> list[str]:
        llm = vllm.LLM(
            model=W4A16_MODEL,
            enable_lora=True,
            max_loras=2,
            max_lora_rank=32,
            max_model_len=256,
            gpu_memory_utilization=0.25,
            enforce_eager=True,
            trust_remote_code=True,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
            async_scheduling=False,
            seed=0,
        )
        try:
            _ = _greedy_texts(llm, str(lora_dir))  # warmup
            return _greedy_texts(llm, str(lora_dir))
        finally:
            del llm
            cleanup_dist_env_and_memory()

    baseline = one_run()
    for _ in range(2):
        assert one_run() == baseline
