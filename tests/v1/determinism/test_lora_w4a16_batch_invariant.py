# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Greedy LoRA determinism on a compressed-tensors W4A16 model (#50059).

The default Punica shrink config uses split_k > 1, which reduces over K with
``tl.atomic_add`` and is not run-to-run deterministic. With
``VLLM_BATCH_INVARIANT=1`` (enabled for every test in this directory by
``conftest.py``) the shrink uses fixed K partials and the W4A16 linear layers
use the dequantize + batch-invariant matmul path, so repeated greedy requests
with an all-layer LoRA must produce identical tokens.
"""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from utils import skip_if_not_cuda

import vllm.lora.ops.triton_ops.lora_shrink_op as shrink_op
import vllm.lora.ops.triton_ops.utils as lora_utils
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.lora.request import LoRARequest

# TinyLlama-1.1B, compressed-tensors W4A16, group size 128, no act-order.
MODEL = "nm-testing/tinyllama-oneshot-w4a16-group128-v2"

# TinyLlama-1.1B dims.
HIDDEN = 2048
INTERMEDIATE = 5632
NUM_LAYERS = 22
KV_OUT = 4 * 64  # num_key_value_heads * head_dim

PROMPTS = [
    "The capital of France is",
    "Write one short sentence about the ocean:",
]


@pytest.fixture(autouse=True)
def batch_invariant_lora(monkeypatch: pytest.MonkeyPatch):
    # The LoRA op modules read VLLM_BATCH_INVARIANT at import time; patch them
    # too in case they were imported before conftest set the env var.
    monkeypatch.setattr(lora_utils, "is_batch_invariant", True)
    monkeypatch.setattr(shrink_op, "is_batch_invariant", True)
    lora_utils.get_lora_op_configs.cache_clear()
    yield
    lora_utils.get_lora_op_configs.cache_clear()


def _write_all_layer_lora(save_dir: Path, rank: int, seed: int = 0) -> str:
    """Write a synthetic PEFT LoRA on every q/k/v/o/gate/up/down projection."""
    save_dir.mkdir(parents=True, exist_ok=True)
    per_layer = [
        ("self_attn.q_proj", HIDDEN, HIDDEN),
        ("self_attn.k_proj", HIDDEN, KV_OUT),
        ("self_attn.v_proj", HIDDEN, KV_OUT),
        ("self_attn.o_proj", HIDDEN, HIDDEN),
        ("mlp.gate_proj", HIDDEN, INTERMEDIATE),
        ("mlp.up_proj", HIDDEN, INTERMEDIATE),
        ("mlp.down_proj", INTERMEDIATE, HIDDEN),
    ]
    adapter_config = {
        "peft_type": "LORA",
        "base_model_name_or_path": MODEL,
        "task_type": "CAUSAL_LM",
        "r": rank,
        "lora_alpha": rank,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": [name.split(".")[-1] for name, _, _ in per_layer],
        "inference_mode": True,
    }
    (save_dir / "adapter_config.json").write_text(json.dumps(adapter_config))

    gen = torch.Generator().manual_seed(seed)
    weights: dict[str, torch.Tensor] = {}
    for layer_idx in range(NUM_LAYERS):
        for name, in_f, out_f in per_layer:
            prefix = f"base_model.model.model.layers.{layer_idx}.{name}"
            weights[f"{prefix}.lora_A.weight"] = (
                torch.randn(rank, in_f, generator=gen) / in_f**0.5
            ).half()
            # Non-zero B so the adapter actually changes the logits.
            weights[f"{prefix}.lora_B.weight"] = (
                torch.randn(out_f, rank, generator=gen) * 0.01
            ).half()
    save_file(weights, str(save_dir / "adapter_model.safetensors"))
    return str(save_dir)


def _make_llm() -> LLM:
    return LLM(
        model=MODEL,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=32,
        max_model_len=256,
        max_num_seqs=len(PROMPTS),
        gpu_memory_utilization=0.3,
        enforce_eager=True,
        enable_prefix_caching=False,
        attention_config={"backend": "TRITON_ATTN"},
        seed=0,
    )


def _greedy_token_ids(llm: LLM, lora_path: str | None) -> list[list[int]]:
    sampling = SamplingParams(temperature=0.0, max_tokens=16)
    lora_request = LoRARequest("lora", 1, lora_path) if lora_path else None
    outputs = llm.generate(PROMPTS, sampling, lora_request=lora_request)
    return [list(o.outputs[0].token_ids) for o in outputs]


@skip_if_not_cuda
@pytest.mark.timeout(1000)
@pytest.mark.parametrize("rank", [8, 16, 32])
def test_w4a16_lora_greedy_deterministic_same_engine(tmp_path, rank):
    lora_path = _write_all_layer_lora(tmp_path / f"lora_r{rank}", rank)
    llm = _make_llm()
    try:
        base = _greedy_token_ids(llm, None)
        baseline = _greedy_token_ids(llm, lora_path)
        assert baseline != base, "LoRA adapter did not change the output"
        for _ in range(4):
            assert _greedy_token_ids(llm, lora_path) == baseline
    finally:
        del llm
        cleanup_dist_env_and_memory()


@skip_if_not_cuda
@pytest.mark.timeout(1000)
@pytest.mark.parametrize("rank", [8, 16, 32])
def test_w4a16_lora_greedy_deterministic_across_restarts(tmp_path, rank):
    lora_path = _write_all_layer_lora(tmp_path / f"lora_r{rank}", rank)

    def one_run() -> list[list[int]]:
        llm = _make_llm()
        try:
            return _greedy_token_ids(llm, lora_path)
        finally:
            del llm
            cleanup_dist_env_and_memory()

    baseline = one_run()
    assert one_run() == baseline
