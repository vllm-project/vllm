# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Integration tests for NomicBertModel with LoRA adapters.

Tests verify that NomicBertModel can load and use LoRA adapters for embedding
tasks, producing distinct outputs per adapter.
"""

import json
import os

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

import vllm
from vllm.lora.request import LoRARequest

from ..utils import create_new_process_for_each_test

NOMIC_MODEL = "nomic-ai/nomic-embed-text-v1"

# NomicBertModel config: hidden_size=768, num_heads=12, num_layers=12
# LoRA targets use HF naming (pre weights-mapper):
#   encoder.layers.{i}.attn.Wqkv      (768 -> 2304, fused)
#   encoder.layers.{i}.attn.out_proj   (768 -> 768)
LORA_RANK = 8
LORA_ALPHA = 16
NUM_LAYERS = 12
HIDDEN_SIZE = 768
QKV_SIZE = HIDDEN_SIZE * 3  # 2304


def _create_random_nomic_lora(save_dir: str, seed: int = 0):
    """Generate a synthetic PEFT LoRA adapter for NomicBertModel."""
    rng = torch.Generator().manual_seed(seed)
    lora_weights = {}

    for i in range(NUM_LAYERS):
        prefix = f"base_model.model.encoder.layers.{i}.attn"

        # Wqkv: in=768, out=2304 (fused Q+K+V)
        lora_weights[f"{prefix}.Wqkv.lora_A.weight"] = torch.randn(
            LORA_RANK, HIDDEN_SIZE, generator=rng, dtype=torch.float16
        )
        lora_weights[f"{prefix}.Wqkv.lora_B.weight"] = (
            torch.randn(QKV_SIZE, LORA_RANK, generator=rng, dtype=torch.float16) * 0.01
        )

        # out_proj: in=768, out=768
        lora_weights[f"{prefix}.out_proj.lora_A.weight"] = torch.randn(
            LORA_RANK, HIDDEN_SIZE, generator=rng, dtype=torch.float16
        )
        lora_weights[f"{prefix}.out_proj.lora_B.weight"] = (
            torch.randn(HIDDEN_SIZE, LORA_RANK, generator=rng, dtype=torch.float16)
            * 0.01
        )

    adapter_config = {
        "peft_type": "LORA",
        "auto_mapping": None,
        "base_model_name_or_path": NOMIC_MODEL,
        "revision": None,
        "task_type": None,
        "inference_mode": True,
        "r": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": 0.0,
        "fan_in_fan_out": False,
        "bias": "none",
        "modules_to_save": None,
        "init_lora_weights": True,
        "layers_to_transform": None,
        "layers_pattern": None,
        "target_modules": ["Wqkv", "out_proj"],
        "exclude_modules": None,
        "use_rslora": False,
        "use_dora": False,
        "loftq_config": None,
    }

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "adapter_config.json"), "w") as f:
        json.dump(adapter_config, f, indent=2)
    save_file(lora_weights, os.path.join(save_dir, "adapter_model.safetensors"))


@pytest.fixture(scope="session")
def nomic_lora_adapter_1(tmp_path_factory):
    """Generate a random LoRA adapter (seed=42)."""
    adapter_dir = str(tmp_path_factory.mktemp("nomic_lora_1"))
    _create_random_nomic_lora(adapter_dir, seed=42)
    return adapter_dir


@pytest.fixture(scope="session")
def nomic_lora_adapter_2(tmp_path_factory):
    """Generate a different random LoRA adapter (seed=123)."""
    adapter_dir = str(tmp_path_factory.mktemp("nomic_lora_2"))
    _create_random_nomic_lora(adapter_dir, seed=123)
    return adapter_dir


def create_nomic_llm(enable_lora: bool = True, max_loras: int = 4):
    """Create a NomicBertModel LLM instance with optional LoRA support."""
    return vllm.LLM(
        model=NOMIC_MODEL,
        enable_lora=enable_lora,
        max_loras=max_loras if enable_lora else 1,
        max_lora_rank=LORA_RANK,
        max_model_len=4096,
        dtype="half",
        enforce_eager=True,
        trust_remote_code=True,
        gpu_memory_utilization=0.8,
        hf_overrides={"architectures": ["NomicBertModel"]},
    )


TEST_PROMPTS = [
    "What is machine learning?",
    "The quick brown fox jumps over the lazy dog.",
]


@create_new_process_for_each_test()
def test_nomic_lora_basic_inference(nomic_lora_adapter_1):
    """Test that NomicBertModel can load and run with a LoRA adapter."""
    llm = create_nomic_llm(enable_lora=True)

    lora_request = LoRARequest(
        lora_name="adapter-1",
        lora_int_id=1,
        lora_path=nomic_lora_adapter_1,
    )

    outputs = llm.embed(TEST_PROMPTS, lora_request=lora_request)

    assert len(outputs) == len(TEST_PROMPTS)
    for output in outputs:
        embedding = output.outputs.embedding
        assert len(embedding) > 0
        assert not all(v == 0.0 for v in embedding)


@create_new_process_for_each_test()
def test_nomic_lora_changes_output(nomic_lora_adapter_1):
    """Test that LoRA adapter produces different embeddings than base model."""
    llm = create_nomic_llm(enable_lora=True)

    base_outputs = llm.embed(TEST_PROMPTS)

    lora_request = LoRARequest(
        lora_name="adapter-1",
        lora_int_id=1,
        lora_path=nomic_lora_adapter_1,
    )
    lora_outputs = llm.embed(TEST_PROMPTS, lora_request=lora_request)

    for base_out, lora_out in zip(base_outputs, lora_outputs):
        base_emb = np.array(base_out.outputs.embedding)
        lora_emb = np.array(lora_out.outputs.embedding)
        cosine_sim = np.dot(base_emb, lora_emb) / (
            np.linalg.norm(base_emb) * np.linalg.norm(lora_emb)
        )
        assert cosine_sim < 1.0, (
            "LoRA adapter should produce different embeddings than base model"
        )


@create_new_process_for_each_test()
def test_nomic_multi_lora(nomic_lora_adapter_1, nomic_lora_adapter_2):
    """Test that different LoRA adapters produce different embeddings."""
    llm = create_nomic_llm(enable_lora=True, max_loras=4)

    lora_request_1 = LoRARequest(
        lora_name="adapter-1",
        lora_int_id=1,
        lora_path=nomic_lora_adapter_1,
    )
    lora_request_2 = LoRARequest(
        lora_name="adapter-2",
        lora_int_id=2,
        lora_path=nomic_lora_adapter_2,
    )

    outputs_1 = llm.embed(TEST_PROMPTS, lora_request=lora_request_1)
    outputs_2 = llm.embed(TEST_PROMPTS, lora_request=lora_request_2)

    for out_1, out_2 in zip(outputs_1, outputs_2):
        emb_1 = np.array(out_1.outputs.embedding)
        emb_2 = np.array(out_2.outputs.embedding)
        assert not np.allclose(emb_1, emb_2, atol=1e-5), (
            "Different LoRA adapters should produce different embeddings"
        )


@create_new_process_for_each_test()
def test_nomic_multi_lora_mixed_batch(nomic_lora_adapter_1, nomic_lora_adapter_2):
    """Test multi-adapter batching: different adapters in one call."""
    llm = create_nomic_llm(enable_lora=True, max_loras=4)

    lora_request_1 = LoRARequest(
        lora_name="adapter-1",
        lora_int_id=1,
        lora_path=nomic_lora_adapter_1,
    )
    lora_request_2 = LoRARequest(
        lora_name="adapter-2",
        lora_int_id=2,
        lora_path=nomic_lora_adapter_2,
    )

    prompt = TEST_PROMPTS[0]

    ref_1 = llm.embed([prompt], lora_request=lora_request_1)[0]
    ref_2 = llm.embed([prompt], lora_request=lora_request_2)[0]

    mixed = llm.embed(
        [prompt, prompt],
        lora_request=[lora_request_1, lora_request_2],
    )

    mixed_emb_1 = np.array(mixed[0].outputs.embedding)
    mixed_emb_2 = np.array(mixed[1].outputs.embedding)
    ref_emb_1 = np.array(ref_1.outputs.embedding)
    ref_emb_2 = np.array(ref_2.outputs.embedding)

    np.testing.assert_allclose(
        mixed_emb_1,
        ref_emb_1,
        atol=1e-3,
        err_msg="Mixed-batch adapter-1 output differs from single-adapter run",
    )
    np.testing.assert_allclose(
        mixed_emb_2,
        ref_emb_2,
        atol=1e-3,
        err_msg="Mixed-batch adapter-2 output differs from single-adapter run",
    )

    assert not np.allclose(mixed_emb_1, mixed_emb_2, atol=1e-5), (
        "Different adapters in mixed batch should produce different embeddings"
    )
