# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LoRA adapters trained on Transformers' native GTE (`model_type: gte`) name
split `q/k/v_proj` and `gate/up_proj` modules. Served on vLLM's fused layers,
they must match Transformers with the same deltas merged in, so a slice packed
into the wrong half of `qkv_proj` or `up_gate_proj` shows up here."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

import vllm
from vllm import PoolingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.lora.request import LoRARequest
from vllm.platforms import current_platform

transformers = pytest.importorskip("transformers", minversion="5.18")

MODEL = "alibaba-nlp-community/gte-multilingual-reranker-base"
RANK = 8
ALPHA = 16
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
PROMPTS = [
    "What is the capital of France?",
    "Paris is the capital and largest city of France.",
]


def _transformers_logits(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Base and merged logits from Transformers; writes the adapter to `path`."""
    model = transformers.GteForSequenceClassification.from_pretrained(
        MODEL, dtype=torch.float32
    ).eval()
    inputs = transformers.AutoTokenizer.from_pretrained(MODEL)(
        PROMPTS, padding=True, return_tensors="pt"
    )
    gen = torch.Generator().manual_seed(0)
    adapter = {}
    with torch.no_grad():
        base = model(**inputs).logits
        for name, module in model.named_modules():
            if not name.endswith(TARGETS):
                continue
            out_dim, in_dim = module.weight.shape
            lora_a = torch.randn(RANK, in_dim, generator=gen) * 0.05
            lora_b = torch.randn(out_dim, RANK, generator=gen) * 0.05
            adapter[f"base_model.model.{name}.lora_A.weight"] = lora_a
            adapter[f"base_model.model.{name}.lora_B.weight"] = lora_b
            module.weight += ALPHA / RANK * lora_b @ lora_a
        # The adapter also retrains the head (`modules_to_save`).
        for name, param in model.classifier.named_parameters():
            param += torch.randn(param.shape, generator=gen) * 0.05
            adapter[f"base_model.model.classifier.{name}"] = param.clone()
        merged = model(**inputs).logits
    assert len(adapter) == 2 * len(TARGETS) * 12 + 2  # 12 layers + head
    path.mkdir()
    save_file(adapter, path / "adapter_model.safetensors")
    config = {
        "peft_type": "LORA",
        "r": RANK,
        "lora_alpha": ALPHA,
        "target_modules": list(TARGETS),
        "modules_to_save": ["classifier"],
        "bias": "none",
    }
    (path / "adapter_config.json").write_text(json.dumps(config))
    return base, merged


def _vllm_logits(lora: str) -> tuple[torch.Tensor, torch.Tensor]:
    llm = vllm.LLM(
        model=MODEL,
        hf_overrides={"architectures": ["GteForSequenceClassification"]},
        runner="pooling",
        dtype="float32" if current_platform.is_cpu() else "float16",
        enforce_eager=True,
        max_model_len=512,
        enable_lora=True,
        max_lora_rank=RANK,
        gpu_memory_utilization=0.3,
    )
    params = PoolingParams(task="classify", use_activation=False)
    results = []
    for request in [None, LoRARequest("native", 1, lora)]:
        outputs = llm.classify(
            PROMPTS, pooling_params=params, lora_request=request, use_tqdm=False
        )
        results.append(torch.tensor([o.outputs.probs for o in outputs]))
    del llm
    cleanup_dist_env_and_memory()
    return results[0], results[1]


def test_native_gte_lora_matches_transformers(tmp_path):
    hf_base, hf_merged = _transformers_logits(tmp_path / "lora")
    base, lora = _vllm_logits(str(tmp_path / "lora"))

    signal = (hf_merged - hf_base).abs().mean()
    assert signal > 1e-2
    assert (base - hf_base).abs().mean() < 0.05 * signal
    assert (lora - hf_merged).abs().mean() < 0.05 * signal
