# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LoRA on `bert_with_rope` encoders: serving a PEFT adapter must match serving
the base model with the LoRA delta merged into its weights. GTE's MLP is the
checkpoint's fused `up_gate_proj` (`[up; gate]`) and Nomic BERT's separate
`fc11`/`fc12` are packed into `gate_up_proj`, so an adapter applied to the wrong
half shows up here. The reranker adapter also replaces the classification head
through `modules_to_save`."""

import json
import shutil
from pathlib import Path
from typing import cast

import pytest
import torch
from safetensors.torch import load_file, save_file

from vllm import EmbeddingRequestOutput, PoolingParams
from vllm.lora.request import LoRARequest
from vllm.platforms import current_platform
from vllm.transformers_utils.repo_utils import hf_api

RANK = 8
ALPHA = 16
GTE_TARGETS = ("qkv_proj", "o_proj", "up_gate_proj", "down_proj")
PROMPTS = [
    "What is the capital of France?",
    "Paris is the capital and largest city of France.",
]
MODELS = {  # case: (model, vLLM architecture, task, checkpoint modules to adapt)
    "gte": ("Alibaba-NLP/gte-multilingual-base", "GteNewModel", "embed", GTE_TARGETS),
    "gte-reranker": (
        "Alibaba-NLP/gte-multilingual-reranker-base",
        "GteNewForSequenceClassification",
        "classify",
        GTE_TARGETS,
    ),
    "nomic": (
        "nomic-ai/nomic-embed-text-v1.5",
        "NomicBertModel",
        "embed",
        ("Wqkv", "out_proj", "fc11", "fc12", "fc2"),
    ),
    # Expert weights are not LoRA targets; the dense MLP is in every other layer.
    "nomic-moe": (
        "nomic-ai/nomic-embed-text-v2-moe",
        "NomicBertModel",
        "embed",
        ("Wqkv", "out_proj", "fc1", "fc2"),
    ),
}


def _build(root: Path, case: str) -> tuple[str, str, str]:
    model_id, _, task, targets = MODELS[case]
    base = Path(
        hf_api().snapshot_download(
            repo_id=model_id,
            allow_patterns=["*.json", "*.safetensors", "*.model"],
        )
    )
    merged, lora = root / "merged", root / "lora"
    shutil.copytree(base, merged)
    weights = load_file(merged / "model.safetensors")
    gen = torch.Generator().manual_seed(0)
    adapter = {}
    for name, weight in weights.items():
        module = name.removesuffix(".weight")
        if module == name or not module.endswith(targets):
            continue
        out_dim, in_dim = weight.shape
        lora_a = torch.randn(RANK, in_dim, generator=gen) * 0.05
        lora_b = torch.randn(out_dim, RANK, generator=gen) * 0.05
        adapter[f"base_model.model.{module}.lora_A.weight"] = lora_a
        adapter[f"base_model.model.{module}.lora_B.weight"] = lora_b
        weights[name] = (weight.float() + ALPHA / RANK * lora_b @ lora_a).to(
            weight.dtype
        )
    assert {name.split(".")[-3] for name in adapter} == set(targets)
    # The reranker adapter also retrains its head (`modules_to_save`).
    head = [n for n in weights if n.startswith("classifier.") and task == "classify"]
    for name in head:
        weight = weights[name]
        noise = torch.randn(weight.shape, generator=gen) * 0.05
        weights[name] = adapter[f"base_model.model.{name}"] = (
            weight.float() + noise
        ).to(weight.dtype)
    save_file(weights, merged / "model.safetensors", metadata={"format": "pt"})

    lora.mkdir()
    save_file(adapter, lora / "adapter_model.safetensors")
    config = {
        "peft_type": "LORA",
        "r": RANK,
        "lora_alpha": ALPHA,
        "target_modules": list(targets),
        "bias": "none",
    }
    if head:
        config["modules_to_save"] = ["classifier"]
    (lora / "adapter_config.json").write_text(json.dumps(config))
    return str(base), str(merged), str(lora)


def _outputs(
    vllm_runner, case: str, model: str, lora: str | None = None
) -> list[torch.Tensor]:
    _, architecture, task, _ = MODELS[case]
    with vllm_runner(
        model,
        hf_overrides={"architectures": [architecture]},
        runner="pooling",
        trust_remote_code=True,
        dtype="float32" if current_platform.is_cpu() else "float16",
        enforce_eager=True,
        max_model_len=512,
        enable_lora=lora is not None,
        max_lora_rank=RANK,
        gpu_memory_utilization=0.3,
        enable_chunked_prefill=None,
    ) as runner:
        requests = [None] if lora is None else [None, LoRARequest(case, 1, lora)]
        results = []
        for request in requests:
            if task == "embed":
                embeds = cast(
                    list[EmbeddingRequestOutput],
                    runner.llm.embed(PROMPTS, lora_request=request, use_tqdm=False),
                )
                results.append(torch.tensor([o.outputs.embedding for o in embeds]))
            else:
                params = PoolingParams(task="classify", use_activation=False)
                logits = runner.llm.classify(
                    PROMPTS, pooling_params=params, lora_request=request, use_tqdm=False
                )
                results.append(torch.tensor([o.outputs.probs for o in logits]))
        return results


@pytest.mark.parametrize("case", list(MODELS))
def test_lora_matches_merged(vllm_runner, tmp_path, case):
    if case == "nomic-moe" and current_platform.is_cpu():
        pytest.skip("Nomic MoE routing needs Triton")
    base, merged, lora = _build(tmp_path, case)

    base_out, lora_out = _outputs(vllm_runner, case, base, lora)
    (merged_out,) = _outputs(vllm_runner, case, merged)

    signal = (merged_out - base_out).abs().mean()
    assert signal > 1e-2
    assert (lora_out - merged_out).abs().mean() < 0.05 * signal
