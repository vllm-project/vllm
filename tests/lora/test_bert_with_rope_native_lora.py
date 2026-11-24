# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LoRA adapters trained on Transformers' native `bert_with_rope` models (GTE,
Nomic BERT) name split `q/k/v_proj` and `gate/up_proj` modules. Served on
vLLM's fused layers, they must match Transformers with the same deltas merged
in, so a slice packed into the wrong part of a fused layer shows up here."""

import json
from pathlib import Path
from typing import cast

import pytest
import torch
from safetensors.torch import save_file

from vllm import EmbeddingRequestOutput, PoolingParams
from vllm.lora.request import LoRARequest
from vllm.platforms import current_platform

transformers = pytest.importorskip("transformers", minversion="5.18")

RANK = 8
ALPHA = 16
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
PROMPTS = [
    "What is the capital of France?",
    "Paris is the capital and largest city of France.",
]
MODELS = {  # case: (model, Transformers and vLLM architecture, task)
    "gte-reranker": (
        "alibaba-nlp-community/gte-multilingual-reranker-base",
        "GteForSequenceClassification",
        "classify",
    ),
    "nomic": ("nomic-ai/nomic-embed-text-v1.5", "NomicBertModel", "embed"),
}


def _transformers_outputs(case: str, path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Base and merged outputs from Transformers; writes the adapter to `path`."""
    model_id, architecture, task = MODELS[case]
    model = getattr(transformers, architecture).from_pretrained(
        model_id, dtype=torch.float32
    )
    inputs = transformers.AutoTokenizer.from_pretrained(model_id)(
        PROMPTS, padding=True, return_tensors="pt"
    )

    def forward() -> torch.Tensor:
        output = model.eval()(**inputs)
        if task == "classify":
            return output.logits
        mask = inputs["attention_mask"].unsqueeze(-1)
        return (output.last_hidden_state * mask).sum(1) / mask.sum(1)

    gen = torch.Generator().manual_seed(0)
    adapter = {}
    with torch.no_grad():
        base = forward()
        for name, module in model.named_modules():
            if not name.endswith(TARGETS):
                continue
            out_dim, in_dim = module.weight.shape
            lora_a = torch.randn(RANK, in_dim, generator=gen) * 0.05
            lora_b = torch.randn(out_dim, RANK, generator=gen) * 0.05
            adapter[f"base_model.model.{name}.lora_A.weight"] = lora_a
            adapter[f"base_model.model.{name}.lora_B.weight"] = lora_b
            module.weight += ALPHA / RANK * lora_b @ lora_a
        assert len(adapter) == 2 * len(TARGETS) * 12  # 12 layers
        config = {
            "peft_type": "LORA",
            "r": RANK,
            "lora_alpha": ALPHA,
            "target_modules": list(TARGETS),
            "bias": "none",
        }
        if task == "classify":  # the adapter also retrains the head
            for name, param in model.classifier.named_parameters():
                param += torch.randn(param.shape, generator=gen) * 0.05
                adapter[f"base_model.model.classifier.{name}"] = param.clone()
            config["modules_to_save"] = ["classifier"]
        merged = forward()
    path.mkdir()
    save_file(adapter, path / "adapter_model.safetensors")
    (path / "adapter_config.json").write_text(json.dumps(config))
    return base, merged


def _vllm_outputs(
    vllm_runner, case: str, lora: str
) -> tuple[torch.Tensor, torch.Tensor]:
    model_id, architecture, task = MODELS[case]
    with vllm_runner(
        model_id,
        hf_overrides={"architectures": [architecture]},
        runner="pooling",
        trust_remote_code=True,
        dtype="float32" if current_platform.is_cpu() else "float16",
        enforce_eager=True,
        max_model_len=512,
        enable_lora=True,
        max_lora_rank=RANK,
        gpu_memory_utilization=0.3,
        enable_chunked_prefill=None,
    ) as runner:
        params = PoolingParams(task=task, use_activation=False)
        results = []
        for request in [None, LoRARequest(case, 1, lora)]:
            if task == "embed":
                embeds = cast(
                    list[EmbeddingRequestOutput],
                    runner.llm.embed(
                        PROMPTS,
                        pooling_params=params,
                        lora_request=request,
                        use_tqdm=False,
                    ),
                )
                results.append(torch.tensor([o.outputs.embedding for o in embeds]))
            else:
                logits = runner.llm.classify(
                    PROMPTS, pooling_params=params, lora_request=request, use_tqdm=False
                )
                results.append(torch.tensor([o.outputs.probs for o in logits]))
        return results[0], results[1]


@pytest.mark.parametrize("case", list(MODELS))
def test_native_lora_matches_transformers(vllm_runner, tmp_path, case):
    if not hasattr(transformers, MODELS[case][1]):
        pytest.skip(f"Transformers has no native {MODELS[case][1]}")
    hf_base, hf_merged = _transformers_outputs(case, tmp_path / "lora")
    base, lora = _vllm_outputs(vllm_runner, case, str(tmp_path / "lora"))

    signal = (hf_merged - hf_base).abs().mean()
    assert signal > 1e-2
    assert (base - hf_base).abs().mean() < 0.05 * signal
    assert (lora - hf_merged).abs().mean() < 0.05 * signal
