# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma4 MoE expert LoRA in the fused 3D PEFT layout.

PEFT `target_parameters` adapters store one flat tensor pair per projection:
`experts.base_layer.lora_{A,B}` for `gate_up_proj` and `experts.lora_{A,B}` for
`down_proj`. Serving such an adapter must match serving the base model with the
LoRA delta merged into the expert weights.
"""

import json
import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import Gemma4Config
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4ForCausalLM,
    Gemma4ForConditionalGeneration,
)

import vllm
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.lora.request import LoRARequest
from vllm.transformers_utils.repo_utils import hf_api

NUM_EXPERTS = 8
HIDDEN = 256
MOE_INTERMEDIATE = 128
NUM_LAYERS = 2
RANK = 8
ALPHA = 16
VOCAB = 262144
TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "processor_config.json",
    "chat_template.jinja",
]


def _config() -> Gemma4Config:
    return Gemma4Config(
        text_config=dict(
            vocab_size=VOCAB,
            vocab_size_per_layer_input=VOCAB,
            hidden_size=HIDDEN,
            hidden_size_per_layer_input=0,
            intermediate_size=512,
            moe_intermediate_size=MOE_INTERMEDIATE,
            num_hidden_layers=NUM_LAYERS,
            layer_types=["sliding_attention", "full_attention"],
            num_attention_heads=4,
            num_key_value_heads=2,
            num_global_key_value_heads=1,
            head_dim=64,
            global_head_dim=128,
            num_experts=NUM_EXPERTS,
            top_k_experts=2,
            enable_moe_block=True,
            sliding_window=64,
            max_position_embeddings=1024,
            num_kv_shared_layers=0,
        ),
        vision_config=dict(num_hidden_layers=1, default_output_length=280),
    )


def _merge(weight: torch.Tensor, lora_a: torch.Tensor, lora_b: torch.Tensor):
    """Add the per-expert delta of a fused 3D LoRA pair to a 3D expert weight.

    lora_a is (num_experts * rank, in), expert-major; lora_b is
    (out, rank * num_experts), expert-minor.
    """
    a = lora_a.view(NUM_EXPERTS, RANK, -1)
    b = lora_b.view(lora_b.shape[0], RANK, NUM_EXPERTS).permute(2, 0, 1)
    return weight + (ALPHA / RANK) * torch.bmm(b, a)


@pytest.fixture(scope="module")
def gemma4_tokenizer_dir() -> Path:
    return Path(
        hf_api().snapshot_download(
            repo_id="google/gemma-4-E2B-it", allow_patterns=TOKENIZER_FILES
        )
    )


def _build(root: Path, tokenizer_dir: Path, multimodal: bool) -> tuple[str, str, str]:
    torch.manual_seed(0)
    config = _config()
    if multimodal:
        model = Gemma4ForConditionalGeneration(config)
        ckpt_prefix = "model.language_model.layers"
    else:
        model = Gemma4ForCausalLM(config.text_config)
        ckpt_prefix = "model.layers"
    base_dir, merged_dir, lora_dir = (root / n for n in ("base", "merged", "lora"))
    for model_dir in (base_dir, merged_dir):
        model.save_pretrained(model_dir)
        for name in TOKENIZER_FILES:
            shutil.copy(tokenizer_dir / name, model_dir / name)

    weights = load_file(merged_dir / "model.safetensors")
    adapter = {}
    for layer in range(NUM_LAYERS):
        experts = f"{ckpt_prefix}.{layer}.experts"
        peft = f"base_model.model.{experts}"
        for proj, module, in_dim, out_dim in (
            ("gate_up_proj", "base_layer.", HIDDEN, 2 * MOE_INTERMEDIATE),
            ("down_proj", "", MOE_INTERMEDIATE, HIDDEN),
        ):
            lora_a = torch.randn(NUM_EXPERTS * RANK, in_dim) * 0.1
            lora_b = torch.randn(out_dim, RANK * NUM_EXPERTS) * 0.1
            adapter[f"{peft}.{module}lora_A.weight"] = lora_a
            adapter[f"{peft}.{module}lora_B.weight"] = lora_b
            name = f"{experts}.{proj}"
            weights[name] = _merge(weights[name].float(), lora_a, lora_b).to(
                weights[name].dtype
            )
    save_file(weights, merged_dir / "model.safetensors", metadata={"format": "pt"})

    lora_dir.mkdir()
    save_file(adapter, lora_dir / "adapter_model.safetensors")
    (lora_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": RANK,
                "lora_alpha": ALPHA,
                "target_modules": [],
                "target_parameters": ["experts.gate_up_proj", "experts.down_proj"],
                "bias": "none",
            }
        )
    )
    return str(base_dir), str(merged_dir), str(lora_dir)


def _prompt_logprobs(
    model: str, lora_path: str | None = None, hf_overrides: dict | None = None
) -> list[list[list[float]]]:
    """Prompt logprobs without LoRA, then with `lora_path` if given."""
    llm = vllm.LLM(
        model,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=256,
        max_num_seqs=4,
        gpu_memory_utilization=0.3,
        enable_lora=True,
        max_lora_rank=RANK,
        limit_mm_per_prompt={"image": 0},
        hf_overrides=hf_overrides,
    )
    try:
        generator = torch.Generator().manual_seed(1)
        prompts = [
            torch.randint(1000, 10000, (32,), generator=generator).tolist()
            for _ in range(3)
        ]
        params = vllm.SamplingParams(max_tokens=1, prompt_logprobs=0, detokenize=False)
        loras = [None]
        if lora_path is not None:
            loras.append(LoRARequest("expert-3d", 1, lora_path))
        results = []
        for lora in loras:
            outputs = llm.generate(
                [{"prompt_token_ids": p} for p in prompts], params, lora_request=lora
            )
            results.append(
                [_token_logprobs(out, p) for out, p in zip(outputs, prompts)]
            )
        return results
    finally:
        del llm
        cleanup_dist_env_and_memory()


def _token_logprobs(output: vllm.RequestOutput, prompt: list[int]) -> list[float]:
    assert output.prompt_logprobs is not None
    logprobs = []
    for token, position in zip(prompt[1:], output.prompt_logprobs[1:]):
        assert position is not None
        logprobs.append(position[token].logprob)
    return logprobs


def _mean_abs_diff(x: list[list[float]], y: list[list[float]]) -> float:
    diffs = [abs(a - b) for xs, ys in zip(x, y) for a, b in zip(xs, ys)]
    return sum(diffs) / len(diffs)


@pytest.mark.parametrize(
    ("multimodal", "hf_overrides"),
    [
        (True, None),
        (False, None),
        (True, {"architectures": ["Gemma4ForCausalLM"]}),
    ],
    ids=["mm", "text", "mm-checkpoint-text-model"],
)
def test_gemma4_moe_3d_expert_lora_matches_merged(
    tmp_path, gemma4_tokenizer_dir, multimodal, hf_overrides
):
    base, merged, lora = _build(tmp_path, gemma4_tokenizer_dir, multimodal)

    base_lp, lora_lp = _prompt_logprobs(base, lora, hf_overrides)
    (merged_lp,) = _prompt_logprobs(merged, hf_overrides=hf_overrides)

    signal = _mean_abs_diff(merged_lp, base_lp)
    assert signal > 0.1
    assert _mean_abs_diff(lora_lp, merged_lp) < 0.1 * signal
