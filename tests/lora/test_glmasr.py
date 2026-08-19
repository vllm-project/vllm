# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

import librosa
import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from transformers import AutoConfig

import vllm
from vllm.assets.audio import AudioAsset
from vllm.lora.request import LoRARequest

MODEL_PATH = "zai-org/GLM-ASR-Nano-2512"

PROMPT_TEMPLATE = (
    "<|user|>\n<|pad|>can you transcribe the speech into a written format?"
    "<|assistant|>\n"
)

SAMPLE_RATE = 16000
LORA_RANK = 8
LORA_B_STD = 0.02


def _tower_connector_modules(config) -> dict[str, tuple[int, int]]:
    """Every LoRA-able linear in the audio tower and projector, mapped to its
    (in_features, out_features)."""
    audio_config = config.audio_config
    text_config = config.text_config
    hidden = audio_config.hidden_size
    q_size = audio_config.num_attention_heads * audio_config.head_dim
    kv_size = audio_config.num_key_value_heads * audio_config.head_dim

    modules: dict[str, tuple[int, int]] = {}
    for i in range(audio_config.num_hidden_layers):
        prefix = f"audio_tower.layers.{i}"
        modules[f"{prefix}.self_attn.q_proj"] = (hidden, q_size)
        modules[f"{prefix}.self_attn.k_proj"] = (hidden, kv_size)
        modules[f"{prefix}.self_attn.v_proj"] = (hidden, kv_size)
        modules[f"{prefix}.self_attn.o_proj"] = (q_size, hidden)
        modules[f"{prefix}.mlp.fc1"] = (hidden, audio_config.intermediate_size)
        modules[f"{prefix}.mlp.fc2"] = (audio_config.intermediate_size, hidden)

    modules["multi_modal_projector.linear_1"] = (
        audio_config.intermediate_size,
        text_config.hidden_size * 2,
    )
    modules["multi_modal_projector.linear_2"] = (
        text_config.hidden_size * 2,
        text_config.hidden_size,
    )
    return modules


def _save_peft_adapter(
    save_dir: Path,
    modules: dict[str, tuple[int, int]],
    perturbed_prefix: str | None,
) -> None:
    """Write a PEFT-format adapter targeting every tower/projector module.

    `lora_B` is zero (a no-op adapter) except for modules under
    `perturbed_prefix`, whose outputs get perturbed.
    """
    rng = np.random.default_rng(0)
    weights: dict[str, torch.Tensor] = {}
    for name, (in_features, out_features) in modules.items():
        # Matches PEFT's default kaiming-uniform init of lora_A (a=sqrt(5)).
        bound = 1 / np.sqrt(in_features)
        lora_a = rng.uniform(-bound, bound, size=(LORA_RANK, in_features))
        lora_b = rng.standard_normal(size=(out_features, LORA_RANK)) * LORA_B_STD
        if perturbed_prefix is None or not name.startswith(perturbed_prefix):
            lora_b[:] = 0
        weights[f"base_model.model.{name}.lora_A.weight"] = torch.from_numpy(
            lora_a.astype(np.float16)
        )
        weights[f"base_model.model.{name}.lora_B.weight"] = torch.from_numpy(
            lora_b.astype(np.float16)
        )

    adapter_config = {
        "peft_type": "LORA",
        "base_model_name_or_path": MODEL_PATH,
        "task_type": "CAUSAL_LM",
        "r": LORA_RANK,
        "lora_alpha": 2 * LORA_RANK,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": sorted(modules),
    }
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "adapter_config.json").write_text(json.dumps(adapter_config))
    save_file(weights, str(save_dir / "adapter_model.safetensors"))


@pytest.fixture(scope="module")
def glmasr_tower_connector_loras(tmp_path_factory) -> dict[str, LoRARequest]:
    config = AutoConfig.from_pretrained(MODEL_PATH)
    modules = _tower_connector_modules(config)
    root = tmp_path_factory.mktemp("glmasr_lora")
    perturbed_prefixes = {
        "identity": None,
        "tower": "audio_tower.",
        "connector": "multi_modal_projector.",
    }
    lora_requests = {}
    for lora_id, (name, prefix) in enumerate(perturbed_prefixes.items(), start=1):
        _save_peft_adapter(root / name, modules, prefix)
        lora_requests[name] = LoRARequest(name, lora_id, str(root / name))
    return lora_requests


@pytest.fixture(scope="module")
def audios() -> list[tuple[np.ndarray, int]]:
    short, sr = AudioAsset("mary_had_lamb").audio_and_sample_rate  # ~16s
    assert sr == SAMPLE_RATE
    other, other_sr = AudioAsset("winning_call").audio_and_sample_rate
    other = librosa.resample(other, orig_sr=other_sr, target_sr=SAMPLE_RATE)
    # ~55s -> two 30s chunks, the second one partially filled.
    long = np.concatenate([short, other, short])[: 55 * SAMPLE_RATE]
    return [(short, SAMPLE_RATE), (long, SAMPLE_RATE)]


def _generate(
    llm: vllm.LLM,
    audios: list[tuple[np.ndarray, int]],
    lora_requests: list[LoRARequest | None],
) -> list[tuple[str, float]]:
    """Greedy (text, cumulative_logprob) per prompt; the logprob is sensitive
    to perturbations too small to flip the argmax."""
    prompts = [
        {"prompt": PROMPT_TEMPLATE, "multi_modal_data": {"audio": audio}}
        for audio in audios
    ]
    sampling_params = vllm.SamplingParams(temperature=0, max_tokens=32, logprobs=0)
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_requests)
    return [
        (output.outputs[0].text, output.outputs[0].cumulative_logprob)
        for output in outputs
    ]


def test_glmasr_tower_connector_lora(
    glmasr_tower_connector_loras, audios, monkeypatch: pytest.MonkeyPatch
):
    """Adapters that only target the audio tower and projector must load, be
    applied per request, and align with the tower/connector token counts for
    single- and multi-chunk audio."""
    # Force SPLIT_K=1 in the Triton LoRA shrink kernel so the identity adapter
    # reproduces the base outputs bit-exactly (see tests/lora/test_qwenvl.py).
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    identity = glmasr_tower_connector_loras["identity"]
    tower = glmasr_tower_connector_loras["tower"]
    connector = glmasr_tower_connector_loras["connector"]

    llm = vllm.LLM(
        MODEL_PATH,
        max_model_len=2048,
        max_num_seqs=4,
        enable_lora=True,
        max_loras=3,
        max_lora_rank=LORA_RANK,
        enable_tower_connector_lora=True,
        limit_mm_per_prompt={"audio": 1},
        mm_processor_cache_gb=0,
        gpu_memory_utilization=0.6,
    )

    base = _generate(llm, audios, [None] * len(audios))
    assert all(text for text, _ in base)

    # A zero-initialised lora_B must be a no-op even though every tower and
    # projector linear is wrapped.
    assert _generate(llm, audios, [identity] * len(audios)) == base

    # Perturbing only the tower (resp. only the projector) must change the
    # outputs, proving the LoRA is applied in each forward.
    tower_outputs = _generate(llm, audios, [tower] * len(audios))
    connector_outputs = _generate(llm, audios, [connector] * len(audios))
    for outputs in (tower_outputs, connector_outputs):
        assert all(
            logprob != base_logprob
            for (_, logprob), (_, base_logprob) in zip(outputs, base)
        )

    # Mixed adapters in one batch: each item must get its own adapter.
    mixed = _generate(llm, audios + audios, [identity, tower, connector, identity])
    assert mixed == [base[0], tower_outputs[1], connector_outputs[0], base[1]]
