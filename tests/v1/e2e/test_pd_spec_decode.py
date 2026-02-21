# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for PD disaggregation combined with speculative decoding.

Verifies that KV transfer via ExampleConnector works correctly when
speculative decoding is enabled (both target and drafter KV must be
saved/loaded).

Tests cover:
  - MTP (DeepSeek-V3-4layers)
  - EAGLE (Llama-3.1-8B + EAGLE drafter)
  - EAGLE3 (GPT-OSS-20B + EAGLE3 speculator)
"""

import os
from dataclasses import dataclass, field
from typing import Any

import pytest
import safetensors.torch
import torch

from tests.utils import large_gpu_mark
from tests.v1.e2e.test_spec_decode import compute_acceptance_rate, get_test_prompts
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.platforms import current_platform

# ── Thresholds ──────────────────────────────────────────────────────────

MTP_MATCH_RATE = 0.8
EAGLE_MATCH_RATE = 0.6
ACCEPTANCE_RATE_TOLERANCE = 0.10
MAX_MODEL_LEN = 2048


# ── Model configs ───────────────────────────────────────────────────────


@dataclass
class SDModelConfig:
    model: str
    speculative_config: dict[str, Any]
    gpu_memory_utilization: float = 0.8
    enable_chunked_prefill: bool = True
    attention_config: dict[str, str] | None = None
    extra_llm_kwargs: dict[str, Any] = field(default_factory=dict)


LLAMA_EAGLE = SDModelConfig(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_config={
        "method": "eagle",
        "model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
        "num_speculative_tokens": 3,
        "max_model_len": MAX_MODEL_LEN,
    },
    attention_config={"backend": "FLASH_ATTN"},
)

GPT_OSS_EAGLE3 = SDModelConfig(
    model="openai/gpt-oss-20b",
    speculative_config={
        "method": "eagle3",
        "model": "RedHatAI/gpt-oss-20b-speculator.eagle3",
        "num_speculative_tokens": 3,
        "max_model_len": MAX_MODEL_LEN,
    },
    attention_config={"backend": "TRITON_ATTN"},
    gpu_memory_utilization=0.9,
)

# ExampleConnector is incompatible with GPT-OSS's hybrid sliding/full
# attention: slot mappings assume a single block table (block_ids[0]).
# GPT-OSS + NixlConnector works (raw block transfer, no layout parsing).
_gpt_oss_skip = pytest.mark.skip(
    reason="ExampleConnector incompatible with hybrid attention models"
)


# ── Shared helpers ──────────────────────────────────────────────────────


def _count_output_matches(outputs_a: list, outputs_b: list) -> int:
    matches = 0
    for a, b in zip(outputs_a, outputs_b):
        if a.outputs[0].text == b.outputs[0].text:
            matches += 1
        else:
            print(f"ref_output: {a.outputs[0].text}")
            print(f"spec_output: {b.outputs[0].text}")
    return matches


def _make_kv_config(tmp_path) -> KVTransferConfig:
    return KVTransferConfig(
        kv_connector="ExampleConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"shared_storage_path": str(tmp_path)},
    )


def _run_sd(
    config: SDModelConfig,
    prompts,
    sampling,
    kv_transfer_config=None,
) -> tuple[list, float]:
    """Run speculative decoding with given config.

    Returns (outputs, acceptance_rate).
    """
    llm = LLM(
        model=config.model,
        trust_remote_code=True,
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=True,
        gpu_memory_utilization=config.gpu_memory_utilization,
        speculative_config=config.speculative_config,
        enable_chunked_prefill=config.enable_chunked_prefill,
        disable_log_stats=False,
        attention_config=config.attention_config,
        kv_transfer_config=kv_transfer_config,
        **config.extra_llm_kwargs,
    )
    outputs = llm.chat(prompts, sampling)
    acceptance_rate = compute_acceptance_rate(llm.get_metrics())
    del llm
    torch.cuda.empty_cache()
    cleanup_dist_env_and_memory()
    return outputs, acceptance_rate


def _zero_out_drafter_kv(storage_path: str) -> int:
    """Zero out the highest-numbered layer (drafter) in each hash dir.

    Returns the number of files zeroed.
    """
    count = 0
    for hash_dir in os.listdir(storage_path):
        hash_path = os.path.join(storage_path, hash_dir)
        if not os.path.isdir(hash_path):
            continue
        layer_files = [f for f in os.listdir(hash_path) if f.endswith(".safetensors")]
        max_idx = -1
        drafter_file = None
        for f in layer_files:
            parts = f.split(".")
            try:
                idx = int(parts[2])
                if idx > max_idx:
                    max_idx = idx
                    drafter_file = f
            except (IndexError, ValueError):
                continue
        if drafter_file is None:
            continue
        filepath = os.path.join(hash_path, drafter_file)
        data = safetensors.torch.load_file(filepath)
        zeroed = {k: torch.zeros_like(v) for k, v in data.items()}
        safetensors.torch.save_file(zeroed, filepath)
        count += 1
    return count


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def reset_torch_dynamo():
    yield
    torch._dynamo.reset()


# ── MTP Tests ───────────────────────────────────────────────────────────


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
def test_pd_mtp_output_match(tmp_path, monkeypatch):
    """PD KV transfer must not change MTP output."""
    monkeypatch.setenv("VLLM_MLA_DISABLE", "1")
    monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_FP8", "0")

    sampling = SamplingParams(temperature=0.0, max_tokens=30)
    prompts = get_test_prompts(mm_enabled=False, num_prompts=50)

    mtp_config = SDModelConfig(
        model="ZixiQi/DeepSeek-V3-4layers-MTP-FP8",
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": 1,
            "max_model_len": MAX_MODEL_LEN,
        },
        gpu_memory_utilization=0.5,
    )

    ref_outputs, _ = _run_sd(mtp_config, prompts, sampling)
    pd_outputs, _ = _run_sd(
        mtp_config, prompts, sampling, kv_transfer_config=_make_kv_config(tmp_path)
    )

    matches = _count_output_matches(ref_outputs, pd_outputs)
    rate = matches / len(ref_outputs)
    print(f"\n  PD+MTP match rate: {matches}/{len(ref_outputs)} = {rate:.0%}")

    assert matches > int(MTP_MATCH_RATE * len(ref_outputs)), (
        f"PD+MTP match rate {rate:.0%} below threshold {MTP_MATCH_RATE:.0%}"
    )


# ── EAGLE / EAGLE3 Tests ────────────────────────────────────────────────


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
@pytest.mark.parametrize(
    "config",
    [
        pytest.param(LLAMA_EAGLE, id="llama3_eagle", marks=large_gpu_mark(min_gb=40)),
        pytest.param(
            GPT_OSS_EAGLE3,
            id="gpt_oss_eagle3",
            marks=[large_gpu_mark(min_gb=80), _gpt_oss_skip],
        ),
    ],
)
def test_pd_eagle_acceptance(tmp_path, config: SDModelConfig):
    """PD+SD must not degrade acceptance rate."""
    sampling = SamplingParams(temperature=0.0, max_tokens=30)
    prompts = get_test_prompts(mm_enabled=False, num_prompts=50)

    sd_outputs, sd_acceptance = _run_sd(config, prompts, sampling)
    print(f"\n  SD-only acceptance rate: {sd_acceptance:.2%}")

    pd_outputs, pd_acceptance = _run_sd(
        config, prompts, sampling, kv_transfer_config=_make_kv_config(tmp_path)
    )
    print(f"  PD+SD  acceptance rate: {pd_acceptance:.2%}")
    print(f"  Acceptance delta: {pd_acceptance - sd_acceptance:+.2%}")

    matches = _count_output_matches(sd_outputs, pd_outputs)
    output_rate = matches / len(sd_outputs)
    print(f"  Output match rate: {matches}/{len(sd_outputs)} = {output_rate:.0%}")

    assert matches > int(EAGLE_MATCH_RATE * len(sd_outputs)), (
        f"Output match rate {output_rate:.0%} below {EAGLE_MATCH_RATE:.0%}"
    )
    assert abs(pd_acceptance - sd_acceptance) <= ACCEPTANCE_RATE_TOLERANCE, (
        f"Acceptance rate diverged: SD={sd_acceptance:.2%}, "
        f"PD+SD={pd_acceptance:.2%}, "
        f"delta={pd_acceptance - sd_acceptance:+.2%} "
        f"exceeds tolerance {ACCEPTANCE_RATE_TOLERANCE:.0%}"
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
@pytest.mark.parametrize(
    "config",
    [
        pytest.param(LLAMA_EAGLE, id="llama3_eagle", marks=large_gpu_mark(min_gb=40)),
        pytest.param(
            GPT_OSS_EAGLE3,
            id="gpt_oss_eagle3",
            marks=[large_gpu_mark(min_gb=80), _gpt_oss_skip],
        ),
    ],
)
def test_pd_eagle_prefill_decode(tmp_path, config: SDModelConfig):
    """Simulates PD disagg: prefill saves KV, decode loads it.

    Prefill instance: computes and saves KV for both target and drafter
    layers via ExampleConnector.
    Decode instance: loads saved KV and continues generation.
    Outputs must mostly match a reference run without connector.
    """
    sampling = SamplingParams(temperature=0.0, max_tokens=30)
    prompts = get_test_prompts(mm_enabled=False, num_prompts=50)

    ref_outputs, _ = _run_sd(config, prompts, sampling)

    kv_config = _make_kv_config(tmp_path)
    prefill_sampling = SamplingParams(temperature=0.0, max_tokens=1)
    _run_sd(config, prompts, prefill_sampling, kv_transfer_config=kv_config)

    pd_outputs, _ = _run_sd(config, prompts, sampling, kv_transfer_config=kv_config)

    matches = _count_output_matches(ref_outputs, pd_outputs)
    output_rate = matches / len(ref_outputs)
    print(
        f"\n  Prefill/decode match rate: "
        f"{matches}/{len(ref_outputs)} = {output_rate:.0%}"
    )

    assert matches > int(EAGLE_MATCH_RATE * len(ref_outputs)), (
        f"Output match rate {output_rate:.0%} below {EAGLE_MATCH_RATE:.0%}"
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
@pytest.mark.parametrize(
    "config",
    [
        pytest.param(LLAMA_EAGLE, id="llama3_eagle", marks=large_gpu_mark(min_gb=40)),
        pytest.param(
            GPT_OSS_EAGLE3,
            id="gpt_oss_eagle3",
            marks=[large_gpu_mark(min_gb=80), _gpt_oss_skip],
        ),
    ],
)
def test_pd_eagle_zeroed_drafter_kv(tmp_path, config: SDModelConfig):
    """Zeroing drafter KV must degrade acceptance rate.

    Proves drafter KV transfer is meaningful:
    1. Prefill instance saves KV (target + drafter) to disk.
    2. Zero out only drafter layer files.
    3. Decode instance loads real target KV + zeroed drafter KV.
    4. Assert acceptance rate drops vs reference.
    """
    sampling = SamplingParams(temperature=0.0, max_tokens=30)
    prompts = get_test_prompts(mm_enabled=False, num_prompts=50)

    _, ref_acc = _run_sd(config, prompts, sampling)
    print(f"\n  Reference acceptance rate: {ref_acc:.2%}")

    kv_config = _make_kv_config(tmp_path)
    prefill_sampling = SamplingParams(temperature=0.0, max_tokens=1)
    _run_sd(config, prompts, prefill_sampling, kv_transfer_config=kv_config)

    n_zeroed = _zero_out_drafter_kv(str(tmp_path))
    print(f"  Zeroed drafter KV in {n_zeroed} hash dirs.")

    _, corrupted_acc = _run_sd(config, prompts, sampling, kv_transfer_config=kv_config)
    print(f"  Corrupted drafter acceptance rate: {corrupted_acc:.2%}")
    print(f"  Delta vs reference: {corrupted_acc - ref_acc:+.2%}")

    assert corrupted_acc < ref_acc, (
        f"Expected degraded acceptance with zeroed drafter KV, "
        f"but got {corrupted_acc:.2%} >= reference {ref_acc:.2%}"
    )
