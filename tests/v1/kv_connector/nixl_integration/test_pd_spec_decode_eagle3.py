# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
PD Disaggregation + EAGLE3 Acceptance Length Tests.

Validates that EAGLE3 speculative decoding acceptance lengths do not regress
when running in prefill-decode disaggregation mode via NixlConnector.

These tests are driven by the shell script pd_spec_decode_eagle3_test.sh,
which starts the prefill/decode servers and proxy before invoking pytest.
The shell script sets these environment variables:
  TEST_CONFIG  - config name (e.g. "qwen3-8b-eagle3")
  TEST_MODEL   - model name (e.g. "Qwen/Qwen3-8B")
  DECODE_PORT  - port of the decode server (for /metrics)
  PROXY_PORT   - port of the proxy server (for /v1/completions)
"""

import os
from dataclasses import dataclass, field
from types import SimpleNamespace
from urllib.request import urlopen

import openai
import regex as re
from transformers import AutoTokenizer

from vllm.benchmarks.datasets import get_samples

# ── Config from environment (set by shell script) ────────────────────────

TEST_CONFIG = os.environ.get("TEST_CONFIG", "")
MODEL_NAME = os.environ.get("TEST_MODEL", "")
DECODE_PORT = int(os.environ.get("DECODE_PORT", "8200"))
PROXY_PORT = int(os.environ.get("PROXY_PORT", "8192"))
PROXY_BASE_URL = f"http://localhost:{PROXY_PORT}/v1"

# ── Test parameters ──────────────────────────────────────────────────────

DEFAULT_NUM_PROMPTS = 80
DEFAULT_OUTPUT_LEN = 256
DEFAULT_NUM_SPEC_TOKENS = 3
DEFAULT_RTOL = 0.05


@dataclass
class Eagle3PdModelConfig:
    verifier: str
    drafter: str
    expected_acceptance_length: float
    expected_acceptance_lengths_per_pos: list[float] = field(default_factory=list)
    id: str = ""
    rtol: float | None = None


EAGLE3_PD_CONFIGS = [
    Eagle3PdModelConfig(
        verifier="Qwen/Qwen3-8B",
        drafter="RedHatAI/Qwen3-8B-speculator.eagle3",
        expected_acceptance_length=2.26,
        expected_acceptance_lengths_per_pos=[0.6541, 0.3993, 0.2020],
        id="qwen3-8b-eagle3",
    ),
    Eagle3PdModelConfig(
        verifier="openai/gpt-oss-20b",
        drafter="RedHatAI/gpt-oss-20b-speculator.eagle3",
        expected_acceptance_length=2.56,
        expected_acceptance_lengths_per_pos=[0.7165, 0.5120, 0.3337],
        id="gpt-oss-20b-eagle3",
    ),
]


def _get_model_config() -> Eagle3PdModelConfig:
    """Resolve config by TEST_CONFIG env var or TEST_MODEL."""
    for config in EAGLE3_PD_CONFIGS:
        if config.id == TEST_CONFIG or config.verifier == MODEL_NAME:
            return config
    raise ValueError(
        f"No config found for TEST_CONFIG={TEST_CONFIG!r}, "
        f"TEST_MODEL={MODEL_NAME!r}. "
        f"Available: {[c.id for c in EAGLE3_PD_CONFIGS]}"
    )


# ── Dataset loading ──────────────────────────────────────────────────────


def _get_mt_bench_prompts() -> list[str]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    args = SimpleNamespace(
        dataset_name="hf",
        dataset_path="philschmid/mt-bench",
        num_prompts=DEFAULT_NUM_PROMPTS,
        seed=42,
        no_oversample=False,
        endpoint_type="openai-chat",
        backend="openai-chat",
        input_len=None,
        output_len=DEFAULT_OUTPUT_LEN,
        sharegpt_output_len=DEFAULT_OUTPUT_LEN,
        hf_name=None,
        hf_split="train",
        hf_subset=None,
        hf_output_len=DEFAULT_OUTPUT_LEN,
        no_stream=True,
        disable_shuffle=False,
        skip_chat_template=False,
        trust_remote_code=False,
        enable_multimodal_chat=False,
        request_id_prefix="",
    )
    samples = get_samples(args, tokenizer)
    return [sample.prompt for sample in samples]


# ── Metrics helpers ──────────────────────────────────────────────────────


def _fetch_metric(metric_name: str) -> float:
    url = f"http://localhost:{DECODE_PORT}/metrics"
    body = urlopen(url).read().decode()
    for line in body.split("\n"):
        if line.startswith(metric_name + "{") or line.startswith(metric_name + " "):
            return float(line.rsplit(" ", 1)[-1])
    raise ValueError(f"Metric {metric_name} not found in decode /metrics")


def _fetch_per_position_acceptance() -> dict[int, float]:
    url = f"http://localhost:{DECODE_PORT}/metrics"
    body = urlopen(url).read().decode()
    counts: dict[int, float] = {}
    for line in body.split("\n"):
        if (
            "spec_decode_num_accepted_tokens_per_pos_total" in line
            and not line.startswith("#")
        ):
            m = re.search(r'position="(\d+)"', line)
            if m:
                counts[int(m.group(1))] = float(line.rsplit(" ", 1)[-1])
    return counts


# ── Test ─────────────────────────────────────────────────────────────────


def test_pd_eagle3_acceptance_length():
    """Validate PD+SD acceptance length against standalone baseline.

    Sends MT-Bench prompts through the PD proxy, then checks that the
    decode server's speculative decoding metrics match the known
    standalone EAGLE3 baselines.
    """
    config = _get_model_config()
    rtol = config.rtol if config.rtol is not None else DEFAULT_RTOL

    prompts = _get_mt_bench_prompts()
    assert len(prompts) == DEFAULT_NUM_PROMPTS, (
        f"Expected {DEFAULT_NUM_PROMPTS} prompts, got {len(prompts)}"
    )

    client = openai.OpenAI(api_key="EMPTY", base_url=PROXY_BASE_URL)
    for i, prompt in enumerate(prompts):
        resp = client.completions.create(
            model=MODEL_NAME,
            prompt=prompt,
            max_tokens=DEFAULT_OUTPUT_LEN,
            temperature=0.0,
            top_p=1.0,
        )
        if i < 3:
            text = resp.choices[0].text.strip()[:100]
            print(f"  [{i}] {prompt[:60]}... -> {text}...")

    # ── Extract metrics ───────────────────────────────────────────────
    n_drafts = _fetch_metric("vllm:spec_decode_num_drafts_total")
    n_accepted = _fetch_metric("vllm:spec_decode_num_accepted_tokens_total")

    assert n_drafts > 0, "No spec-decode drafts were generated"

    acceptance_length = 1 + (n_accepted / n_drafts)

    per_pos_counts = _fetch_per_position_acceptance()
    per_pos_rates = [
        per_pos_counts.get(i, 0) / n_drafts
        for i in range(len(config.expected_acceptance_lengths_per_pos))
    ]

    # ── Report ────────────────────────────────────────────────────────
    expected = config.expected_acceptance_length
    expected_per_pos = config.expected_acceptance_lengths_per_pos

    print(
        f"\n{config.id}: acceptance_length={acceptance_length:.3f} "
        f"(expected={expected:.3f})"
    )
    print(f"  Drafts: {n_drafts:.0f}, Accepted: {n_accepted:.0f}")
    for i, (actual, exp) in enumerate(zip(per_pos_rates, expected_per_pos)):
        print(f"  Position {i}: {actual:.4f} (expected: {exp:.4f})")

    # ── Assert overall acceptance length ──────────────────────────────
    rel_error = abs(acceptance_length - expected) / expected

    assert rel_error <= rtol, (
        f"Acceptance length regression for {config.id}! "
        f"Expected: {expected:.3f}, "
        f"Got: {acceptance_length:.3f}, "
        f"Relative error: {rel_error:.2%} (tolerance: {rtol:.0%}). "
        f"This may indicate drafter KV was not correctly transferred."
    )

    # ── Assert per-position acceptance ────────────────────────────────
    for i, (actual, exp) in enumerate(zip(per_pos_rates, expected_per_pos)):
        if exp > 0:
            pos_err = abs(actual - exp) / exp
            assert pos_err <= rtol, (
                f"Per-position acceptance regression at position {i} "
                f"for {config.id}! "
                f"Expected: {exp:.4f}, Got: {actual:.4f}, "
                f"Relative error: {pos_err:.2%} "
                f"(tolerance: {rtol:.0%})"
            )

    print(
        f"\n=== PASS: {config.id} PD+SD acceptance length "
        f"{acceptance_length:.3f} within {rtol:.0%} of {expected:.3f} ==="
    )
