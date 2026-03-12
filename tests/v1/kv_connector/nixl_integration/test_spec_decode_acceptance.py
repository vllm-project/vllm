# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NixlConnector PD + EAGLE3 speculative decoding acceptance length test.

  - Loads MT-Bench prompts (80 prompts, 256 output tokens)
  - Sends through the PD proxy (completions API)
  - Scrapes Prometheus metrics from the decode server
  - Asserts acceptance length matches standalone EAGLE3 baselines

Baselines from tests/v1/spec_decode/test_acceptance_length.py
(standalone EAGLE3 with same model/drafter on MT-Bench, temp=0).
PD disaggregation via NixlConnector should match within tolerance.

Environment variables (set by spec_decode_acceptance_test.sh):
    TEST_MODEL   - target model name
    DECODE_PORT  - port of the decode vLLM server (for /metrics)
"""

import os
from dataclasses import dataclass, field
from types import SimpleNamespace
from urllib.request import urlopen

import openai
import regex as re
from transformers import AutoTokenizer

from vllm.benchmarks.datasets import get_samples

PROXY_BASE_URL = "http://localhost:8192/v1"
DECODE_PORT = os.environ.get("DECODE_PORT", "8200")
MODEL_NAME = os.environ.get("TEST_MODEL", "meta-llama/Llama-3.1-8B-Instruct")


@dataclass
class Eagle3ModelConfig:
    verifier: str
    drafter: str
    expected_acceptance_length: float
    expected_acceptance_lengths_per_pos: list[float] = field(default_factory=list)
    id: str = ""
    rtol: float | None = None


# Standalone EAGLE3 baselines (MT-Bench, 80 prompts, 256 tokens, temp=0).
# Source: tests/v1/spec_decode/test_acceptance_length.py
EAGLE3_MODEL_CONFIGS = [
    Eagle3ModelConfig(
        verifier="meta-llama/Llama-3.1-8B-Instruct",
        drafter="RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3",
        expected_acceptance_length=2.60,
        expected_acceptance_lengths_per_pos=[0.7296, 0.5208, 0.3545],
        id="llama3-8b-eagle3",
    ),
]

DEFAULT_NUM_PROMPTS = 80
DEFAULT_OUTPUT_LEN = 256
DEFAULT_RTOL = 0.05


def _get_model_config() -> Eagle3ModelConfig:
    """Get the model config matching MODEL_NAME."""
    for config in EAGLE3_MODEL_CONFIGS:
        if config.verifier == MODEL_NAME:
            return config
    raise ValueError(
        f"No Eagle3ModelConfig found for model {MODEL_NAME}. "
        f"Available: {[c.verifier for c in EAGLE3_MODEL_CONFIGS]}"
    )


def _get_mt_bench_prompts() -> list[str]:
    """Load MT-Bench prompts via vllm.benchmarks.datasets.get_samples."""
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


def _fetch_metric(metric_name: str) -> float:
    """Fetch a single counter metric from the decode server's /metrics."""
    url = f"http://localhost:{DECODE_PORT}/metrics"
    body = urlopen(url).read().decode()
    for line in body.split("\n"):
        if line.startswith(metric_name + "{") or line.startswith(metric_name + " "):
            return float(line.rsplit(" ", 1)[-1])
    raise ValueError(f"Metric {metric_name} not found in decode /metrics")


def _fetch_per_position_acceptance() -> dict[int, float]:
    """Fetch per-position acceptance counts from decode /metrics."""
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


def test_spec_decode_acceptance_length():
    """Validate PD+SD acceptance length against standalone baseline.

    Sends MT-Bench prompts through the PD proxy (completions API),
    then checks that the decode server's speculative decoding metrics
    match the known standalone baselines.
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

    # ── Extract metrics from decode server ────────────────────────────
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
        f"\n=== PASS: {config.id} acceptance length {acceptance_length:.3f} "
        f"within {rtol:.0%} of {expected:.3f} ==="
    )
