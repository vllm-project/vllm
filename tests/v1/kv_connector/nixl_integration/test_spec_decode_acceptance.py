# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NixlConnector PD + speculative decoding acceptance length test.

  - Loads MT-Bench prompts (80 prompts, 256 output tokens)
  - Sends through the PD proxy (completions API)
  - Scrapes Prometheus metrics from the decode server
  - Asserts acceptance metrics match standalone baselines

Supports EAGLE3 (default) and MTP, selected via SD_METHOD env var.

Environment variables (set by spec_decode_acceptance_test.sh):
    TEST_MODEL   - target model name
    DECODE_PORT  - port of the decode vLLM server (for /metrics)
    SD_METHOD    - "eagle3" (default) or "mtp"
"""

import os
from dataclasses import dataclass, field
from types import SimpleNamespace
from urllib.request import urlopen

import openai
import regex as re
from transformers import AutoTokenizer

from vllm.benchmarks.datasets import get_samples

SERVER_HOST = os.environ.get("SERVER_HOST", "127.0.0.1")
PROXY_BASE_URL = f"http://{SERVER_HOST}:8192/v1"
DECODE_PORT = os.environ.get("DECODE_PORT", "8200")
MODEL_NAME = os.environ.get("TEST_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
SD_METHOD = os.environ.get("SD_METHOD", "eagle3").lower()


@dataclass
class ModelConfig:
    model: str
    method: str
    expected_acceptance_length: float
    drafter: str = ""
    expected_acceptance_lengths_per_pos: list[float] = field(default_factory=list)
    expected_acceptance_rate: float | None = None
    id: str = ""
    rtol: float | None = None


# Standalone baselines (MT-Bench, 80 prompts, 256 tokens, temp=0).
# EAGLE3 source: tests/v1/spec_decode/test_acceptance_length.py
MODEL_CONFIGS = [
    ModelConfig(
        model="meta-llama/Llama-3.1-8B-Instruct",
        method="eagle3",
        drafter="RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3",
        expected_acceptance_length=2.60,
        expected_acceptance_lengths_per_pos=[0.7296, 0.5208, 0.3545],
        id="llama3-8b-eagle3",
    ),
    ModelConfig(
        model="Qwen/Qwen3.5-0.8B-Base",
        method="mtp",
        expected_acceptance_length=1.798,
        id="qwen35-0.8b-mtp",
    ),
]

DEFAULT_NUM_PROMPTS = 80
DEFAULT_OUTPUT_LEN = 256
DEFAULT_RTOL = 0.05


def _get_model_config() -> ModelConfig:
    """Get the model config matching MODEL_NAME and SD_METHOD."""
    for config in MODEL_CONFIGS:
        if config.model == MODEL_NAME and config.method == SD_METHOD:
            return config
    raise ValueError(
        f"No config for model={MODEL_NAME}, method={SD_METHOD}. "
        f"Available: {[(c.model, c.method) for c in MODEL_CONFIGS]}"
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
    url = f"http://{SERVER_HOST}:{DECODE_PORT}/metrics"
    body = urlopen(url).read().decode()
    for line in body.split("\n"):
        if line.startswith(metric_name + "{") or line.startswith(metric_name + " "):
            return float(line.rsplit(" ", 1)[-1])
    raise ValueError(f"Metric {metric_name} not found in decode /metrics")


def _fetch_per_position_acceptance() -> dict[int, float]:
    """Fetch per-position acceptance counts from decode /metrics."""
    url = f"http://{SERVER_HOST}:{DECODE_PORT}/metrics"
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
            # Prompts are already chat-templated (contain BOS); avoid the
            # completions API prepending a second BOS, which would lower
            # acceptance ~5% vs the add_special_tokens=False standalone baselines.
            extra_body={"add_special_tokens": False},
        )
        if i < 3:
            text = resp.choices[0].text.strip()[:100]
            print(f"  [{i}] {prompt[:60]}... -> {text}...")

    # ── Extract metrics from decode server ────────────────────────────
    n_drafts = _fetch_metric("vllm:spec_decode_num_drafts_total")
    n_accepted = _fetch_metric("vllm:spec_decode_num_accepted_tokens_total")

    assert n_drafts > 0, "No spec-decode drafts were generated"

    acceptance_length = 1 + (n_accepted / n_drafts)
    expected = config.expected_acceptance_length

    print(
        f"\n{config.id}: acceptance_length={acceptance_length:.3f} "
        f"(expected={expected:.3f})"
    )
    print(f"  Drafts: {n_drafts:.0f}, Accepted: {n_accepted:.0f}")

    # ── Assert acceptance length (all methods) ────────────────────────
    rel_error = abs(acceptance_length - expected) / expected
    assert rel_error <= rtol, (
        f"Acceptance length regression for {config.id}! "
        f"Expected: {expected:.3f}, "
        f"Got: {acceptance_length:.3f}, "
        f"Relative error: {rel_error:.2%} (tolerance: {rtol:.0%})"
    )

    # ── Assert per-position acceptance (EAGLE3) ───────────────────────
    if config.expected_acceptance_lengths_per_pos:
        per_pos_counts = _fetch_per_position_acceptance()
        per_pos_rates = [
            per_pos_counts.get(i, 0) / n_drafts
            for i in range(len(config.expected_acceptance_lengths_per_pos))
        ]
        for i, (actual, exp) in enumerate(
            zip(per_pos_rates, config.expected_acceptance_lengths_per_pos)
        ):
            print(f"  Position {i}: {actual:.4f} (expected: {exp:.4f})")
            if exp > 0:
                pos_err = abs(actual - exp) / exp
                assert pos_err <= rtol, (
                    f"Per-position regression at pos {i} for {config.id}! "
                    f"Expected: {exp:.4f}, Got: {actual:.4f}, "
                    f"Relative error: {pos_err:.2%} (tolerance: {rtol:.0%})"
                )

    # ── Assert acceptance rate (MTP) ──────────────────────────────────
    if config.expected_acceptance_rate is not None:
        n_draft_tokens = _fetch_metric("vllm:spec_decode_num_draft_tokens_total")
        acceptance_rate = n_accepted / n_draft_tokens if n_draft_tokens > 0 else 0.0
        print(
            f"  Acceptance rate: {acceptance_rate:.3f} "
            f"(expected: {config.expected_acceptance_rate:.3f})"
        )
        rate_err = (
            abs(acceptance_rate - config.expected_acceptance_rate)
            / config.expected_acceptance_rate
        )
        assert rate_err <= rtol, (
            f"Acceptance rate regression for {config.id}! "
            f"Expected: {config.expected_acceptance_rate:.3f}, "
            f"Got: {acceptance_rate:.3f}, "
            f"Relative error: {rate_err:.2%} (tolerance: {rtol:.0%})"
        )

    print(
        f"\n=== PASS: {config.id} acceptance length {acceptance_length:.3f} "
        f"within {rtol:.0%} of {expected:.3f} ==="
    )
