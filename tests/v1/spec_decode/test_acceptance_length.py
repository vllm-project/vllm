# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
EAGLE3 Acceptance Length Regression Tests.

These tests verify that acceptance lengths for EAGLE3 speculative decoding
do not regress across vLLM commits. Each test runs inference on the MT-Bench
dataset and asserts that the mean acceptance length is within tolerance of
the expected baseline.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from vllm import LLM, SamplingParams
from vllm.benchmarks.datasets import get_samples
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.inputs import TokensPrompt
from vllm.v1.metrics.reader import Counter


@dataclass
class Eagle3ModelConfig:
    """Configuration for an EAGLE3 model pair."""

    verifier: str
    drafter: str
    expected_acceptance_length: float
    id: str


# Model configurations for EAGLE3 acceptance length tests.
# Expected acceptance lengths are determined by running the baseline script.
# TODO: Fill in expected_acceptance_length values after running baselines.
EAGLE3_MODEL_CONFIGS = [
    Eagle3ModelConfig(
        verifier="meta-llama/Llama-3.1-8B-Instruct",
        drafter="RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3",
        expected_acceptance_length=0.0,  # TODO: Run baseline to determine
        id="llama3-8b-eagle3",
    ),
    Eagle3ModelConfig(
        verifier="Qwen/Qwen3-8B",
        drafter="RedHatAI/Qwen3-8B-speculator.eagle3",
        expected_acceptance_length=0.0,  # TODO: Run baseline to determine
        id="qwen3-8b-eagle3",
    ),
    Eagle3ModelConfig(
        verifier="openai/gpt-oss-20b",
        drafter="RedHatAI/gpt-oss-20b-speculator.eagle3",
        expected_acceptance_length=0.0,  # TODO: Run baseline to determine
        id="gpt-oss-20b-eagle3",
    ),
]

# Default test parameters
DEFAULT_NUM_SPEC_TOKENS = 3
DEFAULT_NUM_PROMPTS = 80
DEFAULT_OUTPUT_LEN = 256
DEFAULT_MAX_MODEL_LEN = 16384
DEFAULT_RTOL = 0.02  # 2% relative tolerance


def get_mt_bench_prompts(tokenizer, num_prompts: int = DEFAULT_NUM_PROMPTS):
    """Load prompts from MT-Bench dataset."""
    args = SimpleNamespace(
        dataset_name="hf",
        dataset_path="philschmid/mt-bench",
        num_prompts=num_prompts,
        seed=42,
        no_oversample=False,
        endpoint_type="openai-chat",
        input_len=None,
        output_len=DEFAULT_OUTPUT_LEN,
        sharegpt_output_len=DEFAULT_OUTPUT_LEN,
    )
    samples = get_samples(args, tokenizer)
    prompt_ids = [
        tokenizer.encode(sample.prompt, add_special_tokens=False) for sample in samples
    ]
    return prompt_ids


def extract_acceptance_metrics(metrics) -> dict:
    """Extract acceptance length metrics from LLM metrics."""
    num_drafts = 0
    num_accepted_tokens = 0

    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            assert isinstance(metric, Counter)
            num_accepted_tokens += metric.value

    # Calculate mean acceptance length
    # Formula: 1 + (accepted_tokens / num_drafts)
    acceptance_length = 1 + (num_accepted_tokens / num_drafts) if num_drafts > 0 else 1

    return {
        "acceptance_length": acceptance_length,
        "num_drafts": num_drafts,
        "num_accepted_tokens": num_accepted_tokens,
    }


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up GPU memory after each test."""
    yield
    torch.cuda.empty_cache()
    cleanup_dist_env_and_memory()
    torch._dynamo.reset()


@pytest.mark.parametrize(
    "model_config",
    [pytest.param(config, id=config.id) for config in EAGLE3_MODEL_CONFIGS],
)
@pytest.mark.parametrize("num_spec_tokens", [DEFAULT_NUM_SPEC_TOKENS])
def test_eagle3_acceptance_length(
    model_config: Eagle3ModelConfig,
    num_spec_tokens: int,
    monkeypatch: pytest.MonkeyPatch,
):
    """
    Test EAGLE3 acceptance length does not regress.

    This test:
    1. Loads the MT-Bench dataset
    2. Runs inference with EAGLE3 speculative decoding
    3. Extracts acceptance length metrics
    4. Asserts the acceptance length is within tolerance of expected baseline

    Args:
        model_config: Configuration for verifier/drafter model pair
        num_spec_tokens: Number of speculative tokens to generate
        monkeypatch: Pytest monkeypatch fixture
    """
    # Skip if expected acceptance length is not set
    if model_config.expected_acceptance_length <= 0:
        pytest.skip(
            f"Expected acceptance length not set for {model_config.id}. "
            "Run baseline script to determine expected value."
        )

    # Allow insecure serialization for speculators
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    # Initialize LLM with speculative decoding
    llm = LLM(
        model=model_config.verifier,
        speculative_config={
            "method": "eagle3",
            "model": model_config.drafter,
            "num_speculative_tokens": num_spec_tokens,
        },
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        disable_log_stats=False,
        max_model_len=DEFAULT_MAX_MODEL_LEN,
    )

    # Load MT-Bench prompts
    tokenizer = llm.get_tokenizer()
    prompt_ids = get_mt_bench_prompts(tokenizer, DEFAULT_NUM_PROMPTS)

    # Run inference
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=DEFAULT_OUTPUT_LEN,
    )
    llm.generate(
        [TokensPrompt(prompt_token_ids=ids) for ids in prompt_ids],
        sampling_params=sampling_params,
    )

    # Extract and validate metrics
    metrics = llm.get_metrics()
    results = extract_acceptance_metrics(metrics)

    actual_acceptance_length = results["acceptance_length"]
    expected = model_config.expected_acceptance_length

    # Calculate relative error
    rel_error = abs(actual_acceptance_length - expected) / expected

    # Assert within tolerance
    assert rel_error <= DEFAULT_RTOL, (
        f"Acceptance length regression detected for {model_config.id}!\n"
        f"  Expected: {expected:.3f}\n"
        f"  Actual:   {actual_acceptance_length:.3f}\n"
        f"  Relative error: {rel_error:.2%} (tolerance: {DEFAULT_RTOL:.2%})\n"
        f"  Drafts: {results['num_drafts']}, "
        f"Accepted tokens: {results['num_accepted_tokens']}"
    )

    print(
        f"\n{model_config.id}: acceptance_length={actual_acceptance_length:.3f} "
        f"(expected={expected:.3f}, rel_error={rel_error:.2%})"
    )

    # Cleanup
    del llm
