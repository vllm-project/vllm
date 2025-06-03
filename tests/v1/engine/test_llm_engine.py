# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random
from typing import Optional

import pytest

from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import Counter, Gauge, Histogram, Metric, Vector

MODEL = "facebook/opt-125m"
DTYPE = "half"


def _vllm_model(apc: bool, vllm_runner, monkeypatch):
    """Set up VllmRunner instance."""
    monkeypatch.setenv("VLLM_USE_V1", "1")
    return vllm_runner(
        MODEL,
        dtype=DTYPE,
        max_model_len=128,
        enforce_eager=True,
        enable_prefix_caching=apc,
        gpu_memory_utilization=0.5,
    )


@pytest.fixture(
    # Function scope decouples tests & allows
    # env var adjustment via monkeypatch
    scope="function",
    # Prefix caching
    params=[False, True])
def vllm_model(vllm_runner, request, monkeypatch):
    """VllmRunner test fixture parameterized by APC True/False."""
    with _vllm_model(request.param, vllm_runner, monkeypatch) as vllm_model:
        yield vllm_model


@pytest.fixture(scope="function")
def vllm_model_apc(vllm_runner, monkeypatch):
    """VllmRunner test fixture with APC."""
    with _vllm_model(True, vllm_runner, monkeypatch) as vllm_model:
        yield vllm_model


def _get_test_sampling_params(
    prompt_list: list[str],
    seed: Optional[int] = 42,
) -> tuple[list[SamplingParams], list[int]]:
    """Generate random sampling params for a batch."""

    def get_mostly_n_gt1() -> int:
        r"""Mostly n \in [2,20], ~1/3 n=1"""
        x = random.randint(0, 28)
        if x < 10:
            return 1
        else:
            return x - 8

    n_list = [get_mostly_n_gt1() for _ in range(len(prompt_list))]
    # High temperature to maximize the chance of unique completions
    return [
        SamplingParams(temperature=0.95, top_p=0.95, n=n, seed=seed)
        for n in n_list
    ], n_list


def test_parallel_sampling(vllm_model, example_prompts) -> None:
    """Test passes if parallel sampling `n>1` yields `n` unique completions.
    
    Args:
      vllm_model: VllmRunner instance under test.
      example_prompt: test fixture providing prompts for testing.
    """
    sampling_params_list, n_list = _get_test_sampling_params(example_prompts)
    model: LLM = vllm_model.model
    outputs = model.generate(example_prompts, sampling_params_list)

    # Validate each request response
    for out, n in zip(outputs, n_list):
        completion_counts: dict[str, int] = {}
        # Assert correct number of completions
        assert len(out.outputs) == n, (
            f"{len(out.outputs)} completions; {n} expected.")
        for idx in range(n):
            comp = out.outputs[idx]
            # Assert correct completion indices
            assert comp.index == idx, (f"Index {comp.index}; expected {idx}.")
            text = comp.text
            completion_counts[text] = completion_counts.get(text, 0) + 1
        # Assert unique completions
        if len(completion_counts) != n:
            repeats = {
                txt: num
                for (txt, num) in completion_counts.items() if num > 1
            }
            raise AssertionError(
                f"{len(completion_counts)} unique completions; expected"
                f" {n}. Repeats: {repeats}")


def test_engine_metrics(vllm_runner, monkeypatch, example_prompts):
    max_tokens = 100
    # Use spec decoding to test num_accepted_tokens_per_pos
    speculative_config = {
        "method": "ngram",
        "prompt_lookup_max": 5,
        "prompt_lookup_min": 3,
        "num_speculative_tokens": 5,
    }
    monkeypatch.setenv("VLLM_USE_V1", "1")
    with vllm_runner(
            MODEL,
            speculative_config=speculative_config,
            disable_log_stats=False,
    ) as vllm_model:
        model: LLM = vllm_model.model
        sampling_params = SamplingParams(temperature=0.0,
                                         max_tokens=max_tokens)
        outputs = model.generate(example_prompts, sampling_params)

        n_prompts = len(example_prompts)
        assert len(outputs) == n_prompts

        total_tokens = 0
        for out in outputs:
            assert len(out.outputs) == 1
            total_tokens += len(out.outputs[0].token_ids)
        assert total_tokens == max_tokens * n_prompts

        metrics = model.get_metrics()

        def find_metric(name) -> list[Metric]:
            found = []
            for metric in metrics:
                if metric.name == name:
                    found.append(metric)
            return found

        num_requests_running = find_metric("vllm:num_requests_running")
        assert len(num_requests_running) == 1
        assert isinstance(num_requests_running[0], Gauge)
        assert num_requests_running[0].value == .0

        generation_tokens = find_metric("vllm:generation_tokens")
        assert len(generation_tokens) == 1
        assert isinstance(generation_tokens[0], Counter)
        assert generation_tokens[0].value == total_tokens

        request_generation_tokens = find_metric(
            "vllm:request_generation_tokens")
        assert len(request_generation_tokens) == 1
        assert isinstance(request_generation_tokens[0], Histogram)
        assert "+Inf" in request_generation_tokens[0].buckets
        assert request_generation_tokens[0].buckets["+Inf"] == n_prompts
        assert request_generation_tokens[0].count == n_prompts
        assert request_generation_tokens[0].sum == total_tokens

        num_accepted_tokens_per_pos = find_metric(
            "vllm:spec_decode_num_accepted_tokens_per_pos")
        assert len(num_accepted_tokens_per_pos) == 1
        assert isinstance(num_accepted_tokens_per_pos[0], Vector)
        assert len(num_accepted_tokens_per_pos[0].values) == 5
