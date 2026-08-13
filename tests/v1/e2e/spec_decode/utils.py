# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random
from collections.abc import Iterable, Sequence
from typing import Any

import pytest
import torch

from tests.evals.gsm8k.gsm8k_eval import (
    _build_gsm8k_prompts,
    assert_min_accuracy,
    evaluate_gsm8k_offline,
)
from vllm import LLM, SamplingParams
from vllm.assets.base import VLLM_S3_BUCKET_URL
from vllm.assets.image import VLM_IMAGES_DIR
from vllm.outputs import RequestOutput
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.metrics.reader import Metric


def _skip_if_insufficient_gpus_for_tp(tp_size: int):
    """Skip if fewer than ``tp_size`` accelerators are available."""
    available_gpus = torch.accelerator.device_count()
    if available_gpus < tp_size:
        pytest.skip(
            f"Test requires {tp_size} GPUs, but only {available_gpus} available"
        )


Messages = list[dict[str, Any]]


def get_test_prompts(mm_enabled: bool, num_prompts: int = 100) -> list[Messages]:
    prompt_types = ["repeat", "gsm8k"]
    if mm_enabled:
        prompt_types.append("mm")
    prompts: list[Messages] = []

    num_repeat_prompts = num_prompts // len(prompt_types)
    if mm_enabled:
        num_gsm8k_prompts = num_prompts // len(prompt_types)
        num_mm_prompts = num_prompts - num_repeat_prompts - num_gsm8k_prompts
    else:
        num_mm_prompts = 0
        num_gsm8k_prompts = num_prompts - num_repeat_prompts

    # Generate a mixed batch of prompts, some of which can be easily
    # predicted by n-gram matching and some which likely cannot.
    set_random_seed(0)
    for _ in range(num_repeat_prompts):
        word_choices = ["test", "temp", "hello", "where"]
        word = random.choice(word_choices)
        prompts.append(
            [
                {
                    "role": "user",
                    "content": f"""
        please repeat the word '{word}' 10 times.
        give no other output than the word at least ten times in a row,
        in lowercase with spaces between each word and without quotes.
        """,
                }
            ]
        )
    prompts.extend(
        [{"role": "user", "content": prompt}]
        for prompt in _build_gsm8k_prompts(
            num_questions=num_gsm8k_prompts, num_shots=5
        )[0]
    )
    for _ in range(num_mm_prompts):
        placeholders = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"{VLLM_S3_BUCKET_URL}/{VLM_IMAGES_DIR}/stop_sign.jpg"
                },
            }
        ]
        prompt = [
            *placeholders,
            {"type": "text", "text": "The meaning of the image is"},
        ]
        prompts.append([{"role": "user", "content": prompt}])

    return prompts


def get_instruct_coder_messages(n: int) -> list[Messages]:
    from vllm.benchmarks.datasets import InstructCoderDataset

    dataset = InstructCoderDataset(
        dataset_path="likaixin/InstructCoder", dataset_split="train"
    )
    prompts: Iterable[str] = dataset.sample_prompts(n=n)
    return [[{"role": "user", "content": prompt}] for prompt in prompts]


def greedy_sampling() -> SamplingParams:
    return SamplingParams(temperature=0, max_tokens=10, ignore_eos=False)


def stochastic_sampling() -> SamplingParams:
    return SamplingParams(temperature=1.0, max_tokens=10, ignore_eos=False)


def evaluate_llm_for_gsm8k(llm: LLM, expected_accuracy_threshold: float = 0.70) -> None:
    """Evaluate the LLM on GSM8K and check that accuracy is above a sanity threshold.

    The default threshold assumes the LLM uses the same target model as the "model_name"
    fixture, with max model len == 4096. Precomputed reference value is 75% to 80%
    on GSM8K with greedy decoding, so we check that it's above a sanity threshold of 70%
    to verify that the model is correct.
    """
    if expected_accuracy_threshold <= 0.0:
        print("Skipping GSM8K evaluation")
        return
    results = evaluate_gsm8k_offline(llm)
    accuracy = results.accuracy
    print(f"GSM8K accuracy: {accuracy:.3f}")
    assert_min_accuracy(results, expected_accuracy_threshold)


def assert_request_outputs_match(
    ref_outputs: Sequence[RequestOutput],
    spec_outputs: Sequence[RequestOutput],
    *,
    required_matches: int,
    context: str,
    max_mismatches: int = 3,
) -> None:
    """Assert a thresholded exact-text match with bounded failure diagnostics."""
    assert ref_outputs, f"{context}: no reference outputs"
    assert len(ref_outputs) == len(spec_outputs), (
        f"{context}: output count differs: "
        f"reference={len(ref_outputs)}, speculative={len(spec_outputs)}"
    )
    assert 0 <= required_matches <= len(ref_outputs), (
        f"{context}: invalid required_matches={required_matches} for "
        f"{len(ref_outputs)} outputs"
    )

    mismatches: list[str] = []
    matches = 0
    for index, (ref_output, spec_output) in enumerate(zip(ref_outputs, spec_outputs)):
        assert ref_output.outputs, (
            f"{context}: reference output {index} has no candidate"
        )
        assert spec_output.outputs, (
            f"{context}: speculative output {index} has no candidate"
        )
        ref_candidate = ref_output.outputs[0]
        spec_candidate = spec_output.outputs[0]
        if ref_candidate.text == spec_candidate.text:
            matches += 1
        elif len(mismatches) < max_mismatches:
            mismatches.append(
                f"[{index}] ref_text={ref_candidate.text[:240]!r}, "
                f"spec_text={spec_candidate.text[:240]!r}\n"
                f"    ref_token_ids={list(ref_candidate.token_ids)[:64]}\n"
                f"    spec_token_ids={list(spec_candidate.token_ids)[:64]}"
            )

    print(
        f"{context}: exact text matches={matches}/{len(ref_outputs)} "
        f"(required={required_matches})"
    )
    mismatch_summary = "\n".join(mismatches) or "no mismatches captured"
    assert matches >= required_matches, (
        f"{context}: only {matches}/{len(ref_outputs)} outputs matched; "
        f"required at least {required_matches}. First mismatches:\n"
        f"{mismatch_summary}"
    )


def get_spec_decode_metric_value(metrics: Sequence[Metric], metric_name: str) -> float:
    """Get a spec-decode metric with an actionable error when stats are absent."""
    name2metric = {metric.name: metric for metric in metrics}
    metric = name2metric.get(metric_name)
    assert metric is not None, (
        f"Missing metric {metric_name!r}. Ensure disable_log_stats=False. "
        "Available spec-decode metrics: "
        f"{sorted(name for name in name2metric if 'spec_decode' in name) or ['<none>']}"
    )
    return float(metric.value)


def compute_acceptance_rate(
    metrics: list[Metric], prev_metrics: list[Metric] | None = None
) -> float:
    n_draft_toks = get_spec_decode_metric_value(
        metrics, "vllm:spec_decode_num_draft_tokens"
    )
    if n_draft_toks == 0:
        return float("nan")
    n_accepted_toks = get_spec_decode_metric_value(
        metrics, "vllm:spec_decode_num_accepted_tokens"
    )
    if prev_metrics is not None:
        n_draft_toks -= get_spec_decode_metric_value(
            prev_metrics, "vllm:spec_decode_num_draft_tokens"
        )
        n_accepted_toks -= get_spec_decode_metric_value(
            prev_metrics, "vllm:spec_decode_num_accepted_tokens"
        )
        if n_draft_toks <= 0:
            return float("nan")
    return n_accepted_toks / n_draft_toks


def compute_acceptance_len(
    metrics: list[Metric], prev_metrics: list[Metric] | None = None
) -> float:
    n_drafts = get_spec_decode_metric_value(metrics, "vllm:spec_decode_num_drafts")
    n_accepted_toks = get_spec_decode_metric_value(
        metrics, "vllm:spec_decode_num_accepted_tokens"
    )
    if n_drafts == 0:
        return 1
    if prev_metrics is not None:
        n_drafts -= get_spec_decode_metric_value(
            prev_metrics, "vllm:spec_decode_num_drafts"
        )
        n_accepted_toks -= get_spec_decode_metric_value(
            prev_metrics, "vllm:spec_decode_num_accepted_tokens"
        )
        if n_drafts <= 0:
            return 1
    return 1 + (n_accepted_toks / n_drafts)
